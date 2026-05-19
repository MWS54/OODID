from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .io import load_records
from .preprocessing import GroupAwareMetadataPreprocessor, MetadataPreprocessor
from .utils import parse_csv_list
from .windowing import (
    WindowedData,
    attach_parsed_labels,
    build_grouped_windows,
    filter_id_windows,
    infer_all_labels,
    mark_ood_records,
)


@dataclass
class PreparedWindowDataset:
    df: pd.DataFrame
    split_source: str
    split_frames: Dict[str, pd.DataFrame]
    split_windows: Dict[str, WindowedData]
    id_classes: list[str]
    ood_classes: list[str]
    class_to_idx: Dict[str, int]
    preprocessor: MetadataPreprocessor
    normalization_summary: dict
    leakage_report: dict
    label_col: str
    timestamp_col: str
    record_id_col: str
    group_col: Optional[str]
    window_config: dict

    def window_counts(self) -> dict[str, int]:
        return {name: int(len(windows)) for name, windows in self.split_windows.items()}


def sort_records_for_windowing(df: pd.DataFrame, group_col: Optional[str], timestamp_col: str) -> pd.DataFrame:
    if group_col and group_col in df.columns:
        if timestamp_col in df.columns:
            return df.sort_values([group_col, timestamp_col], kind="stable").reset_index(drop=True)
        return df.sort_values(group_col, kind="stable").reset_index(drop=True)
    if timestamp_col in df.columns:
        return df.sort_values(timestamp_col, kind="stable").reset_index(drop=True)
    return df.reset_index(drop=True)


def build_windows_by_mode(features: np.ndarray, df: pd.DataFrame, class_to_idx: Dict[str, int], args) -> WindowedData:
    return build_grouped_windows(
        features,
        df,
        class_to_idx,
        group_col=args.group_col,
        mode=args.window_mode,
        timestamp_col=args.timestamp_col,
        label_col=args.label_col,
        record_id_col=args.record_id_col,
        window_size=args.window_size,
        stride=args.stride,
        time_seconds=args.time_window_seconds,
        adaptive_min_size=args.adaptive_min_size,
        adaptive_max_size=args.adaptive_max_size,
    )


def chronological_split_id_records(df: pd.DataFrame, ratios=(0.70, 0.15, 0.15)):
    id_df = df[~df["__is_ood_record"]].copy()
    n = len(id_df)
    n_train = max(1, int(n * ratios[0]))
    n_val = max(1, int(n * ratios[1]))
    train_df = id_df.iloc[:n_train].copy()
    val_df = id_df.iloc[n_train:n_train + n_val].copy()
    test_df = id_df.iloc[n_train + n_val:].copy()
    if len(test_df) == 0 and len(val_df) > 1:
        test_df = val_df.iloc[len(val_df) // 2 :].copy()
        val_df = val_df.iloc[: len(val_df) // 2].copy()
    return train_df, val_df, test_df, df[df["__is_ood_record"]].copy()


def resolve_dataset_splits(df: pd.DataFrame):
    if "split" in df.columns:
        split_values = df["split"].fillna("").astype(str).str.strip().str.lower()
        return (
            df[split_values == "train"].copy(),
            df[split_values == "val"].copy(),
            df[split_values == "test_id"].copy(),
            df[split_values == "test_ood"].copy(),
            "split_column",
        )
    train_df, val_df, test_df, ood_df = chronological_split_id_records(df)
    return train_df, val_df, test_df, ood_df, "chronological_fallback"


def _empty_features(preprocessor: MetadataPreprocessor, length: int) -> np.ndarray:
    return np.zeros((length, len(preprocessor.feature_cols)), dtype=np.float32)


def _transform_or_empty(preprocessor: MetadataPreprocessor, df: pd.DataFrame) -> np.ndarray:
    if len(df) == 0:
        return _empty_features(preprocessor, 0)
    return preprocessor.transform(df)


def prepare_window_dataset(args, require_test_ood: bool = False) -> PreparedWindowDataset:
    args.group_col = (args.group_col or "").strip()
    if args.normalization_mode == "group" and not args.group_col:
        raise ValueError("group normalization requires --group_col")

    df = load_records(args.input)
    if args.record_id_col not in df.columns:
        df[args.record_id_col] = np.arange(len(df))
    if args.group_col and args.group_col not in df.columns:
        raise ValueError(f"group_col {args.group_col!r} not found in input data.")

    df = sort_records_for_windowing(df, group_col=args.group_col or None, timestamp_col=args.timestamp_col)
    df = attach_parsed_labels(df, args.label_col)
    all_labels = infer_all_labels(df, args.label_col)
    ood_classes = parse_csv_list(args.ood_classes)
    id_classes = parse_csv_list(args.id_classes)
    if not id_classes:
        id_classes = [label for label in all_labels if label not in set(ood_classes)]
    if not id_classes:
        raise ValueError("No ID classes found. Provide --id_classes or check label column.")
    class_to_idx = {name: idx for idx, name in enumerate(id_classes)}
    df = mark_ood_records(df, id_classes)

    train_df, val_df, test_df, ood_df, split_source = resolve_dataset_splits(df)
    split_frames = {
        "train": train_df,
        "val": val_df,
        "test_id": test_df,
        "test_ood": ood_df,
    }

    pre_cls = GroupAwareMetadataPreprocessor if args.normalization_mode == "group" else MetadataPreprocessor
    preprocessor = pre_cls(
        label_col=args.label_col,
        timestamp_col=args.timestamp_col,
        record_id_col=args.record_id_col,
        group_col=args.group_col or None,
        allow_ports=args.allow_ports,
    )
    preprocessor.fit(train_df)

    split_features = {
        name: _transform_or_empty(preprocessor, split_df)
        for name, split_df in split_frames.items()
    }
    split_windows = {
        "train": filter_id_windows(build_windows_by_mode(split_features["train"], train_df, class_to_idx, args)),
        "val": filter_id_windows(build_windows_by_mode(split_features["val"], val_df, class_to_idx, args)),
        "test_id": filter_id_windows(build_windows_by_mode(split_features["test_id"], test_df, class_to_idx, args)),
        "test_ood": build_windows_by_mode(split_features["test_ood"], ood_df, class_to_idx, args),
    }

    if len(split_windows["train"]) == 0:
        raise ValueError("Not enough ID windows for train split. Reduce window size or check the split.")
    if len(split_windows["val"]) == 0:
        raise ValueError("Not enough ID windows for val split. Reduce window size or check the split.")
    if len(split_windows["test_id"]) == 0:
        raise ValueError("Not enough ID windows for test_id split. Reduce window size or check the split.")
    if require_test_ood and len(split_windows["test_ood"]) == 0:
        raise ValueError("Not enough OOD windows for test_ood split. OOD evaluation requires real OOD windows.")

    return PreparedWindowDataset(
        df=df,
        split_source=split_source,
        split_frames=split_frames,
        split_windows=split_windows,
        id_classes=id_classes,
        ood_classes=ood_classes,
        class_to_idx=class_to_idx,
        preprocessor=preprocessor,
        normalization_summary=preprocessor.normalization_summary(),
        leakage_report=preprocessor.leakage_report(df),
        label_col=args.label_col,
        timestamp_col=args.timestamp_col,
        record_id_col=args.record_id_col,
        group_col=args.group_col or None,
        window_config={
            "mode": args.window_mode,
            "size": int(args.window_size),
            "stride": int(args.stride),
            "time_seconds": float(args.time_window_seconds),
            "adaptive_min_size": int(args.adaptive_min_size),
            "adaptive_max_size": int(args.adaptive_max_size),
        },
    )
