#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


LABEL_MAP = {
    "Normal": "benign",
    "DoS": "dos",
    "DDoS": "ddos",
    "Bruteforce": "bruteforce",
    "MITM": "mitm",
    "Reco": "recon_scanning",
    "Scanning": "recon_scanning",
    "Reply": "replay",
    "FakeLanding": "fake_landing",
    "Evil": "evil",
}

DEFAULT_ID_CLASSES = ["benign", "recon_scanning", "dos", "ddos", "bruteforce", "mitm"]
DEFAULT_OOD_CLASSES = ["replay", "fake_landing", "evil"]


def parse_csv_list(value: str) -> List[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def label_count_dict(df: pd.DataFrame, classes: Sequence[str] | None = None, label_col: str = "label") -> dict:
    if label_col not in df.columns:
        return {}
    raw_counts = {str(k): int(v) for k, v in df[label_col].astype(str).value_counts().to_dict().items()}
    if classes is None:
        return {k: raw_counts[k] for k in sorted(raw_counts)}
    ordered: dict[str, int] = {}
    seen = set()
    for cls in classes:
        key = str(cls)
        ordered[key] = int(raw_counts.get(key, 0))
        seen.add(key)
    for key in sorted(raw_counts):
        if key not in seen:
            ordered[key] = int(raw_counts[key])
    return ordered


def normalise_label(value: str) -> str:
    raw = str(value).strip()
    if raw in LABEL_MAP:
        return LABEL_MAP[raw]
    return raw.replace(" ", "_").replace("-", "_").lower()


def fingerprint_frame(df: pd.DataFrame, feature_cols: Sequence[str]) -> pd.Series:
    return pd.util.hash_pandas_object(df.loc[:, feature_cols], index=False)


def drop_ambiguous_feature_patterns(df: pd.DataFrame, feature_cols: Sequence[str]) -> tuple[pd.DataFrame, int]:
    fp = fingerprint_frame(df, feature_cols)
    label_counts = df.groupby(fp)["label"].nunique()
    ambiguous = set(label_counts[label_counts > 1].index.tolist())
    if not ambiguous:
        return df.copy(), 0
    keep = ~fp.isin(ambiguous)
    dropped = int((~keep).sum())
    return df.loc[keep].copy(), dropped


def interleave_by_label(df: pd.DataFrame, label_col: str, chunk_size: int, seed: int) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    rng = np.random.default_rng(seed)
    queues: Dict[str, List[int]] = {}
    for label, group in df.groupby(label_col, sort=True):
        idx = group.index.to_numpy(copy=True)
        rng.shuffle(idx)
        queues[str(label)] = idx.tolist()
    order: List[int] = []
    labels = sorted(queues)
    while True:
        active = [label for label in labels if queues[label]]
        if not active:
            break
        rng.shuffle(active)
        for label in active:
            take = min(chunk_size, len(queues[label]))
            order.extend(queues[label][:take])
            del queues[label][:take]
    return df.loc[order].reset_index(drop=True)


def window_mix_stats(labels: Iterable[str], window_size: int = 16, stride: int = 8) -> dict:
    seq = list(labels)
    total = 0
    mixed = 0
    pure = 0
    for start in range(0, max(len(seq) - window_size + 1, 0), stride):
        total += 1
        uniq = set(seq[start:start + window_size])
        if len(uniq) > 1:
            mixed += 1
        else:
            pure += 1
    return {
        "window_size": int(window_size),
        "stride": int(stride),
        "total_windows": int(total),
        "mixed_windows": int(mixed),
        "pure_windows": int(pure),
        "mixed_ratio": float(mixed / total) if total else 0.0,
    }


def exact_id_split(
    df: pd.DataFrame,
    label_col: str,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n = len(df)
    n_train = max(1, int(n * train_ratio))
    n_val = max(1, int(n * val_ratio))
    n_test = n - n_train - n_val
    if n_test < 1:
        raise ValueError("ID data is too small for the requested split ratios.")
    train_idx, temp_idx = train_test_split(
        df.index.to_numpy(),
        train_size=n_train,
        stratify=df[label_col].to_numpy(),
        random_state=seed,
    )
    temp = df.loc[temp_idx]
    val_idx, test_idx = train_test_split(
        temp.index.to_numpy(),
        train_size=n_val,
        test_size=n_test,
        stratify=temp[label_col].to_numpy(),
        random_state=seed + 1,
    )
    return df.loc[train_idx].copy(), df.loc[val_idx].copy(), df.loc[test_idx].copy()


def summarise_split(df: pd.DataFrame, classes: Sequence[str]) -> dict:
    return {
        "rows": int(len(df)),
        "label_counts": label_count_dict(df, classes),
        "mixed_window_stats": window_mix_stats(df["label"].tolist()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare an experiment-ready UCS-OODID CSV from the GCS-to-UAV Updated dataset.")
    parser.add_argument(
        "--input",
        default=r"..\..\GCS-to-UAV Updated\Nomral+Attacks\Normal+Attacks11.csv",
        help="Path to the merged CSV shipped with GCS-to-UAV Updated.",
    )
    parser.add_argument(
        "--output",
        default=r"data\gcs_to_uav_updated_experiment.csv",
        help="Path to the prepared output CSV.",
    )
    parser.add_argument(
        "--notes_json",
        default=r"data\gcs_to_uav_updated_experiment_notes.json",
        help="Path to the preparation summary JSON.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chunk_size", type=int, default=4, help="How many same-label records to emit before switching labels in the pseudo-stream.")
    parser.add_argument("--train_ratio", type=float, default=0.70)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--id_classes", default=",".join(DEFAULT_ID_CLASSES))
    parser.add_argument("--ood_classes", default=",".join(DEFAULT_OOD_CLASSES))
    parser.add_argument("--keep_duplicates", action="store_true", help="Do not remove exact duplicate rows after label normalization.")
    parser.add_argument(
        "--keep_conflicting_patterns",
        action="store_true",
        help="Do not drop feature patterns that appear under multiple cleaned labels.",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    workspace_root = project_root.parent
    input_path = Path(args.input)
    if not input_path.is_absolute():
        candidate_paths = [
            (script_dir / input_path).resolve(),
            (project_root / input_path).resolve(),
            (workspace_root / input_path).resolve(),
        ]
        for candidate in candidate_paths:
            if candidate.exists():
                input_path = candidate
                break
        else:
            input_path = candidate_paths[0]
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = (project_root / output_path).resolve()
    notes_path = Path(args.notes_json)
    if not notes_path.is_absolute():
        notes_path = (project_root / notes_path).resolve()

    id_classes = parse_csv_list(args.id_classes)
    ood_classes = parse_csv_list(args.ood_classes)
    known_classes = set(id_classes) | set(ood_classes)
    ordered_classes = list(dict.fromkeys(id_classes + ood_classes))

    df = pd.read_csv(input_path)
    if "label" not in df.columns:
        raise ValueError("Expected a 'label' column in the input CSV.")

    work = df.copy()
    work["source_label"] = work["label"].astype(str).str.strip()
    work["label"] = work["source_label"].map(normalise_label)
    class_counts_before_cleaning = label_count_dict(work, ordered_classes)

    numeric_feature_cols = [c for c in work.columns if c not in {"label", "source_label"} and pd.api.types.is_numeric_dtype(work[c])]
    if not numeric_feature_cols:
        raise ValueError("No numeric feature columns found after loading the dataset.")

    ambiguous_dropped = 0
    if not args.keep_conflicting_patterns:
        work, ambiguous_dropped = drop_ambiguous_feature_patterns(work, numeric_feature_cols)

    duplicate_rows_removed = 0
    if not args.keep_duplicates:
        before = len(work)
        work = work.drop_duplicates(subset=numeric_feature_cols + ["label"]).copy()
        duplicate_rows_removed = before - len(work)
    class_counts_after_cleaning = label_count_dict(work, ordered_classes)

    unknown_labels = sorted(set(work["label"]) - known_classes)
    if unknown_labels:
        raise ValueError(f"Found labels with no ID/OOD assignment: {unknown_labels}")

    work["recommended_partition"] = np.where(work["label"].isin(id_classes), "id", "ood")
    id_df = work[work["recommended_partition"] == "id"].copy()
    ood_df = work[work["recommended_partition"] == "ood"].copy()
    if id_df.empty or ood_df.empty:
        raise ValueError("Both ID and OOD partitions must be non-empty.")

    train_df, val_df, test_df = exact_id_split(id_df, "label", args.train_ratio, args.val_ratio, args.seed)
    train_df = interleave_by_label(train_df, "label", args.chunk_size, args.seed + 10)
    val_df = interleave_by_label(val_df, "label", args.chunk_size, args.seed + 20)
    test_df = interleave_by_label(test_df, "label", args.chunk_size, args.seed + 30)
    ood_df = interleave_by_label(ood_df, "label", args.chunk_size, args.seed + 40)

    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test_id"
    ood_df["split"] = "test_ood"
    split_frames = {
        "train": train_df,
        "val": val_df,
        "test_id": test_df,
        "test_ood": ood_df,
    }
    final_split_rows = {name: int(len(frame)) for name, frame in split_frames.items()}
    split_label_counts = {name: label_count_dict(frame, ordered_classes) for name, frame in split_frames.items()}
    ood_holdout_class_counts = label_count_dict(ood_df, ood_classes)

    prepared = pd.concat([train_df, val_df, test_df, ood_df], ignore_index=True)
    prepared["timestamp"] = np.arange(len(prepared), dtype=np.float64)
    prepared["record_id"] = np.arange(len(prepared), dtype=np.int64)

    ordered_cols = numeric_feature_cols + ["label", "source_label", "recommended_partition", "split", "timestamp", "record_id"]
    prepared = prepared.loc[:, ordered_cols]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    notes_path.parent.mkdir(parents=True, exist_ok=True)
    prepared.to_csv(output_path, index=False)

    summary = {
        "input_csv": str(input_path),
        "output_csv": str(output_path),
        "notes_json": str(notes_path),
        "raw_sample_count": int(len(df)),
        "input_rows": int(len(df)),
        "output_rows": int(len(prepared)),
        "numeric_feature_columns": numeric_feature_cols,
        "label_map": LABEL_MAP,
        "duplicate_rows_removed": int(duplicate_rows_removed),
        "ambiguous_rows_removed": int(ambiguous_dropped),
        "rows_after_cleaning": int(len(work)),
        "class_counts_before_cleaning": class_counts_before_cleaning,
        "class_counts_after_cleaning": class_counts_after_cleaning,
        "id_classes": id_classes,
        "ood_classes": ood_classes,
        "ood_holdout": {
            "classes": ood_classes,
            "class_counts": ood_holdout_class_counts,
            "rows": int(len(ood_df)),
        },
        "final_split_rows": final_split_rows,
        "split_label_counts": split_label_counts,
        "split_summary": {
            "train": summarise_split(prepared[prepared["split"] == "train"], ordered_classes),
            "val": summarise_split(prepared[prepared["split"] == "val"], ordered_classes),
            "test_id": summarise_split(prepared[prepared["split"] == "test_id"], ordered_classes),
            "test_ood": summarise_split(prepared[prepared["split"] == "test_ood"], ordered_classes),
        },
        "full_label_counts": {str(k): int(v) for k, v in prepared["label"].value_counts().to_dict().items()},
        "preparation_config": {
            "synthetic_timestamp_generated": True,
            "interleave_by_label": True,
            "chunk_size": int(args.chunk_size),
            "random_seed": int(args.seed),
            "split_random_seeds": {"train_val_test_split": int(args.seed), "val_test_split": int(args.seed + 1)},
            "interleave_random_seeds": {
                "train": int(args.seed + 10),
                "val": int(args.seed + 20),
                "test_id": int(args.seed + 30),
                "test_ood": int(args.seed + 40),
            },
            "keep_duplicates": bool(args.keep_duplicates),
            "keep_conflicting_patterns": bool(args.keep_conflicting_patterns),
            "train_ratio": float(args.train_ratio),
            "val_ratio": float(args.val_ratio),
        },
        "notes": [
            "The file is ordered as ID-train -> ID-val -> ID-test -> OOD-test so scripts/train.py can reuse its chronological ID split safely.",
            "A synthetic monotonic timestamp was created because the source dataset has no real timestamp column.",
            "Reco and Scanning were merged into recon_scanning because their unique feature patterns are identical in the merged source CSV.",
            "Exact duplicates were removed by default to reduce trivial leakage across splits.",
            "Rows with identical feature patterns but conflicting cleaned labels were removed by default.",
        ],
        "recommended_train_command": (
            "python scripts/train.py "
            f"--input {output_path} "
            "--output_dir runs/gcs_to_uav_updated_default "
            "--label_col label "
            "--timestamp_col timestamp "
            "--record_id_col record_id "
            f"--id_classes {','.join(id_classes)} "
            f"--ood_classes {','.join(ood_classes)} "
            "--window_mode count --window_size 16 --stride 8 --epochs 10"
        ),
    }
    notes_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    console_summary = {
        "input_rows": int(len(df)),
        "ambiguous_rows_removed": int(ambiguous_dropped),
        "duplicate_rows_removed": int(duplicate_rows_removed),
        "rows_after_cleaning": int(len(work)),
        "class_counts_before_cleaning": class_counts_before_cleaning,
        "class_counts_after_cleaning": class_counts_after_cleaning,
        "ood_holdout_class_counts": ood_holdout_class_counts,
        "final_split_rows": final_split_rows,
        "synthetic_timestamp_generated": True,
        "interleave_by_label": True,
        "chunk_size": int(args.chunk_size),
        "random_seed": int(args.seed),
        "notes_json": str(notes_path),
    }
    print(json.dumps({"prepare_gcs_to_uav_updated_summary": console_summary}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
