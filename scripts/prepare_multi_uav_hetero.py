#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, NamedTuple, Sequence

import numpy as np
import pandas as pd

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None

DEFAULT_CONFIG_PATH = Path("configs") / "datasets.yaml"
DEFAULT_SUMMARY_FILENAME = "merged_dataset_summary.json"
DEFAULT_LABEL_CANDIDATES = (
    "label",
    "source_label",
    "label_normalized",
    "attack_cat",
    "attack_type",
    "class",
    "Type of Attack",
)
DEFAULT_LABEL_NORMALIZED_CANDIDATES = ("label_normalized",)
DEFAULT_TIMESTAMP_CANDIDATES = ("timestamp",)
DEFAULT_SPLIT_CANDIDATES = ("split",)
DEFAULT_RECORD_ID_CANDIDATES = ("record_id",)
FIXED_NON_FEATURE_COLUMNS = {
    "label",
    "label_normalized",
    "source_label",
    "source_type",
    "recommended_partition",
    "split",
    "timestamp",
    "record_id",
    "uav_id",
    "dataset_name",
    "__labels",
    "__is_ood_record",
}
REQUIRED_METADATA_COLUMNS = [
    "record_id",
    "timestamp",
    "uav_id",
    "dataset_name",
    "source_type",
    "label",
    "label_normalized",
    "split",
]
LEGACY_SOURCE_TYPE_BY_DATASET = {
    "unsw_nb15": "external_non_uav",
    "ecu_ioft": "uav_iot_wifi",
    "uavids": "uav",
}


class DatasetConfig(NamedTuple):
    uav_id: str
    dataset_name: str
    csv_path: str
    source_type: str | None = None
    label_column: str | None = None
    label_normalized_column: str | None = None
    split_column: str | None = None
    timestamp_column: str | None = None
    record_id_column: str | None = None


class PreparedDatasetSpec(NamedTuple):
    dataframe: pd.DataFrame
    uav_id: str
    dataset_name: str
    source_type: str | None = None


def ordered_unique(values: Sequence[str]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for value in values:
        key = str(value)
        if key not in seen:
            ordered.append(key)
            seen.add(key)
    return ordered


def resolve_existing_path(base_candidates: Sequence[Path], value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    for base in base_candidates:
        candidate = (base / path).resolve()
        if candidate.exists():
            return candidate
    return (base_candidates[0] / path).resolve()


def resolve_output_path(project_root: Path, value: str | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return (project_root / path).resolve()


def normalize_uav_id_list(values: Sequence[str] | str | None) -> list[str]:
    if values is None:
        return []
    if isinstance(values, str):
        raw_values = values.split(",")
    else:
        raw_values = []
        for value in values:
            raw_values.extend(str(value).split(","))
    normalized: list[str] = []
    seen: set[str] = set()
    for value in raw_values:
        text = str(value).strip()
        if text and text not in seen:
            normalized.append(text)
            seen.add(text)
    return normalized


def require_unique_uav_ids(datasets: Sequence[PreparedDatasetSpec | DatasetConfig]) -> None:
    seen: set[str] = set()
    duplicates: list[str] = []
    for spec in datasets:
        if spec.uav_id in seen and spec.uav_id not in duplicates:
            duplicates.append(spec.uav_id)
        seen.add(spec.uav_id)
    if duplicates:
        raise ValueError(f"Duplicate uav_id values are not allowed: {duplicates}")


def require_unique_dataset_names(datasets: Sequence[PreparedDatasetSpec | DatasetConfig]) -> None:
    seen: set[str] = set()
    duplicates: list[str] = []
    for spec in datasets:
        if spec.dataset_name in seen and spec.dataset_name not in duplicates:
            duplicates.append(spec.dataset_name)
        seen.add(spec.dataset_name)
    if duplicates:
        raise ValueError(f"Duplicate dataset_name values are not allowed: {duplicates}")


def build_non_feature_columns(
    label_col: str,
    split_col: str,
    timestamp_col: str,
    record_id_col: str,
) -> set[str]:
    names = set(FIXED_NON_FEATURE_COLUMNS)
    names.update({label_col, split_col, timestamp_col, record_id_col})
    return {str(name) for name in names if str(name).strip()}


def candidate_feature_columns(df: pd.DataFrame, non_feature_columns: set[str]) -> list[str]:
    return [
        str(column)
        for column in df.columns
        if str(column) not in non_feature_columns and pd.api.types.is_numeric_dtype(df[column])
    ]


def require_columns(df: pd.DataFrame, required: Sequence[str], dataset_tag: str) -> None:
    missing = [str(column) for column in required if str(column) not in df.columns]
    if missing:
        raise ValueError(f"{dataset_tag} is missing required columns: {missing}")


def build_prefixed_record_ids(df: pd.DataFrame, record_id_col: str, uav_id: str) -> list[str]:
    if record_id_col in df.columns:
        source_ids = df[record_id_col]
    else:
        source_ids = pd.Series(np.arange(len(df), dtype=np.int64), index=df.index)
    prefixed: list[str] = []
    prefix = f"{uav_id}_"
    for idx, value in enumerate(source_ids.tolist()):
        if pd.isna(value):
            suffix = str(idx)
        else:
            text = str(value).strip()
            if text.startswith(prefix):
                prefixed.append(text)
                continue
            suffix = text if text else str(idx)
        prefixed.append(f"{uav_id}_{suffix}")
    return prefixed


def attach_dataset_identity(
    df: pd.DataFrame,
    uav_id: str,
    dataset_name: str,
    record_id_col: str,
    source_type: str | None = None,
) -> pd.DataFrame:
    out = df.copy()
    out["uav_id"] = uav_id
    out["dataset_name"] = dataset_name
    if source_type is not None:
        out["source_type"] = source_type
    out[record_id_col] = build_prefixed_record_ids(out, record_id_col, uav_id)
    return out


def align_feature_union(df: pd.DataFrame, feature_union: Sequence[str]) -> tuple[pd.DataFrame, list[str]]:
    out = df.copy()
    missing_features: list[str] = []
    aligned_feature_data: dict[str, pd.Series] = {}
    for column in feature_union:
        if column in out.columns:
            numeric = pd.to_numeric(out[column], errors="coerce")
        else:
            numeric = pd.Series(0.0, index=out.index, dtype=np.float64)
            missing_features.append(str(column))
        numeric = numeric.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float64)
        aligned_feature_data[str(column)] = numeric
    metadata = out.drop(columns=[column for column in feature_union if column in out.columns], errors="ignore")
    aligned_features = pd.DataFrame(aligned_feature_data, index=out.index)
    out = pd.concat([metadata, aligned_features], axis=1)
    return out, missing_features


def value_counts_dict(df: pd.DataFrame, column: str) -> dict[str, int]:
    if column not in df.columns:
        return {}
    counts = df[column].astype(str).value_counts().to_dict()
    return {str(key): int(value) for key, value in counts.items()}


def grouped_value_counts(df: pd.DataFrame, group_col: str, value_col: str) -> dict[str, dict[str, int]]:
    if group_col not in df.columns or value_col not in df.columns:
        return {}
    summary: dict[str, dict[str, int]] = {}
    for group_value, group_df in df.groupby(group_col, sort=True):
        summary[str(group_value)] = value_counts_dict(group_df, value_col)
    return summary


def dataset_tag(config: DatasetConfig) -> str:
    return f"{config.dataset_name} ({config.uav_id})"


def clean_string_series(series: pd.Series, *, dataset_name: str, column_name: str) -> pd.Series:
    cleaned = series.astype("string").fillna("").str.strip()
    blank_mask = cleaned.eq("")
    if bool(blank_mask.any()):
        raise ValueError(
            f"{dataset_name} contains {int(blank_mask.sum())} blank values in column '{column_name}'. "
            "Please fix the dataset before merging."
        )
    return cleaned.astype(str)


def cleaned_string_or_none(value: object | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def resolve_config_entry_path(row: dict[str, Any]) -> str | None:
    for key in ("csv", "csv_path", "path", "input_csv"):
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def load_dataset_configs(config_path: Path) -> list[DatasetConfig]:
    if yaml is None:
        raise RuntimeError("pyyaml is required to load configs/datasets.yaml")
    if not config_path.exists():
        raise FileNotFoundError(f"Dataset config file was not found: {config_path}")

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    dataset_rows = raw.get("datasets", raw) if isinstance(raw, dict) else raw
    if isinstance(dataset_rows, dict):
        rows: list[dict[str, Any]] = []
        for uav_id, value in dataset_rows.items():
            row = dict(value or {})
            row.setdefault("uav_id", str(uav_id))
            rows.append(row)
    elif isinstance(dataset_rows, list):
        rows = [dict(value or {}) for value in dataset_rows]
    else:
        raise ValueError(
            f"{config_path} must contain a 'datasets' list or mapping, but found {type(dataset_rows).__name__}."
        )

    configs: list[DatasetConfig] = []
    for idx, row in enumerate(rows, start=1):
        uav_id = cleaned_string_or_none(row.get("uav_id"))
        dataset_name = cleaned_string_or_none(row.get("dataset_name"))
        csv_path = resolve_config_entry_path(row)
        if uav_id is None:
            raise ValueError(f"{config_path}: datasets entry #{idx} is missing 'uav_id'.")
        if dataset_name is None:
            raise ValueError(f"{config_path}: datasets entry #{idx} is missing 'dataset_name'.")
        if csv_path is None:
            raise ValueError(f"{config_path}: datasets entry #{idx} is missing a CSV path field such as 'csv'.")
        configs.append(
            DatasetConfig(
                uav_id=uav_id,
                dataset_name=dataset_name,
                csv_path=csv_path,
                source_type=cleaned_string_or_none(row.get("source_type")),
                label_column=cleaned_string_or_none(row.get("label_column")),
                label_normalized_column=cleaned_string_or_none(row.get("label_normalized_column")),
                split_column=cleaned_string_or_none(row.get("split_column")),
                timestamp_column=cleaned_string_or_none(row.get("timestamp_column")),
                record_id_column=cleaned_string_or_none(row.get("record_id_column")),
            )
        )
    return configs


def resolve_dataset_column(
    df: pd.DataFrame,
    *,
    explicit: str | None,
    candidates: Sequence[str],
    dataset_name: str,
    purpose: str,
    required: bool,
) -> str | None:
    columns = {str(column): column for column in df.columns}
    if explicit is not None:
        if explicit in columns:
            return explicit
        raise ValueError(
            f"{dataset_name} configured {purpose}_column '{explicit}', but that column was not found. "
            f"Available columns: {list(map(str, df.columns))}"
        )
    for candidate in candidates:
        if candidate in columns:
            return candidate
    if required:
        raise ValueError(
            f"{dataset_name} is missing a recognizable {purpose} column. "
            f"Tried {list(candidates)}. Available columns: {list(map(str, df.columns))}"
        )
    return None


def build_timestamp_series(df: pd.DataFrame, column_name: str | None) -> pd.Series:
    if column_name is None or column_name not in df.columns:
        return pd.Series(np.arange(len(df), dtype=np.int64), index=df.index)
    return df[column_name]


def build_split_series(df: pd.DataFrame, column_name: str | None) -> pd.Series:
    if column_name is None or column_name not in df.columns:
        return pd.Series(["all"] * len(df), index=df.index, dtype="string")
    series = df[column_name].astype("string").fillna("").str.strip().str.lower()
    return series.mask(series.eq(""), "all").astype(str)


def infer_source_type(df: pd.DataFrame, config: DatasetConfig) -> str:
    tag = dataset_tag(config)
    configured = cleaned_string_or_none(config.source_type)
    if "source_type" not in df.columns:
        if configured is None:
            raise ValueError(
                f"{tag} is missing source_type. Add a source_type column in the CSV or set source_type in configs/datasets.yaml."
            )
        return configured

    series = clean_string_series(df["source_type"], dataset_name=tag, column_name="source_type")
    unique_values = ordered_unique(series.tolist())
    if len(unique_values) != 1:
        raise ValueError(f"{tag} contains multiple source_type values: {unique_values}")
    detected = unique_values[0]
    if configured is not None and configured != detected:
        raise ValueError(f"{tag} source_type mismatch: config has '{configured}' but CSV has '{detected}'.")
    return detected


def standardize_dataset_frame(df: pd.DataFrame, config: DatasetConfig) -> pd.DataFrame:
    tag = dataset_tag(config)
    label_column = resolve_dataset_column(
        df,
        explicit=config.label_column,
        candidates=DEFAULT_LABEL_CANDIDATES,
        dataset_name=tag,
        purpose="label",
        required=False,
    )
    label_normalized_column = resolve_dataset_column(
        df,
        explicit=config.label_normalized_column,
        candidates=DEFAULT_LABEL_NORMALIZED_CANDIDATES,
        dataset_name=tag,
        purpose="label_normalized",
        required=False,
    )
    source_label_column = "source_label" if "source_label" in df.columns else None

    if label_normalized_column is not None:
        resolved_label_column = label_column or source_label_column or label_normalized_column
        resolved_label_normalized_column = label_normalized_column
    elif label_column is not None and source_label_column is not None and label_column == "label":
        resolved_label_column = source_label_column
        resolved_label_normalized_column = label_column
    elif label_column is not None:
        resolved_label_column = label_column
        resolved_label_normalized_column = label_column
    elif label_normalized_column is not None:
        resolved_label_column = source_label_column or label_normalized_column
        resolved_label_normalized_column = label_normalized_column
    else:
        raise ValueError(
            f"{tag} is missing a recognizable label column. "
            f"Tried {list(DEFAULT_LABEL_CANDIDATES)} and {list(DEFAULT_LABEL_NORMALIZED_CANDIDATES)}. "
            f"Available columns: {list(map(str, df.columns))}"
        )

    record_id_column = resolve_dataset_column(
        df,
        explicit=config.record_id_column,
        candidates=DEFAULT_RECORD_ID_CANDIDATES,
        dataset_name=tag,
        purpose="record_id",
        required=False,
    )
    timestamp_column = resolve_dataset_column(
        df,
        explicit=config.timestamp_column,
        candidates=DEFAULT_TIMESTAMP_CANDIDATES,
        dataset_name=tag,
        purpose="timestamp",
        required=False,
    )
    split_column = resolve_dataset_column(
        df,
        explicit=config.split_column,
        candidates=DEFAULT_SPLIT_CANDIDATES,
        dataset_name=tag,
        purpose="split",
        required=False,
    )
    source_type = infer_source_type(df, config)

    label_series = clean_string_series(
        df[resolved_label_column],
        dataset_name=tag,
        column_name=resolved_label_column,
    )
    label_normalized_series = clean_string_series(
        df[resolved_label_normalized_column],
        dataset_name=tag,
        column_name=resolved_label_normalized_column,
    )
    record_id_series = (
        df[record_id_column]
        if record_id_column is not None and record_id_column in df.columns
        else pd.Series(np.arange(len(df), dtype=np.int64), index=df.index)
    )

    metadata = pd.DataFrame(
        {
            "record_id": record_id_series,
            "timestamp": build_timestamp_series(df, timestamp_column),
            "uav_id": [config.uav_id] * len(df),
            "dataset_name": [config.dataset_name] * len(df),
            "source_type": [source_type] * len(df),
            "label": label_series,
            "label_normalized": label_normalized_series,
            "split": build_split_series(df, split_column),
        },
        index=df.index,
    )

    excluded_columns = build_non_feature_columns(
        resolved_label_column,
        split_column or "split",
        timestamp_column or "timestamp",
        record_id_column or "record_id",
    )
    excluded_columns.update(
        {
            str(resolved_label_normalized_column),
            str(source_label_column or ""),
            "source_type",
            "uav_id",
            "dataset_name",
        }
    )
    remaining_columns = [column for column in df.columns if str(column) not in excluded_columns]
    return pd.concat(
        [metadata.reset_index(drop=True), df.loc[:, remaining_columns].reset_index(drop=True)],
        axis=1,
    )


def load_standardized_dataset(config: DatasetConfig, base_candidates: Sequence[Path]) -> tuple[PreparedDatasetSpec, Path]:
    csv_path = resolve_existing_path(base_candidates, config.csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"{dataset_tag(config)} CSV was not found: {csv_path}")
    raw_df = pd.read_csv(csv_path, low_memory=False)
    standardized_df = standardize_dataset_frame(raw_df, config)
    return (
        PreparedDatasetSpec(
            dataframe=standardized_df,
            uav_id=config.uav_id,
            dataset_name=config.dataset_name,
            source_type=standardized_df["source_type"].iloc[0] if len(standardized_df) else config.source_type,
        ),
        csv_path,
    )


def select_dataset_configs(
    dataset_configs: Sequence[DatasetConfig],
    include_uavs: Sequence[str] | str | None = None,
    exclude_uav_ids: Sequence[str] | str | None = None,
) -> tuple[list[DatasetConfig], list[str]]:
    require_unique_uav_ids(dataset_configs)
    require_unique_dataset_names(dataset_configs)

    by_uav_id = {config.uav_id: config for config in dataset_configs}
    selected_uavs = normalize_uav_id_list(include_uavs)
    if selected_uavs:
        unknown_uavs = [uav_id for uav_id in selected_uavs if uav_id not in by_uav_id]
        if unknown_uavs:
            raise ValueError(
                f"Unknown UAV IDs in --include_uavs: {unknown_uavs}. "
                f"Available UAV IDs: {list(by_uav_id.keys())}"
            )
        selected = [by_uav_id[uav_id] for uav_id in selected_uavs]
    else:
        selected = list(dataset_configs)

    excluded_uavs = normalize_uav_id_list(exclude_uav_ids)
    if excluded_uavs:
        exclude_set = set(excluded_uavs)
        selected = [config for config in selected if config.uav_id not in exclude_set]
    if not selected:
        raise ValueError("No datasets remain after applying include/exclude UAV filters.")
    require_unique_uav_ids(selected)
    require_unique_dataset_names(selected)
    return selected, excluded_uavs


def merge_named_prepared_frames(
    datasets: Sequence[PreparedDatasetSpec],
    label_col: str = "label",
    split_col: str = "split",
    timestamp_col: str = "timestamp",
    record_id_col: str = "record_id",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not datasets:
        raise ValueError("At least one prepared dataset is required to build a heterogeneous multi-UAV CSV.")

    non_feature_columns = build_non_feature_columns(label_col, split_col, timestamp_col, record_id_col)
    feature_union: list[str] = []
    for spec in datasets:
        require_columns(spec.dataframe, [label_col, "label_normalized", split_col, "source_type"], spec.dataset_name)
        feature_union.extend(candidate_feature_columns(spec.dataframe, non_feature_columns))
    feature_union = ordered_unique(feature_union)
    if not feature_union:
        raise ValueError("No numeric feature columns were found across the prepared input CSV files.")

    aligned_frames: list[pd.DataFrame] = []
    input_rows: dict[str, int] = {}
    missing_features_filled: dict[str, list[str]] = {}
    for spec in datasets:
        ready = attach_dataset_identity(
            spec.dataframe,
            uav_id=spec.uav_id,
            dataset_name=spec.dataset_name,
            record_id_col=record_id_col,
            source_type=cleaned_string_or_none(spec.source_type),
        )
        aligned, missing = align_feature_union(ready, feature_union)
        aligned_frames.append(aligned)
        input_rows[spec.dataset_name] = int(len(spec.dataframe))
        missing_features_filled[spec.dataset_name] = list(missing)

    merged = pd.concat(aligned_frames, ignore_index=True)
    output_columns = list(feature_union) + [column for column in REQUIRED_METADATA_COLUMNS if column in merged.columns]
    merged = merged.loc[:, output_columns]

    summary: dict[str, Any] = {
        "input_rows": input_rows,
        "output_rows": int(len(merged)),
        "feature_count": int(len(feature_union)),
        "feature_columns": list(feature_union),
        "missing_features_filled": missing_features_filled,
        "sample_count_by_uav_id": value_counts_dict(merged, "uav_id"),
        "sample_count_by_dataset_name": value_counts_dict(merged, "dataset_name"),
        "sample_count_by_label_normalized": value_counts_dict(merged, "label_normalized"),
        "sample_count_by_source_type": value_counts_dict(merged, "source_type"),
    }
    return merged, summary


def infer_legacy_source_type(dataset_name: str) -> str:
    return LEGACY_SOURCE_TYPE_BY_DATASET.get(str(dataset_name).strip(), "uav")


def build_legacy_dataset_configs_from_values(
    *,
    uav_ndd_csv: str | None = None,
    gcs_csv: str | None = None,
    third_csv: str | None = None,
    third_uav_id: str = "uav_03",
    third_dataset_name: str = "isot_drone",
    fourth_csv: str | None = None,
    fourth_uav_id: str = "uav_04",
    fourth_dataset_name: str = "uavs_normal_cyberattacks",
    fifth_csv: str | None = None,
    fifth_uav_id: str = "uav_05",
    fifth_dataset_name: str = "unsw_nb15",
    sixth_csv: str | None = None,
    sixth_uav_id: str = "uav_06",
    sixth_dataset_name: str = "ecu_ioft",
    seventh_csv: str | None = None,
    seventh_uav_id: str = "uav_07",
    seventh_dataset_name: str = "uavids",
    label_col: str | None = None,
    split_col: str | None = None,
    timestamp_col: str | None = None,
    record_id_col: str | None = None,
) -> list[DatasetConfig]:
    label_column = cleaned_string_or_none(label_col)
    split_column = cleaned_string_or_none(split_col)
    timestamp_column = cleaned_string_or_none(timestamp_col)
    record_id_column = cleaned_string_or_none(record_id_col)

    configs: list[DatasetConfig] = []
    legacy_rows = [
        (uav_ndd_csv, "uav_01", "uav_ndd"),
        (gcs_csv, "uav_02", "gcs_to_uav_updated"),
        (third_csv, third_uav_id, third_dataset_name),
        (fourth_csv, fourth_uav_id, fourth_dataset_name),
        (fifth_csv, fifth_uav_id, fifth_dataset_name),
        (sixth_csv, sixth_uav_id, sixth_dataset_name),
        (seventh_csv, seventh_uav_id, seventh_dataset_name),
    ]
    for csv_path, uav_id, dataset_name in legacy_rows:
        resolved_csv_path = cleaned_string_or_none(csv_path)
        if resolved_csv_path is None:
            continue
        configs.append(
            DatasetConfig(
                uav_id=str(uav_id).strip(),
                dataset_name=str(dataset_name).strip(),
                csv_path=resolved_csv_path,
                source_type=infer_legacy_source_type(str(dataset_name).strip()),
                label_column=label_column,
                split_column=split_column,
                timestamp_column=timestamp_column,
                record_id_column=record_id_column,
            )
        )
    if configs and not cleaned_string_or_none(uav_ndd_csv):
        raise ValueError("--uav_ndd_csv is required when using legacy CSV arguments.")
    if configs and not cleaned_string_or_none(gcs_csv):
        raise ValueError("--gcs_csv is required when using legacy CSV arguments.")
    return configs


def merge_prepared_frames(
    uav_ndd_df: pd.DataFrame,
    gcs_df: pd.DataFrame,
    label_col: str = "label",
    split_col: str = "split",
    timestamp_col: str = "timestamp",
    record_id_col: str = "record_id",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    return merge_named_prepared_frames(
        [
            PreparedDatasetSpec(dataframe=uav_ndd_df, uav_id="uav_01", dataset_name="uav_ndd", source_type="uav"),
            PreparedDatasetSpec(
                dataframe=gcs_df,
                uav_id="uav_02",
                dataset_name="gcs_to_uav_updated",
                source_type="uav",
            ),
        ],
        label_col=label_col,
        split_col=split_col,
        timestamp_col=timestamp_col,
        record_id_col=record_id_col,
    )


def resolve_summary_path(
    *,
    project_root: Path,
    output_path: Path,
    summary_json: str | None,
    notes_json: str | None,
) -> Path:
    if summary_json and notes_json and str(summary_json).strip() != str(notes_json).strip():
        raise ValueError("summary_json and notes_json refer to different paths. Please provide only one of them.")
    target = summary_json or notes_json
    if target is None:
        return output_path.with_name(DEFAULT_SUMMARY_FILENAME)
    resolved = resolve_output_path(project_root, target)
    if resolved is None:
        raise ValueError("Could not resolve the summary JSON path.")
    return resolved


def prepare_multi_uav_hetero_dataset(
    output: str,
    summary_json: str | None = None,
    notes_json: str | None = None,
    config_path: str | None = str(DEFAULT_CONFIG_PATH),
    include_uavs: Sequence[str] | str | None = None,
    exclude_uav_ids: Sequence[str] | str | None = None,
    dataset_configs: Sequence[DatasetConfig] | None = None,
    uav_ndd_csv: str | None = None,
    gcs_csv: str | None = None,
    third_csv: str | None = None,
    third_uav_id: str = "uav_03",
    third_dataset_name: str = "isot_drone",
    fourth_csv: str | None = None,
    fourth_uav_id: str = "uav_04",
    fourth_dataset_name: str = "uavs_normal_cyberattacks",
    fifth_csv: str | None = None,
    fifth_uav_id: str = "uav_05",
    fifth_dataset_name: str = "unsw_nb15",
    sixth_csv: str | None = None,
    sixth_uav_id: str = "uav_06",
    sixth_dataset_name: str = "ecu_ioft",
    seventh_csv: str | None = None,
    seventh_uav_id: str = "uav_07",
    seventh_dataset_name: str = "uavids",
    label_col: str | None = None,
    split_col: str | None = None,
    timestamp_col: str | None = None,
    record_id_col: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    workspace_root = project_root.parent
    output_path = resolve_output_path(project_root, output)
    if output_path is None:
        raise ValueError("output must be provided.")
    summary_path = resolve_summary_path(
        project_root=project_root,
        output_path=output_path,
        summary_json=summary_json,
        notes_json=notes_json,
    )

    resolved_config_path: Path | None = None
    if dataset_configs is None:
        dataset_configs = build_legacy_dataset_configs_from_values(
            uav_ndd_csv=uav_ndd_csv,
            gcs_csv=gcs_csv,
            third_csv=third_csv,
            third_uav_id=third_uav_id,
            third_dataset_name=third_dataset_name,
            fourth_csv=fourth_csv,
            fourth_uav_id=fourth_uav_id,
            fourth_dataset_name=fourth_dataset_name,
            fifth_csv=fifth_csv,
            fifth_uav_id=fifth_uav_id,
            fifth_dataset_name=fifth_dataset_name,
            sixth_csv=sixth_csv,
            sixth_uav_id=sixth_uav_id,
            sixth_dataset_name=sixth_dataset_name,
            seventh_csv=seventh_csv,
            seventh_uav_id=seventh_uav_id,
            seventh_dataset_name=seventh_dataset_name,
            label_col=label_col,
            split_col=split_col,
            timestamp_col=timestamp_col,
            record_id_col=record_id_col,
        )
        if not dataset_configs:
            if config_path is None or not str(config_path).strip():
                raise ValueError("config_path must be provided when dataset_configs is not supplied.")
            resolved_config_path = resolve_existing_path([script_dir, project_root, workspace_root], config_path)
            dataset_configs = load_dataset_configs(resolved_config_path)

    selected_configs, excluded_uavs = select_dataset_configs(
        dataset_configs,
        include_uavs=include_uavs,
        exclude_uav_ids=exclude_uav_ids,
    )

    load_base_candidates = [script_dir, project_root, workspace_root]
    if resolved_config_path is not None:
        load_base_candidates = [resolved_config_path.parent, *load_base_candidates]

    dataset_specs: list[PreparedDatasetSpec] = []
    input_csv: dict[str, str] = {}
    for config in selected_configs:
        spec, csv_path = load_standardized_dataset(config, load_base_candidates)
        dataset_specs.append(spec)
        input_csv[config.dataset_name] = str(csv_path)

    merged, summary = merge_named_prepared_frames(dataset_specs)
    included_uav_ids = [config.uav_id for config in selected_configs]
    included_dataset_names = [config.dataset_name for config in selected_configs]
    summary["config_path"] = str(resolved_config_path) if resolved_config_path is not None else None
    summary["input_csv"] = input_csv
    summary["excluded_uav_ids"] = excluded_uavs
    summary["included_uav_ids"] = included_uav_ids
    summary["included_dataset_names"] = included_dataset_names
    summary["output_csv"] = str(output_path)
    summary["summary_json"] = str(summary_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    console_summary = {
        "output_rows": summary["output_rows"],
        "feature_count": summary["feature_count"],
        "included_uav_ids": summary["included_uav_ids"],
        "sample_count_by_uav_id": summary["sample_count_by_uav_id"],
        "sample_count_by_dataset_name": summary["sample_count_by_dataset_name"],
        "sample_count_by_label_normalized": summary["sample_count_by_label_normalized"],
        "sample_count_by_source_type": summary["sample_count_by_source_type"],
        "summary_json": str(summary_path),
    }
    print(json.dumps({"prepare_multi_uav_hetero_summary": console_summary}, indent=2, ensure_ascii=False))
    return merged, summary


def build_legacy_dataset_configs(args: argparse.Namespace) -> list[DatasetConfig]:
    return build_legacy_dataset_configs_from_values(
        uav_ndd_csv=args.uav_ndd_csv,
        gcs_csv=args.gcs_csv,
        third_csv=args.third_csv,
        third_uav_id=args.third_uav_id,
        third_dataset_name=args.third_dataset_name,
        fourth_csv=args.fourth_csv,
        fourth_uav_id=args.fourth_uav_id,
        fourth_dataset_name=args.fourth_dataset_name,
        fifth_csv=args.fifth_csv,
        fifth_uav_id=args.fifth_uav_id,
        fifth_dataset_name=args.fifth_dataset_name,
        sixth_csv=args.sixth_csv,
        sixth_uav_id=args.sixth_uav_id,
        sixth_dataset_name=args.sixth_dataset_name,
        seventh_csv=args.seventh_csv,
        seventh_uav_id=args.seventh_uav_id,
        seventh_dataset_name=args.seventh_dataset_name,
        label_col=args.label_col,
        split_col=args.split_col,
        timestamp_col=args.timestamp_col,
        record_id_col=args.record_id_col,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Merge configured prepared UAV datasets into one heterogeneous multi-UAV dataset."
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to configs/datasets.yaml. Relative CSV paths inside the config are resolved against the config file.",
    )
    parser.add_argument(
        "--include_uavs",
        default=None,
        help="Optional comma-separated UAV IDs to merge, for example uav_01,uav_02,uav_03,uav_05,uav_06,uav_07.",
    )
    parser.add_argument(
        "--exclude_uav_ids",
        default=None,
        help="Optional comma-separated UAV IDs to exclude after applying --include_uavs.",
    )
    parser.add_argument("--output", required=True, help="Path to the merged output CSV.")
    parser.add_argument(
        "--summary_json",
        default=None,
        help="Optional path to the merged summary JSON. Defaults to merged_dataset_summary.json beside the output CSV.",
    )
    parser.add_argument("--notes_json", default=None, help=argparse.SUPPRESS)

    parser.add_argument("--uav_ndd_csv", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--gcs_csv", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--third_csv", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--third_uav_id", default="uav_03", help=argparse.SUPPRESS)
    parser.add_argument("--third_dataset_name", default="isot_drone", help=argparse.SUPPRESS)
    parser.add_argument("--fourth_csv", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--fourth_uav_id", default="uav_04", help=argparse.SUPPRESS)
    parser.add_argument("--fourth_dataset_name", default="uavs_normal_cyberattacks", help=argparse.SUPPRESS)
    parser.add_argument("--fifth_csv", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--fifth_uav_id", default="uav_05", help=argparse.SUPPRESS)
    parser.add_argument("--fifth_dataset_name", default="unsw_nb15", help=argparse.SUPPRESS)
    parser.add_argument("--sixth_csv", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--sixth_uav_id", default="uav_06", help=argparse.SUPPRESS)
    parser.add_argument("--sixth_dataset_name", default="ecu_ioft", help=argparse.SUPPRESS)
    parser.add_argument("--seventh_csv", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--seventh_uav_id", default="uav_07", help=argparse.SUPPRESS)
    parser.add_argument("--seventh_dataset_name", default="uavids", help=argparse.SUPPRESS)
    parser.add_argument("--label_col", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--split_col", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--timestamp_col", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--record_id_col", default=None, help=argparse.SUPPRESS)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    legacy_configs = build_legacy_dataset_configs(args)
    prepare_multi_uav_hetero_dataset(
        output=args.output,
        summary_json=args.summary_json,
        notes_json=args.notes_json,
        config_path=args.config,
        include_uavs=args.include_uavs,
        exclude_uav_ids=args.exclude_uav_ids,
        dataset_configs=legacy_configs or None,
    )


if __name__ == "__main__":
    main()
