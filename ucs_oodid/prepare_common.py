from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd

COMMON_LABEL_CANDIDATES = [
    "label",
    "class",
    "attack_type",
    "attack type",
    "attack",
    "category",
]

COMMON_TIMESTAMP_CANDIDATES = [
    "timestamp",
    "time",
    "ts",
    "datetime",
    "date_time",
    "flow_time",
]

COMMON_SPLIT_CANDIDATES = [
    "split",
    "partition",
    "subset",
    "set",
    "fold",
]

SPLIT_ALIASES = {
    "train": "train",
    "training": "train",
    "val": "val",
    "valid": "val",
    "validation": "val",
    "dev": "val",
    "test": "test",
    "testing": "test",
    "eval": "test",
    "evaluation": "test",
    "holdout": "test",
    "all": "all",
    "full": "all",
}


def _is_missing_scalar(value: object) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except TypeError:
        return False


def normalize_text_token(value: object) -> str:
    if _is_missing_scalar(value):
        return "unknown"
    text = str(value).strip().lower()
    if not text or text in {"nan", "none", "null"}:
        return "unknown"
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "unknown"


def normalize_raw_label(value: object) -> str:
    if _is_missing_scalar(value):
        return "unknown"
    text = str(value).strip()
    return text or "unknown"


def column_key(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def resolve_base_candidates(script_file: str | Path) -> tuple[Path, Path, list[Path]]:
    script_dir = Path(script_file).resolve().parent
    project_root = script_dir.parent
    workspace_root = project_root.parent
    return script_dir, project_root, [script_dir, project_root, workspace_root]


def resolve_input_path(base_candidates: Sequence[Path], value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    for base in base_candidates:
        candidate = (base / path).resolve()
        if candidate.exists():
            return candidate
    return (base_candidates[0] / path).resolve()


def resolve_output_path(project_root: Path, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (project_root / path).resolve()


def find_csv_files(input_path: Path, preferred_names: Sequence[str] | None = None) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() != ".csv":
            raise ValueError(f"Expected a CSV file, got: {input_path}")
        return [input_path]
    if not input_path.is_dir():
        raise ValueError(f"Input path does not exist: {input_path}")

    csv_files = sorted(path for path in input_path.rglob("*.csv") if path.is_file())
    if not csv_files:
        raise ValueError(f"No CSV files found under: {input_path}")

    if preferred_names:
        lower_map = {path.name.lower(): path for path in csv_files}
        matched: list[Path] = []
        seen: set[Path] = set()
        for preferred in preferred_names:
            candidate = lower_map.get(preferred.lower())
            if candidate is not None and candidate not in seen:
                matched.append(candidate)
                seen.add(candidate)
        if matched:
            return matched
    return csv_files


def load_single_csv_from_input(
    base_candidates: Sequence[Path],
    input_value: str,
    preferred_names: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, Path]:
    input_path = resolve_input_path(base_candidates, input_value)
    csv_files = find_csv_files(input_path, preferred_names=preferred_names)
    if len(csv_files) != 1:
        display = ", ".join(str(path) for path in csv_files[:5])
        raise ValueError(
            f"Expected exactly one CSV under {input_path}, found {len(csv_files)}: {display}"
        )
    csv_path = csv_files[0]
    return pd.read_csv(csv_path, low_memory=False), csv_path


def resolve_column_name(
    df: pd.DataFrame,
    requested: str | None,
    candidates: Sequence[str],
    purpose: str,
    required: bool = True,
) -> str | None:
    lookup: dict[str, str] = {}
    for column in df.columns:
        lookup.setdefault(column_key(column), str(column))

    if requested:
        requested_key = column_key(requested)
        if requested_key in lookup:
            return lookup[requested_key]
        available = ", ".join(map(str, df.columns))
        raise ValueError(f"Could not find {purpose} column '{requested}'. Available columns: {available}")

    for candidate in candidates:
        candidate_key = column_key(candidate)
        if candidate_key in lookup:
            return lookup[candidate_key]

    if required:
        available = ", ".join(map(str, df.columns))
        raise ValueError(f"Could not auto-detect a {purpose} column. Available columns: {available}")
    return None


def build_timestamp_series(df: pd.DataFrame, timestamp_column: str | None) -> pd.Series:
    default_values = pd.Series(np.arange(len(df), dtype=np.int64), index=df.index)
    if timestamp_column is None:
        return default_values

    series = df[timestamp_column]
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().sum() > 0 and numeric.notna().mean() >= 0.8:
        return numeric.where(numeric.notna(), default_values)

    parsed = pd.to_datetime(series, errors="coerce")
    if parsed.notna().sum() > 0 and parsed.notna().mean() >= 0.8:
        formatted = parsed.dt.strftime("%Y-%m-%dT%H:%M:%S")
        return formatted.where(parsed.notna(), default_values.astype(str))

    stripped = series.astype("string").fillna("").str.strip()
    return stripped.where(stripped != "", default_values.astype(str))


def normalize_split_value(value: object) -> str:
    token = normalize_text_token(value)
    if token == "unknown":
        return "all"
    return SPLIT_ALIASES.get(token, token)


def build_split_series(df: pd.DataFrame, split_column: str | None) -> pd.Series:
    if split_column is None:
        return pd.Series(["all"] * len(df), index=df.index)
    return df[split_column].map(normalize_split_value)


def build_prepared_frame(
    raw_df: pd.DataFrame,
    *,
    label_column: str,
    timestamp_column: str | None,
    split_column: str | None,
    uav_id: str,
    dataset_name: str,
    source_type: str,
    label_normalizer: Callable[[object], str],
) -> pd.DataFrame:
    labels = raw_df[label_column].map(normalize_raw_label)
    normalized_labels = labels.map(label_normalizer)

    metadata = pd.DataFrame(
        {
            "record_id": np.arange(len(raw_df), dtype=np.int64),
            "timestamp": build_timestamp_series(raw_df, timestamp_column),
            "uav_id": [str(uav_id)] * len(raw_df),
            "dataset_name": [str(dataset_name)] * len(raw_df),
            "source_type": [str(source_type)] * len(raw_df),
            "label": labels,
            "label_normalized": normalized_labels,
            "split": build_split_series(raw_df, split_column),
        },
        index=raw_df.index,
    )
    remaining_columns = [column for column in raw_df.columns if column not in metadata.columns]
    return pd.concat(
        [metadata.reset_index(drop=True), raw_df.loc[:, remaining_columns].reset_index(drop=True)],
        axis=1,
    )


def missing_value_counts(df: pd.DataFrame) -> dict[str, int]:
    counts: dict[str, int] = {}
    for column in df.columns:
        series = df[column]
        missing = int(series.isna().sum())
        if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
            blanks = int(series.astype("string").fillna("").str.strip().eq("").sum())
            missing = max(missing, blanks)
        if missing > 0:
            counts[str(column)] = missing
    return counts


def build_processing_summary(
    prepared_df: pd.DataFrame,
    *,
    output_path: Path,
    label_column: str,
    timestamp_column: str | None,
    split_column: str | None,
    dataset_name: str,
    source_type: str,
    input_files: Sequence[Path],
    extra_notes: Sequence[str] | None = None,
) -> dict:
    label_counts = prepared_df["label_normalized"].astype(str).value_counts().to_dict()
    benign_count = int(label_counts.get("benign", 0))
    attack_counts = {
        str(label): int(count)
        for label, count in sorted(label_counts.items())
        if str(label) != "benign"
    }
    summary = {
        "dataset_name": dataset_name,
        "source_type": source_type,
        "input_files": [str(path) for path in input_files],
        "label_column_used": label_column,
        "timestamp_column_used": timestamp_column,
        "split_column_used": split_column,
        "total_samples": int(len(prepared_df)),
        "benign_samples": benign_count,
        "attack_samples_by_category": attack_counts,
        "missing_values": missing_value_counts(prepared_df),
        "output_path": str(output_path),
    }
    if extra_notes:
        summary["notes"] = list(extra_notes)
    return summary


def write_prepared_csv(prepared_df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prepared_df.to_csv(output_path, index=False)


def print_processing_summary(tag: str, summary: dict) -> None:
    print(json.dumps({tag: summary}, indent=2, ensure_ascii=False))
