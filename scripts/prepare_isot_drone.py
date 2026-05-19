#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, NamedTuple, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

FOLDER_LABEL_MAP = {
    "Regular": "benign",
    "DoS": "dos",
    "MITM": "mitm",
    "Password Cracking": "bruteforce",
    "Injection": "injection",
    "Ip Spoofing": "ip_spoofing",
    "Manipulation": "payload_manipulation",
    "Replay": "replay",
    "Video": "video_interception",
    "Unauth": "unauthorized_udp",
}

DEFAULT_ID_CLASSES = [
    "benign",
    "dos",
    "mitm",
    "bruteforce",
    "injection",
    "ip_spoofing",
    "payload_manipulation",
]

DEFAULT_OOD_CLASSES = [
    "replay",
    "video_interception",
    "unauthorized_udp",
]

DEFAULT_DROP_FEATURES = [
    "ts",
    "payload_length",
    "var_payload",
    "drone_port",
    "ds_status",
    "sequence_number",
    "max_duration",
    "min_duration",
    "sum_duration",
    "average_duration",
]


class IsotCsvFile(NamedTuple):
    path: Path
    source_folder: str
    label: str
    rows: int


def parse_csv_list(value: str | Sequence[str]) -> List[str]:
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return [str(item).strip() for item in value if str(item).strip()]


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


def sanitize_column_name(name: str) -> str:
    text = str(name).strip().lower()
    text = re.sub(r"[^0-9a-zA-Z]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "unnamed"


def sanitize_columns(columns: Sequence[str]) -> list[str]:
    sanitized: list[str] = []
    seen: dict[str, int] = {}
    for raw in columns:
        base = sanitize_column_name(raw)
        count = seen.get(base, 0)
        seen[base] = count + 1
        sanitized.append(base if count == 0 else f"{base}_{count + 1}")
    return sanitized


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
    label_counts = df[label_col].astype(str).value_counts()
    if int(label_counts.min()) < 3:
        raise ValueError("At least three samples per ID class are required for stratified train/val/test splitting.")
    rng = np.random.default_rng(seed)
    train_parts: list[pd.DataFrame] = []
    val_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []

    for _, group in df.groupby(label_col, sort=True):
        shuffled = group.sample(frac=1.0, random_state=int(rng.integers(0, np.iinfo(np.int32).max))).reset_index(drop=False)
        n = len(shuffled)
        n_train = max(1, int(round(n * train_ratio)))
        n_val = max(1, int(round(n * val_ratio)))
        n_test = n - n_train - n_val
        while n_test < 1:
            if n_train >= n_val and n_train > 1:
                n_train -= 1
            elif n_val > 1:
                n_val -= 1
            else:
                break
            n_test = n - n_train - n_val
        if min(n_train, n_val, n_test) < 1:
            raise ValueError("ID data is too small for the requested split ratios.")
        train_parts.append(group.loc[shuffled.iloc[:n_train]["index"]].copy())
        val_parts.append(group.loc[shuffled.iloc[n_train:n_train + n_val]["index"]].copy())
        test_parts.append(group.loc[shuffled.iloc[n_train + n_val:]["index"]].copy())

    train_df = pd.concat(train_parts, ignore_index=False)
    val_df = pd.concat(val_parts, ignore_index=False)
    test_df = pd.concat(test_parts, ignore_index=False)
    return train_df.copy(), val_df.copy(), test_df.copy()


def summarise_split(df: pd.DataFrame, classes: Sequence[str]) -> dict:
    return {
        "rows": int(len(df)),
        "label_counts": label_count_dict(df, classes),
        "mixed_window_stats": window_mix_stats(df["label"].tolist()),
    }


def allocate_counts(raw_counts: Dict[str, int], target_rows: int, minimum_per_label: int = 0) -> dict[str, int]:
    positive = {str(k): int(v) for k, v in raw_counts.items() if int(v) > 0}
    if not positive:
        return {}
    total_rows = sum(positive.values())
    capped_target = min(int(target_rows), total_rows)
    if capped_target <= 0:
        return {label: 0 for label in sorted(positive)}
    if capped_target >= total_rows:
        return {label: int(positive[label]) for label in sorted(positive)}

    base = {label: 0 for label in positive}
    if minimum_per_label > 0:
        desired_base = {label: min(int(count), int(minimum_per_label)) for label, count in positive.items()}
        if sum(desired_base.values()) <= capped_target:
            base = desired_base

    remaining_target = capped_target - sum(base.values())
    headroom = {label: positive[label] - base[label] for label in positive}
    headroom_total = sum(headroom.values())
    if remaining_target <= 0 or headroom_total <= 0:
        return {label: int(base[label]) for label in sorted(base)}

    floors: dict[str, int] = {}
    fractions: list[tuple[float, str]] = []
    for label, count in headroom.items():
        exact = remaining_target * count / headroom_total
        floor_val = int(math.floor(exact))
        floors[label] = floor_val
        fractions.append((exact - floor_val, label))
    allocated = sum(floors.values())
    remainder = remaining_target - allocated
    for _, label in sorted(fractions, key=lambda item: (-item[0], item[1]))[:remainder]:
        floors[label] += 1
    return {label: int(base[label] + floors[label]) for label in sorted(positive)}


def rebalance_shortfall(available_counts: Dict[str, int], desired_counts: Dict[str, int]) -> dict[str, int]:
    labels = sorted(set(available_counts) | set(desired_counts))
    selected = {
        label: min(int(available_counts.get(label, 0)), int(desired_counts.get(label, 0)))
        for label in labels
    }
    target_rows = sum(int(v) for v in desired_counts.values())
    remaining = target_rows - sum(selected.values())
    if remaining <= 0:
        return selected
    surplus = {
        label: max(int(available_counts.get(label, 0)) - int(selected.get(label, 0)), 0)
        for label in labels
    }
    if sum(surplus.values()) <= 0:
        return selected
    extras = allocate_counts(surplus, remaining, minimum_per_label=0)
    for label, extra in extras.items():
        selected[label] = int(selected.get(label, 0) + extra)
    return selected


def scan_isot_files(input_root: Path) -> tuple[list[IsotCsvFile], list[str]]:
    if not input_root.exists():
        raise FileNotFoundError(f"ISOT input root does not exist: {input_root}")
    files: list[IsotCsvFile] = []
    header_reference: list[str] | None = None
    for folder in sorted([path for path in input_root.iterdir() if path.is_dir()], key=lambda p: p.name.lower()):
        if folder.name not in FOLDER_LABEL_MAP:
            raise ValueError(f"Unsupported ISOT folder '{folder.name}'. Add it to FOLDER_LABEL_MAP before continuing.")
        label = FOLDER_LABEL_MAP[folder.name]
        for csv_path in sorted(folder.glob("*.csv"), key=lambda p: p.name.lower()):
            with csv_path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.reader(handle)
                header = next(reader, None)
                if not header:
                    continue
                row_count = sum(1 for _ in reader)
            if header_reference is None:
                header_reference = list(header)
            elif list(header) != header_reference:
                raise ValueError(f"Schema mismatch detected in {csv_path}.")
            files.append(IsotCsvFile(path=csv_path, source_folder=folder.name, label=label, rows=row_count))
    if not files:
        raise ValueError(f"No CSV files were found under {input_root}.")
    if header_reference is None:
        raise ValueError("Unable to determine the ISOT CSV schema.")
    return files, header_reference


def file_row_counts_by_label(files: Sequence[IsotCsvFile]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in files:
        counts[item.label] = counts.get(item.label, 0) + int(item.rows)
    return {label: counts[label] for label in sorted(counts)}


def folder_row_counts(files: Sequence[IsotCsvFile]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in files:
        counts[item.source_folder] = counts.get(item.source_folder, 0) + int(item.rows)
    return {name: counts[name] for name in sorted(counts)}


def file_counts_by_folder(files: Sequence[IsotCsvFile]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in files:
        counts[item.source_folder] = counts.get(item.source_folder, 0) + 1
    return {name: counts[name] for name in sorted(counts)}


def allocate_file_quotas(files: Sequence[IsotCsvFile], label_quotas: Dict[str, int]) -> dict[Path, int]:
    quotas: dict[Path, int] = {}
    files_by_label: dict[str, list[IsotCsvFile]] = {}
    for item in files:
        files_by_label.setdefault(item.label, []).append(item)
    for label, group in files_by_label.items():
        target = int(label_quotas.get(label, 0))
        if target <= 0:
            continue
        row_counts = {str(item.path): int(item.rows) for item in group}
        allocated = allocate_counts(row_counts, target, minimum_per_label=0)
        for item in group:
            quotas[item.path] = int(allocated.get(str(item.path), 0))
    return quotas


def load_sampled_rows(
    files: Sequence[IsotCsvFile],
    file_sample_quotas: Dict[Path, int],
    sanitized_columns: Sequence[str],
    dropped_features: Sequence[str],
    seed: int,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    rng = np.random.default_rng(seed)
    dropped = {str(col) for col in dropped_features}
    for item in files:
        sample_rows = int(file_sample_quotas.get(item.path, 0))
        if sample_rows <= 0:
            continue
        frame = pd.read_csv(item.path, low_memory=False)
        frame.columns = list(sanitized_columns)
        keep_columns = [column for column in frame.columns if column not in dropped]
        frame = frame.loc[:, keep_columns]
        if sample_rows < len(frame):
            random_state = int(rng.integers(0, np.iinfo(np.int32).max))
            frame = frame.sample(n=sample_rows, random_state=random_state, replace=False)
        frame = frame.reset_index(drop=True)
        frame["label"] = item.label
        frame["source_label"] = item.source_folder
        frames.append(frame)
    if not frames:
        raise ValueError("Sampling produced no rows. Check target_rows or input_root.")
    return pd.concat(frames, ignore_index=True)


def coerce_numeric_features(df: pd.DataFrame, protected_columns: Sequence[str]) -> tuple[pd.DataFrame, list[str]]:
    out = df.copy()
    protected = {str(column) for column in protected_columns}
    feature_cols: list[str] = []
    for column in out.columns:
        if column in protected:
            continue
        out[column] = pd.to_numeric(out[column], errors="coerce")
        feature_cols.append(str(column))
    if not feature_cols:
        raise ValueError("No retained numeric feature columns remained after ISOT column cleanup.")
    return out, feature_cols


def downsample_to_final_quotas(df: pd.DataFrame, final_label_quotas: Dict[str, int], seed: int) -> tuple[pd.DataFrame, dict, dict]:
    available_counts = label_count_dict(df, label_col="label")
    selected_counts = rebalance_shortfall(available_counts, final_label_quotas)
    rng = np.random.default_rng(seed)
    frames: list[pd.DataFrame] = []
    for label in sorted(selected_counts):
        group = df[df["label"] == label].copy()
        target = int(selected_counts[label])
        if target <= 0 or group.empty:
            continue
        if target < len(group):
            random_state = int(rng.integers(0, np.iinfo(np.int32).max))
            group = group.sample(n=target, random_state=random_state, replace=False)
        frames.append(group.reset_index(drop=True))
    final_df = pd.concat(frames, ignore_index=True) if frames else df.iloc[0:0].copy()
    return final_df, available_counts, selected_counts


def prepare_isot_drone_dataset(
    input_root: str,
    output: str,
    notes_json: str | None = None,
    seed: int = 42,
    target_rows: int = 35599,
    chunk_size: int = 4,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    id_classes: Sequence[str] | None = None,
    ood_classes: Sequence[str] | None = None,
    oversample_factor: float = 2.0,
    min_rows_per_label: int = 0,
    drop_features: Sequence[str] | None = None,
    keep_duplicates: bool = False,
    keep_conflicting_patterns: bool = False,
) -> tuple[pd.DataFrame, dict]:
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    workspace_root = project_root.parent
    base_candidates = [script_dir, project_root, workspace_root]

    input_path = resolve_existing_path(base_candidates, input_root)
    output_path = resolve_output_path(project_root, output)
    notes_path = resolve_output_path(project_root, notes_json)
    if output_path is None:
        raise ValueError("output must be provided.")

    id_classes = parse_csv_list(id_classes or DEFAULT_ID_CLASSES)
    ood_classes = parse_csv_list(ood_classes or DEFAULT_OOD_CLASSES)
    known_classes = set(id_classes) | set(ood_classes)
    ordered_classes = list(dict.fromkeys(id_classes + ood_classes))

    files, raw_header = scan_isot_files(input_path)
    sanitized_header = sanitize_columns(raw_header)
    dropped_features = [sanitize_column_name(name) for name in (drop_features or DEFAULT_DROP_FEATURES)]
    raw_label_counts = file_row_counts_by_label(files)
    raw_folder_counts = folder_row_counts(files)
    raw_file_counts = file_counts_by_folder(files)
    total_input_rows = sum(int(item.rows) for item in files)

    unknown_labels = sorted(set(raw_label_counts) - known_classes)
    if unknown_labels:
        raise ValueError(f"Found labels with no ID/OOD assignment: {unknown_labels}")

    final_label_quotas = allocate_counts(raw_label_counts, target_rows, minimum_per_label=min_rows_per_label)
    sampled_label_quotas = {
        label: min(
            int(raw_label_counts[label]),
            max(int(final_label_quotas.get(label, 0)), int(math.ceil(float(final_label_quotas.get(label, 0)) * oversample_factor))),
        )
        for label in raw_label_counts
    }
    sampled_file_quotas = allocate_file_quotas(files, sampled_label_quotas)
    sampled = load_sampled_rows(files, sampled_file_quotas, sanitized_header, dropped_features, seed=seed)
    sampled, feature_cols = coerce_numeric_features(sampled, protected_columns=["label", "source_label"])

    ambiguous_dropped = 0
    if not keep_conflicting_patterns:
        sampled, ambiguous_dropped = drop_ambiguous_feature_patterns(sampled, feature_cols)

    duplicate_rows_removed = 0
    if not keep_duplicates:
        before = len(sampled)
        sampled = sampled.drop_duplicates(subset=feature_cols + ["label"]).copy()
        duplicate_rows_removed = int(before - len(sampled))

    final_sampled, cleaned_available_counts, final_selected_counts = downsample_to_final_quotas(
        sampled,
        final_label_quotas=final_label_quotas,
        seed=seed + 7,
    )

    work = final_sampled.copy()
    work["recommended_partition"] = np.where(work["label"].isin(id_classes), "id", "ood")
    id_df = work[work["recommended_partition"] == "id"].copy()
    ood_df = work[work["recommended_partition"] == "ood"].copy()
    if id_df.empty or ood_df.empty:
        raise ValueError("Both ID and OOD partitions must be non-empty.")

    train_df, val_df, test_df = exact_id_split(id_df, "label", train_ratio, val_ratio, seed)
    train_df = interleave_by_label(train_df, "label", chunk_size, seed + 10)
    val_df = interleave_by_label(val_df, "label", chunk_size, seed + 20)
    test_df = interleave_by_label(test_df, "label", chunk_size, seed + 30)
    ood_df = interleave_by_label(ood_df, "label", chunk_size, seed + 40)

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

    prepared = pd.concat([train_df, val_df, test_df, ood_df], ignore_index=True)
    prepared["timestamp"] = np.arange(len(prepared), dtype=np.float64)
    prepared["record_id"] = np.arange(len(prepared), dtype=np.int64)
    output_columns = feature_cols + ["label", "source_label", "recommended_partition", "split", "timestamp", "record_id"]
    prepared = prepared.loc[:, output_columns]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    prepared.to_csv(output_path, index=False)

    summary = {
        "input_root": str(input_path),
        "output_csv": str(output_path),
        "notes_json": str(notes_path) if notes_path is not None else None,
        "input_rows": int(total_input_rows),
        "output_rows": int(len(prepared)),
        "input_file_count": int(len(files)),
        "raw_folder_counts": raw_folder_counts,
        "raw_file_counts_by_folder": raw_file_counts,
        "raw_label_counts": raw_label_counts,
        "target_rows_requested": int(target_rows),
        "target_rows_by_label": final_label_quotas,
        "sampled_rows_by_label_before_cleaning": sampled_label_quotas,
        "cleaned_available_rows_by_label": cleaned_available_counts,
        "final_selected_rows_by_label": final_selected_counts,
        "feature_columns": feature_cols,
        "dropped_feature_columns": dropped_features,
        "label_map": FOLDER_LABEL_MAP,
        "duplicate_rows_removed": int(duplicate_rows_removed),
        "ambiguous_rows_removed": int(ambiguous_dropped),
        "id_classes": id_classes,
        "ood_classes": ood_classes,
        "ood_holdout": {
            "classes": ood_classes,
            "class_counts": label_count_dict(prepared[prepared["split"] == "test_ood"], ood_classes),
            "rows": int(len(prepared[prepared["split"] == "test_ood"])),
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
            "chunk_size": int(chunk_size),
            "random_seed": int(seed),
            "split_random_seeds": {"train_val_test_split": int(seed), "val_test_split": int(seed + 1)},
            "interleave_random_seeds": {
                "train": int(seed + 10),
                "val": int(seed + 20),
                "test_id": int(seed + 30),
                "test_ood": int(seed + 40),
            },
            "oversample_factor": float(oversample_factor),
            "min_rows_per_label": int(min_rows_per_label),
            "keep_duplicates": bool(keep_duplicates),
            "keep_conflicting_patterns": bool(keep_conflicting_patterns),
            "train_ratio": float(train_ratio),
            "val_ratio": float(val_ratio),
            "reference_dataset_size": 35599,
        },
        "notes": [
            "Folder names, not 'DS status', are used as the source label because DS status is mostly constant outside Password Cracking and behaves like a scenario-specific side channel.",
            "Regular is treated as benign traffic; Video maps to video_interception; Unauth maps to unauthorized_udp; Manipulation maps to payload_manipulation.",
            "The default target_rows=35599 matches the prepared GCS-to-UAV Updated dataset so ISOT can be added as a third UAV without making the group imbalance worse.",
            "Payload, explicit timestamp-like, port-like, status-like, and sequence-like columns were dropped conservatively before writing the prepared CSV.",
            "The file is ordered as ID-train -> ID-val -> ID-test -> OOD-test so scripts/train.py can reuse its explicit split column safely.",
            "A synthetic monotonic timestamp was created so the prepared CSV matches the existing UCS-OODID experiment format.",
        ],
        "recommended_train_command": (
            "python scripts/train.py "
            f"--input {output_path} "
            "--output_dir runs/isot_drone_uav03_default "
            "--label_col label "
            "--timestamp_col timestamp "
            "--record_id_col record_id "
            f"--id_classes {','.join(id_classes)} "
            f"--ood_classes {','.join(ood_classes)} "
            "--window_mode count --window_size 16 --stride 8 --epochs 10"
        ),
    }

    if notes_path is not None:
        notes_path.parent.mkdir(parents=True, exist_ok=True)
        notes_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    console_summary = {
        "input_rows": int(total_input_rows),
        "output_rows": int(len(prepared)),
        "raw_label_counts": raw_label_counts,
        "final_selected_rows_by_label": final_selected_counts,
        "duplicate_rows_removed": int(duplicate_rows_removed),
        "ambiguous_rows_removed": int(ambiguous_dropped),
        "notes_json": str(notes_path) if notes_path is not None else None,
    }
    print(json.dumps({"prepare_isot_drone_summary": console_summary}, indent=2, ensure_ascii=False))
    return prepared, summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare a downsampled UCS-OODID experiment CSV from the ISOT Drone Dataset so it can be used as a third UAV source."
    )
    parser.add_argument(
        "--input_root",
        default=r"..\ISOT Drone Dataset\Dataset\new_feature_csv",
        help="Path to ISOT Drone Dataset/Dataset/new_feature_csv.",
    )
    parser.add_argument(
        "--output",
        default=r"data\isot_drone_uav03_experiment.csv",
        help="Path to the prepared output CSV.",
    )
    parser.add_argument(
        "--notes_json",
        default=r"data\isot_drone_uav03_notes.json",
        help="Path to the preparation summary JSON.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target_rows", type=int, default=35599, help="Target prepared row count. Default matches gcs_to_uav_updated_experiment.csv.")
    parser.add_argument("--chunk_size", type=int, default=4, help="How many same-label records to emit before switching labels in the pseudo-stream.")
    parser.add_argument("--train_ratio", type=float, default=0.70)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--id_classes", default=",".join(DEFAULT_ID_CLASSES))
    parser.add_argument("--ood_classes", default=",".join(DEFAULT_OOD_CLASSES))
    parser.add_argument("--oversample_factor", type=float, default=2.0, help="Temporary oversampling factor applied before duplicate cleanup.")
    parser.add_argument("--min_rows_per_label", type=int, default=0, help="Optional minimum rows to reserve per label before proportional downsampling.")
    parser.add_argument("--drop_features", default=",".join(DEFAULT_DROP_FEATURES), help="Comma-separated sanitized feature names to drop before writing the prepared CSV.")
    parser.add_argument("--keep_duplicates", action="store_true", help="Do not remove exact duplicate rows after sampling.")
    parser.add_argument(
        "--keep_conflicting_patterns",
        action="store_true",
        help="Do not drop feature patterns that appear under multiple cleaned labels in the sampled subset.",
    )
    args = parser.parse_args()

    prepare_isot_drone_dataset(
        input_root=args.input_root,
        output=args.output,
        notes_json=args.notes_json,
        seed=args.seed,
        target_rows=args.target_rows,
        chunk_size=args.chunk_size,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        id_classes=parse_csv_list(args.id_classes),
        ood_classes=parse_csv_list(args.ood_classes),
        oversample_factor=args.oversample_factor,
        min_rows_per_label=args.min_rows_per_label,
        drop_features=parse_csv_list(args.drop_features),
        keep_duplicates=args.keep_duplicates,
        keep_conflicting_patterns=args.keep_conflicting_patterns,
    )


if __name__ == "__main__":
    main()
