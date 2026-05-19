#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


LABEL_MAP = {
    "Normal": "benign",
    "ICMP Flooding": "icmp_flooding",
    "UDP Flooding": "udp_flooding",
    "DoS": "dos",
    "DDoS": "ddos",
    "BruteForce": "bruteforce",
    "MITM": "mitm",
    "De-authentication": "deauthentication",
    "FakeLanding": "fake_landing",
    "replay": "replay",
    "Replay": "replay",
    "Jamming": "jamming",
    "Scanning": "recon_scanning",
    "Reconnassiance": "recon_scanning",
}

DEFAULT_ID_CLASSES = [
    "benign",
    "icmp_flooding",
    "udp_flooding",
    "dos",
    "ddos",
    "bruteforce",
    "mitm",
    "deauthentication",
    "jamming",
    "recon_scanning",
]

DEFAULT_OOD_CLASSES = [
    "replay",
    "fake_landing",
]

FORCE_DROP_COLUMNS = {
    "frame.number",
    "frame.time_epoch",
    "frame.time_relative",
    "radiotap.mactime",
    "radiotap.present.tsft",
    "radiotap.timestamp.ts",
    "radiotap.vendor_oui",
    "wlan.bssid",
    "wlan.seq",
    "wlan.tag",
    "wlan_radio.start_tsf",
    "wlan_radio.timestamp",
    "ip.dst",
    "ip.proto",
    "ip.src",
    "ip.ttl",
    "tcp.ack",
    "udp.dstport",
    "udp.srcport",
    "udp.length",
}

DROP_PATTERNS = [
    "epoch",
    "relative",
    "timestamp",
    "mactime",
    "tsf",
    "vendor",
    "bssid",
    "seq",
    "tag",
    "ip.",
    "udp.",
    "tcp.",
    "port",
]

HEX_RE = re.compile(r"^0x[0-9a-fA-F]+$")
SANITIZE_RE = re.compile(r"[^a-z0-9_]+")


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


def resolve_path(base_candidates: Sequence[Path], value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    for base in base_candidates:
        candidate = (base / path).resolve()
        if candidate.exists():
            return candidate
    return (base_candidates[0] / path).resolve()


def scan_valid_csv_prefix(path: Path) -> dict:
    with path.open("r", encoding="latin1", errors="replace") as f:
        header = f.readline().replace("\x00", "").rstrip("\r\n")
        if not header:
            raise ValueError("Input CSV is empty.")
        expected_fields = header.count(",") + 1
        valid_rows = 0
        first_invalid_line = None
        invalid_field_count = None
        total_lines = 1
        for line_number, line in enumerate(f, start=2):
            total_lines = line_number
            cleaned = line.replace("\x00", "").rstrip("\r\n")
            field_count = cleaned.count(",") + 1 if cleaned else 1
            if field_count != expected_fields:
                first_invalid_line = line_number
                invalid_field_count = field_count
                break
            valid_rows += 1
        return {
            "header": header,
            "expected_fields": expected_fields,
            "valid_rows": valid_rows,
            "first_invalid_line": first_invalid_line,
            "invalid_field_count": invalid_field_count,
            "total_lines_scanned": total_lines,
            "invalid_tail_detected": first_invalid_line is not None,
        }


def coerce_mixed_numeric(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce").astype(np.float64)
    text = series.astype(str).str.strip()
    text = text.replace({"": np.nan, "nan": np.nan, "None": np.nan, "<NA>": np.nan})
    hex_mask = text.str.match(HEX_RE, na=False)
    out = pd.Series(np.nan, index=series.index, dtype=np.float64)
    if hex_mask.any():
        out.loc[hex_mask] = text.loc[hex_mask].map(lambda x: float(int(str(x), 16)))
    non_hex = ~hex_mask
    if non_hex.any():
        out.loc[non_hex] = pd.to_numeric(text.loc[non_hex], errors="coerce")
    return out


def monotonic_non_decreasing_fraction(series: pd.Series) -> float:
    values = series.dropna().to_numpy(dtype=np.float64, copy=False)
    if values.size < 2:
        return 1.0
    return float(np.mean(np.diff(values) >= 0.0))


def sanitize_column_name(name: str) -> str:
    lowered = name.strip().lower().replace(".", "_").replace("/", "_").replace("-", "_").replace(" ", "_")
    lowered = SANITIZE_RE.sub("_", lowered).strip("_")
    return lowered or "feature"


def build_output_name(source_name: str, used: set[str]) -> str:
    base = f"raw_{sanitize_column_name(source_name)}"
    candidate = base
    suffix = 2
    while candidate in used:
        candidate = f"{base}_{suffix}"
        suffix += 1
    used.add(candidate)
    return candidate


def should_force_drop(column_name: str) -> bool:
    lowered = column_name.lower()
    if lowered in {name.lower() for name in FORCE_DROP_COLUMNS}:
        return True
    return any(pattern in lowered for pattern in DROP_PATTERNS)


def select_numeric_features(
    df: pd.DataFrame,
    label_col: str,
    min_numeric_ratio: float = 0.95,
) -> tuple[pd.DataFrame, list[str], dict, dict]:
    selected: dict[str, pd.Series] = {}
    feature_cols: list[str] = []
    selection_report: dict[str, dict] = {}
    output_name_map: dict[str, str] = {}
    used_output_names: set[str] = set()

    for column in df.columns:
        if column == label_col:
            continue
        numeric = coerce_mixed_numeric(df[column])
        non_na = int(numeric.notna().sum())
        numeric_ratio = float(non_na / len(df)) if len(df) else 0.0
        unique_count = int(numeric.nunique(dropna=True)) if non_na else 0
        unique_ratio = float(unique_count / non_na) if non_na else 0.0
        monotonic_fraction = monotonic_non_decreasing_fraction(numeric) if non_na else 0.0
        reason = None

        if should_force_drop(column):
            reason = "forced_leakage_or_identifier_drop"
        elif numeric_ratio < min_numeric_ratio:
            reason = "mostly_non_numeric"
        elif unique_count <= 1:
            reason = "constant_or_empty"
        elif unique_ratio > 0.90 and monotonic_fraction > 0.95:
            reason = "capture_counter_like"

        output_name = None
        if reason is None:
            output_name = build_output_name(column, used_output_names)
            selected[output_name] = numeric.astype(np.float64)
            feature_cols.append(output_name)
            output_name_map[column] = output_name

        selection_report[column] = {
            "selected": reason is None,
            "reason": reason or "kept",
            "output_name": output_name,
            "numeric_ratio": numeric_ratio,
            "unique_count": unique_count,
            "unique_ratio": unique_ratio,
            "monotonic_non_decreasing_fraction": monotonic_fraction,
        }

    if not feature_cols:
        raise ValueError("No numeric metadata features remained after conservative filtering.")

    feature_frame = pd.DataFrame(selected)
    return feature_frame, feature_cols, selection_report, output_name_map


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare an experiment-ready UCS-OODID CSV from UAV-NDD UAV-Case1-Label.csv.")
    parser.add_argument(
        "--input",
        default=r"..\UAV-NDD CSV\UAV-Case1-Label.csv",
        help="Path to UAV-Case1-Label.csv.",
    )
    parser.add_argument(
        "--output",
        default=r"data\uav_ndd_case1_experiment.csv",
        help="Path to the prepared output CSV.",
    )
    parser.add_argument(
        "--notes_json",
        default=r"data\uav_ndd_case1_notes.json",
        help="Path to the preparation summary JSON.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chunk_size", type=int, default=4, help="How many same-label rows to emit before switching labels in the pseudo-stream.")
    parser.add_argument("--train_ratio", type=float, default=0.70)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--min_numeric_ratio", type=float, default=0.95)
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
    base_candidates = [script_dir, project_root, workspace_root]

    input_path = resolve_path(base_candidates, args.input)
    output_path = resolve_path([project_root], args.output)
    notes_path = resolve_path([project_root], args.notes_json)

    id_classes = parse_csv_list(args.id_classes)
    ood_classes = parse_csv_list(args.ood_classes)
    known_classes = set(id_classes) | set(ood_classes)
    ordered_classes = list(dict.fromkeys(id_classes + ood_classes))

    prefix_scan = scan_valid_csv_prefix(input_path)
    valid_rows = int(prefix_scan["valid_rows"])
    if valid_rows < 1:
        raise ValueError("No valid data rows found before the malformed tail.")

    df = pd.read_csv(input_path, encoding="latin1", nrows=valid_rows, low_memory=False)
    if "Label" not in df.columns:
        raise ValueError("Expected a 'Label' column in the input CSV.")

    source_columns = [c for c in df.columns if c != "Label"]
    feature_frame, numeric_feature_cols, selection_report, output_name_map = select_numeric_features(
        df,
        label_col="Label",
        min_numeric_ratio=float(args.min_numeric_ratio),
    )

    work = feature_frame.copy()
    work["source_label"] = df["Label"].astype(str).str.strip()
    work["label"] = work["source_label"].map(normalise_label)
    class_counts_before_cleaning = label_count_dict(work, ordered_classes)

    del feature_frame
    gc.collect()

    duplicate_rows_removed = 0
    if not args.keep_duplicates:
        before = len(work)
        work = work.drop_duplicates(subset=numeric_feature_cols + ["label"]).copy()
        duplicate_rows_removed = before - len(work)

    ambiguous_dropped = 0
    if not args.keep_conflicting_patterns:
        work, ambiguous_dropped = drop_ambiguous_feature_patterns(work, numeric_feature_cols)

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

    prepared = pd.concat([train_df, val_df, test_df, ood_df], ignore_index=True)
    prepared["timestamp"] = np.arange(len(prepared), dtype=np.float64)
    prepared["record_id"] = np.arange(len(prepared), dtype=np.int64)
    ordered_cols = numeric_feature_cols + ["label", "source_label", "recommended_partition", "split", "timestamp", "record_id"]
    prepared = prepared.loc[:, ordered_cols]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    notes_path.parent.mkdir(parents=True, exist_ok=True)
    prepared.to_csv(output_path, index=False)

    final_split_rows = {name: int(len(frame)) for name, frame in split_frames.items()}
    split_label_counts = {name: label_count_dict(frame, ordered_classes) for name, frame in split_frames.items()}
    ood_holdout_class_counts = label_count_dict(ood_df, ood_classes)
    dropped_columns = {
        column: info
        for column, info in selection_report.items()
        if not info["selected"]
    }

    summary = {
        "input_csv": str(input_path),
        "output_csv": str(output_path),
        "notes_json": str(notes_path),
        "scan_summary": prefix_scan,
        "input_rows": int(valid_rows),
        "raw_feature_columns": source_columns,
        "selected_feature_columns": numeric_feature_cols,
        "source_to_output_feature_map": output_name_map,
        "feature_selection_report": selection_report,
        "dropped_feature_columns": dropped_columns,
        "output_rows": int(len(prepared)),
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
            "min_numeric_ratio": float(args.min_numeric_ratio),
        },
        "notes": [
            "The raw CSV was truncated at the first malformed line because the file contains a non-tabular binary tail after the valid records.",
            "Several source columns behaved like capture-order counters or identifiers and were dropped conservatively even when their header names looked numeric.",
            "Scanning and Reconnassiance were merged into recon_scanning.",
            "The file is ordered as ID-train -> ID-val -> ID-test -> OOD-test so scripts/train.py can reuse the explicit split column safely.",
            "A synthetic monotonic timestamp was created because the source capture order is class-blocked and the original timing fields are not reliable enough for direct chronological splitting.",
            "Exact duplicates were removed by default to reduce trivial leakage across splits.",
            "Rows with identical retained feature patterns but conflicting cleaned labels were removed by default.",
        ],
        "recommended_train_command": (
            "python scripts/train.py "
            f"--input {output_path} "
            "--output_dir runs/uav_ndd_case1_default "
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
        "input_rows": int(valid_rows),
        "selected_feature_count": int(len(numeric_feature_cols)),
        "duplicate_rows_removed": int(duplicate_rows_removed),
        "ambiguous_rows_removed": int(ambiguous_dropped),
        "rows_after_cleaning": int(len(work)),
        "final_split_rows": final_split_rows,
        "invalid_tail_detected": bool(prefix_scan["invalid_tail_detected"]),
        "first_invalid_line": prefix_scan["first_invalid_line"],
        "notes_json": str(notes_path),
    }
    print(json.dumps({"prepare_uav_ndd_case1_summary": console_summary}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
