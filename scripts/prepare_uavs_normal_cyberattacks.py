#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd

LABEL_MAP = {
    "benign": "benign",
    "DoS attack": "dos",
    "Replay": "replay",
}

DEFAULT_ID_CLASSES = ["benign", "dos"]
DEFAULT_OOD_CLASSES = ["replay"]

FORCE_DROP_COLUMNS = {
    "timestamp_c",
    "frame.number",
    "wlan.ra",
    "wlan.ta",
    "wlan.da",
    "wlan.sa",
    "wlan.bssid",
    "wlan.seq",
    "wlan.frag",
    "ip.id",
    "ip.src",
    "ip.dst",
    "tcp.srcport",
    "tcp.dstport",
    "tcp.seq_raw",
    "tcp.ack_raw",
    "udp.srcport",
    "udp.dstport",
    "data.data",
    "class",
}

SOURCE_TO_OUTPUT_FEATURE_MAP = {
    "frame.len": "raw_frame_len",
    "frame.protocols": "raw_frame_protocols",
    "wlan.duration": "raw_wlan_duration",
    "llc.type": "raw_llc_type",
    "ip.hdr_len": "raw_ip_hdr_len",
    "ip.len": "raw_ip_len",
    "ip.flags": "raw_ip_flags",
    "ip.ttl": "raw_ip_ttl",
    "ip.proto": "raw_ip_proto",
    "tcp.hdr_len": "raw_tcp_hdr_len",
    "tcp.flags": "raw_tcp_flags",
    "tcp.window_size": "raw_tcp_window_size",
    "tcp.options": "raw_tcp_options",
    "udp.length": "raw_udp_length",
    "data.len": "raw_data_len",
    "wlan.fc.type": "raw_wlan_fc_type",
    "wlan.fc.subtype": "raw_wlan_fc_subtype",
    "time_since_last_packet": "raw_time_since_last_packet",
}


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


def resolve_path(base_candidates: Sequence[Path], value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    for base in base_candidates:
        candidate = (base / path).resolve()
        if candidate.exists():
            return candidate
    return (base_candidates[0] / path).resolve()


def coerce_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def load_clean_subset(input_path: Path) -> tuple[pd.DataFrame, dict]:
    raw = pd.read_csv(input_path, low_memory=False)
    total_rows = int(len(raw))
    raw["source_class"] = raw["class"].fillna("").astype(str).str.strip()
    direct_valid_mask = raw["source_class"].isin(LABEL_MAP)
    repeated_header_rows = int((raw["source_class"] == "class").sum())
    ignored_blank_or_mixed_rows = int((raw["source_class"] == "").sum())
    clean = raw.loc[direct_valid_mask].copy()
    clean["label"] = clean["source_class"].map(LABEL_MAP)
    meta = {
        "input_rows": total_rows,
        "direct_valid_rows": int(len(clean)),
        "ignored_blank_or_mixed_rows": ignored_blank_or_mixed_rows,
        "repeated_header_rows": repeated_header_rows,
        "raw_direct_label_counts": {str(k): int(v) for k, v in clean["source_class"].value_counts().to_dict().items()},
    }
    return clean, meta


def select_and_rename_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], dict[str, str], list[str]]:
    selected_map: dict[str, str] = {}
    dropped: list[str] = []
    work = df.copy()
    for source, output in SOURCE_TO_OUTPUT_FEATURE_MAP.items():
        if source in FORCE_DROP_COLUMNS:
            dropped.append(source)
            continue
        if source in work.columns:
            selected_map[source] = output
        else:
            dropped.append(source)
    for column in work.columns:
        if column in {"label", "source_class"}:
            continue
        if column in FORCE_DROP_COLUMNS:
            dropped.append(column)
    if not selected_map:
        raise ValueError("No usable numeric feature columns were found in the clean subset.")
    selected_columns = list(selected_map.keys())
    work = work.loc[:, selected_columns + ["label", "source_class"]].rename(columns=selected_map)
    feature_cols = list(selected_map.values())
    for column in feature_cols:
        work[column] = coerce_numeric(work[column])
    return work, feature_cols, selected_map, sorted(set(dropped))


def prepare_uavs_normal_cyberattacks_dataset(
    input_csv: str,
    output: str,
    notes_json: str | None = None,
    seed: int = 42,
    chunk_size: int = 4,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    id_classes: Sequence[str] | None = None,
    ood_classes: Sequence[str] | None = None,
    keep_duplicates: bool = False,
    keep_conflicting_patterns: bool = False,
) -> tuple[pd.DataFrame, dict]:
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    workspace_root = project_root.parent
    base_candidates = [script_dir, project_root, workspace_root]

    input_path = resolve_path(base_candidates, input_csv)
    output_path = resolve_path([project_root], output)
    notes_path = resolve_path([project_root], notes_json) if notes_json is not None else None

    id_classes = parse_csv_list(id_classes or DEFAULT_ID_CLASSES)
    ood_classes = parse_csv_list(ood_classes or DEFAULT_OOD_CLASSES)
    ordered_classes = list(dict.fromkeys(id_classes + ood_classes))
    known_classes = set(ordered_classes)

    clean, source_meta = load_clean_subset(input_path)
    work, feature_cols, selected_map, dropped_columns = select_and_rename_features(clean)

    unknown_labels = sorted(set(work["label"]) - known_classes)
    if unknown_labels:
        raise ValueError(f"Found labels with no ID/OOD assignment: {unknown_labels}")

    ambiguous_dropped = 0
    if not keep_conflicting_patterns:
        work, ambiguous_dropped = drop_ambiguous_feature_patterns(work, feature_cols)

    duplicate_rows_removed = 0
    if not keep_duplicates:
        before = len(work)
        work = work.drop_duplicates(subset=feature_cols + ["label"]).copy()
        duplicate_rows_removed = int(before - len(work))

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

    output_columns = feature_cols + ["label", "source_class", "recommended_partition", "split", "timestamp", "record_id"]
    prepared = prepared.loc[:, output_columns]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    prepared.to_csv(output_path, index=False)

    summary = {
        "input_csv": str(input_path),
        "output_csv": str(output_path),
        "notes_json": str(notes_path) if notes_path is not None else None,
        "input_rows": int(source_meta["input_rows"]),
        "direct_valid_rows": int(source_meta["direct_valid_rows"]),
        "ignored_blank_or_mixed_rows": int(source_meta["ignored_blank_or_mixed_rows"]),
        "repeated_header_rows": int(source_meta["repeated_header_rows"]),
        "raw_direct_label_counts": source_meta["raw_direct_label_counts"],
        "output_rows": int(len(prepared)),
        "selected_feature_columns": feature_cols,
        "source_to_output_feature_map": selected_map,
        "dropped_columns": dropped_columns,
        "label_map": LABEL_MAP,
        "duplicate_rows_removed": int(duplicate_rows_removed),
        "ambiguous_rows_removed": int(ambiguous_dropped),
        "rows_after_cleaning": int(len(work)),
        "class_counts_after_cleaning": label_count_dict(work, ordered_classes),
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
            "keep_duplicates": bool(keep_duplicates),
            "keep_conflicting_patterns": bool(keep_conflicting_patterns),
            "train_ratio": float(train_ratio),
            "val_ratio": float(val_ratio),
        },
        "notes": [
            "Only rows with a direct valid 'class' label were retained because the source CSV mixes another schema into the same file.",
            "Rows where the class looked shifted into a different column were ignored instead of heuristically reconstructed, to keep the fourth UAV dataset clean and reproducible.",
            "DoS attack was normalized to dos, Replay to replay, and benign remained benign.",
            "A synthetic monotonic timestamp was created so the prepared CSV matches the existing UCS-OODID experiment format.",
            "The file is ordered as ID-train -> ID-val -> ID-test -> OOD-test so scripts/train.py can reuse its explicit split column safely.",
        ],
        "recommended_train_command": (
            "python scripts/train.py "
            f"--input {output_path} "
            "--output_dir runs/uav04_normal_cyberattacks_default "
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
        "input_rows": int(source_meta["input_rows"]),
        "direct_valid_rows": int(source_meta["direct_valid_rows"]),
        "output_rows": int(len(prepared)),
        "ignored_blank_or_mixed_rows": int(source_meta["ignored_blank_or_mixed_rows"]),
        "duplicate_rows_removed": int(duplicate_rows_removed),
        "ambiguous_rows_removed": int(ambiguous_dropped),
        "full_label_counts": summary["full_label_counts"],
        "notes_json": str(notes_path) if notes_path is not None else None,
    }
    print(json.dumps({"prepare_uavs_normal_cyberattacks_summary": console_summary}, indent=2, ensure_ascii=False))
    return prepared, summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare a UCS-OODID experiment CSV from UAVs-Dataset-Under-Normal-and-Cyberattacks-main."
    )
    parser.add_argument(
        "--input",
        default=r"..\UAVs-Dataset-Under-Normal-and-Cyberattacks-main\Dataset_T-ITS.csv",
        help="Path to Dataset_T-ITS.csv.",
    )
    parser.add_argument(
        "--output",
        default=r"data\uavs_normal_cyberattacks_uav04_experiment.csv",
        help="Path to the prepared output CSV.",
    )
    parser.add_argument(
        "--notes_json",
        default=r"data\uavs_normal_cyberattacks_uav04_notes.json",
        help="Path to the preparation summary JSON.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chunk_size", type=int, default=4)
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

    prepare_uavs_normal_cyberattacks_dataset(
        input_csv=args.input,
        output=args.output,
        notes_json=args.notes_json,
        seed=args.seed,
        chunk_size=args.chunk_size,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        id_classes=parse_csv_list(args.id_classes),
        ood_classes=parse_csv_list(args.ood_classes),
        keep_duplicates=args.keep_duplicates,
        keep_conflicting_patterns=args.keep_conflicting_patterns,
    )


if __name__ == "__main__":
    main()
