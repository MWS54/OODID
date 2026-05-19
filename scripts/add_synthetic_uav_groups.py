#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from ucs_oodid.io import load_records

DEFAULT_SPLIT_ORDER = ["train", "val", "test_id", "test_ood"]


def _ordered_partition_values(values: np.ndarray) -> List[str]:
    seen = [str(v) for v in pd.unique(values)]
    known = [name for name in DEFAULT_SPLIT_ORDER if name in seen]
    unknown = [name for name in seen if name not in DEFAULT_SPLIT_ORDER]
    return known + unknown


def _assign_block(n: int, num_groups: int) -> np.ndarray:
    if n <= 0:
        return np.zeros(0, dtype=np.int64)
    base, extra = divmod(n, num_groups)
    out = np.empty(n, dtype=np.int64)
    start = 0
    for group_idx in range(num_groups):
        width = base + (1 if group_idx < extra else 0)
        stop = start + width
        out[start:stop] = group_idx
        start = stop
    return out


def _assign_partition(
    frame: pd.DataFrame,
    num_groups: int,
    strategy: str,
    label_col: Optional[str],
) -> np.ndarray:
    n = len(frame)
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    if strategy == "round_robin":
        return np.arange(n, dtype=np.int64) % num_groups
    if strategy == "block":
        return _assign_block(n, num_groups)
    if not label_col or label_col not in frame.columns:
        raise ValueError(f"label_col {label_col!r} is required for strategy {strategy!r}.")

    labels = frame[label_col].fillna("<NA>").astype(str).reset_index(drop=True)
    out = np.empty(n, dtype=np.int64)
    offset = 0
    for _, positions in labels.groupby(labels, sort=False).indices.items():
        pos = np.asarray(positions, dtype=np.int64)
        out[pos] = (np.arange(len(pos), dtype=np.int64) + offset) % num_groups
        offset = (offset + len(pos)) % num_groups
    return out


def add_group_column(
    df: pd.DataFrame,
    num_groups: int,
    group_col: str,
    strategy: str,
    split_col: Optional[str],
    label_col: Optional[str],
    prefix: str,
) -> tuple[pd.DataFrame, Dict[str, object]]:
    if not group_col:
        raise ValueError("group_col must be a non-empty string.")
    if num_groups < 2:
        raise ValueError("num_groups must be at least 2 for a multi-UAV experiment.")
    if group_col in df.columns:
        raise ValueError(f"group_col {group_col!r} already exists in the input data.")

    result = df.copy()
    assigned = np.empty(len(result), dtype=object)

    if split_col and split_col in result.columns:
        split_values = result[split_col].fillna("<NA>").astype(str).to_numpy()
        partitions = _ordered_partition_values(split_values)
        partition_positions = {
            split_value: np.flatnonzero(split_values == split_value) for split_value in partitions
        }
    else:
        partitions = ["all_rows"]
        partition_positions = {"all_rows": np.arange(len(result), dtype=np.int64)}

    for partition_name in partitions:
        positions = partition_positions[partition_name]
        local_assign = _assign_partition(
            result.iloc[positions].reset_index(drop=True),
            num_groups=num_groups,
            strategy=strategy,
            label_col=label_col,
        )
        assigned[positions] = np.asarray([f"{prefix}{idx:02d}" for idx in local_assign], dtype=object)

    result[group_col] = assigned

    split_group_counts: Dict[str, Dict[str, int]] = {}
    for partition_name in partitions:
        positions = partition_positions[partition_name]
        counts = result.iloc[positions][group_col].astype(str).value_counts().to_dict()
        split_group_counts[partition_name] = {str(k): int(v) for k, v in counts.items()}

    summary: Dict[str, object] = {
        "rows": int(len(result)),
        "group_col": group_col,
        "num_groups": int(num_groups),
        "strategy": strategy,
        "split_col": split_col if split_col and split_col in result.columns else None,
        "label_col": label_col if label_col and label_col in result.columns else None,
        "global_group_counts": {str(k): int(v) for k, v in result[group_col].astype(str).value_counts().to_dict().items()},
        "partition_group_counts": split_group_counts,
    }
    return result, summary


def main() -> None:
    p = argparse.ArgumentParser(description="Add a synthetic UAV grouping column to a CSV that lacks a true uav_id.")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--notes_json", default="")
    p.add_argument("--group_col", default="uav_id")
    p.add_argument("--num_uavs", type=int, default=4)
    p.add_argument("--uav_prefix", default="uav_")
    p.add_argument("--strategy", choices=["stratified_label_round_robin", "round_robin", "block"], default="stratified_label_round_robin")
    p.add_argument("--split_col", default="split")
    p.add_argument("--label_col", default="label")
    args = p.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    notes_path = Path(args.notes_json) if args.notes_json else output_path.with_name(f"{output_path.stem}_notes.json")

    df = load_records(input_path)
    grouped, summary = add_group_column(
        df,
        num_groups=args.num_uavs,
        group_col=args.group_col,
        strategy=args.strategy,
        split_col=args.split_col or None,
        label_col=args.label_col or None,
        prefix=args.uav_prefix,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    grouped.to_csv(output_path, index=False)

    notes = {
        "input_csv": str(input_path.resolve()),
        "output_csv": str(output_path.resolve()),
        "notes_json": str(notes_path.resolve()),
        "summary": summary,
        "notes": [
            "The original CSV does not contain a true UAV identifier, so the generated group column is synthetic.",
            "Synthetic UAV IDs were assigned independently inside each split to avoid cross-split leakage.",
            "The default stratified_label_round_robin strategy balances labels across UAV groups while preserving each split boundary.",
            "Use the generated group column with scripts/train.py and scripts/detect.py via --group_col.",
        ],
        "recommended_train_command": (
            "python scripts/train.py "
            f"--input {output_path} "
            "--output_dir runs/uav_ndd_case1_multi_uav "
            f"--label_col {args.label_col} "
            "--timestamp_col timestamp "
            "--record_id_col record_id "
            f"--group_col {args.group_col} "
            "--ood_classes fake_landing,recon_scanning,replay "
            "--window_mode count --window_size 16 --stride 8 --epochs 10"
        ),
    }
    notes_path.write_text(json.dumps(notes, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps({
        "synthetic_uav_grouping": {
            "output_csv": str(output_path),
            "notes_json": str(notes_path),
            **summary,
        }
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
