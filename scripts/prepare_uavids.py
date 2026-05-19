#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ucs_oodid.prepare_common import (
    COMMON_LABEL_CANDIDATES,
    COMMON_SPLIT_CANDIDATES,
    COMMON_TIMESTAMP_CANDIDATES,
    build_prepared_frame,
    build_processing_summary,
    load_single_csv_from_input,
    normalize_text_token,
    print_processing_summary,
    resolve_base_candidates,
    resolve_column_name,
    resolve_output_path,
    write_prepared_csv,
)

SOURCE_TYPE = "uav"
DEFAULT_INPUT = r"..\UAVIDS"
DEFAULT_OUTPUT = r"data\prepared_uavids.csv"
DEFAULT_UAV_ID = "uavids"
DEFAULT_DATASET_NAME = "uavids"
LABEL_CANDIDATES = ["label", "class", "attack_type", "attack type", "attack", *COMMON_LABEL_CANDIDATES]
TIMESTAMP_CANDIDATES = list(COMMON_TIMESTAMP_CANDIDATES)
SPLIT_CANDIDATES = list(COMMON_SPLIT_CANDIDATES)
UAVIDS_LABEL_MAP = {
    "benign": "benign",
    "normal": "benign",
    "normal_traffic": "benign",
    "flooding_attack": "flooding",
    "flood_attack": "flooding",
    "flooding": "flooding",
    "dos": "dos",
    "dos_attack": "dos",
    "sybil_attack": "sybil",
    "sybil": "sybil",
    "blackhole_attack": "blackhole",
    "blackhole": "blackhole",
    "wormhole_attack": "wormhole",
    "wormhole": "wormhole",
}


def normalise_label(value: object) -> str:
    token = normalize_text_token(value)
    if token in {"benign", "normal", "normal_traffic"}:
        return "benign"
    return UAVIDS_LABEL_MAP.get(token, token)


def prepare_uavids_dataset(
    input_csv: str,
    output: str,
    uav_id: str = DEFAULT_UAV_ID,
    dataset_name: str = DEFAULT_DATASET_NAME,
    label_column: str | None = None,
    timestamp_column: str | None = None,
    split_column: str | None = None,
) -> tuple[pd.DataFrame, dict]:
    _, project_root, base_candidates = resolve_base_candidates(__file__)
    raw_df, input_path = load_single_csv_from_input(
        base_candidates,
        input_csv,
        preferred_names=["UAVIDS-2025.csv"],
    )
    resolved_label_column = resolve_column_name(
        raw_df,
        label_column,
        LABEL_CANDIDATES,
        purpose="label",
        required=True,
    )
    resolved_timestamp_column = resolve_column_name(
        raw_df,
        timestamp_column,
        TIMESTAMP_CANDIDATES,
        purpose="timestamp",
        required=False,
    )
    resolved_split_column = resolve_column_name(
        raw_df,
        split_column,
        SPLIT_CANDIDATES,
        purpose="split",
        required=False,
    )
    output_path = resolve_output_path(project_root, output)

    prepared = build_prepared_frame(
        raw_df,
        label_column=resolved_label_column,
        timestamp_column=resolved_timestamp_column,
        split_column=resolved_split_column,
        uav_id=uav_id,
        dataset_name=dataset_name,
        source_type=SOURCE_TYPE,
        label_normalizer=normalise_label,
    )
    summary = build_processing_summary(
        prepared,
        output_path=output_path,
        label_column=resolved_label_column,
        timestamp_column=resolved_timestamp_column,
        split_column=resolved_split_column,
        dataset_name=dataset_name,
        source_type=SOURCE_TYPE,
        input_files=[input_path],
        extra_notes=["UAVIDS labels are normalized with common aliases such as normal->benign and blackhole/wormhole attack variants."],
    )
    write_prepared_csv(prepared, output_path)
    print_processing_summary("prepare_uavids", summary)
    return prepared, summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare a unified CSV from the UAVIDS dataset.")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Path to UAVIDS-2025.csv or the UAVIDS dataset root directory.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Path to the unified output CSV.")
    parser.add_argument("--uav_id", default=DEFAULT_UAV_ID, help="Value written to the output uav_id column.")
    parser.add_argument("--dataset_name", default=DEFAULT_DATASET_NAME, help="Value written to the output dataset_name column.")
    parser.add_argument(
        "--label_column",
        default=None,
        help="Optional label column override. If omitted, the script auto-detects label/class/attack_type style columns.",
    )
    parser.add_argument(
        "--timestamp_column",
        default=None,
        help="Optional timestamp column override. If omitted, common timestamp columns are auto-detected and otherwise a synthetic timestamp is generated.",
    )
    parser.add_argument(
        "--split_column",
        default=None,
        help="Optional split column override. If omitted, split is filled with 'all'.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    prepare_uavids_dataset(
        input_csv=args.input,
        output=args.output,
        uav_id=args.uav_id,
        dataset_name=args.dataset_name,
        label_column=args.label_column,
        timestamp_column=args.timestamp_column,
        split_column=args.split_column,
    )


if __name__ == "__main__":
    main()
