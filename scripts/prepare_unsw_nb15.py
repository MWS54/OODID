#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

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
    find_csv_files,
    normalize_text_token,
    print_processing_summary,
    resolve_base_candidates,
    resolve_column_name,
    resolve_input_path,
    resolve_output_path,
    write_prepared_csv,
)

SOURCE_TYPE = "external_non_uav"
DEFAULT_INPUT = r"..\NUSW"
DEFAULT_OUTPUT = r"data\prepared_unsw_nb15.csv"
DEFAULT_UAV_ID = "external_unsw_nb15"
DEFAULT_DATASET_NAME = "unsw_nb15"
LABEL_CANDIDATES = ["attack_cat", "attack_type", "label", *COMMON_LABEL_CANDIDATES]
TIMESTAMP_CANDIDATES = ["stime", "ltime", *COMMON_TIMESTAMP_CANDIDATES]
SPLIT_CANDIDATES = list(COMMON_SPLIT_CANDIDATES)
UNSW_LABEL_MAP = {
    "0": "benign",
    "1": "attack",
    "benign": "benign",
    "normal": "benign",
    "generic": "generic",
    "exploits": "exploits",
    "fuzzers": "fuzzers",
    "dos": "dos",
    "reconnaissance": "reconnaissance",
    "recon": "reconnaissance",
    "analysis": "analysis",
    "backdoor": "backdoor",
    "backdoors": "backdoor",
    "shellcode": "shellcode",
    "worms": "worms",
}


def normalise_label(value: object) -> str:
    token = normalize_text_token(value)
    if token in {"benign", "normal"}:
        return "benign"
    return UNSW_LABEL_MAP.get(token, token)


def load_unsw_frame(input_value: str, base_candidates: Sequence[Path]) -> tuple[pd.DataFrame, list[Path]]:
    input_path = resolve_input_path(base_candidates, input_value)
    if input_path.is_file():
        csv_files = [input_path]
        if "training" in input_path.name.lower():
            sibling = input_path.with_name("UNSW_NB15_testing-set.csv")
            if sibling.exists():
                csv_files.append(sibling)
        elif "testing" in input_path.name.lower():
            sibling = input_path.with_name("UNSW_NB15_training-set.csv")
            if sibling.exists():
                csv_files.insert(0, sibling)
    else:
        csv_files = find_csv_files(
            input_path,
            preferred_names=["UNSW_NB15_training-set.csv", "UNSW_NB15_testing-set.csv"],
        )

    unique_files: list[Path] = []
    seen: set[Path] = set()
    for path in csv_files:
        if path not in seen:
            unique_files.append(path)
            seen.add(path)

    frames: list[pd.DataFrame] = []
    for path in unique_files:
        frame = pd.read_csv(path, low_memory=False)
        frame["source_file"] = path.stem
        frames.append(frame)
    return pd.concat(frames, ignore_index=True), unique_files


def prepare_unsw_nb15_dataset(
    input_path: str,
    output: str,
    uav_id: str = DEFAULT_UAV_ID,
    dataset_name: str = DEFAULT_DATASET_NAME,
    label_column: str | None = None,
    timestamp_column: str | None = None,
    split_column: str | None = None,
) -> tuple[pd.DataFrame, dict]:
    _, project_root, base_candidates = resolve_base_candidates(__file__)
    raw_df, input_files = load_unsw_frame(input_path, base_candidates)

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
        input_files=input_files,
        extra_notes=[
            "UNSW-NB15 is handled here as an external non-UAV benchmark dataset.",
            "Do not describe UNSW-NB15 as real UAV traffic in downstream documentation.",
        ],
    )
    write_prepared_csv(prepared, output_path)
    print_processing_summary("prepare_unsw_nb15", summary)
    return prepared, summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare a unified CSV from UNSW-NB15 as an external non-UAV benchmark dataset."
    )
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Path to a UNSW CSV file or the directory containing the official train/test CSV files.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Path to the unified output CSV.")
    parser.add_argument("--uav_id", default=DEFAULT_UAV_ID, help="Value written to the output uav_id column.")
    parser.add_argument("--dataset_name", default=DEFAULT_DATASET_NAME, help="Value written to the output dataset_name column.")
    parser.add_argument(
        "--label_column",
        default=None,
        help="Optional label column override. If omitted, the script prefers attack_cat over generic label columns.",
    )
    parser.add_argument(
        "--timestamp_column",
        default=None,
        help="Optional timestamp column override. If omitted, a synthetic monotonic timestamp is generated.",
    )
    parser.add_argument(
        "--split_column",
        default=None,
        help="Optional split column override. If omitted, split is filled with 'all'.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    prepare_unsw_nb15_dataset(
        input_path=args.input,
        output=args.output,
        uav_id=args.uav_id,
        dataset_name=args.dataset_name,
        label_column=args.label_column,
        timestamp_column=args.timestamp_column,
        split_column=args.split_column,
    )


if __name__ == "__main__":
    main()
