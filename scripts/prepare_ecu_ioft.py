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

SOURCE_TYPE = "uav_iot_wifi"
DEFAULT_INPUT = r"..\ECU-IoFT-main"
DEFAULT_OUTPUT = r"data\prepared_ecu_ioft.csv"
DEFAULT_UAV_ID = "uav_ecu_ioft"
DEFAULT_DATASET_NAME = "ecu_ioft_main"
LABEL_CANDIDATES = [
    "Type of Attack",
    "Attack Scenario",
    "attack_type",
    "attack type",
    "label",
    "class",
    "Type",
    *COMMON_LABEL_CANDIDATES,
]
TIMESTAMP_CANDIDATES = ["Time", *COMMON_TIMESTAMP_CANDIDATES]
SPLIT_CANDIDATES = list(COMMON_SPLIT_CANDIDATES)
ECU_LABEL_MAP = {
    "benign": "benign",
    "normal": "benign",
    "no_attack": "benign",
    "wi_fi_deauthentication_attack": "wifi_deauth",
    "wifi_deauthentication_attack": "wifi_deauth",
    "wifi_deauth": "wifi_deauth",
    "wi_fi_deauth": "wifi_deauth",
    "deauthentication": "wifi_deauth",
    "wpa2_psk_wifi_cracking_attack": "wpa_cracking",
    "wpa2_psk_cracking_attack": "wpa_cracking",
    "wpa_cracking": "wpa_cracking",
    "wifi_cracking": "wpa_cracking",
    "wifi_cracking_attack": "wpa_cracking",
    "tello_api_exploit": "tello_api_exploit",
}


def normalise_label(value: object) -> str:
    token = normalize_text_token(value)
    if token in {"benign", "normal", "no_attack"}:
        return "benign"
    return ECU_LABEL_MAP.get(token, token)


def prepare_ecu_ioft_dataset(
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
        preferred_names=["ECU-IoFT-Dataset.csv"],
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
        extra_notes=[
            "Common ECU-IoFT Wi-Fi attack labels are normalized to benign, wifi_deauth, wpa_cracking, and tello_api_exploit where possible."
        ],
    )
    write_prepared_csv(prepared, output_path)
    print_processing_summary("prepare_ecu_ioft", summary)
    return prepared, summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare a unified CSV from the ECU-IoFT Wi-Fi UAV/IoT dataset.")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Path to ECU-IoFT-Dataset.csv or the ECU-IoFT dataset root directory.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Path to the unified output CSV.")
    parser.add_argument("--uav_id", default=DEFAULT_UAV_ID, help="Value written to the output uav_id column.")
    parser.add_argument("--dataset_name", default=DEFAULT_DATASET_NAME, help="Value written to the output dataset_name column.")
    parser.add_argument(
        "--label_column",
        default=None,
        help="Optional label column override. If omitted, the script prefers Type of Attack or Attack Scenario.",
    )
    parser.add_argument(
        "--timestamp_column",
        default=None,
        help="Optional timestamp column override. If omitted, Time is auto-detected when present, otherwise a synthetic timestamp is generated.",
    )
    parser.add_argument(
        "--split_column",
        default=None,
        help="Optional split column override. If omitted, split is filled with 'all'.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    prepare_ecu_ioft_dataset(
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
