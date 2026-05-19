from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "prepare_ecu_ioft.py"
SPEC = importlib.util.spec_from_file_location("prepare_ecu_ioft", MODULE_PATH)
prepare_ecu_ioft = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(prepare_ecu_ioft)

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


def run_command(cmd: list[object]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(part) for part in cmd],
        cwd=str(ROOT),
        check=True,
        capture_output=True,
        text=True,
    )


def assert_required_metadata_prefix(testcase: unittest.TestCase, frame: pd.DataFrame) -> None:
    testcase.assertEqual(
        frame.columns[: len(REQUIRED_METADATA_COLUMNS)].tolist(),
        REQUIRED_METADATA_COLUMNS,
    )


def make_row(idx: int, attack_label: str, split_value: str, comment: str | None) -> dict:
    return {
        "ID": idx,
        "Time": f"2021-12-09 04:34:{idx:02d}",
        "Source": "192.168.10.1",
        "Destination": "192.168.10.2",
        "Protocol": "UDP" if "TELLO" in attack_label else "802.11",
        "Length": 100 + idx,
        "Info": "Deauthentication" if "Deauthentication" in attack_label else "Normal frame",
        "Type": "Normal" if attack_label == "No Attack" else "Attack",
        "Type of Attack": attack_label,
        "Attack Scenario": attack_label,
        "dataset_split": split_value,
        "comment": comment,
    }


class PrepareEcuIoftTests(unittest.TestCase):
    def test_normalise_label_maps_common_wifi_attacks(self):
        self.assertEqual(prepare_ecu_ioft.normalise_label("No Attack"), "benign")
        self.assertEqual(prepare_ecu_ioft.normalise_label("Wifi Deauthentication Attack"), "wifi_deauth")
        self.assertEqual(prepare_ecu_ioft.normalise_label("WPA2-PSK WIFI Cracking Attack"), "wpa_cracking")
        self.assertEqual(prepare_ecu_ioft.normalise_label("TELLO API Exploit"), "tello_api_exploit")

    def test_prepare_ecu_ioft_builds_unified_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "ECU-IoFT-Dataset.csv"
            output_path = root / "prepared.csv"

            pd.DataFrame(
                [
                    make_row(0, "No Attack", "TRAIN", None),
                    make_row(1, "Wifi Deauthentication Attack", "TRAIN", "ok"),
                    make_row(2, "WPA2-PSK WIFI Cracking Attack", "TEST", "ok"),
                    make_row(3, "TELLO API Exploit", "TEST", "ok"),
                ]
            ).to_csv(input_path, index=False)

            prepared, summary = prepare_ecu_ioft.prepare_ecu_ioft_dataset(
                input_csv=str(input_path),
                output=str(output_path),
                uav_id="uav-ecu-01",
                dataset_name="ecu_demo",
                split_column="dataset_split",
            )

            required_columns = set(REQUIRED_METADATA_COLUMNS)
            self.assertTrue(output_path.exists())
            self.assertTrue(required_columns.issubset(prepared.columns))
            assert_required_metadata_prefix(self, prepared)
            self.assertEqual(prepared["source_type"].unique().tolist(), ["uav_iot_wifi"])
            self.assertEqual(prepared["uav_id"].unique().tolist(), ["uav-ecu-01"])
            self.assertEqual(prepared["dataset_name"].unique().tolist(), ["ecu_demo"])
            self.assertEqual(
                prepared["label_normalized"].value_counts().to_dict(),
                {"benign": 1, "wifi_deauth": 1, "wpa_cracking": 1, "tello_api_exploit": 1},
            )
            self.assertEqual(set(prepared["split"].tolist()), {"train", "test"})
            self.assertTrue(all("T" in str(value) for value in prepared["timestamp"].tolist()))

            self.assertEqual(summary["label_column_used"], "Type of Attack")
            self.assertEqual(summary["timestamp_column_used"], "Time")
            self.assertEqual(summary["split_column_used"], "dataset_split")
            self.assertEqual(summary["benign_samples"], 1)
            self.assertEqual(
                summary["attack_samples_by_category"],
                {"tello_api_exploit": 1, "wifi_deauth": 1, "wpa_cracking": 1},
            )
            self.assertEqual(summary["missing_values"]["comment"], 1)

            written = pd.read_csv(output_path)
            self.assertTrue(required_columns.issubset(written.columns))
            assert_required_metadata_prefix(self, written)

    def test_prepare_ecu_ioft_defaults_metadata_for_uav06_when_optional_columns_are_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "ECU-IoFT-Dataset.csv"
            output_path = root / "prepared.csv"

            raw = pd.DataFrame(
                [
                    make_row(0, "No Attack", "TRAIN", "ok"),
                    make_row(1, "TELLO API Exploit", "TEST", "ood"),
                ]
            ).drop(columns=["Time", "dataset_split"])
            raw.to_csv(input_path, index=False)

            prepared, summary = prepare_ecu_ioft.prepare_ecu_ioft_dataset(
                input_csv=str(input_path),
                output=str(output_path),
                uav_id="uav_06",
                dataset_name="ecu_ioft",
            )

            self.assertTrue(output_path.exists())
            assert_required_metadata_prefix(self, prepared)
            self.assertEqual(prepared["uav_id"].unique().tolist(), ["uav_06"])
            self.assertEqual(prepared["dataset_name"].unique().tolist(), ["ecu_ioft"])
            self.assertEqual(prepared["source_type"].unique().tolist(), ["uav_iot_wifi"])
            self.assertEqual(prepared["split"].unique().tolist(), ["all"])
            self.assertEqual(prepared["timestamp"].tolist(), [0, 1])
            self.assertIsNone(summary["timestamp_column_used"])
            self.assertIsNone(summary["split_column_used"])
            self.assertEqual(prepared["label_normalized"].tolist(), ["benign", "tello_api_exploit"])

            written = pd.read_csv(output_path)
            assert_required_metadata_prefix(self, written)

    def test_prepare_ecu_ioft_cli_smoke(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_root = root / "dataset"
            dataset_root.mkdir(parents=True, exist_ok=True)
            input_path = dataset_root / "ECU-IoFT-Dataset.csv"
            output_path = root / "prepared.csv"

            pd.DataFrame(
                [
                    make_row(0, "No Attack", "TRAIN", "ok"),
                    make_row(1, "Wifi Deauthentication Attack", "TEST", "ok"),
                    make_row(2, "TELLO API Exploit", "TEST", "ok"),
                ]
            ).to_csv(input_path, index=False)

            completed = run_command(
                [
                    sys.executable,
                    MODULE_PATH,
                    "--input",
                    root,
                    "--output",
                    output_path,
                    "--uav_id",
                    "uav-cli",
                    "--dataset_name",
                    "ecu_cli",
                ]
            )

            self.assertTrue(output_path.exists(), msg=completed.stdout + completed.stderr)
            payload = json.loads(completed.stdout)
            self.assertEqual(payload["prepare_ecu_ioft"]["total_samples"], 3)
            self.assertEqual(payload["prepare_ecu_ioft"]["source_type"], "uav_iot_wifi")

            prepared = pd.read_csv(output_path)
            self.assertEqual(prepared["uav_id"].unique().tolist(), ["uav-cli"])
            self.assertEqual(prepared["split"].unique().tolist(), ["all"])


if __name__ == "__main__":
    unittest.main()
