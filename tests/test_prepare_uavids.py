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
MODULE_PATH = ROOT / "scripts" / "prepare_uavids.py"
SPEC = importlib.util.spec_from_file_location("prepare_uavids", MODULE_PATH)
prepare_uavids = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(prepare_uavids)

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


def make_row(idx: int, attack_label: str, split_value: str, notes: str) -> dict:
    return {
        "FlowID": idx,
        "FlowDuration/s": 100.0 + idx,
        "TxPackets": 200 + idx,
        "RxPackets": 180 + idx,
        "Protocol": "UDP",
        "class": attack_label,
        "capture_split": split_value,
        "notes": notes,
    }


class PrepareUavidsTests(unittest.TestCase):
    def test_normalise_label_maps_normal_and_wormhole(self):
        self.assertEqual(prepare_uavids.normalise_label("Normal Traffic"), "benign")
        self.assertEqual(prepare_uavids.normalise_label("Flooding Attack"), "flooding")
        self.assertEqual(prepare_uavids.normalise_label("Wormhole Attack"), "wormhole")

    def test_prepare_uavids_builds_unified_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "UAVIDS-2025.csv"
            output_path = root / "prepared.csv"

            pd.DataFrame(
                [
                    make_row(1, "Normal Traffic", "TRAIN", ""),
                    make_row(2, "Flooding Attack", "TRAIN", "ok"),
                    make_row(3, "Blackhole Attack", "TEST", "ok"),
                    make_row(4, "Wormhole Attack", "TEST", "ok"),
                ]
            ).to_csv(input_path, index=False)

            prepared, summary = prepare_uavids.prepare_uavids_dataset(
                input_csv=str(input_path),
                output=str(output_path),
                uav_id="uav-07",
                dataset_name="uavids_demo",
                split_column="capture_split",
            )

            required_columns = set(REQUIRED_METADATA_COLUMNS)
            self.assertTrue(output_path.exists())
            self.assertTrue(required_columns.issubset(prepared.columns))
            assert_required_metadata_prefix(self, prepared)
            self.assertEqual(prepared["source_type"].unique().tolist(), ["uav"])
            self.assertEqual(prepared["uav_id"].unique().tolist(), ["uav-07"])
            self.assertEqual(prepared["dataset_name"].unique().tolist(), ["uavids_demo"])
            self.assertEqual(
                prepared["label_normalized"].value_counts().to_dict(),
                {"benign": 1, "flooding": 1, "blackhole": 1, "wormhole": 1},
            )
            self.assertEqual(set(prepared["split"].tolist()), {"train", "test"})
            self.assertEqual(prepared["timestamp"].tolist(), [0, 1, 2, 3])

            self.assertEqual(summary["label_column_used"], "class")
            self.assertIsNone(summary["timestamp_column_used"])
            self.assertEqual(summary["split_column_used"], "capture_split")
            self.assertEqual(summary["benign_samples"], 1)
            self.assertEqual(
                summary["attack_samples_by_category"],
                {"blackhole": 1, "flooding": 1, "wormhole": 1},
            )
            self.assertEqual(summary["missing_values"]["notes"], 1)

            written = pd.read_csv(output_path)
            self.assertTrue(required_columns.issubset(written.columns))
            assert_required_metadata_prefix(self, written)

    def test_prepare_uavids_defaults_metadata_for_uav07_when_split_column_is_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "UAVIDS-2025.csv"
            output_path = root / "prepared.csv"

            raw = pd.DataFrame(
                [
                    make_row(1, "Normal Traffic", "TRAIN", "ok"),
                    make_row(2, "Sybil Attack", "TEST", "ood"),
                ]
            ).drop(columns=["capture_split"])
            raw.to_csv(input_path, index=False)

            prepared, summary = prepare_uavids.prepare_uavids_dataset(
                input_csv=str(input_path),
                output=str(output_path),
                uav_id="uav_07",
                dataset_name="uavids",
            )

            self.assertTrue(output_path.exists())
            assert_required_metadata_prefix(self, prepared)
            self.assertEqual(prepared["uav_id"].unique().tolist(), ["uav_07"])
            self.assertEqual(prepared["dataset_name"].unique().tolist(), ["uavids"])
            self.assertEqual(prepared["source_type"].unique().tolist(), ["uav"])
            self.assertEqual(prepared["split"].unique().tolist(), ["all"])
            self.assertEqual(prepared["timestamp"].tolist(), [0, 1])
            self.assertIsNone(summary["timestamp_column_used"])
            self.assertIsNone(summary["split_column_used"])
            self.assertEqual(prepared["label_normalized"].tolist(), ["benign", "sybil"])

            written = pd.read_csv(output_path)
            assert_required_metadata_prefix(self, written)

    def test_prepare_uavids_cli_smoke(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "UAVIDS-2025.csv"
            output_path = root / "prepared.csv"

            pd.DataFrame(
                [
                    make_row(1, "Normal Traffic", "TRAIN", "ok"),
                    make_row(2, "Sybil Attack", "TEST", "ok"),
                    make_row(3, "Blackhole Attack", "TEST", "ok"),
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
                    "uavids_cli",
                    "--label_column",
                    "class",
                ]
            )

            self.assertTrue(output_path.exists(), msg=completed.stdout + completed.stderr)
            payload = json.loads(completed.stdout)
            self.assertEqual(payload["prepare_uavids"]["total_samples"], 3)
            self.assertEqual(payload["prepare_uavids"]["source_type"], "uav")

            prepared = pd.read_csv(output_path)
            self.assertEqual(prepared["uav_id"].unique().tolist(), ["uav-cli"])
            self.assertEqual(prepared["split"].unique().tolist(), ["all"])


if __name__ == "__main__":
    unittest.main()
