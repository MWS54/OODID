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
MODULE_PATH = ROOT / "scripts" / "prepare_multi_uav_hetero.py"
SPEC = importlib.util.spec_from_file_location("prepare_multi_uav_hetero", MODULE_PATH)
prepare_multi_uav_hetero = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(prepare_multi_uav_hetero)


class PrepareMultiUavHeteroTests(unittest.TestCase):
    def write_csv(self, root: Path, name: str, rows: dict) -> Path:
        path = root / name
        pd.DataFrame(rows).to_csv(path, index=False)
        return path

    def write_config(self, root: Path, datasets: list[dict]) -> Path:
        path = root / "datasets.yaml"
        path.write_text(json.dumps({"datasets": datasets}, indent=2), encoding="utf-8")
        return path

    def test_prepare_multi_uav_hetero_reads_yaml_and_supports_uav05_uav06_uav07_without_uav04(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_csv(
                root,
                "uav01.csv",
                {
                    "feat_a": [1.0],
                    "label": ["benign"],
                    "source_label": ["Normal"],
                    "split": ["train"],
                    "timestamp": [0.0],
                    "record_id": [1],
                },
            )
            self.write_csv(
                root,
                "uav02.csv",
                {
                    "feat_b": [2.0],
                    "label": ["DoS"],
                    "label_normalized": ["dos"],
                    "split": ["val"],
                    "timestamp": [1.0],
                    "record_id": [2],
                },
            )
            self.write_csv(
                root,
                "uav03.csv",
                {
                    "feat_c": [3.0],
                    "label": ["Replay Attack"],
                    "label_normalized": ["replay"],
                    "split": ["test_ood"],
                    "timestamp": [2.0],
                    "record_id": [3],
                },
            )
            self.write_csv(
                root,
                "uav05.csv",
                {
                    "feat_e": [5.0, 6.0],
                    "label": ["Generic", "Backdoors"],
                    "label_normalized": ["generic", "backdoor"],
                    "split": ["train", "test_ood"],
                    "timestamp": [3.0, 4.0],
                    "record_id": [4, 5],
                },
            )
            self.write_csv(
                root,
                "uav06.csv",
                {
                    "feat_f": [6.0],
                    "label": ["WPA2-PSK WIFI Cracking Attack"],
                    "label_normalized": ["wpa_cracking"],
                    "split": ["train"],
                    "timestamp": [5.0],
                },
            )
            self.write_csv(
                root,
                "uav07.csv",
                {
                    "feat_g": [7.0, 8.0],
                    "label": ["wormhole", "sybil"],
                    "source_label": ["Wormhole Attack", "Sybil Attack"],
                    "split": ["train", "test_ood"],
                    "timestamp": [6.0, 7.0],
                    "record_id": [6, 7],
                },
            )
            config_path = self.write_config(
                root,
                [
                    {"uav_id": "uav_01", "dataset_name": "uav_ndd", "csv": "uav01.csv", "source_type": "uav"},
                    {"uav_id": "uav_02", "dataset_name": "gcs_to_uav_updated", "csv": "uav02.csv", "source_type": "uav"},
                    {"uav_id": "uav_03", "dataset_name": "isot_drone", "csv": "uav03.csv", "source_type": "uav"},
                    {"uav_id": "uav_05", "dataset_name": "unsw_nb15", "csv": "uav05.csv", "source_type": "external_non_uav"},
                    {"uav_id": "uav_06", "dataset_name": "ecu_ioft", "csv": "uav06.csv", "source_type": "uav_iot_wifi"},
                    {"uav_id": "uav_07", "dataset_name": "uavids", "csv": "uav07.csv", "source_type": "uav"},
                ],
            )
            output_path = root / "merged.csv"
            summary_path = root / "merged_dataset_summary.json"

            merged, summary = prepare_multi_uav_hetero.prepare_multi_uav_hetero_dataset(
                output=str(output_path),
                summary_json=str(summary_path),
                config_path=str(config_path),
                include_uavs="uav_01,uav_02,uav_03,uav_05,uav_06,uav_07",
            )

            self.assertEqual(
                merged.columns.tolist(),
                [
                    "feat_a",
                    "feat_b",
                    "feat_c",
                    "feat_e",
                    "feat_f",
                    "feat_g",
                    "record_id",
                    "timestamp",
                    "uav_id",
                    "dataset_name",
                    "source_type",
                    "label",
                    "label_normalized",
                    "split",
                ],
            )
            self.assertEqual(
                merged["record_id"].tolist(),
                ["uav_01_1", "uav_02_2", "uav_03_3", "uav_05_4", "uav_05_5", "uav_06_0", "uav_07_6", "uav_07_7"],
            )
            self.assertEqual(summary["included_uav_ids"], ["uav_01", "uav_02", "uav_03", "uav_05", "uav_06", "uav_07"])
            self.assertEqual(
                summary["included_dataset_names"],
                ["uav_ndd", "gcs_to_uav_updated", "isot_drone", "unsw_nb15", "ecu_ioft", "uavids"],
            )
            self.assertEqual(
                summary["sample_count_by_uav_id"],
                {
                    "uav_05": 2,
                    "uav_07": 2,
                    "uav_01": 1,
                    "uav_02": 1,
                    "uav_03": 1,
                    "uav_06": 1,
                },
            )
            self.assertEqual(
                summary["sample_count_by_dataset_name"],
                {
                    "unsw_nb15": 2,
                    "uavids": 2,
                    "uav_ndd": 1,
                    "gcs_to_uav_updated": 1,
                    "isot_drone": 1,
                    "ecu_ioft": 1,
                },
            )
            self.assertEqual(
                summary["sample_count_by_label_normalized"],
                {
                    "benign": 1,
                    "dos": 1,
                    "replay": 1,
                    "generic": 1,
                    "backdoor": 1,
                    "wpa_cracking": 1,
                    "wormhole": 1,
                    "sybil": 1,
                },
            )
            self.assertEqual(
                summary["sample_count_by_source_type"],
                {
                    "uav": 5,
                    "external_non_uav": 2,
                    "uav_iot_wifi": 1,
                },
            )
            self.assertTrue(output_path.exists())
            self.assertTrue(summary_path.exists())
            written_summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(written_summary["summary_json"], str(summary_path))

    def test_prepare_multi_uav_hetero_falls_back_to_source_label_when_label_normalized_is_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_csv(
                root,
                "uav01.csv",
                {
                    "feat_a": [1.0, 2.0],
                    "label": ["benign", "wormhole"],
                    "source_label": ["Normal Traffic", "Wormhole Attack"],
                    "split": ["train", "test_ood"],
                    "timestamp": [0.0, 1.0],
                    "record_id": [10, 11],
                },
            )
            self.write_csv(
                root,
                "uav05.csv",
                {
                    "feat_b": [3.0],
                    "label": ["Backdoor"],
                    "source_label": ["Backdoors"],
                    "split": ["train"],
                    "timestamp": [2.0],
                    "record_id": [12],
                },
            )
            config_path = self.write_config(
                root,
                [
                    {"uav_id": "uav_01", "dataset_name": "uav_ndd", "csv": "uav01.csv", "source_type": "uav"},
                    {"uav_id": "uav_05", "dataset_name": "unsw_nb15", "csv": "uav05.csv", "source_type": "external_non_uav"},
                ],
            )
            output_path = root / "fallback.csv"

            merged, summary = prepare_multi_uav_hetero.prepare_multi_uav_hetero_dataset(
                output=str(output_path),
                config_path=str(config_path),
                include_uavs=["uav_01", "uav_05"],
            )

            self.assertEqual(merged["label"].tolist(), ["Normal Traffic", "Wormhole Attack", "Backdoors"])
            self.assertEqual(merged["label_normalized"].tolist(), ["benign", "wormhole", "Backdoor"])
            self.assertEqual(merged["source_type"].tolist(), ["uav", "uav", "external_non_uav"])
            self.assertEqual(summary["sample_count_by_source_type"], {"uav": 2, "external_non_uav": 1})
            self.assertTrue((root / "merged_dataset_summary.json").exists())

    def test_prepare_multi_uav_hetero_raises_clear_error_when_label_column_is_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_csv(
                root,
                "uav01.csv",
                {
                    "feat_a": [1.0],
                    "label": ["benign"],
                    "label_normalized": ["benign"],
                    "split": ["train"],
                    "timestamp": [0.0],
                    "record_id": [1],
                },
            )
            self.write_csv(
                root,
                "uav05.csv",
                {
                    "feat_b": [2.0],
                    "wrong_label": ["Backdoors"],
                    "split": ["train"],
                    "timestamp": [1.0],
                    "record_id": [2],
                },
            )
            config_path = self.write_config(
                root,
                [
                    {"uav_id": "uav_01", "dataset_name": "uav_ndd", "csv": "uav01.csv", "source_type": "uav"},
                    {
                        "uav_id": "uav_05",
                        "dataset_name": "unsw_nb15",
                        "csv": "uav05.csv",
                        "source_type": "external_non_uav",
                        "label_column": "missing_label",
                    },
                ],
            )

            with self.assertRaisesRegex(ValueError, r"unsw_nb15 \(uav_05\).*missing_label"):
                prepare_multi_uav_hetero.prepare_multi_uav_hetero_dataset(
                    output=str(root / "error.csv"),
                    config_path=str(config_path),
                    include_uavs="uav_01,uav_05",
                )

    def test_prepare_multi_uav_hetero_cli_supports_include_uavs_and_default_summary_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_csv(
                root,
                "uav01.csv",
                {
                    "feat_a": [1.0],
                    "label": ["benign"],
                    "label_normalized": ["benign"],
                    "split": ["train"],
                    "timestamp": [0.0],
                    "record_id": [1],
                },
            )
            self.write_csv(
                root,
                "uav05.csv",
                {
                    "feat_b": [2.0],
                    "label": ["Backdoors"],
                    "label_normalized": ["backdoor"],
                    "split": ["test_ood"],
                    "timestamp": [1.0],
                    "record_id": [2],
                },
            )
            self.write_csv(
                root,
                "uav07.csv",
                {
                    "feat_c": [3.0],
                    "label": ["Sybil Attack"],
                    "label_normalized": ["sybil"],
                    "split": ["train"],
                    "timestamp": [2.0],
                    "record_id": [3],
                },
            )
            config_path = self.write_config(
                root,
                [
                    {"uav_id": "uav_01", "dataset_name": "uav_ndd", "csv": "uav01.csv", "source_type": "uav"},
                    {"uav_id": "uav_05", "dataset_name": "unsw_nb15", "csv": "uav05.csv", "source_type": "external_non_uav"},
                    {"uav_id": "uav_07", "dataset_name": "uavids", "csv": "uav07.csv", "source_type": "uav"},
                ],
            )
            output_path = root / "cli_merged.csv"

            result = subprocess.run(
                [
                    sys.executable,
                    str(MODULE_PATH),
                    "--config",
                    str(config_path),
                    "--include_uavs",
                    "uav_01,uav_05",
                    "--output",
                    str(output_path),
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            summary_path = root / "merged_dataset_summary.json"
            self.assertTrue(output_path.exists())
            self.assertTrue(summary_path.exists())
            payload = json.loads(result.stdout)
            self.assertEqual(
                payload["prepare_multi_uav_hetero_summary"]["included_uav_ids"],
                ["uav_01", "uav_05"],
            )


if __name__ == "__main__":
    unittest.main()
