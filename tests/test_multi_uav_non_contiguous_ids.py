from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from ucs_oodid.dataset_registry import active_simulation_dataset_bindings, default_simulation_uav_ids

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "prepare_multi_uav_hetero.py"
SPEC = importlib.util.spec_from_file_location("prepare_multi_uav_hetero", MODULE_PATH)
prepare_multi_uav_hetero = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(prepare_multi_uav_hetero)

NON_CONTIGUOUS_UAV_IDS = ["uav_01", "uav_02", "uav_03", "uav_05", "uav_06", "uav_07"]
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


def write_csv(path: Path, rows: list[dict[str, object]]) -> Path:
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def write_config(root: Path, datasets: list[dict[str, object]]) -> Path:
    path = root / "datasets.yaml"
    path.write_text(json.dumps({"datasets": datasets}, indent=2), encoding="utf-8")
    return path


def build_merge_inputs(root: Path) -> Path:
    write_csv(
        root / "uav01.csv",
        [
            {
                "feat_a": 1.0,
                "record_id": 1,
                "timestamp": 0.0,
                "uav_id": "uav_01",
                "dataset_name": "uav_ndd",
                "source_type": "uav",
                "label": "benign",
                "label_normalized": "benign",
                "split": "train",
            }
        ],
    )
    write_csv(
        root / "uav02.csv",
        [
            {
                "feat_b": 2.0,
                "record_id": 2,
                "timestamp": 1.0,
                "uav_id": "uav_02",
                "dataset_name": "gcs_to_uav_updated",
                "source_type": "uav",
                "label": "reply",
                "label_normalized": "replay",
                "split": "val",
            }
        ],
    )
    write_csv(
        root / "uav03.csv",
        [
            {
                "feat_c": 3.0,
                "record_id": 3,
                "timestamp": 2.0,
                "uav_id": "uav_03",
                "dataset_name": "isot_drone",
                "source_type": "uav",
                "label": "Injection Attack",
                "label_normalized": "injection",
                "split": "test_id",
            }
        ],
    )
    write_csv(
        root / "uav05.csv",
        [
            {
                "feat_e": 5.0,
                "record_id": 5,
                "timestamp": 3.0,
                "uav_id": "uav_05",
                "dataset_name": "unsw_nb15",
                "source_type": "external_non_uav",
                "label": "Backdoors",
                "label_normalized": "backdoor",
                "split": "test_ood",
            }
        ],
    )
    write_csv(
        root / "uav06.csv",
        [
            {
                "feat_f": 6.0,
                "record_id": 6,
                "timestamp": 4.0,
                "uav_id": "uav_06",
                "dataset_name": "ecu_ioft",
                "source_type": "uav_iot_wifi",
                "label": "WPA2-PSK WIFI Cracking Attack",
                "label_normalized": "wpa_cracking",
                "split": "test_ood",
            }
        ],
    )
    write_csv(
        root / "uav07.csv",
        [
            {
                "feat_g": 7.0,
                "record_id": 7,
                "timestamp": 5.0,
                "uav_id": "uav_07",
                "dataset_name": "uavids",
                "source_type": "uav",
                "label": "Wormhole Attack",
                "label_normalized": "wormhole",
                "split": "test_ood",
            }
        ],
    )
    return write_config(
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


def build_entrypoint_dataset(path: Path) -> Path:
    rows: list[dict[str, object]] = []
    uav_specs = [
        ("uav_01", "uav_ndd", "uav", "dos"),
        ("uav_02", "gcs_to_uav_updated", "uav", "replay"),
        ("uav_03", "isot_drone", "uav", "injection"),
        ("uav_05", "unsw_nb15", "external_non_uav", "generic"),
        ("uav_06", "ecu_ioft", "uav_iot_wifi", "unauthorized_udp"),
        ("uav_07", "uavids", "uav", "wormhole"),
    ]
    timestamp = 0
    for idx, (uav_id, dataset_name, source_type, id_attack_label) in enumerate(uav_specs, start=1):
        base = float(idx)
        rows.extend(
            [
                {
                    "feat_main": base,
                    "feat_aux": base + 0.1,
                    "record_id": f"{uav_id}_train",
                    "timestamp": timestamp,
                    "uav_id": uav_id,
                    "dataset_name": dataset_name,
                    "source_type": source_type,
                    "label": "benign",
                    "label_normalized": "benign",
                    "split": "train",
                },
                {
                    "feat_main": base + 1.0,
                    "feat_aux": base + 1.1,
                    "record_id": f"{uav_id}_val",
                    "timestamp": timestamp + 1,
                    "uav_id": uav_id,
                    "dataset_name": dataset_name,
                    "source_type": source_type,
                    "label": "benign",
                    "label_normalized": "benign",
                    "split": "val",
                },
                {
                    "feat_main": base + 2.0,
                    "feat_aux": base + 2.1,
                    "record_id": f"{uav_id}_test_id",
                    "timestamp": timestamp + 2,
                    "uav_id": uav_id,
                    "dataset_name": dataset_name,
                    "source_type": source_type,
                    "label": id_attack_label,
                    "label_normalized": id_attack_label,
                    "split": "test_id",
                },
            ]
        )
        timestamp += 10

    rows.extend(
        [
            {
                "feat_main": 15.0,
                "feat_aux": 15.1,
                "record_id": "uav_05_test_ood",
                "timestamp": 100,
                "uav_id": "uav_05",
                "dataset_name": "unsw_nb15",
                "source_type": "external_non_uav",
                "label": "backdoor",
                "label_normalized": "backdoor",
                "split": "test_ood",
            },
            {
                "feat_main": 16.0,
                "feat_aux": 16.1,
                "record_id": "uav_06_test_ood",
                "timestamp": 101,
                "uav_id": "uav_06",
                "dataset_name": "ecu_ioft",
                "source_type": "uav_iot_wifi",
                "label": "wpa_cracking",
                "label_normalized": "wpa_cracking",
                "split": "test_ood",
            },
            {
                "feat_main": 17.0,
                "feat_aux": 17.1,
                "record_id": "uav_07_test_ood",
                "timestamp": 102,
                "uav_id": "uav_07",
                "dataset_name": "uavids",
                "source_type": "uav",
                "label": "sybil",
                "label_normalized": "sybil",
                "split": "test_ood",
            },
        ]
    )
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


class MultiUavNonContiguousIdTests(unittest.TestCase):
    def test_default_simulation_registry_skips_uav04(self):
        bindings = active_simulation_dataset_bindings(6)
        self.assertEqual([row["uav_id"] for row in bindings], NON_CONTIGUOUS_UAV_IDS)
        self.assertEqual(default_simulation_uav_ids(6), NON_CONTIGUOUS_UAV_IDS)
        self.assertNotIn("uav_04", {row["uav_id"] for row in bindings})

    def test_default_dataset_config_skips_uav04(self):
        configs = prepare_multi_uav_hetero.load_dataset_configs(ROOT / "configs" / "datasets.yaml")
        self.assertEqual([config.uav_id for config in configs], NON_CONTIGUOUS_UAV_IDS)
        self.assertEqual(
            [config.dataset_name for config in configs],
            ["uav_ndd", "gcs_to_uav_updated", "isot_drone", "unsw_nb15", "ecu_ioft", "uavids"],
        )
        self.assertNotIn("uav_04", [config.uav_id for config in configs])

    def test_prepare_multi_uav_hetero_merges_non_contiguous_ids_without_uav04(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = build_merge_inputs(root)
            output_path = root / "merged.csv"
            summary_path = root / "merged_dataset_summary.json"

            merged, summary = prepare_multi_uav_hetero.prepare_multi_uav_hetero_dataset(
                output=str(output_path),
                summary_json=str(summary_path),
                config_path=str(config_path),
                include_uavs=",".join(NON_CONTIGUOUS_UAV_IDS),
            )

            self.assertTrue(output_path.exists())
            self.assertTrue(summary_path.exists())
            self.assertEqual(merged["uav_id"].tolist(), NON_CONTIGUOUS_UAV_IDS)
            self.assertNotIn("uav_04", merged["uav_id"].tolist())
            self.assertEqual(
                merged["record_id"].tolist(),
                ["uav_01_1", "uav_02_2", "uav_03_3", "uav_05_5", "uav_06_6", "uav_07_7"],
            )
            self.assertEqual(merged.columns.tolist()[-len(REQUIRED_METADATA_COLUMNS) :], REQUIRED_METADATA_COLUMNS)
            self.assertEqual(summary["included_uav_ids"], NON_CONTIGUOUS_UAV_IDS)
            self.assertNotIn("uavs_normal_cyberattacks", summary["included_dataset_names"])
            self.assertEqual(
                summary["sample_count_by_uav_id"],
                {uav_id: 1 for uav_id in NON_CONTIGUOUS_UAV_IDS},
            )

            written = pd.read_csv(output_path)
            self.assertEqual(written.columns.tolist()[-len(REQUIRED_METADATA_COLUMNS) :], REQUIRED_METADATA_COLUMNS)
            self.assertNotIn("uav_04", written["uav_id"].tolist())

            written_summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(written_summary["included_uav_ids"], NON_CONTIGUOUS_UAV_IDS)
            self.assertNotIn("uav_04", written_summary["sample_count_by_uav_id"])

    def test_train_detect_and_benchmark_accept_non_contiguous_multi_uav_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_path = build_entrypoint_dataset(root / "multi_uav_non_contiguous.csv")
            run_dir = root / "run"
            benchmark_json = run_dir / "benchmark_report.json"

            train_result = run_command(
                [
                    sys.executable,
                    ROOT / "scripts" / "train.py",
                    "--input",
                    data_path,
                    "--output_dir",
                    run_dir,
                    "--label_col",
                    "label_normalized",
                    "--group_col",
                    "uav_id",
                    "--id_classes",
                    "benign,dos,replay,injection,generic,unauthorized_udp,wormhole",
                    "--ood_classes",
                    "backdoor,wpa_cracking,sybil",
                    "--epochs",
                    "1",
                    "--batch_size",
                    "8",
                    "--hidden_dim",
                    "16",
                    "--num_heads",
                    "2",
                    "--num_layers",
                    "1",
                    "--window_size",
                    "1",
                    "--stride",
                    "1",
                ]
            )
            self.assertTrue((run_dir / "artifact.pt").exists(), msg=train_result.stdout + train_result.stderr)
            self.assertTrue((run_dir / "eval_report.json").exists(), msg=train_result.stdout + train_result.stderr)

            detect_result = run_command(
                [
                    sys.executable,
                    ROOT / "scripts" / "detect.py",
                    "--input",
                    data_path,
                    "--artifact",
                    run_dir / "artifact.pt",
                    "--output_jsonl",
                    run_dir / "detections.jsonl",
                    "--record_scores_json",
                    run_dir / "record_scores.json",
                    "--summary_json",
                    run_dir / "group_detection_summary.json",
                    "--label_col",
                    "label_normalized",
                    "--group_col",
                    "uav_id",
                    "--batch_size",
                    "16",
                ]
            )
            self.assertTrue((run_dir / "detections.jsonl").exists(), msg=detect_result.stdout + detect_result.stderr)
            self.assertTrue((run_dir / "record_scores.json").exists(), msg=detect_result.stdout + detect_result.stderr)
            self.assertTrue((run_dir / "group_detection_summary.json").exists(), msg=detect_result.stdout + detect_result.stderr)

            group_summary = json.loads((run_dir / "group_detection_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(set(group_summary["groups"]), set(NON_CONTIGUOUS_UAV_IDS))
            self.assertNotIn("uav_04", group_summary["groups"])

            benchmark_result = run_command(
                [
                    sys.executable,
                    ROOT / "scripts" / "benchmark_onboard.py",
                    "--artifact",
                    run_dir / "artifact.pt",
                    "--input",
                    data_path,
                    "--group_col",
                    "uav_id",
                    "--batch_size",
                    "16",
                    "--warmup_runs",
                    "1",
                    "--repeat_runs",
                    "2",
                    "--output_json",
                    benchmark_json,
                ]
            )
            self.assertTrue(benchmark_json.exists(), msg=benchmark_result.stdout + benchmark_result.stderr)
            report = json.loads(benchmark_json.read_text(encoding="utf-8"))

        self.assertGreater(report["num_windows"], 0)
        self.assertGreater(report["throughput_windows_per_sec"], 0.0)
        self.assertIn("deployment_profile", report)


if __name__ == "__main__":
    unittest.main()
