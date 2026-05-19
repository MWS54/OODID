from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run_command(cmd: list[object]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(part) for part in cmd],
        cwd=str(ROOT),
        check=True,
        capture_output=True,
        text=True,
    )


class EntrypointRegressionTests(unittest.TestCase):
    def test_train_detect_and_benchmark_scripts_still_run(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            data_path = tmp_path / "smoke_synthetic.csv"
            run_dir = tmp_path / "run"
            benchmark_json = run_dir / "benchmark_report.json"

            run_command([sys.executable, ROOT / "scripts" / "make_synthetic.py", "--output", data_path, "--records", "320", "--seed", "7"])

            train_result = run_command(
                [
                    sys.executable,
                    ROOT / "scripts" / "train.py",
                    "--input",
                    data_path,
                    "--output_dir",
                    run_dir,
                    "--id_classes",
                    "benign,dos,mitm,spoof,replay,injection,scan",
                    "--ood_classes",
                    "unknown_probe,unknown_burst",
                    "--epochs",
                    "1",
                    "--batch_size",
                    "32",
                    "--hidden_dim",
                    "32",
                    "--num_heads",
                    "4",
                    "--num_layers",
                    "1",
                    "--window_size",
                    "8",
                    "--stride",
                    "4",
                ]
            )
            self.assertTrue((run_dir / "artifact.pt").exists(), msg=train_result.stdout + train_result.stderr)
            self.assertTrue((run_dir / "eval_report.json").exists(), msg=train_result.stdout + train_result.stderr)
            self.assertTrue((run_dir / "feature_columns.json").exists(), msg=train_result.stdout + train_result.stderr)
            eval_report = json.loads((run_dir / "eval_report.json").read_text(encoding="utf-8"))
            self.assertEqual(eval_report["method"], "recda_proxy")
            self.assertIn("run_config", eval_report)
            self.assertEqual(eval_report["run_config"]["method"], "recda_proxy")
            self.assertIn("ood_test", eval_report)
            self.assertIn("fpr_at_threshold", eval_report["ood_test"])
            self.assertIn("timing", eval_report)
            self.assertEqual(
                eval_report["timing"]["test_windows"],
                eval_report["timing"]["test_id_windows"] + eval_report["timing"]["test_ood_windows"],
            )
            self.assertIsNotNone(eval_report["timing"]["average_detection_time_ms"])
            self.assertIsNotNone(eval_report["timing"]["throughput_windows_per_s"])

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
                ]
            )
            self.assertTrue((run_dir / "detections.jsonl").exists(), msg=detect_result.stdout + detect_result.stderr)
            self.assertTrue((run_dir / "record_scores.json").exists(), msg=detect_result.stdout + detect_result.stderr)

            benchmark_result = run_command(
                [
                    sys.executable,
                    ROOT / "scripts" / "benchmark_onboard.py",
                    "--artifact",
                    run_dir / "artifact.pt",
                    "--input",
                    data_path,
                    "--batch_size",
                    "64",
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
