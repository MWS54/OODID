from __future__ import annotations

import importlib.util
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from ucs_oodid.comparison import (
    evaluate_tabular_baseline,
    get_method_config,
    prepare_comparison_dataset,
)
from ucs_oodid.io import load_records

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "run_comparison_experiments.py"
SPEC = importlib.util.spec_from_file_location("run_comparison_experiments", SCRIPT_PATH)
run_comparison_experiments = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(run_comparison_experiments)


def make_args(input_path: Path, output_dir: Path, methods: str = "") -> SimpleNamespace:
    return SimpleNamespace(
        input=str(input_path),
        output_dir=str(output_dir),
        methods=methods,
        id_classes="benign,dos",
        ood_classes="evil",
        label_col="label",
        timestamp_col="timestamp",
        record_id_col="record_id",
        group_col="uav_id",
        benign_label="benign",
        allow_ports=False,
        window_mode="count",
        window_size=2,
        stride=1,
        time_window_seconds=2.0,
        adaptive_min_size=8,
        adaptive_max_size=64,
        graph_k=8,
        graph_tau=0.5,
        graph_metric="cosine",
        graph_variant="sym_weighted",
        hidden_dim=32,
        num_heads=2,
        num_layers=1,
        gcn_layers=1,
        dropout=0.1,
        gate="learned",
        record_head=False,
        epochs=1,
        batch_size=8,
        lr=1e-3,
        weight_decay=1e-4,
        lambda_record=0.0,
        q_ood=0.9,
        bank_k=3,
        fusion="correlation_aware",
        ood_direction_calibration="none",
        phase_aware_threshold=False,
        phase_column="mission_phase_proxy",
        phase_threshold_min_samples=32,
        phase_threshold_quantile=None,
        phase_threshold_fallback="global",
        normalization_mode="group",
        ood_threshold_mode="group",
        group_threshold_strategy="conservative",
        group_threshold_shrink_k=1000.0,
        group_threshold_min_ratio=1.0,
        group_threshold_min_samples=1,
        present_class_min_support=1,
        group_embedding_dim=16,
        seed=42,
        device="auto",
        use_group_embedding=False,
    )


def write_toy_dataset(path: Path) -> None:
    rows = []
    for idx in range(20):
        rows.append(
            {
                "record_id": f"id_{idx}",
                "timestamp": idx,
                "uav_id": "uav_01",
                "label": "dos" if idx % 4 == 0 else "benign",
                "feat_a": float(idx),
                "feat_b": float(idx % 5),
            }
        )
    for idx in range(10):
        rows.append(
            {
                "record_id": f"ood_{idx}",
                "timestamp": 20 + idx,
                "uav_id": "uav_01",
                "label": "evil",
                "feat_a": float(100 + idx),
                "feat_b": float((idx + 1) % 5),
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


class ComparisonPipelineTests(unittest.TestCase):
    def test_build_commands_apply_fairness_protocol_by_method(self):
        args = make_args(Path("toy.csv"), Path("runs"))

        rapier_cmd = [str(part) for part in run_comparison_experiments.build_sklearn_command(args, run_comparison_experiments.METHOD_BY_NAME["rapier_proxy"], Path("runs"))]
        self.assertEqual(rapier_cmd[rapier_cmd.index("--baseline") + 1], "random_forest")
        self.assertEqual(rapier_cmd[rapier_cmd.index("--method_name") + 1], "rapier_proxy")
        self.assertEqual(rapier_cmd[rapier_cmd.index("--normalization_mode") + 1], "global")

        recda_cmd = [str(part) for part in run_comparison_experiments.build_neural_command(args, run_comparison_experiments.METHOD_BY_NAME["recda_proxy"], Path("runs"))]
        self.assertEqual(recda_cmd[recda_cmd.index("--normalization_mode") + 1], "global")
        self.assertEqual(recda_cmd[recda_cmd.index("--ood_threshold_mode") + 1], "global")
        self.assertEqual(recda_cmd[recda_cmd.index("--group_threshold_strategy") + 1], "raw")
        self.assertNotIn("--use_group_embedding", recda_cmd)

        ucs_cmd = [str(part) for part in run_comparison_experiments.build_neural_command(args, run_comparison_experiments.METHOD_BY_NAME["ucs_oodid"], Path("runs"))]
        self.assertEqual(ucs_cmd[ucs_cmd.index("--normalization_mode") + 1], "group")
        self.assertEqual(ucs_cmd[ucs_cmd.index("--ood_threshold_mode") + 1], "group")
        self.assertEqual(ucs_cmd[ucs_cmd.index("--group_threshold_strategy") + 1], "conservative")
        self.assertEqual(ucs_cmd[ucs_cmd.index("--group_threshold_min_ratio") + 1], "1.0")
        self.assertIn("--use_group_embedding", ucs_cmd)

    def test_prepare_comparison_dataset_splits_records_before_windowing(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_path = tmp_path / "toy.csv"
            output_dir = tmp_path / "runs"
            write_toy_dataset(input_path)

            bundle = prepare_comparison_dataset(make_args(input_path, output_dir), output_dir)

            self.assertEqual(bundle.split_source, "chronological_fallback")
            prepared = load_records(bundle.prepared_input_path)
            self.assertIn("split", prepared.columns)
            self.assertEqual(set(prepared["split"].unique()), {"train", "val", "test_id", "test_ood"})

            train_ids = set(bundle.split_windows["train"].record_ids.ravel().tolist())
            val_ids = set(bundle.split_windows["val"].record_ids.ravel().tolist())
            test_id_ids = set(bundle.split_windows["test_id"].record_ids.ravel().tolist())
            test_ood_ids = set(bundle.split_windows["test_ood"].record_ids.ravel().tolist())
            self.assertTrue(train_ids.isdisjoint(val_ids))
            self.assertTrue(train_ids.isdisjoint(test_id_ids))
            self.assertTrue(train_ids.isdisjoint(test_ood_ids))
            self.assertTrue(val_ids.isdisjoint(test_id_ids))
            self.assertTrue(val_ids.isdisjoint(test_ood_ids))
            self.assertTrue(test_id_ids.isdisjoint(test_ood_ids))
            self.assertGreater(len(bundle.split_windows["train"]), 0)
            self.assertGreater(len(bundle.split_windows["val"]), 0)
            self.assertGreater(len(bundle.split_windows["test_id"]), 0)
            self.assertGreater(len(bundle.split_windows["test_ood"]), 0)

    def test_run_comparison_experiments_maps_legacy_aliases_to_paper_methods(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_path = tmp_path / "toy.csv"
            output_dir = tmp_path / "runs"
            write_toy_dataset(input_path)
            args = make_args(input_path, output_dir, methods="random_forest,ucs_oodid")
            seen_commands: list[list[str]] = []

            def fake_runner(cmd):
                cmd = [str(part) for part in cmd]
                seen_commands.append(cmd)
                out_dir = Path(cmd[cmd.index("--output_dir") + 1])
                if cmd[1].endswith("train_sklearn_baselines.py"):
                    out_dir = out_dir / cmd[cmd.index("--method_name") + 1]
                out_dir.mkdir(parents=True, exist_ok=True)
                report = {
                    "status": "success",
                    "id_test": {
                        "micro_f1": 0.91,
                        "macro_f1": 0.82,
                        "mAP": 0.88,
                        "hamming_loss": 0.02,
                        "subset_accuracy": 0.75,
                    },
                    "ood_test": {
                        "auroc": 0.87,
                        "aupr_out": 0.81,
                        "fpr95": 0.12,
                        "precision": 0.70,
                        "tpr": 0.78,
                        "recall": 0.78,
                        "ood_f1": 0.74,
                        "fpr_at_threshold": 0.09,
                    },
                    "timing": {
                        "test_id_windows": 12,
                        "test_ood_windows": 6,
                        "test_windows": 18,
                        "detection_time_s": 0.036,
                        "average_detection_time_ms": 2.0,
                        "throughput_windows_per_s": 500.0,
                    },
                }
                (out_dir / "eval_report.json").write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
                return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")

            summary = run_comparison_experiments.run_comparison_experiments(
                args,
                runner=fake_runner,
            )

            self.assertEqual(summary["selected_methods"], ["rapier_proxy", "ucs_oodid"])
            self.assertEqual(summary["successful_methods"], ["rapier_proxy", "ucs_oodid"])
            self.assertEqual(summary["skipped_methods"], [])
            self.assertEqual(summary["failed_methods"], [])
            rapier_entry = next(item for item in summary["methods"] if item["method_name"] == "rapier_proxy")
            self.assertEqual(rapier_entry["display_name"], "RAPIER-style")
            self.assertEqual(rapier_entry["backend_name"], "random_forest")
            self.assertEqual(rapier_entry["heterogeneity_handling"], "global")
            ucs_entry = next(item for item in summary["methods"] if item["method_name"] == "ucs_oodid")
            self.assertEqual(ucs_entry["status"], "success")
            self.assertEqual(ucs_entry["heterogeneity_handling"], "group")
            sklearn_cmd = next(cmd for cmd in seen_commands if cmd[1].endswith("train_sklearn_baselines.py"))
            self.assertIn("random_forest", sklearn_cmd)
            self.assertIn("rapier_proxy", sklearn_cmd)
            self.assertEqual(
                sklearn_cmd[sklearn_cmd.index("--normalization_mode") + 1],
                "global",
            )
            neural_cmd = next(cmd for cmd in seen_commands if cmd[1].endswith("train.py"))
            self.assertEqual(
                neural_cmd[neural_cmd.index("--normalization_mode") + 1],
                "group",
            )
            self.assertEqual(
                neural_cmd[neural_cmd.index("--ood_threshold_mode") + 1],
                "group",
            )
            self.assertEqual(
                neural_cmd[neural_cmd.index("--group_threshold_strategy") + 1],
                "conservative",
            )
            self.assertIn("--use_group_embedding", neural_cmd)
            self.assertTrue((output_dir / "comparison_summary.json").exists())
            self.assertTrue((output_dir / "failed_methods.json").exists())
            self.assertTrue((output_dir / "known_detection_table.csv").exists())
            self.assertTrue((output_dir / "ood_detection_table.csv").exists())
            self.assertTrue((output_dir / "detection_time_table.csv").exists())
            self.assertTrue((output_dir / "baseline_config_table.csv").exists())
            self.assertTrue((output_dir / "known_detection_table.tex").exists())
            self.assertTrue((output_dir / "ood_detection_table.tex").exists())
            self.assertTrue((output_dir / "detection_time_table.tex").exists())
            self.assertTrue((output_dir / "baseline_config_table.tex").exists())
            known_table = pd.read_csv(output_dir / "known_detection_table.csv")
            self.assertEqual(
                known_table.columns.tolist(),
                ["Method", "Micro-F1", "Macro-F1", "mAP", "Hamming Loss", "Subset Acc."],
            )
            ood_table = pd.read_csv(output_dir / "ood_detection_table.csv")
            self.assertEqual(
                ood_table.columns.tolist(),
                ["Method", "AUROC", "AUPR-Out", "FPR95", "Precision", "Recall", "OOD-F1", "FPR@theta"],
            )
            time_table = pd.read_csv(output_dir / "detection_time_table.csv")
            self.assertEqual(
                time_table.columns.tolist(),
                ["Method", "Avg. Detection Time (ms/window)", "Throughput (windows/s)", "Test Windows"],
            )
            self.assertEqual(time_table["Method"].tolist(), ["RAPIER-style", "UCS-OODID"])
            self.assertEqual(time_table["Test Windows"].tolist(), [18, 18])
            baseline_config_table = pd.read_csv(output_dir / "baseline_config_table.csv")
            self.assertEqual(
                baseline_config_table.columns.tolist(),
                [
                    "Method",
                    "Input",
                    "Classifier/Encoder",
                    "OOD Score",
                    "Threshold Calibration",
                    "Heterogeneity Handling",
                    "Notes",
                ],
            )
            self.assertEqual(baseline_config_table["Method"].tolist(), ["RAPIER-style", "UCS-OODID"])
            self.assertEqual(
                baseline_config_table.loc[baseline_config_table["Method"] == "UCS-OODID", "OOD Score"].iloc[0],
                "Correlation-aware fusion",
            )
            self.assertEqual(
                baseline_config_table.loc[baseline_config_table["Method"] == "RAPIER-style", "Heterogeneity Handling"].iloc[0],
                "Global normalization + global threshold",
            )
            known_tex = (output_dir / "known_detection_table.tex").read_text(encoding="utf-8")
            self.assertIn("Known-attack multi-label detection comparison under the same metadata-window protocol.", known_tex)
            self.assertIn("The cited systems are used as representative research directions.", known_tex)
            self.assertIn(r"\textbf{UCS-OODID}", known_tex)
            self.assertNotIn("Random Forest", known_tex)
            baseline_config_tex = (output_dir / "baseline_config_table.tex").read_text(encoding="utf-8")
            self.assertIn("Baseline configuration summary under the same metadata-window protocol.", baseline_config_tex)
            self.assertIn("The reported results are obtained from input-compatible proxy implementations", baseline_config_tex)
            self.assertIn(r"\textbf{UCS-OODID}", baseline_config_tex)
            self.assertNotIn("Random Forest", baseline_config_tex)
            self.assertEqual(
                summary["paper_note"],
                "The cited systems are used as representative research directions. "
                "The reported results are obtained from input-compatible proxy implementations "
                "under the same metadata-window open-set protocol.",
            )
            self.assertEqual(
                summary["tables"]["detection_time_csv"],
                str(output_dir / "detection_time_table.csv"),
            )
            self.assertEqual(
                summary["tables"]["detection_time_tex"],
                str(output_dir / "detection_time_table.tex"),
            )

    def test_run_comparison_experiments_records_failed_methods(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_path = tmp_path / "toy.csv"
            output_dir = tmp_path / "runs"
            write_toy_dataset(input_path)
            args = make_args(input_path, output_dir, methods="transformer_gcn")

            def failing_runner(cmd):
                return subprocess.CompletedProcess([str(part) for part in cmd], 1, stdout="partial stdout", stderr="boom")

            summary = run_comparison_experiments.run_comparison_experiments(args, runner=failing_runner)

            self.assertEqual(summary["failed_methods"], ["hypervision_proxy"])
            failed = json.loads((output_dir / "failed_methods.json").read_text(encoding="utf-8"))
            self.assertEqual(len(failed), 1)
            self.assertEqual(failed[0]["method"], "hypervision_proxy")
            self.assertEqual(failed[0]["return_code"], 1)
            self.assertIn("boom", failed[0]["stderr_tail"])
            report = json.loads((output_dir / "hypervision_proxy" / "eval_report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "failed")

    def test_real_tabular_baseline_report_is_json_serializable(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_path = tmp_path / "toy.csv"
            output_dir = tmp_path / "runs"
            write_toy_dataset(input_path)
            args = make_args(input_path, output_dir)

            bundle = prepare_comparison_dataset(args, output_dir)
            report = evaluate_tabular_baseline(bundle, get_method_config("random_forest"), args)

            self.assertEqual(report["status"], "success")
            self.assertIn("id_test", report)
            self.assertIn("ood_test", report)
            json.dumps(report, ensure_ascii=False)


if __name__ == "__main__":
    unittest.main()
