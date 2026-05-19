from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "app" / "streamlit_app.py"
SPEC = importlib.util.spec_from_file_location("streamlit_app_diagnostics_module", MODULE_PATH)
streamlit_app = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(streamlit_app)


class DashboardDiagnosticsTests(unittest.TestCase):
    def test_dashboard_defaults_use_artifact_calibrator_thresholds(self):
        config = streamlit_app.clone_dashboard_default_config()
        simulation_config = streamlit_app.apply_data_source_mode_to_config(
            config,
            streamlit_app.SIMULATION_MIXED_REPLAY_LABEL,
        )

        self.assertTrue(bool(config["use_offline_calibrator_for_demo"]))
        self.assertEqual(simulation_config["dataset_replay_mode"], "")
        self.assertTrue(bool(simulation_config["use_offline_calibrator_for_demo"]))

    def test_pure_ids_csv_replay_defaults_export_full_records(self):
        config = streamlit_app.clone_pure_ids_csv_replay_config()

        self.assertEqual(config["dataset_replay_mode"], "pure_ids_csv")
        self.assertTrue(bool(config["use_offline_calibrator_for_demo"]))
        self.assertFalse(bool(config["export_summary_only"]))

    def test_configure_demo_detector_for_artifact_thresholds_sets_detector_modes(self):
        detector = SimpleNamespace(
            use_artifact_calibrator_decision=False,
            score_threshold_mode="per_uav_benign_warmup_raw_quantile",
            threshold_config={"threshold_mode": "per_uav_benign_warmup_raw_quantile"},
        )

        streamlit_app.configure_demo_detector_for_artifact_thresholds(detector)

        self.assertTrue(bool(detector.use_artifact_calibrator_decision))
        self.assertEqual(detector.score_threshold_mode, "artifact_ood_calibrator")
        self.assertEqual(detector.threshold_config["threshold_mode"], "artifact_ood_calibrator")

    def test_detections_frame_uses_top_level_ground_truth_fallbacks(self):
        rows = [
            {
                "window_id": 7,
                "record_ids": ["uav_01:3.000:42"],
                "ood_score": 0.95,
                "ood_threshold": 0.55,
                "is_ood": True,
                "alert_level": "warning",
                "gt_attack_active": True,
                "ground_truth": {
                    "is_ood": True,
                    "attack_types": ["scan"],
                },
            }
        ]

        frame = streamlit_app.detections_frame(rows)

        self.assertEqual(frame.iloc[0]["group_id"], "uav_01")
        self.assertTrue(bool(frame.iloc[0]["attack_active"]))
        self.assertTrue(bool(frame.iloc[0]["ground_truth_is_ood"]))
        self.assertTrue(bool(frame.iloc[0]["ood_alert"]))
        self.assertFalse(bool(frame.iloc[0]["known_attack_alert"]))
        self.assertEqual(frame.iloc[0]["alert_reason"], "ood")

    def test_detections_frame_preserves_detector_alert_flags(self):
        rows = [
            {
                "window_id": 3,
                "record_ids": ["uav_01:3.000:42"],
                "raw_ood_score": 12.0,
                "normalized_ood_score": 0.42,
                "threshold": 0.55,
                "is_ood": False,
                "alert_level": "warning",
                "has_alert": False,
                "false_alert": False,
                "gt_attack_active": False,
                "ground_truth_is_ood": False,
            }
        ]

        frame = streamlit_app.detections_frame(rows)

        self.assertFalse(bool(frame.iloc[0]["has_alert"]))
        self.assertFalse(bool(frame.iloc[0]["false_alert"]))
        self.assertAlmostEqual(float(frame.iloc[0]["raw_ood_score"]), 12.0)
        self.assertAlmostEqual(float(frame.iloc[0]["normalized_ood_score"]), 0.42)
        self.assertAlmostEqual(float(frame.iloc[0]["threshold"]), 0.55)

    def test_detections_frame_preserves_alert_channel_fields(self):
        rows = [
            {
                "window_id": 9,
                "record_ids": ["uav_01:9.000:52"],
                "ood_score": 0.51,
                "threshold": 0.50,
                "is_ood": False,
                "ood_alert": False,
                "known_attack_alert": True,
                "known_attack_pred_labels": ["recon_scanning", "injection"],
                "alert_reason": "known_attack",
                "alert_level": "warning",
                "has_alert": True,
                "false_alert": False,
                "gt_attack_active": True,
                "ground_truth_is_ood": False,
            }
        ]

        frame = streamlit_app.detections_frame(rows)

        self.assertTrue(bool(frame.iloc[0]["known_attack_alert"]))
        self.assertFalse(bool(frame.iloc[0]["ood_alert"]))
        self.assertEqual(frame.iloc[0]["known_attack_pred_labels"], "recon_scanning, injection")
        self.assertEqual(frame.iloc[0]["alert_reason"], "known_attack")

    def test_save_streamlit_payload_json_writes_latest_and_timestamped_files(self):
        payload = {
            "config": {"uav_count": 2, "artifact_path": "runs/example/artifact.pt"},
            "summary": {"alert_count": 3, "mission_success": True},
            "time_points": [0.0, 1.0],
            "records": pd.DataFrame([{"timestamp": 0.0, "uav_id": "uav_01", "attack_active": False}]),
            "detections": pd.DataFrame(
                [
                    {
                        "simulation_time_s": 1.0,
                        "group_id": "uav_01",
                        "has_alert": True,
                        "known_attack_alert": True,
                        "ood_alert": False,
                        "alert_reason": "known_attack",
                        "known_attack_pred_labels": "recon_scanning",
                    }
                ]
            ),
            "responses": pd.DataFrame([{"uav_id": "uav_01", "response_action": "conservative_mode"}]),
            "attack_schedule": pd.DataFrame([{"uav_id": "uav_01", "attack_type": "scan", "start_s": 1.0, "end_s": 5.0}]),
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            export_paths = streamlit_app.save_streamlit_payload_json(payload, output_dir=Path(tmp_dir))

            snapshot_path = Path(export_paths["dashboard_json"])
            latest_path = Path(export_paths["dashboard_json_latest"])

            self.assertTrue(snapshot_path.exists())
            self.assertTrue(latest_path.exists())
            self.assertEqual(latest_path.name, streamlit_app.STREAMLIT_EXPORT_LATEST_FILENAME)

            saved = json.loads(latest_path.read_text(encoding="utf-8"))

        self.assertEqual(saved["export_source"], "streamlit_dashboard")
        self.assertEqual(saved["summary"]["alert_count"], 3)
        self.assertEqual(saved["detections"][0]["alert_reason"], "known_attack")
        self.assertEqual(saved["detections"][0]["known_attack_pred_labels"], "recon_scanning")
        self.assertEqual(saved["output_files"]["dashboard_json_latest"], str(latest_path))

    def test_save_streamlit_payload_json_omits_records_when_summary_only_enabled(self):
        payload = {
            "config": {
                "uav_count": 1,
                "artifact_path": "runs/example/artifact.pt",
                "export_summary_only": True,
            },
            "summary": {"alert_count": 1, "mission_success": True, "record_count": 17},
            "time_points": [0.0, 1.0],
            "records": pd.DataFrame(
                [
                    {"timestamp": 0.0, "uav_id": "uav_01", "attack_active": False},
                    {"timestamp": 1.0, "uav_id": "uav_01", "attack_active": True},
                ]
            ),
            "detections": pd.DataFrame([{"simulation_time_s": 1.0, "group_id": "uav_01", "has_alert": True}]),
            "responses": pd.DataFrame([{"uav_id": "uav_01", "response_action": "conservative_mode"}]),
            "attack_schedule": pd.DataFrame([{"uav_id": "uav_01", "attack_type": "scan", "start_s": 1.0, "end_s": 5.0}]),
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            export_paths = streamlit_app.save_streamlit_payload_json(payload, output_dir=Path(tmp_dir))
            latest_path = Path(export_paths["dashboard_json_latest"])
            saved = json.loads(latest_path.read_text(encoding="utf-8"))

        self.assertEqual(saved["records"], [])
        self.assertEqual(saved["time_points"], [])
        self.assertTrue(bool(saved["records_omitted"]))
        self.assertEqual(saved["config"]["artifact_path"], "runs/example/artifact.pt")
        self.assertNotIn("output_files", saved)
        self.assertEqual(saved["records_omitted_count"], 17)
        self.assertEqual(saved["summary"]["alert_count"], 1)
        self.assertEqual(saved["responses"][0]["response_action"], "conservative_mode")

    def test_build_detection_diagnostics_returns_requested_outputs(self):
        records = pd.DataFrame(
            [
                {
                    "timestamp": 0.0,
                    "uav_id": "uav_01",
                    "detection_energy_wh": 0.10,
                    "total_energy_wh": 1.00,
                },
                {
                    "timestamp": 1.0,
                    "uav_id": "uav_05",
                    "detection_energy_wh": 0.20,
                    "total_energy_wh": 2.00,
                },
            ]
        )
        records = streamlit_app.attach_uav_metadata(records)

        detections = pd.DataFrame(
            [
                {
                    "simulation_time_s": 0.0,
                    "window_id": 0,
                    "group_id": "uav_01",
                    "raw_ood_score": 2.0,
                    "normalized_ood_score": 0.20,
                    "ood_score": 0.20,
                    "threshold": 0.60,
                    "ood_threshold": 0.60,
                    "threshold_source": "warmup_quantile",
                    "is_ood": False,
                    "ood_alert": False,
                    "known_attack_alert": False,
                    "alert_reason": "none",
                    "alert_level": "normal",
                    "has_alert": False,
                    "false_alert": False,
                    "attack_active": False,
                    "ground_truth_is_ood": False,
                },
                {
                    "simulation_time_s": 1.0,
                    "window_id": 1,
                    "group_id": "uav_01",
                    "raw_ood_score": 9.0,
                    "normalized_ood_score": 0.90,
                    "ood_score": 0.90,
                    "threshold": 0.60,
                    "ood_threshold": 0.60,
                    "threshold_source": "warmup_quantile",
                    "is_ood": True,
                    "ood_alert": True,
                    "known_attack_alert": True,
                    "alert_reason": "known_attack+ood",
                    "alert_level": "warning",
                    "has_alert": True,
                    "attack_active": True,
                    "false_alert": False,
                    "ground_truth_is_ood": False,
                },
                {
                    "simulation_time_s": 2.0,
                    "window_id": 2,
                    "group_id": "uav_05",
                    "raw_ood_score": 3.0,
                    "normalized_ood_score": 0.30,
                    "ood_score": 0.30,
                    "threshold": 0.80,
                    "ood_threshold": 0.80,
                    "threshold_source": "warmup_quantile",
                    "is_ood": True,
                    "ood_alert": True,
                    "known_attack_alert": False,
                    "alert_reason": "ood",
                    "alert_level": "warning",
                    "has_alert": True,
                    "false_alert": True,
                    "attack_active": False,
                    "ground_truth_is_ood": False,
                },
                {
                    "simulation_time_s": 3.0,
                    "window_id": 3,
                    "group_id": "uav_05",
                    "raw_ood_score": 12.0,
                    "normalized_ood_score": 1.20,
                    "ood_score": 1.20,
                    "threshold": 0.80,
                    "ood_threshold": 0.80,
                    "threshold_source": "warmup_quantile",
                    "is_ood": True,
                    "ood_alert": True,
                    "known_attack_alert": True,
                    "alert_reason": "known_attack+ood",
                    "alert_level": "critical",
                    "has_alert": True,
                    "false_alert": False,
                    "attack_active": True,
                    "ground_truth_is_ood": True,
                },
            ]
        )

        detector = SimpleNamespace(
            threshold_config={
                "global_ood_threshold": 0.95,
                "group_ood_thresholds": {
                    "uav_01": 0.55,
                    "uav_05": 0.75,
                },
            },
            simulation_diagnostics=lambda: {
                "model_input_columns": ["feat_a", "feat_b"],
                "score_mode": "normalized",
                "threshold_mode": "per_uav_benign_warmup_raw_quantile",
                "score_direction": "higher_is_more_anomalous",
                "normalized_threshold": 1.5,
                "per_uav_threshold": {"uav_01": 0.60, "uav_05": 0.80},
                "per_uav_raw_threshold": {"uav_01": 2.5, "uav_05": 3.5},
                "per_uav_normalized_threshold": {"uav_01": 1.5, "uav_05": 1.5},
                "per_uav_threshold_source": {"uav_01": "warmup_quantile", "uav_05": "warmup_quantile"},
                "per_uav_benign_score_q50": {"uav_01": 2.0, "uav_05": 3.0},
                "per_uav_benign_score_q90": {"uav_01": 2.2, "uav_05": 3.2},
                "per_uav_benign_score_q95": {"uav_01": 2.3, "uav_05": 3.3},
                "per_uav_benign_score_q99": {"uav_01": 2.4, "uav_05": 3.4},
                "per_uav_raw_score_q50": {"uav_01": 5.5, "uav_05": 7.5},
                "per_uav_raw_score_q90": {"uav_01": 8.3, "uav_05": 11.1},
                "per_uav_raw_score_q95": {"uav_01": 8.7, "uav_05": 11.6},
                "per_uav_raw_score_q99": {"uav_01": 8.9, "uav_05": 11.9},
                "per_uav_normalized_score_q50": {"uav_01": 0.55, "uav_05": 0.75},
                "per_uav_normalized_score_q90": {"uav_01": 0.83, "uav_05": 1.11},
                "per_uav_normalized_score_q95": {"uav_01": 0.87, "uav_05": 1.16},
                "per_uav_normalized_score_q99": {"uav_01": 0.89, "uav_05": 1.19},
            },
            ood_cal=SimpleNamespace(
                direction_label_source="bootstrap_benign_only",
                directions={"proto": 1.0},
                direction_report=[
                    {
                        "score_name": "proto",
                        "raw_auroc": None,
                        "flipped": False,
                        "effective_auroc": None,
                        "direction": 1.0,
                    }
                ],
            ),
        )

        summary = streamlit_app.build_detection_diagnostics(records, detections, detector=detector)

        self.assertEqual(summary["model_input_columns"], ["feat_a", "feat_b"])
        self.assertEqual(summary["score_mode"], "normalized")
        self.assertEqual(summary["threshold_mode"], "per_uav_benign_warmup_raw_quantile")
        self.assertEqual(summary["score_direction"], "higher_is_more_anomalous")
        self.assertEqual(summary["normalized_threshold"], 1.5)
        self.assertEqual(summary["per_uav_threshold"]["uav_01"], 0.60)
        self.assertEqual(summary["per_uav_threshold"]["uav_05"], 0.80)
        self.assertEqual(summary["per_uav_raw_threshold"]["uav_01"], 2.5)
        self.assertEqual(summary["per_uav_normalized_threshold"]["uav_05"], 1.5)
        self.assertEqual(summary["per_uav_threshold_source"]["uav_01"], "warmup_quantile")
        self.assertAlmostEqual(summary["per_uav_id_score_mean"]["uav_01"], 0.20)
        self.assertIsNone(summary["per_uav_ood_score_mean"]["uav_01"])
        self.assertAlmostEqual(summary["per_uav_id_score_mean"]["uav_05"], 0.30)
        self.assertAlmostEqual(summary["per_uav_ood_score_mean"]["uav_05"], 1.20)
        self.assertEqual(summary["per_uav_benign_window_count"]["uav_01"], 1)
        self.assertEqual(summary["per_uav_gt_attack_window_count"]["uav_01"], 1)
        self.assertEqual(summary["per_uav_gt_ood_window_count"]["uav_01"], 0)
        self.assertEqual(summary["per_uav_pred_alert_window_count"]["uav_01"], 1)
        self.assertEqual(summary["per_uav_pred_ood_window_count"]["uav_01"], 1)
        self.assertEqual(summary["per_uav_benign_window_count"]["uav_05"], 1)
        self.assertEqual(summary["per_uav_gt_attack_window_count"]["uav_05"], 0)
        self.assertEqual(summary["per_uav_gt_ood_window_count"]["uav_05"], 1)
        self.assertEqual(summary["per_uav_pred_alert_window_count"]["uav_05"], 2)
        self.assertEqual(summary["per_uav_pred_ood_window_count"]["uav_05"], 2)
        self.assertAlmostEqual(summary["per_uav_raw_ood_score_mean"]["uav_01"], 5.5)
        self.assertAlmostEqual(summary["per_uav_normalized_ood_score_mean"]["uav_05"], 0.75)
        self.assertAlmostEqual(summary["per_uav_alert_rate"]["uav_01"], 0.50)
        self.assertAlmostEqual(summary["per_uav_false_alert_rate_on_benign_windows"]["uav_01"], 0.00)
        self.assertAlmostEqual(summary["per_uav_alert_rate"]["uav_05"], 1.00)
        self.assertAlmostEqual(summary["per_uav_false_alert_rate_on_benign_windows"]["uav_05"], 1.00)
        self.assertAlmostEqual(summary["per_uav_score_separation"]["uav_01"], 0.70)
        self.assertFalse(bool(summary["per_uav_direction_warning"]["uav_01"]))
        self.assertAlmostEqual(summary["per_uav_reversed_score_separation"]["uav_01"], -0.70)
        self.assertAlmostEqual(summary["per_uav_raw_score_q50"]["uav_01"], 5.5)
        self.assertAlmostEqual(summary["per_uav_normalized_score_q99"]["uav_05"], 1.19)
        self.assertAlmostEqual(summary["per_uav_benign_score_q95"]["uav_01"], 2.3)
        self.assertEqual(summary["gt_attack_window_count"], 1)
        self.assertEqual(summary["gt_ood_window_count"], 1)
        self.assertEqual(summary["benign_window_count"], 2)
        self.assertEqual(summary["pred_alert_window_count"], 3)
        self.assertEqual(summary["pred_known_attack_alert_window_count"], 2)
        self.assertEqual(summary["pred_ood_window_count"], 3)
        self.assertEqual(summary["pred_dual_alert_window_count"], 2)
        self.assertAlmostEqual(summary["ids_energy_wh"], 0.30)
        self.assertAlmostEqual(summary["ids_energy_ratio"], 0.10)
        self.assertAlmostEqual(summary["uav_only_false_alert_rate"], 0.0)
        self.assertAlmostEqual(summary["external_non_uav_false_alert_rate"], 1.0)

        direction = summary["score_direction_diagnostics"]
        self.assertEqual(direction["configured_label_source"], "bootstrap_benign_only")
        self.assertFalse(bool(direction["fused_ood_score_flip_suspected"]))
        self.assertAlmostEqual(float(direction["fused_ood_score_raw_auroc"]), 1.0)


if __name__ == "__main__":
    unittest.main()
