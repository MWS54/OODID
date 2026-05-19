from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "app" / "streamlit_app.py"
SPEC = importlib.util.spec_from_file_location("streamlit_app_module", MODULE_PATH)
streamlit_app = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(streamlit_app)


class DashboardAnalysisTests(unittest.TestCase):
    def test_build_runtime_scenario_config_uses_fixed_dataset_bindings(self):
        config = streamlit_app.clone_default_config()
        config["uav_count"] = 3
        config["attack_mode"] = "mixed_attack"
        config["attack_replay_mode"] = "loop"
        config["per_uav_attack_plans"] = {
            "uav_01": {
                "attack_types": ["flood"],
                "attack_start_s": 10.0,
                "attack_duration_s": 8.0,
                "attack_intensity": 0.8,
            },
            "uav_02": {
                "attack_types": ["replay"],
                "attack_start_s": 12.0,
                "attack_duration_s": 6.0,
                "attack_intensity": 0.7,
            },
            "uav_03": {
                "attack_types": ["command_injection"],
                "attack_start_s": 14.0,
                "attack_duration_s": 5.0,
                "attack_intensity": 0.9,
            },
        }

        scenario = streamlit_app.build_runtime_scenario_config(config)

        self.assertEqual(scenario.scenario_type, "mixed_attack")
        self.assertEqual([event.uav_id for event in scenario.attack_events], ["uav_01", "uav_02", "uav_03"])
        self.assertEqual(
            [event.source_dataset for event in scenario.attack_events],
            ["uav_ndd", "gcs_to_uav_updated", "isot_drone"],
        )
        self.assertEqual(
            [event.attack_types for event in scenario.attack_events],
            [("flood",), ("replay",), ("command_injection",)],
        )
        self.assertTrue(all(event.replay_mode == "loop" for event in scenario.attack_events))

    def test_build_uavs_skips_uav04_for_extended_bindings(self):
        config = streamlit_app.clone_default_config()
        config["uav_count"] = 4

        uavs = streamlit_app.build_uavs(config)

        self.assertEqual([uav.uav_id for uav in uavs], ["uav_01", "uav_02", "uav_03", "uav_05"])

    def test_dataset_display_marks_unsw_as_external_non_uav(self):
        self.assertEqual(streamlit_app.dataset_display_for_uav("uav_05"), "UNSW-NB15 [external_non_uav]")

    def test_runtime_scenario_uses_unsw_after_uav03_without_uav04(self):
        config = streamlit_app.clone_default_config()
        config["uav_count"] = 4
        config["attack_mode"] = "single_attack"
        config["attack_replay_mode"] = "sequential"
        config["per_uav_attack_plans"]["uav_05"] = {
            "attack_types": ["scan"],
            "attack_start_s": 40.0,
            "attack_duration_s": 8.0,
            "attack_intensity": 0.8,
        }

        scenario = streamlit_app.build_runtime_scenario_config(config)

        self.assertEqual([event.uav_id for event in scenario.attack_events], ["uav_01", "uav_02", "uav_03", "uav_05"])
        self.assertEqual([event.source_dataset for event in scenario.attack_events][-1], "unsw_nb15")

    def test_runtime_scenario_respects_explicit_selected_uavs_without_contiguous_indices(self):
        config = streamlit_app.clone_default_config()
        config["selected_uav_ids"] = ["uav_05", "uav_07"]
        config["uav_count"] = 2
        config["attack_mode"] = "single_attack"
        config["attack_replay_mode"] = "sequential"

        scenario = streamlit_app.build_runtime_scenario_config(config)
        uavs = streamlit_app.build_uavs(config)

        self.assertEqual([uav.uav_id for uav in uavs], ["uav_05", "uav_07"])
        self.assertEqual([event.uav_id for event in scenario.attack_events], ["uav_05", "uav_07"])
        self.assertEqual([event.source_dataset for event in scenario.attack_events], ["unsw_nb15", "uavids"])

    def test_fleet_status_frame_reports_latest_per_uav_status(self):
        records = pd.DataFrame(
            [
                {
                    "uav_id": "uav_01",
                    "timestamp": 1.0,
                    "mission_phase": "cruise",
                    "battery_soc": 91.0,
                    "rssi": -58.0,
                    "latency_ms": 17.0,
                    "attack_active": True,
                    "attack_type": "flood",
                },
                {
                    "uav_id": "uav_02",
                    "timestamp": 1.0,
                    "mission_phase": "hover",
                    "battery_soc": 87.5,
                    "rssi": -63.0,
                    "latency_ms": 23.0,
                    "attack_active": False,
                    "attack_type": "benign",
                },
            ]
        )
        detections = pd.DataFrame(
            [
                {
                    "group_id": "uav_01",
                    "simulation_time_s": 1.0,
                    "ood_score": 1.2,
                    "alert_level": "warning",
                    "known_attack_alert": True,
                    "ood_alert": False,
                    "alert_reason": "known_attack",
                },
                {
                    "group_id": "uav_02",
                    "simulation_time_s": 1.0,
                    "ood_score": 0.2,
                    "alert_level": "normal",
                    "known_attack_alert": False,
                    "ood_alert": False,
                    "alert_reason": "none",
                },
            ]
        )

        status = streamlit_app.fleet_status_frame(records, detections, current_time_s=1.0)
        by_uav = {row["uav_id"]: row for row in status.to_dict("records")}

        self.assertEqual(
            list(status.columns),
            [
                "uav_id",
                "mission_phase",
                "battery_soc",
                "attack_active",
                "attack_type",
                "ood_score",
                "alert_level",
                "alert_reason",
                "known_attack_alert",
                "ood_alert",
            ],
        )
        self.assertEqual(by_uav["uav_01"]["mission_phase"], "cruise")
        self.assertTrue(by_uav["uav_01"]["attack_active"])
        self.assertEqual(by_uav["uav_01"]["alert_level"], "warning")
        self.assertEqual(by_uav["uav_01"]["alert_reason"], "known_attack")
        self.assertTrue(by_uav["uav_01"]["known_attack_alert"])
        self.assertFalse(by_uav["uav_01"]["ood_alert"])
        self.assertFalse(by_uav["uav_02"]["attack_active"])
        self.assertAlmostEqual(by_uav["uav_02"]["ood_score"], 0.2)

    def test_detections_with_attack_scene_classifies_single_and_mixed_windows(self):
        records = pd.DataFrame(
            [
                {"uav_id": "uav_01", "timestamp": 1.0, "attack_active": True, "attack_type": "flood"},
                {"uav_id": "uav_02", "timestamp": 1.0, "attack_active": False, "attack_type": "benign"},
                {"uav_id": "uav_01", "timestamp": 2.0, "attack_active": True, "attack_type": "flood"},
                {"uav_id": "uav_02", "timestamp": 2.0, "attack_active": True, "attack_type": "replay"},
            ]
        )
        detections = pd.DataFrame(
            [
                {
                    "group_id": "uav_01",
                    "simulation_time_s": 1.0,
                    "attack_active": True,
                    "has_alert": True,
                    "ood_score": 1.1,
                    "alert_level": "warning",
                },
                {
                    "group_id": "uav_01",
                    "simulation_time_s": 2.0,
                    "attack_active": True,
                    "has_alert": True,
                    "ood_score": 1.3,
                    "alert_level": "critical",
                },
                {
                    "group_id": "uav_02",
                    "simulation_time_s": 2.0,
                    "attack_active": True,
                    "has_alert": False,
                    "ood_score": 0.9,
                    "alert_level": "normal",
                },
            ]
        )

        classified = streamlit_app.detections_with_attack_scene(records, detections)
        summary = streamlit_app.detection_mode_summary(records, detections)
        by_time = {(row["group_id"], row["simulation_time_s"]): row for row in classified.to_dict("records")}
        by_mode = {row["attack_mode"]: row for row in summary.to_dict("records")}

        self.assertEqual(by_time[("uav_01", 1.0)]["attack_scene_mode"], "single_attack")
        self.assertEqual(by_time[("uav_01", 2.0)]["attack_scene_mode"], "mixed_attack")
        self.assertEqual(by_time[("uav_02", 2.0)]["attack_scene_mode"], "mixed_attack")
        self.assertEqual(by_mode["single_attack"]["attack_windows"], 1)
        self.assertEqual(by_mode["mixed_attack"]["attack_windows"], 2)

    def test_detections_with_phase_handles_interleaved_multi_uav_timestamps(self):
        records = pd.DataFrame(
            [
                {"uav_id": "uav_01", "timestamp": 0.0, "mission_phase": "idle", "mission_context": "surveillance"},
                {"uav_id": "uav_01", "timestamp": 1.0, "mission_phase": "cruise", "mission_context": "surveillance"},
                {"uav_id": "uav_02", "timestamp": 0.0, "mission_phase": "takeoff", "mission_context": "delivery"},
                {"uav_id": "uav_02", "timestamp": 1.0, "mission_phase": "hover", "mission_context": "delivery"},
            ]
        )
        detections = pd.DataFrame(
            [
                {"group_id": "uav_01", "simulation_time_s": 0.0, "has_alert": True, "attack_active": False},
                {"group_id": "uav_01", "simulation_time_s": 1.0, "has_alert": False, "attack_active": False},
                {"group_id": "uav_02", "simulation_time_s": 0.0, "has_alert": True, "attack_active": False},
                {"group_id": "uav_02", "simulation_time_s": 1.0, "has_alert": False, "attack_active": True},
            ]
        )

        merged = streamlit_app.detections_with_phase(records, detections)

        self.assertEqual(list(merged["simulation_time_s"]), [0.0, 0.0, 1.0, 1.0])
        self.assertEqual(list(merged["uav_id"]), ["uav_01", "uav_02", "uav_01", "uav_02"])
        self.assertEqual(
            list(merged["mission_phase"].astype(str)),
            ["idle", "takeoff", "cruise", "hover"],
        )
        self.assertEqual(
            list(merged["mission_context"]),
            ["surveillance", "delivery", "surveillance", "delivery"],
        )

    def test_phase_false_positive_rates_uses_phase_mapping_without_sort_error(self):
        records = pd.DataFrame(
            [
                {"uav_id": "uav_01", "timestamp": 0.0, "mission_phase": "idle", "mission_context": "surveillance"},
                {"uav_id": "uav_01", "timestamp": 1.0, "mission_phase": "cruise", "mission_context": "surveillance"},
                {"uav_id": "uav_02", "timestamp": 0.0, "mission_phase": "takeoff", "mission_context": "delivery"},
                {"uav_id": "uav_02", "timestamp": 1.0, "mission_phase": "hover", "mission_context": "delivery"},
            ]
        )
        detections = pd.DataFrame(
            [
                {"group_id": "uav_01", "simulation_time_s": 0.0, "has_alert": True, "attack_active": False},
                {"group_id": "uav_01", "simulation_time_s": 1.0, "has_alert": False, "attack_active": False},
                {"group_id": "uav_02", "simulation_time_s": 0.0, "has_alert": True, "attack_active": False},
                {"group_id": "uav_02", "simulation_time_s": 1.0, "has_alert": True, "attack_active": True},
            ]
        )

        rates = streamlit_app.phase_false_positive_rates(records, detections)
        by_phase = {row["mission_phase"]: row for row in rates.to_dict("records")}

        self.assertEqual(by_phase["idle"]["false_positive_windows"], 1)
        self.assertEqual(by_phase["idle"]["benign_windows"], 1)
        self.assertEqual(by_phase["idle"]["false_positive_rate"], 1.0)
        self.assertEqual(by_phase["takeoff"]["false_positive_windows"], 1)
        self.assertEqual(by_phase["takeoff"]["benign_windows"], 1)
        self.assertEqual(by_phase["cruise"]["false_positive_rate"], 0.0)
        self.assertEqual(by_phase["hover"]["benign_windows"], 0)

    def test_analysis_metric_frames_include_dataset_source_type_and_external_ood_views(self):
        records = pd.DataFrame(
            [
                {
                    "uav_id": "uav_05",
                    "timestamp": 0.0,
                    "battery_soc": 92.0,
                    "attack_active": False,
                    "total_energy_wh": 1.1,
                },
                {
                    "uav_id": "uav_05",
                    "timestamp": 1.0,
                    "battery_soc": 89.0,
                    "attack_active": True,
                    "total_energy_wh": 1.4,
                },
                {
                    "uav_id": "uav_06",
                    "timestamp": 0.0,
                    "battery_soc": 95.0,
                    "attack_active": False,
                    "total_energy_wh": 0.9,
                },
                {
                    "uav_id": "uav_06",
                    "timestamp": 1.0,
                    "battery_soc": 91.0,
                    "attack_active": True,
                    "total_energy_wh": 1.2,
                },
            ]
        )
        detections = pd.DataFrame(
            [
                {
                    "group_id": "uav_05",
                    "simulation_time_s": 1.0,
                    "is_ood": True,
                    "has_alert": True,
                    "attack_active": True,
                    "ood_score": 1.3,
                    "alert_level": "critical",
                },
                {
                    "group_id": "uav_06",
                    "simulation_time_s": 1.0,
                    "is_ood": True,
                    "has_alert": False,
                    "attack_active": True,
                    "ood_score": 0.8,
                    "alert_level": "normal",
                },
            ]
        )

        per_dataset = streamlit_app.per_dataset_metrics_frame(records, detections)
        per_source_type = streamlit_app.per_source_type_metrics_frame(records, detections)
        external = streamlit_app.external_ood_metrics_frame(records, detections)

        by_dataset = {row["dataset_name"]: row for row in per_dataset.to_dict("records")}
        by_source_type = {row["source_type"]: row for row in per_source_type.to_dict("records")}

        self.assertEqual(by_dataset["unsw_nb15"]["source_type"], "external_non_uav")
        self.assertEqual(by_dataset["unsw_nb15"]["simulation_role"], "external_ood")
        self.assertEqual(by_dataset["unsw_nb15"]["detected_attack_windows"], 1)
        self.assertEqual(by_dataset["ecu_ioft"]["source_type"], "uav_iot_wifi")
        self.assertEqual(by_source_type["external_non_uav"]["uav_count"], 1)
        self.assertEqual(by_source_type["uav_iot_wifi"]["dataset_count"], 1)
        self.assertEqual(len(external), 1)
        self.assertEqual(external.iloc[0]["dataset_name"], "unsw_nb15")
        self.assertIn("External OOD", external.iloc[0]["source_note"])


if __name__ == "__main__":
    unittest.main()
