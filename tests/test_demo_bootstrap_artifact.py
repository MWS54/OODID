from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from ucs_oodid.ood import OODCalibrator

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "simulate_live_demo.py"
SPEC = importlib.util.spec_from_file_location("simulate_live_demo_script", MODULE_PATH)
simulate_live_demo = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(simulate_live_demo)


class DemoBootstrapArtifactTests(unittest.TestCase):
    def test_build_bootstrap_preprocessor_excludes_reserved_simulator_fields(self):
        df = pd.DataFrame(
            [
                {
                    "record_id": "uav_01:0",
                    "timestamp": 0.0,
                    "uav_id": "uav_01",
                    "label": "benign",
                    "mission_phase": "cruise",
                    "battery_soc": 99.0,
                    "speed": 18.0,
                    "altitude": 90.0,
                    "rssi": -52.0,
                    "snr": 21.0,
                    "latency_ms": 15.0,
                    "loss_rate": 0.01,
                    "dataset_name": "uav_ndd",
                    "source_type": "uav",
                    "simulation_role": "uav_replay",
                    "source_missing": 0,
                    "source_is_ipv4": 1,
                    "source_has_mac_like": 0,
                    "feat_a": 1.0,
                    "feat_b": 2.0,
                },
                {
                    "record_id": "uav_02:0",
                    "timestamp": 1.0,
                    "uav_id": "uav_02",
                    "label": "benign",
                    "mission_phase": "hover",
                    "battery_soc": 98.5,
                    "speed": 12.0,
                    "altitude": 70.0,
                    "rssi": -61.0,
                    "snr": 17.0,
                    "latency_ms": 18.0,
                    "loss_rate": 0.02,
                    "dataset_name": "unsw_nb15",
                    "source_type": "external_non_uav",
                    "simulation_role": "external_ood",
                    "source_missing": 1,
                    "source_is_ipv4": 0,
                    "source_has_mac_like": 1,
                    "feat_a": 1.5,
                    "feat_b": 2.5,
                },
            ]
        )

        pre = simulate_live_demo.build_bootstrap_preprocessor(df)

        self.assertEqual(pre.feature_cols, ["feat_a", "feat_b"])

    def test_apply_bootstrap_thresholds_creates_group_thresholds(self):
        raw_scores = {
            "conf": np.asarray([0.05, 0.10, 0.12, 0.15], dtype=np.float32),
            "energy": np.asarray([0.02, 0.04, 0.08, 0.09], dtype=np.float32),
            "proto": np.asarray([0.10, 0.20, 0.30, 0.40], dtype=np.float32),
            "knn": np.asarray([0.01, 0.03, 0.05, 0.07], dtype=np.float32),
        }
        group_ids = np.asarray(["uav_01", "uav_01", "uav_05", "uav_05"], dtype=object)
        ood_cal = OODCalibrator(fusion="proto", q_ood=0.8, ood_threshold_mode="group")

        simulate_live_demo.apply_bootstrap_thresholds(
            ood_cal,
            raw_scores,
            group_ids,
            margin=0.05,
        )

        transformed = ood_cal.transform(raw_scores, groups=group_ids)

        self.assertEqual(ood_cal.ood_threshold_mode, "group")
        self.assertEqual(ood_cal.direction_label_source, "bootstrap_benign_only")
        self.assertEqual(set(ood_cal.group_thresholds), {"uav_01", "uav_05"})
        self.assertTrue(all(source for source in ood_cal.group_threshold_sources.values()))

        for group_id in ("uav_01", "uav_05"):
            mask = group_ids == group_id
            self.assertGreaterEqual(
                float(ood_cal.group_thresholds[group_id]),
                float(np.max(transformed["fused"][mask])),
            )

    def test_main_uses_uav_grouping_when_loading_artifact(self):
        args = SimpleNamespace(
            online_detection=True,
            artifact="artifact.pt",
            top_records=7,
            window_size=6,
            stride=1,
            head=0,
            output_json="",
        )

        class _StubResult:
            records = []
            online_detection_results = []
            attack_record_count = 0
            total_energy_wh = 0.0
            alert_count = 0
            response_count = 0
            average_alert_delay = 0.0
            false_alert_count = 0
            mission_success = True
            response_events = []

        class _StubEngine:
            def run(self):
                return _StubResult()

        fake_detector = SimpleNamespace(
            buffer=None,
            group_col=None,
            pre=SimpleNamespace(group_col=None),
            window_config={},
        )
        with (
            patch.object(simulate_live_demo, "parse_args", return_value=args),
            patch.object(
                simulate_live_demo.OnlineDetector,
                "from_artifact_path",
                return_value=fake_detector,
            ) as detector_factory,
            patch.object(simulate_live_demo, "build_demo_engine", return_value=_StubEngine()),
            patch.object(simulate_live_demo, "compact_detection_trace", return_value=[]),
            patch.object(simulate_live_demo, "effective_attack_start_s", return_value=0.0),
            patch.object(simulate_live_demo, "summarize_trace", return_value={}),
            patch("builtins.print"),
        ):
            simulate_live_demo.main()

        detector_factory.assert_called_once()
        call = detector_factory.call_args
        self.assertEqual(call.args, ("artifact.pt",))
        self.assertEqual(call.kwargs["top_records"], 7)
        self.assertEqual(call.kwargs["group_col"], "uav_id")
        self.assertNotIn("buffer", call.kwargs)
        self.assertEqual(fake_detector.group_col, "uav_id")
        self.assertEqual(fake_detector.pre.group_col, "uav_id")
        self.assertEqual(fake_detector.buffer.group_col, "uav_id")
        self.assertEqual(fake_detector.buffer.window_size, 6)
        self.assertEqual(fake_detector.buffer.stride, 1)


if __name__ == "__main__":
    unittest.main()
