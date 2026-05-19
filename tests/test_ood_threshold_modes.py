from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ucs_oodid.ood import OODCalibrator

DETECT_MODULE_PATH = ROOT / "scripts" / "detect.py"
SPEC = importlib.util.spec_from_file_location("detect_script_threshold_modes", DETECT_MODULE_PATH)
detect_script = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(detect_script)


def make_raw_scores(values):
    arr = np.asarray(values, dtype=np.float32)
    return {name: arr.copy() for name in ("conf", "energy", "proto", "knn")}


def make_manual_group_calibrator(global_threshold=10.0):
    cal = OODCalibrator(q_ood=0.5, ood_threshold_mode="group")
    cal.threshold = float(global_threshold)
    return cal


class OODThresholdModeTests(unittest.TestCase):
    def test_global_mode_keeps_global_threshold_decisions(self):
        raw = make_raw_scores([0.10, 0.20, 0.40, 0.90])
        group_ids = np.asarray(["uav_01", "uav_02", "uav_01", "uav_02"], dtype=object)

        cal = OODCalibrator(q_ood=0.75, ood_threshold_mode="global")
        cal.fit(raw)
        transformed = cal.transform(raw, groups=group_ids)

        np.testing.assert_allclose(
            transformed["thresholds"],
            np.full(len(group_ids), cal.threshold, dtype=np.float32),
        )
        self.assertEqual(transformed["threshold_sources"], ["global"] * len(group_ids))
        np.testing.assert_array_equal(transformed["decisions"], transformed["fused"] > cal.threshold)

    def test_group_mode_assigns_per_group_thresholds(self):
        raw = make_raw_scores([0.05, 0.10, 0.20, 0.80, 0.90, 1.10])
        group_ids = np.asarray(["uav_01", "uav_01", "uav_01", "uav_02", "uav_02", "uav_02"], dtype=object)

        cal = OODCalibrator(q_ood=0.5, ood_threshold_mode="group")
        cal.fit(raw)
        fused = cal.transform(raw)["fused"]
        cal.calibrate_group_thresholds(fused, group_ids, min_samples=1, quantile=0.5)
        transformed = cal.transform(raw, groups=group_ids)

        self.assertNotEqual(cal.group_thresholds["uav_01"], cal.group_thresholds["uav_02"])
        np.testing.assert_allclose(
            transformed["thresholds"][:3],
            np.full(3, cal.group_thresholds["uav_01"], dtype=np.float32),
        )
        np.testing.assert_allclose(
            transformed["thresholds"][3:],
            np.full(3, cal.group_thresholds["uav_02"], dtype=np.float32),
        )
        self.assertEqual(transformed["threshold_sources"], ["group_raw"] * len(group_ids))

    def test_group_mode_falls_back_to_global_for_small_validation_groups(self):
        raw = make_raw_scores([0.10, 0.20, 0.30, 0.80])
        group_ids = np.asarray(["uav_01", "uav_01", "uav_01", "uav_small"], dtype=object)

        cal = OODCalibrator(q_ood=0.5, ood_threshold_mode="group")
        cal.fit(raw)
        fused = cal.transform(raw)["fused"]
        global_threshold = cal.threshold
        cal.calibrate_group_thresholds(fused, group_ids, min_samples=2, quantile=0.5)
        transformed = cal.transform(raw, groups=group_ids)

        self.assertNotIn("uav_small", cal.group_thresholds)
        self.assertEqual(
            cal.group_threshold_fallbacks["uav_small"],
            "fallback_to_global_due_to_small_validation_size",
        )
        np.testing.assert_allclose(
            transformed["thresholds"][:3],
            np.full(3, cal.group_thresholds["uav_01"], dtype=np.float32),
        )
        self.assertAlmostEqual(float(transformed["thresholds"][-1]), float(global_threshold))
        self.assertEqual(transformed["threshold_sources"][-1], "global_fallback")

    def test_raw_strategy_keeps_legacy_group_quantile(self):
        fused = np.asarray([2.0, 2.0, 2.0], dtype=np.float32)
        group_ids = np.asarray(["uav_02", "uav_02", "uav_02"], dtype=object)
        cal = make_manual_group_calibrator(global_threshold=10.0)

        cal.calibrate_group_thresholds(
            fused,
            group_ids,
            min_samples=1,
            quantile=0.5,
            strategy="raw",
        )

        self.assertEqual(cal.group_raw_thresholds["uav_02"], 2.0)
        self.assertEqual(cal.group_smoothed_thresholds["uav_02"], 2.0)
        self.assertEqual(cal.group_thresholds["uav_02"], 2.0)
        self.assertEqual(cal.group_threshold_sources["uav_02"], "group_raw")

    def test_global_floor_strategy_never_goes_below_global_threshold(self):
        fused = np.asarray([2.0, 2.0, 2.0], dtype=np.float32)
        group_ids = np.asarray(["uav_02", "uav_02", "uav_02"], dtype=object)
        cal = make_manual_group_calibrator(global_threshold=10.0)

        cal.calibrate_group_thresholds(
            fused,
            group_ids,
            min_samples=1,
            quantile=0.5,
            strategy="global_floor",
        )

        self.assertEqual(cal.group_raw_thresholds["uav_02"], 2.0)
        self.assertEqual(cal.group_smoothed_thresholds["uav_02"], 2.0)
        self.assertEqual(cal.group_thresholds["uav_02"], 10.0)
        self.assertEqual(cal.group_threshold_sources["uav_02"], "group_global_floor")

    def test_conservative_strategy_applies_min_ratio_floor(self):
        fused = np.asarray([2.0, 2.0, 2.0], dtype=np.float32)
        group_ids = np.asarray(["uav_02", "uav_02", "uav_02"], dtype=object)
        cal = make_manual_group_calibrator(global_threshold=10.0)

        cal.calibrate_group_thresholds(
            fused,
            group_ids,
            min_samples=1,
            quantile=0.5,
            strategy="conservative",
            shrink_k=1.0,
            min_ratio=0.8,
        )

        self.assertEqual(cal.group_raw_thresholds["uav_02"], 2.0)
        self.assertLess(cal.group_smoothed_thresholds["uav_02"], 8.0)
        self.assertGreaterEqual(cal.group_thresholds["uav_02"], 8.0)
        self.assertEqual(cal.group_threshold_sources["uav_02"], "group_conservative_floor")

    def test_conservative_strategy_with_unit_min_ratio_uses_global_floor(self):
        fused = np.asarray([2.0, 2.0, 2.0], dtype=np.float32)
        group_ids = np.asarray(["uav_02", "uav_02", "uav_02"], dtype=object)
        cal = make_manual_group_calibrator(global_threshold=10.0)

        cal.calibrate_group_thresholds(
            fused,
            group_ids,
            min_samples=1,
            quantile=0.5,
            strategy="conservative",
            shrink_k=1.0,
            min_ratio=1.0,
        )

        self.assertEqual(cal.group_raw_thresholds["uav_02"], 2.0)
        self.assertGreaterEqual(cal.group_thresholds["uav_02"], 10.0)
        self.assertEqual(cal.group_threshold_sources["uav_02"], "group_conservative_floor")

    def test_conservative_strategy_keeps_high_group_thresholds_above_global(self):
        fused = np.asarray([12.0, 12.0, 12.0], dtype=np.float32)
        group_ids = np.asarray(["uav_high", "uav_high", "uav_high"], dtype=object)
        cal = make_manual_group_calibrator(global_threshold=10.0)

        cal.calibrate_group_thresholds(
            fused,
            group_ids,
            min_samples=1,
            quantile=0.5,
            strategy="conservative",
            shrink_k=1.0,
            min_ratio=1.0,
        )

        self.assertEqual(cal.group_raw_thresholds["uav_high"], 12.0)
        self.assertGreater(cal.group_smoothed_thresholds["uav_high"], 10.0)
        self.assertGreater(cal.group_thresholds["uav_high"], 10.0)
        self.assertEqual(cal.group_threshold_sources["uav_high"], "group_conservative")

    def test_to_dict_from_dict_preserves_group_threshold_strategy_fields(self):
        fused = np.asarray([2.0, 2.0, 2.0], dtype=np.float32)
        group_ids = np.asarray(["uav_02", "uav_02", "uav_02"], dtype=object)
        cal = make_manual_group_calibrator(global_threshold=10.0)

        cal.calibrate_group_thresholds(
            fused,
            group_ids,
            min_samples=1,
            quantile=0.5,
            strategy="conservative",
            shrink_k=1.0,
            min_ratio=0.8,
        )
        payload = cal.to_dict()
        restored = OODCalibrator.from_dict(payload)

        self.assertEqual(restored.group_threshold_strategy, "conservative")
        self.assertEqual(restored.group_threshold_shrink_k, 1.0)
        self.assertEqual(restored.group_threshold_min_ratio, 0.8)
        self.assertEqual(restored.group_raw_thresholds, {"uav_02": 2.0})
        self.assertEqual(restored.group_smoothed_thresholds, cal.group_smoothed_thresholds)
        self.assertEqual(restored.group_thresholds, cal.group_thresholds)
        self.assertEqual(restored.group_threshold_sources, {"uav_02": "group_conservative_floor"})
        self.assertEqual(restored.group_validation_counts, {"uav_02": 3})

    def test_detect_reads_group_thresholds_from_artifact(self):
        raw = make_raw_scores([0.10, 0.50, 0.90])
        cal = OODCalibrator(q_ood=0.5)
        cal.fit(raw)

        artifact = {
            "ood_threshold_mode": "group",
            "global_ood_threshold": 0.75,
            "group_threshold_strategy": "conservative",
            "group_threshold_shrink_k": 1000.0,
            "group_threshold_min_ratio": 1.0,
            "group_threshold_min_samples": 10,
            "group_ood_thresholds": {"uav_01": 0.25, "uav_02": 0.55},
            "group_raw_thresholds": {"uav_01": 0.10, "uav_02": 0.55},
            "group_smoothed_thresholds": {"uav_01": 0.25, "uav_02": 0.55},
            "group_threshold_sources": {"uav_01": "group_conservative", "uav_02": "group_raw"},
            "group_validation_counts": {"uav_01": 12, "uav_02": 15},
            "group_ood_threshold_fallbacks": {
                "uav_small": "fallback_to_global_due_to_small_validation_size",
            },
            "calibration_config": {},
        }

        resolved = detect_script.resolve_ood_threshold_config(artifact, cal)
        transformed = cal.transform(
            raw,
            groups=np.asarray(["uav_01", "uav_small", "uav_new"], dtype=object),
        )

        self.assertEqual(resolved["ood_threshold_mode"], "group")
        self.assertEqual(cal.ood_threshold_mode, "group")
        self.assertEqual(resolved["group_threshold_strategy"], "conservative")
        self.assertEqual(cal.group_thresholds, {"uav_01": 0.25, "uav_02": 0.55})
        self.assertEqual(cal.group_threshold_sources, {"uav_01": "group_conservative", "uav_02": "group_raw"})
        np.testing.assert_allclose(
            transformed["thresholds"],
            np.asarray([0.25, 0.75, 0.75], dtype=np.float32),
        )
        self.assertEqual(
            transformed["threshold_sources"],
            ["group_conservative", "global_fallback", "global_fallback"],
        )


if __name__ == "__main__":
    unittest.main()
