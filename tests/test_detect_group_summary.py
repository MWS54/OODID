from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "detect.py"
SPEC = importlib.util.spec_from_file_location("detect_script", MODULE_PATH)
detect_script = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(detect_script)


class DetectGroupSummaryTests(unittest.TestCase):
    def test_summarize_group_detections_aggregates_by_group_id(self):
        rows = [
            {"window_id": 0, "group_id": "uav_01", "is_ood": False, "ood_score": 0.2, "ood_threshold": 0.5},
            {"window_id": 1, "group_id": "uav_01", "is_ood": True, "ood_score": 0.9, "ood_threshold": 0.5},
            {"window_id": 2, "group_id": None, "is_ood": True, "ood_score": 0.8, "ood_threshold": 0.4},
            {"window_id": 3, "group_id": "uav_02", "is_ood": False, "ood_score": 0.1, "ood_threshold": 0.6},
        ]

        summary = detect_script.summarize_group_detections(rows, "uav_id")

        self.assertEqual(summary["group_col"], "uav_id")
        self.assertEqual(set(summary["groups"]), {"uav_01", "uav_02", "__ungrouped__"})
        self.assertEqual(summary["groups"]["uav_01"]["windows"], 2)
        self.assertEqual(summary["groups"]["uav_01"]["ood_alerts"], 1)
        self.assertEqual(summary["groups"]["uav_01"]["alert_rate"], 0.5)
        self.assertEqual(summary["groups"]["uav_01"]["mean_ood_score"], 0.55)
        self.assertEqual(summary["groups"]["uav_01"]["max_ood_score"], 0.9)
        self.assertEqual(summary["groups"]["uav_01"]["threshold"], 0.5)
        self.assertEqual(summary["groups"]["uav_01"]["threshold_source"], "global")
        self.assertEqual(summary["groups"]["uav_01"]["mean_ood_threshold"], 0.5)
        self.assertEqual(summary["groups"]["uav_01"]["top_window_ids"], [1, 0])
        self.assertEqual(summary["groups"]["__ungrouped__"]["windows"], 1)
        self.assertEqual(summary["groups"]["__ungrouped__"]["threshold"], 0.4)
        self.assertEqual(summary["groups"]["__ungrouped__"]["top_window_ids"], [2])
        self.assertEqual(summary["global"]["windows"], 4)
        self.assertEqual(summary["global"]["ood_alerts"], 2)
        self.assertEqual(summary["global"]["alert_rate"], 0.5)
        self.assertEqual(summary["global"]["mean_ood_score"], 0.5)
        self.assertEqual(summary["global"]["max_ood_score"], 0.9)

    def test_canonical_summary_group_id_maps_none_to_ungrouped(self):
        self.assertEqual(detect_script.canonical_summary_group_id(None), "__ungrouped__")

    def test_summarize_group_detections_keeps_real_threshold_source(self):
        rows = [
            {
                "window_id": 0,
                "group_id": "uav_02",
                "is_ood": True,
                "ood_score": 1.2,
                "ood_threshold": 1.0,
                "ood_threshold_source": "group_conservative_floor",
            }
        ]

        summary = detect_script.summarize_group_detections(rows, "uav_id")

        self.assertEqual(summary["groups"]["uav_02"]["threshold_source"], "group_conservative_floor")

    def test_summarize_group_detections_merges_present_class_id_metrics(self):
        rows = [
            {"window_id": 0, "group_id": "uav_01", "is_ood": False, "ood_score": 0.2, "ood_threshold": 0.5},
            {"window_id": 1, "group_id": "uav_01", "is_ood": True, "ood_score": 0.9, "ood_threshold": 0.5},
        ]
        id_metrics_by_group = {
            "uav_01": {
                "windows": 2,
                "macro_f1": 0.4,
                "present_class_macro_f1": 1.0,
                "present_class_count": 2,
                "present_class_names": ["attack_a", "attack_b"],
                "absent_class_count": 3,
                "absent_class_names": ["attack_c", "attack_d", "attack_e"],
                "class_support": {"attack_a": 1, "attack_b": 1, "attack_c": 0, "attack_d": 0, "attack_e": 0},
            }
        }

        summary = detect_script.summarize_group_detections(
            rows,
            "uav_id",
            id_metrics_by_group=id_metrics_by_group,
            present_class_min_support=1,
        )

        self.assertEqual(summary["present_class_min_support"], 1)
        self.assertEqual(summary["groups"]["uav_01"]["macro_f1"], 0.4)
        self.assertEqual(summary["groups"]["uav_01"]["present_class_macro_f1"], 1.0)
        self.assertEqual(summary["groups"]["uav_01"]["present_class_count"], 2)
        self.assertEqual(summary["groups"]["uav_01"]["present_class_names"], ["attack_a", "attack_b"])
        self.assertEqual(summary["groups"]["uav_01"]["absent_class_count"], 3)
        self.assertEqual(summary["groups"]["uav_01"]["class_support"]["attack_c"], 0)


if __name__ == "__main__":
    unittest.main()
