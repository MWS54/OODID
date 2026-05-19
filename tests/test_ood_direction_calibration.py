from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ucs_oodid.ood import OODCalibrator, build_leave_one_class_out_pseudo_ood


def make_raw_scores(values):
    return {
        "conf": np.asarray(values, dtype=np.float32),
        "energy": np.asarray(values, dtype=np.float32),
        "proto": np.asarray(values, dtype=np.float32),
        "knn": np.asarray(values, dtype=np.float32),
    }


class OODDirectionCalibrationTests(unittest.TestCase):
    def test_default_directions_keep_predefined_orientation(self):
        cal = OODCalibrator().set_default_directions(label_source="none")
        cal.fit(make_raw_scores([0.1, 0.2, 0.3, 0.4]))
        self.assertEqual(cal.direction_label_source, "none")
        self.assertEqual(cal.directions, {name: 1.0 for name in cal.score_names})
        for item in cal.direction_report:
            self.assertIsNone(item["raw_auroc"])
            self.assertFalse(item["flipped"])
            self.assertEqual(item["direction"], 1.0)

    def test_pseudo_ood_leave_one_class_out_builds_binary_labels(self):
        raw_scores = {
            "conf": np.asarray([0.10, 0.80, 0.20, 0.70], dtype=np.float32),
            "energy": np.asarray([0.15, 0.75, 0.25, 0.65], dtype=np.float32),
            "proto": np.asarray([0.12, 0.72, 0.22, 0.62], dtype=np.float32),
            "knn": np.asarray([0.11, 0.71, 0.21, 0.61], dtype=np.float32),
        }
        labels = np.asarray(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
                [0.0, 1.0],
            ],
            dtype=np.float32,
        )
        pseudo_raw, pseudo_labels, summary = build_leave_one_class_out_pseudo_ood(
            raw_scores,
            labels,
            class_names=["class_a", "class_b"],
        )
        self.assertEqual(len(summary), 2)
        self.assertEqual({item["class_name"] for item in summary}, {"class_a", "class_b"})
        self.assertEqual(set(np.unique(pseudo_labels).tolist()), {0, 1})
        self.assertGreater(len(pseudo_labels), len(labels))
        for name in ("conf", "energy", "proto", "knn"):
            self.assertEqual(len(pseudo_raw[name]), len(pseudo_labels))

    def test_direction_calibration_flips_inverted_scores(self):
        cal = OODCalibrator()
        cal.calibrate_directions(
            make_raw_scores([0.90, 0.80, 0.20, 0.10]),
            y_true_ood=np.asarray([0, 0, 1, 1], dtype=np.int64),
            label_source="pseudo_ood",
        )
        self.assertEqual(cal.direction_label_source, "pseudo_ood")
        for name in cal.score_names:
            self.assertEqual(cal.directions[name], -1.0)
        for item in cal.direction_report:
            self.assertTrue(item["flipped"])
            self.assertLess(item["raw_auroc"], 0.5)
            self.assertGreater(item["effective_auroc"], 0.5)


if __name__ == "__main__":
    unittest.main()
