from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ucs_oodid.metrics import compute_present_class_macro_f1, multilabel_metrics


class PresentClassMacroF1Tests(unittest.TestCase):
    def test_compute_present_class_macro_f1_ignores_absent_global_labels(self):
        label_names = ["attack_a", "attack_b", "attack_c", "attack_d", "attack_e"]
        y_true = np.asarray(
            [
                [1, 0, 0, 0, 0],
                [0, 1, 0, 0, 0],
                [1, 0, 0, 0, 0],
                [0, 1, 0, 0, 0],
            ],
            dtype=np.int64,
        )
        y_pred = np.asarray(
            [
                [1, 0, 1, 0, 0],
                [0, 1, 0, 1, 0],
                [1, 0, 0, 0, 1],
                [0, 1, 0, 0, 0],
            ],
            dtype=np.int64,
        )
        thresholds = np.full(y_true.shape[1], 0.5, dtype=np.float32)

        global_metrics = multilabel_metrics(y_true, y_pred.astype(np.float32), thresholds)
        present_metrics = compute_present_class_macro_f1(y_true, y_pred, label_names=label_names)

        self.assertAlmostEqual(global_metrics["macro_f1"], 0.4)
        self.assertAlmostEqual(present_metrics["present_class_macro_f1"], 1.0)
        self.assertEqual(present_metrics["present_class_count"], 2)
        self.assertEqual(present_metrics["present_class_names"], ["attack_a", "attack_b"])
        self.assertEqual(present_metrics["absent_class_count"], 3)
        self.assertEqual(
            present_metrics["absent_class_names"],
            ["attack_c", "attack_d", "attack_e"],
        )

    def test_compute_present_class_macro_f1_respects_min_support_threshold(self):
        label_names = ["attack_a", "attack_b", "attack_c", "attack_d", "attack_e"]
        y_true = np.asarray(
            [
                [1, 0, 0, 0, 0],
                [1, 0, 0, 0, 0],
                [0, 1, 0, 0, 0],
            ],
            dtype=np.int64,
        )
        y_pred = np.asarray(
            [
                [1, 0, 0, 0, 0],
                [1, 0, 0, 0, 0],
                [0, 1, 0, 0, 0],
            ],
            dtype=np.int64,
        )

        metrics = compute_present_class_macro_f1(y_true, y_pred, label_names=label_names, min_support=2)

        self.assertAlmostEqual(metrics["present_class_macro_f1"], 1.0)
        self.assertEqual(metrics["present_class_count"], 1)
        self.assertEqual(metrics["present_class_names"], ["attack_a"])
        self.assertEqual(metrics["absent_class_count"], 4)
        self.assertEqual(
            metrics["absent_class_names"],
            ["attack_b", "attack_c", "attack_d", "attack_e"],
        )

    def test_compute_present_class_macro_f1_returns_nan_when_no_labels_present(self):
        label_names = ["attack_a", "attack_b", "attack_c", "attack_d", "attack_e"]
        y_true = np.zeros((3, 5), dtype=np.int64)
        y_pred = np.asarray(
            [
                [0, 0, 1, 0, 0],
                [0, 1, 0, 0, 0],
                [1, 0, 0, 0, 0],
            ],
            dtype=np.int64,
        )

        metrics = compute_present_class_macro_f1(y_true, y_pred, label_names=label_names)

        self.assertTrue(math.isnan(metrics["present_class_macro_f1"]))
        self.assertEqual(metrics["present_class_count"], 0)
        self.assertEqual(metrics["present_class_names"], [])
        self.assertEqual(metrics["absent_class_count"], 5)
        self.assertEqual(metrics["absent_class_names"], label_names)


if __name__ == "__main__":
    unittest.main()
