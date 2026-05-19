from __future__ import annotations

import unittest

import numpy as np

from ucs_oodid.metrics import ood_metrics


class OODMetricFieldTests(unittest.TestCase):
    def test_ood_metrics_reports_recall_and_fpr_at_threshold_for_fixed_decisions(self):
        y_true = np.asarray([0, 0, 1, 1], dtype=np.int64)
        scores = np.asarray([0.1, 0.3, 0.8, 0.9], dtype=np.float32)
        decisions = np.asarray([0, 1, 1, 1], dtype=np.int64)

        metrics = ood_metrics(y_true, scores, decisions)

        self.assertAlmostEqual(metrics["precision"], 2.0 / 3.0)
        self.assertAlmostEqual(metrics["tpr"], 1.0)
        self.assertAlmostEqual(metrics["recall"], 1.0)
        self.assertAlmostEqual(metrics["ood_f1"], 0.8)
        self.assertAlmostEqual(metrics["fpr_at_threshold"], 0.5)

    def test_ood_metrics_fpr_at_threshold_uses_only_id_windows(self):
        y_true = np.asarray([0, 0, 0, 1, 1], dtype=np.int64)
        scores = np.asarray([0.9, 0.2, 0.1, 0.8, 0.3], dtype=np.float32)
        decisions = np.asarray([1, 0, 0, 1, 0], dtype=np.int64)

        metrics = ood_metrics(y_true, scores, decisions)

        self.assertAlmostEqual(metrics["fpr_at_threshold"], 1.0 / 3.0)

    def test_ood_metrics_reports_new_fields_for_report_only_threshold(self):
        y_true = np.asarray([0, 0, 1, 1], dtype=np.int64)
        scores = np.asarray([0.1, 0.2, 0.8, 0.9], dtype=np.float32)

        metrics = ood_metrics(y_true, scores)

        self.assertIn("threshold_report_only", metrics)
        self.assertIn("recall", metrics)
        self.assertIn("fpr_at_threshold", metrics)


if __name__ == "__main__":
    unittest.main()
