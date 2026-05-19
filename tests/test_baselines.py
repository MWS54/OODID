from __future__ import annotations

import unittest
from unittest import mock

import numpy as np
from sklearn.preprocessing import MaxAbsScaler

from ucs_oodid.baselines import (
    OptionalDependencyUnavailable,
    SafeLightweightSVMClassifier,
    SklearnBaseline,
    apply_threshold,
    calibrate_binary_threshold_from_id_scores,
    compute_baseline_uncertainty_scores,
    make_baseline,
    make_svm_baseline,
    logits_to_probabilities,
    probabilities_to_logits,
)


class _PredictOnlyModel:
    def __init__(self):
        self.seen_shape = None

    def predict(self, x):
        xx = np.asarray(x, dtype=np.float32)
        self.seen_shape = xx.shape
        return (np.sum(xx, axis=1) > 0.0).astype(np.float32)


class BaselineTests(unittest.TestCase):
    def test_make_svm_baseline_uses_scaler_pipeline(self):
        baseline = make_svm_baseline(random_state=7)

        estimator = baseline.model.estimator
        self.assertIsInstance(estimator.named_steps["maxabsscaler"], MaxAbsScaler)
        self.assertIsInstance(estimator.named_steps["lightsvm"], SafeLightweightSVMClassifier)
        self.assertEqual(estimator.named_steps["lightsvm"].class_weight, "balanced")
        self.assertEqual(estimator.named_steps["lightsvm"].max_iter, 200)
        self.assertEqual(baseline.model.n_jobs, 1)

    def test_linear_svm_probability_like_uses_decision_function(self):
        baseline = make_svm_baseline(random_state=7)
        x = np.array(
            [
                [-2.0, -1.0],
                [-1.0, -2.0],
                [1.0, 2.0],
                [2.0, 1.0],
            ],
            dtype=np.float32,
        )
        y = np.array(
            [
                [0, 1],
                [0, 1],
                [1, 0],
                [1, 0],
            ],
            dtype=np.int64,
        )

        baseline.fit(x, y)
        probs = baseline.predict_proba(x)
        logits = baseline.predict_logits_like(x)

        self.assertEqual(probs.shape, (4, 2))
        self.assertEqual(logits.shape, (4, 2))
        self.assertTrue(np.all(probs >= 0.0))
        self.assertTrue(np.all(probs <= 1.0))
        np.testing.assert_allclose(logits_to_probabilities(logits), probs, atol=1e-6)

    def test_lightweight_svm_handles_constant_target_column(self):
        baseline = make_svm_baseline(random_state=7)
        x = np.array(
            [
                [-2.0, -1.0],
                [-1.0, -2.0],
                [1.0, 2.0],
                [2.0, 1.0],
            ],
            dtype=np.float32,
        )
        y = np.array(
            [
                [0, 0],
                [0, 0],
                [1, 0],
                [1, 0],
            ],
            dtype=np.int64,
        )

        baseline.fit(x, y)
        probs = baseline.predict_proba(x)
        logits = baseline.predict_logits_like(x)

        self.assertEqual(probs.shape, (4, 2))
        self.assertEqual(logits.shape, (4, 2))
        np.testing.assert_allclose(probs[:, 1], np.zeros(4, dtype=np.float32), atol=1e-6)

    def test_make_baseline_supports_requested_names(self):
        self.assertEqual(make_baseline("svm").name, "svm")
        self.assertEqual(make_baseline("random_forest").name, "random_forest")
        self.assertEqual(make_baseline("mlp").name, "mlp_tabular")

    def test_make_baseline_xgboost_raises_clear_error_when_missing(self):
        with mock.patch("ucs_oodid.baselines.importlib.import_module", side_effect=ModuleNotFoundError("xgboost")):
            with self.assertRaises(OptionalDependencyUnavailable) as ctx:
                make_baseline("xgboost")

        self.assertIn("xgboost is not installed", str(ctx.exception))

    def test_predict_logits_like_flattens_window_inputs(self):
        model = _PredictOnlyModel()
        baseline = SklearnBaseline("dummy", model)
        windows = np.arange(24, dtype=np.float32).reshape(3, 2, 4)

        scores = baseline.predict_logits_like(windows)

        self.assertEqual(model.seen_shape, (3, 8))
        self.assertEqual(scores.shape, (3, 1))
        np.testing.assert_array_equal(baseline.decision_score(windows), scores)

    def test_compute_baseline_uncertainty_scores(self):
        probs = np.array([[0.2, 0.8], [0.7, 0.3]], dtype=np.float32)

        scores = compute_baseline_uncertainty_scores(probs)

        np.testing.assert_allclose(scores["max_prob"], np.array([0.8, 0.7], dtype=np.float32))
        np.testing.assert_allclose(scores["conf"], np.array([0.2, 0.3], dtype=np.float32))
        expected_energy = -np.sum(np.logaddexp(0.0, probabilities_to_logits(probs)), axis=1)
        np.testing.assert_allclose(scores["energy"], expected_energy.astype(np.float32))

    def test_threshold_helpers(self):
        id_scores = np.array([0.1, 0.2, 0.8, 0.9], dtype=np.float32)

        threshold = calibrate_binary_threshold_from_id_scores(id_scores, q=0.75)
        flags = apply_threshold(np.array([0.2, threshold, 0.95], dtype=np.float32), threshold)

        self.assertAlmostEqual(threshold, float(np.quantile(id_scores, 0.75)))
        np.testing.assert_array_equal(flags, np.array([False, False, True]))


if __name__ == "__main__":
    unittest.main()
