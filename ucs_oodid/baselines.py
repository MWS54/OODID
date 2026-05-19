from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Callable, Dict, Optional

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler


class OptionalDependencyUnavailable(RuntimeError):
    """Raised when an optional comparison dependency is unavailable."""


def flatten_windows(
    x: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    include_mask: bool = False,
) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"x must be a 3D window tensor, got shape {arr.shape}")
    flat = arr.copy()
    mask_array = None
    if valid_mask is not None:
        mask_array = np.asarray(valid_mask, dtype=bool)
        if mask_array.shape != arr.shape[:2]:
            raise ValueError(f"valid_mask shape {mask_array.shape} does not match window shape {arr.shape[:2]}")
        flat = flat * mask_array[..., None].astype(np.float32)
    flat = flat.reshape(arr.shape[0], -1)
    if include_mask and mask_array is not None:
        flat = np.concatenate([flat, mask_array.astype(np.float32)], axis=1)
    return flat.astype(np.float32)


def flatten_window_metadata(windows, include_mask: bool = True) -> np.ndarray:
    return flatten_windows(
        windows.x,
        valid_mask=getattr(windows, "valid_mask", None),
        include_mask=include_mask,
    )


def probabilities_to_logits(probs: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    clipped = np.clip(np.asarray(probs, dtype=np.float32), eps, 1.0 - eps)
    return np.log(clipped) - np.log1p(-clipped)


def logits_to_probabilities(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    scaled = np.asarray(logits, dtype=np.float32) / max(float(temperature), 1e-6)
    return (1.0 / (1.0 + np.exp(-scaled))).astype(np.float32)


def _ensure_2d_float_matrix(x: np.ndarray, *, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim == 1:
        return arr[:, None].astype(np.float32)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 1D or 2D array, got shape {arr.shape}")
    return arr.astype(np.float32)


@dataclass
class SklearnBaseline:
    name: str
    model: object

    def _maybe_flatten(self, x: np.ndarray) -> np.ndarray:
        arr = np.asarray(x)
        if arr.ndim == 3:
            return flatten_windows(arr)
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D or 3D input for sklearn baseline, got shape {arr.shape}")
        return arr.astype(np.float32)

    @staticmethod
    def _estimator_classes(estimator: object):
        classes = getattr(estimator, "classes_", None)
        if classes is not None:
            return np.asarray(classes)
        named_steps = getattr(estimator, "named_steps", None)
        if named_steps:
            last_step = list(named_steps.values())[-1]
            classes = getattr(last_step, "classes_", None)
            if classes is not None:
                return np.asarray(classes)
        return None

    def _positive_probability(self, probs: np.ndarray, estimator: object | None) -> np.ndarray:
        arr = np.asarray(probs, dtype=np.float32)
        if arr.ndim == 1:
            return arr.astype(np.float32)
        classes = self._estimator_classes(estimator) if estimator is not None else None
        if arr.shape[1] == 1:
            if classes is not None and len(classes) == 1 and int(classes[0]) == 1:
                return np.ones(arr.shape[0], dtype=np.float32)
            return np.zeros(arr.shape[0], dtype=np.float32)
        if classes is not None:
            positive = np.where(classes == 1)[0]
            if len(positive):
                return arr[:, int(positive[0])].astype(np.float32)
        return arr[:, 1].astype(np.float32)

    def _positive_margin(self, scores: np.ndarray, estimator: object | None) -> np.ndarray:
        arr = np.asarray(scores, dtype=np.float32)
        if arr.ndim == 1:
            return arr.astype(np.float32)
        classes = self._estimator_classes(estimator) if estimator is not None else None
        if arr.shape[1] == 1:
            return arr[:, 0].astype(np.float32)
        if classes is not None:
            positive = np.where(classes == 1)[0]
            if len(positive):
                return arr[:, int(positive[0])].astype(np.float32)
        return arr[:, -1].astype(np.float32)

    def _stack_estimator_outputs(self, xx: np.ndarray, method_name: str) -> np.ndarray | None:
        estimators = list(getattr(self.model, "estimators_", []))
        if not estimators:
            return None
        stacked = []
        for estimator in estimators:
            if not hasattr(estimator, method_name):
                return None
            output = getattr(estimator, method_name)(xx)
            if method_name == "predict_proba":
                stacked.append(self._positive_probability(output, estimator))
            else:
                stacked.append(self._positive_margin(output, estimator))
        return np.stack(stacked, axis=1).astype(np.float32)

    def _decision_function_like(self, xx: np.ndarray) -> np.ndarray | None:
        if hasattr(self.model, "decision_function"):
            decision = self.model.decision_function(xx)
            if isinstance(decision, list):
                estimators = list(getattr(self.model, "estimators_", []))
                stacked = []
                for idx, score in enumerate(decision):
                    estimator = estimators[idx] if idx < len(estimators) else None
                    stacked.append(self._positive_margin(score, estimator))
                return np.stack(stacked, axis=1).astype(np.float32)
            return _ensure_2d_float_matrix(
                self._positive_margin(decision, self.model),
                name="decision_function",
            )
        return self._stack_estimator_outputs(xx, "decision_function")

    def fit(self, x: np.ndarray, y: np.ndarray):
        xx = self._maybe_flatten(x)
        self.model.fit(xx, np.asarray(y, dtype=np.int64))
        return self

    def _predict_probability_like(self, x: np.ndarray, ensure_2d: bool = False) -> np.ndarray:
        xx = self._maybe_flatten(x)
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(xx)
            if isinstance(probs, list):
                estimators = list(getattr(self.model, "estimators_", []))
                stacked = []
                for idx, proba in enumerate(probs):
                    estimator = estimators[idx] if idx < len(estimators) else None
                    stacked.append(self._positive_probability(proba, estimator))
                return np.stack(stacked, axis=1).astype(np.float32)
            probs_array = np.asarray(probs, dtype=np.float32)
            if ensure_2d and probs_array.ndim == 1:
                return probs_array[:, None].astype(np.float32)
            return probs_array.astype(np.float32)
        stacked_probs = self._stack_estimator_outputs(xx, "predict_proba")
        if stacked_probs is not None:
            return stacked_probs.astype(np.float32)
        decision_like = self._decision_function_like(xx)
        if decision_like is not None:
            probs_array = logits_to_probabilities(decision_like)
            if ensure_2d and probs_array.ndim == 1:
                return probs_array[:, None].astype(np.float32)
            return probs_array.astype(np.float32)
        pred = self.model.predict(xx)
        pred_array = np.asarray(pred, dtype=np.float32)
        if ensure_2d and pred_array.ndim == 1:
            return pred_array[:, None].astype(np.float32)
        return pred_array.astype(np.float32)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return self._predict_probability_like(x, ensure_2d=False)

    def predict_logits_like(self, x: np.ndarray) -> np.ndarray:
        xx = self._maybe_flatten(x)
        decision_like = self._decision_function_like(xx)
        if decision_like is not None:
            return decision_like.astype(np.float32)
        probs = self._predict_probability_like(xx, ensure_2d=True)
        return probabilities_to_logits(_ensure_2d_float_matrix(probs, name="probs"))

    def decision_score(self, x: np.ndarray) -> np.ndarray:
        return self.predict_logits_like(x)


class SafeLightweightSVMClassifier(BaseEstimator, ClassifierMixin):
    """A lightweight linear SVM-style classifier with single-class fallback."""

    def __init__(
        self,
        *,
        class_weight: str | dict | None = "balanced",
        alpha: float = 5e-4,
        max_iter: int = 200,
        tol: float = 1e-2,
        average: bool = True,
        random_state: int = 42,
    ) -> None:
        self.class_weight = class_weight
        self.alpha = float(alpha)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.average = bool(average)
        self.random_state = int(random_state)

    def fit(self, x: np.ndarray, y: np.ndarray):
        xx = np.asarray(x, dtype=np.float32)
        yy = np.asarray(y, dtype=np.int64).reshape(-1)
        self.classes_ = np.array([0, 1], dtype=np.int64)
        unique = np.unique(yy)
        self.constant_prediction_ = int(unique[0]) if len(unique) else 0
        self.model_ = None
        if len(unique) < 2:
            return self
        self.model_ = SGDClassifier(
            loss="hinge",
            penalty="l2",
            alpha=self.alpha,
            class_weight=self.class_weight,
            max_iter=self.max_iter,
            tol=self.tol,
            average=self.average,
            random_state=self.random_state,
        )
        self.model_.fit(xx, yy)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        xx = np.asarray(x, dtype=np.float32)
        if self.model_ is None:
            return np.full(xx.shape[0], self.constant_prediction_, dtype=np.int64)
        return np.asarray(self.model_.predict(xx), dtype=np.int64)

    def decision_function(self, x: np.ndarray) -> np.ndarray:
        xx = np.asarray(x, dtype=np.float32)
        if self.model_ is None:
            margin = 12.0 if int(self.constant_prediction_) == 1 else -12.0
            return np.full(xx.shape[0], margin, dtype=np.float32)
        return np.asarray(self.model_.decision_function(xx), dtype=np.float32).reshape(-1)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        xx = np.asarray(x, dtype=np.float32)
        if self.model_ is None:
            if int(self.constant_prediction_) == 1:
                return np.tile(np.array([0.0, 1.0], dtype=np.float32), (xx.shape[0], 1))
            return np.tile(np.array([1.0, 0.0], dtype=np.float32), (xx.shape[0], 1))
        margins = np.asarray(self.decision_function(x), dtype=np.float32).reshape(-1)
        positive = logits_to_probabilities(margins)
        return np.stack([1.0 - positive, positive], axis=1).astype(np.float32)


def make_svm_baseline(random_state: int = 42) -> SklearnBaseline:
    estimator = MultiOutputClassifier(
        Pipeline(
            steps=[
                ("maxabsscaler", MaxAbsScaler(copy=False)),
                (
                    "lightsvm",
                    SafeLightweightSVMClassifier(
                        class_weight="balanced",
                        alpha=5e-4,
                        max_iter=200,
                        tol=1e-2,
                        average=True,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        n_jobs=1,
    )
    return SklearnBaseline("svm", estimator)


def make_mlp_baseline(hidden=(128, 64), random_state: int = 42) -> SklearnBaseline:
    estimator = MultiOutputClassifier(
        MLPClassifier(
            hidden_layer_sizes=hidden,
            max_iter=300,
            random_state=random_state,
        )
    )
    return SklearnBaseline("mlp_tabular", estimator)


def make_random_forest_baseline(random_state: int = 42) -> SklearnBaseline:
    estimator = MultiOutputClassifier(
        RandomForestClassifier(
            n_estimators=200,
            random_state=random_state,
            n_jobs=-1,
        )
    )
    return SklearnBaseline("random_forest", estimator)


def make_xgboost_baseline(random_state: int = 42) -> SklearnBaseline:
    try:
        xgboost = importlib.import_module("xgboost")
        xgb_classifier = xgboost.XGBClassifier
    except ModuleNotFoundError as exc:
        raise OptionalDependencyUnavailable(
            "xgboost is not installed. Install xgboost to use the xgboost baseline."
        ) from exc
    except AttributeError as exc:
        raise OptionalDependencyUnavailable(
            "xgboost.XGBClassifier is unavailable in this xgboost installation."
        ) from exc
    estimator = MultiOutputClassifier(
        xgb_classifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=-1,
            tree_method="hist",
        )
    )
    return SklearnBaseline("xgboost", estimator)


TABULAR_BASELINE_FACTORIES: Dict[str, Callable[..., SklearnBaseline]] = {
    "svm": make_svm_baseline,
    "random_forest": make_random_forest_baseline,
    "xgboost": make_xgboost_baseline,
    "mlp": make_mlp_baseline,
    "mlp_tabular": make_mlp_baseline,
}


def make_baseline(name: str, random_state: int = 42) -> SklearnBaseline:
    key = str(name).strip().lower()
    if key not in TABULAR_BASELINE_FACTORIES:
        raise ValueError(f"Unsupported baseline: {name}. Supported baselines: svm, random_forest, xgboost, mlp")
    return TABULAR_BASELINE_FACTORIES[key](random_state=random_state)


def make_tabular_baseline(name: str, random_state: int = 42) -> SklearnBaseline:
    key = str(name).strip().lower()
    if key not in TABULAR_BASELINE_FACTORIES:
        raise ValueError(f"Unsupported tabular baseline: {name}")
    return make_baseline(key, random_state=random_state)


def compute_baseline_uncertainty_scores(probs: np.ndarray) -> Dict[str, np.ndarray]:
    prob_matrix = _ensure_2d_float_matrix(probs, name="probs")
    max_prob = np.max(prob_matrix, axis=1).astype(np.float32)
    logits = probabilities_to_logits(prob_matrix)
    energy = -np.sum(np.logaddexp(0.0, logits), axis=1).astype(np.float32)
    return {
        "conf": (1.0 - max_prob).astype(np.float32),
        "energy": energy.astype(np.float32),
        "max_prob": max_prob.astype(np.float32),
    }


def calibrate_binary_threshold_from_id_scores(id_scores: np.ndarray, q: float = 0.90) -> float:
    return float(np.quantile(np.asarray(id_scores, dtype=np.float32), q))


def apply_threshold(scores: np.ndarray, threshold: float) -> np.ndarray:
    return np.asarray(scores, dtype=np.float32) > float(threshold)


def fit_isolation_forest_ood(id_embeddings: np.ndarray, random_state=42) -> IsolationForest:
    model = IsolationForest(n_estimators=200, contamination="auto", random_state=random_state, n_jobs=-1)
    model.fit(id_embeddings)
    return model


def isolation_forest_scores(model: IsolationForest, embeddings: np.ndarray) -> np.ndarray:
    return -model.score_samples(embeddings)
