from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    hamming_loss,
    roc_auc_score,
)


def _resolve_label_names(num_classes: int, label_names: Optional[Sequence[str]] = None) -> list[str]:
    if label_names is None:
        return [str(index) for index in range(num_classes)]
    labels = [str(name) for name in label_names]
    if len(labels) != num_classes:
        raise ValueError(f"label_names length {len(labels)} does not match num_classes {num_classes}")
    return labels


def compute_class_support(y_true: np.ndarray, label_names: Optional[Sequence[str]] = None) -> Dict[str, int]:
    y_true_array = np.asarray(y_true)
    if y_true_array.ndim != 2:
        raise ValueError(f"y_true must be 2D, got shape {y_true_array.shape}")
    labels = _resolve_label_names(y_true_array.shape[1], label_names=label_names)
    supports = np.asarray(y_true_array.sum(axis=0), dtype=np.float64)
    return {label: int(round(float(supports[index]))) for index, label in enumerate(labels)}


def compute_present_class_macro_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: Optional[Sequence[str]] = None,
    min_support: int = 1,
) -> Dict[str, Any]:
    if min_support < 0:
        raise ValueError("min_support must be >= 0")
    y_true_array = np.asarray(y_true)
    y_pred_array = np.asarray(y_pred)
    if y_true_array.ndim != 2:
        raise ValueError(f"y_true must be 2D, got shape {y_true_array.shape}")
    if y_true_array.shape != y_pred_array.shape:
        raise ValueError(f"y_true shape {y_true_array.shape} does not match y_pred shape {y_pred_array.shape}")
    labels = _resolve_label_names(y_true_array.shape[1], label_names=label_names)
    supports = np.asarray(y_true_array.sum(axis=0), dtype=np.float64)
    present_indices = np.flatnonzero(supports >= float(min_support))
    absent_indices = np.flatnonzero(supports < float(min_support))
    present_class_macro_f1 = (
        float(f1_score(y_true_array[:, present_indices], y_pred_array[:, present_indices], average="macro", zero_division=0))
        if len(present_indices)
        else float("nan")
    )
    return {
        "present_class_macro_f1": present_class_macro_f1,
        "present_class_count": int(len(present_indices)),
        "present_class_names": [labels[index] for index in present_indices],
        "absent_class_count": int(len(absent_indices)),
        "absent_class_names": [labels[index] for index in absent_indices],
    }


def multilabel_metrics(y_true: np.ndarray, probs: np.ndarray, thresholds: Optional[np.ndarray] = None) -> Dict[str, float]:
    if thresholds is None:
        thresholds = np.full(y_true.shape[1], 0.5)
    pred = (probs >= thresholds.reshape(1, -1)).astype(int)
    out = {
        "micro_f1": float(f1_score(y_true, pred, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        "hamming_loss": float(hamming_loss(y_true, pred)),
        "subset_accuracy": float((pred == y_true).all(axis=1).mean()),
    }
    aps = []
    for j in range(y_true.shape[1]):
        if len(np.unique(y_true[:, j])) > 1:
            aps.append(average_precision_score(y_true[:, j], probs[:, j]))
    out["mAP"] = float(np.mean(aps)) if aps else 0.0
    return out


def fpr_at_tpr(y_true_ood: np.ndarray, scores: np.ndarray, target_tpr: float = 0.95) -> float:
    y = y_true_ood.astype(int)
    if y.max() == 0 or y.min() == 1:
        return float("nan")
    order = np.argsort(scores)[::-1]
    y_sorted = y[order]
    pos = max(y.sum(), 1)
    neg = max((1 - y).sum(), 1)
    tp = np.cumsum(y_sorted == 1)
    fp = np.cumsum(y_sorted == 0)
    tpr = tp / pos
    fpr = fp / neg
    idx = np.where(tpr >= target_tpr)[0]
    return float(fpr[idx[0]]) if len(idx) else 1.0


def _binary_decision_metrics(y_true_ood: np.ndarray, decisions: np.ndarray) -> tuple[float, float, float, float]:
    y = np.asarray(y_true_ood, dtype=int)
    pred = np.asarray(decisions, dtype=int)
    if y.shape != pred.shape:
        raise ValueError(f"y_true_ood shape {y.shape} does not match decisions shape {pred.shape}")
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else float("nan")
    return float(precision), float(recall), float(f1), fpr


def _fpr_at_fixed_threshold(y_true_ood: np.ndarray, decisions: np.ndarray) -> float:
    y = np.asarray(y_true_ood, dtype=int)
    pred = np.asarray(decisions, dtype=int)
    if y.shape != pred.shape:
        raise ValueError(f"y_true_ood shape {y.shape} does not match decisions shape {pred.shape}")
    id_mask = y == 0
    false_positive = int(((pred == 1) & id_mask).sum())
    true_negative = int(((pred == 0) & id_mask).sum())
    return float(false_positive / max(false_positive + true_negative, 1))


def ood_metrics(y_true_ood: np.ndarray, scores: np.ndarray, decisions: Optional[np.ndarray] = None) -> Dict[str, float]:
    y = y_true_ood.astype(int)
    out: Dict[str, float] = {}
    if len(np.unique(y)) > 1:
        out["auroc"] = float(roc_auc_score(y, scores))
        out["aupr_out"] = float(average_precision_score(y, scores))
        out["fpr95"] = fpr_at_tpr(y, scores, 0.95)
    else:
        out["auroc"] = float("nan")
        out["aupr_out"] = float("nan")
        out["fpr95"] = float("nan")
    if decisions is None:
        # Choose best threshold for reporting only; calibration should use ID validation.
        if len(scores) == 0:
            out["precision"] = float("nan")
            out["tpr"] = float("nan")
            out["recall"] = float("nan")
            out["ood_f1"] = float("nan")
            out["threshold_report_only"] = float("nan")
            out["fpr_at_threshold"] = float("nan")
            return out
        thresholds = np.quantile(scores, np.linspace(0.05, 0.95, 19))
        best = (0.0, 0.0, 0.0, 0.0, float("nan"))
        for thr in thresholds:
            pred = scores > thr
            p, r, f, fpr = _binary_decision_metrics(y, pred.astype(int))
            if f > best[2]:
                best = (p, r, f, float(thr), fpr)
        out["precision"] = float(best[0])
        out["tpr"] = float(best[1])
        out["recall"] = float(best[1])
        out["ood_f1"] = float(best[2])
        out["threshold_report_only"] = float(best[3])
        out["fpr_at_threshold"] = float(best[4])
    else:
        fixed_decisions = decisions.astype(int)
        p, r, f, _ = _binary_decision_metrics(y, fixed_decisions)
        out["precision"] = float(p)
        out["tpr"] = float(r)
        out["recall"] = float(r)
        out["ood_f1"] = float(f)
        # Keep this separate from fpr95: this is the realized false-positive rate
        # on ID samples under the provided fixed threshold decisions.
        out["fpr_at_threshold"] = _fpr_at_fixed_threshold(y, fixed_decisions)
    return out


def attribution_metrics(record_scores: Dict[str, float], malicious_record_ids: set, topk: int = 20) -> Dict[str, float]:
    if not record_scores:
        return {"hit_rate": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "analyst_reduction": 0.0}
    ranked = sorted(record_scores.items(), key=lambda kv: kv[1], reverse=True)
    top = [rid for rid, _ in ranked[:topk]]
    hit = len(set(top) & malicious_record_ids)
    precision = hit / max(len(top), 1)
    recall = hit / max(len(malicious_record_ids), 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    reduction = 1.0 - len(top) / max(len(record_scores), 1)
    return {
        "hit_rate": float(1.0 if hit > 0 else 0.0),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "analyst_reduction": float(reduction),
    }
