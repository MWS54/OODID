from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np


def _resolve_thresholds(ood_threshold: float | np.ndarray, t_count: int) -> np.ndarray:
    thresholds = np.asarray(ood_threshold, dtype=np.float32)
    if thresholds.ndim == 0:
        thresholds = np.full(t_count, float(thresholds), dtype=np.float32)
    elif len(thresholds) != t_count:
        raise ValueError("ood_threshold must be scalar or match the number of windows.")
    return thresholds


def build_record_level_suspiciousness_ranking(
    record_ids: np.ndarray,
    attention: np.ndarray,
    fused_scores: np.ndarray,
    ood_threshold: float | np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    window_ids: Optional[np.ndarray] = None,
    eps: float = 1e-8,
) -> tuple[Dict[str, float], List[dict]]:
    """Build analyst-triage record rankings from window excess and attention.

    This produces a suspiciousness ranking for analyst review rather than a
    complete per-record attack-localization output.
    """
    numerator = defaultdict(float)
    denominator = defaultdict(float)
    best_evidence: Dict[str, dict] = {}
    t_count, win_size = record_ids.shape
    if valid_mask is None:
        valid_mask = np.ones_like(attention, dtype=bool)
    thresholds = _resolve_thresholds(ood_threshold, t_count)
    if window_ids is None:
        window_ids = np.arange(t_count, dtype=np.int64)
    else:
        window_ids = np.asarray(window_ids, dtype=np.int64)
        if len(window_ids) != t_count:
            raise ValueError("window_ids must match the number of windows.")
    for t in range(t_count):
        excess = max(0.0, float(fused_scores[t]) - float(thresholds[t]))
        for i in range(win_size):
            if not bool(valid_mask[t, i]):
                continue
            rid = str(record_ids[t, i])
            alpha = float(attention[t, i])
            contribution = alpha * excess
            numerator[rid] += contribution
            denominator[rid] += alpha
            evidence = {
                "window_id": int(window_ids[t]),
                "predicted_ood_score": float(fused_scores[t]),
                "attention_weight": alpha,
                "window_excess_score": float(excess),
                "contribution": float(contribution),
            }
            current = best_evidence.get(rid)
            if current is None or evidence["contribution"] > current["contribution"]:
                best_evidence[rid] = evidence
    ranking: List[dict] = []
    for rid in denominator:
        score = numerator[rid] / (denominator[rid] + eps)
        evidence = best_evidence.get(
            rid,
            {
                "window_id": -1,
                "predicted_ood_score": 0.0,
                "attention_weight": 0.0,
                "window_excess_score": 0.0,
                "contribution": 0.0,
            },
        )
        ranking.append(
            {
                "record_id": rid,
                "score": float(score),
                "window_id": int(evidence["window_id"]),
                "predicted_ood_score": float(evidence["predicted_ood_score"]),
                "attention_weight": float(evidence["attention_weight"]),
                "window_excess_score": float(evidence["window_excess_score"]),
            }
        )
    ranking.sort(key=lambda row: (row["score"], row["predicted_ood_score"], row["attention_weight"]), reverse=True)
    record_scores = {row["record_id"]: float(row["score"]) for row in ranking}
    return record_scores, ranking


def aggregate_record_suspiciousness(
    record_ids: np.ndarray,
    attention: np.ndarray,
    fused_scores: np.ndarray,
    ood_threshold: float | np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    eps: float = 1e-8,
) -> Dict[str, float]:
    """Compute record-level unknown attribution over overlapping windows.

    Padding entries introduced by time/adaptive windows are ignored when valid_mask is provided.
    """
    record_scores, _ = build_record_level_suspiciousness_ranking(
        record_ids,
        attention,
        fused_scores,
        ood_threshold,
        valid_mask=valid_mask,
        window_ids=None,
        eps=eps,
    )
    return record_scores


def top_suspicious_records(record_scores: Dict[str, float], topk: int = 20) -> List[dict]:
    return [
        {"record_id": rid, "suspiciousness": float(score)}
        for rid, score in sorted(record_scores.items(), key=lambda kv: kv[1], reverse=True)[:topk]
    ]


def dominant_ood_source(normalized_scores: dict, idx: int) -> str:
    names = list(normalized_scores.keys())
    vals = [float(normalized_scores[n][idx]) for n in names]
    return names[int(np.argmax(vals))]
