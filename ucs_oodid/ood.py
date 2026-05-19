from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_auc_score

VALID_OOD_FUSIONS = ("correlation_aware", "mean", "hard_voting", "variance_weighted", "conf", "energy", "proto", "knn")
VALID_OOD_THRESHOLD_MODES = ("global", "group")
VALID_GROUP_THRESHOLD_STRATEGIES = ("raw", "shrink", "conservative", "global_floor")
DEPRECATED_VALIDATION_WEIGHTED_MESSAGE = (
    "`validation_weighted` is deprecated and mapped to `variance_weighted`; "
    "the weights are inverse-variance over validation-score dispersion, not label-aware validation performance."
)


def canonicalize_ood_fusion(fusion: str, emit_warning: bool = False) -> str:
    mode = str(fusion).strip().lower()
    if mode == "validation_weighted":
        if emit_warning:
            print(DEPRECATED_VALIDATION_WEIGHTED_MESSAGE)
        return "variance_weighted"
    if mode not in VALID_OOD_FUSIONS:
        raise ValueError(f"Unsupported OOD fusion: {fusion}")
    return mode


def canonicalize_ood_threshold_mode(mode: str | None) -> str:
    key = str(mode or "global").strip().lower()
    if key not in VALID_OOD_THRESHOLD_MODES:
        raise ValueError(f"Unsupported OOD threshold mode: {mode}")
    return key


def canonicalize_group_threshold_strategy(strategy: str | None) -> str:
    key = str(strategy or "raw").strip().lower()
    if key not in VALID_GROUP_THRESHOLD_STRATEGIES:
        raise ValueError(f"Unsupported group threshold strategy: {strategy}")
    return key


@dataclass
class PrototypeBank:
    prototypes: np.ndarray
    precision: np.ndarray
    bank_embeddings: np.ndarray
    bank_labels: np.ndarray
    class_names: List[str]

    @staticmethod
    def fit(embeddings: np.ndarray, labels: np.ndarray, class_names: Sequence[str], shrinkage: bool = True) -> "PrototypeBank":
        c = labels.shape[1]
        d = embeddings.shape[1]
        protos = np.zeros((c, d), dtype=np.float32)
        for i in range(c):
            mask = labels[:, i] > 0.5
            protos[i] = embeddings[mask].mean(axis=0) if mask.any() else embeddings.mean(axis=0)
        centered = embeddings - embeddings.mean(axis=0, keepdims=True)
        if len(embeddings) > d + 2 and shrinkage:
            try:
                precision = LedoitWolf().fit(centered).precision_.astype(np.float32)
            except Exception:
                precision = np.eye(d, dtype=np.float32)
        else:
            cov = np.cov(centered.T) if len(embeddings) > 1 else np.eye(d)
            cov = cov + np.eye(d) * 1e-3
            precision = np.linalg.pinv(cov).astype(np.float32)
        return PrototypeBank(protos, precision, embeddings.astype(np.float32), labels.astype(np.float32), list(class_names))

    def to_dict(self) -> dict:
        return {
            "prototypes": self.prototypes,
            "precision": self.precision,
            "bank_embeddings": self.bank_embeddings,
            "bank_labels": self.bank_labels,
            "class_names": self.class_names,
        }

    @staticmethod
    def from_dict(obj: dict) -> "PrototypeBank":
        return PrototypeBank(
            prototypes=np.asarray(obj["prototypes"], dtype=np.float32),
            precision=np.asarray(obj["precision"], dtype=np.float32),
            bank_embeddings=np.asarray(obj["bank_embeddings"], dtype=np.float32),
            bank_labels=np.asarray(obj["bank_labels"], dtype=np.float32),
            class_names=list(obj["class_names"]),
        )


def calibrate_temperature(logits: np.ndarray, labels: np.ndarray, grid: Sequence[float]) -> float:
    logits_t = torch.tensor(logits, dtype=torch.float32)
    labels_t = torch.tensor(labels, dtype=torch.float32)
    best_t, best_loss = 1.0, float("inf")
    for t in grid:
        loss = F.binary_cross_entropy_with_logits(logits_t / float(t), labels_t).item()
        if loss < best_loss:
            best_loss = loss
            best_t = float(t)
    return best_t


def calibrate_class_thresholds(probs: np.ndarray, labels: np.ndarray, grid: Optional[Sequence[float]] = None) -> np.ndarray:
    if grid is None:
        grid = np.linspace(0.05, 0.95, 19)
    c = labels.shape[1]
    thresholds = np.full(c, 0.5, dtype=np.float32)
    for j in range(c):
        y = labels[:, j].astype(int)
        if y.max() == 0:
            thresholds[j] = 0.95
            continue
        best_f1, best_thr = -1.0, 0.5
        for thr in grid:
            pred = (probs[:, j] >= thr).astype(int)
            tp = ((pred == 1) & (y == 1)).sum()
            fp = ((pred == 1) & (y == 0)).sum()
            fn = ((pred == 0) & (y == 1)).sum()
            precision = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)
            f1 = 2 * precision * recall / max(precision + recall, 1e-12)
            if f1 > best_f1:
                best_f1, best_thr = f1, float(thr)
        thresholds[j] = best_thr
    return thresholds


def compute_raw_ood_scores(
    logits: np.ndarray,
    probs: np.ndarray,
    embeddings: np.ndarray,
    bank: PrototypeBank,
    temperature: float = 1.0,
    k_bank: int = 5,
) -> Dict[str, np.ndarray]:
    logits = np.asarray(logits, dtype=np.float32)
    probs = np.asarray(probs, dtype=np.float32)
    embeddings = np.asarray(embeddings, dtype=np.float32)
    conf = 1.0 - probs.max(axis=1)
    energy = -float(temperature) * torch.logsumexp(torch.tensor(logits / max(temperature, 1e-6), dtype=torch.float32), dim=1).numpy()

    diff = embeddings[:, None, :] - bank.prototypes[None, :, :]
    precision = bank.precision.astype(np.float32)
    mahal = np.einsum("ncd,df,ncf->nc", diff, precision, diff)
    proto = np.sqrt(np.clip(mahal.min(axis=1), 0.0, None))

    emb_norm = embeddings / np.clip(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-8, None)
    bank_emb = bank.bank_embeddings
    bank_norm = bank_emb / np.clip(np.linalg.norm(bank_emb, axis=1, keepdims=True), 1e-8, None)
    sim = emb_norm @ bank_norm.T
    k_eff = min(max(1, int(k_bank)), sim.shape[1])
    topk = np.partition(sim, -k_eff, axis=1)[:, -k_eff:]
    knn = -topk.mean(axis=1)
    return {"conf": conf.astype(np.float32), "energy": energy.astype(np.float32), "proto": proto.astype(np.float32), "knn": knn.astype(np.float32)}


def build_leave_one_class_out_pseudo_ood(
    raw_scores: Dict[str, np.ndarray],
    labels: np.ndarray,
    class_names: Optional[Sequence[str]] = None,
    score_names: Optional[Sequence[str]] = None,
) -> tuple[Dict[str, np.ndarray], np.ndarray, List[dict]]:
    labels = np.asarray(labels, dtype=np.float32)
    if labels.ndim != 2:
        raise ValueError("labels must be a 2D multi-label array for pseudo-OOD construction.")
    names = list(score_names or raw_scores.keys())
    pseudo_parts = {name: [] for name in names}
    pseudo_labels: List[np.ndarray] = []
    pseudo_summary: List[dict] = []
    for class_idx in range(labels.shape[1]):
        class_name = (
            str(class_names[class_idx])
            if class_names is not None and class_idx < len(class_names)
            else f"class_{class_idx}"
        )
        pos_mask = labels[:, class_idx] > 0.5
        neg_mask = ~pos_mask
        pos_count = int(pos_mask.sum())
        neg_count = int(neg_mask.sum())
        used = pos_count > 0 and neg_count > 0
        pseudo_summary.append(
            {
                "class_name": class_name,
                "pseudo_ood_windows": pos_count,
                "pseudo_id_windows": neg_count,
                "used": used,
            }
        )
        if not used:
            continue
        for name in names:
            arr = np.asarray(raw_scores[name], dtype=np.float32)
            if len(arr) != len(labels):
                raise ValueError(f"Score length mismatch for pseudo-OOD construction: {name}")
            pseudo_parts[name].append(arr[neg_mask])
            pseudo_parts[name].append(arr[pos_mask])
        pseudo_labels.append(np.zeros(neg_count, dtype=np.int64))
        pseudo_labels.append(np.ones(pos_count, dtype=np.int64))
    if not pseudo_labels:
        empty = {name: np.empty((0,), dtype=np.float32) for name in names}
        return empty, np.empty((0,), dtype=np.int64), pseudo_summary
    merged = {name: np.concatenate(parts, axis=0).astype(np.float32) for name, parts in pseudo_parts.items()}
    return merged, np.concatenate(pseudo_labels, axis=0), pseudo_summary


@dataclass
class OODCalibrator:
    fusion: str = "correlation_aware"
    q_ood: float = 0.95
    score_names: List[str] = field(default_factory=lambda: ["conf", "energy", "proto", "knn"])
    medians: Dict[str, float] = field(default_factory=dict)
    iqrs: Dict[str, float] = field(default_factory=dict)
    weights: Dict[str, float] = field(default_factory=dict)
    directions: Dict[str, float] = field(default_factory=dict)
    direction_report: List[dict] = field(default_factory=list)
    direction_label_source: Optional[str] = None
    ood_threshold_mode: str = "global"
    group_threshold_min_samples: int = 0
    group_threshold_quantile: Optional[float] = None
    group_threshold_strategy: str = "raw"
    group_threshold_shrink_k: float = 1000.0
    group_threshold_min_ratio: float = 1.0
    group_thresholds: Dict[str, float] = field(default_factory=dict)
    group_raw_thresholds: Dict[str, float] = field(default_factory=dict)
    group_smoothed_thresholds: Dict[str, float] = field(default_factory=dict)
    group_threshold_sources: Dict[str, str] = field(default_factory=dict)
    group_validation_counts: Dict[str, int] = field(default_factory=dict)
    group_threshold_fallbacks: Dict[str, str] = field(default_factory=dict)
    phase_aware_enabled: bool = False
    phase_column: Optional[str] = None
    phase_threshold_min_samples: int = 0
    phase_threshold_quantile: Optional[float] = None
    phase_threshold_fallback: str = "global"
    phase_thresholds: Dict[str, float] = field(default_factory=dict)
    phase_validation_counts: Dict[str, int] = field(default_factory=dict)
    phase_threshold_sources: Dict[str, str] = field(default_factory=dict)
    threshold: float = 0.0
    per_score_thresholds: Dict[str, float] = field(default_factory=dict)
    correlation: Optional[np.ndarray] = None
    single_score: Optional[str] = None

    def __post_init__(self) -> None:
        self.fusion = canonicalize_ood_fusion(self.fusion, emit_warning=True)
        self.ood_threshold_mode = canonicalize_ood_threshold_mode(self.ood_threshold_mode)
        self.group_threshold_strategy = canonicalize_group_threshold_strategy(self.group_threshold_strategy)
        self.group_threshold_shrink_k = float(max(self.group_threshold_shrink_k, 0.0))
        self.group_threshold_min_ratio = float(self.group_threshold_min_ratio)

    def _score_direction_defaults(self, label_source: Optional[str] = None) -> None:
        self.direction_label_source = label_source
        self.directions = {name: 1.0 for name in self.score_names}
        self.direction_report = [
            {
                "score_name": name,
                "raw_auroc": None,
                "flipped": False,
                "effective_auroc": None,
                "direction": 1.0,
            }
            for name in self.score_names
        ]

    def set_default_directions(self, label_source: Optional[str] = None) -> "OODCalibrator":
        self._score_direction_defaults(label_source=label_source)
        return self

    @staticmethod
    def _normalize_phase_label(phase: object) -> Optional[str]:
        if phase is None:
            return None
        if isinstance(phase, (float, np.floating)) and np.isnan(phase):
            return None
        text = str(phase).strip()
        return text or None

    @staticmethod
    def _normalize_group_label(group: object) -> Optional[str]:
        if group is None:
            return None
        if isinstance(group, (float, np.floating)) and np.isnan(group):
            return None
        text = str(group).strip()
        return text or None

    @staticmethod
    def _safe_auroc(y_true_ood: Optional[np.ndarray], scores: np.ndarray) -> Optional[float]:
        if y_true_ood is None:
            return None
        y = np.asarray(y_true_ood, dtype=int)
        if len(y) != len(scores):
            raise ValueError("y_true_ood must match score length for direction calibration.")
        if len(np.unique(y)) < 2:
            return None
        try:
            return float(roc_auc_score(y, scores))
        except ValueError:
            return None

    def estimate_directions(
        self,
        raw_scores: Dict[str, np.ndarray],
        y_true_ood: np.ndarray,
    ) -> tuple[Dict[str, float], List[dict]]:
        y = np.asarray(y_true_ood, dtype=int)
        directions: Dict[str, float] = {}
        report: List[dict] = []
        for name in self.score_names:
            arr = np.asarray(raw_scores[name], dtype=np.float32)
            raw_auroc = self._safe_auroc(y, arr)
            direction = 1.0
            flipped = False
            aligned = arr
            if raw_auroc is not None and raw_auroc < 0.5:
                direction = -1.0
                flipped = True
                aligned = -arr
            effective_auroc = self._safe_auroc(y, aligned)
            directions[name] = float(direction)
            report.append(
                {
                    "score_name": name,
                    "raw_auroc": raw_auroc,
                    "flipped": flipped,
                    "effective_auroc": effective_auroc,
                    "direction": float(direction),
                }
            )
        return directions, report

    def calibrate_directions(
        self,
        raw_scores: Dict[str, np.ndarray],
        y_true_ood: Optional[np.ndarray] = None,
        label_source: Optional[str] = None,
    ) -> "OODCalibrator":
        if y_true_ood is None:
            self.set_default_directions(label_source=label_source)
            return self

        self.direction_label_source = label_source or "provided_ood_labels"
        self.directions, self.direction_report = self.estimate_directions(raw_scores, y_true_ood=np.asarray(y_true_ood, dtype=int))
        return self

    def calibrate_phase_thresholds(
        self,
        fused_scores: np.ndarray,
        phases: Optional[Sequence[object]],
        phase_column: str,
        min_samples: int,
        quantile: Optional[float] = None,
        fallback: str = "global",
    ) -> "OODCalibrator":
        self.phase_aware_enabled = True
        self.phase_column = phase_column
        self.phase_threshold_min_samples = int(max(min_samples, 0))
        self.phase_threshold_quantile = float(self.q_ood if quantile is None else quantile)
        self.phase_threshold_fallback = fallback
        self.phase_thresholds = {}
        self.phase_validation_counts = {}
        self.phase_threshold_sources = {}

        if phases is None:
            return self
        labels = [self._normalize_phase_label(phase) for phase in phases]
        fused = np.asarray(fused_scores, dtype=np.float32)
        if len(labels) != len(fused):
            raise ValueError("phases must match fused score length for phase-aware threshold calibration.")
        unique_phases = sorted({label for label in labels if label is not None})
        for phase in unique_phases:
            mask = np.asarray([label == phase for label in labels], dtype=bool)
            count = int(mask.sum())
            self.phase_validation_counts[phase] = count
            if count >= self.phase_threshold_min_samples:
                threshold = float(np.quantile(fused[mask], self.phase_threshold_quantile))
                source = "phase"
            else:
                threshold = float(self.threshold)
                source = "global_fallback"
            self.phase_thresholds[phase] = threshold
            self.phase_threshold_sources[phase] = source
        return self

    def calibrate_group_thresholds(
        self,
        fused_scores: np.ndarray,
        group_ids: Optional[Sequence[object]],
        min_samples: int = 10,
        quantile: Optional[float] = None,
        strategy: str = "raw",
        shrink_k: float = 1000.0,
        min_ratio: float = 1.0,
    ) -> "OODCalibrator":
        self.ood_threshold_mode = "group"
        self.group_threshold_min_samples = int(max(min_samples, 0))
        self.group_threshold_quantile = float(self.q_ood if quantile is None else quantile)
        self.group_threshold_strategy = canonicalize_group_threshold_strategy(strategy)
        self.group_threshold_shrink_k = float(max(shrink_k, 0.0))
        self.group_threshold_min_ratio = float(min_ratio)
        self.group_thresholds = {}
        self.group_raw_thresholds = {}
        self.group_smoothed_thresholds = {}
        self.group_threshold_sources = {}
        self.group_validation_counts = {}
        self.group_threshold_fallbacks = {}

        if group_ids is None:
            return self

        labels = [self._normalize_group_label(group_id) for group_id in group_ids]
        fused = np.asarray(fused_scores, dtype=np.float32)
        if len(labels) != len(fused):
            raise ValueError("group_ids must match fused score length for group-aware threshold calibration.")
        unique_groups = sorted({label for label in labels if label is not None})
        for group in unique_groups:
            mask = np.asarray([label == group for label in labels], dtype=bool)
            count = int(mask.sum())
            self.group_validation_counts[group] = count
            if count >= self.group_threshold_min_samples:
                raw_threshold = float(np.quantile(fused[mask], self.group_threshold_quantile))
                smoothed_threshold = raw_threshold
                final_threshold = raw_threshold
                source = "group_raw"

                if self.group_threshold_strategy == "shrink":
                    alpha = 1.0 if self.group_threshold_shrink_k <= 0 else float(count / (count + self.group_threshold_shrink_k))
                    smoothed_threshold = float(alpha * raw_threshold + (1.0 - alpha) * float(self.threshold))
                    final_threshold = smoothed_threshold
                    source = "group_shrink"
                elif self.group_threshold_strategy == "conservative":
                    alpha = 1.0 if self.group_threshold_shrink_k <= 0 else float(count / (count + self.group_threshold_shrink_k))
                    smoothed_threshold = float(alpha * raw_threshold + (1.0 - alpha) * float(self.threshold))
                    floor_threshold = float(self.threshold) * float(self.group_threshold_min_ratio)
                    final_threshold = float(max(smoothed_threshold, floor_threshold))
                    source = "group_conservative_floor" if final_threshold == floor_threshold else "group_conservative"
                elif self.group_threshold_strategy == "global_floor":
                    final_threshold = float(max(raw_threshold, float(self.threshold)))
                    source = "group_global_floor"

                self.group_raw_thresholds[group] = float(raw_threshold)
                self.group_smoothed_thresholds[group] = float(smoothed_threshold)
                self.group_thresholds[group] = float(final_threshold)
                self.group_threshold_sources[group] = source
            else:
                self.group_threshold_fallbacks[group] = "fallback_to_global_due_to_small_validation_size"
        return self

    def resolve_thresholds(
        self,
        phases: Optional[Sequence[object]],
        count: int,
        groups: Optional[Sequence[object]] = None,
    ) -> tuple[np.ndarray, List[str]]:
        thresholds = np.full(count, float(self.threshold), dtype=np.float32)
        sources = ["global"] * count
        if self.ood_threshold_mode == "group":
            if groups is None:
                return thresholds, sources
            labels = [self._normalize_group_label(group_id) for group_id in groups]
            if len(labels) != count:
                raise ValueError("group_ids must match sample count for threshold resolution.")
            for i, group in enumerate(labels):
                if group is None:
                    continue
                if group in self.group_thresholds:
                    thresholds[i] = float(self.group_thresholds[group])
                    sources[i] = self.group_threshold_sources.get(group, "group")
                else:
                    sources[i] = "global_fallback"
            return thresholds, sources
        if not self.phase_aware_enabled or phases is None:
            return thresholds, sources
        labels = [self._normalize_phase_label(phase) for phase in phases]
        if len(labels) != count:
            raise ValueError("phases must match sample count for threshold resolution.")
        for i, phase in enumerate(labels):
            if phase is None:
                continue
            if phase in self.phase_thresholds:
                thresholds[i] = float(self.phase_thresholds[phase])
                sources[i] = self.phase_threshold_sources.get(phase, "phase")
            elif self.phase_threshold_fallback == "global":
                thresholds[i] = float(self.threshold)
                sources[i] = "global_fallback"
        return thresholds, sources

    def orient_scores(self, raw_scores: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        if not self.directions:
            self._score_direction_defaults(label_source=None)
        oriented: Dict[str, np.ndarray] = {}
        for name in self.score_names:
            arr = np.asarray(raw_scores[name], dtype=np.float32)
            oriented[name] = (float(self.directions.get(name, 1.0)) * arr).astype(np.float32)
        return oriented

    def fit(
        self,
        raw_scores: Dict[str, np.ndarray],
        y_true_ood: Optional[np.ndarray] = None,
        label_source: Optional[str] = None,
    ) -> "OODCalibrator":
        if y_true_ood is not None:
            self.calibrate_directions(raw_scores, y_true_ood=y_true_ood, label_source=label_source)
        elif not self.directions or not self.direction_report:
            self.set_default_directions(label_source=label_source)
        norm = self.normalize(raw_scores, fit=True)
        matrix = np.stack([norm[k] for k in self.score_names], axis=1)
        if len(matrix) > 2:
            corr = np.corrcoef(matrix.T)
            corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            corr = np.eye(len(self.score_names), dtype=np.float32)
        self.correlation = corr.astype(np.float32)
        self.weights = self._estimate_weights(matrix, corr)
        self.per_score_thresholds = {name: float(np.quantile(norm[name], self.q_ood)) for name in self.score_names}
        fused = self.fuse_normalized(norm)
        self.threshold = float(np.quantile(fused, self.q_ood))
        return self

    def normalize(self, raw_scores: Dict[str, np.ndarray], fit: bool = False) -> Dict[str, np.ndarray]:
        oriented_scores = self.orient_scores(raw_scores)
        out: Dict[str, np.ndarray] = {}
        for name in self.score_names:
            arr = np.asarray(oriented_scores[name], dtype=np.float32)
            if fit:
                q25, q75 = np.quantile(arr, [0.25, 0.75])
                self.medians[name] = float(np.median(arr))
                self.iqrs[name] = float(max(q75 - q25, 1e-6))
            out[name] = (arr - self.medians[name]) / max(self.iqrs[name], 1e-6)
        return out

    def _estimate_weights(self, matrix: np.ndarray, corr: np.ndarray) -> Dict[str, float]:
        if self.fusion in self.score_names:
            return {n: 1.0 if n == self.fusion else 0.0 for n in self.score_names}
        if self.fusion == "mean":
            return {n: 1.0 / len(self.score_names) for n in self.score_names}
        if self.fusion == "variance_weighted":
            std = np.std(matrix, axis=0) + 1e-6
            inv = 1.0 / std
            inv = inv / inv.sum()
            return {n: float(inv[i]) for i, n in enumerate(self.score_names)}
        if self.fusion == "hard_voting":
            return {n: 1.0 / len(self.score_names) for n in self.score_names}
        # correlation-aware default: downweight redundant scores.
        redundancy = 1.0 + (np.abs(corr).sum(axis=1) - 1.0)
        inv = 1.0 / np.clip(redundancy, 1e-6, None)
        inv = inv / inv.sum()
        return {n: float(inv[i]) for i, n in enumerate(self.score_names)}

    def fuse_normalized(self, normalized_scores: Dict[str, np.ndarray]) -> np.ndarray:
        if self.fusion == "hard_voting":
            votes = []
            for name in self.score_names:
                arr = normalized_scores[name]
                thr = self.per_score_thresholds.get(name, 0.0)
                votes.append((arr > thr).astype(np.float32))
            return np.mean(np.stack(votes, axis=1), axis=1)
        fused = np.zeros_like(next(iter(normalized_scores.values())), dtype=np.float32)
        for name, weight in self.weights.items():
            fused += float(weight) * normalized_scores[name]
        return fused.astype(np.float32)

    def transform(
        self,
        raw_scores: Dict[str, np.ndarray],
        phases: Optional[Sequence[object]] = None,
        groups: Optional[Sequence[object]] = None,
    ) -> dict:
        norm = self.normalize(raw_scores, fit=False)
        fused = self.fuse_normalized(norm)
        thresholds, threshold_sources = self.resolve_thresholds(phases, len(fused), groups=groups)
        decisions = fused > thresholds
        return {"normalized": norm, "fused": fused, "thresholds": thresholds, "threshold_sources": threshold_sources, "decisions": decisions}

    def to_dict(self) -> dict:
        return {
            "fusion": self.fusion,
            "q_ood": self.q_ood,
            "score_names": self.score_names,
            "medians": {k: float(v) for k, v in self.medians.items()},
            "iqrs": {k: float(v) for k, v in self.iqrs.items()},
            "weights": {k: float(v) for k, v in self.weights.items()},
            "directions": {k: float(v) for k, v in self.directions.items()},
            "direction_report": self.direction_report,
            "direction_label_source": self.direction_label_source,
            "ood_threshold_mode": self.ood_threshold_mode,
            "group_threshold_min_samples": self.group_threshold_min_samples,
            "group_threshold_quantile": self.group_threshold_quantile,
            "group_threshold_strategy": self.group_threshold_strategy,
            "group_threshold_shrink_k": float(self.group_threshold_shrink_k),
            "group_threshold_min_ratio": float(self.group_threshold_min_ratio),
            "group_thresholds": {k: float(v) for k, v in self.group_thresholds.items()},
            "group_raw_thresholds": {k: float(v) for k, v in self.group_raw_thresholds.items()},
            "group_smoothed_thresholds": {k: float(v) for k, v in self.group_smoothed_thresholds.items()},
            "group_threshold_sources": {k: str(v) for k, v in self.group_threshold_sources.items()},
            "group_validation_counts": {k: int(v) for k, v in self.group_validation_counts.items()},
            "group_threshold_fallbacks": {k: str(v) for k, v in self.group_threshold_fallbacks.items()},
            "phase_aware_enabled": self.phase_aware_enabled,
            "phase_column": self.phase_column,
            "phase_threshold_min_samples": self.phase_threshold_min_samples,
            "phase_threshold_quantile": self.phase_threshold_quantile,
            "phase_threshold_fallback": self.phase_threshold_fallback,
            "phase_thresholds": {k: float(v) for k, v in self.phase_thresholds.items()},
            "phase_validation_counts": {k: int(v) for k, v in self.phase_validation_counts.items()},
            "phase_threshold_sources": {k: str(v) for k, v in self.phase_threshold_sources.items()},
            "threshold": float(self.threshold),
            "per_score_thresholds": {k: float(v) for k, v in self.per_score_thresholds.items()},
            "correlation": self.correlation,
            "single_score": self.single_score,
        }

    @staticmethod
    def from_dict(obj: dict) -> "OODCalibrator":
        fusion = canonicalize_ood_fusion(obj.get("fusion", "correlation_aware"), emit_warning=False)
        threshold_mode = canonicalize_ood_threshold_mode(
            obj.get("ood_threshold_mode", "group" if obj.get("group_thresholds") else "global")
        )
        cal = OODCalibrator(
            fusion=fusion,
            q_ood=float(obj.get("q_ood", 0.95)),
            score_names=list(obj.get("score_names", ["conf", "energy", "proto", "knn"])),
            ood_threshold_mode=threshold_mode,
        )
        cal.medians = {k: float(v) for k, v in obj.get("medians", {}).items()}
        cal.iqrs = {k: float(v) for k, v in obj.get("iqrs", {}).items()}
        cal.weights = {k: float(v) for k, v in obj.get("weights", {}).items()}
        cal.directions = {k: float(v) for k, v in obj.get("directions", {}).items()}
        cal.direction_report = list(obj.get("direction_report", []))
        cal.direction_label_source = obj.get("direction_label_source")
        cal.group_threshold_min_samples = int(obj.get("group_threshold_min_samples", 0))
        group_quantile = obj.get("group_threshold_quantile")
        cal.group_threshold_quantile = None if group_quantile is None else float(group_quantile)
        cal.group_threshold_strategy = canonicalize_group_threshold_strategy(obj.get("group_threshold_strategy", "raw"))
        cal.group_threshold_shrink_k = float(max(float(obj.get("group_threshold_shrink_k", 1000.0)), 0.0))
        cal.group_threshold_min_ratio = float(obj.get("group_threshold_min_ratio", 1.0))
        cal.group_thresholds = {k: float(v) for k, v in obj.get("group_thresholds", {}).items()}
        cal.group_raw_thresholds = {k: float(v) for k, v in obj.get("group_raw_thresholds", {}).items()}
        cal.group_smoothed_thresholds = {k: float(v) for k, v in obj.get("group_smoothed_thresholds", {}).items()}
        cal.group_threshold_sources = {k: str(v) for k, v in obj.get("group_threshold_sources", {}).items()}
        cal.group_validation_counts = {k: int(v) for k, v in obj.get("group_validation_counts", {}).items()}
        cal.group_threshold_fallbacks = {k: str(v) for k, v in obj.get("group_threshold_fallbacks", {}).items()}
        cal.phase_aware_enabled = bool(obj.get("phase_aware_enabled", False))
        cal.phase_column = obj.get("phase_column")
        cal.phase_threshold_min_samples = int(obj.get("phase_threshold_min_samples", 0))
        quantile = obj.get("phase_threshold_quantile")
        cal.phase_threshold_quantile = None if quantile is None else float(quantile)
        cal.phase_threshold_fallback = str(obj.get("phase_threshold_fallback", "global"))
        cal.phase_thresholds = {k: float(v) for k, v in obj.get("phase_thresholds", {}).items()}
        cal.phase_validation_counts = {k: int(v) for k, v in obj.get("phase_validation_counts", {}).items()}
        cal.phase_threshold_sources = {k: str(v) for k, v in obj.get("phase_threshold_sources", {}).items()}
        cal.threshold = float(obj.get("threshold", 0.0))
        cal.per_score_thresholds = {k: float(v) for k, v in obj.get("per_score_thresholds", {}).items()}
        if obj.get("correlation") is not None:
            cal.correlation = np.asarray(obj["correlation"], dtype=np.float32)
        cal.single_score = obj.get("single_score")
        if not cal.directions or not cal.direction_report:
            cal.set_default_directions(label_source=cal.direction_label_source)
        return cal
