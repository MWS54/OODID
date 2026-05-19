from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from ..artifacts import load_artifact
from ..attribution import build_record_level_suspiciousness_ranking, dominant_ood_source
from ..baselines import compute_baseline_uncertainty_scores
from ..inference import collect_model_outputs
from ..model import UCSOODID
from ..ood import OODCalibrator, PrototypeBank, compute_raw_ood_scores
from ..utils import choose_device, parse_label_cell, set_seed
from ..windowing import UNKNOWN_GROUP_TOKEN, WindowedData, attach_group_index, attach_parsed_labels, window_phase_labels
from .simulation_feature_adapter import SimulationFeatureAdapter
from .streaming_buffer import BufferedWindow, StreamingWindowBuffer


def _normalize_group_id(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, (float, np.floating)) and np.isnan(value):
        return None
    text = str(value).strip()
    return text or None


def resolve_preprocessor_from_artifact(artifact: dict):
    pre = artifact["preprocessor"]
    normalization_mode = str(artifact.get("normalization_mode", getattr(pre, "normalization_mode", "global")) or "global").strip().lower()
    if artifact.get("feature_cols") is not None:
        pre.feature_cols = list(artifact["feature_cols"])
    if artifact.get("feature_medians") is not None:
        pre.feature_medians = {str(col): float(value) for col, value in artifact["feature_medians"].items()}
    if artifact.get("group_col") is not None:
        pre.group_col = artifact["group_col"]
    if artifact.get("global_scaler") is not None:
        pre.scaler = artifact["global_scaler"]
    if normalization_mode == "group":
        pre.normalization_mode = "group"
        group_scalers = artifact.get("group_scalers")
        if group_scalers is not None:
            pre.group_scalers = dict(group_scalers)
        elif getattr(pre, "group_scalers", None) is None:
            pre.group_scalers = {}
        group_fallbacks = artifact.get("group_normalization_fallbacks")
        if group_fallbacks is not None:
            pre.group_fallbacks = {str(group): str(reason) for group, reason in group_fallbacks.items()}
        elif getattr(pre, "group_fallbacks", None) is None:
            pre.group_fallbacks = {}
    else:
        pre.normalization_mode = "global"
    return pre, normalization_mode


def resolve_ood_threshold_config(artifact: dict, ood_cal: OODCalibrator, override_mode: str | None = None) -> dict:
    artifact_mode = artifact.get("ood_threshold_mode")
    if artifact_mode is None:
        artifact_mode = artifact.get("calibration_config", {}).get("ood_threshold_mode", getattr(ood_cal, "ood_threshold_mode", "global"))
    resolved_mode = str(override_mode or artifact_mode or "global").strip().lower()
    if resolved_mode not in {"global", "group"}:
        raise ValueError(f"Unsupported OOD threshold mode: {resolved_mode}")
    ood_cal.ood_threshold_mode = resolved_mode
    ood_cal.threshold = float(artifact.get("global_ood_threshold", getattr(ood_cal, "threshold", 0.0)))
    group_thresholds = artifact.get("group_ood_thresholds")
    if group_thresholds is not None:
        ood_cal.group_thresholds = {str(group): float(value) for group, value in group_thresholds.items()}
    elif getattr(ood_cal, "group_thresholds", None) is None:
        ood_cal.group_thresholds = {}
    group_raw_thresholds = artifact.get("group_raw_thresholds")
    if group_raw_thresholds is not None:
        ood_cal.group_raw_thresholds = {str(group): float(value) for group, value in group_raw_thresholds.items()}
    elif getattr(ood_cal, "group_raw_thresholds", None) is None:
        ood_cal.group_raw_thresholds = {}
    group_smoothed_thresholds = artifact.get("group_smoothed_thresholds")
    if group_smoothed_thresholds is not None:
        ood_cal.group_smoothed_thresholds = {str(group): float(value) for group, value in group_smoothed_thresholds.items()}
    elif getattr(ood_cal, "group_smoothed_thresholds", None) is None:
        ood_cal.group_smoothed_thresholds = {}
    group_threshold_sources = artifact.get("group_threshold_sources")
    if group_threshold_sources is not None:
        ood_cal.group_threshold_sources = {str(group): str(source) for group, source in group_threshold_sources.items()}
    elif getattr(ood_cal, "group_threshold_sources", None) is None:
        ood_cal.group_threshold_sources = {}
    group_validation_counts = artifact.get("group_validation_counts")
    if group_validation_counts is not None:
        ood_cal.group_validation_counts = {str(group): int(count) for group, count in group_validation_counts.items()}
    elif getattr(ood_cal, "group_validation_counts", None) is None:
        ood_cal.group_validation_counts = {}
    group_fallbacks = artifact.get("group_ood_threshold_fallbacks")
    if group_fallbacks is not None:
        ood_cal.group_threshold_fallbacks = {str(group): str(reason) for group, reason in group_fallbacks.items()}
    elif getattr(ood_cal, "group_threshold_fallbacks", None) is None:
        ood_cal.group_threshold_fallbacks = {}
    ood_cal.group_threshold_min_samples = int(
        artifact.get(
            "group_threshold_min_samples",
            artifact.get("calibration_config", {}).get(
                "group_threshold_min_samples",
                getattr(ood_cal, "group_threshold_min_samples", 0),
            ),
        )
    )
    quantile = artifact.get("group_threshold_quantile", getattr(ood_cal, "group_threshold_quantile", None))
    ood_cal.group_threshold_quantile = None if quantile is None else float(quantile)
    ood_cal.group_threshold_strategy = str(
        artifact.get(
            "group_threshold_strategy",
            artifact.get("calibration_config", {}).get(
                "group_threshold_strategy",
                getattr(ood_cal, "group_threshold_strategy", "raw"),
            ),
        )
    )
    ood_cal.group_threshold_shrink_k = float(
        artifact.get(
            "group_threshold_shrink_k",
            artifact.get("calibration_config", {}).get(
                "group_threshold_shrink_k",
                getattr(ood_cal, "group_threshold_shrink_k", 1000.0),
            ),
        )
    )
    ood_cal.group_threshold_min_ratio = float(
        artifact.get(
            "group_threshold_min_ratio",
            artifact.get("calibration_config", {}).get(
                "group_threshold_min_ratio",
                getattr(ood_cal, "group_threshold_min_ratio", 1.0),
            ),
        )
    )
    return {
        "ood_threshold_mode": ood_cal.ood_threshold_mode,
        "global_ood_threshold": float(ood_cal.threshold),
        "group_threshold_min_samples": int(ood_cal.group_threshold_min_samples),
        "group_threshold_quantile": ood_cal.group_threshold_quantile,
        "group_threshold_strategy": str(ood_cal.group_threshold_strategy),
        "group_threshold_shrink_k": float(ood_cal.group_threshold_shrink_k),
        "group_threshold_min_ratio": float(ood_cal.group_threshold_min_ratio),
        "group_ood_thresholds": {group: float(value) for group, value in ood_cal.group_thresholds.items()},
        "group_raw_thresholds": {group: float(value) for group, value in ood_cal.group_raw_thresholds.items()},
        "group_smoothed_thresholds": {group: float(value) for group, value in ood_cal.group_smoothed_thresholds.items()},
        "group_threshold_sources": dict(ood_cal.group_threshold_sources),
        "group_validation_counts": {group: int(count) for group, count in ood_cal.group_validation_counts.items()},
        "group_ood_threshold_fallbacks": dict(ood_cal.group_threshold_fallbacks),
    }


def resolve_group_embedding_config(artifact: dict) -> dict:
    model_cfg = artifact.get("model_config", {})
    enabled = bool(artifact.get("use_group_embedding", model_cfg.get("use_group_embedding", False)))
    group_embedding_dim = int(artifact.get("group_embedding_dim", model_cfg.get("group_embedding_dim", 16)))
    group_to_index = artifact.get("group_to_index")
    if group_to_index is None:
        group_to_index = artifact.get("group_embedding", {}).get("group_to_index", {})
    group_to_index = {str(group): int(index) for group, index in (group_to_index or {}).items()}
    unknown_group_index = artifact.get("unknown_group_index", model_cfg.get("unknown_group_index"))
    if enabled and UNKNOWN_GROUP_TOKEN not in group_to_index:
        fallback_index = int(unknown_group_index) if unknown_group_index is not None else len(group_to_index)
        group_to_index[UNKNOWN_GROUP_TOKEN] = fallback_index
    unknown_group_index = group_to_index.get(UNKNOWN_GROUP_TOKEN, unknown_group_index)
    return {
        "enabled": enabled,
        "group_embedding_dim": group_embedding_dim,
        "group_to_index": group_to_index,
        "unknown_group_index": None if unknown_group_index is None else int(unknown_group_index),
    }


def attach_group_embedding_indices(windows: WindowedData, artifact: dict) -> dict:
    config = resolve_group_embedding_config(artifact)
    if config["enabled"]:
        attach_group_index(windows, config["group_to_index"], unknown_group=UNKNOWN_GROUP_TOKEN)
    return config


def alert_level_for_margin(excess: float) -> str:
    if excess >= 1.0:
        return "critical"
    if excess >= 0.0:
        return "warning"
    if excess >= -0.25:
        return "watch"
    return "normal"


def collapse_attack_types(attack_types: Sequence[object]) -> str:
    ordered: list[str] = []
    seen: set[str] = set()
    for item in attack_types:
        text = str(item).strip()
        if not text or text == "benign" or text in seen:
            continue
        ordered.append(text)
        seen.add(text)
    if not ordered:
        return "benign"
    if len(ordered) == 1:
        return ordered[0]
    return "+".join(ordered)


def _threshold_group_key(group_id: str | None) -> str:
    return "__ungrouped__" if group_id is None else str(group_id)


def _fuse_raw_ood_scores(ood_cal: OODCalibrator, raw_scores: Mapping[str, np.ndarray]) -> np.ndarray:
    oriented = ood_cal.orient_scores(dict(raw_scores))
    fused = np.zeros_like(next(iter(oriented.values())), dtype=np.float32)
    weights = dict(getattr(ood_cal, "weights", {}) or {})
    if not weights:
        score_names = list(getattr(ood_cal, "score_names", list(oriented)))
        weights = {name: 1.0 / max(len(score_names), 1) for name in score_names}
    for name in getattr(ood_cal, "score_names", list(oriented)):
        fused += float(weights.get(name, 0.0)) * np.asarray(oriented[name], dtype=np.float32)
    return fused.astype(np.float32)


def _score_quantiles(values: Sequence[float]) -> dict[str, float] | None:
    if not values:
        return None
    arr = np.asarray(values, dtype=np.float32)
    q50, q90, q95, q99 = np.quantile(arr, [0.50, 0.90, 0.95, 0.99])
    return {
        "q50": float(q50),
        "q90": float(q90),
        "q95": float(q95),
        "q99": float(q99),
    }


def _warmup_distribution_stats(
    values: Sequence[float],
    *,
    threshold_quantile: float,
    eps: float,
) -> dict[str, float] | None:
    if not values:
        return None
    arr = np.asarray(values, dtype=np.float32)
    q25, q50, q75 = np.quantile(arr, [0.25, 0.50, 0.75])
    quantiles = _score_quantiles(values) or {}
    return {
        "median": float(q50),
        "iqr": float(max(float(q75 - q25), float(eps))),
        "raw_threshold": float(np.quantile(arr, float(threshold_quantile))),
        **quantiles,
    }


class OnlineDetector:
    """Thin realtime shell over the existing UCS-OODID inference stack."""

    def __init__(
        self,
        artifact: dict,
        *,
        artifact_path: str | Path | None = None,
        device: str | None = None,
        label_col: str = "label",
        timestamp_col: str = "timestamp",
        record_id_col: str = "record_id",
        group_col: str | None = None,
        batch_size: int = 32,
        top_records: int = 10,
        buffer: StreamingWindowBuffer | None = None,
        strict_model_feature_mode: bool = True,
        threshold_quantile: float = 0.95,
    ) -> None:
        self.device = choose_device(device)
        self.artifact = artifact
        self.artifact_path = None if artifact_path is None else Path(artifact_path)
        artifact_seed = artifact.get("seed", artifact.get("run_config", {}).get("seed"))
        if artifact_seed is not None:
            set_seed(int(artifact_seed))

        self.label_col = str(label_col)
        self.timestamp_col = str(timestamp_col)
        self.record_id_col = str(record_id_col)
        self.batch_size = int(max(batch_size, 1))
        self.top_records = int(max(top_records, 1))
        strict_mode_from_artifact = artifact.get("strict_model_feature_mode")
        self.strict_model_feature_mode = (
            bool(strict_model_feature_mode)
            if strict_mode_from_artifact is None
            else bool(strict_mode_from_artifact)
        )
        self.threshold_quantile = float(threshold_quantile)
        if not 0.0 < self.threshold_quantile <= 1.0:
            raise ValueError("threshold_quantile must be within (0.0, 1.0].")
        self.score_mode = str(
            artifact.get(
                "simulation_score_mode",
                artifact.get("online_score_mode", artifact.get("calibration_config", {}).get("simulation_score_mode", "normalized")),
            )
            or "normalized"
        ).strip().lower()
        if self.score_mode not in {"raw", "normalized"}:
            raise ValueError("simulation_score_mode must be 'raw' or 'normalized'.")
        self.score_normalization_mode = str(
            artifact.get(
                "simulation_score_normalization_mode",
                artifact.get("online_score_normalization_mode", "median_iqr"),
            )
            or "median_iqr"
        ).strip().lower()
        if self.score_normalization_mode not in {"median_iqr", "none"}:
            raise ValueError("simulation_score_normalization_mode must be 'median_iqr' or 'none'.")
        self.normalized_threshold = float(
            artifact.get(
                "simulation_normalized_threshold",
                artifact.get("online_normalized_threshold", 1.5),
            )
        )
        self.score_threshold_mode = "per_uav_benign_warmup_raw_quantile"
        self.use_artifact_calibrator_decision = False
        self.score_direction = "higher_is_more_anomalous"
        self.score_eps = float(artifact.get("simulation_score_eps", 1e-6))

        self.pre, self.normalization_mode = resolve_preprocessor_from_artifact(artifact)
        self.feature_adapter = SimulationFeatureAdapter(
            artifact,
            artifact_path=self.artifact_path,
            strict_model_feature_mode=self.strict_model_feature_mode,
        )
        self.model_input_columns = list(self.feature_adapter.model_input_columns)
        self.feature_adapter_enabled = True
        self.group_embedding_config = resolve_group_embedding_config(artifact)
        self.deployment_backend = str(artifact.get("deployment_backend") or "ucs_oodid").strip().lower()
        if self.deployment_backend == "sklearn_tabular":
            self.model = None
            self.bank = None
            self.sklearn_baseline = artifact["sklearn_baseline"]
            self.sklearn_ood_threshold = float(artifact["ood_threshold"])
            self.sklearn_ood_score_name = str(artifact.get("ood_score_name", "uncertainty") or "uncertainty").strip().lower()
            self.ood_cal = None
            self.threshold_config = {
                "ood_threshold_mode": "global",
                "global_ood_threshold": float(self.sklearn_ood_threshold),
                "group_ood_thresholds": {},
                "group_threshold_strategy": "global_only",
            }
        else:
            self.sklearn_baseline = None
            self.sklearn_ood_threshold = float("nan")
            self.sklearn_ood_score_name = ""
            self.model = UCSOODID(**artifact["model_config"]).to(self.device)
            self.model.load_state_dict(artifact["model_state"])
            self.model.eval()
            bank_obj = artifact["prototype_bank"]
            self.bank = bank_obj if isinstance(bank_obj, PrototypeBank) else PrototypeBank.from_dict(bank_obj)
            cal_obj = artifact["ood_calibrator"]
            self.ood_cal = cal_obj if isinstance(cal_obj, OODCalibrator) else OODCalibrator.from_dict(cal_obj)
            self.threshold_config = resolve_ood_threshold_config(artifact, self.ood_cal)
            self.threshold_config.update(
                {
                    "score_mode": self.score_mode,
                    "threshold_mode": self.score_threshold_mode,
                    "score_direction": self.score_direction,
                    "score_normalization_mode": self.score_normalization_mode,
                    "normalized_threshold": float(self.normalized_threshold),
                }
            )
        self.class_names = list(artifact["class_names"])
        self.class_to_idx = dict(artifact["class_to_idx"])
        self.class_thresholds = np.asarray(artifact.get("class_thresholds", np.full(len(self.class_names), 0.5)), dtype=np.float32)
        self.temperature = float(artifact.get("temperature", 1.0))
        self.graph_config = dict(artifact.get("graph_config", {}))
        self.window_config = dict(artifact.get("window_config", {}))
        self.k_bank = int(artifact.get("calibration_config", {}).get("bank_k", 5))
        self._record_counter = 0
        self._warmup_scores_by_group: dict[str, list[float]] = {}
        self._warmup_stats_by_group: dict[str, dict[str, float]] = {}
        self._raw_thresholds_by_group: dict[str, float | None] = {}
        self._raw_threshold_sources_by_group: dict[str, str] = {}
        self._finalized_thresholds_by_group: dict[str, float | None] = {}
        self._finalized_threshold_sources_by_group: dict[str, str] = {}
        self._threshold_finalized_groups: set[str] = set()
        self._id_score_mean_by_group: dict[str, float | None] = {}
        self._id_score_median_by_group: dict[str, float | None] = {}
        self._raw_scores_by_group: dict[str, list[float]] = {}
        self._normalized_scores_by_group: dict[str, list[float]] = {}
        self._score_unique_values_by_group: dict[str, set[float]] = {}
        self._window_raw_ood_scores: list[dict[str, object]] = []
        self._window_normalized_ood_scores: list[dict[str, object]] = []
        self._inference_error_count = 0
        self._fallback_score_used_count = 0
        self._inference_errors_by_group: dict[str, int] = {}

        artifact_group_col = (
            artifact.get("group_col")
            or artifact.get("group_config", {}).get("group_col", "")
            or getattr(self.pre, "group_col", "")
            or ""
        ).strip()
        self.group_col = (group_col or "").strip() or artifact_group_col or None
        self.pre.group_col = self.group_col

        self.buffer = buffer or StreamingWindowBuffer(
            mode=self.window_config.get("mode", "count"),
            window_size=self.window_config.get("size", 32),
            stride=self.window_config.get("stride", 16),
            time_seconds=self.window_config.get("time_seconds", 2.0),
            adaptive_min_size=self.window_config.get("adaptive_min_size", 8),
            adaptive_max_size=self.window_config.get("adaptive_max_size", 64),
            timestamp_col=self.timestamp_col,
            record_id_col=self.record_id_col,
            group_col=self.group_col,
        )
        print(json.dumps({"model_input_columns": self.model_input_columns}, ensure_ascii=False))

    @classmethod
    def from_artifact_path(
        cls,
        artifact_path: str,
        *,
        device: str | None = None,
        label_col: str = "label",
        timestamp_col: str = "timestamp",
        record_id_col: str = "record_id",
        group_col: str | None = None,
        batch_size: int = 32,
        top_records: int = 10,
        buffer: StreamingWindowBuffer | None = None,
        strict_model_feature_mode: bool = True,
        threshold_quantile: float = 0.95,
    ) -> "OnlineDetector":
        artifact = load_artifact(artifact_path, map_location=choose_device(device))
        return cls(
            artifact,
            artifact_path=artifact_path,
            device=device,
            label_col=label_col,
            timestamp_col=timestamp_col,
            record_id_col=record_id_col,
            group_col=group_col,
            batch_size=batch_size,
            top_records=top_records,
            buffer=buffer,
            strict_model_feature_mode=strict_model_feature_mode,
            threshold_quantile=threshold_quantile,
        )

    def reset(self) -> None:
        self.buffer.reset()
        self._record_counter = 0
        self._warmup_scores_by_group.clear()
        self._warmup_stats_by_group.clear()
        self._raw_thresholds_by_group.clear()
        self._raw_threshold_sources_by_group.clear()
        self._finalized_thresholds_by_group.clear()
        self._finalized_threshold_sources_by_group.clear()
        self._threshold_finalized_groups.clear()
        self._id_score_mean_by_group.clear()
        self._id_score_median_by_group.clear()
        self._raw_scores_by_group.clear()
        self._normalized_scores_by_group.clear()
        self._score_unique_values_by_group.clear()
        self._window_raw_ood_scores.clear()
        self._window_normalized_ood_scores.clear()
        self._inference_error_count = 0
        self._fallback_score_used_count = 0
        self._inference_errors_by_group.clear()

    def consume_record(self, record: Mapping[str, object] | object) -> list[dict]:
        return self.consume_records([record])

    def consume_records(self, records: Sequence[Mapping[str, object] | object]) -> list[dict]:
        normalized = [self._normalize_stream_record(record) for record in records]
        windows = self.buffer.extend(normalized)
        if not windows:
            return []
        return self.detect_windows(windows)

    def _observe_scores_for_group(self, group_key: str, *, raw_score: float, normalized_score: float) -> None:
        self._raw_scores_by_group.setdefault(group_key, []).append(float(raw_score))
        self._normalized_scores_by_group.setdefault(group_key, []).append(float(normalized_score))

    def _benign_warmup_stats(self, group_key: str, *, allow_live: bool = True) -> dict[str, float] | None:
        finalized = self._warmup_stats_by_group.get(group_key)
        if finalized is not None:
            return dict(finalized)
        if not allow_live:
            return None
        return _warmup_distribution_stats(
            self._warmup_scores_by_group.get(group_key, []),
            threshold_quantile=self.threshold_quantile,
            eps=self.score_eps,
        )

    def _normalize_live_score(
        self,
        *,
        raw_score: float,
        benign_stats: dict[str, float] | None,
        artifact_normalized_score: float,
    ) -> float:
        if self.score_normalization_mode == "none":
            return float(raw_score)
        if benign_stats is None:
            return float(artifact_normalized_score)
        return float((float(raw_score) - float(benign_stats["median"])) / max(float(benign_stats["iqr"]), self.score_eps))

    def _selected_score_value(self, *, raw_score: float, normalized_score: float) -> float:
        return float(raw_score if self.score_mode == "raw" else normalized_score)

    def _finalize_group_threshold(self, group_key: str) -> tuple[float | None, str]:
        if group_key in self._threshold_finalized_groups:
            return (
                self._finalized_thresholds_by_group.get(group_key),
                str(self._finalized_threshold_sources_by_group.get(group_key, "unavailable")),
            )
        benign_stats = self._benign_warmup_stats(group_key, allow_live=True)
        raw_threshold: float | None = None
        raw_source = "no_benign_warmup"
        selected_threshold: float | None = None
        selected_source = "no_benign_warmup"
        if benign_stats is not None:
            self._warmup_stats_by_group[group_key] = dict(benign_stats)
            raw_threshold = float(benign_stats["raw_threshold"])
            raw_source = "warmup_raw_quantile"
            if self.score_mode == "raw":
                selected_threshold = raw_threshold
                selected_source = raw_source
            else:
                selected_threshold = float(self.normalized_threshold)
                selected_source = "normalized_static"
            warmup_scores = list(self._warmup_scores_by_group.get(group_key, []))
            normalized_warmup_scores = [
                self._normalize_live_score(
                    raw_score=float(score),
                    benign_stats=benign_stats,
                    artifact_normalized_score=0.0,
                )
                for score in warmup_scores
            ]
            selected_scores = warmup_scores if self.score_mode == "raw" else normalized_warmup_scores
            if selected_scores:
                arr = np.asarray(selected_scores, dtype=np.float32)
                self._id_score_mean_by_group[group_key] = float(arr.mean())
                self._id_score_median_by_group[group_key] = float(np.median(arr))
            else:
                self._id_score_mean_by_group[group_key] = None
                self._id_score_median_by_group[group_key] = None
        else:
            if self.score_mode == "normalized":
                selected_threshold = float(self.normalized_threshold)
                selected_source = "normalized_static_no_warmup"
            self._id_score_mean_by_group[group_key] = None
            self._id_score_median_by_group[group_key] = None
        self._raw_thresholds_by_group[group_key] = raw_threshold
        self._raw_threshold_sources_by_group[group_key] = raw_source
        self._finalized_thresholds_by_group[group_key] = selected_threshold
        self._finalized_threshold_sources_by_group[group_key] = selected_source
        self._threshold_finalized_groups.add(group_key)
        return selected_threshold, selected_source

    def _current_threshold_snapshot(
        self,
        *,
        group_id: str | None,
    ) -> dict[str, object]:
        group_key = _threshold_group_key(group_id)
        if group_key in self._threshold_finalized_groups:
            selected_threshold = self._finalized_thresholds_by_group.get(group_key)
            threshold_source = str(self._finalized_threshold_sources_by_group.get(group_key, "unavailable"))
            raw_threshold = self._raw_thresholds_by_group.get(group_key)
            raw_source = str(self._raw_threshold_sources_by_group.get(group_key, "no_benign_warmup"))
            benign_stats = self._benign_warmup_stats(group_key, allow_live=False)
            normalized_threshold = (
                selected_threshold
                if self.use_artifact_calibrator_decision and selected_threshold is not None
                else float(self.normalized_threshold)
            )
            return {
                "group_key": group_key,
                "raw_threshold": raw_threshold,
                "raw_threshold_source": raw_source,
                "normalized_threshold": normalized_threshold,
                "threshold": selected_threshold,
                "threshold_source": threshold_source,
                "warmup_collecting": False,
                "benign_stats": benign_stats,
            }
        benign_stats = self._benign_warmup_stats(group_key, allow_live=True)
        raw_threshold = None if benign_stats is None else float(benign_stats["raw_threshold"])
        raw_source = "warmup_raw_quantile_live" if raw_threshold is not None else "no_benign_warmup"
        selected_threshold = raw_threshold if self.score_mode == "raw" else float(self.normalized_threshold)
        selected_source = "warmup_collecting" if benign_stats is not None else ("normalized_static_no_warmup" if self.score_mode == "normalized" else "no_benign_warmup")
        return {
            "group_key": group_key,
            "raw_threshold": raw_threshold,
            "raw_threshold_source": raw_source,
            "normalized_threshold": float(self.normalized_threshold),
            "threshold": selected_threshold,
            "threshold_source": selected_source,
            "warmup_collecting": benign_stats is not None,
            "benign_stats": benign_stats,
        }

    def _resolve_window_scoring(
        self,
        *,
        group_id: str | None,
        raw_score: float,
        artifact_normalized_score: float,
        artifact_threshold: float | None,
        artifact_threshold_source: str | None,
        gt_is_benign: bool,
    ) -> dict[str, object]:
        group_key = _threshold_group_key(group_id)
        if self.use_artifact_calibrator_decision:
            normalized_score = float(artifact_normalized_score)
            selected_threshold = None if artifact_threshold is None else float(artifact_threshold)
            threshold_source = (
                "artifact_ood_calibrator"
                if artifact_threshold_source is None
                else str(artifact_threshold_source)
            )
            self._raw_thresholds_by_group[group_key] = None
            self._raw_threshold_sources_by_group[group_key] = "artifact_ood_calibrator"
            self._finalized_thresholds_by_group[group_key] = selected_threshold
            self._finalized_threshold_sources_by_group[group_key] = threshold_source
            self._threshold_finalized_groups.add(group_key)
            self._observe_scores_for_group(group_key, raw_score=float(raw_score), normalized_score=normalized_score)
            return {
                "group_key": group_key,
                "raw_threshold": None,
                "raw_threshold_source": "artifact_ood_calibrator",
                "normalized_threshold": selected_threshold,
                "threshold": selected_threshold,
                "threshold_source": threshold_source,
                "warmup_collecting": False,
                "benign_stats": None,
                "raw_score": float(raw_score),
                "normalized_score": normalized_score,
                "score": normalized_score,
                "alert_enabled": bool(selected_threshold is not None),
            }
        if gt_is_benign and group_key not in self._threshold_finalized_groups:
            self._warmup_scores_by_group.setdefault(group_key, []).append(float(raw_score))
        elif group_key not in self._threshold_finalized_groups:
            self._finalize_group_threshold(group_key)
        threshold_snapshot = self._current_threshold_snapshot(group_id=group_id)
        benign_stats = threshold_snapshot.get("benign_stats")
        normalized_score = self._normalize_live_score(
            raw_score=float(raw_score),
            benign_stats=benign_stats if isinstance(benign_stats, dict) else None,
            artifact_normalized_score=float(artifact_normalized_score),
        )
        self._observe_scores_for_group(group_key, raw_score=raw_score, normalized_score=normalized_score)
        selected_score = self._selected_score_value(raw_score=float(raw_score), normalized_score=float(normalized_score))
        return {
            **threshold_snapshot,
            "raw_score": float(raw_score),
            "normalized_score": float(normalized_score),
            "score": float(selected_score),
            "alert_enabled": bool((not bool(threshold_snapshot["warmup_collecting"])) and threshold_snapshot["threshold"] is not None),
        }

    def _finalize_pending_thresholds(self) -> None:
        group_keys = set(self._warmup_scores_by_group)
        group_keys.update(self._raw_scores_by_group)
        group_keys.update(self._normalized_scores_by_group)
        for group_key in sorted(group_keys):
            self._finalize_group_threshold(group_key)

    def _artifact_threshold_for_group(self, group_id: str | None) -> tuple[float, str]:
        groups = None if group_id is None else np.asarray([group_id], dtype=object)
        thresholds, sources = self.ood_cal.resolve_thresholds(phases=None, count=1, groups=groups)
        return float(thresholds[0]), str(sources[0])

    def _slice_windowed_data(self, windowed: WindowedData, indices: Sequence[int]) -> WindowedData:
        idx = np.asarray(indices, dtype=np.int64)
        return WindowedData(
            x=np.asarray(windowed.x[idx], dtype=np.float32),
            y=np.asarray(windowed.y[idx], dtype=np.float32),
            record_indices=np.asarray(windowed.record_indices[idx], dtype=np.int64),
            record_ids=np.asarray(windowed.record_ids[idx], dtype=object),
            ood=np.asarray(windowed.ood[idx], dtype=bool),
            record_y=None if windowed.record_y is None else np.asarray(windowed.record_y[idx]),
            valid_mask=None if windowed.valid_mask is None else np.asarray(windowed.valid_mask[idx], dtype=bool),
            timestamps=None if windowed.timestamps is None else np.asarray(windowed.timestamps[idx]),
            raw_starts=None if windowed.raw_starts is None else np.asarray(windowed.raw_starts[idx]),
            group_ids=None if windowed.group_ids is None else np.asarray(windowed.group_ids[idx], dtype=object),
            group_index=None if windowed.group_index is None else np.asarray(windowed.group_index[idx], dtype=np.int64),
        )

    def _window_input_is_compatible(self, stats: Mapping[str, object]) -> bool:
        missing_ratio = float(stats.get("missing_feature_ratio", 0.0) or 0.0)
        raw_input_std = float(stats.get("raw_input_std", 0.0) or 0.0)
        return not (missing_ratio >= 0.999 or (missing_ratio >= 0.95 and raw_input_std <= 1e-8))

    def _record_window_score(
        self,
        *,
        group_id: str | None,
        window_id: int,
        raw_ood_score: float | None,
        normalized_ood_score: float | None,
        decision_score: float | None,
        inference_error: bool = False,
    ) -> None:
        group_key = _threshold_group_key(group_id)
        self._window_raw_ood_scores.append(
            {
                "window_id": int(window_id),
                "group_id": group_id,
                "raw_ood_score": None if raw_ood_score is None else float(raw_ood_score),
                "inference_error": bool(inference_error),
            }
        )
        self._window_normalized_ood_scores.append(
            {
                "window_id": int(window_id),
                "group_id": group_id,
                "normalized_ood_score": None if normalized_ood_score is None else float(normalized_ood_score),
                "inference_error": bool(inference_error),
            }
        )
        if decision_score is not None:
            self._score_unique_values_by_group.setdefault(group_key, set()).add(round(float(decision_score), 8))

    def _build_inference_error_row(
        self,
        *,
        window: BufferedWindow,
        group_id: str | None,
        valid_records: Sequence[Mapping[str, object]],
        window_stats: Mapping[str, object],
        reason: str,
    ) -> dict[str, object]:
        group_key = _threshold_group_key(group_id)
        threshold_snapshot = self._current_threshold_snapshot(group_id=group_id)
        self._inference_error_count += 1
        self._inference_errors_by_group[group_key] = self._inference_errors_by_group.get(group_key, 0) + 1
        self._record_window_score(
            group_id=group_id,
            window_id=int(window.window_id),
            raw_ood_score=None,
            normalized_ood_score=None,
            decision_score=None,
            inference_error=True,
        )
        ground_truth = self._ground_truth_payload(valid_records, {})
        ground_truth_is_ood = bool(ground_truth.get("is_ood", False))
        window_partition = "ood" if ground_truth_is_ood else ("attack" if bool(ground_truth.get("attack_active", False)) else "benign")
        valid_ids = [str(record.get(self.record_id_col)) for record in valid_records if record.get(self.record_id_col) is not None]
        selected_threshold = threshold_snapshot.get("threshold")
        return {
            "window_id": int(window.window_id),
            "group_id": group_id,
            "record_ids": valid_ids,
            "known_pred_labels": [],
            "known_pred_probs": {},
            "known_attack_alert": False,
            "known_attack_pred_labels": [],
            "raw_ood_score": None,
            "normalized_ood_score": None,
            "ood_score": None,
            "raw_threshold": threshold_snapshot.get("raw_threshold"),
            "normalized_threshold": float(threshold_snapshot.get("normalized_threshold", self.normalized_threshold)),
            "threshold": None if selected_threshold is None else float(selected_threshold),
            "ood_threshold": None if selected_threshold is None else float(selected_threshold),
            "threshold_source": str(threshold_snapshot.get("threshold_source", "unavailable")),
            "ood_threshold_source": str(threshold_snapshot.get("threshold_source", "unavailable")),
            "threshold_mode": self.score_threshold_mode,
            "score_mode": self.score_mode,
            "score_direction": self.score_direction,
            "ood_alert": False,
            "is_ood": False,
            "has_alert": False,
            "alert_level": "normal",
            "alert_reason": "none",
            "top_suspicious_records": [],
            "raw_scores": {},
            "normalized_scores": {},
            "dominant_ood_source": "inference_error",
            "window_mode": window.mode,
            "window_target_length": int(window.target_length),
            "window_valid_count": int(window.valid_count),
            "window_start": float(window.start_marker),
            "window_end": float(window.end_marker),
            "timestamp_range": self._timestamp_range(valid_records),
            "ground_truth": ground_truth,
            "gt_attack_active": bool(ground_truth.get("attack_active", False)),
            "gt_attack_type": collapse_attack_types(ground_truth.get("attack_types", [])),
            "ground_truth_is_ood": ground_truth_is_ood,
            "window_partition": window_partition,
            "false_alert": False,
            "inference_error": True,
            "inference_error_reason": str(reason),
            "window_missing_feature_ratio": float(window_stats.get("missing_feature_ratio", 0.0) or 0.0),
            "window_zero_feature_ratio": float(window_stats.get("zero_feature_ratio", 0.0) or 0.0),
            "window_input_mean": float(window_stats.get("input_mean", 0.0) or 0.0),
            "window_input_std": float(window_stats.get("input_std", 0.0) or 0.0),
        }

    def simulation_diagnostics(self) -> dict[str, object]:
        self._finalize_pending_thresholds()
        adapter_diagnostics = self.feature_adapter.diagnostics()
        score_groups = set(self._score_unique_values_by_group)
        score_groups.update(self._inference_errors_by_group)
        score_groups.update(self._threshold_finalized_groups)
        score_groups.update(self._warmup_scores_by_group)
        score_groups.update(self._raw_scores_by_group)
        score_groups.update(self._normalized_scores_by_group)
        effective_normalized_threshold = None if self.use_artifact_calibrator_decision else float(self.normalized_threshold)
        per_uav_normalized_threshold = (
            {
                group_id: float(value)
                for group_id, value in sorted(self._finalized_thresholds_by_group.items())
                if value is not None
            }
            if self.use_artifact_calibrator_decision
            else {
                group_id: float(self.normalized_threshold)
                for group_id in sorted(score_groups)
            }
        )

        def _quantile_map(score_map: Mapping[str, Sequence[float]], key: str) -> dict[str, float]:
            payload: dict[str, float] = {}
            for group_id, values in sorted(score_map.items()):
                stats = _score_quantiles(values)
                if stats is not None:
                    payload[group_id] = float(stats[key])
            return payload

        def _warmup_quantile_map(key: str) -> dict[str, float]:
            payload: dict[str, float] = {}
            for group_id in sorted(score_groups):
                stats = self._benign_warmup_stats(group_id, allow_live=False)
                if stats is not None:
                    payload[group_id] = float(stats[key])
            return payload

        return {
            **adapter_diagnostics,
            "score_mode": self.score_mode,
            "threshold_mode": self.score_threshold_mode,
            "score_direction": self.score_direction,
            "score_normalization_mode": self.score_normalization_mode,
            "normalized_threshold": effective_normalized_threshold,
            "per_uav_threshold": {
                group_id: float(value)
                for group_id, value in sorted(self._finalized_thresholds_by_group.items())
                if value is not None
            },
            "per_uav_threshold_source": dict(sorted(self._finalized_threshold_sources_by_group.items())),
            "per_uav_raw_threshold": {
                group_id: float(value)
                for group_id, value in sorted(self._raw_thresholds_by_group.items())
                if value is not None
            },
            "per_uav_normalized_threshold": per_uav_normalized_threshold,
            "per_uav_id_score_mean": dict(sorted(self._id_score_mean_by_group.items())),
            "per_uav_id_score_median": dict(sorted(self._id_score_median_by_group.items())),
            "per_uav_benign_warmup_window_count": {
                group_id: int(len(values))
                for group_id, values in sorted(self._warmup_scores_by_group.items())
            },
            "per_uav_benign_score_q50": _warmup_quantile_map("q50"),
            "per_uav_benign_score_q90": _warmup_quantile_map("q90"),
            "per_uav_benign_score_q95": _warmup_quantile_map("q95"),
            "per_uav_benign_score_q99": _warmup_quantile_map("q99"),
            "per_uav_raw_score_q50": _quantile_map(self._raw_scores_by_group, "q50"),
            "per_uav_raw_score_q90": _quantile_map(self._raw_scores_by_group, "q90"),
            "per_uav_raw_score_q95": _quantile_map(self._raw_scores_by_group, "q95"),
            "per_uav_raw_score_q99": _quantile_map(self._raw_scores_by_group, "q99"),
            "per_uav_normalized_score_q50": _quantile_map(self._normalized_scores_by_group, "q50"),
            "per_uav_normalized_score_q90": _quantile_map(self._normalized_scores_by_group, "q90"),
            "per_uav_normalized_score_q95": _quantile_map(self._normalized_scores_by_group, "q95"),
            "per_uav_normalized_score_q99": _quantile_map(self._normalized_scores_by_group, "q99"),
            "per_uav_raw_ood_score_mean": {
                group_id: float(np.mean(values))
                for group_id, values in sorted(self._raw_scores_by_group.items())
                if values
            },
            "per_uav_normalized_ood_score_mean": {
                group_id: float(np.mean(values))
                for group_id, values in sorted(self._normalized_scores_by_group.items())
                if values
            },
            "per_window_raw_ood_score": list(self._window_raw_ood_scores),
            "per_window_normalized_ood_score": list(self._window_normalized_ood_scores),
            "per_uav_score_unique_count": {
                group_id: int(len(self._score_unique_values_by_group.get(group_id, set()))) for group_id in sorted(score_groups)
            },
            "per_uav_inference_error_count": {
                group_id: int(self._inference_errors_by_group.get(group_id, 0)) for group_id in sorted(score_groups)
            },
            "inference_error_count": int(self._inference_error_count),
            "fallback_score_used_count": int(self._fallback_score_used_count),
            "threshold_quantile": float(self.threshold_quantile),
        }

    def _sklearn_ood_score_array(self, probs: np.ndarray) -> np.ndarray:
        pack = compute_baseline_uncertainty_scores(probs)
        if self.sklearn_ood_score_name in {"uncertainty", "conf"}:
            return np.asarray(pack["conf"], dtype=np.float32)
        if self.sklearn_ood_score_name in {"energy"}:
            return np.asarray(pack["energy"], dtype=np.float32)
        return np.asarray(pack["conf"], dtype=np.float32)

    def _detect_windows_sklearn(self, windows: Sequence[BufferedWindow]) -> list[dict]:
        non_empty = [window for window in windows if window.valid_count > 0]
        if not non_empty:
            return []
        frame, windowed, valid_records_by_window, window_input_stats = self._build_windowed_batch(non_empty)
        valid_indices = [idx for idx, stats in enumerate(window_input_stats) if self._window_input_is_compatible(stats)]
        valid_index_set = set(valid_indices)
        invalid_indices = [idx for idx in range(len(non_empty)) if idx not in valid_index_set]
        rows_by_index: dict[int, dict[str, object]] = {}
        for idx in invalid_indices:
            window = non_empty[idx]
            rows_by_index[idx] = self._build_inference_error_row(
                window=window,
                group_id=_normalize_group_id(window.group_id),
                valid_records=valid_records_by_window[idx],
                window_stats=window_input_stats[idx],
                reason="degenerate_model_input",
            )
        if valid_indices:
            valid_windowed = self._slice_windowed_data(windowed, valid_indices)
            probs = self.sklearn_baseline.predict_proba(valid_windowed.x)
            raw_fused_scores = self._sklearn_ood_score_array(probs)
            predicted = probs >= self.class_thresholds.reshape(1, -1)
            for local_idx, original_idx in enumerate(valid_indices):
                window = non_empty[original_idx]
                labels = [self.class_names[j] for j, flag in enumerate(predicted[local_idx]) if flag]
                prob_map = {self.class_names[j]: float(probs[local_idx, j]) for j in range(len(self.class_names))}
                group_id = _normalize_group_id(window.group_id)
                valid_ids = [str(record_id) for record_id in valid_windowed.record_ids[local_idx].tolist()]
                raw_ood_score = float(raw_fused_scores[local_idx])
                selected_threshold = float(self.sklearn_ood_threshold)
                normalized_ood_score = raw_ood_score
                decision_score = raw_ood_score
                ground_truth = self._ground_truth_payload(valid_records_by_window[original_idx], {})
                self._record_window_score(
                    group_id=group_id,
                    window_id=int(window.window_id),
                    raw_ood_score=raw_ood_score,
                    normalized_ood_score=normalized_ood_score,
                    decision_score=decision_score,
                    inference_error=False,
                )
                threshold_source = "sklearn_val_ood_quantile"
                is_ood = bool(decision_score > selected_threshold)
                margin = float(decision_score - selected_threshold)
                benign_label_names = {"benign", "normal", "normal traffic"}
                known_attack_labels = [
                    str(label)
                    for label in labels
                    if str(label).strip() and str(label).strip().lower() not in benign_label_names
                ]
                known_attack_alert = bool(known_attack_labels)
                has_alert = bool(known_attack_alert or is_ood)
                if is_ood:
                    alert_level = alert_level_for_margin(margin)
                elif known_attack_alert:
                    alert_level = "warning"
                else:
                    alert_level = "normal"
                false_alert = bool((not bool(ground_truth.get("attack_active", False))) and has_alert)
                if known_attack_alert and is_ood:
                    alert_reason = "known_attack+ood"
                elif known_attack_alert:
                    alert_reason = "known_attack"
                elif is_ood:
                    alert_reason = "ood"
                else:
                    alert_reason = "none"
                ground_truth_is_ood = bool(ground_truth.get("is_ood", False))
                window_partition = (
                    "ood" if ground_truth_is_ood else ("attack" if bool(ground_truth.get("attack_active", False)) else "benign")
                )
                rows_by_index[original_idx] = {
                    "window_id": int(window.window_id),
                    "group_id": group_id,
                    "record_ids": valid_ids,
                    "known_pred_labels": labels,
                    "known_pred_probs": prob_map,
                    "known_attack_alert": known_attack_alert,
                    "known_attack_pred_labels": known_attack_labels,
                    "raw_ood_score": raw_ood_score,
                    "normalized_ood_score": normalized_ood_score,
                    "ood_score": decision_score,
                    "raw_threshold": None,
                    "normalized_threshold": float(self.normalized_threshold),
                    "threshold": selected_threshold,
                    "ood_threshold": selected_threshold,
                    "threshold_source": threshold_source,
                    "ood_threshold_source": threshold_source,
                    "threshold_mode": "sklearn_global_threshold",
                    "score_mode": self.score_mode,
                    "score_direction": self.score_direction,
                    "ood_alert": is_ood,
                    "is_ood": is_ood,
                    "has_alert": has_alert,
                    "alert_level": alert_level,
                    "alert_reason": alert_reason,
                    "top_suspicious_records": [],
                    "raw_scores": {"sklearn_ood": raw_ood_score},
                    "normalized_scores": {"sklearn_ood": normalized_ood_score},
                    "dominant_ood_source": "sklearn_ood",
                    "window_mode": window.mode,
                    "window_target_length": int(window.target_length),
                    "window_valid_count": int(window.valid_count),
                    "window_start": float(window.start_marker),
                    "window_end": float(window.end_marker),
                    "timestamp_range": self._timestamp_range(valid_records_by_window[original_idx]),
                    "ground_truth": ground_truth,
                    "gt_attack_active": bool(ground_truth.get("attack_active", False)),
                    "gt_attack_type": collapse_attack_types(ground_truth.get("attack_types", [])),
                    "ground_truth_is_ood": ground_truth_is_ood,
                    "window_partition": window_partition,
                    "false_alert": false_alert,
                    "inference_error": False,
                    "inference_error_reason": None,
                    "window_missing_feature_ratio": float(window_input_stats[original_idx]["missing_feature_ratio"]),
                    "window_zero_feature_ratio": float(window_input_stats[original_idx]["zero_feature_ratio"]),
                    "window_input_mean": float(window_input_stats[original_idx]["input_mean"]),
                    "window_input_std": float(window_input_stats[original_idx]["input_std"]),
                }
        return [rows_by_index[idx] for idx in range(len(non_empty))]

    def detect_windows(self, windows: Sequence[BufferedWindow]) -> list[dict]:
        if getattr(self, "deployment_backend", "") == "sklearn_tabular":
            return self._detect_windows_sklearn(windows)
        non_empty = [window for window in windows if window.valid_count > 0]
        if not non_empty:
            return []

        frame, windowed, valid_records_by_window, window_input_stats = self._build_windowed_batch(non_empty)
        valid_indices = [idx for idx, stats in enumerate(window_input_stats) if self._window_input_is_compatible(stats)]
        valid_index_set = set(valid_indices)
        invalid_indices = [idx for idx in range(len(non_empty)) if idx not in valid_index_set]

        rows_by_index: dict[int, dict[str, object]] = {}
        for idx in invalid_indices:
            window = non_empty[idx]
            rows_by_index[idx] = self._build_inference_error_row(
                window=window,
                group_id=_normalize_group_id(window.group_id),
                valid_records=valid_records_by_window[idx],
                window_stats=window_input_stats[idx],
                reason="degenerate_model_input",
            )

        if valid_indices:
            valid_windowed = self._slice_windowed_data(windowed, valid_indices)
            attach_group_embedding_indices(valid_windowed, self.artifact)
            outputs = collect_model_outputs(
                self.model,
                valid_windowed,
                self.graph_config,
                batch_size=min(self.batch_size, max(len(valid_windowed), 1)),
                device=self.device,
                temperature=self.temperature,
            )
            raw_scores = compute_raw_ood_scores(
                outputs["logits"],
                outputs["probs"],
                outputs["embeddings"],
                self.bank,
                temperature=self.temperature,
                k_bank=self.k_bank,
            )
            raw_fused_scores = _fuse_raw_ood_scores(self.ood_cal, raw_scores)
            phases = None
            if self.ood_cal.phase_aware_enabled and self.ood_cal.phase_column and self.ood_cal.phase_column in frame.columns:
                phases = window_phase_labels(frame, valid_windowed, self.ood_cal.phase_column)
            transformed = self.ood_cal.transform(raw_scores, phases=phases, groups=valid_windowed.group_ids)
            predicted = outputs["probs"] >= self.class_thresholds.reshape(1, -1)
            output_mask = outputs.get("mask", valid_windowed.valid_mask)

            for local_idx, original_idx in enumerate(valid_indices):
                window = non_empty[original_idx]
                labels = [self.class_names[j] for j, flag in enumerate(predicted[local_idx]) if flag]
                prob_map = {
                    self.class_names[j]: float(outputs["probs"][local_idx, j]) for j in range(len(self.class_names))
                }
                group_id = _normalize_group_id(window.group_id)
                valid_ids = [
                    str(record_id)
                    for record_id, ok in zip(valid_windowed.record_ids[local_idx].tolist(), output_mask[local_idx].tolist())
                    if ok
                ]
                preliminary_ground_truth = self._ground_truth_payload(valid_records_by_window[original_idx], {})
                raw_ood_score = float(raw_fused_scores[local_idx])
                scoring = self._resolve_window_scoring(
                    group_id=group_id,
                    raw_score=raw_ood_score,
                    artifact_normalized_score=float(transformed["fused"][local_idx]),
                    artifact_threshold=float(transformed["thresholds"][local_idx]),
                    artifact_threshold_source=str(transformed["threshold_sources"][local_idx]),
                    gt_is_benign=bool(
                        (not bool(preliminary_ground_truth.get("attack_active", False)))
                        and (not bool(preliminary_ground_truth.get("is_ood", False)))
                    ),
                )
                normalized_ood_score = float(scoring["normalized_score"])
                decision_score = float(scoring["score"])
                selected_threshold = scoring.get("threshold")
                ranking_threshold = float("inf") if selected_threshold is None else float(selected_threshold)
                record_scores, suspicious = build_record_level_suspiciousness_ranking(
                    valid_windowed.record_ids[local_idx : local_idx + 1],
                    outputs["attention"][local_idx : local_idx + 1],
                    np.asarray([decision_score], dtype=np.float32),
                    np.asarray([ranking_threshold], dtype=np.float32),
                    valid_mask=output_mask[local_idx : local_idx + 1],
                    window_ids=np.asarray([window.window_id], dtype=np.int64),
                )
                label_lookup = {
                    str(record[self.record_id_col]): str(record[self.label_col])
                    for record in valid_records_by_window[original_idx]
                    if self.label_col in record and record.get(self.record_id_col) is not None
                }
                for row in suspicious:
                    row["label"] = label_lookup.get(row["record_id"])
                ground_truth = self._ground_truth_payload(valid_records_by_window[original_idx], record_scores)
                self._record_window_score(
                    group_id=group_id,
                    window_id=int(window.window_id),
                    raw_ood_score=raw_ood_score,
                    normalized_ood_score=normalized_ood_score,
                    decision_score=decision_score,
                    inference_error=False,
                )
                alert_enabled = bool(scoring["alert_enabled"])
                threshold_source = str(scoring.get("threshold_source", "unavailable"))
                is_ood = bool(alert_enabled and selected_threshold is not None and (decision_score > float(selected_threshold)))
                margin = float("-inf") if selected_threshold is None else float(decision_score - float(selected_threshold))
                benign_label_names = {"benign", "normal", "normal traffic"}
                known_attack_labels = [
                    str(label)
                    for label in labels
                    if str(label).strip() and str(label).strip().lower() not in benign_label_names
                ]
                known_attack_alert = bool(known_attack_labels)
                has_alert = bool(known_attack_alert or is_ood)

                if is_ood:
                    alert_level = alert_level_for_margin(margin)
                elif known_attack_alert:
                    alert_level = "warning"
                else:
                    alert_level = "normal"

                false_alert = bool((not bool(ground_truth.get("attack_active", False))) and has_alert)

                if known_attack_alert and is_ood:
                    alert_reason = "known_attack+ood"
                elif known_attack_alert:
                    alert_reason = "known_attack"
                elif is_ood:
                    alert_reason = "ood"
                else:
                    alert_reason = "none"
                ground_truth_is_ood = bool(ground_truth.get("is_ood", False))
                window_partition = (
                    "ood" if ground_truth_is_ood else ("attack" if bool(ground_truth.get("attack_active", False)) else "benign")
                )
                rows_by_index[original_idx] = {
                    "window_id": int(window.window_id),
                    "group_id": group_id,
                    "record_ids": valid_ids,
                    "known_pred_labels": labels,
                    "known_pred_probs": prob_map,
                    "known_attack_alert": known_attack_alert,
                    "known_attack_pred_labels": known_attack_labels,
                    "raw_ood_score": raw_ood_score,
                    "normalized_ood_score": normalized_ood_score,
                    "ood_score": decision_score,
                    "raw_threshold": scoring.get("raw_threshold"),
                    "normalized_threshold": float(scoring.get("normalized_threshold", self.normalized_threshold)),
                    "threshold": None if selected_threshold is None else float(selected_threshold),
                    "ood_threshold": None if selected_threshold is None else float(selected_threshold),
                    "threshold_source": str(threshold_source),
                    "ood_threshold_source": str(threshold_source),
                    "threshold_mode": self.score_threshold_mode,
                    "score_mode": self.score_mode,
                    "score_direction": self.score_direction,
                    "ood_alert": is_ood,
                    "is_ood": is_ood,
                    "has_alert": has_alert,
                    "alert_level": alert_level,
                    "alert_reason": alert_reason,
                    "top_suspicious_records": suspicious[: self.top_records],
                    "raw_scores": {name: float(raw_scores[name][local_idx]) for name in raw_scores},
                    "normalized_scores": {
                        name: float(transformed["normalized"][name][local_idx]) for name in transformed["normalized"]
                    },
                    "dominant_ood_source": dominant_ood_source(transformed["normalized"], local_idx),
                    "window_mode": window.mode,
                    "window_target_length": int(window.target_length),
                    "window_valid_count": int(window.valid_count),
                    "window_start": float(window.start_marker),
                    "window_end": float(window.end_marker),
                    "timestamp_range": self._timestamp_range(valid_records_by_window[original_idx]),
                    "ground_truth": ground_truth,
                    "gt_attack_active": bool(ground_truth.get("attack_active", False)),
                    "gt_attack_type": collapse_attack_types(ground_truth.get("attack_types", [])),
                    "ground_truth_is_ood": ground_truth_is_ood,
                    "window_partition": window_partition,
                    "false_alert": false_alert,
                    "inference_error": False,
                    "inference_error_reason": None,
                    "window_missing_feature_ratio": float(window_input_stats[original_idx]["missing_feature_ratio"]),
                    "window_zero_feature_ratio": float(window_input_stats[original_idx]["zero_feature_ratio"]),
                    "window_input_mean": float(window_input_stats[original_idx]["input_mean"]),
                    "window_input_std": float(window_input_stats[original_idx]["input_std"]),
                }

        return [rows_by_index[idx] for idx in range(len(non_empty))]

    def _normalize_stream_record(self, record: Mapping[str, object] | object) -> dict[str, object]:
        if isinstance(record, Mapping):
            payload = dict(record)
        elif hasattr(record, "to_dict"):
            payload = dict(record.to_dict())
        elif is_dataclass(record):
            payload = asdict(record)
        else:
            raise TypeError(f"Unsupported streaming record type: {type(record)!r}")

        if self.group_col and self.group_col not in payload:
            raise ValueError(f"Streaming record is missing required group_col {self.group_col!r}")
        if self.timestamp_col not in payload:
            raise ValueError(f"Streaming record is missing required timestamp_col {self.timestamp_col!r}")
        if self.record_id_col not in payload or payload.get(self.record_id_col) in {None, ""}:
            payload[self.record_id_col] = self._default_record_id(payload)
        if self.label_col not in payload and ("attack_active" in payload or "attack_type" in payload):
            active = bool(payload.get("attack_active", False))
            attack_type = str(payload.get("attack_type", "benign") or "benign")
            payload[self.label_col] = "benign" if (not active or attack_type == "benign") else attack_type
        self._record_counter += 1
        return payload

    def _default_record_id(self, payload: Mapping[str, object]) -> str:
        group_id = _normalize_group_id(payload.get(self.group_col)) if self.group_col else None
        group_prefix = group_id or "stream"
        timestamp = float(payload.get(self.timestamp_col, 0.0))
        return f"{group_prefix}:{timestamp:.6f}:{self._record_counter}"

    def _build_windowed_batch(
        self,
        windows: Sequence[BufferedWindow],
    ) -> tuple[pd.DataFrame, WindowedData, list[list[dict[str, object]]], list[dict[str, float]]]:
        flat_records: list[dict[str, object]] = []
        ranges: list[tuple[int, int]] = []
        valid_records_by_window: list[list[dict[str, object]]] = []
        for window in windows:
            valid_records = [dict(record) for record in window.records]
            start = len(flat_records)
            flat_records.extend(valid_records)
            end = len(flat_records)
            ranges.append((start, end))
            valid_records_by_window.append(valid_records)

        frame = pd.DataFrame(flat_records)
        frame = attach_parsed_labels(frame, self.label_col) if self.label_col in frame.columns else frame.assign(__labels=[[] for _ in range(len(frame))])
        self._validate_frame(frame)
        features = self.feature_adapter.transform(frame)
        row_diagnostics = self.feature_adapter.last_row_diagnostics

        xs: list[np.ndarray] = []
        ys: list[np.ndarray] = []
        record_indices: list[np.ndarray] = []
        record_ids: list[np.ndarray] = []
        ood_flags: list[bool] = []
        masks: list[np.ndarray] = []
        group_ids: list[str | None] = []
        window_input_stats: list[dict[str, float]] = []

        for window, (start, end) in zip(windows, ranges):
            valid_len = end - start
            if valid_len <= 0:
                continue
            idx = np.arange(start, end, dtype=np.int64)
            valid_mask = np.ones(valid_len, dtype=bool)
            if valid_len < window.target_length:
                pad = np.full(window.target_length - valid_len, idx[-1], dtype=np.int64)
                idx = np.concatenate([idx, pad])
                valid_mask = np.concatenate([valid_mask, np.zeros(len(pad), dtype=bool)])
            elif valid_len > window.target_length:
                idx = idx[: window.target_length]
                valid_mask = valid_mask[: window.target_length]
            xs.append(features[idx])
            labels = list(frame.iloc[idx]["__labels"])
            y, ood_flag = self._window_label(labels, valid_mask)
            ys.append(y)
            record_indices.append(idx)
            record_ids.append(frame.iloc[idx][self.record_id_col].astype(str).to_numpy())
            ood_flags.append(ood_flag)
            masks.append(valid_mask)
            group_ids.append(_normalize_group_id(window.group_id))
            valid_indices = idx[valid_mask]
            if len(valid_indices) > 0:
                window_input_stats.append(
                    {
                        "missing_feature_ratio": float(
                            np.mean(np.asarray(row_diagnostics["missing_feature_ratio"], dtype=np.float32)[valid_indices])
                        ),
                        "zero_feature_ratio": float(
                            np.mean(np.asarray(row_diagnostics["zero_feature_ratio"], dtype=np.float32)[valid_indices])
                        ),
                        "raw_input_mean": float(
                            np.mean(np.asarray(row_diagnostics["raw_input_mean"], dtype=np.float32)[valid_indices])
                        ),
                        "raw_input_std": float(
                            np.mean(np.asarray(row_diagnostics["raw_input_std"], dtype=np.float32)[valid_indices])
                        ),
                        "input_mean": float(
                            np.mean(np.asarray(row_diagnostics["input_mean"], dtype=np.float32)[valid_indices])
                        ),
                        "input_std": float(
                            np.mean(np.asarray(row_diagnostics["input_std"], dtype=np.float32)[valid_indices])
                        ),
                    }
                )
            else:
                window_input_stats.append(
                    {
                        "missing_feature_ratio": 0.0,
                        "zero_feature_ratio": 0.0,
                        "raw_input_mean": 0.0,
                        "raw_input_std": 0.0,
                        "input_mean": 0.0,
                        "input_std": 0.0,
                    }
                )

        use_group_ids = any(group_id is not None for group_id in group_ids)
        windowed = WindowedData(
            x=np.asarray(xs, dtype=np.float32),
            y=np.asarray(ys, dtype=np.float32),
            record_indices=np.asarray(record_indices, dtype=np.int64),
            record_ids=np.asarray(record_ids, dtype=object),
            ood=np.asarray(ood_flags, dtype=bool),
            valid_mask=np.asarray(masks, dtype=bool),
            group_ids=np.asarray(group_ids, dtype=object) if use_group_ids else None,
        )
        return frame, windowed, valid_records_by_window, window_input_stats

    def _validate_frame(self, frame: pd.DataFrame) -> None:
        if self.normalization_mode == "group" and (not self.group_col or self.group_col not in frame.columns):
            raise ValueError("artifact was trained with group normalization but streaming input does not contain the required group_col")
        if self.group_embedding_config["enabled"] and (not self.group_col or self.group_col not in frame.columns):
            raise ValueError("artifact was trained with group embedding but streaming input does not contain the required group_col")
        if self.group_col and self.group_col not in frame.columns:
            raise ValueError(f"group_col {self.group_col!r} not found in streaming input")

    def _window_label(self, labels: Sequence[list[str]], valid_mask: Sequence[bool]) -> tuple[np.ndarray, bool]:
        y = np.zeros(len(self.class_to_idx), dtype=np.float32)
        ood = False
        for is_valid, labs in zip(valid_mask, labels):
            if not bool(is_valid):
                continue
            for label in labs:
                if label in self.class_to_idx:
                    y[self.class_to_idx[label]] = 1.0
                else:
                    ood = True
        return y, ood

    def _timestamp_range(self, records: Sequence[Mapping[str, object]]) -> dict[str, float | None]:
        timestamps = [float(record[self.timestamp_col]) for record in records if self.timestamp_col in record]
        if not timestamps:
            return {"start": None, "end": None}
        return {"start": float(min(timestamps)), "end": float(max(timestamps))}

    def _ground_truth_payload(
        self,
        records: Sequence[Mapping[str, object]],
        record_scores: Mapping[str, float],
    ) -> dict:
        labels: list[str] = []
        attack_types: set[str] = set()
        truth_records: list[dict] = []
        attack_active = False
        window_is_ood = False
        benign_record_count = 0
        attack_record_count = 0
        ood_record_count = 0
        for record in records:
            record_labels = parse_label_cell(record.get(self.label_col)) if self.label_col in record else []
            labels.extend(record_labels)
            current_attack_type = str(record.get("gt_attack_type", record.get("attack_type", "benign")) or "benign")
            current_attack_active = bool(record.get("gt_attack_active", record.get("attack_active", False)))
            current_truth_is_ood_raw = record.get("ground_truth_is_ood")
            if current_truth_is_ood_raw is None:
                current_truth_is_ood = any(label not in self.class_to_idx for label in record_labels if str(label).strip())
            else:
                current_truth_is_ood = bool(current_truth_is_ood_raw)
            attack_active = attack_active or current_attack_active
            window_is_ood = window_is_ood or current_truth_is_ood
            if current_attack_active and current_attack_type != "benign":
                attack_types.add(current_attack_type)
            if current_truth_is_ood:
                ood_record_count += 1
            elif current_attack_active:
                attack_record_count += 1
            else:
                benign_record_count += 1
            truth_records.append(
                {
                    "record_id": str(record.get(self.record_id_col)),
                    "label": record.get(self.label_col),
                    "attack_active": current_attack_active,
                    "attack_type": current_attack_type,
                    "is_ood": current_truth_is_ood,
                    "suspiciousness": float(record_scores.get(str(record.get(self.record_id_col)), 0.0)),
                }
            )
        unique_labels = sorted({str(label) for label in labels if str(label).strip()})
        return {
            "labels": unique_labels,
            "is_ood": bool(window_is_ood or any(label not in self.class_to_idx for label in unique_labels)),
            "attack_active": attack_active,
            "attack_types": sorted(attack_types),
            "benign_record_count": int(benign_record_count),
            "attack_record_count": int(attack_record_count),
            "ood_record_count": int(ood_record_count),
            "records": truth_records,
        }
