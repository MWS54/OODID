from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from ..preprocessing import normalize_group_id

DEFAULT_FEATURE_COLUMNS_FILENAME = "feature_columns.json"
FORBIDDEN_SIMULATION_MODEL_PREFIXES: tuple[str, ...] = ("sim_", "gt_", "source_")

FORBIDDEN_SIMULATION_MODEL_COLUMNS: frozenset[str] = frozenset(
    {
        "label",
        "uav_id",
        "dataset_name",
        "split",
        "timestamp",
        "record_id",
        "mission_phase",
        "mission_context",
        "battery_soc",
        "altitude",
        "speed",
        "rssi",
        "snr",
        "latency_ms",
        "loss_rate",
        "distance_to_gcs",
        "wind_level",
        "obstacle_factor",
        "source_type",
        "sim_source",
        "record_kind",
        "attack_active",
        "attack_type",
        "gt_attack_active",
        "gt_attack_type",
        "source_timestamp",
        "source_record_id",
        "source_uav_id",
        "source_dataset_name",
        "source_label",
        "energy",
        "flight_energy_wh",
        "communication_energy_wh",
        "detection_energy_wh",
        "total_energy_wh",
        "cumulative_energy_wh",
    }
)


def _is_forbidden_simulation_model_column(column: object) -> bool:
    normalized = str(column).strip().lower()
    if not normalized:
        return False
    if normalized in FORBIDDEN_SIMULATION_MODEL_COLUMNS:
        return True
    return normalized.startswith(FORBIDDEN_SIMULATION_MODEL_PREFIXES)


def _load_feature_columns(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, list):
        columns = payload
    elif isinstance(payload, Mapping):
        columns = payload.get("feature_columns", [])
    else:
        raise ValueError(f"Unsupported feature column payload in {path}")
    result = [str(column).strip() for column in columns if str(column).strip()]
    if not result:
        raise ValueError(f"No feature columns found in {path}")
    return result


class SimulationFeatureAdapter:
    """Force simulation windows onto the exact training feature schema."""

    def __init__(
        self,
        artifact: Mapping[str, Any],
        *,
        artifact_path: str | Path | None = None,
        feature_columns_path: str | Path | None = None,
        strict_model_feature_mode: bool = True,
    ) -> None:
        self.artifact = artifact
        self.strict_model_feature_mode = bool(strict_model_feature_mode)
        self.artifact_path = None if artifact_path is None else Path(artifact_path)
        self.feature_columns_path = self._resolve_feature_columns_path(feature_columns_path)
        self.feature_columns_source = "artifact.feature_cols"
        if self.feature_columns_path is not None and self.feature_columns_path.exists():
            self.feature_columns = _load_feature_columns(self.feature_columns_path)
            self.feature_columns_source = str(self.feature_columns_path)
        elif artifact.get("feature_cols") is not None:
            self.feature_columns = [str(column) for column in artifact["feature_cols"]]
        else:
            raise ValueError("SimulationFeatureAdapter requires feature_columns.json or artifact['feature_cols'].")
        if not self.feature_columns:
            raise ValueError("SimulationFeatureAdapter received an empty feature column whitelist.")

        forbidden = sorted({column for column in self.feature_columns if _is_forbidden_simulation_model_column(column)})
        if forbidden and self.strict_model_feature_mode:
            raise ValueError(
                "feature_columns contains simulation-only fields that are forbidden in strict_model_feature_mode: "
                + ", ".join(forbidden)
            )

        self.group_col = str(
            artifact.get("group_col")
            or artifact.get("group_config", {}).get("group_col", "")
            or getattr(artifact.get("preprocessor"), "group_col", "")
            or ""
        ).strip() or None
        self.normalization_mode = str(
            artifact.get("normalization_mode", getattr(artifact.get("preprocessor"), "normalization_mode", "global")) or "global"
        ).strip().lower()
        self.global_scaler = artifact.get("global_scaler")
        self.group_scalers = dict(artifact.get("group_scalers", {}) or {})
        self.model_input_columns = list(self.feature_columns)
        self.last_transform_summary = self._empty_summary()
        self.last_row_diagnostics = self._empty_row_diagnostics()
        self._seen_missing_model_features: set[str] = set()
        self._seen_extra_simulation_columns: set[str] = set()
        self._seen_dropped_simulation_columns: set[str] = set()
        self._group_input_totals: dict[str, dict[str, float]] = {}

    def _resolve_feature_columns_path(self, feature_columns_path: str | Path | None) -> Path | None:
        if feature_columns_path is not None and str(feature_columns_path).strip():
            return Path(feature_columns_path)
        if self.artifact_path is None:
            return None
        return self.artifact_path.resolve().parent / DEFAULT_FEATURE_COLUMNS_FILENAME

    def _empty_summary(self) -> dict[str, object]:
        return {
            "model_input_columns": list(self.model_input_columns) if hasattr(self, "model_input_columns") else [],
            "missing_model_features": [],
            "extra_simulation_columns": [],
            "dropped_simulation_columns": [],
            "per_uav_missing_feature_ratio": {},
            "per_uav_zero_feature_ratio": {},
            "per_uav_input_mean": {},
            "per_uav_input_std": {},
            "feature_adapter_enabled": True,
            "strict_model_feature_mode": bool(self.strict_model_feature_mode),
            "feature_columns_source": self.feature_columns_source,
        }

    def _empty_row_diagnostics(self) -> dict[str, object]:
        return {
            "group_ids": np.empty((0,), dtype=object),
            "missing_feature_ratio": np.empty((0,), dtype=np.float32),
            "zero_feature_ratio": np.empty((0,), dtype=np.float32),
            "raw_input_mean": np.empty((0,), dtype=np.float32),
            "raw_input_std": np.empty((0,), dtype=np.float32),
            "input_mean": np.empty((0,), dtype=np.float32),
            "input_std": np.empty((0,), dtype=np.float32),
        }

    def _diagnostic_group_column(self, df: pd.DataFrame) -> str | None:
        if self.group_col and self.group_col in df.columns:
            return self.group_col
        if "uav_id" in df.columns:
            return "uav_id"
        return None

    def _resolve_group_labels(self, df: pd.DataFrame) -> list[str]:
        if len(df) <= 0:
            return []
        group_col = self._diagnostic_group_column(df)
        if group_col is None:
            return ["__all__" for _ in range(len(df))]
        labels: list[str] = []
        for value in df[group_col].tolist():
            normalized = normalize_group_id(value)
            labels.append(normalized or "__ungrouped__")
        return labels

    def _build_group_metric_map(
        self,
        *,
        group_labels: list[str],
        missing_mask: np.ndarray,
        zero_mask: np.ndarray,
        transformed: np.ndarray,
    ) -> dict[str, dict[str, float]]:
        metrics: dict[str, dict[str, float]] = {}
        if not group_labels:
            return metrics
        labels = np.asarray(group_labels, dtype=object)
        for group_id in sorted({str(label) for label in labels.tolist()}):
            mask = labels == group_id
            if not np.any(mask):
                continue
            feature_count = int(missing_mask[mask].size)
            input_values = transformed[mask]
            metrics[group_id] = {
                "missing_feature_ratio": 0.0
                if feature_count <= 0
                else float(missing_mask[mask].sum()) / float(feature_count),
                "zero_feature_ratio": 0.0
                if feature_count <= 0
                else float(zero_mask[mask].sum()) / float(feature_count),
                "input_mean": 0.0 if input_values.size <= 0 else float(np.mean(input_values)),
                "input_std": 0.0 if input_values.size <= 0 else float(np.std(input_values)),
            }
        return metrics

    def _update_group_input_totals(
        self,
        *,
        group_labels: list[str],
        missing_mask: np.ndarray,
        zero_mask: np.ndarray,
        transformed: np.ndarray,
    ) -> None:
        if not group_labels:
            return
        labels = np.asarray(group_labels, dtype=object)
        for group_id in sorted({str(label) for label in labels.tolist()}):
            mask = labels == group_id
            if not np.any(mask):
                continue
            values = transformed[mask]
            stats = self._group_input_totals.setdefault(
                group_id,
                {
                    "feature_value_count": 0.0,
                    "missing_feature_count": 0.0,
                    "zero_feature_count": 0.0,
                    "input_sum": 0.0,
                    "input_sq_sum": 0.0,
                },
            )
            stats["feature_value_count"] += float(missing_mask[mask].size)
            stats["missing_feature_count"] += float(missing_mask[mask].sum())
            stats["zero_feature_count"] += float(zero_mask[mask].sum())
            stats["input_sum"] += float(np.sum(values))
            stats["input_sq_sum"] += float(np.sum(np.square(values)))

    def _aggregated_group_metrics(self) -> dict[str, dict[str, float]]:
        metrics: dict[str, dict[str, float]] = {}
        for group_id, stats in sorted(self._group_input_totals.items()):
            feature_value_count = float(stats.get("feature_value_count", 0.0))
            input_sum = float(stats.get("input_sum", 0.0))
            input_sq_sum = float(stats.get("input_sq_sum", 0.0))
            input_mean = 0.0 if feature_value_count <= 0.0 else input_sum / feature_value_count
            variance = 0.0
            if feature_value_count > 0.0:
                variance = max((input_sq_sum / feature_value_count) - (input_mean * input_mean), 0.0)
            metrics[group_id] = {
                "missing_feature_ratio": 0.0
                if feature_value_count <= 0.0
                else float(stats.get("missing_feature_count", 0.0)) / feature_value_count,
                "zero_feature_ratio": 0.0
                if feature_value_count <= 0.0
                else float(stats.get("zero_feature_count", 0.0)) / feature_value_count,
                "input_mean": float(input_mean),
                "input_std": float(np.sqrt(variance)),
            }
        return metrics

    def _coerce_series(self, df: pd.DataFrame, column: str) -> pd.Series:
        if column not in df.columns:
            return pd.Series(np.zeros(len(df), dtype=np.float32), index=df.index)
        return (
            pd.to_numeric(df[column], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .astype(np.float32)
        )

    def _apply_scalers(self, df: pd.DataFrame, raw_values: np.ndarray) -> np.ndarray:
        transformed = raw_values.astype(np.float32)
        if self.global_scaler is not None:
            transformed = self.global_scaler.transform(raw_values).astype(np.float32)
        if self.normalization_mode != "group" or not self.group_scalers:
            return transformed
        if not self.group_col:
            return transformed
        if self.group_col not in df.columns:
            if self.strict_model_feature_mode:
                raise ValueError(
                    "artifact was trained with group normalization but simulation input does not contain the required group_col"
                )
            return transformed
        group_labels = [normalize_group_id(value) for value in df[self.group_col].tolist()]
        for group_id, scaler in self.group_scalers.items():
            mask = np.asarray([label == group_id for label in group_labels], dtype=bool)
            if not np.any(mask):
                continue
            transformed[mask] = scaler.transform(raw_values[mask]).astype(np.float32)
        return transformed.astype(np.float32)

    def transform(self, sim_records_df: pd.DataFrame) -> np.ndarray:
        if not isinstance(sim_records_df, pd.DataFrame):
            raise TypeError("SimulationFeatureAdapter.transform expects a pandas DataFrame.")
        missing_model_features = [column for column in self.feature_columns if column not in sim_records_df.columns]
        visible_columns = [str(column) for column in sim_records_df.columns if not str(column).startswith("__")]
        extra_simulation_columns = sorted({column for column in visible_columns if column not in self.feature_columns})
        dropped_simulation_columns = [
            column for column in extra_simulation_columns if not self.group_col or column != self.group_col
        ]

        if len(sim_records_df) == 0:
            raw_values = np.empty((0, len(self.feature_columns)), dtype=np.float32)
            missing_mask = np.empty((0, len(self.feature_columns)), dtype=bool)
            transformed = np.empty((0, len(self.feature_columns)), dtype=np.float32)
            zero_mask = np.empty((0, len(self.feature_columns)), dtype=bool)
        else:
            raw_columns: list[np.ndarray] = []
            missing_columns: list[np.ndarray] = []
            for column in self.feature_columns:
                if column not in sim_records_df.columns:
                    series = pd.Series(np.full(len(sim_records_df), np.nan, dtype=np.float32), index=sim_records_df.index)
                else:
                    series = pd.to_numeric(sim_records_df[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
                missing_columns.append(series.isna().to_numpy(dtype=bool))
                raw_columns.append(series.fillna(0.0).to_numpy(dtype=np.float32))
            raw_values = np.stack(raw_columns, axis=1).astype(np.float32)
            missing_mask = np.stack(missing_columns, axis=1).astype(bool)
            transformed = self._apply_scalers(sim_records_df, raw_values)
            zero_mask = np.isclose(raw_values, 0.0)

        group_labels = self._resolve_group_labels(sim_records_df)
        if len(group_labels) == 0:
            group_metrics: dict[str, dict[str, float]] = {}
            self.last_row_diagnostics = self._empty_row_diagnostics()
        else:
            row_missing_ratio = missing_mask.mean(axis=1).astype(np.float32)
            row_zero_ratio = zero_mask.mean(axis=1).astype(np.float32)
            row_raw_input_mean = raw_values.mean(axis=1).astype(np.float32)
            row_raw_input_std = raw_values.std(axis=1).astype(np.float32)
            row_input_mean = transformed.mean(axis=1).astype(np.float32)
            row_input_std = transformed.std(axis=1).astype(np.float32)
            self.last_row_diagnostics = {
                "group_ids": np.asarray(group_labels, dtype=object),
                "missing_feature_ratio": row_missing_ratio,
                "zero_feature_ratio": row_zero_ratio,
                "raw_input_mean": row_raw_input_mean,
                "raw_input_std": row_raw_input_std,
                "input_mean": row_input_mean,
                "input_std": row_input_std,
            }
            group_metrics = self._build_group_metric_map(
                group_labels=group_labels,
                missing_mask=missing_mask,
                zero_mask=zero_mask,
                transformed=transformed,
            )
            self._update_group_input_totals(
                group_labels=group_labels,
                missing_mask=missing_mask,
                zero_mask=zero_mask,
                transformed=transformed,
            )

        self._seen_missing_model_features.update(missing_model_features)
        self._seen_extra_simulation_columns.update(extra_simulation_columns)
        self._seen_dropped_simulation_columns.update(dropped_simulation_columns)
        self.last_transform_summary = {
            "model_input_columns": list(self.model_input_columns),
            "missing_model_features": sorted(missing_model_features),
            "extra_simulation_columns": sorted(extra_simulation_columns),
            "dropped_simulation_columns": sorted(dropped_simulation_columns),
            "per_uav_missing_feature_ratio": {
                group_id: float(values["missing_feature_ratio"]) for group_id, values in sorted(group_metrics.items())
            },
            "per_uav_zero_feature_ratio": {
                group_id: float(values["zero_feature_ratio"]) for group_id, values in sorted(group_metrics.items())
            },
            "per_uav_input_mean": {
                group_id: float(values["input_mean"]) for group_id, values in sorted(group_metrics.items())
            },
            "per_uav_input_std": {
                group_id: float(values["input_std"]) for group_id, values in sorted(group_metrics.items())
            },
            "feature_adapter_enabled": True,
            "strict_model_feature_mode": bool(self.strict_model_feature_mode),
            "feature_columns_source": self.feature_columns_source,
        }
        return transformed.astype(np.float32)

    def diagnostics(self) -> dict[str, object]:
        group_metrics = self._aggregated_group_metrics()
        return {
            "model_input_columns": list(self.model_input_columns),
            "missing_model_features": sorted(self._seen_missing_model_features),
            "extra_simulation_columns": sorted(self._seen_extra_simulation_columns),
            "dropped_simulation_columns": sorted(self._seen_dropped_simulation_columns),
            "per_uav_missing_feature_ratio": {
                group_id: float(values["missing_feature_ratio"]) for group_id, values in group_metrics.items()
            },
            "per_uav_zero_feature_ratio": {
                group_id: float(values["zero_feature_ratio"]) for group_id, values in group_metrics.items()
            },
            "per_uav_input_mean": {
                group_id: float(values["input_mean"]) for group_id, values in group_metrics.items()
            },
            "per_uav_input_std": {
                group_id: float(values["input_std"]) for group_id, values in group_metrics.items()
            },
            "feature_adapter_enabled": True,
            "strict_model_feature_mode": bool(self.strict_model_feature_mode),
            "feature_columns_source": self.feature_columns_source,
        }


__all__ = [
    "DEFAULT_FEATURE_COLUMNS_FILENAME",
    "FORBIDDEN_SIMULATION_MODEL_COLUMNS",
    "SimulationFeatureAdapter",
]
