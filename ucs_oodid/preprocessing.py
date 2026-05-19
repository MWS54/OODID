from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

DEFAULT_DROP_PATTERNS = [
    "payload", "raw_bytes", "byte_payload", "message_id", "msg_id", "command",
    "mavlink_msg", "application", "app_", "tls_sni", "hostname", "host_name",
    "url", "uri", "dns_query", "src_ip", "dst_ip", "source_ip", "destination_ip",
    "ip_src", "ip_dst", "five_tuple", "flow_id", "session_id"
]

PORT_PATTERNS = ["src_port", "dst_port", "source_port", "destination_port", "sport", "dport"]
RESERVED_METADATA_PREFIXES = ("sim_", "gt_", "source_")
RESERVED_METADATA_ONLY_COLUMNS = {
    "uav_id",
    "domain_id",
    "dataset_name",
    "split",
    "source_type",
    "simulation_role",
    "source_note",
    "source_timestamp",
    "source_record_id",
    "source_uav_id",
    "source_dataset_name",
    "source_label",
    "direction_type",
    "scenario_role",
    "original_group_id",
    "mission_phase",
    "mission_context",
    "sim_source",
    "sim_time",
    "record_kind",
    "attack_source_dataset",
    "attack_active",
    "attack_type",
    "gt_attack_active",
    "gt_attack_type",
    "response_action",
    "response_time",
    "response_reason",
    "battery_soc",
    "speed",
    "altitude",
    "rssi",
    "snr",
    "latency_ms",
    "loss_rate",
    "distance_to_gcs",
    "wind_level",
    "obstacle_factor",
    "bytes_up",
    "bytes_down",
    "throughput_bytes",
    "cpu_load",
    "board_temperature_c",
    "flight_energy_wh",
    "communication_energy_wh",
    "detection_energy_wh",
    "total_energy_wh",
    "cumulative_energy_wh",
    "energy",
}


def normalize_group_id(value: object) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, (float, np.floating)) and np.isnan(value):
        return None
    text = str(value).strip()
    return text or None


@dataclass
class MetadataPreprocessor:
    label_col: str = "label"
    timestamp_col: Optional[str] = "timestamp"
    record_id_col: Optional[str] = "record_id"
    group_col: Optional[str] = None
    drop_patterns: List[str] = field(default_factory=lambda: list(DEFAULT_DROP_PATTERNS))
    identifier_free: bool = True
    allow_ports: bool = False
    fill_value: float = 0.0
    feature_cols: List[str] = field(default_factory=list)
    feature_medians: dict = field(default_factory=dict)
    scaler: Optional[StandardScaler] = None
    normalization_mode: str = "global"

    def _is_dropped(self, col: str) -> bool:
        lower = col.lower()
        protected = {self.label_col, self.timestamp_col, self.record_id_col, "__labels", "__is_ood_record"}
        if self.group_col:
            protected.add(self.group_col)
        protected_lower = {str(name).lower() for name in protected if name is not None}
        if lower in protected_lower:
            return True
        if lower.startswith(RESERVED_METADATA_PREFIXES):
            return True
        if lower in RESERVED_METADATA_ONLY_COLUMNS:
            return True
        patterns = list(self.drop_patterns)
        if self.identifier_free and not self.allow_ports:
            patterns += PORT_PATTERNS
        return any(p.lower() in lower for p in patterns)

    def infer_feature_cols(self, df: pd.DataFrame, extra_drop: Optional[Sequence[str]] = None) -> List[str]:
        extra = set(extra_drop or [])
        candidates: List[str] = []
        for col in df.columns:
            if col in extra or self._is_dropped(col):
                continue
            if pd.api.types.is_numeric_dtype(df[col]):
                candidates.append(col)
            else:
                # Try coercion. If at least 90% values are numeric, keep it.
                coerced = pd.to_numeric(df[col], errors="coerce")
                if coerced.notna().mean() >= 0.9:
                    candidates.append(col)
        if not candidates:
            raise ValueError("No metadata feature columns found. Check leakage-control settings or input schema.")
        return candidates

    def _missing_series(self, df: pd.DataFrame) -> pd.Series:
        return pd.Series(np.full(len(df), self.fill_value, dtype=np.float32), index=df.index)

    def _coerce_feature_series(self, df: pd.DataFrame, col: str, fit_mode: bool) -> pd.Series:
        if col not in df.columns:
            if fit_mode:
                raise ValueError(f"Missing feature column: {col}")
            return self._missing_series(df)
        return pd.to_numeric(df[col], errors="coerce")

    def _fit_feature_statistics(self, df: pd.DataFrame, feature_cols: Optional[Sequence[str]] = None) -> None:
        self.feature_cols = list(feature_cols) if feature_cols else self.infer_feature_cols(df)
        self.feature_medians = {}
        for col in self.feature_cols:
            series = self._coerce_feature_series(df, col, fit_mode=False)
            val = series.replace([np.inf, -np.inf], np.nan).median()
            self.feature_medians[col] = float(val) if pd.notna(val) else float(self.fill_value)

    def fit(self, df: pd.DataFrame, feature_cols: Optional[Sequence[str]] = None) -> "MetadataPreprocessor":
        self._fit_feature_statistics(df, feature_cols=feature_cols)
        x = self._extract(df, fit_mode=True)
        self.scaler = StandardScaler()
        self.scaler.fit(x)
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if self.scaler is None or not self.feature_cols:
            raise RuntimeError("MetadataPreprocessor must be fitted before transform.")
        x = self._extract(df, fit_mode=False)
        return self.scaler.transform(x).astype(np.float32)

    def fit_transform(self, df: pd.DataFrame, feature_cols: Optional[Sequence[str]] = None) -> np.ndarray:
        self.fit(df, feature_cols)
        return self.transform(df)

    def _extract(self, df: pd.DataFrame, fit_mode: bool) -> np.ndarray:
        data = []
        for col in self.feature_cols:
            series = self._coerce_feature_series(df, col, fit_mode=fit_mode)
            values = series.replace([np.inf, -np.inf], np.nan).fillna(self.feature_medians.get(col, 0.0))
            data.append(values.to_numpy(dtype=np.float32))
        return np.stack(data, axis=1).astype(np.float32)

    def normalization_summary(self) -> dict:
        return {
            "mode": self.normalization_mode,
            "group_col": self.group_col,
            "groups": {},
            "fallback_groups": [],
        }

    def leakage_report(self, df: pd.DataFrame) -> dict:
        removed, retained = [], []
        for col in df.columns:
            if col in self.feature_cols:
                retained.append(col)
            elif self._is_dropped(col):
                removed.append(col)
        return {
            "retained_features": retained,
            "removed_or_ignored_features": sorted(set(removed)),
            "identifier_free": self.identifier_free,
            "allow_ports": self.allow_ports,
            "group_col": self.group_col,
        }


@dataclass
class GroupAwareMetadataPreprocessor(MetadataPreprocessor):
    min_group_records: int = 10
    group_scalers: Dict[str, StandardScaler] = field(default_factory=dict)
    group_record_counts: Dict[str, int] = field(default_factory=dict)
    group_fallbacks: Dict[str, str] = field(default_factory=dict)
    normalization_mode: str = "group"

    def fit(self, df: pd.DataFrame, feature_cols: Optional[Sequence[str]] = None) -> "GroupAwareMetadataPreprocessor":
        if not self.group_col:
            raise ValueError("group normalization requires --group_col")
        if self.group_col not in df.columns:
            raise ValueError(f"Missing required group column: {self.group_col}")

        self._fit_feature_statistics(df, feature_cols=feature_cols)
        x = self._extract(df, fit_mode=True)
        self.scaler = StandardScaler()
        self.scaler.fit(x)
        self.group_scalers = {}
        self.group_record_counts = {}
        self.group_fallbacks = {}

        group_labels = [normalize_group_id(value) for value in df[self.group_col].tolist()]
        unique_groups = sorted({label for label in group_labels if label is not None})
        for group_id in unique_groups:
            mask = np.asarray([label == group_id for label in group_labels], dtype=bool)
            count = int(mask.sum())
            self.group_record_counts[group_id] = count
            if count >= self.min_group_records:
                scaler = StandardScaler()
                scaler.fit(x[mask])
                self.group_scalers[group_id] = scaler
            else:
                self.group_fallbacks[group_id] = "fallback_to_global_due_to_small_group_size"
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if self.scaler is None or not self.feature_cols:
            raise RuntimeError("GroupAwareMetadataPreprocessor must be fitted before transform.")

        x = self._extract(df, fit_mode=False)
        transformed = self.scaler.transform(x).astype(np.float32)
        if not self.group_col or self.group_col not in df.columns or not self.group_scalers:
            return transformed

        group_labels = [normalize_group_id(value) for value in df[self.group_col].tolist()]
        for group_id, scaler in self.group_scalers.items():
            mask = np.asarray([label == group_id for label in group_labels], dtype=bool)
            if not np.any(mask):
                continue
            transformed[mask] = scaler.transform(x[mask]).astype(np.float32)
        return transformed.astype(np.float32)

    def normalization_summary(self) -> dict:
        groups = {}
        for group_id in sorted(self.group_record_counts):
            groups[group_id] = {
                "records": int(self.group_record_counts[group_id]),
                "used_group_scaler": group_id in self.group_scalers,
            }
        return {
            "mode": self.normalization_mode,
            "group_col": self.group_col,
            "groups": groups,
            "fallback_groups": sorted(self.group_fallbacks),
        }
