from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd


@dataclass
class AdapterResult:
    dataframe: pd.DataFrame
    dataset_name: str
    feature_columns: List[str]
    label_column: str = "label"
    timestamp_column: str = "timestamp"
    record_id_column: str = "record_id"
    notes: List[str] = field(default_factory=list)


LABEL_CANDIDATES = ["label", "Label", "class", "Class", "attack", "Attack", "category", "Category", "traffic_type", "Traffic Type"]
TIME_CANDIDATES = ["timestamp", "Timestamp", "time", "Time", "ts", "stime", "StartTime", "Flow Start", "flow_start"]
ID_CANDIDATES = ["record_id", "id", "ID", "uid", "flow_id", "Flow ID", "No.", "index"]


def _find_first(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    lower_map = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c in df.columns:
            return c
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    return None


def _normalise_label(v) -> str:
    if pd.isna(v):
        return "benign"
    s = str(v).strip().replace(" ", "_").replace("-", "_").lower()
    benign_alias = {"benign", "normal", "background", "legitimate", "0", "none"}
    return "benign" if s in benign_alias else s


def _numeric_timestamp(series: pd.Series) -> np.ndarray:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().mean() > 0.8:
        return numeric.ffill().fillna(0).to_numpy(float)
    dt = pd.to_datetime(series, errors="coerce", utc=True)
    if dt.notna().any():
        base = dt.dropna().iloc[0].timestamp()
        return dt.map(lambda x: x.timestamp() - base if pd.notna(x) else np.nan).ffill().fillna(0).to_numpy(float)
    return np.arange(len(series), dtype=float)


def _basic_standardise(df: pd.DataFrame, dataset_name: str, label_col: Optional[str] = None, timestamp_col: Optional[str] = None, record_id_col: Optional[str] = None) -> AdapterResult:
    work = df.copy()
    notes: List[str] = []
    label_col = label_col or _find_first(work, LABEL_CANDIDATES)
    if label_col is None:
        work["label"] = "benign"
        notes.append("No label column found; all records were marked benign. Pass --label_col for supervised experiments.")
    elif label_col != "label":
        work["label"] = work[label_col].map(_normalise_label)
    else:
        work["label"] = work["label"].map(_normalise_label)

    timestamp_col = timestamp_col or _find_first(work, TIME_CANDIDATES)
    if timestamp_col is None:
        work["timestamp"] = np.arange(len(work), dtype=float)
        notes.append("No timestamp column found; a monotonic record-order timestamp was created.")
    elif timestamp_col != "timestamp":
        work["timestamp"] = _numeric_timestamp(work[timestamp_col])
    else:
        work["timestamp"] = _numeric_timestamp(work["timestamp"])

    record_id_col = record_id_col or _find_first(work, ID_CANDIDATES)
    if record_id_col is None or record_id_col == "Flow ID":
        work["record_id"] = np.arange(len(work))
    elif record_id_col != "record_id":
        work["record_id"] = work[record_id_col].astype(str)

    # Convert common direction strings to numeric side-channel metadata.
    for col in list(work.columns):
        low = col.lower()
        if low in {"direction", "dir", "flow_direction"} and not pd.api.types.is_numeric_dtype(work[col]):
            work[col + "_numeric"] = work[col].astype(str).str.lower().map(lambda s: 1 if any(k in s for k in ["uav", "src", "out", "up", "forward"]) else 0)
    feature_cols = [c for c in work.columns if c not in {"label", "timestamp", "record_id"} and pd.api.types.is_numeric_dtype(work[c])]
    return AdapterResult(work, dataset_name, feature_cols, notes=notes)


def convert_dataset(df: pd.DataFrame, dataset: str = "generic", label_col: Optional[str] = None, timestamp_col: Optional[str] = None, record_id_col: Optional[str] = None) -> AdapterResult:
    """Convert a public dataset into the UCS-OODID canonical schema.

    The adapter is intentionally conservative: it preserves numeric side-channel metadata, creates canonical
    label/timestamp/record_id columns, and leaves leakage removal to MetadataPreprocessor.
    """
    name = dataset.lower().replace("_", "-")
    res = _basic_standardise(df, name, label_col=label_col, timestamp_col=timestamp_col, record_id_col=record_id_col)
    if name in {"cicids2017", "cic-ids2017", "cicids"}:
        res.notes.append("CIC-IDS2017 adapter: use Flow Duration / Fwd/Bwd packet statistics as encrypted-flow metadata; Flow ID, IPs and ports are later removed unless --allow_ports is used.")
    elif name in {"ustc-tfc2016", "ustc", "ustc-tfc"}:
        res.notes.append("USTC-TFC2016 adapter: expects flow/session statistical features. Malware family labels can be used as OOD hold-out classes.")
    elif name in {"mavlink", "mavlink-sequence"}:
        # Known leakage fields are kept for reporting but will be dropped by preprocessing.
        res.notes.append("MAVLink adapter: message_id/command/payload-like fields are retained only so leakage_report can prove removal.")
    elif name in {"uav-gcs-ids", "uav-gcs", "uavgcs"}:
        res.notes.append("UAV-GCS-IDS adapter: label names are normalised; mission/phase fields are preserved if present for RQ6 grouping.")
    elif name in {"uav-cyber-physical", "uav-cps", "uav"}:
        res.notes.append("UAV cyber-physical adapter: canonical schema generated from available telemetry/network metadata.")
    else:
        res.notes.append("Generic adapter used. Verify label mapping and leakage_report before reporting paper results.")
    return res
