#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ucs_oodid.io import load_records, save_json


PHASE_NAMES = [
    "hovering_proxy",
    "cruising_proxy",
    "maneuvering_proxy",
    "high_rate_sensing_proxy",
]
PHASE_TYPE = "metadata_traffic_regime_proxy"
PHASE_NOTE = "mission_phase_proxy is derived from metadata clustering and is not a real flight-stage label"


def resolve_path(value: str, base: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)


def build_phase_features(df: pd.DataFrame) -> pd.DataFrame:
    flow_pkts_s = safe_numeric(df["flow_pkts_s"]).clip(lower=0.0)
    flow_byts_s = safe_numeric(df["flow_byts_s"]).clip(lower=0.0)
    flow_iat_mean = safe_numeric(df["flow_iat_mean"]).clip(lower=0.0)
    flow_duration = safe_numeric(df["flow_duration"]).clip(lower=0.0)
    pkt_size_avg = safe_numeric(df["pkt_size_avg"]).clip(lower=0.0)
    tot_fwd_pkts = safe_numeric(df["tot_fwd_pkts"]).clip(lower=0.0)
    tot_bwd_pkts = safe_numeric(df["tot_bwd_pkts"]).clip(lower=0.0)
    total_pkts = tot_fwd_pkts + tot_bwd_pkts
    direction_ratio = tot_fwd_pkts / total_pkts.clip(lower=1e-6)
    direction_balance = 1.0 - (direction_ratio - 0.5).abs() * 2.0
    return pd.DataFrame(
        {
            "log_flow_pkts_s": np.log1p(flow_pkts_s),
            "log_flow_byts_s": np.log1p(flow_byts_s),
            "log_flow_iat_mean": np.log1p(flow_iat_mean),
            "log_flow_duration": np.log1p(flow_duration),
            "log_pkt_size_avg": np.log1p(pkt_size_avg),
            "log_total_pkts": np.log1p(total_pkts),
            "direction_ratio": direction_ratio.clip(0.0, 1.0),
            "direction_balance": direction_balance.clip(0.0, 1.0),
        },
        index=df.index,
    )


def choose_fit_mask(df: pd.DataFrame, min_rows: int) -> tuple[pd.Series, str]:
    candidates: list[tuple[str, pd.Series]] = []
    if {"split", "label"}.issubset(df.columns):
        candidates.append(("train_benign", (df["split"].astype(str) == "train") & (df["label"].astype(str) == "benign")))
    if {"split", "recommended_partition"}.issubset(df.columns):
        candidates.append(("train_id", (df["split"].astype(str) == "train") & (df["recommended_partition"].astype(str) == "id")))
    if "split" in df.columns:
        candidates.append(("train_all", df["split"].astype(str) == "train"))
    if "label" in df.columns:
        candidates.append(("all_benign", df["label"].astype(str) == "benign"))
    candidates.append(("all_rows", pd.Series(True, index=df.index)))
    for name, mask in candidates:
        if int(mask.sum()) >= min_rows:
            return mask, name
    return candidates[-1][1], candidates[-1][0]


def zscore(series: pd.Series) -> pd.Series:
    std = float(series.std(ddof=0))
    if std <= 1e-9:
        return pd.Series(0.0, index=series.index)
    return (series - float(series.mean())) / std


def derive_phase_mapping(cluster_profiles: pd.DataFrame) -> tuple[dict[int, str], pd.DataFrame]:
    scores = pd.DataFrame(index=cluster_profiles.index)
    scores["rate_score"] = (
        zscore(cluster_profiles["log_flow_pkts_s"])
        + zscore(cluster_profiles["log_flow_byts_s"])
        - zscore(cluster_profiles["log_flow_iat_mean"])
        + 0.25 * zscore(cluster_profiles["log_total_pkts"])
    )
    scores["cruise_score"] = (
        zscore(cluster_profiles["log_flow_duration"])
        + zscore(cluster_profiles["direction_balance"])
        - 0.25 * scores["rate_score"].abs()
    )

    mapping: dict[int, str] = {}
    hover_cluster = int(scores["rate_score"].idxmin())
    high_cluster = int(scores["rate_score"].idxmax())
    mapping[hover_cluster] = "hovering_proxy"
    mapping[high_cluster] = "high_rate_sensing_proxy"

    remaining = [idx for idx in scores.index.tolist() if idx not in {hover_cluster, high_cluster}]
    if len(remaining) == 2:
        cruise_cluster = int(scores.loc[remaining, "cruise_score"].idxmax())
        maneuver_cluster = int([idx for idx in remaining if idx != cruise_cluster][0])
        mapping[cruise_cluster] = "cruising_proxy"
        mapping[maneuver_cluster] = "maneuvering_proxy"
    else:
        unused = [name for name in PHASE_NAMES if name not in mapping.values()]
        for idx, name in zip(remaining, unused):
            mapping[int(idx)] = name
    return mapping, scores


def main() -> None:
    p = argparse.ArgumentParser(description="Assign mission-phase proxy labels from metadata-only flow statistics.")
    p.add_argument("--input", default="data/gcs_to_uav_updated_experiment.csv")
    p.add_argument("--output", default="data/gcs_to_uav_updated_experiment_with_phase_proxy.csv")
    p.add_argument("--eval_output", default="data/gcs_to_uav_updated_eval_with_phase_proxy.csv")
    p.add_argument("--notes_json", default="data/gcs_to_uav_updated_phase_proxy_notes.json")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n_clusters", type=int, default=4)
    args = p.parse_args()

    input_path = resolve_path(args.input, ROOT)
    output_path = resolve_path(args.output, ROOT)
    eval_output_path = resolve_path(args.eval_output, ROOT)
    notes_path = resolve_path(args.notes_json, ROOT)

    df = load_records(input_path)
    required = {"flow_pkts_s", "flow_byts_s", "flow_iat_mean", "flow_duration", "pkt_size_avg", "tot_fwd_pkts", "tot_bwd_pkts"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Cannot assign phase proxy; missing columns: {missing}")

    phase_features = build_phase_features(df)
    min_rows = max(args.n_clusters * 20, 80)
    fit_mask, fit_source = choose_fit_mask(df, min_rows=min_rows)
    fit_features = phase_features.loc[fit_mask].copy()

    scaler = StandardScaler()
    x_fit = scaler.fit_transform(fit_features.values)
    kmeans = KMeans(n_clusters=args.n_clusters, n_init=20, random_state=args.seed)
    kmeans.fit(x_fit)
    clusters = kmeans.predict(scaler.transform(phase_features.values))

    fit_cluster_profiles = phase_features.loc[fit_mask].assign(cluster=clusters[fit_mask.to_numpy()]).groupby("cluster").mean(numeric_only=True)
    if len(fit_cluster_profiles) != args.n_clusters:
        raise RuntimeError("Not all phase-proxy clusters were populated in the fitting subset.")

    cluster_to_phase, score_df = derive_phase_mapping(fit_cluster_profiles)

    out_df = df.copy()
    out_df["mission_phase_proxy_cluster"] = clusters.astype(int)
    out_df["mission_phase_proxy"] = [cluster_to_phase[int(c)] for c in clusters]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    notes_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path, index=False)

    eval_rows = 0
    if "split" in out_df.columns:
        eval_df = out_df[out_df["split"].astype(str).isin(["test_id", "test_ood"])].copy()
        eval_output_path.parent.mkdir(parents=True, exist_ok=True)
        eval_df.to_csv(eval_output_path, index=False)
        eval_rows = int(len(eval_df))

    summary = {
        "input_csv": str(input_path),
        "output_csv": str(output_path),
        "eval_output_csv": str(eval_output_path),
        "phase_type": PHASE_TYPE,
        "is_ground_truth_mission_phase": False,
        "note": PHASE_NOTE,
        "fit_subset": fit_source,
        "fit_rows": int(fit_mask.sum()),
        "all_rows": int(len(out_df)),
        "eval_rows": eval_rows,
        "phase_feature_columns": phase_features.columns.tolist(),
        "phase_proxy_counts": {str(k): int(v) for k, v in out_df["mission_phase_proxy"].value_counts().to_dict().items()},
        "phase_counts": {str(k): int(v) for k, v in out_df["mission_phase_proxy"].value_counts().to_dict().items()},
        "phase_proxy_counts_by_split": (
            out_df.groupby(["split", "mission_phase_proxy"]).size().rename("rows").reset_index().to_dict(orient="records")
            if "split" in out_df.columns
            else []
        ),
        "phase_counts_by_split": (
            out_df.groupby(["split", "mission_phase_proxy"]).size().rename("rows").reset_index().to_dict(orient="records")
            if "split" in out_df.columns
            else []
        ),
        "cluster_to_phase_proxy": {str(k): v for k, v in cluster_to_phase.items()},
        "cluster_to_phase": {str(k): v for k, v in cluster_to_phase.items()},
        "cluster_profiles_fit_subset": fit_cluster_profiles.reset_index().to_dict(orient="records"),
        "cluster_scores_fit_subset": score_df.reset_index().to_dict(orient="records"),
        "notes": [
            PHASE_NOTE,
            "mission_phase_proxy is a metadata traffic-regime proxy rather than a ground-truth flight-stage label.",
            "Clusters were fit on a train-benign subset when available to reduce attack-label leakage into phase definitions.",
            "Cluster names were assigned heuristically by centroid rate, duration, and direction-balance patterns.",
        ],
    }
    save_json(summary, notes_path)
    print(json.dumps({"mission_phase_proxy_assignment_summary": summary}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
