#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from ucs_oodid.artifacts import load_artifact
from ucs_oodid.attribution import build_record_level_suspiciousness_ranking, dominant_ood_source, top_suspicious_records
from ucs_oodid.inference import collect_model_outputs
from ucs_oodid.io import load_records, save_json, write_jsonl
from ucs_oodid.metrics import compute_class_support, compute_present_class_macro_f1, multilabel_metrics
from ucs_oodid.model import UCSOODID
from ucs_oodid.ood import OODCalibrator, PrototypeBank, compute_raw_ood_scores
from ucs_oodid.utils import choose_device, set_seed
from ucs_oodid.windowing import UNKNOWN_GROUP_TOKEN, attach_group_index, attach_parsed_labels, build_grouped_windows, window_phase_labels


def build_windows(features, df, class_to_idx, artifact, label_col, timestamp_col, record_id_col, group_col):
    cfg = artifact["window_config"]
    return build_grouped_windows(
        features,
        df,
        class_to_idx,
        group_col=group_col,
        mode=cfg.get("mode", "count"),
        timestamp_col=timestamp_col,
        label_col=label_col,
        record_id_col=record_id_col,
        window_size=cfg.get("size", 32),
        stride=cfg.get("stride", 16),
        time_seconds=cfg.get("time_seconds", 2.0),
        adaptive_min_size=cfg.get("adaptive_min_size", 8),
        adaptive_max_size=cfg.get("adaptive_max_size", 64),
    )


def write_top_records_csv(rows: list[dict], path: str | Path) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["record_id", "window_id", "score", "label", "predicted_ood_score", "attention_weight"]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in fieldnames})


def normalize_window_group_id(group_id) -> str | None:
    if group_id is None:
        return None
    if isinstance(group_id, (float, np.floating)) and np.isnan(group_id):
        return None
    text = str(group_id).strip()
    return text or None


def canonical_summary_group_id(group_id) -> str:
    normalized = normalize_window_group_id(group_id)
    return "__ungrouped__" if normalized is None else normalized


def _group_indices(group_ids) -> dict[str, np.ndarray]:
    if group_ids is None:
        return {}
    arr = np.asarray(group_ids, dtype=object)
    if arr.size == 0:
        return {}
    grouped: dict[str, list[int]] = {}
    for index, value in enumerate(arr.tolist()):
        grouped.setdefault(canonical_summary_group_id(value), []).append(index)
    return {group_id: np.asarray(indices, dtype=np.int64) for group_id, indices in grouped.items()}


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


def attach_group_embedding_indices(windows, artifact: dict) -> dict:
    config = resolve_group_embedding_config(artifact)
    if config["enabled"]:
        attach_group_index(windows, config["group_to_index"], unknown_group=UNKNOWN_GROUP_TOKEN)
    return config


def compute_id_metrics_by_group(windows, probs, thresholds, label_names=None, present_class_min_support=1) -> dict:
    groups = _group_indices(getattr(windows, "group_ids", None))
    if not groups:
        return {}
    thresholds_array = np.asarray(thresholds, dtype=np.float32)
    preds = (np.asarray(probs) >= thresholds_array.reshape(1, -1)).astype(int)
    summary = {}
    for group, indices in groups.items():
        group_y = np.asarray(windows.y[indices])
        group_probs = np.asarray(probs[indices])
        summary[group] = {
            "windows": int(len(indices)),
            **multilabel_metrics(group_y, group_probs, thresholds_array),
            **compute_present_class_macro_f1(
                group_y,
                preds[indices],
                label_names=label_names,
                min_support=present_class_min_support,
            ),
            "class_support": compute_class_support(group_y, label_names=label_names),
        }
    return summary


def summarize_group_detections(
    rows: list[dict],
    group_col: str,
    *,
    id_metrics_by_group: dict | None = None,
    present_class_min_support: int = 1,
) -> dict:
    grouped_rows: dict[str, list[dict]] = {}
    for row in rows:
        group_id = canonical_summary_group_id(row.get("group_id"))
        grouped_rows.setdefault(group_id, []).append(row)
    id_metrics_by_group = id_metrics_by_group or {}

    def _score_array(items: list[dict], key: str) -> np.ndarray:
        return np.asarray([float(item.get(key, 0.0)) for item in items], dtype=np.float64)

    def _decision_array(items: list[dict]) -> np.ndarray:
        return np.asarray([1.0 if bool(item.get("is_ood", False)) else 0.0 for item in items], dtype=np.float64)

    def _summary_for_rows(items: list[dict]) -> dict:
        if not items:
            return {
                "windows": 0,
                "ood_alerts": 0,
                "alert_rate": 0.0,
                "mean_ood_score": 0.0,
                "max_ood_score": 0.0,
                "threshold": 0.0,
                "threshold_source": "global",
                "mean_ood_threshold": 0.0,
                "top_window_ids": [],
            }
        scores = _score_array(items, "ood_score")
        thresholds = _score_array(items, "ood_threshold")
        alerts = _decision_array(items)
        threshold = float(thresholds[0]) if len(thresholds) else 0.0
        if len(thresholds) and not np.allclose(thresholds, threshold):
            threshold = float(thresholds.mean())
        ranked = sorted(items, key=lambda item: float(item.get("ood_score", 0.0)), reverse=True)[:10]
        return {
            "windows": int(len(items)),
            "ood_alerts": int(alerts.sum()),
            "alert_rate": float(alerts.mean()),
            "mean_ood_score": float(scores.mean()),
            "max_ood_score": float(scores.max()),
            "threshold": threshold,
            "threshold_source": str(items[0].get("ood_threshold_source", "global")),
            "mean_ood_threshold": float(thresholds.mean()),
            "top_window_ids": [item.get("window_id") for item in ranked],
        }

    all_group_ids = sorted(set(grouped_rows) | {str(group_id).strip() for group_id in id_metrics_by_group if str(group_id).strip()})
    groups = {}
    for group_id in all_group_ids:
        group_summary = _summary_for_rows(grouped_rows.get(group_id, []))
        group_id_metrics = id_metrics_by_group.get(group_id, {})
        if isinstance(group_id_metrics, dict):
            group_summary.update(group_id_metrics)
        groups[group_id] = group_summary
    global_scores = _score_array(rows, "ood_score") if rows else np.asarray([], dtype=np.float64)
    global_alerts = _decision_array(rows) if rows else np.asarray([], dtype=np.float64)
    summary = {
        "group_col": group_col,
        "present_class_min_support": int(present_class_min_support),
        "groups": groups,
        "global": {
            "windows": int(len(rows)),
            "ood_alerts": int(global_alerts.sum()) if len(global_alerts) else 0,
            "alert_rate": float(global_alerts.mean()) if len(global_alerts) else 0.0,
            "mean_ood_score": float(global_scores.mean()) if len(global_scores) else 0.0,
            "max_ood_score": float(global_scores.max()) if len(global_scores) else 0.0,
        },
    }
    return summary


def main():
    p = argparse.ArgumentParser(description="Run UCS-OODID online detection/OOD rejection.")
    p.add_argument("--input", required=True)
    p.add_argument("--artifact", required=True)
    p.add_argument("--output_jsonl", required=True)
    p.add_argument("--record_scores_json", required=True)
    p.add_argument("--top_records_csv", default="", help="Optional CSV export for the Top-K record-level suspiciousness ranking.")
    p.add_argument("--summary_json", default="", help="Optional JSON export for group-level detection summary.")
    p.add_argument("--label_col", default="label")
    p.add_argument("--timestamp_col", default="timestamp")
    p.add_argument("--record_id_col", default="record_id")
    p.add_argument(
        "--group_col",
        default="",
        help="Optional grouping column, e.g., uav_id. If omitted, use group_col stored in artifact.",
    )
    p.add_argument(
        "--ood_threshold_mode",
        default=None,
        choices=["global", "group"],
        help="Optional override. If omitted, use the OOD threshold mode stored in the artifact.",
    )
    p.add_argument(
        "--present_class_min_support",
        type=int,
        default=1,
        help="Minimum y_true support required for a class to contribute to per-group present_class_macro_f1.",
    )
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--top_records", type=int, default=20)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    device = choose_device(args.device)
    artifact = load_artifact(args.artifact, map_location=device)
    artifact_seed = artifact.get("seed", artifact.get("run_config", {}).get("seed"))
    if artifact_seed is not None:
        set_seed(int(artifact_seed))
    model_cfg = artifact["model_config"]
    model = UCSOODID(**model_cfg).to(device)
    model.load_state_dict(artifact["model_state"])
    model.eval()
    pre, normalization_mode = resolve_preprocessor_from_artifact(artifact)
    group_embedding_config = resolve_group_embedding_config(artifact)
    bank = PrototypeBank.from_dict(artifact["prototype_bank"])
    ood_cal = OODCalibrator.from_dict(artifact["ood_calibrator"])
    threshold_config = resolve_ood_threshold_config(artifact, ood_cal, override_mode=args.ood_threshold_mode)
    class_names = artifact["class_names"]
    class_to_idx = artifact["class_to_idx"]
    thresholds = np.asarray(artifact["class_thresholds"], dtype=np.float32)
    temperature = float(artifact["temperature"])
    k_bank = int(artifact.get("calibration_config", {}).get("bank_k", 5))
    artifact_group_col = (
        artifact.get("group_col")
        or artifact.get("group_config", {}).get("group_col", "")
        or getattr(pre, "group_col", "")
        or ""
    ).strip()
    effective_group_col = (args.group_col or "").strip() or artifact_group_col

    df = load_records(args.input)
    if args.record_id_col not in df.columns:
        df[args.record_id_col] = np.arange(len(df))
    if normalization_mode == "group" and (not effective_group_col or effective_group_col not in df.columns):
        raise ValueError("artifact was trained with group normalization but input data does not contain required group_col")
    if group_embedding_config["enabled"] and (not effective_group_col or effective_group_col not in df.columns):
        raise ValueError("artifact was trained with group embedding but input data does not contain required group_col")
    if effective_group_col and effective_group_col not in df.columns:
        raise ValueError(f"group_col {effective_group_col!r} not found in input data.")
    pre.group_col = effective_group_col or getattr(pre, "group_col", None)
    if effective_group_col:
        if args.timestamp_col in df.columns:
            df = df.sort_values([effective_group_col, args.timestamp_col], kind="stable").reset_index(drop=True)
        else:
            df = df.sort_values(effective_group_col, kind="stable").reset_index(drop=True)
    elif args.timestamp_col in df.columns:
        df = df.sort_values(args.timestamp_col, kind="stable").reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)
    df = attach_parsed_labels(df, args.label_col) if args.label_col in df.columns else df.assign(__labels=[[] for _ in range(len(df))])
    features = pre.transform(df)
    windows = build_windows(
        features,
        df,
        class_to_idx,
        artifact,
        args.label_col,
        args.timestamp_col,
        args.record_id_col,
        effective_group_col,
    )
    attach_group_embedding_indices(windows, artifact)
    outs = collect_model_outputs(model, windows, artifact["graph_config"], batch_size=args.batch_size, device=device, temperature=temperature)
    raw_scores = compute_raw_ood_scores(outs["logits"], outs["probs"], outs["embeddings"], bank, temperature=temperature, k_bank=k_bank)
    window_phases = None
    if ood_cal.phase_aware_enabled and ood_cal.phase_column and ood_cal.phase_column in df.columns:
        window_phases = window_phase_labels(df, windows, ood_cal.phase_column)
    transformed = ood_cal.transform(raw_scores, phases=window_phases, groups=windows.group_ids)
    pred_multi = outs["probs"] >= thresholds.reshape(1, -1)
    id_metrics_by_group = compute_id_metrics_by_group(
        windows,
        outs["probs"],
        thresholds,
        label_names=class_names,
        present_class_min_support=args.present_class_min_support,
    )

    record_scores, suspiciousness_ranking = build_record_level_suspiciousness_ranking(
        windows.record_ids,
        outs["attention"],
        transformed["fused"],
        transformed["thresholds"],
        valid_mask=outs.get("mask"),
        window_ids=np.arange(len(windows), dtype=np.int64),
    )
    label_lookup = {}
    if args.label_col in df.columns:
        for rid, label in zip(df[args.record_id_col].astype(str), df[args.label_col].astype(str)):
            label_lookup.setdefault(str(rid), str(label))
    for row in suspiciousness_ranking:
        row["label"] = label_lookup.get(row["record_id"])
    top_records_global = suspiciousness_ranking[: args.top_records]
    legacy_top_records = top_suspicious_records(record_scores, topk=args.top_records)
    rows = []
    for i in range(len(windows)):
        labels = [class_names[j] for j, flag in enumerate(pred_multi[i]) if flag]
        norm_scores = {name: float(transformed["normalized"][name][i]) for name in transformed["normalized"]}
        phase_label = None if window_phases is None else window_phases[i]
        if getattr(windows, "group_ids", None) is not None:
            group_id = normalize_window_group_id(windows.group_ids[i])
        else:
            group_id = None
        local_records = []
        for rid, ok in zip(windows.record_ids[i].tolist(), outs.get("mask", windows.valid_mask)[i].tolist()):
            if not ok:
                continue
            local_records.append({"record_id": str(rid), "suspiciousness": float(record_scores.get(str(rid), 0.0))})
        local_records = sorted(local_records, key=lambda x: x["suspiciousness"], reverse=True)[:args.top_records]
        rows.append({
            "window_id": int(i),
            "group_id": group_id,
            "record_ids": [str(x) for x, ok in zip(windows.record_ids[i].tolist(), outs.get("mask", windows.valid_mask)[i].tolist()) if ok],
            "phase_label": None if phase_label is None else str(phase_label),
            "known_labels": labels,
            "probabilities": {class_names[j]: float(outs["probs"][i, j]) for j in range(len(class_names))},
            "is_ood": bool(transformed["decisions"][i]),
            "ood_score": float(transformed["fused"][i]),
            "ood_threshold": float(transformed["thresholds"][i]),
            "ood_threshold_source": str(transformed["threshold_sources"][i]),
            "raw_scores": {name: float(raw_scores[name][i]) for name in raw_scores},
            "normalized_scores": norm_scores,
            "dominant_ood_source": dominant_ood_source(transformed["normalized"], i),
            "embedding": outs["embeddings"][i].astype(float).tolist(),
            "attention": outs["attention"][i].astype(float).tolist(),
            "top_suspicious_records": local_records,
        })
    write_jsonl(rows, args.output_jsonl)
    save_json(
        {
            "record_level_suspiciousness_ranking": {
                "ranking_type": "analyst_triage_ranking",
                "note": "record-level suspiciousness ranks analyst triage priority and is not complete per-record attack localization.",
                "topk_default": int(args.top_records),
                "ranked_records": suspiciousness_ranking,
            },
            "record_scores": record_scores,
            "top_records": legacy_top_records,
        },
        args.record_scores_json,
    )
    if args.top_records_csv:
        write_top_records_csv(top_records_global, args.top_records_csv)
    if args.summary_json:
        summary = summarize_group_detections(
            rows,
            effective_group_col,
            id_metrics_by_group=id_metrics_by_group,
            present_class_min_support=args.present_class_min_support,
        )
        summary.update(threshold_config)
        save_json(summary, args.summary_json)
    print(f"Saved {len(rows)} window detections to {args.output_jsonl}")
    print(f"Saved record-level suspiciousness ranking to {args.record_scores_json}")
    if args.top_records_csv:
        print(f"Saved Top-{args.top_records} suspicious records CSV to {args.top_records_csv}")
    if args.summary_json:
        print(f"Saved group-level detection summary to {args.summary_json}")


if __name__ == "__main__":
    main()
