#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

try:
    import psutil
except ImportError:
    psutil = None

from ucs_oodid.artifacts import load_artifact
from ucs_oodid.io import load_records, save_json
from ucs_oodid.inference import collect_model_outputs
from ucs_oodid.model import UCSOODID
from ucs_oodid.ood import OODCalibrator, PrototypeBank, compute_raw_ood_scores
from ucs_oodid.utils import choose_device, set_seed
from ucs_oodid.windowing import UNKNOWN_GROUP_TOKEN, attach_group_index, attach_parsed_labels, build_grouped_windows, window_phase_labels


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
        if artifact.get("group_scalers") is not None:
            pre.group_scalers = dict(artifact["group_scalers"])
        elif getattr(pre, "group_scalers", None) is None:
            pre.group_scalers = {}
        if artifact.get("group_normalization_fallbacks") is not None:
            pre.group_fallbacks = {str(group): str(reason) for group, reason in artifact["group_normalization_fallbacks"].items()}
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
    return {
        "enabled": enabled,
        "group_embedding_dim": group_embedding_dim,
        "group_to_index": group_to_index,
    }


def resolve_effective_group_col(args, artifact: dict, pre) -> str:
    artifact_group_col = (
        artifact.get("group_col")
        or artifact.get("group_config", {}).get("group_col", "")
        or getattr(pre, "group_col", "")
        or ""
    ).strip()
    return (args.group_col or "").strip() or artifact_group_col


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


def count_model_parameters(model: torch.nn.Module) -> tuple[int, int]:
    total = sum(int(param.numel()) for param in model.parameters())
    trainable = sum(int(param.numel()) for param in model.parameters() if param.requires_grad)
    return total, trainable


def estimate_model_weights_size_mb(model: torch.nn.Module) -> float:
    total_bytes = 0
    for tensor in list(model.parameters()) + list(model.buffers()):
        total_bytes += int(tensor.numel()) * int(tensor.element_size())
    return float(total_bytes / (1024.0 * 1024.0))


def sync_device(device: str) -> None:
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def estimate_sklearn_parameter_count(estimator: object) -> int:
    total = 0
    if estimator is None:
        return 0
    inner = getattr(estimator, "model", None)
    target = inner if inner is not None else estimator
    tree_struct = getattr(target, "tree_", None)
    if tree_struct is not None:
        return int(getattr(tree_struct, "node_count", 0)) * 4
    estimators = getattr(target, "estimators_", None)
    if estimators is not None and hasattr(estimators, "__iter__"):
        for sub in estimators:
            total += estimate_sklearn_parameter_count(sub)
        if total > 0:
            return total
    named_steps = getattr(target, "named_steps", None)
    if named_steps:
        for _, step in named_steps.items():
            total += estimate_sklearn_parameter_count(step)
        return total
    coef = getattr(target, "coef_", None)
    if coef is not None:
        total += int(np.asarray(coef).size)
    intercept = getattr(target, "intercept_", None)
    if intercept is not None:
        total += int(np.asarray(intercept).size)
    return total


def benchmark_inference(
    model,
    windows,
    graph_cfg,
    bank,
    ood_cal,
    temperature: float,
    bank_k: int,
    batch_size: int,
    device: str,
    repeat_runs: int,
    warmup_runs: int,
    window_phases=None,
):
    if len(windows) == 0:
        raise ValueError("No windows were constructed from the provided input data.")

    for _ in range(max(int(warmup_runs), 0)):
        _ = collect_model_outputs(model, windows, graph_cfg, batch_size=batch_size, device=device, temperature=temperature)
        sync_device(device)

    run_times_s = []
    for _ in range(max(int(repeat_runs), 1)):
        sync_device(device)
        start = time.perf_counter()
        outs = collect_model_outputs(model, windows, graph_cfg, batch_size=batch_size, device=device, temperature=temperature)
        raw_scores = compute_raw_ood_scores(
            outs["logits"],
            outs["probs"],
            outs["embeddings"],
            bank,
            temperature=temperature,
            k_bank=bank_k,
        )
        _ = ood_cal.transform(raw_scores, phases=window_phases, groups=windows.group_ids)
        sync_device(device)
        run_times_s.append(time.perf_counter() - start)
    return outs, run_times_s


def benchmark_sklearn_inference(baseline, windows, repeat_runs: int, warmup_runs: int) -> list[float]:
    if len(windows) == 0:
        raise ValueError("No windows were constructed from the provided input data.")
    x = windows.x
    for _ in range(max(int(warmup_runs), 0)):
        _ = baseline.predict_proba(x)
    run_times_s = []
    for _ in range(max(int(repeat_runs), 1)):
        start = time.perf_counter()
        _ = baseline.predict_proba(x)
        run_times_s.append(time.perf_counter() - start)
    return run_times_s


def resource_snapshot() -> dict[str, Any]:
    out: dict[str, Any] = {"memory_usage_mb": None, "cpu_usage_percent": None}
    if psutil is None:
        return out
    proc = psutil.Process()
    rss = float(proc.memory_info().rss) / (1024.0 * 1024.0)
    cpu = float(proc.cpu_percent(interval=0.05))
    out["memory_usage_mb"] = rss
    out["cpu_usage_percent"] = cpu
    return out


def infer_output_path(artifact_path: Path, output_json: str) -> Path:
    if output_json and output_json.strip():
        return Path(output_json)
    return artifact_path.resolve().parent / "benchmark_report.json"


def apply_split_filter(df, split_filter: str):
    text = str(split_filter or "").strip()
    if not text:
        return df
    if "split" not in df.columns:
        raise ValueError("--split_filter requires a split column in the input CSV")
    names = [s.strip().lower() for s in text.split(",") if s.strip()]
    mask = df["split"].astype(str).str.strip().str.lower().isin(names)
    return df.loc[mask].reset_index(drop=True)


def main():
    p = argparse.ArgumentParser(description="Benchmark deployment-oriented onboard inference for a saved UCS-OODID artifact.")
    p.add_argument("--artifact", required=True)
    p.add_argument("--input", required=True)
    p.add_argument("--output_json", default="", help="Optional output path. Defaults to <artifact_dir>/benchmark_report.json.")
    p.add_argument("--label_col", default="label")
    p.add_argument("--timestamp_col", default="timestamp")
    p.add_argument("--record_id_col", default="record_id")
    p.add_argument("--group_col", default="", help="Optional grouping column. If omitted, use the group column stored in the artifact.")
    p.add_argument("--split_filter", default="", help="Comma-separated split names (e.g. test_id,test_ood) to subset rows before windowing.")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--warmup_runs", type=int, default=5)
    p.add_argument("--repeat_runs", type=int, default=20)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    artifact_path = Path(args.artifact)
    output_path = infer_output_path(artifact_path, args.output_json)
    device = choose_device(args.device)
    artifact = load_artifact(artifact_path, map_location=device)
    artifact_seed = artifact.get("seed", artifact.get("run_config", {}).get("seed"))
    if artifact_seed is not None:
        set_seed(int(artifact_seed))

    pre, normalization_mode = resolve_preprocessor_from_artifact(artifact)
    group_embedding_config = resolve_group_embedding_config(artifact)
    class_to_idx = artifact["class_to_idx"]

    df = load_records(args.input)
    if args.record_id_col not in df.columns:
        df[args.record_id_col] = np.arange(len(df))
    df = apply_split_filter(df, args.split_filter)

    effective_group_col = resolve_effective_group_col(args, artifact, pre)
    if normalization_mode == "group" and (not effective_group_col or effective_group_col not in df.columns):
        raise ValueError("artifact was trained with group normalization but input data does not contain the required group column")
    if group_embedding_config["enabled"] and (not effective_group_col or effective_group_col not in df.columns):
        raise ValueError("artifact was trained with group embedding but input data does not contain the required group column")
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

    prep_start = time.perf_counter()
    features = pre.transform(df)
    feature_prep_s = time.perf_counter() - prep_start

    window_start = time.perf_counter()
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
    if group_embedding_config["enabled"]:
        attach_group_index(windows, group_embedding_config["group_to_index"], unknown_group=UNKNOWN_GROUP_TOKEN)
    window_build_s = time.perf_counter() - window_start
    if len(windows) == 0:
        raise ValueError("No windows were generated from the provided input. Check the data split and window configuration.")

    deployment_backend = str(artifact.get("deployment_backend") or "ucs_oodid").strip().lower()
    gpu_mem_mb = None
    if deployment_backend == "sklearn_tabular":
        baseline = artifact["sklearn_baseline"]
        run_times_s = benchmark_sklearn_inference(baseline, windows, args.repeat_runs, args.warmup_runs)
        total_params = int(estimate_sklearn_parameter_count(baseline))
        trainable_params = total_params
        model_cfg = artifact.get("model_config", {})
        encoder_ablation = str(model_cfg.get("encoder_ablation", "sklearn_tabular"))
        graph_enabled = False
        model_weights_mb = float(total_params * 4) / (1024.0 * 1024.0)
    else:
        model_cfg = artifact["model_config"]
        model = UCSOODID(**model_cfg).to(device)
        model.load_state_dict(artifact["model_state"])
        model.eval()
        bank = PrototypeBank.from_dict(artifact["prototype_bank"])
        ood_cal = OODCalibrator.from_dict(artifact["ood_calibrator"])
        temperature = float(artifact["temperature"])
        bank_k = int(artifact.get("calibration_config", {}).get("bank_k", 5))
        window_phases = None
        if ood_cal.phase_aware_enabled and ood_cal.phase_column and ood_cal.phase_column in df.columns:
            window_phases = window_phase_labels(df, windows, ood_cal.phase_column)
        if str(device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)
        _, run_times_s = benchmark_inference(
            model=model,
            windows=windows,
            graph_cfg=artifact["graph_config"],
            bank=bank,
            ood_cal=ood_cal,
            temperature=temperature,
            bank_k=bank_k,
            batch_size=args.batch_size,
            device=device,
            repeat_runs=args.repeat_runs,
            warmup_runs=args.warmup_runs,
            window_phases=window_phases,
        )
        total_params, trainable_params = count_model_parameters(model)
        encoder_ablation = str(model_cfg.get("encoder_ablation", "unknown"))
        graph_enabled = bool(getattr(model, "uses_graph_encoder", False))
        model_weights_mb = estimate_model_weights_size_mb(model)
        if str(device).startswith("cuda") and torch.cuda.is_available():
            gpu_mem_mb = float(torch.cuda.max_memory_allocated(device)) / (1024.0 * 1024.0)

    avg_run_s = float(np.mean(run_times_s)) if run_times_s else 0.0
    num_windows = int(len(windows))
    throughput = float(num_windows / avg_run_s) if avg_run_s > 0 else 0.0
    avg_window_ms = float((avg_run_s * 1000.0) / max(num_windows, 1))

    snap = resource_snapshot()

    report = {
        "artifact": str(artifact_path),
        "input": str(args.input),
        "split_filter": str(args.split_filter or ""),
        "output_json": str(output_path),
        "device": device,
        "deployment_backend": deployment_backend,
        "batch_size": int(args.batch_size),
        "warmup_runs": int(args.warmup_runs),
        "repeat_runs": int(args.repeat_runs),
        "benchmark_window_count": num_windows,
        "num_windows": num_windows,
        "window_mode": artifact.get("window_config", {}).get("mode", "count"),
        "window_size": int(artifact.get("window_config", {}).get("size", 32)),
        "stride": int(artifact.get("window_config", {}).get("stride", 16)),
        "encoder_ablation": encoder_ablation,
        "graph_enabled": graph_enabled,
        "deployment_profile": artifact.get("deployment_profile", artifact.get("run_config", {}).get("deployment_profile", {})),
        "parameter_count": int(total_params),
        "trainable_parameter_count": int(trainable_params),
        "model_size_mb": float(artifact_path.stat().st_size / (1024.0 * 1024.0)),
        "model_weights_size_mb": float(model_weights_mb),
        "feature_preparation_ms": float(feature_prep_s * 1000.0),
        "window_build_ms": float(window_build_s * 1000.0),
        "average_run_ms": float(avg_run_s * 1000.0),
        "average_window_inference_ms": avg_window_ms,
        "average_inference_time_ms_per_window": avg_window_ms,
        "throughput_windows_per_sec": throughput,
        "throughput_windows_per_second": throughput,
        "memory_usage_mb": snap.get("memory_usage_mb"),
        "cpu_usage_percent": snap.get("cpu_usage_percent"),
        "gpu_memory_usage_mb": gpu_mem_mb,
        "run_times_ms": [float(value * 1000.0) for value in run_times_s],
    }

    save_json(report, output_path)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
