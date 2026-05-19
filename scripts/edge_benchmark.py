#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import psutil
import torch

from ucs_oodid.artifacts import load_artifact
from ucs_oodid.graph import build_behavior_graph
from ucs_oodid.io import load_records, save_json
from ucs_oodid.model import UCSOODID
from ucs_oodid.ood import OODCalibrator, PrototypeBank, compute_raw_ood_scores
from ucs_oodid.utils import choose_device, set_seed
from ucs_oodid.windowing import UNKNOWN_GROUP_TOKEN, attach_group_index, attach_parsed_labels, build_grouped_windows, window_phase_labels


def median_ms(values):
    return float(np.median(values) * 1000.0) if values else 0.0


def process_cpu_seconds(proc: psutil.Process) -> float:
    times = proc.cpu_times()
    return float(times.user + times.system)


def run_benchmark_iteration(
    df,
    pre,
    class_to_idx,
    artifact,
    model,
    bank,
    ood_cal,
    temperature: float,
    bank_k: int,
    device: str,
    label_col: str,
    record_id_col: str,
    win_size: int,
    stride: int,
    timings: dict | None = None,
) -> bool:
    t_total = time.perf_counter()

    t0 = time.perf_counter()
    local_df = df.copy()
    if timings is not None:
        timings["metadata_extraction"].append(time.perf_counter() - t0)

    t0 = time.perf_counter()
    feats = pre.transform(local_df)
    if timings is not None:
        timings["feature_normalization"].append(time.perf_counter() - t0)

    t0 = time.perf_counter()
    group_col = (
        artifact.get("group_col")
        or artifact.get("group_config", {}).get("group_col")
        or getattr(pre, "group_col", None)
    )
    windows = build_grouped_windows(
        feats,
        local_df,
        class_to_idx,
        group_col=group_col,
        label_col=label_col,
        timestamp_col="timestamp",
        record_id_col=record_id_col,
        window_size=win_size,
        stride=stride,
    )
    if artifact.get("use_group_embedding") and artifact.get("group_to_index"):
        attach_group_index(windows, artifact["group_to_index"], unknown_group=UNKNOWN_GROUP_TOKEN)
    if timings is not None:
        timings["windowing"].append(time.perf_counter() - t0)
    if len(windows) == 0:
        return False

    x = torch.tensor(windows.x[:1], dtype=torch.float32, device=device)
    group_index = None
    if windows.group_index is not None:
        group_index = torch.tensor(windows.group_index[:1], dtype=torch.long, device=device)

    t0 = time.perf_counter()
    adj = None
    if model.uses_graph_encoder:
        adj = build_behavior_graph(x, **artifact["graph_config"]).adj
    if timings is not None:
        timings["behavioral_graph"].append(time.perf_counter() - t0)

    t0 = time.perf_counter()
    with torch.no_grad():
        out = model(x, adj, temperature=temperature, group_index=group_index)
    if device == "cuda":
        torch.cuda.synchronize()
    if timings is not None:
        timings["temporal_graph_inference"].append(time.perf_counter() - t0)

    t0 = time.perf_counter()
    raw = compute_raw_ood_scores(
        out["logits"].cpu().numpy(),
        out["probs"].cpu().numpy(),
        out["embedding"].cpu().numpy(),
        bank,
        temperature=temperature,
        k_bank=bank_k,
    )
    phases = None
    if ood_cal.phase_aware_enabled and ood_cal.phase_column and ood_cal.phase_column in local_df.columns:
        phases = window_phase_labels(local_df, windows, ood_cal.phase_column)[:1]
    trans = ood_cal.transform(raw, phases=phases)
    if timings is not None:
        timings["ood_scoring_fusion"].append(time.perf_counter() - t0)

    t0 = time.perf_counter()
    _ = float(trans["fused"][0]) - float(trans["thresholds"][0])
    if timings is not None:
        timings["record_attribution"].append(time.perf_counter() - t0)
        timings["total"].append(time.perf_counter() - t_total)
    return True


def main():
    p = argparse.ArgumentParser(description="Module-level latency benchmark for edge feasibility.")
    p.add_argument("--input", required=True)
    p.add_argument("--artifact", required=True)
    p.add_argument("--output_json", required=True)
    p.add_argument("--iterations", type=int, default=100)
    p.add_argument("--warmup_iterations", type=int, default=10)
    p.add_argument("--label_col", default="label")
    p.add_argument("--record_id_col", default="record_id")
    p.add_argument("--device", default=None)
    args = p.parse_args()

    device = choose_device(args.device)
    artifact = load_artifact(args.artifact, map_location=device)
    artifact_seed = artifact.get("seed", artifact.get("run_config", {}).get("seed"))
    if artifact_seed is not None:
        set_seed(int(artifact_seed))
    pre = artifact["preprocessor"]
    df = load_records(args.input).head(max(256, artifact["window_config"].get("size", 32) * 10)).copy()
    if args.record_id_col not in df.columns:
        df[args.record_id_col] = np.arange(len(df))
    df = attach_parsed_labels(df, args.label_col) if args.label_col in df.columns else df.assign(__labels=[[] for _ in range(len(df))])
    class_to_idx = artifact["class_to_idx"]
    model = UCSOODID(**artifact["model_config"]).to(device)
    model.load_state_dict(artifact["model_state"])
    model.eval()
    bank = PrototypeBank.from_dict(artifact["prototype_bank"])
    ood_cal = OODCalibrator.from_dict(artifact["ood_calibrator"])
    temperature = float(artifact["temperature"])
    bank_k = int(artifact.get("calibration_config", {}).get("bank_k", 5))
    win_size = int(artifact["window_config"].get("size", 32))
    stride = int(artifact["window_config"].get("stride", 16))

    timings = {"metadata_extraction": [], "feature_normalization": [], "windowing": [], "behavioral_graph": [], "temporal_graph_inference": [], "ood_scoring_fusion": [], "record_attribution": [], "total": []}
    proc = psutil.Process()
    for _ in range(args.warmup_iterations):
        ok = run_benchmark_iteration(
            df,
            pre,
            class_to_idx,
            artifact,
            model,
            bank,
            ood_cal,
            temperature,
            bank_k,
            device,
            args.label_col,
            args.record_id_col,
            win_size,
            stride,
            timings=None,
        )
        if not ok:
            break

    cpu_time_start = process_cpu_seconds(proc)
    wall_time_start = time.perf_counter()
    completed = 0
    for _ in range(args.iterations):
        ok = run_benchmark_iteration(
            df,
            pre,
            class_to_idx,
            artifact,
            model,
            bank,
            ood_cal,
            temperature,
            bank_k,
            device,
            args.label_col,
            args.record_id_col,
            win_size,
            stride,
            timings=timings,
        )
        if ok:
            completed += 1

    wall_time_s = max(time.perf_counter() - wall_time_start, 1e-9)
    cpu_time_s = max(process_cpu_seconds(proc) - cpu_time_start, 0.0)
    logical_cpus = max(psutil.cpu_count(logical=True) or 1, 1)

    report = {k + "_ms": median_ms(v) for k, v in timings.items()}
    report["throughput_windows_per_sec"] = float(1000.0 / max(report.get("total_ms", 1e-9), 1e-9))
    report["iterations_completed"] = int(completed)
    report["warmup_iterations"] = int(args.warmup_iterations)
    report["benchmark_wall_ms"] = float(wall_time_s * 1000.0)
    report["cpu_percent_avg"] = float(100.0 * cpu_time_s / wall_time_s)
    report["cpu_percent_system_avg"] = float(report["cpu_percent_avg"] / logical_cpus)
    report["rss_mb"] = float(proc.memory_info().rss / 1024 / 1024)
    report["device"] = device
    save_json(report, args.output_json)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
