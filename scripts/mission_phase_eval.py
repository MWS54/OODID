#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ucs_oodid.artifacts import load_artifact
from ucs_oodid.io import load_json, load_records, read_jsonl
from ucs_oodid.metrics import ood_metrics
from ucs_oodid.windowing import attach_parsed_labels, build_adaptive_windows, build_count_windows, build_time_windows

PHASE_TYPE = "metadata_traffic_regime_proxy"
PHASE_NOTE = "mission_phase_proxy is derived from metadata clustering and is not a real flight-stage label"


def run(cmd):
    print("RUN", " ".join(map(str, cmd)), flush=True)
    subprocess.run(list(map(str, cmd)), check=True)


def summarize_detection_rows(rows):
    if not rows:
        return {"windows": 0}
    return {
        "windows": len(rows),
        "ood_alerts": sum(1 for r in rows if r.get("is_ood")),
        "ood_alert_rate": sum(1 for r in rows if r.get("is_ood")) / max(len(rows), 1),
        "mean_ood_score": sum(float(r.get("ood_score", 0.0)) for r in rows) / max(len(rows), 1),
    }


def build_windows(features, df, class_to_idx, artifact, label_col, timestamp_col, record_id_col):
    cfg = artifact["window_config"]
    mode = cfg.get("mode", "count")
    if mode == "time":
        return build_time_windows(
            features,
            df,
            class_to_idx,
            timestamp_col=timestamp_col,
            label_col=label_col,
            record_id_col=record_id_col,
            time_seconds=cfg.get("time_seconds", 2.0),
            max_records=cfg.get("size", 16),
        )
    if mode == "adaptive":
        return build_adaptive_windows(
            features,
            df,
            class_to_idx,
            timestamp_col=timestamp_col,
            label_col=label_col,
            record_id_col=record_id_col,
            min_size=cfg.get("adaptive_min_size", 8),
            max_size=cfg.get("adaptive_max_size", 64),
            target_records=cfg.get("size", 16),
            stride=cfg.get("stride", 8),
        )
    return build_count_windows(
        features,
        df,
        class_to_idx,
        label_col=label_col,
        record_id_col=record_id_col,
        window_size=cfg.get("size", 16),
        stride=cfg.get("stride", 8),
    )


def prepare_phase_df(df, label_col, timestamp_col, record_id_col):
    work = df.copy()
    if record_id_col not in work.columns:
        work[record_id_col] = np.arange(len(work))
    if timestamp_col in work.columns:
        work = work.sort_values(timestamp_col).reset_index(drop=True)
    else:
        work = work.reset_index(drop=True)
    if label_col in work.columns:
        work = attach_parsed_labels(work, label_col)
    else:
        work = work.assign(__labels=[[] for _ in range(len(work))])
    return work


def compute_phase_metrics(df, artifact, detections, label_col, timestamp_col, record_id_col):
    work = prepare_phase_df(df, label_col, timestamp_col, record_id_col)
    features = artifact["preprocessor"].transform(work)
    windows = build_windows(features, work, artifact["class_to_idx"], artifact, label_col, timestamp_col, record_id_col)
    summary = {
        "id_windows_true": int((~windows.ood).sum()),
        "ood_windows_true": int(windows.ood.sum()),
    }
    if len(detections) != len(windows):
        summary["metric_error"] = f"detection/window length mismatch: {len(detections)} vs {len(windows)}"
        return summary
    scores = np.asarray([float(row.get("ood_score", 0.0)) for row in detections], dtype=np.float32)
    decisions = np.asarray([bool(row.get("is_ood", False)) for row in detections], dtype=bool)
    summary.update(ood_metrics(windows.ood.astype(int), scores, decisions))
    return summary


def main():
    p = argparse.ArgumentParser(description="RQ6 mission-phase proxy evaluation by filtering records per traffic-regime proxy and running detection.")
    p.add_argument("--input", required=True)
    p.add_argument("--artifact", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--phase_col", default="mission_phase_proxy")
    p.add_argument("--label_col", default="label")
    p.add_argument("--timestamp_col", default="timestamp")
    p.add_argument("--record_id_col", default="record_id")
    p.add_argument("--detect_args", default="")
    p.add_argument("--run_benchmark", action="store_true")
    p.add_argument("--benchmark_device", default=None)
    p.add_argument("--benchmark_iterations", type=int, default=100)
    p.add_argument("--benchmark_warmup_iterations", type=int, default=10)
    args = p.parse_args()

    df = load_records(args.input)
    if args.phase_col not in df.columns:
        raise ValueError(f"phase_col {args.phase_col!r} not found. Add mission-phase proxy labels or use --phase_col.")
    artifact = load_artifact(args.artifact, map_location="cpu")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    detect_py = Path(__file__).with_name("detect.py")
    edge_py = Path(__file__).with_name("edge_benchmark.py")
    summary = []
    is_proxy_phase_column = args.phase_col == "mission_phase_proxy"
    for phase in sorted(df[args.phase_col].dropna().astype(str).unique()):
        safe = phase.replace("/", "_").replace(" ", "_")
        phase_dir = out / safe
        phase_dir.mkdir(parents=True, exist_ok=True)
        phase_df = df[df[args.phase_col].astype(str) == phase].copy()
        data_path = phase_dir / "phase_records.csv"
        phase_df.to_csv(data_path, index=False)
        det_path = phase_dir / "detections.jsonl"
        rec_path = phase_dir / "record_scores.json"
        cmd = [
            sys.executable,
            detect_py,
            "--input",
            data_path,
            "--artifact",
            args.artifact,
            "--output_jsonl",
            det_path,
            "--record_scores_json",
            rec_path,
            "--label_col",
            args.label_col,
            "--timestamp_col",
            args.timestamp_col,
            "--record_id_col",
            args.record_id_col,
        ]
        if args.detect_args:
            cmd += args.detect_args.split()
        run(cmd)
        detections = read_jsonl(det_path)
        phase_summary = {
            "phase": phase,
            "phase_display_name": f"{phase} (traffic-regime proxy)" if is_proxy_phase_column else phase,
            "phase_column": args.phase_col,
            "phase_type": PHASE_TYPE if is_proxy_phase_column else "custom_phase_group_column",
            "is_ground_truth_mission_phase": False if is_proxy_phase_column else None,
            "note": PHASE_NOTE if is_proxy_phase_column else f"{args.phase_col} is treated as the grouping column for this report.",
            "records": int(len(phase_df)),
            **summarize_detection_rows(detections),
            **compute_phase_metrics(phase_df, artifact, detections, args.label_col, args.timestamp_col, args.record_id_col),
            "dir": str(phase_dir),
        }
        if args.run_benchmark:
            bench_path = phase_dir / "edge_benchmark.json"
            bench_cmd = [
                sys.executable,
                edge_py,
                "--input",
                data_path,
                "--artifact",
                args.artifact,
                "--output_json",
                bench_path,
                "--label_col",
                args.label_col,
                "--record_id_col",
                args.record_id_col,
                "--iterations",
                str(args.benchmark_iterations),
                "--warmup_iterations",
                str(args.benchmark_warmup_iterations),
            ]
            if args.benchmark_device is not None:
                bench_cmd += ["--device", args.benchmark_device]
            run(bench_cmd)
            bench = load_json(bench_path)
            phase_summary.update(
                {
                    "latency_ms": bench.get("total_ms"),
                    "throughput_windows_per_sec": bench.get("throughput_windows_per_sec"),
                    "cpu_percent_avg": bench.get("cpu_percent_avg"),
                    "cpu_percent_system_avg": bench.get("cpu_percent_system_avg"),
                    "rss_mb": bench.get("rss_mb"),
                    "benchmark_device": bench.get("device"),
                    "module_latency_ms": {
                        "metadata_extraction": bench.get("metadata_extraction_ms"),
                        "feature_normalization": bench.get("feature_normalization_ms"),
                        "windowing": bench.get("windowing_ms"),
                        "behavioral_graph": bench.get("behavioral_graph_ms"),
                        "temporal_graph_inference": bench.get("temporal_graph_inference_ms"),
                        "ood_scoring_fusion": bench.get("ood_scoring_fusion_ms"),
                        "record_attribution": bench.get("record_attribution_ms"),
                    },
                }
            )
        summary.append(phase_summary)
    (out / "mission_phase_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"mission_phase_proxy_summary": summary}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
