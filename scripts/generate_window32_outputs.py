#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return read_json(path)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def backup_raw(path: Path) -> Path:
    raw_path = path.with_name(f"{path.stem}_raw{path.suffix}")
    if path.exists() and not raw_path.exists():
        shutil.copyfile(path, raw_path)
    return raw_path if raw_path.exists() else path


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )


def parse_metric(value: Any) -> float:
    if value is None:
        return float("nan")
    if isinstance(value, (int, float, np.number)):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else float("nan")
    text = str(value).strip()
    if not text:
        return float("nan")
    if "±" in text:
        text = text.split("±", 1)[0].strip()
    try:
        numeric = float(text)
    except ValueError:
        return float("nan")
    return numeric if math.isfinite(numeric) else float("nan")


def format_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(numeric):
        return ""
    if abs(numeric - round(numeric)) < 1e-12 and abs(numeric) >= 100:
        return str(int(round(numeric)))
    return f"{numeric:.4f}"


def parse_bool_like(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y", "on"}


def markdown_table(frame: pd.DataFrame) -> str:
    columns = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(format_cell(value) for value in row) + " |")
    return "\n".join(lines)


def save_single_row_csv(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(path, index=False)


def normalize_comparison_tables(comparison_dir: Path) -> None:
    known_csv = comparison_dir / "known_detection_table.csv"
    raw_known_csv = backup_raw(known_csv)
    known_frame = pd.read_csv(raw_known_csv)
    if "Subset Acc." in known_frame.columns and "Subset Accuracy" not in known_frame.columns:
        known_frame = known_frame.rename(columns={"Subset Acc.": "Subset Accuracy"})
    known_frame.to_csv(known_csv, index=False)


def normalize_robustness_table(robustness_dir: Path) -> None:
    table_csv = robustness_dir / "homogeneous_vs_heterogeneous_table.csv"
    raw_csv = backup_raw(table_csv)
    frame = pd.read_csv(raw_csv)
    if "FPR95 Increase" in frame.columns and "FPR95 Change" not in frame.columns:
        frame = frame.rename(columns={"FPR95 Increase": "FPR95 Change"})
    frame.to_csv(table_csv, index=False)


def normalize_ablation_table(ablation_dir: Path) -> None:
    csv_path = ablation_dir / "ablation_summary.csv"
    raw_csv = backup_raw(csv_path)
    frame = pd.read_csv(raw_csv)
    friendly = pd.DataFrame(
        {
            "Variant": frame["experiment_name"],
            "Normalization": frame["normalization_mode"],
            "Fusion": frame["fusion"],
            "OOD Threshold Mode": frame["ood_threshold_mode"],
            "Group Threshold Strategy": frame["group_threshold_strategy"],
            "Group Embedding": frame["use_group_embedding"].map(
                lambda value: "Yes" if parse_bool_like(value) else "No"
            ),
            "Micro-F1": frame["id_micro_f1"],
            "Macro-F1": frame["id_macro_f1"],
            "mAP": frame["id_mAP"],
            "AUROC": frame["ood_auroc"],
            "AUPR-Out": frame["ood_aupr_out"],
            "OOD-F1": frame["ood_f1"],
            "FPR95": frame["ood_fpr95"],
            "FPR@theta": frame["fpr_at_threshold"],
            "Avg. Detection Time (ms/window)": frame["average_detection_time_ms"],
            "Throughput (windows/s)": frame["throughput_windows_per_s"],
            "Test Windows": frame["test_windows"],
            "Status": frame["status"],
            "Output Dir": frame["output_dir"],
        }
    )
    friendly.to_csv(csv_path, index=False)


def create_window_sensitivity_aliases(window_dir: Path) -> tuple[Path, Path]:
    summary_csv = window_dir / "window_size_summary.csv"
    summary_json = window_dir / "window_size_summary.json"
    if not summary_csv.exists() or not summary_json.exists():
        raise FileNotFoundError("window-size sensitivity outputs were not found")
    frame = pd.read_csv(summary_csv)
    alias_frame = frame.rename(
        columns={
            "Window Size": "Window",
            "Avg. Detection Time (ms/window)": "Avg. Time",
            "Throughput (windows/s)": "Throughput",
        }
    ).reindex(
        columns=[
            "Window",
            "Stride",
            "Method",
            "Micro-F1",
            "Macro-F1",
            "mAP",
            "AUROC",
            "OOD-F1",
            "FPR95",
            "Avg. Time",
            "Throughput",
        ]
    )
    alias_csv = window_dir / "window_sensitivity_table.csv"
    alias_frame.to_csv(alias_csv, index=False)

    payload = read_json(summary_json)
    payload["tables"]["window_sensitivity_table_csv"] = str(alias_csv)
    payload["confirmed_settings"] = [
        {"window": 8, "stride": 4},
        {"window": 16, "stride": 8},
        {"window": 32, "stride": 16},
        {"window": 64, "stride": 32},
    ]
    alias_json = window_dir / "window_sensitivity_summary.json"
    write_json(alias_json, payload)
    return alias_csv, alias_json


def build_benchmark_outputs(deployment_dir: Path, artifact_path: Path) -> tuple[dict[str, Any], Path]:
    onboard_raw = read_json(deployment_dir / "benchmark_onboard_raw.json")
    edge_raw = read_json(deployment_dir / "edge_benchmark_raw.json")
    simulation_path = deployment_dir / "simulation_online.json"
    simulation_payload = read_json_if_exists(simulation_path)
    window_count = int(simulation_payload.get("window_count", 0) or 0) if simulation_payload else 0
    ids_energy_wh = float(simulation_payload.get("ids_inference_energy_wh", 0.0) or 0.0) if simulation_payload else 0.0
    energy_per_window_wh = None if window_count <= 0 else float(ids_energy_wh / float(window_count))
    energy_per_window_j = None if energy_per_window_wh is None else float(energy_per_window_wh * 3600.0)
    raw_reports = {
        "benchmark_onboard_raw": str(deployment_dir / "benchmark_onboard_raw.json"),
        "edge_benchmark_raw": str(deployment_dir / "edge_benchmark_raw.json"),
    }
    if simulation_payload is not None:
        raw_reports["simulation_online"] = str(simulation_path)

    report = {
        "artifact": str(artifact_path),
        "input": str(onboard_raw.get("input", "")),
        "window_mode": str(onboard_raw.get("window_mode", "count")),
        "window_size": int(onboard_raw.get("window_size", 32)),
        "stride": int(onboard_raw.get("stride", 16)),
        "device": str(onboard_raw.get("device", edge_raw.get("device", "cpu"))),
        "parameter_count": int(onboard_raw.get("parameter_count", 0) or 0),
        "trainable_parameter_count": int(onboard_raw.get("trainable_parameter_count", 0) or 0),
        "model_size_mb": float(onboard_raw.get("model_size_mb", 0.0) or 0.0),
        "model_weights_size_mb": float(onboard_raw.get("model_weights_size_mb", 0.0) or 0.0),
        "average_inference_time_ms_per_window": float(
            onboard_raw.get("average_window_inference_ms", 0.0) or 0.0
        ),
        "throughput_windows_per_s": float(onboard_raw.get("throughput_windows_per_sec", 0.0) or 0.0),
        "num_windows": int(onboard_raw.get("num_windows", 0) or 0),
        "memory_usage_mb": float(edge_raw.get("rss_mb", 0.0) or 0.0),
        "cpu_usage_percent": float(edge_raw.get("cpu_percent_avg", 0.0) or 0.0),
        "cpu_usage_system_percent": float(edge_raw.get("cpu_percent_system_avg", 0.0) or 0.0),
        "edge_total_latency_ms": float(edge_raw.get("total_ms", 0.0) or 0.0),
        "edge_iterations_completed": int(edge_raw.get("iterations_completed", 0) or 0),
        "energy_per_window_wh": energy_per_window_wh,
        "energy_per_window_j": energy_per_window_j,
        "energy_metrics_ready": simulation_payload is not None,
        "energy_metrics_note": None if simulation_payload is not None else "Unavailable until online replay results are generated.",
        "raw_reports": raw_reports,
    }
    report_path = deployment_dir / "benchmark_report.json"
    write_json(report_path, report)
    table_row = {
        "artifact": str(artifact_path),
        "device": report["device"],
        "window_size": report["window_size"],
        "stride": report["stride"],
        "parameter_count": report["parameter_count"],
        "trainable_parameter_count": report["trainable_parameter_count"],
        "model_size_mb": report["model_size_mb"],
        "model_weights_size_mb": report["model_weights_size_mb"],
        "average_inference_time_ms_per_window": report["average_inference_time_ms_per_window"],
        "throughput_windows_per_s": report["throughput_windows_per_s"],
        "num_windows": report["num_windows"],
        "memory_usage_mb": report["memory_usage_mb"],
        "cpu_usage_percent": report["cpu_usage_percent"],
        "cpu_usage_system_percent": report["cpu_usage_system_percent"],
        "energy_per_window_wh": report["energy_per_window_wh"],
        "energy_per_window_j": report["energy_per_window_j"],
    }
    table_path = deployment_dir / "benchmark_table.csv"
    save_single_row_csv(table_path, table_row)
    return report, table_path


def build_online_replay_summary(deployment_dir: Path) -> tuple[dict[str, Any], Path]:
    payload = read_json(deployment_dir / "simulation_online.json")
    row = {
        "artifact_path": str(payload.get("artifact_path", "")),
        "window_size": int(payload.get("window_size", 32) or 32),
        "stride": int(payload.get("stride", 16) or 16),
        "record_count": int(payload.get("record_count", 0) or 0),
        "window_count": int(payload.get("window_count", 0) or 0),
        "attack_record_count": int(payload.get("attack_record_count", 0) or 0),
        "attack_window_count": int(payload.get("attack_window_count", 0) or 0),
        "ood_window_count": int(payload.get("ood_window_count", 0) or 0),
        "benign_window_count": int(payload.get("benign_window_count", 0) or 0),
        "predicted_alert_window_count": int(payload.get("predicted_alert_window_count", 0) or 0),
        "alert_count": int(payload.get("alert_count", 0) or 0),
        "false_alert_window_count": int(payload.get("false_alert_window_count", 0) or 0),
        "false_alert_count": int(payload.get("false_alert_count", 0) or 0),
        "false_alert_ratio": float(payload.get("false_alert_ratio", 0.0) or 0.0),
        "response_action_count": int(payload.get("response_action_count", 0) or 0),
        "average_alert_delay": float(payload.get("average_alert_delay", 0.0) or 0.0),
        "peak_ood_score": float(payload.get("peak_ood_score", 0.0) or 0.0),
        "average_ood_score": float(payload.get("average_ood_score", 0.0) or 0.0),
        "ids_inference_energy_wh": float(payload.get("ids_inference_energy_wh", 0.0) or 0.0),
        "total_energy_wh": float(payload.get("total_energy_wh", 0.0) or 0.0),
        "average_latency_ms": float(payload.get("average_latency_ms", 0.0) or 0.0),
        "average_cpu_load": float(payload.get("average_cpu_load", 0.0) or 0.0),
        "peak_temperature_c": float(payload.get("peak_temperature_c", 0.0) or 0.0),
        "average_packet_loss": float(payload.get("average_packet_loss", 0.0) or 0.0),
        "average_rssi": float(payload.get("average_rssi", 0.0) or 0.0),
    }
    csv_path = deployment_dir / "online_replay_summary.csv"
    save_single_row_csv(csv_path, row)
    return row, csv_path


def save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def grouped_bar(ax: plt.Axes, frame: pd.DataFrame, metrics: list[str], title: str) -> None:
    methods = frame["Method"].tolist()
    x = np.arange(len(methods))
    width = 0.8 / max(len(metrics), 1)
    for index, metric in enumerate(metrics):
        values = pd.to_numeric(frame[metric], errors="coerce").to_numpy(dtype=float)
        offset = (index - (len(metrics) - 1) / 2.0) * width
        ax.bar(x + offset, values, width=width, label=metric)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=20, ha="right")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.2)


def plot_known_attack_results(comparison_dir: Path, figures_dir: Path) -> Path:
    frame = pd.read_csv(comparison_dir / "known_detection_table.csv")
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), height_ratios=[3, 2])
    grouped_bar(axes[0], frame, ["Micro-F1", "Macro-F1", "mAP", "Subset Accuracy"], "Known-Attack Detection")
    axes[0].legend(ncol=2, frameon=False)
    axes[1].bar(frame["Method"], pd.to_numeric(frame["Hamming Loss"], errors="coerce"), color="#4c78a8")
    axes[1].set_title("Hamming Loss")
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].grid(axis="y", alpha=0.2)
    output_path = figures_dir / "fig_known_attack_results.pdf"
    save_figure(fig, output_path)
    return output_path


def plot_ood_results(comparison_dir: Path, figures_dir: Path) -> Path:
    frame = pd.read_csv(comparison_dir / "ood_detection_table.csv")
    fig, axes = plt.subplots(2, 1, figsize=(9, 8), height_ratios=[3, 2])
    grouped_bar(axes[0], frame, ["AUROC", "AUPR-Out", "Precision", "Recall", "OOD-F1"], "OOD Detection")
    axes[0].legend(ncol=3, frameon=False)
    grouped_bar(axes[1], frame, ["FPR95", "FPR@theta"], "Lower-Is-Better OOD Error")
    axes[1].legend(frameon=False)
    output_path = figures_dir / "fig_ood_results.pdf"
    save_figure(fig, output_path)
    return output_path


def plot_detection_time(comparison_dir: Path, figures_dir: Path) -> Path:
    frame = pd.read_csv(comparison_dir / "detection_time_table.csv")
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.bar(frame["Method"], pd.to_numeric(frame["Avg. Detection Time (ms/window)"], errors="coerce"), color="#f58518")
    ax.set_ylabel("ms/window")
    ax.set_title("Detection Time")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.2)
    output_path = figures_dir / "fig_detection_time.pdf"
    save_figure(fig, output_path)
    return output_path


def plot_throughput(comparison_dir: Path, figures_dir: Path) -> Path:
    frame = pd.read_csv(comparison_dir / "detection_time_table.csv")
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.bar(frame["Method"], pd.to_numeric(frame["Throughput (windows/s)"], errors="coerce"), color="#54a24b")
    ax.set_ylabel("windows/s")
    ax.set_title("Throughput")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.2)
    output_path = figures_dir / "fig_throughput.pdf"
    save_figure(fig, output_path)
    return output_path


def plot_window_sensitivity(window_csv: Path, figures_dir: Path) -> Path:
    frame = pd.read_csv(window_csv)
    ucs = frame[frame["Method"] == "UCS-OODID"].copy()
    ucs = ucs.sort_values("Window")
    fig, axes = plt.subplots(2, 2, figsize=(9, 7))
    metric_specs = [
        ("Micro-F1", "Micro-F1"),
        ("AUROC", "AUROC"),
        ("OOD-F1", "OOD-F1"),
        ("FPR95", "FPR95"),
    ]
    for ax, (column, title) in zip(axes.flatten(), metric_specs):
        ax.plot(ucs["Window"], pd.to_numeric(ucs[column], errors="coerce"), marker="o", color="#4c78a8")
        ax.set_title(title)
        ax.set_xlabel("Window")
        ax.grid(alpha=0.2)
    output_path = figures_dir / "fig_window_sensitivity.pdf"
    save_figure(fig, output_path)
    return output_path


def plot_ablation_results(ablation_dir: Path, figures_dir: Path) -> Path:
    frame = pd.read_csv(ablation_dir / "ablation_summary.csv")
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), height_ratios=[3, 2])
    grouped_bar(axes[0], frame.rename(columns={"Variant": "Method"}), ["Micro-F1", "AUROC", "OOD-F1"], "Ablation Metrics")
    axes[0].legend(frameon=False)
    grouped_bar(
        axes[1],
        frame.rename(columns={"Variant": "Method"}),
        ["FPR95", "FPR@theta"],
        "Threshold Error",
    )
    axes[1].legend(frameon=False)
    output_path = figures_dir / "fig_ablation_results.pdf"
    save_figure(fig, output_path)
    return output_path


def plot_homogeneous_vs_heterogeneous(robustness_dir: Path, figures_dir: Path) -> Path:
    frame = pd.read_csv(robustness_dir / "homogeneous_vs_heterogeneous_table.csv")
    methods = frame["Method"].tolist()
    x = np.arange(len(methods))
    width = 0.35

    homo_micro = [parse_metric(value) for value in frame["Homogeneous Micro-F1"]]
    hetero_micro = [parse_metric(value) for value in frame["Heterogeneous Micro-F1"]]
    homo_ood = [parse_metric(value) for value in frame["Homogeneous OOD-F1"]]
    hetero_ood = [parse_metric(value) for value in frame["Heterogeneous OOD-F1"]]
    fpr95_change = [parse_metric(value) for value in frame["FPR95 Change"]]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))
    axes[0].bar(x - width / 2.0, homo_micro, width=width, label="Homogeneous", color="#4c78a8")
    axes[0].bar(x + width / 2.0, hetero_micro, width=width, label="Heterogeneous", color="#f58518")
    axes[0].set_title("Micro-F1")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(methods, rotation=20, ha="right")
    axes[0].grid(axis="y", alpha=0.2)
    axes[0].legend(frameon=False)

    axes[1].bar(x - width / 2.0, homo_ood, width=width, color="#4c78a8")
    axes[1].bar(x + width / 2.0, hetero_ood, width=width, color="#f58518")
    axes[1].set_title("OOD-F1")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(methods, rotation=20, ha="right")
    axes[1].grid(axis="y", alpha=0.2)

    axes[2].bar(methods, fpr95_change, color="#54a24b")
    axes[2].set_title("FPR95 Change")
    axes[2].tick_params(axis="x", rotation=20)
    axes[2].grid(axis="y", alpha=0.2)

    output_path = figures_dir / "fig_homogeneous_heterogeneous.pdf"
    save_figure(fig, output_path)
    return output_path


def plot_sim_alert_response(simulation_payload: dict[str, Any], figures_dir: Path) -> Path:
    trace = list(simulation_payload.get("ood_trace", []))
    times = [float(item.get("simulation_time_s", 0.0)) for item in trace]
    scores = [float(item.get("ood_score", 0.0)) for item in trace]
    threshold = [float(item.get("ood_threshold", 0.0)) for item in trace]
    alerts = [bool(item.get("is_ood", False)) for item in trace]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(times, scores, marker="o", color="#4c78a8", label="OOD score")
    ax.plot(times, threshold, linestyle="--", color="#e45756", label="Threshold")
    if any(alerts):
        alert_times = [time for time, flag in zip(times, alerts) if flag]
        alert_scores = [score for score, flag in zip(scores, alerts) if flag]
        ax.scatter(alert_times, alert_scores, color="#e45756", s=36, label="Alert")
    ax.set_xlabel("Simulation time (s)")
    ax.set_ylabel("OOD score")
    ax.set_title("Replay Alert Response")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    output_path = figures_dir / "fig_sim_alert_response.pdf"
    save_figure(fig, output_path)
    return output_path


def plot_sim_energy_metrics(simulation_payload: dict[str, Any], figures_dir: Path) -> Path:
    ids_energy = float(simulation_payload.get("ids_inference_energy_wh", 0.0) or 0.0)
    total_energy = float(simulation_payload.get("total_energy_wh", 0.0) or 0.0)
    other_energy = max(total_energy - ids_energy, 0.0)
    energy_per_window_mwh = (
        ids_energy / float(simulation_payload.get("window_count", 1) or 1) * 1000.0
        if float(simulation_payload.get("window_count", 0) or 0) > 0
        else 0.0
    )
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.4))
    axes[0].bar(["IDS", "Other"], [ids_energy, other_energy], color=["#4c78a8", "#bab0ab"])
    axes[0].set_ylabel("Wh")
    axes[0].set_title("Mission Energy Split")
    axes[0].grid(axis="y", alpha=0.2)
    axes[1].bar(["Total energy", "IDS/window"], [total_energy, energy_per_window_mwh], color=["#54a24b", "#f58518"])
    axes[1].set_title("Replay Energy")
    axes[1].grid(axis="y", alpha=0.2)
    output_path = figures_dir / "fig_sim_energy_metrics.pdf"
    save_figure(fig, output_path)
    return output_path


def plot_sim_ids_energy(simulation_payload: dict[str, Any], figures_dir: Path) -> Path:
    ids_energy = float(simulation_payload.get("ids_inference_energy_wh", 0.0) or 0.0)
    total_energy = float(simulation_payload.get("total_energy_wh", 0.0) or 0.0)
    ratio_percent = 0.0 if total_energy <= 0.0 else ids_energy / total_energy * 100.0
    energy_per_window_mwh = (
        ids_energy / float(simulation_payload.get("window_count", 1) or 1) * 1000.0
        if float(simulation_payload.get("window_count", 0) or 0) > 0
        else 0.0
    )
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.bar(
        ["IDS energy (Wh)", "IDS ratio (%)", "IDS/window (mWh)"],
        [ids_energy, ratio_percent, energy_per_window_mwh],
        color=["#4c78a8", "#54a24b", "#f58518"],
    )
    ax.set_title("IDS Energy")
    ax.grid(axis="y", alpha=0.2)
    output_path = figures_dir / "fig_sim_ids_energy.pdf"
    save_figure(fig, output_path)
    return output_path


def plot_sim_system_metrics(simulation_payload: dict[str, Any], figures_dir: Path) -> Path:
    metrics = {
        "Latency (ms)": float(simulation_payload.get("average_latency_ms", 0.0) or 0.0),
        "CPU Load (%)": float(simulation_payload.get("average_cpu_load", 0.0) or 0.0) * 100.0,
        "Peak Temp (C)": float(simulation_payload.get("peak_temperature_c", 0.0) or 0.0),
        "Packet Loss (%)": float(simulation_payload.get("average_packet_loss", 0.0) or 0.0) * 100.0,
        "RSSI (dBm)": float(simulation_payload.get("average_rssi", 0.0) or 0.0),
    }
    fig, axes = plt.subplots(1, len(metrics), figsize=(14, 4.2))
    for ax, (label, value) in zip(axes, metrics.items()):
        ax.bar([label], [value], color="#4c78a8")
        ax.set_title(label)
        ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="y", alpha=0.2)
    output_path = figures_dir / "fig_sim_system_metrics.pdf"
    save_figure(fig, output_path)
    return output_path


def plot_energy_comparison_combined(
    benchmark_report: dict[str, Any],
    simulation_payload: dict[str, Any],
    figures_dir: Path,
) -> Path:
    benchmark_latency = float(benchmark_report.get("average_inference_time_ms_per_window", 0.0) or 0.0)
    replay_latency = float(simulation_payload.get("average_latency_ms", 0.0) or 0.0)
    energy_per_window_mwh = (
        float(benchmark_report.get("energy_per_window_wh", 0.0) or 0.0) * 1000.0
    )
    model_size_mb = float(benchmark_report.get("model_size_mb", 0.0) or 0.0)
    fig, axes = plt.subplots(1, 3, figsize=(10, 4.2))
    axes[0].bar(["Model"], [model_size_mb], color="#4c78a8")
    axes[0].set_title("Model Size (MB)")
    axes[0].grid(axis="y", alpha=0.2)
    axes[1].bar(["Benchmark", "Replay"], [benchmark_latency, replay_latency], color=["#54a24b", "#f58518"])
    axes[1].set_title("Latency Comparison")
    axes[1].grid(axis="y", alpha=0.2)
    axes[2].bar(["Energy/window"], [energy_per_window_mwh], color="#e45756")
    axes[2].set_title("IDS Energy per Window (mWh)")
    axes[2].grid(axis="y", alpha=0.2)
    output_path = figures_dir / "fig_energy_comparison_combined.pdf"
    save_figure(fig, output_path)
    return output_path


def build_report(
    results_root: Path,
    artifact_path: Path,
    benchmark_report: dict[str, Any],
    online_summary_row: dict[str, Any] | None,
) -> Path:
    comparison_dir = results_root / "comparison"
    robustness_dir = results_root / "homogeneous_vs_heterogeneous"
    ablation_dir = results_root / "ablation"
    window_dir = results_root / "window_sensitivity"

    known_frame = pd.read_csv(comparison_dir / "known_detection_table.csv")
    ood_frame = pd.read_csv(comparison_dir / "ood_detection_table.csv")
    time_frame = pd.read_csv(comparison_dir / "detection_time_table.csv")
    robustness_frame = pd.read_csv(robustness_dir / "homogeneous_vs_heterogeneous_table.csv")
    ablation_frame = pd.read_csv(ablation_dir / "ablation_summary.csv")
    window_frame = pd.read_csv(window_dir / "window_sensitivity_table.csv")
    benchmark_frame = pd.read_csv(results_root / "deployment" / "benchmark_table.csv")
    online_summary_csv = results_root / "deployment" / "online_replay_summary.csv"
    online_frame = pd.read_csv(online_summary_csv) if online_summary_csv.exists() else None

    lines = [
        "# Full Experiment Results Report (Window 32 / Stride 16)",
        "",
        f"Results root: `{results_root}`",
        "",
        "Default main setting:",
        "",
        "- `window_size=32`",
        "- `stride=16`",
        "",
        f"Final artifact: `{artifact_path}`",
        "",
        "## Main Comparison",
        "",
        "### Known-Attack Detection",
        "",
        markdown_table(known_frame),
        "",
        "### Unknown-Attack / OOD Detection",
        "",
        markdown_table(ood_frame),
        "",
        "### Detection Efficiency",
        "",
        markdown_table(time_frame),
        "",
        "## Homogeneous vs Heterogeneous Robustness",
        "",
        markdown_table(robustness_frame),
        "",
        "## Ablation Study",
        "",
        markdown_table(ablation_frame),
        "",
        "## Window-Size Sensitivity",
        "",
        markdown_table(window_frame),
        "",
        "## Deployment Benchmark",
        "",
        markdown_table(benchmark_frame),
        "",
        "## Final Artifact",
        "",
        f"- Path: `{artifact_path}`",
        f"- Window size: `{benchmark_report.get('window_size', 32)}`",
        f"- Stride: `{benchmark_report.get('stride', 16)}`",
        "",
    ]
    if online_frame is not None:
        lines.extend(
            [
                "## Online Replay",
                "",
                markdown_table(online_frame),
                "",
            ]
        )
    else:
        lines.extend(
            [
                "## Online Replay",
                "",
                "Online replay generation is currently paused, so this section is intentionally omitted from the current report.",
                "",
            ]
        )
    if online_summary_row is not None:
        lines.extend(
            [
                "## Key Replay Metrics",
                "",
                f"- `record_count`: `{online_summary_row['record_count']}`",
                f"- `window_count`: `{online_summary_row['window_count']}`",
                f"- `predicted_alert_window_count`: `{online_summary_row['predicted_alert_window_count']}`",
                f"- `false_alert_window_count`: `{online_summary_row['false_alert_window_count']}`",
                f"- `false_alert_ratio`: `{format_cell(online_summary_row['false_alert_ratio'])}`",
                f"- `response_action_count`: `{online_summary_row['response_action_count']}`",
                f"- `average_alert_delay`: `{format_cell(online_summary_row['average_alert_delay'])}`",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "## Pending Sections",
                "",
                "- Online replay metrics are pending.",
                "- Replay energy-derived benchmark fields remain unavailable until replay is regenerated.",
                "",
            ]
        )
    report_path = results_root / "full_experiment_results_report_window32_stride16.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def generate_figures(
    results_root: Path,
    benchmark_report: dict[str, Any],
    simulation_payload: dict[str, Any] | None,
) -> list[Path]:
    figures_dir = results_root / "figures"
    comparison_dir = results_root / "comparison"
    robustness_dir = results_root / "homogeneous_vs_heterogeneous"
    ablation_dir = results_root / "ablation"
    window_dir = results_root / "window_sensitivity"
    paths = [
        plot_known_attack_results(comparison_dir, figures_dir),
        plot_ood_results(comparison_dir, figures_dir),
        plot_detection_time(comparison_dir, figures_dir),
        plot_throughput(comparison_dir, figures_dir),
        plot_window_sensitivity(window_dir / "window_sensitivity_table.csv", figures_dir),
        plot_ablation_results(ablation_dir, figures_dir),
        plot_homogeneous_vs_heterogeneous(robustness_dir, figures_dir),
    ]
    if simulation_payload is not None:
        paths.extend(
            [
                plot_sim_alert_response(simulation_payload, figures_dir),
                plot_sim_energy_metrics(simulation_payload, figures_dir),
                plot_sim_ids_energy(simulation_payload, figures_dir),
                plot_sim_system_metrics(simulation_payload, figures_dir),
                plot_energy_comparison_combined(benchmark_report, simulation_payload, figures_dir),
            ]
        )
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate normalized window32 experiment tables, figures, and Markdown report.")
    parser.add_argument("--results_root", required=True)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--deployment_dir", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_matplotlib()

    results_root = Path(args.results_root)
    artifact_path = Path(args.artifact)
    deployment_dir = Path(args.deployment_dir) if str(args.deployment_dir).strip() else results_root / "deployment"

    normalize_comparison_tables(results_root / "comparison")
    normalize_robustness_table(results_root / "homogeneous_vs_heterogeneous")
    normalize_ablation_table(results_root / "ablation")
    window_csv, window_json = create_window_sensitivity_aliases(results_root / "window_sensitivity")
    benchmark_report, benchmark_table = build_benchmark_outputs(deployment_dir, artifact_path)
    simulation_payload = read_json_if_exists(deployment_dir / "simulation_online.json")
    online_summary_row = None
    online_csv = None
    if simulation_payload is not None:
        online_summary_row, online_csv = build_online_replay_summary(deployment_dir)
    figure_paths = generate_figures(results_root, benchmark_report, simulation_payload)
    report_path = build_report(results_root, artifact_path, benchmark_report, online_summary_row)

    manifest = {
        "results_root": str(results_root),
        "artifact_path": str(artifact_path),
        "normalized_outputs": {
            "window_sensitivity_table_csv": str(window_csv),
            "window_sensitivity_summary_json": str(window_json),
            "benchmark_report_json": str(deployment_dir / "benchmark_report.json"),
            "benchmark_table_csv": str(benchmark_table),
            "report_markdown": str(report_path),
        },
        "figure_files": [str(path) for path in figure_paths],
    }
    if online_csv is not None:
        manifest["normalized_outputs"]["online_replay_summary_csv"] = str(online_csv)
    write_json(results_root / "generation_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
