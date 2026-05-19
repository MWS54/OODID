#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ucs_oodid.artifacts import load_artifact


def flatten_report(report: dict, prefix: str = ""):
    rows = {}
    for k, v in report.items():
        key = f"{prefix}{k}" if not prefix else f"{prefix}.{k}"
        if isinstance(v, dict):
            rows.update(flatten_report(v, key))
        elif isinstance(v, (int, float, str, bool)) or v is None:
            rows[key] = v
    return rows


def write_eval_csv(root: Path, output_csv: Path):
    reports = []
    for path in root.rglob("eval_report.json"):
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        row = {"run_dir": str(path.parent)}
        row.update(flatten_report(report))
        reports.append(row)
    if not reports:
        return 0
    keys = sorted({k for r in reports for k in r})
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader(); writer.writerows(reports)
    return len(reports)


def plot_correlation(artifact_path: Path, output_pdf: Path):
    import matplotlib.pyplot as plt
    import numpy as np
    artifact = load_artifact(artifact_path, map_location="cpu")
    cal = artifact.get("ood_calibrator", {})
    corr = cal.get("correlation")
    names = cal.get("score_names", ["conf", "energy", "proto", "knn"])
    if corr is None:
        raise ValueError("No correlation matrix found in artifact. Train first with OOD calibration.")
    corr = np.asarray(corr, dtype=float)
    fig = plt.figure(figsize=(4.8, 4.2))
    ax = fig.add_subplot(111)
    im = ax.imshow(corr, vmin=-1, vmax=1)
    ax.set_xticks(range(len(names))); ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha="right"); ax.set_yticklabels(names)
    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            ax.text(j, i, f"{corr[i,j]:.2f}", ha="center", va="center", fontsize=8)
    ax.set_title("OOD score correlation")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_pdf)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description="Generate paper tables and score-correlation heatmap from experiment outputs.")
    p.add_argument("--results_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--artifact", default="", help="Optional artifact.pt for score_correlation_heatmap.pdf")
    args = p.parse_args()
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    n = write_eval_csv(Path(args.results_dir), out / "eval_report_summary.csv")
    made = {"eval_rows": n, "eval_csv": str(out / "eval_report_summary.csv")}
    if args.artifact:
        heatmap = out / "score_correlation_heatmap.pdf"
        plot_correlation(Path(args.artifact), heatmap)
        made["score_correlation_heatmap"] = str(heatmap)
    (out / "figure_table_manifest.json").write_text(json.dumps(made, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(made, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
