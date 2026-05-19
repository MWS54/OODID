#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ucs_oodid.reporting.text_summary import export_metric_summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export Markdown/TXT/JSON/CSV/LaTeX metric summaries, including per-UAV Macro-F1 and Present-Class Macro-F1 breakdowns, from reports/*.json artifacts.",
    )
    parser.add_argument(
        "--reports_dir",
        default=str(ROOT / "reports"),
        help="Directory containing report.json, simulation_summary.json, benchmark_report.json, group_detection_summary.json, ablation_summary.json, and cross_uav_summary.json.",
    )
    args = parser.parse_args()
    payload = export_metric_summary(args.reports_dir)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
