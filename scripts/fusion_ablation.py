#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

FUSIONS = ["conf", "energy", "proto", "knn", "hard_voting", "mean", "variance_weighted", "correlation_aware"]


def main():
    p = argparse.ArgumentParser(description="Run OOD fusion-strategy ablation by retraining/calibrating each fusion setting.")
    p.add_argument("--input", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--train_args", default="", help="Extra arguments passed verbatim to scripts/train.py")
    args = p.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    results = {}
    for fusion in FUSIONS:
        sub = out / fusion
        cmd = [sys.executable, str(Path(__file__).with_name("train.py")), "--input", args.input, "--output_dir", str(sub), "--fusion", fusion]
        if args.train_args:
            cmd += args.train_args.split()
        print("RUN", " ".join(cmd))
        subprocess.run(cmd, check=True)
        report = json.loads((sub / "eval_report.json").read_text(encoding="utf-8"))
        results[fusion] = report.get("ood_test", report.get("id_test", {}))
    (out / "fusion_ablation_summary.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
