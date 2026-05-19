#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

VARIANTS = [
    ("cosine_sym_weighted", ["--graph_metric", "cosine", "--graph_variant", "sym_weighted"]),
    ("rbf_sym_weighted", ["--graph_metric", "rbf", "--graph_variant", "sym_weighted"]),
    ("directed_knn", ["--graph_metric", "cosine", "--graph_variant", "directed"]),
    ("mutual_knn", ["--graph_metric", "cosine", "--graph_variant", "mutual"]),
    ("binary_edges", ["--graph_metric", "cosine", "--graph_variant", "binary"]),
    ("identity_graph", ["--graph_metric", "cosine", "--graph_variant", "identity_graph", "--encoder_ablation", "full"]),
]


def main():
    p = argparse.ArgumentParser(description="Run behavioral graph-construction ablation.")
    p.add_argument("--input", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--train_args", default="", help="Extra arguments passed to scripts/train.py")
    args = p.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    results = {}
    for name, extra in VARIANTS:
        sub = out / name
        cmd = [sys.executable, str(Path(__file__).with_name("train.py")), "--input", args.input, "--output_dir", str(sub)] + extra
        if args.train_args:
            cmd += args.train_args.split()
        print("RUN", " ".join(cmd))
        subprocess.run(cmd, check=True)
        report = json.loads((sub / "eval_report.json").read_text(encoding="utf-8"))
        merged = {}
        merged.update(report.get("id_test", {}))
        merged.update({f"ood_{k}": v for k, v in report.get("ood_test", {}).items()})
        results[name] = merged
    (out / "graph_ablation_summary.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
