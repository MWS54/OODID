#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ucs_oodid.offline import run_offline_triage


def main():
    p = argparse.ArgumentParser(description="Cluster OOD windows and generate analyst report.")
    p.add_argument("--detections", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--method", default="dbscan", choices=["dbscan", "kmeans"])
    p.add_argument("--eps", type=float, default=1.2)
    p.add_argument("--min_samples", type=int, default=3)
    p.add_argument("--n_clusters", type=int, default=None)
    args = p.parse_args()
    summaries = run_offline_triage(args.detections, args.output_dir, method=args.method, eps=args.eps, min_samples=args.min_samples, n_clusters=args.n_clusters)
    print(f"Generated {len(summaries)} cluster summaries in {args.output_dir}")


if __name__ == "__main__":
    main()
