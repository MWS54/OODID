#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ucs_oodid.synthetic import save_synthetic


def main():
    p = argparse.ArgumentParser(description="Generate synthetic metadata-only UAV traffic.")
    p.add_argument("--output", required=True)
    p.add_argument("--records", type=int, default=5000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    save_synthetic(args.output, records=args.records, seed=args.seed)
    print(f"Saved synthetic dataset to {args.output}")


if __name__ == "__main__":
    main()
