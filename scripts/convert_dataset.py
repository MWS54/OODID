#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ucs_oodid.dataset_adapters import convert_dataset
from ucs_oodid.io import load_records


def main():
    p = argparse.ArgumentParser(description="Convert public UAV/CIC/USTC datasets to UCS-OODID canonical metadata schema.")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--dataset", default="generic", choices=["generic", "cicids2017", "ustc-tfc2016", "mavlink", "uav-gcs-ids", "uav-cyber-physical"])
    p.add_argument("--label_col", default=None)
    p.add_argument("--timestamp_col", default=None)
    p.add_argument("--record_id_col", default=None)
    p.add_argument("--notes_json", default=None)
    args = p.parse_args()
    df = load_records(args.input)
    res = convert_dataset(df, args.dataset, label_col=args.label_col, timestamp_col=args.timestamp_col, record_id_col=args.record_id_col)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix.lower() == ".parquet":
        res.dataframe.to_parquet(out, index=False)
    else:
        res.dataframe.to_csv(out, index=False)
    notes = {"dataset": res.dataset_name, "rows": len(res.dataframe), "feature_columns": res.feature_columns, "notes": res.notes}
    if args.notes_json:
        Path(args.notes_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.notes_json).write_text(json.dumps(notes, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(notes, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
