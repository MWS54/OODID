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

from ucs_oodid.io import load_json, load_records
from ucs_oodid.metrics import attribution_metrics
from ucs_oodid.utils import parse_csv_list


def extract_record_scores(payload: dict) -> tuple[dict, list[dict], dict]:
    ranking_payload = payload.get("record_level_suspiciousness_ranking", {})
    scores = payload.get("record_scores", payload if isinstance(payload, dict) else {})
    ranked_records = ranking_payload.get("ranked_records", [])
    return scores, ranked_records, ranking_payload


def write_topk_csv(rows: list[dict], path: str | Path) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["record_id", "window_id", "score", "label", "predicted_ood_score", "attention_weight"]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in fieldnames})


def main():
    p = argparse.ArgumentParser(description="RQ7 record-level suspiciousness-ranking metrics from record_scores.json and malicious record labels.")
    p.add_argument("--record_scores_json", required=True)
    p.add_argument("--records", required=True, help="Canonical record file containing labels and record_id.")
    p.add_argument("--label_col", default="label")
    p.add_argument("--record_id_col", default="record_id")
    p.add_argument("--malicious_labels", default="", help="Comma-separated malicious labels. If omitted, all non-benign labels are malicious.")
    p.add_argument("--benign_label", default="benign")
    p.add_argument("--topk", default="20,50,100")
    p.add_argument("--output_topk_csv", default="", help="Optional CSV export for Top-K suspicious records.")
    p.add_argument("--output_topk_csv_k", type=int, default=0, help="Optional K for CSV export. Defaults to max(--topk).")
    p.add_argument("--output_json", required=True)
    args = p.parse_args()
    obj = load_json(args.record_scores_json)
    scores, ranked_records, ranking_payload = extract_record_scores(obj)
    df = load_records(args.records)
    if args.record_id_col not in df.columns:
        df[args.record_id_col] = range(len(df))
    mal_labels = set(parse_csv_list(args.malicious_labels))
    if mal_labels:
        mask = df[args.label_col].astype(str).isin(mal_labels)
    else:
        mask = df[args.label_col].astype(str) != args.benign_label
    malicious_ids = set(df.loc[mask, args.record_id_col].astype(str))
    label_lookup = {str(rid): str(label) for rid, label in zip(df[args.record_id_col].astype(str), df[args.label_col].astype(str))}
    topk_values = [int(k) for k in parse_csv_list(args.topk)]
    results = {str(k): attribution_metrics(scores, malicious_ids, topk=int(k)) for k in topk_values}

    if args.output_topk_csv:
        export_k = int(args.output_topk_csv_k) if args.output_topk_csv_k > 0 else max(topk_values)
        if ranked_records:
            rows = [
                {
                    "record_id": str(row.get("record_id")),
                    "window_id": row.get("window_id"),
                    "score": float(row.get("score", row.get("suspiciousness", 0.0))),
                    "label": row.get("label", label_lookup.get(str(row.get("record_id")))),
                    "predicted_ood_score": row.get("predicted_ood_score"),
                    "attention_weight": row.get("attention_weight"),
                }
                for row in ranked_records[:export_k]
            ]
        else:
            rows = []
            for rid, score in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:export_k]:
                rows.append(
                    {
                        "record_id": str(rid),
                        "window_id": None,
                        "score": float(score),
                        "label": label_lookup.get(str(rid)),
                        "predicted_ood_score": None,
                        "attention_weight": None,
                    }
                )
        write_topk_csv(rows, args.output_topk_csv)

    report = {
        "record_level_suspiciousness_ranking": {
            "ranking_type": ranking_payload.get("ranking_type", "analyst_triage_ranking"),
            "note": ranking_payload.get(
                "note",
                "record-level suspiciousness ranks analyst triage priority and is not complete per-record attack localization.",
            ),
            "malicious_records": len(malicious_ids),
            "metric_notes": {
                "hit_rate": "Top-K hit_rate indicates whether at least one malicious record appears in the K most suspicious records.",
                "precision": "Top-K precision is the proportion of malicious records among the K most suspicious records.",
                "recall": "Top-K recall may remain low because this ranking is designed for analyst triage rather than full malicious-record coverage.",
                "f1": "Top-K f1 summarizes the precision/recall trade-off for the suspiciousness ranking at the chosen K.",
                "analyst_reduction": "analyst_reduction is the fraction of records that analysts do not need to inspect after restricting review to the Top-K suspicious records.",
            },
            "topk_metrics": results,
        },
        "malicious_records": len(malicious_ids),
        "metrics": results,
    }
    if args.output_topk_csv:
        report["record_level_suspiciousness_ranking"]["topk_csv"] = str(Path(args.output_topk_csv))
        report["record_level_suspiciousness_ranking"]["topk_csv_k"] = int(args.output_topk_csv_k) if args.output_topk_csv_k > 0 else max(topk_values)
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(results, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
