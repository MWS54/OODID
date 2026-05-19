from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
from sklearn.cluster import DBSCAN, KMeans

from .io import read_jsonl, save_json


def cluster_ood_windows(rows: List[dict], method: str = "dbscan", eps: float = 1.2, min_samples: int = 3, n_clusters: Optional[int] = None) -> np.ndarray:
    ood_rows = [r for r in rows if r.get("is_ood", False)]
    if not ood_rows:
        return np.asarray([], dtype=int)
    vectors = []
    for r in ood_rows:
        emb = np.asarray(r.get("embedding", []), dtype=np.float32)
        scores = r.get("normalized_scores", {})
        svec = np.asarray([scores.get(k, 0.0) for k in ["conf", "energy", "proto", "knn"]], dtype=np.float32)
        vectors.append(np.concatenate([emb, svec]))
    x = np.stack(vectors, axis=0)
    if method == "kmeans":
        k = n_clusters or max(1, min(8, int(np.sqrt(len(x)))))
        labels = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(x)
    else:
        labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(x)
    return labels.astype(int)


def summarize_clusters(rows: List[dict], labels: np.ndarray) -> List[dict]:
    ood_rows = [r for r in rows if r.get("is_ood", False)]
    if len(ood_rows) != len(labels):
        raise ValueError("labels must match OOD rows")
    grouped: Dict[int, List[dict]] = defaultdict(list)
    for row, lab in zip(ood_rows, labels):
        grouped[int(lab)].append(row)
    summaries = []
    for lab, items in sorted(grouped.items(), key=lambda kv: kv[0]):
        pred_counter = Counter()
        source_counter = Counter()
        top_records = []
        for r in items:
            pred_counter.update(r.get("known_labels", []))
            source_counter.update([r.get("dominant_ood_source", "unknown")])
            top_records.extend(r.get("top_suspicious_records", [])[:5])
        record_counter = Counter([str(x.get("record_id")) for x in top_records])
        avg_scores = {
            name: float(np.mean([r.get("normalized_scores", {}).get(name, 0.0) for r in items]))
            for name in ["conf", "energy", "proto", "knn"]
        }
        fused = [r.get("ood_score", 0.0) for r in items]
        summaries.append({
            "cluster_id": int(lab),
            "num_windows": len(items),
            "avg_ood_score": float(np.mean(fused)),
            "dominant_known_predictions": pred_counter.most_common(5),
            "dominant_ood_source": source_counter.most_common(1)[0][0] if source_counter else "unknown",
            "avg_normalized_scores": avg_scores,
            "representative_windows": [r.get("window_id") for r in items[:5]],
            "representative_suspicious_records": record_counter.most_common(10),
            "semantic_hypothesis": infer_semantic_hypothesis(avg_scores, source_counter.most_common(1)[0][0] if source_counter else "unknown"),
        })
    return summaries


def infer_semantic_hypothesis(avg_scores: dict, dominant_source: str) -> str:
    if dominant_source == "knn":
        return "weak neighborhood consistency; possible new mission behavior, unseen relay path, or low-rate probing."
    if dominant_source == "proto":
        return "large prototype distance; possible attack-family shift or mission-phase distribution drift."
    if dominant_source == "energy":
        return "abnormal logit energy; possible high-rate burst, replay-like repetition, or dense anomalous block."
    if dominant_source == "conf":
        return "low known-class confidence; possible ambiguous mixed window or unseen traffic subtype."
    return "unresolved unknown pattern requiring analyst review."


def write_markdown_report(summaries: List[dict], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# UCS-OODID Offline Unknown-pattern Triage Report", ""]
    for s in summaries:
        lines += [
            f"## Cluster {s['cluster_id']}",
            f"- Windows: {s['num_windows']}",
            f"- Average OOD score: {s['avg_ood_score']:.4f}",
            f"- Dominant OOD evidence source: {s['dominant_ood_source']}",
            f"- Semantic hypothesis: {s['semantic_hypothesis']}",
            f"- Dominant known predictions: {s['dominant_known_predictions']}",
            f"- Representative windows: {s['representative_windows']}",
            f"- Representative suspicious records: {s['representative_suspicious_records']}",
            "",
        ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def run_offline_triage(detections_path: str | Path, output_dir: str | Path, method: str = "dbscan", eps: float = 1.2, min_samples: int = 3, n_clusters: Optional[int] = None) -> List[dict]:
    rows = read_jsonl(detections_path)
    labels = cluster_ood_windows(rows, method=method, eps=eps, min_samples=min_samples, n_clusters=n_clusters)
    summaries = summarize_clusters(rows, labels)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_json(summaries, out / "cluster_summary.json")
    write_markdown_report(summaries, out / "report.md")
    return summaries
