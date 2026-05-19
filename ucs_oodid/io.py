from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd


def load_records(path: str | Path) -> pd.DataFrame:
    """Load metadata records from CSV, JSONL, JSON, or Parquet."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".jsonl", ".ndjson"}:
        return pd.read_json(path, lines=True)
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, list):
            return pd.DataFrame(obj)
        if isinstance(obj, dict):
            for key in ("records", "data", "items"):
                if key in obj and isinstance(obj[key], list):
                    return pd.DataFrame(obj[key])
            return pd.DataFrame([obj])
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported input format: {path}")


def save_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def save_records(df: pd.DataFrame, path: str | Path) -> None:
    """Save metadata records to CSV, JSONL, JSON, or Parquet."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df.to_csv(path, index=False)
        return
    if suffix in {".jsonl", ".ndjson"}:
        df.to_json(path, orient="records", lines=True, force_ascii=False)
        return
    if suffix == ".json":
        records = json.loads(df.to_json(orient="records", force_ascii=False))
        with path.open("w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)
        return
    if suffix in {".parquet", ".pq"}:
        df.to_parquet(path, index=False)
        return
    raise ValueError(f"Unsupported output format: {path}")


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def write_jsonl(rows: Iterable[Dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def ensure_record_id(df: pd.DataFrame, record_id_col: Optional[str] = None) -> tuple[pd.DataFrame, str]:
    df = df.copy()
    if record_id_col and record_id_col in df.columns:
        return df, record_id_col
    if "record_id" not in df.columns:
        df["record_id"] = list(range(len(df)))
    return df, "record_id"
