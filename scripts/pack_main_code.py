#!/usr/bin/env python3
"""Pack main project source into a zip (excludes runs, data, models, caches)."""
from __future__ import annotations

import zipfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT.parent

SKIP_DIRS = {
    "runs",
    "data",
    "__pycache__",
    ".pytest_cache",
    ".git",
    ".venv",
    "venv",
    "collected_main_project_code",
    "collected_multi_uav_modified_code_20260502_1116",
    "node_modules",
    ".mypy_cache",
    ".ruff_cache",
}
SKIP_SUFFIXES = {".pyc", ".pyo", ".pt", ".pth", ".joblib", ".ckpt", ".npy", ".npz"}
SKIP_NAMES = {".DS_Store", "Thumbs.db"}


def should_skip(rel: Path) -> bool:
    if any(part in SKIP_DIRS for part in rel.parts):
        return True
    if rel.name in SKIP_NAMES:
        return True
    return rel.suffix.lower() in SKIP_SUFFIXES


def main() -> None:
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    out = OUT_DIR / f"ucs_oodid_project_main_code_{stamp}.zip"
    file_count = 0
    source_bytes = 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for path in ROOT.rglob("*"):
            if not path.is_file():
                continue
            rel = path.relative_to(ROOT)
            if should_skip(rel):
                continue
            zf.write(path, f"ucs_oodid_project/{rel.as_posix()}")
            file_count += 1
            source_bytes += path.stat().st_size
    zip_mb = out.stat().st_size / (1024 * 1024)
    print(f"archive: {out}")
    print(f"files: {file_count}")
    print(f"uncompressed_source_mb: {source_bytes / (1024 * 1024):.2f}")
    print(f"zip_size_mb: {zip_mb:.2f}")


if __name__ == "__main__":
    main()
