"""Minimal end-to-end smoke test.

Run from project root after installing requirements:

python tests/smoke_test.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "examples" / "smoke_synthetic.csv"
RUN = ROOT / "runs" / "smoke"


def run(cmd):
    print("RUN", " ".join(map(str, cmd)))
    subprocess.run(list(map(str, cmd)), check=True)


def main():
    run([sys.executable, ROOT / "scripts" / "make_synthetic.py", "--output", DATA, "--records", 800, "--seed", 7])
    run([
        sys.executable, ROOT / "scripts" / "train.py",
        "--input", DATA,
        "--output_dir", RUN,
        "--id_classes", "benign,dos,mitm,spoof,replay,injection,scan",
        "--ood_classes", "unknown_probe,unknown_burst",
        "--epochs", "1",
        "--batch_size", "64",
        "--hidden_dim", "32",
        "--num_heads", "4",
        "--num_layers", "1",
    ])
    run([
        sys.executable, ROOT / "scripts" / "detect.py",
        "--input", DATA,
        "--artifact", RUN / "artifact.pt",
        "--output_jsonl", RUN / "detections.jsonl",
        "--record_scores_json", RUN / "record_scores.json",
    ])
    run([
        sys.executable, ROOT / "scripts" / "offline_triage.py",
        "--detections", RUN / "detections.jsonl",
        "--output_dir", RUN / "offline_report",
    ])
    print("Smoke test completed.")


if __name__ == "__main__":
    main()
