#!/usr/bin/env python3
"""Build replay-compatible artifacts under runs/final_* for paper simulation (32/16 suite settings)."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_INPUT = ROOT / "data" / "multi_uav_hetero_no_uav04_plus_uav06_uav07_experiment.csv"
DEFAULT_ID = (
    "benign,icmp_flooding,udp_flooding,dos,ddos,bruteforce,mitm,"
    "deauthentication,jamming,injection,ip_spoofing,payload_manipulation,"
    "recon_scanning,generic,exploits,fuzzers,sybil"
)
DEFAULT_OOD = (
    "replay,reply,fake_landing,evil,video_interception,unauthorized_udp,"
    "analysis,backdoor,shellcode,wormhole,blackhole,worms"
)


def run(cmd: list[object]) -> None:
    cmd_list = [str(part) for part in cmd]
    print("RUN", " ".join(cmd_list), flush=True)
    subprocess.run(cmd_list, check=True, cwd=str(ROOT))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train/export methods into runs/final_* deployment dirs.")
    p.add_argument("--input", type=str, default=str(DEFAULT_INPUT))
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--q_ood", type=float, default=0.90)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    p.add_argument("--group_col", default="uav_id")
    p.add_argument("--id_classes", default=DEFAULT_ID)
    p.add_argument("--ood_classes", default=DEFAULT_OOD)
    p.add_argument("--skip_sklearn", action="store_true")
    p.add_argument("--skip_neural", action="store_true")
    return p.parse_args()


def shared_train_prefix(args: argparse.Namespace) -> list[object]:
    return [
        sys.executable,
        ROOT / "scripts" / "train.py",
        "--input",
        args.input,
        "--label_col",
        "label",
        "--timestamp_col",
        "timestamp",
        "--record_id_col",
        "record_id",
        "--group_col",
        args.group_col,
        "--id_classes",
        args.id_classes,
        "--ood_classes",
        args.ood_classes,
        "--window_mode",
        "count",
        "--window_size",
        "32",
        "--stride",
        "16",
        "--time_window_seconds",
        "2.0",
        "--adaptive_min_size",
        "8",
        "--adaptive_max_size",
        "64",
        "--q_ood",
        str(args.q_ood),
        "--seed",
        str(args.seed),
        "--epochs",
        str(args.epochs),
        "--normalization_mode",
        "global",
        "--ood_threshold_mode",
        "global",
        "--group_threshold_strategy",
        "raw",
        "--bank_k",
        "5",
        "--device",
        args.device,
    ]


def main() -> None:
    args = parse_args()
    if not args.skip_sklearn:
        run(
            [
                sys.executable,
                ROOT / "scripts" / "export_sklearn_deployment_artifact.py",
                "--input",
                args.input,
                "--output_dir",
                ROOT / "runs" / "final_rapier_style_window32_stride16",
                "--baseline",
                "random_forest",
                "--method_name",
                "rapier_proxy",
                "--display_name",
                "RAPIER-style",
                "--normalization_mode",
                "global",
                "--window_size",
                "32",
                "--stride",
                "16",
                "--group_col",
                args.group_col,
                "--id_classes",
                args.id_classes,
                "--ood_classes",
                args.ood_classes,
                "--q_ood",
                str(args.q_ood),
                "--seed",
                str(args.seed),
            ]
        )
        run(
            [
                sys.executable,
                ROOT / "scripts" / "export_sklearn_deployment_artifact.py",
                "--input",
                args.input,
                "--output_dir",
                ROOT / "runs" / "final_rids_style_window32_stride16",
                "--baseline",
                "svm",
                "--method_name",
                "rids_lite_proxy",
                "--display_name",
                "RIDS-style",
                "--normalization_mode",
                "global",
                "--window_size",
                "32",
                "--stride",
                "16",
                "--group_col",
                args.group_col,
                "--id_classes",
                args.id_classes,
                "--ood_classes",
                args.ood_classes,
                "--q_ood",
                str(args.q_ood),
                "--seed",
                str(args.seed),
            ]
        )

    if not args.skip_neural:
        hyper_cmd = shared_train_prefix(args) + [
            "--output_dir",
            ROOT / "runs" / "final_hypervision_style_window32_stride16",
            "--encoder_ablation",
            "full",
            "--fusion",
            "mean",
        ]
        recda_cmd = shared_train_prefix(args) + [
            "--output_dir",
            ROOT / "runs" / "final_recda_style_window32_stride16",
            "--encoder_ablation",
            "transformer_only",
            "--fusion",
            "correlation_aware",
        ]
        run(hyper_cmd)
        run(recda_cmd)


if __name__ == "__main__":
    main()
