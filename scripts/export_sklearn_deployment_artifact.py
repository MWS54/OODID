#!/usr/bin/env python3
"""Export a replay-compatible sklearn-tabular artifact from the same training protocol as train_sklearn_baselines."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from ucs_oodid.artifacts import save_artifact
from ucs_oodid.baselines import (
    calibrate_binary_threshold_from_id_scores,
    compute_baseline_uncertainty_scores,
    make_baseline,
)
from ucs_oodid.experiment_utils import prepare_window_dataset
from ucs_oodid.io import save_json
from ucs_oodid.ood import calibrate_class_thresholds

DEFAULT_OOD_SCORE_BY_BASELINE = {
    "svm": "uncertainty",
    "random_forest": "uncertainty",
    "xgboost": "uncertainty",
    "mlp": "energy",
}
from ucs_oodid.utils import set_seed


def _ensure_prob_matrix(probs: np.ndarray) -> np.ndarray:
    arr = np.asarray(probs, dtype=np.float32)
    if arr.ndim == 1:
        return arr[:, None].astype(np.float32)
    return arr.astype(np.float32)


def _class_names_ordered(class_to_idx: dict[str, int]) -> list[str]:
    return [name for name, _ in sorted(class_to_idx.items(), key=lambda kv: kv[1])]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train sklearn baseline on training splits and export deployment artifact.pt.")
    p.add_argument("--input", required=True)
    p.add_argument("--output_dir", required=True, help="Directory to write artifact.pt and eval_report.json sidecar.")
    p.add_argument("--label_col", default="label")
    p.add_argument("--timestamp_col", default="timestamp")
    p.add_argument("--record_id_col", default="record_id")
    p.add_argument("--group_col", default="uav_id")
    p.add_argument("--id_classes", default="", help="Comma-separated ID classes (same as comparison suite when nonempty).")
    p.add_argument("--ood_classes", default="", help="Comma-separated OOD-only classes used at test time.")
    p.add_argument("--benign_label", default="benign")
    p.add_argument("--allow_ports", action="store_true")
    p.add_argument("--window_mode", choices=["count", "time", "adaptive"], default="count")
    p.add_argument("--window_size", type=int, default=32)
    p.add_argument("--stride", type=int, default=16)
    p.add_argument("--time_window_seconds", type=float, default=2.0)
    p.add_argument("--adaptive_min_size", type=int, default=8)
    p.add_argument("--adaptive_max_size", type=int, default=64)
    p.add_argument("--normalization_mode", default="global", choices=["global", "group"])
    p.add_argument("--baseline", required=True, help="Backend name: random_forest or svm.")
    p.add_argument("--method_name", default="", help="Stored method_name / proxy key (e.g. rapier_proxy).")
    p.add_argument("--display_name", default="", help="Human-readable method label.")
    p.add_argument("--q_ood", type=float, default=0.90)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bundle = prepare_window_dataset(args, require_test_ood=True)
    baseline = make_baseline(args.baseline, random_state=args.seed)
    train_w = bundle.split_windows["train"]
    val_w = bundle.split_windows["val"]
    baseline.fit(train_w.x, train_w.y)

    val_probs = _ensure_prob_matrix(baseline.predict_proba(val_w.x))
    class_thresholds = calibrate_class_thresholds(val_probs, val_w.y)
    ood_score_name = DEFAULT_OOD_SCORE_BY_BASELINE[str(args.baseline).strip().lower()]
    score_key = "conf" if ood_score_name == "uncertainty" else ("energy" if ood_score_name == "energy" else "conf")
    val_ood_scores = np.asarray(compute_baseline_uncertainty_scores(val_probs)[score_key], dtype=np.float32)
    ood_threshold = float(calibrate_binary_threshold_from_id_scores(val_ood_scores, q=args.q_ood))

    class_names = _class_names_ordered(bundle.class_to_idx)
    method_key = str(args.method_name).strip() or str(args.baseline).strip().lower()
    display_name = str(args.display_name).strip() or method_key

    window_cfg = dict(bundle.window_config)
    pre = bundle.preprocessor

    artifact: dict = {
        "deployment_backend": "sklearn_tabular",
        "method_name": method_key,
        "display_name": display_name,
        "backend_name": str(args.baseline).strip().lower(),
        "seed": int(args.seed),
        "normalization_mode": str(args.normalization_mode),
        "feature_cols": list(pre.feature_cols),
        "preprocessor": pre,
        "global_scaler": getattr(pre, "scaler", None),
        "group_col": getattr(pre, "group_col", None),
        "window_config": window_cfg,
        "class_names": class_names,
        "class_to_idx": dict(bundle.class_to_idx),
        "class_thresholds": np.asarray(class_thresholds, dtype=np.float32),
        "sklearn_baseline": baseline,
        "ood_threshold": float(ood_threshold),
        "calibration_threshold": float(ood_threshold),
        "ood_score_name": ood_score_name,
        "calibration_config": {
            "q_ood": float(args.q_ood),
            "ood_score_key": score_key,
            "threshold_fit_split": "val_id_only",
        },
        "strict_model_feature_mode": True,
        "simulation_score_mode": "raw",
        "simulation_score_normalization_mode": "none",
        "run_config": {
            "seed": int(args.seed),
            "normalization_mode": str(args.normalization_mode),
            "ood_threshold_mode": "global",
            "group_threshold_strategy": "raw",
        },
        "group_ood_thresholds": {},
        "conservative_group_thresholds": {},
        "temperature": 1.0,
        "model_config": {"input_dim": len(pre.feature_cols), "num_classes": len(class_names), "encoder_ablation": "sklearn_tabular"},
        "graph_config": {},
    }

    save_artifact(output_dir / "artifact.pt", artifact)
    save_json(list(pre.feature_cols), output_dir / "feature_columns.json")
    save_json(
        {
            "status": "success",
            "method_name": method_key,
            "display_name": display_name,
            "ood_threshold": float(ood_threshold),
            "ood_score_name": ood_score_name,
            "artifact_path": str(output_dir / "artifact.pt"),
        },
        output_dir / "eval_report.json",
    )
    print(f"Wrote {output_dir / 'artifact.pt'}")


if __name__ == "__main__":
    main()
