#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ucs_oodid.io import load_records
from ucs_oodid.preprocessing import MetadataPreprocessor
from ucs_oodid.tabular_gan import TabularGanConfig, augment_class_with_wgan
from ucs_oodid.utils import choose_device, parse_csv_list, set_seed


def label_count_dict(df: pd.DataFrame, classes: Sequence[str] | None = None, label_col: str = "label") -> dict:
    if label_col not in df.columns:
        return {}
    raw_counts = {str(k): int(v) for k, v in df[label_col].astype(str).value_counts().to_dict().items()}
    if classes is None:
        return {k: raw_counts[k] for k in sorted(raw_counts)}
    ordered: dict[str, int] = {}
    seen = set()
    for cls in classes:
        key = str(cls)
        ordered[key] = int(raw_counts.get(key, 0))
        seen.add(key)
    for key in sorted(raw_counts):
        if key not in seen:
            ordered[key] = int(raw_counts[key])
    return ordered


def interleave_by_label(df: pd.DataFrame, label_col: str, chunk_size: int, seed: int) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    rng = np.random.default_rng(seed)
    queues: Dict[str, List[int]] = {}
    for label, group in df.groupby(label_col, sort=True):
        idx = group.index.to_numpy(copy=True)
        rng.shuffle(idx)
        queues[str(label)] = idx.tolist()
    order: List[int] = []
    labels = sorted(queues)
    while True:
        active = [label for label in labels if queues[label]]
        if not active:
            break
        rng.shuffle(active)
        for label in active:
            take = min(chunk_size, len(queues[label]))
            order.extend(queues[label][:take])
            del queues[label][:take]
    return df.loc[order].reset_index(drop=True)


def window_mix_stats(labels: Iterable[str], window_size: int = 16, stride: int = 8) -> dict:
    seq = list(labels)
    total = 0
    mixed = 0
    pure = 0
    for start in range(0, max(len(seq) - window_size + 1, 0), stride):
        total += 1
        uniq = set(seq[start:start + window_size])
        if len(uniq) > 1:
            mixed += 1
        else:
            pure += 1
    return {
        "window_size": int(window_size),
        "stride": int(stride),
        "total_windows": int(total),
        "mixed_windows": int(mixed),
        "pure_windows": int(pure),
        "mixed_ratio": float(mixed / total) if total else 0.0,
    }


def resolve_target_count(counts: pd.Series, strategy: str, fixed_target: int, quantile: float) -> int:
    values = counts.to_numpy(dtype=np.int64)
    if values.size == 0:
        raise ValueError("No label counts available to determine augmentation target.")
    if strategy == "fixed":
        if fixed_target <= 0:
            raise ValueError("--target_count must be > 0 when --target_strategy=fixed.")
        return int(fixed_target)
    if strategy == "max":
        return int(values.max())
    if strategy == "quantile":
        return int(np.quantile(values, quantile))
    return int(np.median(values))


def summarise_split(df: pd.DataFrame, label_col: str, ordered_labels: Sequence[str]) -> dict:
    return {
        "rows": int(len(df)),
        "label_counts": label_count_dict(df, ordered_labels, label_col=label_col),
        "mixed_window_stats": window_mix_stats(df[label_col].astype(str).tolist()),
    }


def infer_partition_labels(df: pd.DataFrame, label_col: str) -> tuple[list[str], list[str]]:
    if "recommended_partition" in df.columns:
        id_labels = sorted(df.loc[df["recommended_partition"].astype(str) == "id", label_col].astype(str).unique().tolist())
        ood_labels = sorted(df.loc[df["recommended_partition"].astype(str) == "ood", label_col].astype(str).unique().tolist())
        return id_labels, ood_labels
    if "split" in df.columns:
        ood_labels = sorted(df.loc[df["split"].astype(str) == "test_ood", label_col].astype(str).unique().tolist())
        id_labels = sorted(set(df[label_col].astype(str).unique().tolist()) - set(ood_labels))
        return id_labels, ood_labels
    return sorted(df[label_col].astype(str).unique().tolist()), []


def compute_feature_medians(df: pd.DataFrame, feature_cols: Sequence[str]) -> dict[str, float]:
    medians: dict[str, float] = {}
    for col in feature_cols:
        series = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        value = series.median()
        medians[col] = float(value) if pd.notna(value) else 0.0
    return medians


def extract_feature_matrix(df: pd.DataFrame, feature_cols: Sequence[str], medians: dict[str, float]) -> np.ndarray:
    cols = []
    for col in feature_cols:
        series = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(medians[col])
        cols.append(series.to_numpy(dtype=np.float32))
    return np.stack(cols, axis=1).astype(np.float32)


def main() -> None:
    p = argparse.ArgumentParser(description="Augment UAV-NDD Case1 training split with a tabular WGAN-GP.")
    p.add_argument("--input", default=r"data\uav_ndd_case1_experiment.csv")
    p.add_argument("--output", default=r"data\uav_ndd_case1_experiment_gan_augmented.csv")
    p.add_argument("--notes_json", default=r"data\uav_ndd_case1_gan_notes.json")
    p.add_argument("--label_col", default="label")
    p.add_argument("--timestamp_col", default="timestamp")
    p.add_argument("--record_id_col", default="record_id")
    p.add_argument("--train_split_value", default="train")
    p.add_argument("--labels", default="", help="Optional comma-separated labels to augment. Defaults to all eligible minority ID labels.")
    p.add_argument("--target_strategy", choices=["median", "quantile", "fixed", "max"], default="median")
    p.add_argument("--target_count", type=int, default=0)
    p.add_argument("--target_quantile", type=float, default=0.5)
    p.add_argument("--min_real_samples", type=int, default=500)
    p.add_argument("--chunk_size", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--noise_dim", type=int, default=32)
    p.add_argument("--critic_steps", type=int, default=3)
    p.add_argument("--lambda_gp", type=float, default=10.0)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--distance_quantile", type=float, default=0.95)
    p.add_argument("--distance_scale", type=float, default=1.25)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    set_seed(args.seed)
    device = choose_device(args.device)
    input_path = (ROOT / args.input).resolve() if not Path(args.input).is_absolute() else Path(args.input)
    output_path = (ROOT / args.output).resolve() if not Path(args.output).is_absolute() else Path(args.output)
    notes_path = (ROOT / args.notes_json).resolve() if not Path(args.notes_json).is_absolute() else Path(args.notes_json)

    df = load_records(input_path)
    if "split" not in df.columns:
        raise ValueError("Expected a 'split' column so only the training split can be augmented.")

    split_values = df["split"].fillna("").astype(str).str.strip().str.lower()
    train_split = str(args.train_split_value).strip().lower()
    train_df = df.loc[split_values == train_split].copy().reset_index(drop=True)
    val_df = df.loc[split_values == "val"].copy().reset_index(drop=True)
    test_id_df = df.loc[split_values == "test_id"].copy().reset_index(drop=True)
    test_ood_df = df.loc[split_values == "test_ood"].copy().reset_index(drop=True)
    if train_df.empty:
        raise ValueError(f"No rows found for split='{args.train_split_value}'.")

    pre = MetadataPreprocessor(label_col=args.label_col, timestamp_col=args.timestamp_col, record_id_col=args.record_id_col)
    feature_cols = pre.infer_feature_cols(train_df)
    feature_medians = compute_feature_medians(train_df, feature_cols)

    if "row_origin" in df.columns:
        raise ValueError("Input dataset already contains 'row_origin'; aborting to avoid double augmentation confusion.")
    if "synthetic_model" in df.columns:
        raise ValueError("Input dataset already contains 'synthetic_model'; aborting to avoid double augmentation confusion.")

    id_labels, ood_labels = infer_partition_labels(df, args.label_col)
    train_counts = train_df[args.label_col].astype(str).value_counts()
    id_train_counts = train_counts[[lab for lab in train_counts.index.tolist() if lab in set(id_labels)]]
    target_count = resolve_target_count(id_train_counts, args.target_strategy, args.target_count, args.target_quantile)

    requested_labels = set(parse_csv_list(args.labels))
    eligible_labels: list[str] = []
    for label, count in id_train_counts.items():
        if count < args.min_real_samples:
            continue
        if count >= target_count:
            continue
        if requested_labels and label not in requested_labels:
            continue
        eligible_labels.append(str(label))
    eligible_labels = sorted(eligible_labels)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    notes_path.parent.mkdir(parents=True, exist_ok=True)

    train_df = train_df.copy()
    val_df = val_df.copy()
    test_id_df = test_id_df.copy()
    test_ood_df = test_ood_df.copy()
    for frame in (train_df, val_df, test_id_df, test_ood_df):
        frame["row_origin"] = "real"
        frame["synthetic_model"] = ""

    gan_cfg = TabularGanConfig(
        noise_dim=args.noise_dim,
        hidden_dim=args.hidden_dim,
        batch_size=args.batch_size,
        epochs=args.epochs,
        critic_steps=args.critic_steps,
        lambda_gp=args.lambda_gp,
        lr=args.lr,
        device=device,
    )

    synthetic_frames: list[pd.DataFrame] = []
    augmentation_summary: dict[str, dict] = {}
    synthetic_counts: dict[str, int] = {}
    for idx, label in enumerate(eligible_labels):
        class_df = train_df.loc[train_df[args.label_col].astype(str) == label].copy().reset_index(drop=True)
        needed = int(target_count - len(class_df))
        if needed <= 0:
            continue
        real_x = extract_feature_matrix(class_df, feature_cols, feature_medians)
        generated_x, gan_summary = augment_class_with_wgan(
            real_x,
            n_generate=needed,
            config=gan_cfg,
            seed=args.seed + idx * 1000,
            distance_quantile=args.distance_quantile,
            distance_scale=args.distance_scale,
        )
        source_label = class_df["source_label"].mode().iloc[0] if "source_label" in class_df.columns else label
        synthetic = pd.DataFrame(generated_x, columns=feature_cols)
        synthetic[args.label_col] = label
        if "source_label" in class_df.columns:
            synthetic["source_label"] = source_label
        if "recommended_partition" in class_df.columns:
            synthetic["recommended_partition"] = class_df["recommended_partition"].mode().iloc[0]
        synthetic["split"] = args.train_split_value
        synthetic["row_origin"] = "gan_synthetic"
        synthetic["synthetic_model"] = "tabular_wgan_gp"
        synthetic_frames.append(synthetic)
        synthetic_counts[label] = int(len(synthetic))
        augmentation_summary[label] = {
            "real_train_rows": int(len(class_df)),
            "target_rows": int(target_count),
            "requested_synthetic_rows": int(needed),
            "generated_synthetic_rows": int(len(synthetic)),
            **gan_summary,
        }

    if synthetic_frames:
        synthetic_train_df = pd.concat(synthetic_frames, ignore_index=True)
    else:
        synthetic_train_df = pd.DataFrame(columns=train_df.columns.tolist())

    train_aug_df = pd.concat([train_df, synthetic_train_df], ignore_index=True, sort=False)
    ordered_train_cols = list(dict.fromkeys(
        feature_cols
        + [c for c in ["label", "source_label", "recommended_partition", "split", "row_origin", "synthetic_model"] if c in train_aug_df.columns]
    ))
    train_aug_df = train_aug_df.loc[:, ordered_train_cols]
    train_aug_df = interleave_by_label(train_aug_df, args.label_col, args.chunk_size, args.seed + 77)

    ordered_meta_cols = [c for c in ["label", "source_label", "recommended_partition", "split", "row_origin", "synthetic_model"] if c in df.columns or c in {"row_origin", "synthetic_model"}]
    ordered_cols = list(dict.fromkeys(feature_cols + ordered_meta_cols))
    val_df = val_df.loc[:, [c for c in ordered_cols if c in val_df.columns]].copy()
    test_id_df = test_id_df.loc[:, [c for c in ordered_cols if c in test_id_df.columns]].copy()
    test_ood_df = test_ood_df.loc[:, [c for c in ordered_cols if c in test_ood_df.columns]].copy()

    final = pd.concat([train_aug_df, val_df, test_id_df, test_ood_df], ignore_index=True, sort=False)
    final[args.timestamp_col] = np.arange(len(final), dtype=np.float64)
    final[args.record_id_col] = np.arange(len(final), dtype=np.int64)

    final_cols = ordered_cols + [args.timestamp_col, args.record_id_col]
    final = final.loc[:, [c for c in final_cols if c in final.columns]]
    final.to_csv(output_path, index=False)

    before_counts = label_count_dict(train_df, id_labels, label_col=args.label_col)
    after_counts = label_count_dict(train_aug_df, id_labels, label_col=args.label_col)
    split_frames = {
        "train": final[final["split"].astype(str).str.lower() == train_split].copy(),
        "val": final[final["split"].astype(str).str.lower() == "val"].copy(),
        "test_id": final[final["split"].astype(str).str.lower() == "test_id"].copy(),
        "test_ood": final[final["split"].astype(str).str.lower() == "test_ood"].copy(),
    }
    ordered_all_labels = list(dict.fromkeys(id_labels + ood_labels))

    summary = {
        "input_csv": str(input_path),
        "output_csv": str(output_path),
        "notes_json": str(notes_path),
        "device": device,
        "feature_columns": feature_cols,
        "train_rows_before": int(len(train_df)),
        "train_rows_after": int(len(train_aug_df)),
        "synthetic_rows_added": int(len(synthetic_train_df)),
        "train_label_counts_before": before_counts,
        "train_label_counts_after": after_counts,
        "synthetic_rows_per_label": synthetic_counts,
        "eligible_labels": eligible_labels,
        "requested_labels": sorted(requested_labels),
        "id_labels": id_labels,
        "ood_labels": ood_labels,
        "target_strategy": args.target_strategy,
        "target_count": int(target_count),
        "min_real_samples": int(args.min_real_samples),
        "chunk_size": int(args.chunk_size),
        "gan_config": {
            "noise_dim": int(args.noise_dim),
            "hidden_dim": int(args.hidden_dim),
            "batch_size": int(args.batch_size),
            "epochs": int(args.epochs),
            "critic_steps": int(args.critic_steps),
            "lambda_gp": float(args.lambda_gp),
            "lr": float(args.lr),
            "distance_quantile": float(args.distance_quantile),
            "distance_scale": float(args.distance_scale),
            "seed": int(args.seed),
        },
        "per_label_augmentation": augmentation_summary,
        "split_summary": {
            "train": summarise_split(split_frames["train"], args.label_col, ordered_all_labels),
            "val": summarise_split(split_frames["val"], args.label_col, ordered_all_labels),
            "test_id": summarise_split(split_frames["test_id"], args.label_col, ordered_all_labels),
            "test_ood": summarise_split(split_frames["test_ood"], args.label_col, ordered_all_labels),
        },
        "notes": [
            "Only the training split was augmented; validation and test splits were left unchanged.",
            "Synthetic rows were generated label-by-label with a tabular WGAN-GP trained on each minority class.",
            "Generated rows were post-processed by clipping/snapping to observed feature support and filtered by nearest-neighbor distance to the real class manifold.",
            "The final dataset was reordered as train -> val -> test_id -> test_ood and assigned fresh monotonic timestamp/record_id values.",
        ],
        "recommended_train_command": (
            "python scripts/train.py "
            f"--input {output_path} "
            "--output_dir runs/uav_ndd_case1_gan_augmented "
            "--label_col label "
            "--timestamp_col timestamp "
            "--record_id_col record_id "
            f"--id_classes {','.join(id_labels)} "
            f"--ood_classes {','.join(ood_labels)} "
            "--window_mode count --window_size 16 --stride 8 --epochs 10"
        ),
    }
    notes_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "gan_augmentation_summary": {
            "train_rows_before": int(len(train_df)),
            "train_rows_after": int(len(train_aug_df)),
            "synthetic_rows_added": int(len(synthetic_train_df)),
            "target_count": int(target_count),
            "eligible_labels": eligible_labels,
            "output_csv": str(output_path),
            "notes_json": str(notes_path),
        }
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
