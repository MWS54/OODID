from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd


def parse_csv_list(value: str | Sequence[str]) -> List[str]:
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return [str(item).strip() for item in value if str(item).strip()]


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


def fingerprint_frame(df: pd.DataFrame, feature_cols: Sequence[str]) -> pd.Series:
    return pd.util.hash_pandas_object(df.loc[:, feature_cols], index=False)


def drop_ambiguous_feature_patterns(df: pd.DataFrame, feature_cols: Sequence[str]) -> tuple[pd.DataFrame, int]:
    fp = fingerprint_frame(df, feature_cols)
    label_counts = df.groupby(fp)["label"].nunique()
    ambiguous = set(label_counts[label_counts > 1].index.tolist())
    if not ambiguous:
        return df.copy(), 0
    keep = ~fp.isin(ambiguous)
    dropped = int((~keep).sum())
    return df.loc[keep].copy(), dropped


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


def exact_id_split(
    df: pd.DataFrame,
    label_col: str,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    label_counts = df[label_col].astype(str).value_counts()
    if int(label_counts.min()) < 3:
        raise ValueError("At least three samples per ID class are required for stratified train/val/test splitting.")
    rng = np.random.default_rng(seed)
    train_parts: list[pd.DataFrame] = []
    val_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []

    for _, group in df.groupby(label_col, sort=True):
        shuffled = group.sample(frac=1.0, random_state=int(rng.integers(0, np.iinfo(np.int32).max))).reset_index(drop=False)
        n = len(shuffled)
        n_train = max(1, int(round(n * train_ratio)))
        n_val = max(1, int(round(n * val_ratio)))
        n_test = n - n_train - n_val
        while n_test < 1:
            if n_train >= n_val and n_train > 1:
                n_train -= 1
            elif n_val > 1:
                n_val -= 1
            else:
                break
            n_test = n - n_train - n_val
        if min(n_train, n_val, n_test) < 1:
            raise ValueError("ID data is too small for the requested split ratios.")
        train_parts.append(group.loc[shuffled.iloc[:n_train]["index"]].copy())
        val_parts.append(group.loc[shuffled.iloc[n_train:n_train + n_val]["index"]].copy())
        test_parts.append(group.loc[shuffled.iloc[n_train + n_val:]["index"]].copy())

    train_df = pd.concat(train_parts, ignore_index=False)
    val_df = pd.concat(val_parts, ignore_index=False)
    test_df = pd.concat(test_parts, ignore_index=False)
    return train_df.copy(), val_df.copy(), test_df.copy()


def summarise_split(df: pd.DataFrame, classes: Sequence[str], label_col: str = "label") -> dict:
    return {
        "rows": int(len(df)),
        "label_counts": label_count_dict(df, classes, label_col=label_col),
        "mixed_window_stats": window_mix_stats(df[label_col].tolist()),
    }


def resolve_existing_path(base_candidates: Sequence[Path], value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    for base in base_candidates:
        candidate = (base / path).resolve()
        if candidate.exists():
            return candidate
    return (base_candidates[0] / path).resolve()


def resolve_output_path(project_root: Path, value: str | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return (project_root / path).resolve()


def allocate_counts(raw_counts: Dict[str, int], target_rows: int, minimum_per_label: int = 0) -> dict[str, int]:
    positive = {str(k): int(v) for k, v in raw_counts.items() if int(v) > 0}
    if not positive:
        return {}
    total_rows = sum(positive.values())
    capped_target = min(int(target_rows), total_rows)
    if capped_target <= 0:
        return {label: 0 for label in sorted(positive)}
    if capped_target >= total_rows:
        return {label: int(positive[label]) for label in sorted(positive)}

    base = {label: 0 for label in positive}
    if minimum_per_label > 0:
        desired_base = {label: min(int(count), int(minimum_per_label)) for label, count in positive.items()}
        if sum(desired_base.values()) <= capped_target:
            base = desired_base

    remaining_target = capped_target - sum(base.values())
    headroom = {label: positive[label] - base[label] for label in positive}
    headroom_total = sum(headroom.values())
    if remaining_target <= 0 or headroom_total <= 0:
        return {label: int(base[label]) for label in sorted(base)}

    floors: dict[str, int] = {}
    fractions: list[tuple[float, str]] = []
    for label, count in headroom.items():
        exact = remaining_target * count / headroom_total
        floor_val = int(math.floor(exact))
        floors[label] = floor_val
        fractions.append((exact - floor_val, label))
    allocated = sum(floors.values())
    remainder = remaining_target - allocated
    for _, label in sorted(fractions, key=lambda item: (-item[0], item[1]))[:remainder]:
        floors[label] += 1
    return {label: int(base[label] + floors[label]) for label in sorted(positive)}


def rebalance_shortfall(available_counts: Dict[str, int], desired_counts: Dict[str, int]) -> dict[str, int]:
    labels = sorted(set(available_counts) | set(desired_counts))
    selected = {
        label: min(int(available_counts.get(label, 0)), int(desired_counts.get(label, 0)))
        for label in labels
    }
    target_rows = sum(int(v) for v in desired_counts.values())
    remaining = target_rows - sum(selected.values())
    if remaining <= 0:
        return selected
    surplus = {
        label: max(int(available_counts.get(label, 0)) - int(selected.get(label, 0)), 0)
        for label in labels
    }
    if sum(surplus.values()) <= 0:
        return selected
    extras = allocate_counts(surplus, remaining, minimum_per_label=0)
    for label, extra in extras.items():
        selected[label] = int(selected.get(label, 0) + extra)
    return selected


def downsample_by_label(
    df: pd.DataFrame,
    label_col: str,
    target_rows: int | None,
    seed: int,
    minimum_per_label: int = 0,
) -> tuple[pd.DataFrame, dict[str, int], dict[str, int]]:
    available_counts = label_count_dict(df, label_col=label_col)
    if target_rows is None or int(target_rows) <= 0 or int(target_rows) >= len(df):
        return df.copy(), available_counts, available_counts

    desired_counts = allocate_counts(available_counts, int(target_rows), minimum_per_label=minimum_per_label)
    selected_counts = rebalance_shortfall(available_counts, desired_counts)
    rng = np.random.default_rng(seed)
    frames: list[pd.DataFrame] = []
    for label in sorted(selected_counts):
        group = df[df[label_col].astype(str) == str(label)].copy()
        target = int(selected_counts[label])
        if target <= 0 or group.empty:
            continue
        if target < len(group):
            group = group.sample(n=target, random_state=int(rng.integers(0, np.iinfo(np.int32).max)), replace=False)
        frames.append(group.reset_index(drop=True))
    if not frames:
        return df.iloc[0:0].copy(), available_counts, selected_counts
    return pd.concat(frames, ignore_index=True), available_counts, selected_counts
