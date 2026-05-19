from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd

from .baselines import (
    OptionalDependencyUnavailable,
    flatten_window_metadata,
    logits_to_probabilities,
    make_tabular_baseline,
    probabilities_to_logits,
)
from .io import load_records, save_records
from .metrics import compute_class_support, compute_present_class_macro_f1, multilabel_metrics, ood_metrics
from .ood import (
    OODCalibrator,
    PrototypeBank,
    build_leave_one_class_out_pseudo_ood,
    calibrate_class_thresholds,
    calibrate_temperature,
    compute_raw_ood_scores,
)
from .preprocessing import GroupAwareMetadataPreprocessor, MetadataPreprocessor
from .utils import parse_csv_list
from .windowing import (
    WindowedData,
    attach_parsed_labels,
    build_grouped_windows,
    filter_id_windows,
    infer_all_labels,
    mark_ood_records,
    window_phase_labels,
)

ALLOWED_SPLIT_NAMES = ("train", "val", "test_id", "test_ood")
TABULAR_BASELINE_METHODS = ("svm", "random_forest", "xgboost", "mlp_tabular")


@dataclass(frozen=True)
class ComparisonMethodConfig:
    method_name: str
    display_name: str
    category: str
    encoder_ablation: Optional[str] = None
    fusion: Optional[str] = None
    ood_threshold_mode: Optional[str] = None
    group_threshold_strategy: Optional[str] = None
    tabular_factory: Optional[str] = None


COMPARISON_METHOD_CONFIGS = [
    ComparisonMethodConfig("svm", "SVM", "sklearn_tabular", tabular_factory="svm"),
    ComparisonMethodConfig("random_forest", "Random Forest", "sklearn_tabular", tabular_factory="random_forest"),
    ComparisonMethodConfig("xgboost", "XGBoost", "sklearn_tabular", tabular_factory="xgboost"),
    ComparisonMethodConfig("mlp_tabular", "MLP (tabular)", "sklearn_tabular", tabular_factory="mlp_tabular"),
    ComparisonMethodConfig("mlp_only", "MLP-Only", "neural_encoder", encoder_ablation="mlp_only"),
    ComparisonMethodConfig("gcn_only", "GCN-Only", "neural_encoder", encoder_ablation="gcn_only"),
    ComparisonMethodConfig("random_graph", "Random-Graph", "neural_encoder", encoder_ablation="random_graph"),
    ComparisonMethodConfig("full", "Full (Transformer+GCN)", "neural_encoder", encoder_ablation="full"),
    ComparisonMethodConfig(
        "transformer_only_mean_fusion",
        "Transformer-Only + Mean Fusion",
        "neural_encoder",
        encoder_ablation="transformer_only",
        fusion="mean",
    ),
    ComparisonMethodConfig(
        "ucs_oodid",
        "UCS-OODID",
        "neural_encoder",
        encoder_ablation="transformer_only",
        fusion="correlation_aware",
        ood_threshold_mode="group",
        group_threshold_strategy="conservative",
    ),
]


@dataclass
class ComparisonDatasetBundle:
    prepared_input_path: Path
    split_source: str
    split_frames: Dict[str, pd.DataFrame]
    split_windows: Dict[str, WindowedData]
    id_classes: list[str]
    ood_classes: list[str]
    class_to_idx: Dict[str, int]
    preprocessor: MetadataPreprocessor
    normalization_summary: dict
    leakage_report: dict
    split_summary: dict
    label_col: str
    timestamp_col: str
    record_id_col: str
    group_col: Optional[str]
    window_config: dict


def get_method_config(method_name: str) -> ComparisonMethodConfig:
    key = str(method_name).strip().lower()
    for config in COMPARISON_METHOD_CONFIGS:
        if config.method_name == key:
            return config
    raise ValueError(f"Unsupported comparison method: {method_name}")


def select_method_configs(methods: str | None) -> list[ComparisonMethodConfig]:
    if methods is None or str(methods).strip() == "":
        return list(COMPARISON_METHOD_CONFIGS)
    names = [item.strip().lower() for item in str(methods).split(",") if item.strip()]
    if not names or names == ["all"]:
        return list(COMPARISON_METHOD_CONFIGS)
    seen = set()
    selected = []
    for name in names:
        config = get_method_config(name)
        if config.method_name in seen:
            continue
        seen.add(config.method_name)
        selected.append(config)
    return selected


def sort_records_for_windowing(df: pd.DataFrame, group_col: Optional[str], timestamp_col: str) -> pd.DataFrame:
    if group_col and group_col in df.columns:
        if timestamp_col in df.columns:
            return df.sort_values([group_col, timestamp_col], kind="stable").reset_index(drop=True)
        return df.sort_values(group_col, kind="stable").reset_index(drop=True)
    if timestamp_col in df.columns:
        return df.sort_values(timestamp_col, kind="stable").reset_index(drop=True)
    return df.reset_index(drop=True)


def build_windows_by_mode(features: np.ndarray, df: pd.DataFrame, class_to_idx: Dict[str, int], args) -> WindowedData:
    return build_grouped_windows(
        features,
        df,
        class_to_idx,
        group_col=args.group_col,
        mode=args.window_mode,
        timestamp_col=args.timestamp_col,
        label_col=args.label_col,
        record_id_col=args.record_id_col,
        window_size=args.window_size,
        stride=args.stride,
        time_seconds=args.time_window_seconds,
        adaptive_min_size=args.adaptive_min_size,
        adaptive_max_size=args.adaptive_max_size,
    )


def chronological_split_id_records(df: pd.DataFrame, ratios=(0.70, 0.15, 0.15)):
    id_df = df[~df["__is_ood_record"]].copy()
    n = len(id_df)
    n_train = max(1, int(n * ratios[0]))
    n_val = max(1, int(n * ratios[1]))
    train = id_df.iloc[:n_train].copy()
    val = id_df.iloc[n_train:n_train + n_val].copy()
    test = id_df.iloc[n_train + n_val:].copy()
    if len(test) == 0 and len(val) > 1:
        test = val.iloc[len(val) // 2 :].copy()
        val = val.iloc[: len(val) // 2].copy()
    return train, val, test, df[df["__is_ood_record"]].copy()


def _normalize_split_column(df: pd.DataFrame) -> pd.Series:
    return df["split"].fillna("").astype(str).str.strip().str.lower()


def _validate_explicit_split_frames(split_frames: Dict[str, pd.DataFrame]) -> None:
    invalid_ood_targets = {}
    for split_name in ("train", "val", "test_id"):
        bad = int(split_frames[split_name]["__is_ood_record"].sum()) if len(split_frames[split_name]) else 0
        if bad:
            invalid_ood_targets[split_name] = bad
    if invalid_ood_targets:
        raise ValueError(
            "Explicit split column places OOD records in ID-only splits: "
            + ", ".join(f"{name}={count}" for name, count in sorted(invalid_ood_targets.items()))
        )
    test_ood_df = split_frames["test_ood"]
    if len(test_ood_df):
        non_ood = int((~test_ood_df["__is_ood_record"]).sum())
        if non_ood:
            raise ValueError(f"Explicit split column places {non_ood} ID records in test_ood.")


def resolve_dataset_splits_for_comparison(df: pd.DataFrame):
    if "split" in df.columns:
        split_values = _normalize_split_column(df)
        blank_count = int((split_values == "").sum())
        if blank_count:
            raise ValueError(f"Explicit split column contains {blank_count} blank/unassigned records.")
        invalid = sorted({value for value in split_values.unique().tolist() if value and value not in ALLOWED_SPLIT_NAMES})
        if invalid:
            raise ValueError(
                "Unsupported values found in split column: "
                + ", ".join(invalid)
                + f". Expected only {', '.join(ALLOWED_SPLIT_NAMES)}."
            )
        split_frames = {name: df[split_values == name].copy() for name in ALLOWED_SPLIT_NAMES}
        _validate_explicit_split_frames(split_frames)
        return split_frames, "split_column"
    train_df, val_df, test_df, ood_df = chronological_split_id_records(df)
    return {
        "train": train_df,
        "val": val_df,
        "test_id": test_df,
        "test_ood": ood_df,
    }, "chronological_fallback"


def _counter_to_dict(counter):
    return {str(key): int(counter[key]) for key in sorted(counter)}


def _group_indices(group_ids):
    if group_ids is None:
        return {}
    arr = np.asarray(group_ids)
    if arr.size == 0:
        return {}
    groups = {}
    for idx, value in enumerate(arr.tolist()):
        if value is None or (isinstance(value, (float, np.floating)) and np.isnan(value)):
            key = "nan"
        else:
            key = str(value)
        groups.setdefault(key, []).append(idx)
    return {group: np.asarray(indices, dtype=np.int64) for group, indices in groups.items()}


def _group_count_dict(group_ids):
    counts = Counter()
    for group, indices in _group_indices(group_ids).items():
        counts[group] = len(indices)
    return _counter_to_dict(counts)


def summarize_record_labels(df: pd.DataFrame, id_classes):
    label_counts = Counter()
    ood_counts = Counter()
    id_set = {str(label) for label in id_classes}
    for labels in df.get("__labels", []):
        for label in labels:
            label = str(label)
            label_counts[label] += 1
            if label not in id_set:
                ood_counts[label] += 1
    return _counter_to_dict(label_counts), _counter_to_dict(ood_counts)


def build_split_summary(split_source, split_frames, split_windows, id_classes, group_col=None):
    summary = {"split_source": split_source, "splits": {}}
    for split_name, split_df in split_frames.items():
        label_counts, ood_counts = summarize_record_labels(split_df, id_classes)
        windows = split_windows.get(split_name)
        split_summary = {
            "records": int(len(split_df)),
            "windows": int(len(windows)) if windows is not None else 0,
            "label_distribution": label_counts,
            "ood_class_distribution": ood_counts,
        }
        if group_col and group_col in split_df.columns:
            split_summary["group_distribution"] = _group_count_dict(split_df[group_col].to_numpy())
        if windows is not None and getattr(windows, "group_ids", None) is not None:
            split_summary["window_group_distribution"] = _group_count_dict(windows.group_ids)
        summary["splits"][split_name] = split_summary
    return summary


def compute_id_metrics_by_group(windows, probs, thresholds, label_names=None, present_class_min_support=1):
    groups = _group_indices(getattr(windows, "group_ids", None))
    if not groups:
        return {}
    thresholds_array = np.asarray(thresholds, dtype=np.float32)
    preds = (np.asarray(probs) >= thresholds_array.reshape(1, -1)).astype(int)
    summary = {}
    for group, indices in groups.items():
        group_y = np.asarray(windows.y[indices])
        group_probs = np.asarray(probs[indices])
        metrics = multilabel_metrics(group_y, group_probs, thresholds_array)
        summary[group] = {
            "windows": int(len(indices)),
            **metrics,
            **compute_present_class_macro_f1(
                group_y,
                preds[indices],
                label_names=label_names,
                min_support=present_class_min_support,
            ),
            "class_support": compute_class_support(group_y, label_names=label_names),
        }
    return summary


def compute_ood_metrics_by_group(group_ids, y_true_ood, scores, decisions):
    groups = _group_indices(group_ids)
    if not groups:
        return {}
    y_true_ood = np.asarray(y_true_ood)
    scores = np.asarray(scores)
    decisions_array = None if decisions is None else np.asarray(decisions)
    summary = {}
    for group, indices in groups.items():
        group_y = y_true_ood[indices].astype(int)
        group_scores = scores[indices]
        group_decisions = None if decisions_array is None else decisions_array[indices]
        metrics = ood_metrics(group_y, group_scores, group_decisions)
        summary[group] = {
            "windows": int(len(indices)),
            "known_windows": int(np.sum(group_y == 0)),
            "ood_windows": int(np.sum(group_y == 1)),
            "alert_rate": float(np.mean(group_decisions.astype(float))) if group_decisions is not None and len(group_decisions) else float("nan"),
            **metrics,
        }
    return summary


def default_direction_item(score_name):
    return {
        "score_name": score_name,
        "raw_auroc": None,
        "flipped": False,
        "effective_auroc": None,
        "direction": 1.0,
    }


def ordered_direction_report(score_names, report):
    report_by_name = {item["score_name"]: item for item in report}
    return [report_by_name.get(name, default_direction_item(name)) for name in score_names]


def ordered_ood_weights(ood_cal):
    return {name: float(ood_cal.weights.get(name, 0.0)) for name in ood_cal.score_names}


def ordered_ood_direction_report(ood_cal):
    return ordered_direction_report(ood_cal.score_names, ood_cal.direction_report)


def ordered_ood_directions(ood_cal):
    return {name: float(ood_cal.directions.get(name, 1.0)) for name in ood_cal.score_names}


def build_direction_diagnostic_payload(score_names, report=None, label_source="none", enabled=False, note=None):
    payload = {
        "enabled": bool(enabled),
        "label_source": label_source,
        "used_for_direction_flip": False,
        "report": ordered_direction_report(score_names, report or []),
    }
    if note:
        payload["note"] = note
    return payload


def phase_threshold_summary(ood_cal):
    return {
        "enabled": bool(ood_cal.phase_aware_enabled),
        "phase_column": ood_cal.phase_column,
        "phase_threshold_min_samples": int(ood_cal.phase_threshold_min_samples),
        "phase_threshold_quantile": ood_cal.phase_threshold_quantile,
        "phase_threshold_fallback": ood_cal.phase_threshold_fallback,
        "global_threshold": float(ood_cal.threshold),
        "phase_thresholds": {phase: float(value) for phase, value in ood_cal.phase_thresholds.items()},
        "phase_validation_counts": {phase: int(value) for phase, value in ood_cal.phase_validation_counts.items()},
        "phase_threshold_sources": dict(ood_cal.phase_threshold_sources),
    }


def ood_threshold_summary(ood_cal):
    return {
        "ood_threshold_mode": ood_cal.ood_threshold_mode,
        "global_ood_threshold": float(ood_cal.threshold),
        "group_threshold_min_samples": int(ood_cal.group_threshold_min_samples),
        "group_threshold_quantile": ood_cal.group_threshold_quantile,
        "group_threshold_strategy": ood_cal.group_threshold_strategy,
        "group_threshold_shrink_k": float(ood_cal.group_threshold_shrink_k),
        "group_threshold_min_ratio": float(ood_cal.group_threshold_min_ratio),
        "group_ood_thresholds": {group: float(value) for group, value in ood_cal.group_thresholds.items()},
        "group_raw_thresholds": {group: float(value) for group, value in ood_cal.group_raw_thresholds.items()},
        "group_smoothed_thresholds": {group: float(value) for group, value in ood_cal.group_smoothed_thresholds.items()},
        "group_threshold_sources": dict(ood_cal.group_threshold_sources),
        "group_validation_counts": {group: int(value) for group, value in ood_cal.group_validation_counts.items()},
        "group_ood_threshold_fallbacks": dict(ood_cal.group_threshold_fallbacks),
    }


def _empty_features(preprocessor: MetadataPreprocessor, length: int) -> np.ndarray:
    return np.zeros((length, len(preprocessor.feature_cols)), dtype=np.float32)


def _transform_or_empty(preprocessor: MetadataPreprocessor, df: pd.DataFrame) -> np.ndarray:
    if len(df) == 0:
        return _empty_features(preprocessor, 0)
    return preprocessor.transform(df)


def prepare_comparison_dataset(args, output_dir: Path) -> ComparisonDatasetBundle:
    output_dir.mkdir(parents=True, exist_ok=True)
    args.group_col = (args.group_col or "").strip()
    if args.normalization_mode == "group" and not args.group_col:
        raise ValueError("group normalization requires --group_col")
    if args.ood_threshold_mode == "group" and not args.group_col:
        raise ValueError("group-aware OOD thresholds require --group_col")

    df = load_records(args.input)
    if args.record_id_col not in df.columns:
        df[args.record_id_col] = np.arange(len(df))
    if args.group_col and args.group_col not in df.columns:
        raise ValueError(f"group_col {args.group_col!r} not found in input data.")

    df = sort_records_for_windowing(df, group_col=args.group_col or None, timestamp_col=args.timestamp_col)
    df = attach_parsed_labels(df, args.label_col)
    all_labels = infer_all_labels(df, args.label_col)
    ood_classes = parse_csv_list(args.ood_classes)
    id_classes = parse_csv_list(args.id_classes)
    if not id_classes:
        id_classes = [label for label in all_labels if label not in set(ood_classes)]
    if not id_classes:
        raise ValueError("No ID classes found. Provide --id_classes or check label column.")
    class_to_idx = {name: idx for idx, name in enumerate(id_classes)}
    df = mark_ood_records(df, id_classes)

    split_frames, split_source = resolve_dataset_splits_for_comparison(df)

    prepared_df = df.copy()
    prepared_df["split"] = ""
    for split_name, split_df in split_frames.items():
        prepared_df.loc[split_df.index, "split"] = split_name
    prepared_input_path = output_dir / "comparison_input_with_split.jsonl"
    save_records(prepared_df.drop(columns=["__labels", "__is_ood_record"], errors="ignore"), prepared_input_path)

    pre_cls = GroupAwareMetadataPreprocessor if args.normalization_mode == "group" else MetadataPreprocessor
    preprocessor = pre_cls(
        label_col=args.label_col,
        timestamp_col=args.timestamp_col,
        record_id_col=args.record_id_col,
        group_col=args.group_col or None,
        allow_ports=args.allow_ports,
    )
    preprocessor.fit(split_frames["train"])

    split_features = {
        name: _transform_or_empty(preprocessor, split_df)
        for name, split_df in split_frames.items()
    }
    split_windows = {
        "train": filter_id_windows(build_windows_by_mode(split_features["train"], split_frames["train"], class_to_idx, args)),
        "val": filter_id_windows(build_windows_by_mode(split_features["val"], split_frames["val"], class_to_idx, args)),
        "test_id": filter_id_windows(build_windows_by_mode(split_features["test_id"], split_frames["test_id"], class_to_idx, args)),
        "test_ood": build_windows_by_mode(split_features["test_ood"], split_frames["test_ood"], class_to_idx, args),
    }
    if len(split_windows["train"]) == 0:
        raise ValueError("Not enough ID windows for train split. Reduce window size or check the split.")
    if len(split_windows["val"]) == 0:
        raise ValueError("Not enough ID windows for val split. Reduce window size or check the split.")
    if len(split_windows["test_id"]) == 0:
        raise ValueError("Not enough ID windows for test_id split. Reduce window size or check the split.")
    if len(split_windows["test_ood"]) == 0:
        raise ValueError("Not enough OOD windows for test_ood split. Comparison experiments require real OOD evaluation windows.")

    split_summary = build_split_summary(
        split_source,
        split_frames,
        split_windows,
        id_classes,
        group_col=args.group_col or None,
    )
    return ComparisonDatasetBundle(
        prepared_input_path=prepared_input_path,
        split_source=split_source,
        split_frames=split_frames,
        split_windows=split_windows,
        id_classes=id_classes,
        ood_classes=ood_classes,
        class_to_idx=class_to_idx,
        preprocessor=preprocessor,
        normalization_summary=preprocessor.normalization_summary(),
        leakage_report=preprocessor.leakage_report(df),
        split_summary=split_summary,
        label_col=args.label_col,
        timestamp_col=args.timestamp_col,
        record_id_col=args.record_id_col,
        group_col=args.group_col or None,
        window_config={
            "mode": args.window_mode,
            "size": int(args.window_size),
            "stride": int(args.stride),
            "time_seconds": float(args.time_window_seconds),
            "adaptive_min_size": int(args.adaptive_min_size),
            "adaptive_max_size": int(args.adaptive_max_size),
        },
    )


def _concatenate_optional_arrays(left, right):
    if left is None or right is None:
        return None
    return np.concatenate([left, right], axis=0)


def _tabular_ood_embedding(probs: np.ndarray) -> np.ndarray:
    return np.asarray(probs, dtype=np.float32)


def build_status_report(
    method: ComparisonMethodConfig,
    status: str,
    *,
    error_message: Optional[str] = None,
    skip_reason: Optional[str] = None,
    prepared_input_path: Optional[Path] = None,
) -> dict:
    report = {
        "status": status,
        "comparison_method": method.method_name,
        "display_name": method.display_name,
        "method_category": method.category,
    }
    if prepared_input_path is not None:
        report["comparison_input"] = str(prepared_input_path)
    if error_message:
        report["error_message"] = error_message
    if skip_reason:
        report["skip_reason"] = skip_reason
    return report


def evaluate_tabular_baseline(bundle: ComparisonDatasetBundle, method: ComparisonMethodConfig, args) -> dict:
    if method.tabular_factory is None:
        raise ValueError(f"Method {method.method_name} is not a tabular baseline.")

    baseline = make_tabular_baseline(method.tabular_factory, random_state=args.seed)
    train_w = bundle.split_windows["train"]
    val_w = bundle.split_windows["val"]
    test_id_w = bundle.split_windows["test_id"]
    test_ood_w = bundle.split_windows["test_ood"]

    train_flat = flatten_window_metadata(train_w, include_mask=True)
    val_flat = flatten_window_metadata(val_w, include_mask=True)
    test_id_flat = flatten_window_metadata(test_id_w, include_mask=True)
    test_ood_flat = flatten_window_metadata(test_ood_w, include_mask=True)

    baseline.fit(train_flat, train_w.y)

    val_logits = probabilities_to_logits(baseline.predict_proba(val_flat))
    temperature = calibrate_temperature(val_logits, val_w.y, grid=[0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 5.0])
    val_probs = logits_to_probabilities(val_logits, temperature=temperature)
    class_thresholds = calibrate_class_thresholds(val_probs, val_w.y)

    bank = PrototypeBank.fit(_tabular_ood_embedding(val_probs), val_w.y, bundle.id_classes)
    raw_val = compute_raw_ood_scores(
        val_logits,
        val_probs,
        _tabular_ood_embedding(val_probs),
        bank,
        temperature=temperature,
        k_bank=args.bank_k,
    )
    ood_cal = OODCalibrator(
        fusion=method.fusion or args.fusion,
        q_ood=args.q_ood,
        ood_threshold_mode=method.ood_threshold_mode or args.ood_threshold_mode,
        group_threshold_strategy=method.group_threshold_strategy or args.group_threshold_strategy,
        group_threshold_shrink_k=args.group_threshold_shrink_k,
        group_threshold_min_ratio=args.group_threshold_min_ratio,
    )
    ood_cal.group_threshold_min_samples = int(max(args.group_threshold_min_samples, 0))
    ood_cal.group_threshold_quantile = float(args.q_ood)
    if args.ood_direction_calibration == "pseudo_ood":
        pseudo_raw, pseudo_labels, pseudo_summary = build_leave_one_class_out_pseudo_ood(
            raw_val,
            val_w.y,
            class_names=bundle.id_classes,
            score_names=ood_cal.score_names,
        )
        if len(pseudo_labels) and len(np.unique(pseudo_labels)) == 2:
            ood_cal.calibrate_directions(
                pseudo_raw,
                y_true_ood=pseudo_labels,
                label_source="pseudo_ood_leave_one_class_out_validation_windows",
            )
        else:
            ood_cal.set_default_directions(label_source="pseudo_ood_unavailable_fallback_to_none")
    else:
        pseudo_summary = []
        ood_cal.set_default_directions(label_source="none")
    ood_cal.fit(raw_val)
    val_threshold_scores = ood_cal.transform(raw_val)["fused"]
    if ood_cal.ood_threshold_mode == "group":
        ood_cal.calibrate_group_thresholds(
            val_threshold_scores,
            getattr(val_w, "group_ids", None),
            min_samples=args.group_threshold_min_samples,
            quantile=args.q_ood,
            strategy=method.group_threshold_strategy or args.group_threshold_strategy,
            shrink_k=args.group_threshold_shrink_k,
            min_ratio=args.group_threshold_min_ratio,
        )

    val_phase_labels = None
    if args.phase_aware_threshold and args.phase_column in bundle.split_frames["val"].columns:
        val_phase_labels = window_phase_labels(bundle.split_frames["val"], val_w, args.phase_column)
        transformed_val = ood_cal.transform(raw_val)
        ood_cal.calibrate_phase_thresholds(
            transformed_val["fused"],
            val_phase_labels,
            phase_column=args.phase_column,
            min_samples=args.phase_threshold_min_samples,
            quantile=args.phase_threshold_quantile,
            fallback=args.phase_threshold_fallback,
        )

    test_id_logits = probabilities_to_logits(baseline.predict_proba(test_id_flat))
    test_id_probs = logits_to_probabilities(test_id_logits, temperature=temperature)
    test_ood_logits = probabilities_to_logits(baseline.predict_proba(test_ood_flat))
    test_ood_probs = logits_to_probabilities(test_ood_logits, temperature=temperature)

    id_test = multilabel_metrics(test_id_w.y, test_id_probs, class_thresholds)
    id_test_by_group = compute_id_metrics_by_group(
        test_id_w,
        test_id_probs,
        class_thresholds,
        label_names=bundle.id_classes,
        present_class_min_support=args.present_class_min_support,
    )

    combo_logits = np.concatenate([test_id_logits, test_ood_logits], axis=0)
    combo_probs = np.concatenate([test_id_probs, test_ood_probs], axis=0)
    combo_embeddings = _tabular_ood_embedding(combo_probs)
    combo_ood = np.concatenate(
        [np.zeros(len(test_id_w), dtype=bool), np.ones(len(test_ood_w), dtype=bool)],
        axis=0,
    )
    combo_group_ids = _concatenate_optional_arrays(test_id_w.group_ids, test_ood_w.group_ids)
    combo_phase_labels = None
    if args.phase_aware_threshold and args.phase_column in bundle.split_frames["test_id"].columns and args.phase_column in bundle.split_frames["test_ood"].columns:
        combo_phase_labels = np.concatenate(
            [
                window_phase_labels(bundle.split_frames["test_id"], test_id_w, args.phase_column),
                window_phase_labels(bundle.split_frames["test_ood"], test_ood_w, args.phase_column),
            ],
            axis=0,
        )

    raw_combo = compute_raw_ood_scores(
        combo_logits,
        combo_probs,
        combo_embeddings,
        bank,
        temperature=temperature,
        k_bank=args.bank_k,
    )
    test_ood_direction_diagnostic_report = build_direction_diagnostic_payload(
        ood_cal.score_names,
        label_source="none" if args.ood_direction_calibration != "test_ood_diagnostic" else "unavailable",
        enabled=False,
        note=(
            "test_ood_diagnostic was not requested"
            if args.ood_direction_calibration != "test_ood_diagnostic"
            else "Diagnostic report will be generated without affecting final score directions."
        ),
    )
    if args.ood_direction_calibration == "test_ood_diagnostic":
        diagnostic_cal = OODCalibrator(score_names=list(ood_cal.score_names))
        diagnostic_cal.calibrate_directions(
            raw_combo,
            y_true_ood=combo_ood.astype(int),
            label_source="test_id_vs_test_ood_windows_diagnostic",
        )
        test_ood_direction_diagnostic_report = build_direction_diagnostic_payload(
            diagnostic_cal.score_names,
            report=diagnostic_cal.direction_report,
            label_source=diagnostic_cal.direction_label_source,
            enabled=True,
            note="Diagnostic only; final OOD score directions remain unchanged.",
        )

    transformed = ood_cal.transform(raw_combo, phases=combo_phase_labels, groups=combo_group_ids)
    ood_test = ood_metrics(combo_ood.astype(int), transformed["fused"], transformed["decisions"])
    ood_test_by_group = compute_ood_metrics_by_group(
        combo_group_ids,
        combo_ood.astype(int),
        transformed["fused"],
        transformed["decisions"],
    )

    report = {
        "status": "success",
        "comparison_method": method.method_name,
        "display_name": method.display_name,
        "method_category": method.category,
        "comparison_input": str(bundle.prepared_input_path),
        "baseline_name": baseline.name,
        "tabular_representation": {
            "flattened_dim": int(train_flat.shape[1]),
            "include_valid_mask": True,
            "invalid_steps_zeroed": True,
            "ood_embedding_source": "calibrated_probabilities",
        },
        "split_source": bundle.split_source,
        "dataset_split_summary": bundle.split_summary,
        "window_config": dict(bundle.window_config),
        "group_config": {
            "group_col": bundle.group_col,
            "present_class_min_support": int(args.present_class_min_support),
        },
        "normalization": bundle.normalization_summary,
        "leakage_report": bundle.leakage_report,
        "id_classes": list(bundle.id_classes),
        "ood_classes": list(bundle.ood_classes),
        "feature_cols": list(bundle.preprocessor.feature_cols),
        "ood_fusion": ood_cal.fusion,
        "ood_fusion_weights": ordered_ood_weights(ood_cal),
        "ood_score_direction_mode": args.ood_direction_calibration,
        "ood_score_direction_label_source": ood_cal.direction_label_source,
        "ood_score_direction_report": ordered_ood_direction_report(ood_cal),
        "direction_used_for_final_scores": ordered_ood_directions(ood_cal),
        "test_ood_direction_diagnostic_report": test_ood_direction_diagnostic_report,
        "phase_aware_threshold": phase_threshold_summary(ood_cal),
        "pseudo_ood_summary": pseudo_summary,
        "calibration_config": {
            "q_ood": args.q_ood,
            "bank_k": args.bank_k,
            "fusion": ood_cal.fusion,
            "ood_direction_calibration": args.ood_direction_calibration,
            "ood_threshold_mode": ood_cal.ood_threshold_mode,
            "group_threshold_strategy": ood_cal.group_threshold_strategy,
            "group_threshold_shrink_k": float(args.group_threshold_shrink_k),
            "group_threshold_min_ratio": float(args.group_threshold_min_ratio),
            "group_threshold_min_samples": int(args.group_threshold_min_samples),
            "phase_aware_threshold": bool(args.phase_aware_threshold),
            "phase_column": args.phase_column,
            "phase_threshold_quantile": args.phase_threshold_quantile,
            "phase_threshold_min_samples": int(args.phase_threshold_min_samples),
            "phase_threshold_fallback": args.phase_threshold_fallback,
        },
        "temperature": float(temperature),
        "class_thresholds": class_thresholds.tolist(),
        "id_test": id_test,
        "id_test_by_group": id_test_by_group,
        "ood_test": ood_test,
        "ood_test_by_group": ood_test_by_group,
        **ood_threshold_summary(ood_cal),
    }
    return report


def report_note(report: dict) -> str:
    if report.get("skip_reason"):
        return str(report["skip_reason"])
    if report.get("error_message"):
        return str(report["error_message"])
    return ""


def build_known_detection_row(method: ComparisonMethodConfig, report: dict) -> dict:
    metrics = report.get("id_test", {})
    return {
        "method": method.display_name,
        "method_name": method.method_name,
        "category": method.category,
        "status": report.get("status", "success"),
        "micro_f1": metrics.get("micro_f1"),
        "macro_f1": metrics.get("macro_f1"),
        "mAP": metrics.get("mAP"),
        "hamming_loss": metrics.get("hamming_loss"),
        "subset_accuracy": metrics.get("subset_accuracy"),
        "note": report_note(report),
    }


def build_ood_detection_row(method: ComparisonMethodConfig, report: dict) -> dict:
    metrics = report.get("ood_test", {})
    return {
        "method": method.display_name,
        "method_name": method.method_name,
        "category": method.category,
        "status": report.get("status", "success"),
        "auroc": metrics.get("auroc"),
        "aupr_out": metrics.get("aupr_out"),
        "fpr95": metrics.get("fpr95"),
        "precision": metrics.get("precision"),
        "tpr": metrics.get("tpr"),
        "recall": metrics.get("recall"),
        "ood_f1": metrics.get("ood_f1"),
        "fpr_at_threshold": metrics.get("fpr_at_threshold"),
        "note": report_note(report),
    }
