from __future__ import annotations

import csv
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..dataset_registry import (
    DATASET_METADATA_BY_NAME,
    DATASET_METADATA_BY_UAV,
    SIMULATION_DATASET_BINDINGS,
    canonical_dataset_name,
    dataset_role_for_name,
)
from ..io import read_jsonl, save_json

NA = "N/A"
UNSW_SUMMARY_NOTE = "UNSW-NB15 用于外部泛化或非 UAV OOD 测试，不作为真实 UAV 飞行流量主实验依据。"

INPUT_FILE_SPECS: dict[str, dict[str, Any]] = {
    "report": {
        "filename": "report.json",
        # The current training pipeline still writes eval_report.json.
        "aliases": ("eval_report.json",),
    },
    "simulation_summary": {"filename": "simulation_summary.json", "aliases": ()},
    "benchmark_report": {"filename": "benchmark_report.json", "aliases": ()},
    "group_detection_summary": {"filename": "group_detection_summary.json", "aliases": ()},
    "ablation_summary": {"filename": "ablation_summary.json", "aliases": ()},
    "cross_uav_summary": {"filename": "cross_uav_summary.json", "aliases": ()},
}

DATASET_MAPPING_COLUMNS = [
    "dataset_display",
    "dataset_name",
    "uav_id",
    "source_type",
    "dataset_role",
    "role",
    "sample_count",
    "attack_count",
    "summary_note",
]

PER_UAV_COLUMNS = [
    "uav_id",
    "dataset_display",
    "dataset_name",
    "source_type",
    "dataset_role",
    "sample_count",
    "attack_count",
    "micro_f1",
    "macro_f1",
    "present_class_macro_f1",
    "present_class_count",
    "present_class_names",
    "absent_class_count",
    "subset_accuracy",
    "mAP",
    "auroc",
    "auprc",
    "fpr95",
    "recall",
    "precision",
    "alert_count",
    "false_alert_count",
]

PER_DATASET_COLUMNS = [
    "dataset_display",
    "dataset_name",
    "uav_id",
    "source_type",
    "dataset_role",
    "sample_count",
    "attack_count",
    "micro_f1",
    "macro_f1",
    "auroc",
    "auprc",
    "fpr95",
    "recall",
    "precision",
    "alert_count",
    "summary_note",
]

PER_SOURCE_TYPE_COLUMNS = [
    "source_type",
    "uav_ids",
    "dataset_names",
    "sample_count",
    "attack_count",
    "micro_f1",
    "macro_f1",
    "auroc",
    "auprc",
    "fpr95",
    "recall",
    "precision",
    "alert_count",
    "summary_note",
]

LATEX_DATASET_MAPPING_COLUMNS = [
    "dataset_display",
    "dataset_name",
    "uav_id",
    "source_type",
    "summary_note",
]

SUMMARY_METRIC_COLUMNS = (
    "sample_count",
    "attack_count",
    "micro_f1",
    "macro_f1",
    "present_class_macro_f1",
    "present_class_count",
    "absent_class_count",
    "subset_accuracy",
    "mAP",
    "auroc",
    "auprc",
    "fpr95",
    "recall",
    "precision",
    "alert_count",
)

REPORTING_SOURCE_TYPE_ORDER = ["uav", "uav_gcs", "uav_iot_wifi", "external_non_uav"]
REPORTING_SOURCE_TYPE_OVERRIDES = {"gcs_to_uav_updated": "uav_gcs"}

ENERGY_COLUMNS = [
    "total_flight_energy_j",
    "total_communication_energy_j",
    "total_sensing_energy_j",
    "total_ids_energy_j",
    "total_energy_j",
    "ids_energy_ratio",
    "average_inference_latency_ms",
    "max_inference_latency_ms",
    "battery_drop_by_ids_percent",
    "power_source",
    "transformer_extra_power_w",
]


def _safe_load_json(path: Path) -> tuple[Any, bool, str | None]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - defensive path
        return {}, False, f"{type(exc).__name__}: {exc}"
    return payload, True, None


def _safe_load_jsonl(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    try:
        rows = read_jsonl(path)
    except Exception:  # pragma: no cover - defensive path
        return []
    return [row for row in rows if isinstance(row, dict)]


def _resolve_related_path(base_dir: Path, raw_path: Any) -> Path | None:
    if raw_path is None:
        return None
    text = str(raw_path).strip()
    if not text:
        return None
    path = Path(text)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, str) and value.strip() in {"", "NA", "N/A"}:
        return True
    if isinstance(value, Mapping) and len(value) == 0:
        return True
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)) and len(value) == 0:
        return True
    return False


def _clean_value(value: Any) -> Any:
    if isinstance(value, float) and math.isnan(value):
        return NA
    if value is None:
        return NA
    if isinstance(value, Mapping):
        cleaned: dict[str, Any] = {}
        for key, item in value.items():
            normalized = _clean_value(item)
            if not _is_missing(normalized):
                cleaned[str(key)] = normalized
        return cleaned or NA
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        cleaned_items = [_clean_value(item) for item in value]
        cleaned_items = [item for item in cleaned_items if not _is_missing(item)]
        return cleaned_items or NA
    if isinstance(value, str):
        text = value.strip()
        return text if text else NA
    return value


def _format_float(value: float) -> str:
    text = f"{float(value):.4f}"
    return text.rstrip("0").rstrip(".")


def _display_value(value: Any) -> str:
    normalized = _clean_value(value)
    if _is_missing(normalized):
        return NA
    if isinstance(normalized, bool):
        return "true" if normalized else "false"
    if isinstance(normalized, int):
        return str(normalized)
    if isinstance(normalized, float):
        return _format_float(normalized)
    if isinstance(normalized, Mapping):
        if all(not isinstance(item, (Mapping, list, tuple)) for item in normalized.values()):
            return "; ".join(f"{key}={_display_value(item)}" for key, item in normalized.items()) or NA
        return json.dumps(normalized, ensure_ascii=False, sort_keys=True)
    if isinstance(normalized, Sequence) and not isinstance(normalized, str):
        return ", ".join(_display_value(item) for item in normalized) or NA
    return str(normalized)


def _to_number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        if isinstance(value, float) and math.isnan(value):
            return None
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if not text or text in {"NA", "N/A"}:
            return None
        try:
            return float(text)
        except ValueError:
            return None
    return None


def _get_nested(payload: Any, dotted_key: str) -> Any:
    current = payload
    for part in dotted_key.split("."):
        if isinstance(current, Mapping):
            if part not in current:
                return None
            current = current[part]
            continue
        if isinstance(current, Sequence) and not isinstance(current, (str, bytes, bytearray)):
            try:
                index = int(part)
            except ValueError:
                return None
            if index < 0 or index >= len(current):
                return None
            current = current[index]
            continue
        return None
    return current


def _pick_first(payloads: Sequence[Any], candidates: Sequence[str]) -> Any:
    for payload in payloads:
        for candidate in candidates:
            value = _get_nested(payload, candidate)
            if not _is_missing(value):
                return value
    return NA


def _first_non_missing(*values: Any) -> Any:
    for value in values:
        if not _is_missing(value):
            return value
    return NA


def _dict_from_pairs(items: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    return {key: _clean_value(value) for key, value in items}


def _pretty_model_name(value: Any) -> str:
    if _is_missing(value):
        return "当前模型"
    mapping = {
        "transformer_only": "Transformer-only",
        "mlp_only": "MLP-only",
        "gcn_only": "GCN-only",
        "random_graph": "Random-graph",
        "full": "Full model",
    }
    key = str(value).strip()
    return mapping.get(key, key.replace("_", " "))


def _summary_language(config: Mapping[str, Any]) -> str:
    raw_value = _clean_value(config.get("language", "zh"))
    if _is_missing(raw_value):
        return "zh"
    language = str(raw_value).strip().lower()
    return "zh" if language in {"zh", "zh-cn", "zh_hans", "cn", "chinese"} else "en"


def _present_class_macro_sentence(config: Mapping[str, Any]) -> str:
    if _summary_language(config) == "zh":
        return (
            "由于每个 UAV 只包含全局标签空间中的部分攻击类型，直接计算 per-UAV Macro-F1 会将不存在类别也纳入平均，"
            "从而低估该 UAV 的真实分类效果。因此本文额外报告 Present-Class Macro-F1，仅在该 UAV 测试集中实际出现过的类别上计算平均 F1。"
        )
    return (
        "per-UAV Macro-F1 may be underestimated when a UAV contains only a subset of global attack labels. "
        "Present-Class Macro-F1 is therefore additionally reported by averaging F1 only over labels that actually "
        "appear in the UAV's test windows."
    )


def _canonical_uav_id(value: Any) -> str:
    text = str(value or "").strip()
    return text or NA


def _resolve_dataset_role(dataset_name: Any) -> Any:
    cleaned = _clean_value(dataset_name)
    if _is_missing(cleaned):
        return NA
    role = dataset_role_for_name(str(cleaned))
    return NA if role == "unknown" else role


def _dataset_metadata_for_summary(uav_id: Any, dataset_name: Any) -> dict[str, Any]:
    key_uav = str(uav_id or "").strip()
    if key_uav and key_uav in DATASET_METADATA_BY_UAV:
        return dict(DATASET_METADATA_BY_UAV[key_uav])
    key_name = canonical_dataset_name(str(dataset_name or ""))
    if key_name and key_name in DATASET_METADATA_BY_NAME:
        return dict(DATASET_METADATA_BY_NAME[key_name])
    return {}


def _dataset_display_for_summary(uav_id: Any, dataset_name: Any) -> Any:
    metadata = _dataset_metadata_for_summary(uav_id, dataset_name)
    display = _clean_value(metadata.get("dataset_display"))
    if not _is_missing(display):
        return display
    key_name = canonical_dataset_name(str(dataset_name or ""))
    return key_name or NA


def _summary_note_for_dataset(uav_id: Any, dataset_name: Any) -> Any:
    metadata = _dataset_metadata_for_summary(uav_id, dataset_name)
    key_name = str(metadata.get("dataset_name") or canonical_dataset_name(str(dataset_name or ""))).strip()
    if key_name == "unsw_nb15":
        return UNSW_SUMMARY_NOTE
    note = _clean_value(metadata.get("source_note"))
    return note if not _is_missing(note) else NA


def _reporting_source_type(uav_id: Any, dataset_name: Any, source_type: Any = None) -> Any:
    metadata = _dataset_metadata_for_summary(uav_id, dataset_name)
    key_name = str(metadata.get("dataset_name") or canonical_dataset_name(str(dataset_name or ""))).strip()
    if key_name in REPORTING_SOURCE_TYPE_OVERRIDES:
        return REPORTING_SOURCE_TYPE_OVERRIDES[key_name]
    candidate = _first_non_missing(metadata.get("source_type"), source_type)
    cleaned = _clean_value(candidate)
    return cleaned if not _is_missing(cleaned) else NA


def _ordered_reporting_uav_ids() -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for binding in SIMULATION_DATASET_BINDINGS:
        uav_id = str(binding.get("uav_id", "")).strip()
        if not uav_id or uav_id in seen:
            continue
        seen.add(uav_id)
        ordered.append(uav_id)
    return ordered


def _unique_text_values(values: Sequence[Any]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        cleaned = _clean_value(value)
        if _is_missing(cleaned):
            continue
        text = str(cleaned).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        ordered.append(text)
    return ordered


def _join_unique_text_values(values: Sequence[Any]) -> Any:
    joined = ", ".join(_unique_text_values(values))
    return joined if joined else NA


def _sum_numeric_values(values: Sequence[Any]) -> Any:
    numeric_values = [_to_number(value) for value in values]
    numeric_values = [float(value) for value in numeric_values if value is not None]
    if not numeric_values:
        return NA
    total = float(sum(numeric_values))
    return int(total) if total.is_integer() else total


def _weighted_average(rows: Sequence[Mapping[str, Any]], metric_key: str, weight_key: str) -> Any:
    weighted_values: list[tuple[float, float]] = []
    fallback_values: list[float] = []
    for row in rows:
        metric_value = _to_number(row.get(metric_key))
        if metric_value is None:
            continue
        fallback_values.append(float(metric_value))
        weight_value = _to_number(row.get(weight_key))
        if weight_value is not None and float(weight_value) > 0.0:
            weighted_values.append((float(metric_value), float(weight_value)))
    if weighted_values:
        total_weight = float(sum(weight for _, weight in weighted_values))
        if total_weight > 0.0:
            return sum(value * weight for value, weight in weighted_values) / total_weight
    if fallback_values:
        return sum(fallback_values) / len(fallback_values)
    return NA


def _row_has_metric_data(row: Mapping[str, Any]) -> bool:
    return any(not _is_missing(row.get(key, NA)) for key in SUMMARY_METRIC_COLUMNS)


def load_report_bundle(reports_dir: str | Path) -> dict[str, Any]:
    reports_path = Path(reports_dir)
    reports_path.mkdir(parents=True, exist_ok=True)

    reports: dict[str, Any] = {}
    sources: dict[str, Any] = {}
    for key, spec in INPUT_FILE_SPECS.items():
        requested_path = reports_path / spec["filename"]
        resolved_path: Path | None = requested_path if requested_path.exists() else None
        if resolved_path is None:
            for alias in spec.get("aliases", ()):
                alias_path = reports_path / alias
                if alias_path.exists():
                    resolved_path = alias_path
                    break

        payload: Any = {}
        valid_json = False
        error = None
        if resolved_path is not None:
            payload, valid_json, error = _safe_load_json(resolved_path)
            if not isinstance(payload, (Mapping, list)):
                payload = {}
        reports[key] = payload
        sources[key] = {
            "requested_path": str(requested_path),
            "resolved_path": None if resolved_path is None else str(resolved_path),
            "exists": resolved_path is not None,
            "used_alias": resolved_path is not None and resolved_path.name != spec["filename"],
            "valid_json": bool(valid_json),
            "error": error,
        }

    simulation_summary = reports.get("simulation_summary", {})
    records_path = None
    detections_path = None
    if isinstance(simulation_summary, Mapping):
        output_files = simulation_summary.get("output_files", {})
        if isinstance(output_files, Mapping):
            records_path = _resolve_related_path(reports_path, output_files.get("simulation_records"))
            detections_path = _resolve_related_path(reports_path, output_files.get("simulation_detections"))

    fallback_records = reports_path / "simulation_records.jsonl"
    if records_path is None and fallback_records.exists():
        records_path = fallback_records
    fallback_detections = reports_path / "simulation_detections.jsonl"
    if detections_path is None and fallback_detections.exists():
        detections_path = fallback_detections
    generic_detections = reports_path / "detections.jsonl"
    if detections_path is None and generic_detections.exists():
        detections_path = generic_detections

    related = {
        "simulation_records": _safe_load_jsonl(records_path),
        "detection_rows": _safe_load_jsonl(detections_path),
    }
    related_sources = {
        "simulation_records": None if records_path is None else str(records_path),
        "detection_rows": None if detections_path is None else str(detections_path),
    }
    return {
        "reports_dir": str(reports_path),
        "reports": reports,
        "sources": sources,
        "related": related,
        "related_sources": related_sources,
    }


def _extract_per_class_maps(report_payload: Any) -> tuple[Any, Any, Any]:
    class_names = _pick_first([report_payload], ["class_names"])
    class_names_list = class_names if isinstance(class_names, list) else []

    def _coerce_named_metric(value: Any) -> Any:
        if _is_missing(value):
            return NA
        if isinstance(value, Mapping):
            return {
                str(key): _clean_value(item)
                for key, item in value.items()
                if not _is_missing(_clean_value(item))
            } or NA
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            if class_names_list and len(class_names_list) == len(value):
                return {
                    str(name): _clean_value(item)
                    for name, item in zip(class_names_list, value)
                    if not _is_missing(_clean_value(item))
                } or NA
            return NA
        return NA

    precision = _coerce_named_metric(
        _pick_first(
            [report_payload],
            [
                "id_test.per_class_precision",
                "known_attack.per_class_precision",
                "per_class_precision",
            ],
        )
    )
    recall = _coerce_named_metric(
        _pick_first(
            [report_payload],
            [
                "id_test.per_class_recall",
                "known_attack.per_class_recall",
                "per_class_recall",
            ],
        )
    )
    f1 = _coerce_named_metric(
        _pick_first(
            [report_payload],
            [
                "id_test.per_class_f1",
                "known_attack.per_class_f1",
                "per_class_f1",
            ],
        )
    )

    per_class_metrics = _pick_first(
        [report_payload],
        [
            "id_test.per_class_metrics",
            "known_attack.per_class_metrics",
            "per_class_metrics",
        ],
    )
    if isinstance(per_class_metrics, Mapping):
        if _is_missing(precision):
            precision = {
                str(label): _clean_value(values.get("precision"))
                for label, values in per_class_metrics.items()
                if isinstance(values, Mapping) and not _is_missing(values.get("precision"))
            } or NA
        if _is_missing(recall):
            recall = {
                str(label): _clean_value(values.get("recall"))
                for label, values in per_class_metrics.items()
                if isinstance(values, Mapping) and not _is_missing(values.get("recall"))
            } or NA
        if _is_missing(f1):
            f1 = {
                str(label): _clean_value(values.get("f1"))
                for label, values in per_class_metrics.items()
                if isinstance(values, Mapping) and not _is_missing(values.get("f1"))
            } or NA
    elif isinstance(per_class_metrics, list):
        def _from_list(metric_name: str) -> Any:
            rows: dict[str, Any] = {}
            for item in per_class_metrics:
                if not isinstance(item, Mapping):
                    continue
                label = _first_non_missing(item.get("class_name"), item.get("label"), item.get("class"))
                if _is_missing(label):
                    continue
                metric_value = item.get(metric_name)
                if _is_missing(metric_value):
                    continue
                rows[str(label)] = _clean_value(metric_value)
            return rows or NA

        if _is_missing(precision):
            precision = _from_list("precision")
        if _is_missing(recall):
            recall = _from_list("recall")
        if _is_missing(f1):
            f1 = _from_list("f1")

    return _clean_value(precision), _clean_value(recall), _clean_value(f1)


def _derive_dominant_ood_source(bundle: dict[str, Any]) -> Any:
    report = bundle["reports"].get("report", {})
    rows = bundle["related"].get("detection_rows", [])
    explicit = _pick_first(
        [report],
        [
            "ood_test.dominant_ood_source",
            "dominant_ood_source",
        ],
    )
    if not _is_missing(explicit):
        return explicit

    sources = [
        str(row.get("dominant_ood_source")).strip()
        for row in rows
        if isinstance(row, Mapping) and bool(row.get("is_ood", False)) and str(row.get("dominant_ood_source", "")).strip()
    ]
    if sources:
        return Counter(sources).most_common(1)[0][0]

    fusion_weights = _pick_first([report], ["ood_fusion_weights"])
    if isinstance(fusion_weights, Mapping):
        numeric_items = [
            (str(name), _to_number(weight))
            for name, weight in fusion_weights.items()
            if _to_number(weight) is not None
        ]
        if numeric_items:
            return max(numeric_items, key=lambda item: float(item[1]))[0]

    direction_report = _pick_first([report], ["ood_score_direction_report"])
    if isinstance(direction_report, list):
        ranked: list[tuple[str, float]] = []
        for item in direction_report:
            if not isinstance(item, Mapping):
                continue
            score_name = str(item.get("score_name", "")).strip()
            effective_auroc = _to_number(item.get("effective_auroc"))
            if score_name and effective_auroc is not None:
                ranked.append((score_name, float(effective_auroc)))
        if ranked:
            return max(ranked, key=lambda item: item[1])[0]

    return NA


def _extract_experiment_config(bundle: dict[str, Any]) -> dict[str, Any]:
    report = bundle["reports"].get("report", {})
    benchmark = bundle["reports"].get("benchmark_report", {})
    payloads = [report, benchmark]
    return _dict_from_pairs(
        [
            ("model_name", _pick_first(payloads, ["encoder_ablation", "run_config.encoder_ablation"])),
            ("graph_enabled", _pick_first(payloads, ["graph_enabled", "run_config.graph_enabled"])),
            ("graph_variant", _pick_first(payloads, ["graph_variant", "run_config.graph_variant"])),
            ("ood_fusion", _pick_first(payloads, ["ood_fusion", "calibration_config.fusion", "fusion"])),
            ("normalization_mode", _pick_first(payloads, ["normalization_mode", "run_config.normalization_mode", "normalization.mode"])),
            ("ood_threshold_mode", _pick_first(payloads, ["ood_threshold_mode", "calibration_config.ood_threshold_mode"])),
            ("group_col", _pick_first(payloads, ["group_config.group_col", "group_col"])),
            ("window_mode", _pick_first(payloads, ["window_config.mode", "window_mode"])),
            ("window_size", _pick_first(payloads, ["window_config.size", "window_size"])),
            ("stride", _pick_first(payloads, ["window_config.stride", "stride"])),
            ("class_names", _pick_first(payloads, ["class_names"])),
            ("language", _pick_first(payloads, ["language", "reporting.language", "reporting_config.language"])),
            ("seed", _pick_first(payloads, ["seed", "run_config.seed"])),
            ("device", _pick_first(payloads, ["device"])),
            ("num_windows", _pick_first(payloads, ["num_windows"])),
        ]
    )


def _extract_known_attack_metrics(bundle: dict[str, Any]) -> dict[str, Any]:
    report = bundle["reports"].get("report", {})
    precision, recall, f1 = _extract_per_class_maps(report)
    return _dict_from_pairs(
        [
            ("micro_f1", _pick_first([report], ["id_test.micro_f1", "known_attack.micro_f1", "micro_f1"])),
            ("macro_f1", _pick_first([report], ["id_test.macro_f1", "known_attack.macro_f1", "macro_f1"])),
            ("mAP", _pick_first([report], ["id_test.mAP", "known_attack.mAP", "known_attack.map", "mAP", "map"])),
            ("subset_accuracy", _pick_first([report], ["id_test.subset_accuracy", "known_attack.subset_accuracy", "subset_accuracy"])),
            ("hamming_loss", _pick_first([report], ["id_test.hamming_loss", "known_attack.hamming_loss", "hamming_loss"])),
            ("per_class_precision", precision),
            ("per_class_recall", recall),
            ("per_class_f1", f1),
        ]
    )


def _extract_ood_metrics(bundle: dict[str, Any]) -> dict[str, Any]:
    report = bundle["reports"].get("report", {})
    payload = _dict_from_pairs(
        [
            ("auroc", _pick_first([report], ["ood_test.auroc", "ood_unknown_attack.auroc", "auroc"])),
            ("auprc", _pick_first([report], ["ood_test.auprc", "ood_test.aupr_out", "ood_unknown_attack.auprc", "ood_unknown_attack.aupr_out", "auprc", "aupr_out"])),
            ("fpr95", _pick_first([report], ["ood_test.fpr95", "ood_unknown_attack.fpr95", "fpr95"])),
            ("ood_f1", _pick_first([report], ["ood_test.ood_f1", "ood_unknown_attack.ood_f1", "ood_f1"])),
            ("ood_precision", _pick_first([report], ["ood_test.precision", "ood_test.ood_precision", "ood_precision", "precision"])),
            ("ood_recall", _pick_first([report], ["ood_test.recall", "ood_test.tpr", "ood_recall", "recall", "tpr"])),
            ("best_threshold", _pick_first([report], ["ood_test.best_threshold", "ood_test.threshold_report_only", "best_threshold", "global_ood_threshold"])),
            ("threshold_mode", _pick_first([report], ["ood_threshold_mode", "calibration_config.ood_threshold_mode", "threshold_mode"])),
            ("dominant_ood_source", _derive_dominant_ood_source(bundle)),
        ]
    )
    return payload


def _group_rows_from_simulation(records: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for record in records:
        uav_id = _canonical_uav_id(record.get("uav_id"))
        if _is_missing(uav_id):
            continue
        entry = rows.setdefault(
            uav_id,
            {
                "sample_count": 0,
                "attack_count": 0,
                "dataset_names": Counter(),
            },
        )
        entry["sample_count"] += 1
        attack_active = _first_non_missing(record.get("gt_attack_active"), record.get("attack_active"))
        if not _is_missing(attack_active) and bool(attack_active):
            entry["attack_count"] += 1
        dataset_name = str(record.get("dataset_name", "")).strip()
        if dataset_name:
            entry["dataset_names"][dataset_name] += 1
    return rows


def _group_rows_from_detections(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for row in rows:
        group_id = _canonical_uav_id(_first_non_missing(row.get("group_id"), row.get("uav_id")))
        if _is_missing(group_id):
            continue
        item = grouped.setdefault(
            group_id,
            {
                "alert_count": 0,
                "false_alert_count": 0,
            },
        )
        is_alert = bool(row.get("is_ood", False)) or str(row.get("alert_level", "")).strip().lower() in {"warning", "critical"}
        if is_alert:
            item["alert_count"] += 1
            gt_attack_active = _first_non_missing(row.get("gt_attack_active"), _get_nested(row, "ground_truth.attack_active"))
            if gt_attack_active is False:
                item["false_alert_count"] += 1
    return grouped


def _extract_cross_uav(bundle: dict[str, Any]) -> dict[str, Any]:
    cross = bundle["reports"].get("cross_uav_summary", {})
    return _dict_from_pairs(
        [
            ("source_uav", _pick_first([cross], ["source_uav"])),
            ("target_uav", _pick_first([cross], ["target_uav"])),
            ("target_micro_f1", _pick_first([cross], ["target_group_metrics.id_test.micro_f1", "target_id_metrics.micro_f1"])),
            ("target_macro_f1", _pick_first([cross], ["target_group_metrics.id_test.macro_f1", "target_id_metrics.macro_f1"])),
            ("target_auroc", _pick_first([cross], ["target_group_metrics.ood_test.auroc", "target_ood_metrics.auroc"])),
            ("target_fpr95", _pick_first([cross], ["target_group_metrics.ood_test.fpr95", "target_ood_metrics.fpr95"])),
            ("target_alert_count", _pick_first([cross], ["target_group_metrics.detection_summary.ood_alerts", "target_group_metrics.detection_summary.alert_count"])),
            ("source_train_samples", _pick_first([cross], ["used_rows.source_train"])),
            ("source_val_samples", _pick_first([cross], ["used_rows.source_val"])),
            ("target_test_id_samples", _pick_first([cross], ["used_rows.target_test_id"])),
            ("target_test_ood_samples", _pick_first([cross], ["used_rows.target_test_ood"])),
        ]
    )


def _build_per_uav_rows(bundle: dict[str, Any]) -> list[dict[str, Any]]:
    report = bundle["reports"].get("report", {})
    group_detection = bundle["reports"].get("group_detection_summary", {})
    cross = bundle["reports"].get("cross_uav_summary", {})
    simulation_rows = _group_rows_from_simulation(bundle["related"].get("simulation_records", []))
    detection_rows = _group_rows_from_detections(bundle["related"].get("detection_rows", []))

    id_by_group = _get_nested(report, "id_test_by_group")
    if not isinstance(id_by_group, Mapping):
        id_by_group = {}
    ood_by_group = _get_nested(report, "ood_test_by_group")
    if not isinstance(ood_by_group, Mapping):
        ood_by_group = {}
    detected_groups = _get_nested(group_detection, "groups")
    if not isinstance(detected_groups, Mapping):
        detected_groups = {}
    cross_target = str(cross.get("target_uav", "")).strip()
    cross_source = str(cross.get("source_uav", "")).strip()
    cross_target_group = cross.get("target_group_metrics", {}) if isinstance(cross.get("target_group_metrics", {}), Mapping) else {}

    all_groups: set[str] = set(_ordered_reporting_uav_ids())
    all_groups.update(str(key).strip() for key in id_by_group)
    all_groups.update(str(key).strip() for key in ood_by_group)
    all_groups.update(str(key).strip() for key in detected_groups)
    all_groups.update(str(key).strip() for key in simulation_rows)
    all_groups.update(str(key).strip() for key in detection_rows)
    if cross_target:
        all_groups.add(cross_target)
    if cross_source:
        all_groups.add(cross_source)
    all_groups.discard("")

    rows: list[dict[str, Any]] = []
    for group in sorted(all_groups):
        id_metrics = id_by_group.get(group, {}) if isinstance(id_by_group.get(group), Mapping) else {}
        ood_metrics = ood_by_group.get(group, {}) if isinstance(ood_by_group.get(group), Mapping) else {}
        detect_metrics = detected_groups.get(group, {}) if isinstance(detected_groups.get(group), Mapping) else {}
        sim_metrics = simulation_rows.get(group, {})
        detection_metrics = detection_rows.get(group, {})
        cross_id = {}
        cross_ood = {}
        cross_detect = {}
        if group == cross_target and isinstance(cross_target_group, Mapping):
            cross_id = cross_target_group.get("id_test", {}) if isinstance(cross_target_group.get("id_test", {}), Mapping) else {}
            cross_ood = cross_target_group.get("ood_test", {}) if isinstance(cross_target_group.get("ood_test", {}), Mapping) else {}
            cross_detect = cross_target_group.get("detection_summary", {}) if isinstance(cross_target_group.get("detection_summary", {}), Mapping) else {}

        if group == cross_source and isinstance(cross, Mapping):
            source_used_rows = cross.get("used_rows", {})
            if isinstance(source_used_rows, Mapping):
                source_sample_count = _to_number(source_used_rows.get("source_train"))
                source_val_count = _to_number(source_used_rows.get("source_val"))
                if source_sample_count is not None:
                    sim_metrics = dict(sim_metrics)
                    sim_metrics["sample_count"] = int(source_sample_count + (source_val_count or 0))

        metadata_guess = _dataset_metadata_for_summary(
            group,
            _first_non_missing(
                id_metrics.get("dataset_name"),
                ood_metrics.get("dataset_name"),
                detect_metrics.get("dataset_name"),
                (
                    sim_metrics.get("dataset_names", Counter()).most_common(1)[0][0]
                    if isinstance(sim_metrics.get("dataset_names"), Counter) and sim_metrics["dataset_names"]
                    else NA
                ),
            ),
        )
        dataset_name = _first_non_missing(
            id_metrics.get("dataset_name"),
            ood_metrics.get("dataset_name"),
            detect_metrics.get("dataset_name"),
            (
                sim_metrics.get("dataset_names", Counter()).most_common(1)[0][0]
                if isinstance(sim_metrics.get("dataset_names"), Counter) and sim_metrics["dataset_names"]
                else NA
            ),
            metadata_guess.get("dataset_name"),
            "source" if group == cross_source and not _is_missing(cross_source) else NA,
            "target" if group == cross_target and not _is_missing(cross_target) else NA,
        )
        dataset_role = _first_non_missing(metadata_guess.get("dataset_role"), _resolve_dataset_role(dataset_name))
        dataset_display = _first_non_missing(
            id_metrics.get("dataset_display"),
            ood_metrics.get("dataset_display"),
            detect_metrics.get("dataset_display"),
            metadata_guess.get("dataset_display"),
            _dataset_display_for_summary(group, dataset_name),
        )
        sample_count = _first_non_missing(
            id_metrics.get("windows"),
            ood_metrics.get("windows"),
            detect_metrics.get("windows"),
            cross_detect.get("windows"),
            sim_metrics.get("sample_count"),
        )
        attack_count = _first_non_missing(
            ood_metrics.get("ood_windows"),
            id_metrics.get("attack_count"),
            detect_metrics.get("attack_count"),
            sim_metrics.get("attack_count"),
            _get_nested(cross, "used_rows.target_test_ood") if group == cross_target else NA,
        )
        row = _dict_from_pairs(
            [
                ("uav_id", group),
                ("dataset_display", dataset_display),
                ("dataset_name", dataset_name),
                (
                    "source_type",
                    _reporting_source_type(
                        group,
                        dataset_name,
                        _first_non_missing(
                            id_metrics.get("source_type"),
                            ood_metrics.get("source_type"),
                            detect_metrics.get("source_type"),
                            metadata_guess.get("source_type"),
                        ),
                    ),
                ),
                ("dataset_role", dataset_role),
                ("sample_count", sample_count),
                ("attack_count", attack_count),
                (
                    "micro_f1",
                    _first_non_missing(id_metrics.get("micro_f1"), cross_id.get("micro_f1"), detect_metrics.get("micro_f1")),
                ),
                (
                    "macro_f1",
                    _first_non_missing(id_metrics.get("macro_f1"), cross_id.get("macro_f1"), detect_metrics.get("macro_f1")),
                ),
                (
                    "present_class_macro_f1",
                    _first_non_missing(
                        id_metrics.get("present_class_macro_f1"),
                        cross_id.get("present_class_macro_f1"),
                        detect_metrics.get("present_class_macro_f1"),
                    ),
                ),
                (
                    "present_class_count",
                    _first_non_missing(
                        id_metrics.get("present_class_count"),
                        cross_id.get("present_class_count"),
                        detect_metrics.get("present_class_count"),
                    ),
                ),
                (
                    "present_class_names",
                    _first_non_missing(
                        id_metrics.get("present_class_names"),
                        cross_id.get("present_class_names"),
                        detect_metrics.get("present_class_names"),
                    ),
                ),
                (
                    "absent_class_count",
                    _first_non_missing(
                        id_metrics.get("absent_class_count"),
                        cross_id.get("absent_class_count"),
                        detect_metrics.get("absent_class_count"),
                    ),
                ),
                (
                    "subset_accuracy",
                    _first_non_missing(
                        id_metrics.get("subset_accuracy"),
                        cross_id.get("subset_accuracy"),
                        detect_metrics.get("subset_accuracy"),
                    ),
                ),
                ("mAP", _first_non_missing(id_metrics.get("mAP"), cross_id.get("mAP"), detect_metrics.get("mAP"))),
                ("auroc", _first_non_missing(ood_metrics.get("auroc"), cross_ood.get("auroc"))),
                (
                    "auprc",
                    _first_non_missing(
                        ood_metrics.get("auprc"),
                        ood_metrics.get("aupr_out"),
                        cross_ood.get("auprc"),
                        cross_ood.get("aupr_out"),
                    ),
                ),
                ("fpr95", _first_non_missing(ood_metrics.get("fpr95"), cross_ood.get("fpr95"))),
                (
                    "recall",
                    _first_non_missing(
                        ood_metrics.get("recall"),
                        ood_metrics.get("tpr"),
                        ood_metrics.get("ood_recall"),
                        cross_ood.get("recall"),
                        cross_ood.get("tpr"),
                        cross_ood.get("ood_recall"),
                    ),
                ),
                (
                    "precision",
                    _first_non_missing(
                        ood_metrics.get("precision"),
                        ood_metrics.get("ood_precision"),
                        cross_ood.get("precision"),
                        cross_ood.get("ood_precision"),
                    ),
                ),
                ("alert_count", _first_non_missing(detect_metrics.get("ood_alerts"), detect_metrics.get("alert_count"), cross_detect.get("ood_alerts"), cross_detect.get("alert_count"), detection_metrics.get("alert_count"))),
                ("false_alert_count", _first_non_missing(detect_metrics.get("false_alert_count"), cross_detect.get("false_alert_count"), detection_metrics.get("false_alert_count"))),
                ("summary_note", _summary_note_for_dataset(group, dataset_name)),
            ]
        )
        rows.append(row)
    return rows


def _extract_dataset_uav_mapping(bundle: dict[str, Any], per_uav_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    cross = bundle["reports"].get("cross_uav_summary", {})
    rows_by_uav = {
        str(item.get("uav_id", "")).strip(): item
        for item in per_uav_rows
        if str(item.get("uav_id", "")).strip()
    }
    source_uav = str(cross.get("source_uav", "")).strip()
    target_uav = str(cross.get("target_uav", "")).strip()
    rows: list[dict[str, Any]] = []
    for binding in SIMULATION_DATASET_BINDINGS:
        uav_id = str(binding.get("uav_id", "")).strip()
        if not uav_id:
            continue
        item = rows_by_uav.get(uav_id, {})
        dataset_name = _first_non_missing(binding.get("dataset_name"), item.get("dataset_name"))
        rows.append(
            _dict_from_pairs(
                [
                    ("dataset_display", _first_non_missing(binding.get("dataset_display"), item.get("dataset_display"), _dataset_display_for_summary(uav_id, dataset_name))),
                    ("dataset_name", dataset_name),
                    ("uav_id", uav_id),
                    ("source_type", _reporting_source_type(uav_id, dataset_name, item.get("source_type", binding.get("source_type")))),
                    ("dataset_role", _first_non_missing(binding.get("dataset_role"), item.get("dataset_role"), _resolve_dataset_role(dataset_name))),
                    ("role", "source" if uav_id == source_uav else "target" if uav_id == target_uav else NA),
                    ("sample_count", item.get("sample_count", NA)),
                    ("attack_count", item.get("attack_count", NA)),
                    ("summary_note", _summary_note_for_dataset(uav_id, dataset_name)),
                ]
            )
        )
    return rows


def _build_per_dataset_rows(
    dataset_mapping: Sequence[Mapping[str, Any]],
    per_uav_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows_by_uav = {
        str(item.get("uav_id", "")).strip(): item
        for item in per_uav_rows
        if str(item.get("uav_id", "")).strip()
    }
    rows: list[dict[str, Any]] = []
    for mapping_row in dataset_mapping:
        uav_id = str(mapping_row.get("uav_id", "")).strip()
        metrics_row = rows_by_uav.get(uav_id, {})
        rows.append(
            _dict_from_pairs(
                [
                    ("dataset_display", _first_non_missing(mapping_row.get("dataset_display"), metrics_row.get("dataset_display"))),
                    ("dataset_name", _first_non_missing(mapping_row.get("dataset_name"), metrics_row.get("dataset_name"))),
                    ("uav_id", uav_id or NA),
                    ("source_type", _first_non_missing(mapping_row.get("source_type"), metrics_row.get("source_type"))),
                    ("dataset_role", _first_non_missing(mapping_row.get("dataset_role"), metrics_row.get("dataset_role"))),
                    ("sample_count", metrics_row.get("sample_count", NA)),
                    ("attack_count", metrics_row.get("attack_count", NA)),
                    ("micro_f1", metrics_row.get("micro_f1", NA)),
                    ("macro_f1", metrics_row.get("macro_f1", NA)),
                    ("auroc", metrics_row.get("auroc", NA)),
                    ("auprc", metrics_row.get("auprc", NA)),
                    ("fpr95", metrics_row.get("fpr95", NA)),
                    ("recall", metrics_row.get("recall", NA)),
                    ("precision", metrics_row.get("precision", NA)),
                    ("alert_count", metrics_row.get("alert_count", NA)),
                    ("summary_note", _first_non_missing(mapping_row.get("summary_note"), metrics_row.get("summary_note"))),
                ]
            )
        )
    return rows


def _build_per_source_type_rows(per_dataset_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped_rows: dict[str, list[Mapping[str, Any]]] = {}
    for row in per_dataset_rows:
        source_type = str(row.get("source_type", "")).strip()
        if not source_type:
            continue
        grouped_rows.setdefault(source_type, []).append(row)

    ordered_source_types = list(REPORTING_SOURCE_TYPE_ORDER)
    ordered_source_types.extend(sorted(key for key in grouped_rows if key not in REPORTING_SOURCE_TYPE_ORDER))

    rows: list[dict[str, Any]] = []
    for source_type in ordered_source_types:
        source_rows = grouped_rows.get(source_type, [])
        present_rows = [row for row in source_rows if _row_has_metric_data(row)]
        rows.append(
            _dict_from_pairs(
                [
                    ("source_type", source_type),
                    ("uav_ids", _join_unique_text_values([row.get("uav_id", NA) for row in present_rows])),
                    ("dataset_names", _join_unique_text_values([row.get("dataset_display", row.get("dataset_name", NA)) for row in present_rows])),
                    ("sample_count", _sum_numeric_values([row.get("sample_count", NA) for row in present_rows])),
                    ("attack_count", _sum_numeric_values([row.get("attack_count", NA) for row in present_rows])),
                    ("micro_f1", _weighted_average(present_rows, "micro_f1", "sample_count")),
                    ("macro_f1", _weighted_average(present_rows, "macro_f1", "sample_count")),
                    ("auroc", _weighted_average(present_rows, "auroc", "attack_count")),
                    ("auprc", _weighted_average(present_rows, "auprc", "attack_count")),
                    ("fpr95", _weighted_average(present_rows, "fpr95", "attack_count")),
                    ("recall", _weighted_average(present_rows, "recall", "attack_count")),
                    ("precision", _weighted_average(present_rows, "precision", "attack_count")),
                    ("alert_count", _sum_numeric_values([row.get("alert_count", NA) for row in present_rows])),
                    ("summary_note", _join_unique_text_values([row.get("summary_note", NA) for row in source_rows])),
                ]
            )
        )
    return rows


def _dataset_mapping_sentence(dataset_rows: Sequence[Mapping[str, Any]]) -> str:
    parts = ["GCS-to-UAV Updated 在 source_type 汇总中按 `uav_gcs` 单独统计。"]
    unsw_notes = [
        row.get("summary_note", NA)
        for row in dataset_rows
        if str(row.get("dataset_name", "")).strip() == "unsw_nb15"
    ]
    note = _join_unique_text_values(unsw_notes)
    if not _is_missing(note):
        parts.append(str(note))
    return " ".join(part.strip() for part in parts if part.strip())


def _extract_single_mixed_results(bundle: dict[str, Any]) -> dict[str, Any]:
    report = bundle["reports"].get("report", {})
    return _dict_from_pairs(
        [
            (
                "single_attack",
                _pick_first(
                    [report],
                    [
                        "single_attack_results",
                        "single_attack_metrics",
                        "attack_breakdown.single_attack",
                        "attack_breakdown.single",
                        "individual_attack_results",
                    ],
                ),
            ),
            (
                "mixed_attack",
                _pick_first(
                    [report],
                    [
                        "mixed_attack_results",
                        "mixed_attack_metrics",
                        "attack_breakdown.mixed_attack",
                        "attack_breakdown.mixed",
                        "mixed_window_results",
                    ],
                ),
            ),
            (
                "mix_window_stats",
                _pick_first(
                    [report],
                    [
                        "mixed_window_stats",
                        "window_mix_stats",
                        "attack_mix_stats",
                    ],
                ),
            ),
        ]
    )


def _extract_simulation_summary(bundle: dict[str, Any]) -> dict[str, Any]:
    simulation = bundle["reports"].get("simulation_summary", {})
    group_detection = bundle["reports"].get("group_detection_summary", {})
    cross = bundle["reports"].get("cross_uav_summary", {})
    detection_rows = bundle["related"].get("detection_rows", [])
    detected_groups = _group_rows_from_detections(detection_rows)
    false_alert_total = sum(
        int(metrics.get("false_alert_count", 0))
        for metrics in detected_groups.values()
        if isinstance(metrics, Mapping)
    )
    cross_payload = _extract_cross_uav(bundle)
    return _dict_from_pairs(
        [
            ("scenario_name", _pick_first([simulation], ["scenario_name"])),
            ("uav_ids", _pick_first([simulation], ["uav_ids"])),
            ("record_count", _pick_first([simulation], ["record_count"])),
            ("detection_count", _pick_first([simulation], ["detection_count"])),
            ("attack_record_count", _pick_first([simulation], ["attack_record_count"])),
            ("alert_count", _first_non_missing(_pick_first([simulation], ["alert_count"]), _pick_first([group_detection], ["global.ood_alerts"]))),
            ("false_alert_count", _first_non_missing(_pick_first([simulation], ["false_alert_count"]), false_alert_total if false_alert_total else NA)),
            ("response_count", _pick_first([simulation], ["response_count"])),
            ("average_alert_delay", _pick_first([simulation], ["average_alert_delay"])),
            ("mission_success", _pick_first([simulation], ["mission_success"])),
            ("global_alert_rate", _pick_first([group_detection], ["global.alert_rate"])),
            ("mean_ood_score", _pick_first([group_detection], ["global.mean_ood_score"])),
            ("max_ood_score", _pick_first([group_detection], ["global.max_ood_score"])),
            ("cross_uav_generalization", cross_payload),
        ]
    )


def _energy_from_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not records:
        return {}

    def _sum_wh(field_name: str) -> float | None:
        total = 0.0
        seen = False
        for row in records:
            value = _to_number(row.get(field_name))
            if value is None:
                continue
            total += float(value)
            seen = True
        return total if seen else None

    timestamps: list[float] = []
    for row in records:
        for field_name in ("simulation_time_s", "sim_time", "timestamp"):
            value = _to_number(row.get(field_name))
            if value is not None:
                timestamps.append(float(value))
                break
    duration_s = None
    if timestamps:
        duration_s = float(max(timestamps) - min(timestamps))
        if duration_s <= 0.0:
            duration_s = float(len(records))
    elif records:
        duration_s = float(len(records))

    total_flight_wh = _sum_wh("flight_energy_wh")
    total_comm_wh = _sum_wh("communication_energy_wh")
    total_ids_wh = _sum_wh("detection_energy_wh")
    total_energy_wh = _sum_wh("total_energy_wh")
    if total_energy_wh is None:
        parts = [value for value in (total_flight_wh, total_comm_wh, total_ids_wh) if value is not None]
        total_energy_wh = sum(parts) if parts else None

    derived: dict[str, Any] = {}
    if total_flight_wh is not None:
        derived["total_flight_energy_j"] = total_flight_wh * 3600.0
    if total_comm_wh is not None:
        derived["total_communication_energy_j"] = total_comm_wh * 3600.0
    if total_ids_wh is not None:
        derived["total_ids_energy_j"] = total_ids_wh * 3600.0
    if total_energy_wh is not None:
        derived["total_energy_j"] = total_energy_wh * 3600.0
    if total_ids_wh is not None and total_energy_wh is not None and total_energy_wh > 0:
        derived["ids_energy_ratio"] = total_ids_wh / total_energy_wh
    if total_ids_wh is not None and duration_s is not None and duration_s > 0:
        derived["transformer_extra_power_w"] = (total_ids_wh * 3600.0) / duration_s
    return derived


def _extract_energy_metrics(bundle: dict[str, Any]) -> dict[str, Any]:
    simulation = bundle["reports"].get("simulation_summary", {})
    benchmark = bundle["reports"].get("benchmark_report", {})
    records = bundle["related"].get("simulation_records", [])
    derived = _energy_from_records(records)

    def _pick_metric(candidates: Sequence[str], *, wh_candidates: Sequence[str] = ()) -> Any:
        value = _pick_first([simulation, benchmark], candidates)
        if not _is_missing(value):
            return value
        for candidate in wh_candidates:
            wh_value = _pick_first([simulation, benchmark], [candidate])
            numeric = _to_number(wh_value)
            if numeric is not None:
                return numeric * 3600.0
        return NA

    run_times_ms = _pick_first([benchmark], ["run_times_ms"])
    num_windows = _to_number(_pick_first([benchmark], ["num_windows"]))
    max_inference_latency = _pick_metric(["max_inference_latency_ms"])
    if _is_missing(max_inference_latency) and isinstance(run_times_ms, list) and num_windows and num_windows > 0:
        numeric_times = [_to_number(item) for item in run_times_ms]
        numeric_times = [float(item) for item in numeric_times if item is not None]
        if numeric_times:
            max_inference_latency = max(numeric_times) / float(num_windows)

    payload = _dict_from_pairs(
        [
            (
                "total_flight_energy_j",
                _first_non_missing(
                    _pick_metric(["total_flight_energy_j", "energy.total_flight_energy_j"], wh_candidates=["total_flight_energy_wh", "energy.total_flight_energy_wh"]),
                    derived.get("total_flight_energy_j", NA),
                ),
            ),
            (
                "total_communication_energy_j",
                _first_non_missing(
                    _pick_metric(["total_communication_energy_j", "energy.total_communication_energy_j"], wh_candidates=["total_communication_energy_wh", "energy.total_communication_energy_wh"]),
                    derived.get("total_communication_energy_j", NA),
                ),
            ),
            (
                "total_sensing_energy_j",
                _pick_metric(["total_sensing_energy_j", "energy.total_sensing_energy_j"], wh_candidates=["total_sensing_energy_wh", "energy.total_sensing_energy_wh"]),
            ),
            (
                "total_ids_energy_j",
                _first_non_missing(
                    _pick_metric(
                        ["total_ids_energy_j", "energy.total_ids_energy_j", "total_detection_energy_j", "energy.total_detection_energy_j"],
                        wh_candidates=["total_ids_energy_wh", "energy.total_ids_energy_wh", "total_detection_energy_wh", "energy.total_detection_energy_wh"],
                    ),
                    derived.get("total_ids_energy_j", NA),
                ),
            ),
            (
                "total_energy_j",
                _first_non_missing(
                    _pick_metric(["total_energy_j", "energy.total_energy_j"], wh_candidates=["total_energy_wh", "energy.total_energy_wh"]),
                    derived.get("total_energy_j", NA),
                ),
            ),
            (
                "ids_energy_ratio",
                _first_non_missing(
                    _pick_metric(["ids_energy_ratio", "energy.ids_energy_ratio", "detection_energy_ratio"]),
                    derived.get("ids_energy_ratio", NA),
                ),
            ),
            ("average_inference_latency_ms", _pick_metric(["average_inference_latency_ms", "average_window_inference_ms"])),
            ("max_inference_latency_ms", max_inference_latency),
            ("battery_drop_by_ids_percent", _pick_metric(["battery_drop_by_ids_percent", "energy.battery_drop_by_ids_percent"])),
            ("power_source", _pick_metric(["power_source", "energy.power_source"])),
            (
                "transformer_extra_power_w",
                _first_non_missing(
                    _pick_metric(["transformer_extra_power_w", "energy.transformer_extra_power_w"]),
                    derived.get("transformer_extra_power_w", NA),
                ),
            ),
        ]
    )

    total_ids = _to_number(payload.get("total_ids_energy_j"))
    total_energy = _to_number(payload.get("total_energy_j"))
    if _is_missing(payload.get("ids_energy_ratio")) and total_ids is not None and total_energy is not None and total_energy > 0:
        payload["ids_energy_ratio"] = total_ids / total_energy
    return payload


def _normalize_experiment_rows(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, Mapping):
        if isinstance(payload.get("experiments"), list):
            return [row for row in payload["experiments"] if isinstance(row, Mapping)]
        rows = []
        for name, value in payload.items():
            if isinstance(value, Mapping):
                row = dict(value)
                row.setdefault("experiment_name", str(name))
                rows.append(row)
        return rows
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, Mapping)]
    return []


def _best_experiment(rows: Sequence[Mapping[str, Any]], metric_keys: Sequence[str]) -> dict[str, Any]:
    best_row: Mapping[str, Any] | None = None
    best_value: float | None = None
    best_metric_name = NA
    for row in rows:
        if str(row.get("status", "success")).strip().lower() == "failed":
            continue
        for metric_name in metric_keys:
            value = _to_number(row.get(metric_name))
            if value is None:
                continue
            if best_value is None or float(value) > best_value:
                best_value = float(value)
                best_row = row
                best_metric_name = metric_name
    if best_row is None or best_value is None:
        return {
            "experiment_name": NA,
            "metric_name": NA,
            "metric_value": NA,
        }
    return {
        "experiment_name": _clean_value(best_row.get("experiment_name", NA)),
        "metric_name": best_metric_name,
        "metric_value": best_value,
    }


def _extract_ablation_summary(bundle: dict[str, Any]) -> dict[str, Any]:
    ablation_payload = bundle["reports"].get("ablation_summary", {})
    rows = _normalize_experiment_rows(ablation_payload)
    failed = []
    if isinstance(ablation_payload, Mapping) and isinstance(ablation_payload.get("failed_experiments"), list):
        failed = [str(item) for item in ablation_payload.get("failed_experiments", [])]
    elif rows:
        failed = [str(row.get("experiment_name")) for row in rows if str(row.get("status", "")).strip().lower() == "failed"]
    successful_rows = [row for row in rows if str(row.get("status", "success")).strip().lower() != "failed"]
    return _dict_from_pairs(
        [
            ("experiment_count", len(rows) if rows else NA),
            ("successful_experiment_count", len(successful_rows) if rows else NA),
            ("failed_experiments", failed or NA),
            ("best_known_attack_experiment", _best_experiment(rows, ["id_micro_f1", "micro_f1", "id_test.micro_f1"])),
            ("best_ood_experiment", _best_experiment(rows, ["ood_auroc", "auroc", "ood_f1", "macro_uav_ood_f1"])),
            ("experiments", rows or NA),
        ]
    )


def _quality_label(value: Any, *, high: float, medium: float, inverse: bool = False) -> str:
    numeric = _to_number(value)
    if numeric is None:
        return "N/A"
    numeric = float(numeric)
    if inverse:
        if numeric <= high:
            return "较低"
        if numeric <= medium:
            return "中等"
        return "偏高"
    if numeric >= high:
        return "较高"
    if numeric >= medium:
        return "中等"
    return "偏低"


def _known_attack_sentence(config: Mapping[str, Any], known: Mapping[str, Any]) -> str:
    model_name = _pretty_model_name(config.get("model_name"))
    micro_f1 = known.get("micro_f1", NA)
    macro_f1 = known.get("macro_f1", NA)
    map_score = known.get("mAP", NA)
    if _is_missing(micro_f1):
        return "已知攻击检测的关键指标缺失，当前汇总仅能输出 N/A。"
    quality = _quality_label(micro_f1, high=0.85, medium=0.70)
    sentence = (
        f"{model_name} 模型在已知攻击检测中取得了{quality}的 Micro-F1（{_display_value(micro_f1)}）"
        f"，Macro-F1 为 {_display_value(macro_f1)}，mAP 为 {_display_value(map_score)}，"
        "说明模型能够较好地识别窗口级已知攻击。"
    )
    return sentence


def _ood_sentence(ood: Mapping[str, Any]) -> str:
    auroc = ood.get("auroc", NA)
    fpr95 = ood.get("fpr95", NA)
    threshold_mode = ood.get("threshold_mode", NA)
    dominant_source = ood.get("dominant_ood_source", NA)
    if _is_missing(auroc):
        return "OOD 未知攻击检测指标不足，AUROC/FPR95 暂时无法给出有效结论。"
    auroc_quality = _quality_label(auroc, high=0.85, medium=0.70)
    fpr_quality = _quality_label(fpr95, high=0.10, medium=0.25, inverse=True)
    source_note = ""
    if not _is_missing(dominant_source):
        dominant_text = _display_value(dominant_source)
        if dominant_text == "energy":
            source_note = " 主导 OOD 判别分数分支为 `energy`，这里指 OOD energy score 分量，不代表无人机物理能量消耗。"
        else:
            source_note = f" 主导 OOD 判别分数分支为 `{dominant_text}`。"
    return (
        f"OOD 检测的 AUROC 为 {_display_value(auroc)}（{auroc_quality}），FPR95 为 {_display_value(fpr95)}（{fpr_quality}），"
        f"阈值模式为 `{_display_value(threshold_mode)}`，反映了模型对未知攻击的拒识能力。{source_note}"
    )


def _single_mixed_sentence(single_mixed: Mapping[str, Any]) -> str:
    single_attack = single_mixed.get("single_attack", NA)
    mixed_attack = single_mixed.get("mixed_attack", NA)
    if _is_missing(single_attack) and _is_missing(mixed_attack):
        return "当前报告中没有单独攻击与混合攻击的专门分项结果，相关指标统一记为 N/A。"

    single_micro = None
    mixed_micro = None
    if isinstance(single_attack, Mapping):
        single_micro = _to_number(_first_non_missing(single_attack.get("micro_f1"), single_attack.get("id_micro_f1")))
    if isinstance(mixed_attack, Mapping):
        mixed_micro = _to_number(_first_non_missing(mixed_attack.get("micro_f1"), mixed_attack.get("id_micro_f1")))
    if single_micro is not None and mixed_micro is not None:
        if single_micro >= mixed_micro:
            return (
                f"单独攻击场景的 Micro-F1（{_display_value(single_micro)}）高于或接近混合攻击场景（{_display_value(mixed_micro)}），"
                "说明 mixed windows 会带来更强的判别难度。"
            )
        return (
            f"混合攻击场景的 Micro-F1（{_display_value(mixed_micro)}）高于单独攻击场景（{_display_value(single_micro)}），"
            "表明当前配置对复合攻击模式具备一定鲁棒性。"
        )
    return "单独攻击与混合攻击结果已导出；若需更细粒度结论，可结合各攻击类型的分项指标继续分析。"


def _simulation_sentence(simulation: Mapping[str, Any]) -> str:
    record_count = simulation.get("record_count", NA)
    detection_count = simulation.get("detection_count", NA)
    alert_count = simulation.get("alert_count", NA)
    mission_success = simulation.get("mission_success", NA)
    if _is_missing(record_count) and _is_missing(detection_count):
        return "飞行仿真与在线检测汇总缺失，当前只能输出 N/A。"
    return (
        f"本次仿真共生成 {_display_value(record_count)} 条记录、{_display_value(detection_count)} 个检测窗口和 {_display_value(alert_count)} 次告警，"
        f"任务完成状态为 `{_display_value(mission_success)}`。"
    )


def _energy_sentence(config: Mapping[str, Any], energy: Mapping[str, Any]) -> str:
    model_name = _pretty_model_name(config.get("model_name"))
    avg_latency = energy.get("average_inference_latency_ms", NA)
    ids_ratio = energy.get("ids_energy_ratio", NA)
    extra_power = energy.get("transformer_extra_power_w", NA)
    latency_note = "平均推理时延为 N/A。"
    if not _is_missing(avg_latency):
        latency_quality = _quality_label(avg_latency, high=5.0, medium=20.0, inverse=True)
        latency_note = f"平均推理时延为 {_display_value(avg_latency)} ms（{latency_quality}）。"

    ratio_note = "IDS 能耗占比暂缺。"
    if not _is_missing(ids_ratio):
        ratio_note = (
            f"IDS 计算能耗占总能耗比例为 {_display_value(ids_ratio)}。"
            if _to_number(ids_ratio) is None
            else f"IDS 计算能耗占总能耗比例为 {_format_float(float(_to_number(ids_ratio)) * 100.0)}%。"
        )
    power_note = ""
    if not _is_missing(extra_power):
        power_note = f" 估计的 Transformer 额外功耗约为 {_display_value(extra_power)} W。"
    return (
        f"{model_name} 的部署开销汇总表明，{latency_note} {ratio_note}{power_note} "
        "这里的能耗仅来自仿真/基准报告中的物理能耗指标，不使用 OOD `energy` 分数替代实际能量消耗。"
    )


def _ablation_sentence(ablation: Mapping[str, Any]) -> str:
    best_known = ablation.get("best_known_attack_experiment", {})
    best_ood = ablation.get("best_ood_experiment", {})
    if not isinstance(best_known, Mapping) or _is_missing(best_known.get("experiment_name")):
        return "当前没有可用的消融实验结果，因此无法给出结构对比结论。"
    parts = [
        f"已知攻击指标最优的实验为 `{_display_value(best_known.get('experiment_name'))}`（{_display_value(best_known.get('metric_name'))}={_display_value(best_known.get('metric_value'))}）。"
    ]
    if isinstance(best_ood, Mapping) and not _is_missing(best_ood.get("experiment_name")):
        parts.append(
            f"OOD 结果最优的实验为 `{_display_value(best_ood.get('experiment_name'))}`（{_display_value(best_ood.get('metric_name'))}={_display_value(best_ood.get('metric_value'))}）。"
        )
    return " ".join(parts)


def _overall_sentence(
    config: Mapping[str, Any],
    known: Mapping[str, Any],
    ood: Mapping[str, Any],
    energy: Mapping[str, Any],
    ablation: Mapping[str, Any],
) -> str:
    parts = [
        _known_attack_sentence(config, known),
        _ood_sentence(ood),
        _energy_sentence(config, energy),
    ]
    ablation_text = _ablation_sentence(ablation)
    if ablation_text:
        parts.append(ablation_text)
    return " ".join(part.strip() for part in parts if part.strip())


def build_metric_summary(bundle_or_reports_dir: dict[str, Any] | str | Path) -> dict[str, Any]:
    bundle = (
        bundle_or_reports_dir
        if isinstance(bundle_or_reports_dir, Mapping) and "reports" in bundle_or_reports_dir
        else load_report_bundle(bundle_or_reports_dir)
    )
    experiment_config = _extract_experiment_config(bundle)
    known_attack = _extract_known_attack_metrics(bundle)
    ood_metrics = _extract_ood_metrics(bundle)
    per_uav_rows = _build_per_uav_rows(bundle)
    dataset_mapping = _extract_dataset_uav_mapping(bundle, per_uav_rows)
    per_dataset_metrics = _build_per_dataset_rows(dataset_mapping, per_uav_rows)
    per_source_type_metrics = _build_per_source_type_rows(per_dataset_metrics)
    single_mixed = _extract_single_mixed_results(bundle)
    simulation = _extract_simulation_summary(bundle)
    energy = _extract_energy_metrics(bundle)
    ablation = _extract_ablation_summary(bundle)

    highlights = {
        "dataset_mapping": _dataset_mapping_sentence(dataset_mapping),
        "known_attack": _known_attack_sentence(experiment_config, known_attack),
        "ood": _ood_sentence(ood_metrics),
        "single_mixed": _single_mixed_sentence(single_mixed),
        "per_uav_present_class": _present_class_macro_sentence(experiment_config),
        "simulation": _simulation_sentence(simulation),
        "energy": _energy_sentence(experiment_config, energy),
        "ablation": _ablation_sentence(ablation),
    }
    overall_conclusion = _overall_sentence(experiment_config, known_attack, ood_metrics, energy, ablation)
    highlights["overall"] = overall_conclusion

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "reports_dir": bundle["reports_dir"],
        "source_files": bundle["sources"],
        "related_files": bundle["related_sources"],
        "experiment_config": experiment_config,
        "dataset_uav_mapping": dataset_mapping or NA,
        "per_dataset_metrics": per_dataset_metrics or NA,
        "per_source_type_metrics": per_source_type_metrics or NA,
        "known_attack_detection": known_attack,
        "ood_unknown_attack_detection": ood_metrics,
        "single_and_mixed_attack_results": single_mixed,
        "multi_uav_group_detection": {
            "rows": per_uav_rows or NA,
            "global_alert_rate": _pick_first([bundle["reports"].get("group_detection_summary", {})], ["global.alert_rate"]),
            "cross_uav_generalization": simulation.get("cross_uav_generalization", NA),
        },
        "simulation_and_online_detection": simulation,
        "transformer_inference_energy": energy,
        "ablation_conclusion": ablation,
        "natural_language_highlights": highlights,
        "overall_conclusion": overall_conclusion,
    }


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    if not rows:
        return "| Value |\n| --- |\n| N/A |\n"
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_display_value(item) for item in row) + " |")
    return "\n".join(lines) + "\n"


def _kv_markdown(metrics: Mapping[str, Any], ordered_keys: Sequence[str]) -> str:
    rows = [(key, metrics.get(key, NA)) for key in ordered_keys]
    return _markdown_table(["指标", "数值"], rows)


def _latex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in text)


def _render_latex_metric_table(caption: str, label: str, rows: Sequence[tuple[str, Any]]) -> str:
    lines = [
        "% Requires \\usepackage{booktabs}",
        r"\begin{table}[htbp]",
        r"\centering",
        rf"\caption{{{_latex_escape(caption)}}}",
        rf"\label{{{_latex_escape(label)}}}",
        r"\begin{tabular}{lp{0.68\linewidth}}",
        r"\toprule",
        r"Metric & Value \\",
        r"\midrule",
    ]
    for metric_name, value in rows:
        lines.append(f"{_latex_escape(metric_name)} & {_latex_escape(_display_value(value))} \\\\")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
    )
    return "\n".join(lines) + "\n"


def _render_latex_grid_table(
    caption: str,
    label: str,
    headers: Sequence[str],
    rows: Sequence[Sequence[Any]],
    *,
    column_spec: str | None = None,
    resize_to_width: bool = False,
) -> str:
    normalized_rows = list(rows)
    if not normalized_rows:
        normalized_rows = [[NA for _ in headers]]
    spec = column_spec or ("l" * len(headers))
    requirement_note = "% Requires \\usepackage{booktabs}"
    if resize_to_width:
        requirement_note += " and \\usepackage{graphicx}"
    lines = [
        requirement_note,
        r"\begin{table}[htbp]",
        r"\centering",
        rf"\caption{{{_latex_escape(caption)}}}",
        rf"\label{{{_latex_escape(label)}}}",
    ]
    if resize_to_width:
        lines.append(r"\resizebox{\linewidth}{!}{%")
    lines.extend(
        [
            rf"\begin{{tabular}}{{{spec}}}",
            r"\toprule",
            " & ".join(_latex_escape(str(header)) for header in headers) + r" \\",
            r"\midrule",
        ]
    )
    for row in normalized_rows:
        lines.append(" & ".join(_latex_escape(_display_value(item)) for item in row) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    if resize_to_width:
        lines.append(r"}")
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


def render_markdown(summary: Mapping[str, Any]) -> str:
    config = summary["experiment_config"]
    known = summary["known_attack_detection"]
    ood = summary["ood_unknown_attack_detection"]
    single_mixed = summary["single_and_mixed_attack_results"]
    multi_uav = summary["multi_uav_group_detection"]
    simulation = summary["simulation_and_online_detection"]
    energy = summary["transformer_inference_energy"]
    ablation = summary["ablation_conclusion"]

    dataset_rows = summary["dataset_uav_mapping"] if isinstance(summary["dataset_uav_mapping"], list) else []
    per_dataset_rows = summary["per_dataset_metrics"] if isinstance(summary.get("per_dataset_metrics"), list) else []
    per_source_type_rows = (
        summary["per_source_type_metrics"] if isinstance(summary.get("per_source_type_metrics"), list) else []
    )
    per_uav_rows = multi_uav.get("rows", []) if isinstance(multi_uav, Mapping) and isinstance(multi_uav.get("rows"), list) else []
    failed_experiments = ablation.get("failed_experiments", NA) if isinstance(ablation, Mapping) else NA

    config_table = _kv_markdown(
        config,
        [
            "model_name",
            "graph_enabled",
            "graph_variant",
            "ood_fusion",
            "normalization_mode",
            "ood_threshold_mode",
            "group_col",
            "window_mode",
            "window_size",
            "stride",
            "device",
            "num_windows",
            "seed",
        ],
    )
    known_table = _kv_markdown(
        known,
        [
            "micro_f1",
            "macro_f1",
            "mAP",
            "subset_accuracy",
            "hamming_loss",
            "per_class_precision",
            "per_class_recall",
            "per_class_f1",
        ],
    )
    ood_table = _kv_markdown(
        ood,
        [
            "auroc",
            "auprc",
            "fpr95",
            "ood_f1",
            "ood_precision",
            "ood_recall",
            "best_threshold",
            "threshold_mode",
            "dominant_ood_source",
        ],
    )
    simulation_table = _kv_markdown(
        simulation,
        [
            "scenario_name",
            "uav_ids",
            "record_count",
            "detection_count",
            "attack_record_count",
            "alert_count",
            "false_alert_count",
            "response_count",
            "average_alert_delay",
            "mission_success",
            "global_alert_rate",
            "mean_ood_score",
            "max_ood_score",
        ],
    )
    energy_table = _kv_markdown(energy, ENERGY_COLUMNS)

    dataset_table = _markdown_table(
        DATASET_MAPPING_COLUMNS,
        [[row.get(column, NA) for column in DATASET_MAPPING_COLUMNS] for row in dataset_rows],
    )
    per_uav_table = _markdown_table(
        PER_UAV_COLUMNS,
        [[row.get(column, NA) for column in PER_UAV_COLUMNS] for row in per_uav_rows],
    )
    per_dataset_table = _markdown_table(
        PER_DATASET_COLUMNS,
        [[row.get(column, NA) for column in PER_DATASET_COLUMNS] for row in per_dataset_rows],
    )
    per_source_type_table = _markdown_table(
        PER_SOURCE_TYPE_COLUMNS,
        [[row.get(column, NA) for column in PER_SOURCE_TYPE_COLUMNS] for row in per_source_type_rows],
    )
    ablation_table = _kv_markdown(
        ablation if isinstance(ablation, Mapping) else {},
        [
            "experiment_count",
            "successful_experiment_count",
            "failed_experiments",
            "best_known_attack_experiment",
            "best_ood_experiment",
        ],
    )

    lines = [
        "# UAV IDS 文字版指标总结",
        "",
        "## 实验配置概述",
        "",
        config_table.rstrip(),
        "",
        summary["natural_language_highlights"]["overall"],
        "",
        "## 数据集与 UAV 映射",
        "",
        dataset_table.rstrip(),
        "",
        summary["natural_language_highlights"].get("dataset_mapping", ""),
        "",
        "## 各数据集指标汇总",
        "",
        per_dataset_table.rstrip(),
        "",
        "## 各 source_type 指标汇总",
        "",
        per_source_type_table.rstrip(),
        "",
        "## 已知攻击检测结果",
        "",
        known_table.rstrip(),
        "",
        summary["natural_language_highlights"]["known_attack"],
        "",
        "## OOD 未知攻击检测结果",
        "",
        ood_table.rstrip(),
        "",
        summary["natural_language_highlights"]["ood"],
        "",
        "## 单独攻击与混合攻击结果",
        "",
        _kv_markdown(single_mixed, ["single_attack", "mixed_attack", "mix_window_stats"]).rstrip(),
        "",
        summary["natural_language_highlights"]["single_mixed"],
        "",
        "## 多 UAV 分组检测结果",
        "",
        per_uav_table.rstrip(),
        "",
        summary["natural_language_highlights"].get("per_uav_present_class", ""),
        "",
        "## 飞行仿真与在线检测结果",
        "",
        simulation_table.rstrip(),
        "",
        summary["natural_language_highlights"]["simulation"],
        "",
        "## Transformer 推理开销与能耗结果",
        "",
        energy_table.rstrip(),
        "",
        summary["natural_language_highlights"]["energy"],
        "",
        "## 消融实验结论",
        "",
        ablation_table.rstrip(),
        "",
        summary["natural_language_highlights"]["ablation"],
        "",
        "## 总体结论",
        "",
        summary["overall_conclusion"],
        "",
    ]
    if not _is_missing(failed_experiments):
        lines.extend(["失败实验列表：" + _display_value(failed_experiments), ""])
    return "\n".join(lines)


def render_text(summary: Mapping[str, Any]) -> str:
    def _render_mapping(title: str, metrics: Mapping[str, Any], keys: Sequence[str]) -> list[str]:
        lines = [title, "-" * len(title)]
        for key in keys:
            lines.append(f"- {key}: {_display_value(metrics.get(key, NA))}")
        lines.append("")
        return lines

    dataset_rows = summary["dataset_uav_mapping"] if isinstance(summary["dataset_uav_mapping"], list) else []
    per_dataset_rows = summary["per_dataset_metrics"] if isinstance(summary.get("per_dataset_metrics"), list) else []
    per_source_type_rows = (
        summary["per_source_type_metrics"] if isinstance(summary.get("per_source_type_metrics"), list) else []
    )
    per_uav_rows = (
        summary["multi_uav_group_detection"].get("rows", [])
        if isinstance(summary["multi_uav_group_detection"], Mapping)
        and isinstance(summary["multi_uav_group_detection"].get("rows"), list)
        else []
    )

    lines = ["UAV IDS Metrics Summary", ""]
    lines.extend(
        _render_mapping(
            "实验配置概述",
            summary["experiment_config"],
            [
                "model_name",
                "graph_enabled",
                "graph_variant",
                "ood_fusion",
                "normalization_mode",
                "ood_threshold_mode",
                "group_col",
                "window_mode",
                "window_size",
                "stride",
                "device",
                "num_windows",
                "seed",
            ],
        )
    )
    lines.append(summary["natural_language_highlights"]["overall"])
    lines.append("")
    lines.append("数据集与 UAV 映射")
    lines.append("--------------")
    if dataset_rows:
        for row in dataset_rows:
            lines.append(
                " | ".join(f"{column}={_display_value(row.get(column, NA))}" for column in DATASET_MAPPING_COLUMNS)
            )
    else:
        lines.append(f"- {NA}")
    lines.append("")
    lines.append(summary["natural_language_highlights"].get("dataset_mapping", ""))
    lines.append("")
    lines.append("各数据集指标汇总")
    lines.append("--------------")
    if per_dataset_rows:
        for row in per_dataset_rows:
            lines.append(" | ".join(f"{column}={_display_value(row.get(column, NA))}" for column in PER_DATASET_COLUMNS))
    else:
        lines.append(NA)
    lines.append("")
    lines.append("各 source_type 指标汇总")
    lines.append("-------------------")
    if per_source_type_rows:
        for row in per_source_type_rows:
            lines.append(
                " | ".join(f"{column}={_display_value(row.get(column, NA))}" for column in PER_SOURCE_TYPE_COLUMNS)
            )
    else:
        lines.append(NA)
    lines.append("")
    lines.extend(
        _render_mapping(
            "已知攻击检测结果",
            summary["known_attack_detection"],
            [
                "micro_f1",
                "macro_f1",
                "mAP",
                "subset_accuracy",
                "hamming_loss",
                "per_class_precision",
                "per_class_recall",
                "per_class_f1",
            ],
        )
    )
    lines.append(summary["natural_language_highlights"]["known_attack"])
    lines.append("")
    lines.extend(
        _render_mapping(
            "OOD 未知攻击检测结果",
            summary["ood_unknown_attack_detection"],
            [
                "auroc",
                "auprc",
                "fpr95",
                "ood_f1",
                "ood_precision",
                "ood_recall",
                "best_threshold",
                "threshold_mode",
                "dominant_ood_source",
            ],
        )
    )
    lines.append(summary["natural_language_highlights"]["ood"])
    lines.append("")
    lines.extend(
        _render_mapping(
            "单独攻击与混合攻击结果",
            summary["single_and_mixed_attack_results"],
            ["single_attack", "mixed_attack", "mix_window_stats"],
        )
    )
    lines.append(summary["natural_language_highlights"]["single_mixed"])
    lines.append("")
    lines.append("多 UAV 分组检测结果")
    lines.append("----------------")
    if per_uav_rows:
        for row in per_uav_rows:
            lines.append(
                " | ".join(f"{column}={_display_value(row.get(column, NA))}" for column in PER_UAV_COLUMNS)
            )
    else:
        lines.append(NA)
    lines.append("")
    lines.append(summary["natural_language_highlights"].get("per_uav_present_class", ""))
    lines.append("")
    lines.extend(
        _render_mapping(
            "飞行仿真与在线检测结果",
            summary["simulation_and_online_detection"],
            [
                "scenario_name",
                "uav_ids",
                "record_count",
                "detection_count",
                "attack_record_count",
                "alert_count",
                "false_alert_count",
                "response_count",
                "average_alert_delay",
                "mission_success",
                "global_alert_rate",
                "mean_ood_score",
                "max_ood_score",
            ],
        )
    )
    lines.append(summary["natural_language_highlights"]["simulation"])
    lines.append("")
    lines.extend(
        _render_mapping(
            "Transformer 推理开销与能耗结果",
            summary["transformer_inference_energy"],
            ENERGY_COLUMNS,
        )
    )
    lines.append(summary["natural_language_highlights"]["energy"])
    lines.append("")
    lines.extend(
        _render_mapping(
            "消融实验结论",
            summary["ablation_conclusion"],
            [
                "experiment_count",
                "successful_experiment_count",
                "failed_experiments",
                "best_known_attack_experiment",
                "best_ood_experiment",
            ],
        )
    )
    lines.append(summary["natural_language_highlights"]["ablation"])
    lines.append("")
    lines.append("总体结论")
    lines.append("--------")
    lines.append(summary["overall_conclusion"])
    lines.append("")
    return "\n".join(lines)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    output_rows = list(rows)
    if not output_rows:
        output_rows = [{column: NA for column in columns}]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        writer.writeheader()
        for row in output_rows:
            writer.writerow({column: _display_value(row.get(column, NA)) for column in columns})


def export_metric_summary(reports_dir: str | Path) -> dict[str, Any]:
    reports_path = Path(reports_dir)
    reports_path.mkdir(parents=True, exist_ok=True)
    summary = build_metric_summary(reports_path)

    known_rows = [
        ("micro_f1", summary["known_attack_detection"].get("micro_f1", NA)),
        ("macro_f1", summary["known_attack_detection"].get("macro_f1", NA)),
        ("mAP", summary["known_attack_detection"].get("mAP", NA)),
        ("subset_accuracy", summary["known_attack_detection"].get("subset_accuracy", NA)),
        ("hamming_loss", summary["known_attack_detection"].get("hamming_loss", NA)),
        ("per_class_precision", summary["known_attack_detection"].get("per_class_precision", NA)),
        ("per_class_recall", summary["known_attack_detection"].get("per_class_recall", NA)),
        ("per_class_f1", summary["known_attack_detection"].get("per_class_f1", NA)),
    ]
    ood_rows = [
        ("auroc", summary["ood_unknown_attack_detection"].get("auroc", NA)),
        ("auprc", summary["ood_unknown_attack_detection"].get("auprc", NA)),
        ("fpr95", summary["ood_unknown_attack_detection"].get("fpr95", NA)),
        ("ood_f1", summary["ood_unknown_attack_detection"].get("ood_f1", NA)),
        ("ood_precision", summary["ood_unknown_attack_detection"].get("ood_precision", NA)),
        ("ood_recall", summary["ood_unknown_attack_detection"].get("ood_recall", NA)),
        ("best_threshold", summary["ood_unknown_attack_detection"].get("best_threshold", NA)),
        ("threshold_mode", summary["ood_unknown_attack_detection"].get("threshold_mode", NA)),
        ("dominant_ood_source", summary["ood_unknown_attack_detection"].get("dominant_ood_source", NA)),
    ]
    energy_rows = [(column, summary["transformer_inference_energy"].get(column, NA)) for column in ENERGY_COLUMNS]
    dataset_mapping_rows = summary["dataset_uav_mapping"] if isinstance(summary["dataset_uav_mapping"], list) else []

    markdown_text = render_markdown(summary)
    plain_text = render_text(summary)
    dataset_mapping_tex = _render_latex_grid_table(
        "Dataset-to-UAV mapping used in textual summary",
        "tab:dataset_uav_mapping",
        LATEX_DATASET_MAPPING_COLUMNS,
        [[row.get(column, NA) for column in LATEX_DATASET_MAPPING_COLUMNS] for row in dataset_mapping_rows],
        column_spec="llllp{0.42\\linewidth}",
        resize_to_width=True,
    )
    known_table_tex = _render_latex_metric_table("Known attack detection metrics", "tab:known_attack_metrics", known_rows)
    ood_table_tex = _render_latex_metric_table("OOD unknown-attack metrics", "tab:ood_metrics", ood_rows)
    energy_table_tex = _render_latex_metric_table("Transformer inference and energy metrics", "tab:energy_metrics", energy_rows)
    combined_tex = (
        "\n".join(
            [
                dataset_mapping_tex.rstrip(),
                "",
                known_table_tex.rstrip(),
                "",
                ood_table_tex.rstrip(),
                "",
                energy_table_tex.rstrip(),
                "",
            ]
        )
        + "\n"
    )

    output_paths = {
        "metrics_summary_md": reports_path / "metrics_summary.md",
        "metrics_summary_txt": reports_path / "metrics_summary.txt",
        "metrics_summary_json": reports_path / "metrics_summary.json",
        "metrics_table_tex": reports_path / "metrics_table.tex",
        "dataset_mapping_table_tex": reports_path / "dataset_mapping_table.tex",
        "known_attack_metrics_table_tex": reports_path / "known_attack_metrics_table.tex",
        "ood_metrics_table_tex": reports_path / "ood_metrics_table.tex",
        "energy_metrics_table_tex": reports_path / "energy_metrics_table.tex",
        "per_uav_metrics_csv": reports_path / "per_uav_metrics.csv",
        "per_dataset_metrics_csv": reports_path / "per_dataset_metrics.csv",
        "per_source_type_metrics_csv": reports_path / "per_source_type_metrics.csv",
        "energy_summary_csv": reports_path / "energy_summary.csv",
    }

    output_paths["metrics_summary_md"].write_text(markdown_text, encoding="utf-8")
    output_paths["metrics_summary_txt"].write_text(plain_text, encoding="utf-8")
    output_paths["metrics_table_tex"].write_text(combined_tex, encoding="utf-8")
    output_paths["dataset_mapping_table_tex"].write_text(dataset_mapping_tex, encoding="utf-8")
    output_paths["known_attack_metrics_table_tex"].write_text(known_table_tex, encoding="utf-8")
    output_paths["ood_metrics_table_tex"].write_text(ood_table_tex, encoding="utf-8")
    output_paths["energy_metrics_table_tex"].write_text(energy_table_tex, encoding="utf-8")

    per_uav_rows = summary["multi_uav_group_detection"].get("rows", []) if isinstance(summary["multi_uav_group_detection"], Mapping) else []
    _write_csv(output_paths["per_uav_metrics_csv"], per_uav_rows if isinstance(per_uav_rows, list) else [], PER_UAV_COLUMNS)
    _write_csv(
        output_paths["per_dataset_metrics_csv"],
        summary["per_dataset_metrics"] if isinstance(summary.get("per_dataset_metrics"), list) else [],
        PER_DATASET_COLUMNS,
    )
    _write_csv(
        output_paths["per_source_type_metrics_csv"],
        summary["per_source_type_metrics"] if isinstance(summary.get("per_source_type_metrics"), list) else [],
        PER_SOURCE_TYPE_COLUMNS,
    )
    _write_csv(
        output_paths["energy_summary_csv"],
        [summary["transformer_inference_energy"]] if isinstance(summary["transformer_inference_energy"], Mapping) else [],
        ENERGY_COLUMNS,
    )

    manifest = {key: str(path) for key, path in output_paths.items()}
    json_payload = dict(summary)
    json_payload["generated_files"] = manifest
    save_json(json_payload, output_paths["metrics_summary_json"])

    return {
        "reports_dir": str(reports_path),
        "generated_files": manifest,
        "source_files": summary["source_files"],
    }

