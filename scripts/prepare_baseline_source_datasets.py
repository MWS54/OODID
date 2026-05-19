#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prepare_ecu_ioft import prepare_ecu_ioft_dataset
from prepare_isot_drone import FOLDER_LABEL_MAP as ISOT_FOLDER_LABEL_MAP
from prepare_multi_uav_hetero import prepare_multi_uav_hetero_dataset
from prepare_uavids import prepare_uavids_dataset
from prepare_uavs_normal_cyberattacks import LABEL_MAP as UAV04_LABEL_MAP
from prepare_unsw_nb15 import prepare_unsw_nb15_dataset
from ucs_oodid.prepare_common import (
    build_prepared_frame,
    build_processing_summary,
    normalize_text_token,
    print_processing_summary,
    resolve_base_candidates,
    resolve_output_path,
    write_prepared_csv,
)


@dataclass(frozen=True)
class DocumentedSource:
    uav_id: str
    dataset_name: str
    source_type: str
    raw_path: str
    prepare_mode: str
    output_name: str


DOCUMENTED_SOURCES: tuple[DocumentedSource, ...] = (
    DocumentedSource("uav_01", "uav_ndd", "uav", r"data\UAV-NDD.csv", "missing_raw", "uav_ndd_uav01_baseline.csv"),
    DocumentedSource("uav_02", "gcs_to_uav_updated", "uav", r"data\GCS-to-UAV-Updated.csv", "missing_raw", "gcs_to_uav_updated_uav02_baseline.csv"),
    DocumentedSource("uav_03", "isot_drone", "uav", r"..\ISOT Drone Dataset\Dataset\new_feature_csv", "isot_light", "isot_drone_uav03_baseline.csv"),
    DocumentedSource("uav_04", "uavs_normal_cyberattacks", "uav", r"..\UAVs-Dataset-Under-Normal-and-Cyberattacks-main\Dataset_T-ITS.csv", "uav04_light", "uavs_normal_cyberattacks_uav04_baseline.csv"),
    DocumentedSource("uav_05", "unsw_nb15", "external_non_uav", r"..\NUSW\UNSW_NB15_training-set.csv", "unsw_light", "unsw_nb15_uav05_baseline.csv"),
    DocumentedSource("uav_06", "ecu_ioft", "uav_iot_wifi", r"..\ECU-IoFT-main\dataset\ECU-IoFT-Dataset.csv", "ecu_light", "ecu_ioft_uav06_baseline.csv"),
    DocumentedSource("uav_07", "uavids", "uav", r"..\UAVIDS\UAVIDS-2025.csv", "uavids_light", "uavids_uav07_baseline.csv"),
)


def _ordered_union(items: list[str], new_items: list[str]) -> list[str]:
    seen = set(items)
    for item in new_items:
        key = str(item)
        if key not in seen:
            items.append(key)
            seen.add(key)
    return items


def _resolve_existing_path(base_candidates: list[Path], value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    for base in base_candidates:
        candidate = (base / path).resolve()
        if candidate.exists():
            return candidate
    fallback_base = base_candidates[1] if len(base_candidates) > 1 else base_candidates[0]
    return (fallback_base / path).resolve()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _normalise_uav04_label(value: object) -> str:
    text = str(value).strip()
    if text in UAV04_LABEL_MAP:
        return UAV04_LABEL_MAP[text]
    return normalize_text_token(text)


def _normalise_passthrough_label(value: object) -> str:
    return normalize_text_token(value)


def prepare_uav04_baseline_dataset(
    input_csv: str,
    output: str,
    *,
    uav_id: str,
    dataset_name: str,
    source_type: str,
) -> tuple[Path, dict[str, Any]]:
    script_dir, project_root, base_candidates = resolve_base_candidates(__file__)
    del script_dir
    input_path = _resolve_existing_path(base_candidates, input_csv)
    output_path = resolve_output_path(project_root, output)
    if output_path is None:
        raise ValueError("output must be provided.")

    raw_df = pd.read_csv(input_path, low_memory=False)
    input_rows = int(len(raw_df))
    source_class = raw_df["class"].fillna("").astype(str).str.strip()
    valid_mask = source_class.isin(UAV04_LABEL_MAP)
    cleaned = raw_df.loc[valid_mask].copy()
    cleaned["label"] = source_class.loc[valid_mask].map(UAV04_LABEL_MAP)

    prepared = build_prepared_frame(
        cleaned,
        label_column="label",
        timestamp_column="timestamp_c" if "timestamp_c" in cleaned.columns else None,
        split_column=None,
        uav_id=uav_id,
        dataset_name=dataset_name,
        source_type=source_type,
        label_normalizer=_normalise_uav04_label,
    )
    summary = build_processing_summary(
        prepared,
        output_path=output_path,
        label_column="class",
        timestamp_column="timestamp_c" if "timestamp_c" in cleaned.columns else None,
        split_column=None,
        dataset_name=dataset_name,
        source_type=source_type,
        input_files=[input_path],
        extra_notes=[
            "Baseline-light preparation keeps raw rows and skips duplicate removal, feature remapping, and label-chunk interleaving.",
            f"Rows with unsupported class labels were discarded: {input_rows - len(cleaned)}",
        ],
    )
    summary["input_rows"] = input_rows
    summary["valid_rows_after_label_filter"] = int(len(cleaned))
    summary["dropped_rows_due_to_unknown_label"] = int(input_rows - len(cleaned))
    summary["label_map"] = dict(UAV04_LABEL_MAP)
    write_prepared_csv(prepared, output_path)
    print_processing_summary("prepare_uav04_baseline", summary)
    return output_path, summary


def prepare_isot_baseline_dataset(
    input_root: str,
    output: str,
    *,
    uav_id: str,
    dataset_name: str,
    source_type: str,
) -> tuple[Path, dict[str, Any]]:
    script_dir, project_root, base_candidates = resolve_base_candidates(__file__)
    del script_dir
    input_path = _resolve_existing_path(base_candidates, input_root)
    output_path = resolve_output_path(project_root, output)
    if output_path is None:
        raise ValueError("output must be provided.")

    csv_specs: list[tuple[Path, str, str]] = []
    raw_header_union: list[str] = []
    folder_counts: dict[str, int] = {}
    for folder in sorted([path for path in input_path.iterdir() if path.is_dir()], key=lambda p: p.name.lower()):
        label = ISOT_FOLDER_LABEL_MAP.get(folder.name)
        if label is None:
            continue
        files = sorted(folder.glob("*.csv"), key=lambda p: p.name.lower())
        folder_counts[folder.name] = len(files)
        for csv_path in files:
            header = list(pd.read_csv(csv_path, nrows=0).columns)
            _ordered_union(raw_header_union, header)
            csv_specs.append((csv_path, folder.name, label))
    if not csv_specs:
        raise ValueError(f"No supported ISOT CSV files were found under {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    total_input_rows = 0
    total_output_rows = 0
    label_counts: dict[str, int] = {}
    first_chunk = True
    for csv_path, folder_name, label in csv_specs:
        frame = pd.read_csv(csv_path, low_memory=False)
        total_input_rows += int(len(frame))
        frame = frame.reindex(columns=raw_header_union, fill_value=pd.NA)
        frame["label"] = label
        frame["source_folder"] = folder_name
        frame["source_file"] = csv_path.name
        prepared = build_prepared_frame(
            frame,
            label_column="label",
            timestamp_column="ts" if "ts" in frame.columns else None,
            split_column=None,
            uav_id=uav_id,
            dataset_name=dataset_name,
            source_type=source_type,
            label_normalizer=_normalise_passthrough_label,
        )
        prepared.to_csv(output_path, index=False, mode="w" if first_chunk else "a", header=first_chunk)
        first_chunk = False
        total_output_rows += int(len(prepared))
        chunk_counts = prepared["label_normalized"].astype(str).value_counts().to_dict()
        for key, value in chunk_counts.items():
            label_counts[str(key)] = int(label_counts.get(str(key), 0)) + int(value)

    summary = {
        "dataset_name": dataset_name,
        "source_type": source_type,
        "input_root": str(input_path),
        "input_file_count": int(len(csv_specs)),
        "raw_files_by_folder": {str(key): int(value) for key, value in folder_counts.items()},
        "input_rows": int(total_input_rows),
        "output_rows": int(total_output_rows),
        "output_path": str(output_path),
        "label_map": dict(ISOT_FOLDER_LABEL_MAP),
        "label_counts": {str(key): int(value) for key, value in sorted(label_counts.items())},
        "notes": [
            "Baseline-light preparation keeps all supported ISOT rows and skips target-size balancing, oversampling, duplicate removal, ambiguous-pattern filtering, and label-chunk interleaving.",
            "Rows are concatenated from the original folder structure and split is left as 'all'.",
        ],
    }
    print_processing_summary("prepare_isot_baseline", summary)
    return output_path, summary


def _relative_to_project(project_root: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(project_root.resolve())).replace("/", "\\")
    except Exception:
        return str(path.resolve())


def _write_config(config_path: Path, entries: list[dict[str, str]]) -> None:
    lines = ["datasets:"]
    for entry in entries:
        lines.extend(
            [
                f"  - uav_id: {entry['uav_id']}",
                f"    dataset_name: {entry['dataset_name']}",
                f"    csv: {entry['csv']}",
                f"    source_type: {entry['source_type']}",
            ]
        )
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare documented raw/original dataset sources into baseline-light canonical CSVs without the heavy project-specific reconstruction pipeline."
    )
    parser.add_argument("--output_root", default=r"data\baseline_raw", help="Directory for per-source baseline-light CSVs and notes.")
    parser.add_argument(
        "--manifest_output",
        default=r"data\baseline_raw\documented_raw_dataset_manifest.json",
        help="JSON manifest summarizing documented raw sources, availability, and generated baseline outputs.",
    )
    parser.add_argument(
        "--config_output",
        default=r"configs\baseline_raw_datasets_available.yaml",
        help="YAML config listing the generated baseline-light source CSVs that are actually available on disk.",
    )
    parser.add_argument(
        "--merged_output",
        default=None,
        help="Optional merged heterogeneous baseline CSV built from the available baseline-light sources.",
    )
    parser.add_argument(
        "--merged_summary_json",
        default=None,
        help="Optional summary JSON path for --merged_output. Defaults beside the merged CSV when omitted.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    script_dir, project_root, base_candidates = resolve_base_candidates(__file__)
    del script_dir

    output_root = resolve_output_path(project_root, args.output_root)
    manifest_path = resolve_output_path(project_root, args.manifest_output)
    config_path = resolve_output_path(project_root, args.config_output)
    if output_root is None or manifest_path is None or config_path is None:
        raise ValueError("output_root, manifest_output, and config_output must be provided.")
    output_root.mkdir(parents=True, exist_ok=True)

    available_entries: list[dict[str, str]] = []
    manifest_records: list[dict[str, Any]] = []

    for source in DOCUMENTED_SOURCES:
        raw_path = _resolve_existing_path(base_candidates, source.raw_path)
        output_csv = output_root / source.output_name
        notes_json = output_root / f"{Path(source.output_name).stem}_notes.json"
        record: dict[str, Any] = {
            "uav_id": source.uav_id,
            "dataset_name": source.dataset_name,
            "source_type": source.source_type,
            "documented_raw_path": source.raw_path,
            "resolved_raw_path": str(raw_path),
            "raw_exists": bool(raw_path.exists()),
            "prepare_mode": source.prepare_mode,
            "baseline_output_csv": str(output_csv),
            "notes_json": str(notes_json),
        }
        if not raw_path.exists():
            record["status"] = "missing_raw_source"
            manifest_records.append(record)
            continue

        if source.prepare_mode == "ecu_light":
            _, summary = prepare_ecu_ioft_dataset(
                input_csv=str(raw_path),
                output=_relative_to_project(project_root, output_csv),
                uav_id=source.uav_id,
                dataset_name=source.dataset_name,
            )
        elif source.prepare_mode == "unsw_light":
            _, summary = prepare_unsw_nb15_dataset(
                input_path=str(raw_path),
                output=_relative_to_project(project_root, output_csv),
                uav_id=source.uav_id,
                dataset_name=source.dataset_name,
            )
        elif source.prepare_mode == "uavids_light":
            _, summary = prepare_uavids_dataset(
                input_csv=str(raw_path),
                output=_relative_to_project(project_root, output_csv),
                uav_id=source.uav_id,
                dataset_name=source.dataset_name,
            )
        elif source.prepare_mode == "uav04_light":
            _, summary = prepare_uav04_baseline_dataset(
                input_csv=str(raw_path),
                output=_relative_to_project(project_root, output_csv),
                uav_id=source.uav_id,
                dataset_name=source.dataset_name,
                source_type=source.source_type,
            )
        elif source.prepare_mode == "isot_light":
            _, summary = prepare_isot_baseline_dataset(
                input_root=str(raw_path),
                output=_relative_to_project(project_root, output_csv),
                uav_id=source.uav_id,
                dataset_name=source.dataset_name,
                source_type=source.source_type,
            )
        else:
            record["status"] = "unsupported_prepare_mode"
            manifest_records.append(record)
            continue

        _write_json(notes_json, summary)
        record["status"] = "prepared"
        record["summary"] = summary
        available_entries.append(
            {
                "uav_id": source.uav_id,
                "dataset_name": source.dataset_name,
                "csv": _relative_to_project(project_root, output_csv),
                "source_type": source.source_type,
            }
        )
        manifest_records.append(record)

    _write_config(config_path, available_entries)

    manifest_payload: dict[str, Any] = {
        "document_source": "README.md documented raw dataset paths plus UAV04_EXPERIMENT_SUMMARY.md",
        "generated_output_root": str(output_root),
        "generated_config": str(config_path),
        "documented_sources": manifest_records,
    }

    if args.merged_output:
        merged_output = resolve_output_path(project_root, args.merged_output)
        if merged_output is None:
            raise ValueError("merged_output could not be resolved.")
        merged_summary_json = resolve_output_path(project_root, args.merged_summary_json) if args.merged_summary_json else None
        _, merged_summary = prepare_multi_uav_hetero_dataset(
            output=_relative_to_project(project_root, merged_output),
            summary_json=_relative_to_project(project_root, merged_summary_json) if merged_summary_json is not None else None,
            config_path=_relative_to_project(project_root, config_path),
        )
        manifest_payload["merged_output"] = str(merged_output)
        manifest_payload["merged_summary"] = merged_summary

    _write_json(manifest_path, manifest_payload)
    print(json.dumps({"prepare_baseline_source_datasets": manifest_payload}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
