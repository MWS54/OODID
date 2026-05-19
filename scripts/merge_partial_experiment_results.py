#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import shutil
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_comparison_experiments as comparison_runner
import run_homogeneous_experiments as homogeneous_runner
from ucs_oodid.io import load_json, save_json


COMPARISON_METHOD_ORDER = [spec.name for spec in comparison_runner.METHOD_SPECS]
GROUP_STATS_COLUMNS = getattr(
    homogeneous_runner,
    "GROUP_STATS_COLUMNS",
    [
        "Group",
        "Records",
        "ID Records",
        "OOD Records",
        "Unique Labels",
        "ID Labels Present",
        "OOD Labels Present",
        "ID Classes Used",
        "OOD Classes Used",
        "Status",
        "Reason",
    ],
)


def _resolve_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return ROOT / path


def _deepcopy_payload(payload: Any) -> Any:
    return copy.deepcopy(payload)


def _ordered_union(*sequences: list[Any]) -> list[Any]:
    seen: set[str] = set()
    ordered: list[Any] = []
    for sequence in sequences:
        for item in sequence or []:
            key = str(item)
            if key in seen:
                continue
            seen.add(key)
            ordered.append(item)
    return ordered


def _choose_preferred_method(existing: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    existing_status = str((existing or {}).get("status", ""))
    incoming_status = str((incoming or {}).get("status", ""))
    if existing_status == "success" and incoming_status != "success":
        return existing
    if incoming_status == "success" and existing_status != "success":
        return incoming
    return incoming if incoming_status else existing


def _known_row_for_method(group: str | None, method_summary: dict[str, Any], *, include_group: bool) -> dict[str, Any]:
    metrics = dict(method_summary.get("known_detection", {}) or {})
    row = {
        "Method": str(method_summary.get("display_name", method_summary.get("method_name", ""))),
        "Micro-F1": metrics.get("micro_f1"),
        "Macro-F1": metrics.get("macro_f1"),
        "mAP": metrics.get("mAP"),
        "Hamming Loss": metrics.get("hamming_loss"),
        "Subset Acc.": metrics.get("subset_accuracy"),
    }
    if include_group:
        row = {"Group": str(group or ""), **row}
    return row


def _ood_row_for_method(group: str | None, method_summary: dict[str, Any], *, include_group: bool) -> dict[str, Any]:
    metrics = dict(method_summary.get("ood_detection", {}) or {})
    row = {
        "Method": str(method_summary.get("display_name", method_summary.get("method_name", ""))),
        "AUROC": metrics.get("auroc"),
        "AUPR-Out": metrics.get("aupr_out"),
        "FPR95": metrics.get("fpr95"),
        "Precision": metrics.get("precision"),
        "Recall": metrics.get("recall", metrics.get("tpr")),
        "OOD-F1": metrics.get("ood_f1"),
        "FPR@theta": metrics.get("fpr_at_threshold"),
    }
    if include_group:
        row = {"Group": str(group or ""), **row}
    return row


def _time_row_for_method(group: str | None, method_summary: dict[str, Any], *, include_group: bool) -> dict[str, Any]:
    timing = dict(method_summary.get("timing", {}) or {})
    row = {
        "Method": str(method_summary.get("display_name", method_summary.get("method_name", ""))),
        "Avg. Detection Time (ms/window)": timing.get("average_detection_time_ms"),
        "Throughput (windows/s)": timing.get("throughput_windows_per_s"),
        "Test Windows": timing.get("test_windows"),
    }
    if include_group:
        row = {"Group": str(group or ""), **row}
    return row


def _populate_method_summary_defaults(method_summary: dict[str, Any], spec) -> dict[str, Any]:
    if spec is None:
        return method_summary
    method_summary.setdefault("display_name", spec.display_name)
    method_summary.setdefault("category", spec.category)
    method_summary.setdefault("backend_name", spec.backend_name)
    method_summary.setdefault("heterogeneity_handling", comparison_runner._heterogeneity_handling_mode(spec))
    method_summary.setdefault(
        "heterogeneity_handling_description",
        comparison_runner._heterogeneity_handling_description(spec),
    )
    method_summary.setdefault("timing", {})
    return method_summary


def _copy_method_dir(src_dir: Path, dest_dir: Path) -> None:
    if not src_dir.exists() or not src_dir.is_dir():
        return
    shutil.copytree(src_dir, dest_dir, dirs_exist_ok=True)


def _build_group_comparison_summary(
    merged_group: dict[str, Any],
    group_output_dir: Path,
    method_names: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    methods_by_name = dict(merged_group.get("methods", {}) or {})
    method_summaries: list[dict[str, Any]] = []
    known_rows: list[dict[str, Any]] = []
    ood_rows: list[dict[str, Any]] = []
    time_rows: list[dict[str, Any]] = []
    failed_method_records: list[dict[str, Any]] = []
    baseline_rows: list[dict[str, Any]] = []

    for method_name in method_names:
        method_summary = methods_by_name.get(method_name)
        if method_summary is None:
            continue
        method_summary = _deepcopy_payload(method_summary)
        spec = comparison_runner.METHOD_BY_NAME.get(method_name)
        method_summary.setdefault("method_name", method_name)
        method_summary = _populate_method_summary_defaults(method_summary, spec)
        method_dir = group_output_dir / method_name
        method_summary["output_dir"] = str(method_dir)
        method_summary["eval_report"] = str(method_dir / "eval_report.json")
        method_summaries.append(method_summary)
        known_rows.append(_known_row_for_method(None, method_summary, include_group=False))
        ood_rows.append(_ood_row_for_method(None, method_summary, include_group=False))
        time_rows.append(_time_row_for_method(None, method_summary, include_group=False))
        if spec is not None:
            baseline_rows.append(comparison_runner._baseline_config_row(spec))
        if str(method_summary.get("status", "")) != "success":
            failed_method_records.append(
                {
                    "method_name": method_name,
                    "display_name": method_summary.get("display_name", method_name),
                    "status": method_summary.get("status", ""),
                    "note": method_summary.get("note", ""),
                }
            )

    comparison_summary = {
        "input": str(merged_group.get("temp_input", "")),
        "output_root": str(group_output_dir),
        "output_dir": str(group_output_dir),
        "paper_note": str(
            merged_group.get("paper_note", comparison_runner.PROXY_IMPLEMENTATION_NOTE)
            or comparison_runner.PROXY_IMPLEMENTATION_NOTE
        ),
        "selected_methods": [item["method_name"] for item in method_summaries],
        "successful_methods": [item["method_name"] for item in method_summaries if item.get("status") == "success"],
        "skipped_methods": [item["method_name"] for item in method_summaries if item.get("status") == "skipped"],
        "failed_methods": [item["method_name"] for item in method_summaries if item.get("status") == "failed"],
        "methods": method_summaries,
        "tables": {
            "known_detection_csv": str(group_output_dir / "known_detection_table.csv"),
            "known_detection_tex": str(group_output_dir / "known_detection_table.tex"),
            "ood_detection_csv": str(group_output_dir / "ood_detection_table.csv"),
            "ood_detection_tex": str(group_output_dir / "ood_detection_table.tex"),
            "detection_time_csv": str(group_output_dir / "detection_time_table.csv"),
            "detection_time_tex": str(group_output_dir / "detection_time_table.tex"),
            "baseline_config_csv": str(group_output_dir / "baseline_config_table.csv"),
            "baseline_config_tex": str(group_output_dir / "baseline_config_table.tex"),
        },
        "failed_methods_file": str(group_output_dir / "failed_methods.json"),
    }
    return time_rows, known_rows, ood_rows, baseline_rows, comparison_summary | {"failed_method_records": failed_method_records}


def merge_homogeneous(source_roots: list[Path], output_root: Path) -> None:
    summaries = [load_json(root / "homogeneous_summary.json") for root in source_roots]
    if not summaries:
        raise ValueError("No homogeneous summaries were provided.")

    output_root.mkdir(parents=True, exist_ok=True)
    groups = _ordered_union(*(summary.get("groups", []) for summary in summaries))
    resolved_label_configs: dict[str, Any] = {}
    skipped_by_group: dict[str, dict[str, Any]] = {}
    failed_by_group: dict[str, dict[str, Any]] = {}
    stats_by_group: dict[str, dict[str, Any]] = {}
    merged_groups: dict[str, dict[str, Any]] = {}

    for source_root, summary in zip(source_roots, summaries):
        for group, config in dict(summary.get("resolved_group_label_configs", {}) or {}).items():
            resolved_label_configs[str(group)] = _deepcopy_payload(config)
        for row in list(summary.get("group_dataset_stats", []) or []):
            group = str(row.get("Group", ""))
            if group:
                stats_by_group[group] = _deepcopy_payload(row)
        for record in list(summary.get("skipped_groups", []) or []):
            group = str(record.get("group", ""))
            if group and group not in skipped_by_group:
                skipped_by_group[group] = _deepcopy_payload(record)
        for record in list(summary.get("failed_groups", []) or []):
            group = str(record.get("group", ""))
            if group and group not in failed_by_group:
                failed_by_group[group] = _deepcopy_payload(record)

        for group_record in list(summary.get("group_results", []) or []):
            group = str(group_record.get("group", ""))
            if not group:
                continue
            safe_group = str(group_record.get("safe_group", group))
            dest_group_dir = output_root / f"group_{safe_group}"
            dest_group_dir.mkdir(parents=True, exist_ok=True)

            incoming = _deepcopy_payload(group_record)
            incoming["output_root"] = str(dest_group_dir)
            incoming["comparison_summary_path"] = str(dest_group_dir / "comparison_summary.json")
            incoming["known_detection_table"] = str(dest_group_dir / "known_detection_table.csv")
            incoming["ood_detection_table"] = str(dest_group_dir / "ood_detection_table.csv")

            source_group_dir = _resolve_path(group_record.get("output_root", source_root / f"group_{safe_group}"))
            for method_name in dict(group_record.get("methods", {}) or {}):
                _copy_method_dir(source_group_dir / method_name, dest_group_dir / method_name)

            existing = merged_groups.get(group)
            if existing is None:
                merged_groups[group] = incoming
                continue

            existing["selected_methods"] = _ordered_union(
                list(existing.get("selected_methods", []) or []),
                list(incoming.get("selected_methods", []) or []),
            )
            existing["successful_methods"] = _ordered_union(
                list(existing.get("successful_methods", []) or []),
                list(incoming.get("successful_methods", []) or []),
            )
            existing["skipped_methods"] = _ordered_union(
                list(existing.get("skipped_methods", []) or []),
                list(incoming.get("skipped_methods", []) or []),
            )
            existing["failed_methods"] = _ordered_union(
                list(existing.get("failed_methods", []) or []),
                list(incoming.get("failed_methods", []) or []),
            )
            existing_methods = dict(existing.get("methods", {}) or {})
            for method_name, method_summary in dict(incoming.get("methods", {}) or {}).items():
                if method_name in existing_methods:
                    existing_methods[method_name] = _choose_preferred_method(existing_methods[method_name], method_summary)
                else:
                    existing_methods[method_name] = _deepcopy_payload(method_summary)
            existing["methods"] = existing_methods
            selected_methods = _ordered_union(
                list(existing.get("selected_methods", []) or []),
                list(existing_methods.keys()),
            )
            existing["selected_methods"] = selected_methods
            existing["successful_methods"] = [
                method_name
                for method_name in selected_methods
                if str(dict(existing_methods.get(method_name, {}) or {}).get("status", "")) == "success"
            ]
            existing["skipped_methods"] = [
                method_name
                for method_name in selected_methods
                if str(dict(existing_methods.get(method_name, {}) or {}).get("status", "")) == "skipped"
            ]
            existing["failed_methods"] = [
                method_name
                for method_name in selected_methods
                if str(dict(existing_methods.get(method_name, {}) or {}).get("status", "")) == "failed"
            ]
            if not existing.get("split_source") and incoming.get("split_source"):
                existing["split_source"] = incoming.get("split_source")
            if not dict(existing.get("window_counts", {}) or {}) and dict(incoming.get("window_counts", {}) or {}):
                existing["window_counts"] = dict(incoming.get("window_counts", {}) or {})
            merged_groups[group] = existing

    method_order = [name for name in COMPARISON_METHOD_ORDER if any(name in dict(item.get("methods", {}) or {}) for item in merged_groups.values())]
    merged_group_results = [merged_groups[group] for group in groups if group in merged_groups]

    for group_record in merged_group_results:
        group = str(group_record.get("group", ""))
        safe_group = str(group_record.get("safe_group", group))
        group_output_dir = output_root / f"group_{safe_group}"
        time_rows, known_rows, ood_rows, baseline_rows, comparison_summary = _build_group_comparison_summary(
            group_record,
            group_output_dir,
            method_order,
        )
        comparison_runner._write_tables(
            time_rows,
            comparison_runner.TIME_COLUMNS,
            group_output_dir / "detection_time_table.csv",
            group_output_dir / "detection_time_table.tex",
            comparison_runner.TIME_CAPTION,
        )
        comparison_runner._write_tables(
            known_rows,
            comparison_runner.KNOWN_COLUMNS,
            group_output_dir / "known_detection_table.csv",
            group_output_dir / "known_detection_table.tex",
            comparison_runner.KNOWN_CAPTION,
        )
        comparison_runner._write_tables(
            ood_rows,
            comparison_runner.OOD_COLUMNS,
            group_output_dir / "ood_detection_table.csv",
            group_output_dir / "ood_detection_table.tex",
            comparison_runner.OOD_CAPTION,
        )
        comparison_runner._write_tables(
            baseline_rows,
            comparison_runner.BASELINE_CONFIG_COLUMNS,
            group_output_dir / "baseline_config_table.csv",
            group_output_dir / "baseline_config_table.tex",
            comparison_runner.BASELINE_CONFIG_CAPTION,
            round_numeric=False,
            aligns="lllllll",
        )
        save_json(comparison_summary["failed_method_records"], group_output_dir / "failed_methods.json")
        save_json(
            {key: value for key, value in comparison_summary.items() if key != "failed_method_records"},
            group_output_dir / "comparison_summary.json",
        )

        group_record["output_root"] = str(group_output_dir)
        group_record["comparison_summary_path"] = str(group_output_dir / "comparison_summary.json")
        group_record["known_detection_table"] = str(group_output_dir / "known_detection_table.csv")
        group_record["ood_detection_table"] = str(group_output_dir / "ood_detection_table.csv")
        group_record["detection_time_table"] = str(group_output_dir / "detection_time_table.csv")

    merged_known_rows: list[dict[str, Any]] = []
    merged_ood_rows: list[dict[str, Any]] = []
    merged_time_rows: list[dict[str, Any]] = []
    for group_record in merged_group_results:
        methods_by_name = dict(group_record.get("methods", {}) or {})
        for method_name in method_order:
            method_summary = methods_by_name.get(method_name)
            if method_summary is None:
                continue
            merged_known_rows.append(_known_row_for_method(group_record.get("group"), method_summary, include_group=True))
            merged_ood_rows.append(_ood_row_for_method(group_record.get("group"), method_summary, include_group=True))
            merged_time_rows.append(_time_row_for_method(group_record.get("group"), method_summary, include_group=True))

    successful_groups = [group for group in groups if group in merged_groups]
    skipped_groups = [skipped_by_group[group] for group in groups if group in skipped_by_group and group not in merged_groups]
    failed_groups = [failed_by_group[group] for group in groups if group in failed_by_group and group not in merged_groups]

    stats_rows: list[dict[str, Any]] = []
    for group in groups:
        row = _deepcopy_payload(stats_by_group.get(group, {"Group": group}))
        if group in merged_groups:
            row["Status"] = "success"
            row["Reason"] = ""
        elif group in skipped_by_group:
            row["Status"] = "skipped"
            row["Reason"] = str(skipped_by_group[group].get("skip_reason", ""))
        elif group in failed_by_group:
            row["Status"] = "failed"
            row["Reason"] = str(failed_by_group[group].get("error_message", ""))
        stats_rows.append(row)

    pd.DataFrame(stats_rows).reindex(columns=GROUP_STATS_COLUMNS).to_csv(output_root / "group_dataset_stats.csv", index=False)
    save_json(failed_groups, output_root / "failed_groups.json")
    homogeneous_runner._write_tables(
        merged_known_rows,
        homogeneous_runner.KNOWN_COLUMNS,
        output_root / "homogeneous_known_detection_table.csv",
        output_root / "homogeneous_known_detection_table.tex",
        homogeneous_runner.KNOWN_CAPTION,
    )
    homogeneous_runner._write_tables(
        merged_ood_rows,
        homogeneous_runner.OOD_COLUMNS,
        output_root / "homogeneous_ood_detection_table.csv",
        output_root / "homogeneous_ood_detection_table.tex",
        homogeneous_runner.OOD_CAPTION,
    )
    homogeneous_runner._write_tables(
        merged_time_rows,
        ["Group", *comparison_runner.TIME_COLUMNS],
        output_root / "homogeneous_detection_time_table.csv",
        output_root / "homogeneous_detection_time_table.tex",
        comparison_runner.TIME_CAPTION,
    )
    main_rows = homogeneous_runner._build_main_table_rows(merged_group_results)
    homogeneous_runner._write_main_table(
        main_rows,
        output_root / "homogeneous_main_table.csv",
        output_root / "homogeneous_main_table.tex",
    )

    merged_config = dict(summaries[0].get("config", {}) or {})
    merged_config["skip_sklearn"] = False
    merged_config["skip_neural"] = False
    merged_summary = {
        "input": str(summaries[0].get("input", "")),
        "output_root": str(output_root),
        "group_col": str(summaries[0].get("group_col", "")),
        "paper_note": str(
            summaries[0].get("paper_note", comparison_runner.PROXY_IMPLEMENTATION_NOTE)
            or comparison_runner.PROXY_IMPLEMENTATION_NOTE
        ),
        "groups": groups,
        "successful_groups": successful_groups,
        "skipped_groups": skipped_groups,
        "failed_groups": failed_groups,
        "resolved_group_label_configs": resolved_label_configs,
        "group_dataset_stats": stats_rows,
        "group_results": merged_group_results,
        "config": merged_config,
        "tables": {
            "group_dataset_stats_csv": str(output_root / "group_dataset_stats.csv"),
            "homogeneous_known_detection_csv": str(output_root / "homogeneous_known_detection_table.csv"),
            "homogeneous_known_detection_tex": str(output_root / "homogeneous_known_detection_table.tex"),
            "homogeneous_ood_detection_csv": str(output_root / "homogeneous_ood_detection_table.csv"),
            "homogeneous_ood_detection_tex": str(output_root / "homogeneous_ood_detection_table.tex"),
            "homogeneous_detection_time_csv": str(output_root / "homogeneous_detection_time_table.csv"),
            "homogeneous_detection_time_tex": str(output_root / "homogeneous_detection_time_table.tex"),
            "homogeneous_main_table_csv": str(output_root / "homogeneous_main_table.csv"),
            "homogeneous_main_table_tex": str(output_root / "homogeneous_main_table.tex"),
        },
        "failed_groups_file": str(output_root / "failed_groups.json"),
    }
    save_json(merged_summary, output_root / "homogeneous_summary.json")


def merge_comparison(source_roots: list[Path], output_root: Path) -> None:
    summaries = [load_json(root / "comparison_summary.json") for root in source_roots]
    if not summaries:
        raise ValueError("No comparison summaries were provided.")

    output_root.mkdir(parents=True, exist_ok=True)
    merged_methods: dict[str, dict[str, Any]] = {}
    input_path = str(summaries[0].get("input", ""))

    for source_root, summary in zip(source_roots, summaries):
        for method_summary in list(summary.get("methods", []) or []):
            method_name = str(method_summary.get("method_name", ""))
            if not method_name:
                continue
            method_dir = _resolve_path(method_summary.get("output_dir", source_root / method_name))
            _copy_method_dir(method_dir, output_root / method_name)

            incoming = _deepcopy_payload(method_summary)
            incoming = _populate_method_summary_defaults(
                incoming,
                comparison_runner.METHOD_BY_NAME.get(method_name),
            )
            incoming["output_dir"] = str(output_root / method_name)
            incoming["eval_report"] = str(output_root / method_name / "eval_report.json")
            if method_name in merged_methods:
                merged_methods[method_name] = _choose_preferred_method(merged_methods[method_name], incoming)
            else:
                merged_methods[method_name] = incoming

    selected_methods = [name for name in COMPARISON_METHOD_ORDER if name in merged_methods]
    method_summaries = [merged_methods[name] for name in selected_methods]
    time_rows = [_time_row_for_method(None, item, include_group=False) for item in method_summaries]
    known_rows = [_known_row_for_method(None, item, include_group=False) for item in method_summaries]
    ood_rows = [_ood_row_for_method(None, item, include_group=False) for item in method_summaries]
    baseline_rows = [
        comparison_runner._baseline_config_row(comparison_runner.METHOD_BY_NAME[name])
        for name in selected_methods
        if name in comparison_runner.METHOD_BY_NAME
    ]
    failed_method_records = [
        {
            "method_name": item.get("method_name", ""),
            "display_name": item.get("display_name", ""),
            "status": item.get("status", ""),
            "note": item.get("note", ""),
        }
        for item in method_summaries
        if str(item.get("status", "")) != "success"
    ]

    comparison_runner._write_tables(
        known_rows,
        comparison_runner.KNOWN_COLUMNS,
        output_root / "known_detection_table.csv",
        output_root / "known_detection_table.tex",
        comparison_runner.KNOWN_CAPTION,
    )
    comparison_runner._write_tables(
        ood_rows,
        comparison_runner.OOD_COLUMNS,
        output_root / "ood_detection_table.csv",
        output_root / "ood_detection_table.tex",
        comparison_runner.OOD_CAPTION,
    )
    comparison_runner._write_tables(
        time_rows,
        comparison_runner.TIME_COLUMNS,
        output_root / "detection_time_table.csv",
        output_root / "detection_time_table.tex",
        comparison_runner.TIME_CAPTION,
    )
    comparison_runner._write_tables(
        baseline_rows,
        comparison_runner.BASELINE_CONFIG_COLUMNS,
        output_root / "baseline_config_table.csv",
        output_root / "baseline_config_table.tex",
        comparison_runner.BASELINE_CONFIG_CAPTION,
        round_numeric=False,
        aligns="lllllll",
    )
    save_json(failed_method_records, output_root / "failed_methods.json")

    summary = {
        "input": input_path,
        "output_root": str(output_root),
        "output_dir": str(output_root),
        "paper_note": str(
            summaries[0].get("paper_note", comparison_runner.PROXY_IMPLEMENTATION_NOTE)
            or comparison_runner.PROXY_IMPLEMENTATION_NOTE
        ),
        "selected_methods": selected_methods,
        "successful_methods": [item["method_name"] for item in method_summaries if item.get("status") == "success"],
        "skipped_methods": [item["method_name"] for item in method_summaries if item.get("status") == "skipped"],
        "failed_methods": [item["method_name"] for item in method_summaries if item.get("status") == "failed"],
        "methods": method_summaries,
        "tables": {
            "known_detection_csv": str(output_root / "known_detection_table.csv"),
            "known_detection_tex": str(output_root / "known_detection_table.tex"),
            "ood_detection_csv": str(output_root / "ood_detection_table.csv"),
            "ood_detection_tex": str(output_root / "ood_detection_table.tex"),
            "detection_time_csv": str(output_root / "detection_time_table.csv"),
            "detection_time_tex": str(output_root / "detection_time_table.tex"),
            "baseline_config_csv": str(output_root / "baseline_config_table.csv"),
            "baseline_config_tex": str(output_root / "baseline_config_table.tex"),
        },
        "failed_methods_file": str(output_root / "failed_methods.json"),
    }
    save_json(summary, output_root / "comparison_summary.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge partial experiment outputs into a standard result root.")
    parser.add_argument("--mode", required=True, choices=["homogeneous", "comparison"])
    parser.add_argument("--source_roots", required=True, help="Comma-separated source roots to merge.")
    parser.add_argument("--output_root", required=True, help="Destination root for merged outputs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_roots = [_resolve_path(item.strip()) for item in str(args.source_roots).split(",") if item.strip()]
    output_root = _resolve_path(args.output_root)
    if args.mode == "homogeneous":
        merge_homogeneous(source_roots, output_root)
    else:
        merge_comparison(source_roots, output_root)


if __name__ == "__main__":
    main()
