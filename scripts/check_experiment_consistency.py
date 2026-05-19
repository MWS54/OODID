#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ucs_oodid.artifacts import load_artifact

EXPECTED_WINDOW_SIZE = 32
EXPECTED_STRIDE = 16
ALLOWED_WINDOW_SENSITIVITY_SETTINGS = {(8, 4), (16, 8), (32, 16), (64, 32)}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def normalize_path(path: Path | str) -> str:
    return os.path.normcase(str(Path(path).resolve()))


def check_equal(name: str, actual: Any, expected: Any, details: dict[str, Any] | None = None) -> dict[str, Any]:
    passed = actual == expected
    payload = {
        "name": name,
        "passed": passed,
        "actual": actual,
        "expected": expected,
    }
    if details:
        payload["details"] = details
    return payload


def check_true(name: str, passed: bool, details: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = {
        "name": name,
        "passed": bool(passed),
    }
    if details:
        payload["details"] = details
    return payload


def scan_for_old_defaults(results_dir: Path) -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []
    patterns = {
        "window_size_16": re.compile(r"window(?:_| )?size[^0-9]{0,12}16\b", re.IGNORECASE),
        "stride_8": re.compile(r"stride[^0-9]{0,12}8\b", re.IGNORECASE),
    }
    for path in results_dir.rglob("*"):
        if not path.is_file():
            continue
        if "window_sensitivity" in path.parts or "figures" in path.parts:
            continue
        if path.suffix.lower() not in {".json", ".csv"}:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for label, pattern in patterns.items():
            if pattern.search(text):
                findings.append({"file": str(path), "pattern": label})
    return findings


def check_window_sensitivity(window_dir: Path) -> list[dict[str, Any]]:
    payload = read_json(window_dir / "window_sensitivity_summary.json")
    confirmed = {
        (int(item["window"]), int(item["stride"]))
        for item in payload.get("confirmed_settings", [])
        if isinstance(item, dict) and "window" in item and "stride" in item
    }
    frame = pd.read_csv(window_dir / "window_sensitivity_table.csv")
    observed = {
        (int(row.Window), int(row.Stride))
        for row in frame.itertuples(index=False)
        if not pd.isna(row.Window) and not pd.isna(row.Stride)
    }
    return [
        check_equal(
            "window_sensitivity_confirmed_settings",
            sorted(confirmed),
            sorted(ALLOWED_WINDOW_SENSITIVITY_SETTINGS),
        ),
        check_equal(
            "window_sensitivity_observed_settings",
            sorted(observed),
            sorted(ALLOWED_WINDOW_SENSITIVITY_SETTINGS),
        ),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Check 32/16 experiment output consistency.")
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--artifact", required=True)
    args = parser.parse_args()

    results_dir = Path(args.results_dir).resolve()
    artifact_path = Path(args.artifact).resolve()

    checks: list[dict[str, Any]] = []

    comparison = read_json(results_dir / "comparison" / "comparison_summary.json")
    comparison_cfg = dict(comparison.get("config", {}) or {})
    checks.append(
        check_equal(
            "comparison_window_size",
            int(comparison_cfg.get("window_size", -1)),
            EXPECTED_WINDOW_SIZE,
        )
    )
    checks.append(
        check_equal(
            "comparison_stride",
            int(comparison_cfg.get("stride", -1)),
            EXPECTED_STRIDE,
        )
    )

    robustness = read_json(results_dir / "homogeneous_vs_heterogeneous" / "homogeneous_summary.json")
    robustness_cfg = dict(robustness.get("config", {}) or {})
    checks.append(
        check_equal(
            "robustness_window_size",
            int(robustness_cfg.get("window_size", -1)),
            EXPECTED_WINDOW_SIZE,
        )
    )
    checks.append(
        check_equal(
            "robustness_stride",
            int(robustness_cfg.get("stride", -1)),
            EXPECTED_STRIDE,
        )
    )

    ablation = read_json(results_dir / "ablation" / "ablation_summary.json")
    ablation_conditions = dict(ablation.get("common_conditions", {}) or {})
    checks.append(
        check_equal(
            "ablation_window_size",
            int(ablation_conditions.get("window_size", -1)),
            EXPECTED_WINDOW_SIZE,
        )
    )
    checks.append(
        check_equal(
            "ablation_stride",
            int(ablation_conditions.get("stride", -1)),
            EXPECTED_STRIDE,
        )
    )

    benchmark = read_json(results_dir / "deployment" / "benchmark_report.json")
    checks.append(
        check_equal(
            "benchmark_window_size",
            int(benchmark.get("window_size", -1)),
            EXPECTED_WINDOW_SIZE,
        )
    )
    checks.append(
        check_equal(
            "benchmark_stride",
            int(benchmark.get("stride", -1)),
            EXPECTED_STRIDE,
        )
    )
    checks.append(
        check_equal(
            "benchmark_artifact_path",
            normalize_path(benchmark.get("artifact", "")),
            normalize_path(artifact_path),
        )
    )

    replay = read_json(results_dir / "deployment" / "simulation_online.json")
    checks.append(
        check_equal(
            "online_replay_window_size",
            int(replay.get("window_size", -1)),
            EXPECTED_WINDOW_SIZE,
        )
    )
    checks.append(
        check_equal(
            "online_replay_stride",
            int(replay.get("stride", -1)),
            EXPECTED_STRIDE,
        )
    )
    checks.append(
        check_equal(
            "online_replay_artifact_path",
            normalize_path(replay.get("artifact_path", "")),
            normalize_path(artifact_path),
        )
    )

    online_summary = pd.read_csv(results_dir / "deployment" / "online_replay_summary.csv")
    online_row = online_summary.iloc[0].to_dict()
    checks.append(
        check_equal(
            "online_summary_window_size",
            int(online_row.get("window_size", -1)),
            EXPECTED_WINDOW_SIZE,
        )
    )
    checks.append(
        check_equal(
            "online_summary_stride",
            int(online_row.get("stride", -1)),
            EXPECTED_STRIDE,
        )
    )

    benchmark_summary = pd.read_csv(results_dir / "deployment" / "benchmark_table.csv")
    benchmark_row = benchmark_summary.iloc[0].to_dict()
    checks.append(
        check_equal(
            "benchmark_table_window_size",
            int(benchmark_row.get("window_size", -1)),
            EXPECTED_WINDOW_SIZE,
        )
    )
    checks.append(
        check_equal(
            "benchmark_table_stride",
            int(benchmark_row.get("stride", -1)),
            EXPECTED_STRIDE,
        )
    )

    artifact = load_artifact(artifact_path, map_location="cpu")
    window_config = dict(artifact.get("window_config", {}) or {})
    checks.append(
        check_equal(
            "artifact_window_size",
            int(window_config.get("size", -1)),
            EXPECTED_WINDOW_SIZE,
        )
    )
    checks.append(
        check_equal(
            "artifact_stride",
            int(window_config.get("stride", -1)),
            EXPECTED_STRIDE,
        )
    )

    checks.extend(check_window_sensitivity(results_dir / "window_sensitivity"))

    stale_findings = scan_for_old_defaults(results_dir)
    checks.append(
        check_true(
            "no_stale_16_8_defaults_outside_window_sensitivity",
            not stale_findings,
            details={"findings": stale_findings},
        )
    )

    passed = all(check["passed"] for check in checks)
    report = {
        "results_dir": str(results_dir),
        "artifact": str(artifact_path),
        "expected_main_setting": {
            "window_size": EXPECTED_WINDOW_SIZE,
            "stride": EXPECTED_STRIDE,
        },
        "overall_passed": passed,
        "checks": checks,
    }

    report_json = results_dir / "consistency_report.json"
    write_json(report_json, report)

    markdown_lines = [
        "# Experiment Consistency Report",
        "",
        f"Overall: {'PASS' if passed else 'FAIL'}",
        "",
        f"Results dir: `{results_dir}`",
        f"Artifact: `{artifact_path}`",
        "",
        "| Check | Status | Actual | Expected |",
        "| --- | --- | --- | --- |",
    ]
    for item in checks:
        actual = item.get("actual", "")
        expected = item.get("expected", "")
        markdown_lines.append(
            f"| {item['name']} | {'PASS' if item['passed'] else 'FAIL'} | {actual} | {expected} |"
        )
    if stale_findings:
        markdown_lines.extend(
            [
                "",
                "## Stale Findings",
                "",
            ]
        )
        for finding in stale_findings:
            markdown_lines.append(f"- `{finding['file']}` matched `{finding['pattern']}`")

    report_md = results_dir / "consistency_report.md"
    report_md.write_text("\n".join(markdown_lines) + "\n", encoding="utf-8")

    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
