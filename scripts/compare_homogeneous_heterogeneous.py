#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_comparison_experiments as comparison_runner
from ucs_oodid.utils import parse_csv_list

OUTPUT_COLUMNS = [
    "Method",
    "Homogeneous OOD-F1",
    "Homogeneous AUROC",
    "Heterogeneous OOD-F1",
    "Heterogeneous AUROC",
    "OOD-F1 Drop",
    "Drop Ratio",
]
TABLE_CAPTION = "Homogeneous-to-heterogeneous performance degradation comparison."
LEGACY_DISPLAY_ALIASES = {
    "random forest": "RAPIER-style",
    "transformer+gcn": "HyperVision-style",
    "transformer-only + mean": "ReCDA-style",
    "transformer-only + mean fusion": "ReCDA-style",
    "transformer-only + mean ood fusion": "ReCDA-style",
    "svm": "RIDS-style",
}


def latex_escape(text: str) -> str:
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


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(numeric):
        return None
    return float(numeric)


def _format_float(value: Optional[float]) -> str:
    return "" if value is None else f"{value:.4f}"


def _format_mean_std(mean: Optional[float], std: Optional[float], count: int) -> str:
    if count <= 0 or mean is None or std is None:
        return ""
    return f"{mean:.4f} ± {std:.4f}"


def _display_name_maps() -> tuple[dict[str, str], dict[str, str]]:
    by_name = {}
    by_lower_display = {}
    for spec in comparison_runner.METHOD_SPECS:
        by_name[str(spec.name)] = str(spec.display_name)
        by_lower_display[str(spec.display_name).lower()] = str(spec.display_name)
    for alias, canonical in getattr(comparison_runner, "METHOD_ALIASES", {}).items():
        spec = comparison_runner.METHOD_BY_NAME.get(str(canonical))
        if spec is not None:
            by_name[str(alias)] = str(spec.display_name)
    by_lower_display.update(LEGACY_DISPLAY_ALIASES)
    return by_name, by_lower_display


def _canonical_method_display(value: Any) -> str:
    text = str(value or "").strip()
    by_name, by_lower_display = _display_name_maps()
    return by_name.get(text, by_lower_display.get(text.lower(), text))


def _resolve_focus_methods(focus_methods: str) -> list[str]:
    method_name_to_display, display_lookup = _display_name_maps()
    requested = parse_csv_list(focus_methods)
    if not requested:
        requested = [str(spec.name) for spec in comparison_runner.METHOD_SPECS]

    resolved: list[str] = []
    seen = set()
    for item in requested:
        key = str(item).strip()
        display_name = method_name_to_display.get(key, method_name_to_display.get(key.lower(), display_lookup.get(key.lower(), key)))
        if display_name in seen:
            continue
        seen.add(display_name)
        resolved.append(display_name)
    return resolved


def _prepare_homogeneous_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Homogeneous OOD table was not found: {path}")
    frame = pd.read_csv(path)
    required = {"Group", "Method", "AUROC", "OOD-F1"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Homogeneous OOD table is missing required columns: {', '.join(missing)}")
    frame = frame.copy()
    frame["Method"] = frame["Method"].map(_canonical_method_display)
    frame["AUROC"] = pd.to_numeric(frame["AUROC"], errors="coerce")
    frame["OOD-F1"] = pd.to_numeric(frame["OOD-F1"], errors="coerce")
    return frame


def _prepare_heterogeneous_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Heterogeneous OOD table was not found: {path}")
    frame = pd.read_csv(path)
    required = {"Method", "AUROC", "OOD-F1"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Heterogeneous OOD table is missing required columns: {', '.join(missing)}")
    frame = frame.copy()
    frame["Method"] = frame["Method"].map(_canonical_method_display)
    frame["AUROC"] = pd.to_numeric(frame["AUROC"], errors="coerce")
    frame["OOD-F1"] = pd.to_numeric(frame["OOD-F1"], errors="coerce")
    return frame


def _first_valid(series: pd.Series) -> Optional[float]:
    for value in series.tolist():
        numeric = _safe_float(value)
        if numeric is not None:
            return numeric
    return None


def _format_tex_cell(value: Any, highlight: bool = False) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if " ± " in text:
        mean, std = text.split(" ± ", 1)
        mean_text = latex_escape(mean)
        std_text = latex_escape(std)
        if highlight:
            return rf"\textbf{{{mean_text}}} $\pm$ \textbf{{{std_text}}}"
        return rf"{mean_text} $\pm$ {std_text}"
    escaped = latex_escape(text)
    if highlight:
        return rf"\textbf{{{escaped}}}"
    return escaped


def _write_table(rows: list[dict[str, Any]], output_csv: Path, output_tex: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows).reindex(columns=OUTPUT_COLUMNS)
    frame.to_csv(output_csv, index=False)

    aligns = "l" + "c" * (len(OUTPUT_COLUMNS) - 1)
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        rf"\caption{{{TABLE_CAPTION}}}",
        r"\begin{tabular}{" + aligns + "}",
        r"\toprule",
        " & ".join(latex_escape(str(column)) for column in OUTPUT_COLUMNS) + r" \\",
        r"\midrule",
    ]
    for row in frame.itertuples(index=False, name=None):
        highlight = str(row[0]) == "UCS-OODID"
        cells = [_format_tex_cell(value, highlight=highlight) for value in row]
        lines.append(" & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    output_tex.write_text("\n".join(lines) + "\n", encoding="utf-8")


def compare_homogeneous_heterogeneous(args) -> dict[str, Any]:
    homogeneous_root = Path(getattr(args, "homogeneous_root"))
    heterogeneous_root = Path(getattr(args, "heterogeneous_root"))
    output_dir = Path(getattr(args, "output_dir", "runs/homo_hetero_comparison"))
    output_dir.mkdir(parents=True, exist_ok=True)

    homogeneous_path = homogeneous_root / "homogeneous_ood_detection_table.csv"
    heterogeneous_path = heterogeneous_root / "ood_detection_table.csv"
    homogeneous_frame = _prepare_homogeneous_frame(homogeneous_path)
    heterogeneous_frame = _prepare_heterogeneous_frame(heterogeneous_path)
    focus_methods = _resolve_focus_methods(str(getattr(args, "focus_methods", "") or ""))

    rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for display_name in focus_methods:
        homo_subset = homogeneous_frame.loc[homogeneous_frame["Method"].astype(str) == display_name].copy()
        hetero_subset = heterogeneous_frame.loc[heterogeneous_frame["Method"].astype(str) == display_name].copy()

        homo_ood_values = homo_subset["OOD-F1"].dropna().astype(float).to_numpy(dtype=np.float64)
        homo_auroc_values = homo_subset["AUROC"].dropna().astype(float).to_numpy(dtype=np.float64)
        homo_ood_mean = float(np.mean(homo_ood_values)) if homo_ood_values.size else None
        homo_ood_std = float(np.std(homo_ood_values)) if homo_ood_values.size else None
        homo_auroc_mean = float(np.mean(homo_auroc_values)) if homo_auroc_values.size else None

        hetero_ood_f1 = _first_valid(hetero_subset["OOD-F1"]) if "OOD-F1" in hetero_subset else None
        hetero_auroc = _first_valid(hetero_subset["AUROC"]) if "AUROC" in hetero_subset else None

        drop = None
        if homo_ood_mean is not None and hetero_ood_f1 is not None:
            drop = float(homo_ood_mean - hetero_ood_f1)

        drop_ratio = None
        if drop is not None and homo_ood_mean not in (None, 0.0):
            drop_ratio = float(drop / homo_ood_mean)

        rows.append(
            {
                "Method": display_name,
                "Homogeneous OOD-F1": _format_mean_std(homo_ood_mean, homo_ood_std, int(homo_ood_values.size)),
                "Homogeneous AUROC": _format_float(homo_auroc_mean),
                "Heterogeneous OOD-F1": _format_float(hetero_ood_f1),
                "Heterogeneous AUROC": _format_float(hetero_auroc),
                "OOD-F1 Drop": _format_float(drop),
                "Drop Ratio": _format_float(drop_ratio),
            }
        )
        summary_rows.append(
            {
                "method": display_name,
                "homogeneous_group_count": int(homo_ood_values.size),
                "homogeneous_mean_ood_f1": homo_ood_mean,
                "homogeneous_std_ood_f1": homo_ood_std,
                "homogeneous_mean_auroc": homo_auroc_mean,
                "heterogeneous_ood_f1": hetero_ood_f1,
                "heterogeneous_auroc": hetero_auroc,
                "degradation_ood_f1": drop,
                "degradation_ratio": drop_ratio,
            }
        )

    output_csv = output_dir / "homogeneous_vs_heterogeneous_table.csv"
    output_tex = output_dir / "homogeneous_vs_heterogeneous_table.tex"
    _write_table(rows, output_csv, output_tex)

    summary = {
        "homogeneous_root": str(homogeneous_root),
        "heterogeneous_root": str(heterogeneous_root),
        "output_dir": str(output_dir),
        "focus_methods": focus_methods,
        "rows": summary_rows,
        "tables": {
            "csv": str(output_csv),
            "tex": str(output_tex),
        },
    }
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare OOD degradation from homogeneous single-source runs to a heterogeneous reference."
    )
    parser.add_argument("--homogeneous_root", required=True)
    parser.add_argument("--heterogeneous_root", required=True)
    parser.add_argument("--output_dir", default="runs/homo_hetero_comparison")
    parser.add_argument(
        "--focus_methods",
        default="rapier_proxy,hypervision_proxy,recda_proxy,rids_lite_proxy,ucs_oodid",
        help="Comma-separated focus methods. Supports internal method names or display names.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = compare_homogeneous_heterogeneous(args)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
