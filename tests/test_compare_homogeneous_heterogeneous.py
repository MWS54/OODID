from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "compare_homogeneous_heterogeneous.py"
SPEC = importlib.util.spec_from_file_location("compare_homogeneous_heterogeneous", SCRIPT_PATH)
compare_script = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(compare_script)


def make_args(homogeneous_root: Path, heterogeneous_root: Path, output_dir: Path, *, focus_methods: str = "") -> SimpleNamespace:
    return SimpleNamespace(
        homogeneous_root=str(homogeneous_root),
        heterogeneous_root=str(heterogeneous_root),
        output_dir=str(output_dir),
        focus_methods=focus_methods or "rapier_proxy,hypervision_proxy,recda_proxy,rids_lite_proxy,ucs_oodid",
    )


class CompareHomogeneousHeterogeneousTests(unittest.TestCase):
    def test_compare_homogeneous_heterogeneous_aggregates_and_formats_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            homogeneous_root = tmp_path / "homogeneous"
            heterogeneous_root = tmp_path / "heterogeneous"
            output_dir = tmp_path / "comparison"
            homogeneous_root.mkdir(parents=True, exist_ok=True)
            heterogeneous_root.mkdir(parents=True, exist_ok=True)

            pd.DataFrame(
                [
                    {"Group": "uav_01", "Method": "RAPIER-style", "AUROC": 0.80, "OOD-F1": 0.60},
                    {"Group": "uav_02", "Method": "RAPIER-style", "AUROC": 0.70, "OOD-F1": 0.40},
                    {"Group": "uav_01", "Method": "HyperVision-style", "AUROC": 0.76, "OOD-F1": 0.55},
                    {"Group": "uav_02", "Method": "HyperVision-style", "AUROC": 0.66, "OOD-F1": 0.45},
                    {"Group": "uav_01", "Method": "ReCDA-style", "AUROC": 0.88, "OOD-F1": 0.80},
                    {"Group": "uav_02", "Method": "ReCDA-style", "AUROC": 0.78, "OOD-F1": 0.60},
                    {"Group": "uav_01", "Method": "RIDS-style", "AUROC": "", "OOD-F1": ""},
                    {"Group": "uav_02", "Method": "RIDS-style", "AUROC": 0.75, "OOD-F1": 0.50},
                    {"Group": "uav_01", "Method": "UCS-OODID", "AUROC": 0.95, "OOD-F1": 0.90},
                    {"Group": "uav_02", "Method": "UCS-OODID", "AUROC": 0.85, "OOD-F1": 0.70},
                ]
            ).to_csv(homogeneous_root / "homogeneous_ood_detection_table.csv", index=False)

            pd.DataFrame(
                [
                    {"Method": "RAPIER-style", "AUROC": 0.45, "OOD-F1": 0.30},
                    {"Method": "HyperVision-style", "AUROC": 0.58, "OOD-F1": 0.40},
                    {"Method": "ReCDA-style", "AUROC": 0.70, "OOD-F1": 0.50},
                    {"Method": "RIDS-style", "AUROC": 0.55, "OOD-F1": 0.45},
                    {"Method": "UCS-OODID", "AUROC": 0.82, "OOD-F1": 0.65},
                ]
            ).to_csv(heterogeneous_root / "ood_detection_table.csv", index=False)

            summary = compare_script.compare_homogeneous_heterogeneous(
                make_args(homogeneous_root, heterogeneous_root, output_dir)
            )

            self.assertEqual(
                summary["focus_methods"],
                ["RAPIER-style", "HyperVision-style", "ReCDA-style", "RIDS-style", "UCS-OODID"],
            )
            self.assertTrue((output_dir / "homogeneous_vs_heterogeneous_table.csv").exists())
            self.assertTrue((output_dir / "homogeneous_vs_heterogeneous_table.tex").exists())

            table = pd.read_csv(output_dir / "homogeneous_vs_heterogeneous_table.csv")
            self.assertEqual(table.columns.tolist(), compare_script.OUTPUT_COLUMNS)
            self.assertEqual(
                table["Method"].tolist(),
                ["RAPIER-style", "HyperVision-style", "ReCDA-style", "RIDS-style", "UCS-OODID"],
            )
            self.assertEqual(
                table.loc[table["Method"] == "RAPIER-style", "Homogeneous OOD-F1"].iloc[0],
                "0.5000 ± 0.1000",
            )
            self.assertEqual(
                table.loc[table["Method"] == "RAPIER-style", "Homogeneous AUROC"].iloc[0],
                0.7500,
            )
            self.assertEqual(
                table.loc[table["Method"] == "RAPIER-style", "OOD-F1 Drop"].iloc[0],
                0.2000,
            )
            self.assertEqual(
                table.loc[table["Method"] == "RAPIER-style", "Drop Ratio"].iloc[0],
                0.4000,
            )
            self.assertEqual(
                table.loc[table["Method"] == "UCS-OODID", "Homogeneous OOD-F1"].iloc[0],
                "0.8000 ± 0.1000",
            )
            self.assertEqual(
                table.loc[table["Method"] == "UCS-OODID", "Heterogeneous OOD-F1"].iloc[0],
                0.6500,
            )
            self.assertEqual(
                table.loc[table["Method"] == "UCS-OODID", "Drop Ratio"].iloc[0],
                0.1875,
            )

            tex = (output_dir / "homogeneous_vs_heterogeneous_table.tex").read_text(encoding="utf-8")
            self.assertIn(compare_script.TABLE_CAPTION, tex)
            self.assertIn(r"\textbf{UCS-OODID}", tex)
            self.assertIn(r"\textbf{0.8000} $\pm$ \textbf{0.1000}", tex)

    def test_compare_homogeneous_heterogeneous_accepts_legacy_internal_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            homogeneous_root = tmp_path / "homogeneous"
            heterogeneous_root = tmp_path / "heterogeneous"
            output_dir = tmp_path / "comparison"
            homogeneous_root.mkdir(parents=True, exist_ok=True)
            heterogeneous_root.mkdir(parents=True, exist_ok=True)

            pd.DataFrame(
                [{"Group": "uav_01", "Method": "UCS-OODID", "AUROC": 0.90, "OOD-F1": 0.80}]
            ).to_csv(homogeneous_root / "homogeneous_ood_detection_table.csv", index=False)
            pd.DataFrame(
                [{"Method": "UCS-OODID", "AUROC": 0.70, "OOD-F1": 0.60}]
            ).to_csv(heterogeneous_root / "ood_detection_table.csv", index=False)

            summary = compare_script.compare_homogeneous_heterogeneous(
                make_args(
                    homogeneous_root,
                    heterogeneous_root,
                    output_dir,
                    focus_methods="random_forest,ucs_oodid",
                )
            )

            self.assertEqual(summary["focus_methods"], ["RAPIER-style", "UCS-OODID"])
            table = pd.read_csv(output_dir / "homogeneous_vs_heterogeneous_table.csv")
            self.assertEqual(table["Method"].tolist(), ["RAPIER-style", "UCS-OODID"])


if __name__ == "__main__":
    unittest.main()
