from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ucs_oodid.preprocessing import GroupAwareMetadataPreprocessor, MetadataPreprocessor

DETECT_MODULE_PATH = ROOT / "scripts" / "detect.py"
SPEC = importlib.util.spec_from_file_location("detect_script_group_normalization", DETECT_MODULE_PATH)
detect_script = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(detect_script)


def make_group_df() -> pd.DataFrame:
    rows = []
    for idx in range(12):
        rows.append(
            {
                "record_id": f"a_{idx}",
                "timestamp": idx,
                "label": "benign",
                "uav_id": "uav_01",
                "feat_a": float(idx),
                "feat_b": float(idx * 2 + 10),
            }
        )
    for idx in range(12):
        rows.append(
            {
                "record_id": f"b_{idx}",
                "timestamp": 100 + idx,
                "label": "benign",
                "uav_id": "uav_02",
                "feat_a": float(100 + idx),
                "feat_b": float(-50 + idx * 3),
            }
        )
    return pd.DataFrame(rows)


class GroupNormalizationTests(unittest.TestCase):
    def test_reserved_metadata_columns_are_not_used_as_features(self):
        df = pd.DataFrame(
            [
                {
                    "record_id": "r0",
                    "timestamp": 0,
                    "label": "benign",
                    "uav_id": "uav_01",
                    "domain_id": 1,
                    "source_type": 10,
                    "direction_type": 100,
                    "scenario_role": 1000,
                    "original_group_id": 10000,
                    "mission_phase": "cruise",
                    "sim_time": 10.0,
                    "battery_soc": 91.0,
                    "speed": 14.0,
                    "altitude": 120.0,
                    "source_missing": 0,
                    "source_is_ipv4": 1,
                    "source_has_mac_like": 0,
                    "flight_energy_wh": 1.2,
                    "communication_energy_wh": 0.3,
                    "detection_energy_wh": 0.1,
                    "total_energy_wh": 1.6,
                    "feat_a": 0.1,
                    "feat_b": 1.0,
                },
                {
                    "record_id": "r1",
                    "timestamp": 1,
                    "label": "dos",
                    "uav_id": "uav_02",
                    "domain_id": 2,
                    "source_type": 20,
                    "direction_type": 200,
                    "scenario_role": 2000,
                    "original_group_id": 20000,
                    "mission_phase": "hover",
                    "sim_time": 11.0,
                    "battery_soc": 89.0,
                    "speed": 12.0,
                    "altitude": 80.0,
                    "source_missing": 1,
                    "source_is_ipv4": 0,
                    "source_has_mac_like": 1,
                    "flight_energy_wh": 1.4,
                    "communication_energy_wh": 0.4,
                    "detection_energy_wh": 0.2,
                    "total_energy_wh": 2.0,
                    "feat_a": 0.2,
                    "feat_b": 2.0,
                },
            ]
        )

        pre = MetadataPreprocessor(group_col="uav_id")
        pre.fit(df)

        self.assertEqual(pre.feature_cols, ["feat_a", "feat_b"])

    def test_group_normalization_centers_each_group_independently(self):
        df = make_group_df()
        pre = GroupAwareMetadataPreprocessor(group_col="uav_id")
        x = pre.fit_transform(df)

        mask_1 = df["uav_id"].eq("uav_01").to_numpy()
        mask_2 = df["uav_id"].eq("uav_02").to_numpy()
        np.testing.assert_allclose(x[mask_1].mean(axis=0), np.zeros(x.shape[1]), atol=1e-6)
        np.testing.assert_allclose(x[mask_2].mean(axis=0), np.zeros(x.shape[1]), atol=1e-6)
        self.assertEqual(set(pre.group_scalers), {"uav_01", "uav_02"})

    def test_small_group_falls_back_to_global_scaler(self):
        df = make_group_df().copy()
        small = pd.DataFrame(
            [
                {
                    "record_id": "small_0",
                    "timestamp": 500,
                    "label": "benign",
                    "uav_id": "uav_small",
                    "feat_a": 999.0,
                    "feat_b": -999.0,
                },
                {
                    "record_id": "small_1",
                    "timestamp": 501,
                    "label": "benign",
                    "uav_id": "uav_small",
                    "feat_a": 1001.0,
                    "feat_b": -1001.0,
                },
            ]
        )
        df = pd.concat([df, small], ignore_index=True)
        pre = GroupAwareMetadataPreprocessor(group_col="uav_id", min_group_records=3)
        transformed = pre.fit_transform(df)

        self.assertNotIn("uav_small", pre.group_scalers)
        self.assertEqual(
            pre.group_fallbacks["uav_small"],
            "fallback_to_global_due_to_small_group_size",
        )
        small_mask = df["uav_id"].eq("uav_small").to_numpy()
        raw = pre._extract(df, fit_mode=False)
        expected = pre.scaler.transform(raw[small_mask]).astype(np.float32)
        np.testing.assert_allclose(transformed[small_mask], expected, atol=1e-6)
        summary = pre.normalization_summary()
        self.assertEqual(summary["fallback_groups"], ["uav_small"])
        self.assertFalse(summary["groups"]["uav_small"]["used_group_scaler"])

    def test_detect_restores_group_scaler_from_artifact(self):
        df = make_group_df()
        pre = GroupAwareMetadataPreprocessor(group_col="uav_id")
        pre.fit(df)

        artifact = {
            "preprocessor": GroupAwareMetadataPreprocessor(group_col="uav_id"),
            "normalization_mode": "group",
            "group_col": "uav_id",
            "feature_cols": list(pre.feature_cols),
            "feature_medians": dict(pre.feature_medians),
            "global_scaler": pre.scaler,
            "group_scalers": pre.group_scalers,
            "group_normalization_fallbacks": pre.group_fallbacks,
        }

        restored, mode = detect_script.resolve_preprocessor_from_artifact(artifact)

        self.assertEqual(mode, "group")
        self.assertEqual(restored.normalization_mode, "group")
        self.assertEqual(set(restored.group_scalers), {"uav_01", "uav_02"})
        np.testing.assert_allclose(restored.transform(df), pre.transform(df), atol=1e-6)


if __name__ == "__main__":
    unittest.main()
