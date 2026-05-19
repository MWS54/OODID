from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from ucs_oodid.simulator.attack_replay import AttackReplayPool


def write_csv(path: Path, rows: list[dict[str, object]]) -> Path:
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def build_new_dataset_paths(root: Path) -> dict[str, Path]:
    unsw_path = write_csv(
        root / "unsw_nb15.csv",
        [
            {"flow_feature": 1.0, "label": "benign", "timestamp": 0.0, "record_id": "unsw0"},
            {"flow_feature": 2.0, "label": "recon_scanning", "timestamp": 1.0, "record_id": "unsw1"},
            {"flow_feature": 3.0, "label": "backdoor", "timestamp": 2.0, "record_id": "unsw2"},
        ],
    )
    ecu_path = write_csv(
        root / "ecu_ioft.csv",
        [
            {"flow_feature": 4.0, "label": "benign", "timestamp": 0.0, "record_id": "ecu0"},
            {"flow_feature": 5.0, "label": "unauthorized_udp", "timestamp": 1.0, "record_id": "ecu1"},
            {"flow_feature": 6.0, "label": "wifi_deauth", "timestamp": 2.0, "record_id": "ecu2"},
        ],
    )
    uavids_path = write_csv(
        root / "uavids.csv",
        [
            {"flow_feature": 7.0, "label": "benign", "timestamp": 0.0, "record_id": "uavids0"},
            {"flow_feature": 8.0, "label": "wormhole", "timestamp": 1.0, "record_id": "uavids1"},
            {"flow_feature": 9.0, "label": "sybil", "timestamp": 2.0, "record_id": "uavids2"},
        ],
    )
    return {
        "unsw_nb15": unsw_path,
        "ecu_ioft": ecu_path,
        "uavids": uavids_path,
    }


class AttackReplayNewDatasetsTests(unittest.TestCase):
    def test_attack_replay_pool_samples_from_uav05_uav06_uav07_without_uav04(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            pool = AttackReplayPool(
                dataset_paths=build_new_dataset_paths(Path(tmp_dir)),
                uav_dataset_bindings={
                    "uav_05": "unsw_nb15",
                    "uav_06": "ecu_ioft",
                    "uav_07": "uavids",
                },
                seed=23,
            )

            self.assertFalse(pool.has_binding("uav_04"))

            unsw_rows = pool.sample_attack_records(
                "uav_05",
                attack_type="scan",
                attack_intensity=2,
                attack_replay_mode="loop",
                sim_fields={"sim_time": 5.0},
            )
            ecu_rows = pool.sample_attack_records(
                "uav_06",
                attack_type="command_injection",
                attack_intensity=1,
                attack_replay_mode="sequential",
            )
            uavids_rows = pool.sample_attack_records(
                "uav_07",
                attack_type="command_injection",
                attack_intensity=2,
                attack_replay_mode="loop",
            )

        self.assertEqual(len(unsw_rows), 2)
        self.assertTrue(all(row["uav_id"] == "uav_05" for row in unsw_rows))
        self.assertTrue(all(row["dataset_name"] == "unsw_nb15" for row in unsw_rows))
        self.assertTrue(all(row["attack_source_dataset"] == "unsw_nb15" for row in unsw_rows))
        self.assertTrue(all(row["source_type"] == "external_non_uav" for row in unsw_rows))
        self.assertTrue(all(row["simulation_role"] == "external_ood" for row in unsw_rows))
        self.assertTrue(all(row["attack_type"] == "recon_scanning" for row in unsw_rows))
        self.assertTrue(all(row["sim_time"] == 5.0 for row in unsw_rows))
        self.assertTrue(all("not real UAV flight" in str(row["source_note"]) for row in unsw_rows))

        self.assertEqual(len(ecu_rows), 1)
        self.assertEqual(ecu_rows[0]["uav_id"], "uav_06")
        self.assertEqual(ecu_rows[0]["dataset_name"], "ecu_ioft")
        self.assertEqual(ecu_rows[0]["attack_source_dataset"], "ecu_ioft")
        self.assertEqual(ecu_rows[0]["source_type"], "uav_iot_wifi")
        self.assertEqual(ecu_rows[0]["simulation_role"], "uav_replay")
        self.assertEqual(ecu_rows[0]["attack_type"], "unauthorized_udp")
        self.assertEqual(ecu_rows[0]["source_record_id"], "ecu1")

        self.assertEqual(len(uavids_rows), 2)
        self.assertTrue(all(row["uav_id"] == "uav_07" for row in uavids_rows))
        self.assertTrue(all(row["dataset_name"] == "uavids" for row in uavids_rows))
        self.assertTrue(all(row["attack_source_dataset"] == "uavids" for row in uavids_rows))
        self.assertTrue(all(row["source_type"] == "uav" for row in uavids_rows))
        self.assertTrue(all(row["simulation_role"] == "uav_replay" for row in uavids_rows))
        self.assertEqual([row["attack_type"] for row in uavids_rows], ["wormhole", "sybil"])
        self.assertEqual([row["source_record_id"] for row in uavids_rows], ["uavids1", "uavids2"])


if __name__ == "__main__":
    unittest.main()
