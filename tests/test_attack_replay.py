from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from ucs_oodid.simulator import Attacker, GCS, SimulationConfig, SimulationEngine, UAV
from ucs_oodid.simulator.attack_replay import AttackReplayPool
from ucs_oodid.simulator.attacks import AttackEvent, AttackInjector


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def build_dataset_paths(tmp_path: Path) -> dict[str, Path]:
    uav_ndd_path = tmp_path / "uav_ndd.csv"
    gcs_path = tmp_path / "gcs_to_uav.csv"
    isot_path = tmp_path / "isot.csv"
    write_csv(
        uav_ndd_path,
        [
            {"raw_feature": 1.0, "label": "benign", "timestamp": 0.0, "record_id": "u0"},
            {"raw_feature": 10.0, "label": "jamming", "timestamp": 1.0, "record_id": "u1"},
            {"raw_feature": 11.0, "label": "jamming", "timestamp": 2.0, "record_id": "u2"},
            {"raw_feature": 12.0, "label": "replay", "timestamp": 3.0, "record_id": "u3"},
        ],
    )
    write_csv(
        gcs_path,
        [
            {"flow_feature": 21.0, "label": "reply", "timestamp": 0.0, "record_id": "g0"},
            {"flow_feature": 22.0, "label": "fake_landing", "timestamp": 1.0, "record_id": "g1"},
            {"flow_feature": 23.0, "label": "reply", "timestamp": 2.0, "record_id": "g2"},
        ],
    )
    write_csv(
        isot_path,
        [
            {"proto_feature": 31.0, "label": "injection", "timestamp": 0.0, "record_id": "i0"},
            {"proto_feature": 32.0, "label": "payload_manipulation", "timestamp": 1.0, "record_id": "i1"},
        ],
    )
    return {
        "uav_ndd": uav_ndd_path,
        "gcs_to_uav_updated": gcs_path,
        "isot_drone": isot_path,
    }


def build_extended_dataset_paths(tmp_path: Path) -> dict[str, Path]:
    unsw_path = tmp_path / "unsw.csv"
    ecu_path = tmp_path / "ecu.csv"
    uavids_path = tmp_path / "uavids.csv"
    write_csv(
        unsw_path,
        [
            {"flow_feature": 1.0, "label": "benign", "timestamp": 0.0, "record_id": "n0"},
            {"flow_feature": 2.0, "label": "recon_scanning", "timestamp": 1.0, "record_id": "n1"},
        ],
    )
    write_csv(
        ecu_path,
        [
            {"flow_feature": 3.0, "label": "benign", "timestamp": 0.0, "record_id": "e0"},
            {"flow_feature": 4.0, "label": "unauthorized_udp", "timestamp": 1.0, "record_id": "e1"},
        ],
    )
    write_csv(
        uavids_path,
        [
            {"flow_feature": 5.0, "label": "benign", "timestamp": 0.0, "record_id": "v0"},
            {"flow_feature": 6.0, "label": "wormhole", "timestamp": 1.0, "record_id": "v1"},
        ],
    )
    return {
        "unsw_nb15": unsw_path,
        "ecu_ioft": ecu_path,
        "uavids": uavids_path,
    }


class AttackReplayPoolTests(unittest.TestCase):
    def test_pool_uses_bound_dataset_and_single_attack_type_filter(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            pool = AttackReplayPool(dataset_paths=build_dataset_paths(Path(tmp_dir)), seed=7)

            rows = pool.sample_attack_records(
                "uav_01",
                attack_type="jamming_proxy",
                attack_intensity=2,
                attack_replay_mode="sequential",
                sim_fields={"sim_time": 7.0, "mission_phase": "cruise"},
            )

        self.assertEqual(len(rows), 2)
        self.assertEqual([row["attack_type"] for row in rows], ["jamming", "jamming"])
        self.assertEqual([row["source_timestamp"] for row in rows], [1.0, 2.0])
        self.assertTrue(all(row["dataset_name"] == "uav_ndd" for row in rows))
        self.assertTrue(all(row["attack_source_dataset"] == "uav_ndd" for row in rows))
        self.assertTrue(all(row["uav_id"] == "uav_01" for row in rows))
        self.assertTrue(all(row["sim_time"] == 7.0 for row in rows))
        self.assertTrue(all(row["sim_mission_phase"] == "cruise" for row in rows))

    def test_pool_supports_mixed_labels_and_loop_mode(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            pool = AttackReplayPool(dataset_paths=build_dataset_paths(Path(tmp_dir)), seed=9)

            first = pool.sample_attack_records(
                "uav_03",
                label=["injection", "payload_manipulation"],
                attack_intensity=3,
                attack_replay_mode="loop",
            )
            second = pool.sample_attack_records(
                "uav_03",
                label=["injection", "payload_manipulation"],
                attack_intensity=3,
                attack_replay_mode="loop",
            )

        self.assertEqual([row["attack_type"] for row in first], ["injection", "payload_manipulation", "injection"])
        self.assertEqual(
            [row["attack_type"] for row in second],
            ["payload_manipulation", "injection", "payload_manipulation"],
        )

    def test_pool_supports_random_mode(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            pool = AttackReplayPool(dataset_paths=build_dataset_paths(Path(tmp_dir)), seed=11)

            rows = pool.sample_attack_records(
                "uav_02",
                label=["replay", "fake_landing"],
                attack_intensity=4,
                attack_replay_mode="random",
            )

        self.assertEqual(len(rows), 4)
        self.assertTrue(all(row["dataset_name"] == "gcs_to_uav_updated" for row in rows))
        self.assertTrue(all(row["attack_type"] in {"replay", "fake_landing"} for row in rows))

    def test_pool_supports_explicit_dataset_override(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            pool = AttackReplayPool(dataset_paths=build_dataset_paths(Path(tmp_dir)), seed=17)

            rows = pool.sample_attack_records(
                "uav_01",
                dataset_name="gcs_to_uav_updated",
                attack_type="replay",
                attack_intensity=1,
                attack_replay_mode="sequential",
            )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["dataset_name"], "gcs_to_uav_updated")
        self.assertEqual(rows[0]["attack_source_dataset"], "gcs_to_uav_updated")
        self.assertEqual(rows[0]["attack_type"], "replay")
        self.assertEqual(rows[0]["source_record_id"], "g0")

    def test_pool_supports_extended_dataset_bindings_without_uav04(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            pool = AttackReplayPool(
                dataset_paths=build_extended_dataset_paths(Path(tmp_dir)),
                uav_dataset_bindings={"uav_05": "unsw_nb15", "uav_06": "ecu_ioft", "uav_07": "uavids"},
                seed=19,
            )

            unsw_rows = pool.sample_attack_records("uav_05", attack_type="scan", attack_intensity=1)
            ecu_rows = pool.sample_attack_records("uav_06", attack_type="command_injection", attack_intensity=1)
            uavids_rows = pool.sample_attack_records("uav_07", attack_type="command_injection", attack_intensity=1)

        self.assertEqual(unsw_rows[0]["attack_source_dataset"], "unsw_nb15")
        self.assertEqual(unsw_rows[0]["attack_type"], "recon_scanning")
        self.assertEqual(unsw_rows[0]["source_type"], "external_non_uav")
        self.assertEqual(unsw_rows[0]["simulation_role"], "external_ood")
        self.assertIn("not real UAV flight", str(unsw_rows[0]["source_note"]))
        self.assertEqual(ecu_rows[0]["attack_source_dataset"], "ecu_ioft")
        self.assertEqual(ecu_rows[0]["attack_type"], "unauthorized_udp")
        self.assertEqual(ecu_rows[0]["source_type"], "uav_iot_wifi")
        self.assertEqual(ecu_rows[0]["simulation_role"], "uav_replay")
        self.assertEqual(uavids_rows[0]["attack_source_dataset"], "uavids")
        self.assertEqual(uavids_rows[0]["attack_type"], "wormhole")
        self.assertEqual(uavids_rows[0]["source_type"], "uav")
        self.assertEqual(uavids_rows[0]["simulation_role"], "uav_replay")


class AttackReplayEngineIntegrationTests(unittest.TestCase):
    def test_engine_emits_replayed_attack_rows_with_simulation_context(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            pool = AttackReplayPool(dataset_paths=build_dataset_paths(Path(tmp_dir)), seed=13)
            uav = UAV(
                uav_id="uav_01",
                route_length_m=12.0,
                hover_duration_s=1.0,
                cruise_altitude_m=18.0,
                cruise_speed_mps=5.0,
            )
            engine = SimulationEngine(
                uavs=[uav],
                gcs=GCS(),
                attacker=Attacker(),
                attack_injector=AttackInjector(
                    [AttackEvent(start_s=0.0, end_s=2.0, attack_type="jamming_proxy", intensity=2.0)]
                ),
                attack_replay_pool=pool,
                config=SimulationConfig(
                    duration_s=2.0,
                    dt_s=1.0,
                    seed=13,
                    attack_replay_mode="loop",
                    records_per_uav_per_step=2,
                ),
            )

            result = engine.run()

        self.assertEqual(len(result.records), 4)
        self.assertEqual(result.attack_record_count, 4)
        self.assertTrue(all(record.record_kind == "attack_replay" for record in result.records))
        first = result.records[0].to_dict()
        self.assertEqual(first["dataset_name"], "uav_ndd")
        self.assertEqual(first["attack_source_dataset"], "uav_ndd")
        self.assertEqual(first["attack_type"], "jamming")
        self.assertEqual(first["sim_time"], 0.0)
        self.assertIn("raw_feature", first)
        self.assertIn("source_record_id", first)
        self.assertTrue(all(record.attack_active for record in result.records))
        self.assertEqual({record.timestamp for record in result.records[:2]}, {0.0})
        self.assertEqual({record.timestamp for record in result.records[2:]}, {1.0})

    def test_engine_evenly_distributes_target_records_per_uav(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            pool = AttackReplayPool(dataset_paths=build_dataset_paths(Path(tmp_dir)), seed=29)
            uav = UAV(
                uav_id="uav_01",
                route_length_m=12.0,
                hover_duration_s=1.0,
                cruise_altitude_m=18.0,
                cruise_speed_mps=5.0,
            )
            engine = SimulationEngine(
                uavs=[uav],
                gcs=GCS(),
                attacker=Attacker(),
                attack_injector=AttackInjector(),
                attack_replay_pool=pool,
                config=SimulationConfig(
                    duration_s=3.0,
                    dt_s=1.0,
                    seed=29,
                    attack_replay_mode="loop",
                    records_per_uav_per_step=99,
                    target_records_per_uav=5,
                ),
            )

            result = engine.run()

        counts_by_time: dict[float, int] = {}
        for record in result.records:
            counts_by_time[float(record.timestamp)] = counts_by_time.get(float(record.timestamp), 0) + 1

        self.assertEqual(len(result.records), 5)
        self.assertEqual(counts_by_time, {0.0: 2, 1.0: 2, 2.0: 1})
        self.assertTrue(all(not record.attack_active for record in result.records))


if __name__ == "__main__":
    unittest.main()
