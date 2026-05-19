from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "prepare_isot_drone.py"
SPEC = importlib.util.spec_from_file_location("prepare_isot_drone", MODULE_PATH)
prepare_isot_drone = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(prepare_isot_drone)


def make_rows(count: int, offset: int) -> list[dict]:
    rows: list[dict] = []
    for idx in range(count):
        value = offset + idx
        rows.append(
            {
                "ts": 1716000000 + value,
                "Payload_Length": 100.0 + value,
                "Var_Payload": 10.0 + value,
                "Duration": float((idx % 5) + 1),
                "Entropy": float((idx % 7) + 0.5),
                "Drone_port": 8889 if offset % 2 == 0 else 62514,
                "DS status": float(idx % 3),
                "Sequence number": float(value),
                "flow_active_time": float((idx % 11) + 0.25),
                "Protocol Type": 17.0,
            }
        )
    return rows


class PrepareIsotDroneTests(unittest.TestCase):
    def test_prepare_isot_drone_maps_folders_and_downsamples(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_root = root / "isot_root"
            output_path = root / "prepared.csv"
            notes_path = root / "prepared_notes.json"

            folder_rows = {
                "Regular": make_rows(20, 0),
                "DoS": make_rows(20, 100),
                "Replay": make_rows(8, 200),
                "Video": make_rows(8, 300),
            }
            for folder_name, rows in folder_rows.items():
                folder = input_root / folder_name
                folder.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(rows).to_csv(folder / f"{folder_name.lower()}.csv", index=False)

            prepared, summary = prepare_isot_drone.prepare_isot_drone_dataset(
                input_root=str(input_root),
                output=str(output_path),
                notes_json=str(notes_path),
                target_rows=24,
                seed=7,
                train_ratio=0.70,
                val_ratio=0.15,
                oversample_factor=1.0,
                id_classes=["benign", "dos"],
                ood_classes=["replay", "video_interception"],
                keep_conflicting_patterns=True,
            )

            self.assertTrue(output_path.exists())
            self.assertTrue(notes_path.exists())
            self.assertEqual(len(prepared), 24)
            self.assertEqual(summary["target_rows_requested"], 24)
            self.assertEqual(summary["raw_file_counts_by_folder"], {"DoS": 1, "Regular": 1, "Replay": 1, "Video": 1})
            self.assertEqual(set(prepared["label"].unique().tolist()), {"benign", "dos", "replay", "video_interception"})
            self.assertEqual(set(prepared["source_label"].unique().tolist()), {"Regular", "DoS", "Replay", "Video"})
            self.assertEqual(
                set(prepared["split"].unique().tolist()),
                {"train", "val", "test_id", "test_ood"},
            )
            self.assertEqual(prepared["record_id"].tolist(), list(range(24)))
            self.assertEqual(prepared["timestamp"].tolist(), [float(i) for i in range(24)])

            self.assertIn("duration", prepared.columns)
            self.assertIn("entropy", prepared.columns)
            self.assertIn("flow_active_time", prepared.columns)
            self.assertIn("protocol_type", prepared.columns)
            self.assertNotIn("ts", prepared.columns)
            self.assertNotIn("payload_length", prepared.columns)
            self.assertNotIn("var_payload", prepared.columns)
            self.assertNotIn("drone_port", prepared.columns)
            self.assertNotIn("ds_status", prepared.columns)
            self.assertNotIn("sequence_number", prepared.columns)

            notes = json.loads(notes_path.read_text(encoding="utf-8"))
            self.assertEqual(notes["output_rows"], 24)
            self.assertEqual(notes["final_selected_rows_by_label"], {"benign": 9, "dos": 9, "replay": 3, "video_interception": 3})


if __name__ == "__main__":
    unittest.main()
