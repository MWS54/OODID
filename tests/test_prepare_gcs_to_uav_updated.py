from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "prepare_gcs_to_uav_updated.py"
SPEC = importlib.util.spec_from_file_location("prepare_gcs_to_uav_updated", MODULE_PATH)
prepare_gcs_to_uav_updated = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(prepare_gcs_to_uav_updated)


class PrepareGcsToUavUpdatedTests(unittest.TestCase):
    def test_reply_label_normalises_to_replay(self):
        self.assertEqual(prepare_gcs_to_uav_updated.normalise_label("Reply"), "replay")
        self.assertEqual(
            prepare_gcs_to_uav_updated.DEFAULT_OOD_CLASSES,
            ["replay", "fake_landing", "evil"],
        )


if __name__ == "__main__":
    unittest.main()
