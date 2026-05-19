from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ucs_oodid.artifacts import load_artifact, save_artifact
from ucs_oodid.dataset import WindowTorchDataset
from ucs_oodid.model import UCSOODID
from ucs_oodid.windowing import WindowedData

DETECT_MODULE_PATH = ROOT / "scripts" / "detect.py"
SPEC = importlib.util.spec_from_file_location("detect_script_group_embedding", DETECT_MODULE_PATH)
detect_script = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(detect_script)


def make_windows() -> WindowedData:
    return WindowedData(
        x=np.asarray(
            [
                [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
                [[1.1, 1.2], [1.3, 1.4], [1.5, 1.6]],
                [[2.1, 2.2], [2.3, 2.4], [2.5, 2.6]],
                [[3.1, 3.2], [3.3, 3.4], [3.5, 3.6]],
            ],
            dtype=np.float32,
        ),
        y=np.asarray(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
                [0.0, 0.0],
            ],
            dtype=np.float32,
        ),
        record_indices=np.asarray([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]], dtype=np.int64),
        record_ids=np.asarray(
            [
                ["r0", "r1", "r2"],
                ["r3", "r4", "r5"],
                ["r6", "r7", "r8"],
                ["r9", "r10", "r11"],
            ],
            dtype=object,
        ),
        ood=np.zeros(4, dtype=bool),
        valid_mask=np.ones((4, 3), dtype=bool),
        group_ids=np.asarray(["uav_01", "uav_02", "uav_01", "uav_02"], dtype=object),
        group_index=np.asarray([0, 1, 0, 1], dtype=np.int64),
    )


class GroupEmbeddingTests(unittest.TestCase):
    def test_disabled_group_embedding_keeps_model_outputs_unchanged(self):
        x = torch.randn(2, 3, 4)
        mask = torch.ones((2, 3), dtype=torch.bool)

        torch.manual_seed(7)
        model_default = UCSOODID(
            input_dim=4,
            num_classes=2,
            hidden_dim=8,
            num_heads=2,
            num_layers=1,
            gcn_layers=1,
            encoder_ablation="temporal_only",
            dropout=0.0,
        )
        torch.manual_seed(7)
        model_disabled = UCSOODID(
            input_dim=4,
            num_classes=2,
            hidden_dim=8,
            num_heads=2,
            num_layers=1,
            gcn_layers=1,
            encoder_ablation="temporal_only",
            dropout=0.0,
            use_group_embedding=False,
            group_embedding_dim=16,
        )

        out_default = model_default(x, mask=mask)
        out_disabled = model_disabled(x, mask=mask)

        torch.testing.assert_close(out_default["embedding"], out_disabled["embedding"])
        torch.testing.assert_close(out_default["logits"], out_disabled["logits"])
        self.assertEqual(model_default.embedding_dim, model_disabled.embedding_dim)

    def test_enabled_group_embedding_supports_batched_forward(self):
        windows = make_windows()
        loader = DataLoader(WindowTorchDataset(windows), batch_size=2, shuffle=False)
        batch = next(iter(loader))

        model = UCSOODID(
            input_dim=2,
            num_classes=2,
            hidden_dim=8,
            num_heads=2,
            num_layers=1,
            gcn_layers=1,
            encoder_ablation="temporal_only",
            dropout=0.0,
            use_group_embedding=True,
            num_groups=3,
            group_embedding_dim=4,
            unknown_group_index=2,
        )
        out = model(batch["x"], mask=batch["mask"], group_index=batch["group_index"])

        self.assertEqual(tuple(out["logits"].shape), (2, 2))
        self.assertEqual(tuple(out["embedding"].shape), (2, 12))
        self.assertEqual(tuple(out["group_embedding"].shape), (2, 4))

    def test_artifact_round_trip_preserves_group_to_index(self):
        artifact = {
            "use_group_embedding": True,
            "group_embedding_dim": 16,
            "group_to_index": {"uav_01": 0, "uav_02": 1, "__unknown__": 2},
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "artifact.pt"
            save_artifact(path, artifact)
            loaded = load_artifact(path)

        self.assertTrue(loaded["use_group_embedding"])
        self.assertEqual(loaded["group_embedding_dim"], 16)
        self.assertEqual(loaded["group_to_index"]["uav_02"], 1)
        self.assertEqual(loaded["group_to_index"]["__unknown__"], 2)

    def test_detect_unknown_group_uses_unknown_index(self):
        windows = WindowedData(
            x=np.asarray([[[0.1, 0.2], [0.3, 0.4]]], dtype=np.float32),
            y=np.asarray([[1.0, 0.0]], dtype=np.float32),
            record_indices=np.asarray([[0, 1]], dtype=np.int64),
            record_ids=np.asarray([["r0", "r1"]], dtype=object),
            ood=np.zeros(1, dtype=bool),
            valid_mask=np.ones((1, 2), dtype=bool),
            group_ids=np.asarray(["uav_new"], dtype=object),
        )
        artifact = {
            "use_group_embedding": True,
            "group_embedding_dim": 4,
            "group_to_index": {"uav_01": 0, "uav_02": 1, "__unknown__": 2},
            "unknown_group_index": 2,
        }

        config = detect_script.attach_group_embedding_indices(windows, artifact)

        self.assertTrue(config["enabled"])
        self.assertEqual(int(windows.group_index[0]), 2)
        model = UCSOODID(
            input_dim=2,
            num_classes=2,
            hidden_dim=8,
            num_heads=2,
            num_layers=1,
            gcn_layers=1,
            encoder_ablation="temporal_only",
            dropout=0.0,
            use_group_embedding=True,
            num_groups=3,
            group_embedding_dim=4,
            unknown_group_index=2,
        )
        out = model(
            torch.tensor(windows.x, dtype=torch.float32),
            mask=torch.tensor(windows.valid_mask, dtype=torch.bool),
            group_index=torch.tensor(windows.group_index, dtype=torch.long),
        )
        self.assertEqual(tuple(out["embedding"].shape), (1, 12))


if __name__ == "__main__":
    unittest.main()
