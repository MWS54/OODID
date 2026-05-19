from __future__ import annotations

import unittest
from unittest import mock

import numpy as np
import torch

from ucs_oodid.graph import build_behavior_graph
from ucs_oodid.inference import collect_model_outputs
from ucs_oodid.model import UCSOODID, canonicalize_encoder_ablation
from ucs_oodid.windowing import WindowedData


class EncoderAblationTests(unittest.TestCase):
    def test_canonicalize_encoder_ablation_accepts_new_and_legacy_names(self):
        self.assertEqual(canonicalize_encoder_ablation("full"), "full")
        self.assertEqual(canonicalize_encoder_ablation("transformer_only"), "transformer_only")
        self.assertEqual(canonicalize_encoder_ablation("gcn_only"), "gcn_only")
        self.assertEqual(canonicalize_encoder_ablation("mlp_only"), "mlp_only")
        self.assertEqual(canonicalize_encoder_ablation("random_graph"), "random_graph")
        self.assertEqual(canonicalize_encoder_ablation("temporal_only"), "transformer_only")
        self.assertEqual(canonicalize_encoder_ablation("graph_only"), "gcn_only")
        self.assertEqual(canonicalize_encoder_ablation("identity_graph"), "full")
        with self.assertRaises(ValueError):
            canonicalize_encoder_ablation("unsupported_mode")

    def test_transformer_only_bypasses_learned_gate_and_skips_graph(self):
        x = torch.randn(2, 3, 4)
        mask = torch.tensor([[True, True, True], [True, True, False]])
        model = UCSOODID(
            input_dim=4,
            num_classes=2,
            hidden_dim=8,
            num_heads=2,
            num_layers=1,
            gcn_layers=1,
            dropout=0.0,
            gate="learned",
            encoder_ablation="transformer_only",
        )

        out = model(x, mask=mask)

        self.assertEqual(tuple(out["logits"].shape), (2, 2))
        self.assertTrue(torch.allclose(out["h_graph"], torch.zeros_like(out["h_graph"])))
        self.assertTrue(torch.allclose(out["gate"], torch.ones_like(out["gate"])))

    def test_transformer_only_collect_outputs_does_not_call_build_behavior_graph(self):
        windows = WindowedData(
            x=np.random.randn(2, 3, 4).astype(np.float32),
            y=np.zeros((2, 2), dtype=np.float32),
            record_indices=np.asarray([[0, 1, 2], [3, 4, 5]], dtype=np.int64),
            record_ids=np.asarray([["r0", "r1", "r2"], ["r3", "r4", "r5"]], dtype=object),
            ood=np.zeros(2, dtype=bool),
            valid_mask=np.asarray([[True, True, True], [True, True, False]], dtype=bool),
        )
        model = UCSOODID(
            input_dim=4,
            num_classes=2,
            hidden_dim=8,
            num_heads=2,
            num_layers=1,
            gcn_layers=1,
            dropout=0.0,
            gate="learned",
            encoder_ablation="transformer_only",
        )

        with mock.patch("ucs_oodid.inference.build_behavior_graph", side_effect=AssertionError("build_behavior_graph should not be called")):
            outs = collect_model_outputs(
                model,
                windows,
                {"k": 1, "tau": 0.5, "metric": "cosine", "variant": "sym_weighted"},
                batch_size=1,
                device="cpu",
            )

        self.assertEqual(tuple(outs["logits"].shape), (2, 2))

    def test_gcn_only_bypasses_learned_gate_and_uses_graph(self):
        x = torch.randn(2, 3, 4)
        mask = torch.tensor([[True, True, True], [True, True, False]])
        adj = build_behavior_graph(x, k=1, tau=0.5, metric="cosine", variant="sym_weighted", mask=mask).adj
        model = UCSOODID(
            input_dim=4,
            num_classes=2,
            hidden_dim=8,
            num_heads=2,
            num_layers=1,
            gcn_layers=1,
            dropout=0.0,
            gate="learned",
            encoder_ablation="gcn_only",
        )

        out = model(x, adj=adj, mask=mask)

        self.assertEqual(tuple(out["logits"].shape), (2, 2))
        self.assertTrue(torch.allclose(out["h_temporal"], torch.zeros_like(out["h_temporal"])))
        self.assertTrue(torch.allclose(out["gate"], torch.zeros_like(out["gate"])))

    def test_mlp_only_uses_uniform_attention_without_graph_inputs(self):
        x = torch.tensor(
            [
                [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
                [[1.0, 1.1], [1.2, 1.3], [1.4, 1.5]],
            ],
            dtype=torch.float32,
        )
        mask = torch.tensor([[True, True, True], [True, False, False]])
        model = UCSOODID(
            input_dim=2,
            num_classes=2,
            hidden_dim=8,
            num_heads=2,
            num_layers=1,
            gcn_layers=1,
            dropout=0.0,
            gate="learned",
            encoder_ablation="mlp_only",
        )

        out = model(x, mask=mask)

        expected_attention = torch.tensor([[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], [1.0, 0.0, 0.0]], dtype=torch.float32)
        self.assertEqual(tuple(out["logits"].shape), (2, 2))
        torch.testing.assert_close(out["attention"], expected_attention)
        self.assertIsNone(out["gate"])
        self.assertTrue(torch.allclose(out["h_temporal"], torch.zeros_like(out["h_temporal"])))
        self.assertTrue(torch.allclose(out["h_graph"], torch.zeros_like(out["h_graph"])))

    def test_random_neighbor_strategy_differs_from_knn_graph(self):
        torch.manual_seed(7)
        x = torch.randn(2, 4, 3)
        mask = torch.tensor([[True, True, True, True], [True, True, True, False]])

        knn_graph = build_behavior_graph(
            x,
            k=2,
            tau=0.5,
            metric="cosine",
            variant="sym_weighted",
            mask=mask,
            neighbor_strategy="knn",
        )
        torch.manual_seed(7)
        random_graph = build_behavior_graph(
            x,
            k=2,
            tau=0.5,
            metric="cosine",
            variant="sym_weighted",
            mask=mask,
            neighbor_strategy="random",
        )

        self.assertEqual(tuple(knn_graph.adj.shape), tuple(random_graph.adj.shape))
        self.assertTrue(torch.isfinite(random_graph.adj).all())
        self.assertGreater(float(random_graph.raw_affinity.sum().item()), 0.0)
        self.assertFalse(torch.allclose(knn_graph.raw_affinity, random_graph.raw_affinity))


if __name__ == "__main__":
    unittest.main()
