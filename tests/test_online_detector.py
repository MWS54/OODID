from __future__ import annotations

import unittest

import numpy as np
import torch

from ucs_oodid.inference import collect_model_outputs
from ucs_oodid.model import UCSOODID
from ucs_oodid.ood import OODCalibrator, PrototypeBank, compute_raw_ood_scores
from ucs_oodid.preprocessing import MetadataPreprocessor
from ucs_oodid.realtime.online_detector import OnlineDetector
from ucs_oodid.windowing import attach_parsed_labels, build_grouped_windows

FEATURE_COLS = [
    "flow_duration_ms",
    "packet_count",
    "byte_rate",
    "dst_port_entropy",
]


def make_row(idx: int, **overrides) -> dict[str, object]:
    row = {
        "record_id": f"u1:{idx}",
        "timestamp": float(idx),
        "uav_id": "u1",
        "label": "benign",
        "attack_active": False,
        "attack_type": "benign",
        "flow_duration_ms": 8.0 + idx * 0.2,
        "packet_count": 32.0 + idx,
        "byte_rate": 2400.0 + idx * 10.0,
        "dst_port_entropy": 0.30 + idx * 0.01,
        "battery_soc": 92.0 - idx * 0.2,
        "speed": 14.0,
        "altitude": 80.0,
        "rssi": -58.0,
        "snr": 24.0,
        "latency_ms": 18.0,
        "loss_rate": 0.01,
        "bytes_up": 2600.0,
        "bytes_down": 2400.0,
    }
    row.update(overrides)
    return row


def initialize_identity_model(model: UCSOODID) -> None:
    hidden_dim = int(model.hidden_dim)
    input_dim = int(model.input_dim)
    diag_dim = min(hidden_dim, input_dim)
    with torch.no_grad():
        model.input_proj.weight.zero_()
        model.input_proj.bias.zero_()
        model.input_proj.weight[:diag_dim, :diag_dim] = torch.eye(diag_dim)
        model.fused_norm.weight.fill_(1.0)
        model.fused_norm.bias.zero_()
        model.mlp_pool_proj.weight.zero_()
        model.mlp_pool_proj.bias.zero_()
        model.mlp_pool_proj.weight[:diag_dim, :diag_dim] = torch.eye(diag_dim)
        model.mlp_pool_norm.weight.fill_(1.0)
        model.mlp_pool_norm.bias.zero_()
        model.classifier.weight.zero_()
        model.classifier.bias.fill_(2.0)


def build_test_artifact() -> dict:
    benign_rows = [make_row(idx) for idx in range(10)]
    import pandas as pd

    df = pd.DataFrame(benign_rows)
    pre = MetadataPreprocessor(label_col="label", timestamp_col="timestamp", record_id_col="record_id", group_col="uav_id")
    pre.fit(df, feature_cols=FEATURE_COLS)
    work = attach_parsed_labels(df, "label")
    features = pre.transform(work)
    class_names = ["benign"]
    class_to_idx = {"benign": 0}
    windows = build_grouped_windows(
        features,
        work,
        class_to_idx,
        group_col="uav_id",
        mode="count",
        timestamp_col="timestamp",
        label_col="label",
        record_id_col="record_id",
        window_size=4,
        stride=1,
    )

    model_cfg = {
        "input_dim": len(FEATURE_COLS),
        "num_classes": 1,
        "hidden_dim": len(FEATURE_COLS),
        "num_heads": 1,
        "num_layers": 1,
        "gcn_layers": 1,
        "dropout": 0.0,
        "gate": "mean",
        "encoder_ablation": "mlp_only",
        "record_head": False,
        "max_window_size": 4,
        "num_groups": None,
        "use_group_embedding": False,
        "group_embedding_dim": 16,
        "unknown_group_index": None,
    }
    model = UCSOODID(**model_cfg)
    initialize_identity_model(model)
    model.eval()
    outputs = collect_model_outputs(model, windows, graph_cfg={}, batch_size=8, device="cpu", temperature=1.0)
    bank = PrototypeBank.fit(outputs["embeddings"], windows.y, class_names)
    raw_scores = compute_raw_ood_scores(
        outputs["logits"],
        outputs["probs"],
        outputs["embeddings"],
        bank,
        temperature=1.0,
        k_bank=5,
    )
    ood_cal = OODCalibrator(fusion="proto", q_ood=0.8)
    ood_cal.fit(raw_scores)
    return {
        "model_state": model.state_dict(),
        "model_config": model_cfg,
        "graph_config": {},
        "run_config": {"seed": 7},
        "seed": 7,
        "normalization_mode": "global",
            "strict_model_feature_mode": True,
        "group_col": "uav_id",
        "feature_cols": list(pre.feature_cols),
        "feature_medians": dict(pre.feature_medians),
        "global_scaler": pre.scaler,
        "group_scalers": {},
        "group_normalization_fallbacks": {},
        "use_group_embedding": False,
        "group_embedding_dim": 16,
        "group_to_index": {},
        "unknown_group_index": None,
        "ood_threshold_mode": "global",
        "global_ood_threshold": float(ood_cal.threshold),
        "window_config": {
            "mode": "count",
            "size": 4,
            "stride": 1,
            "time_seconds": 1.0,
            "adaptive_min_size": 4,
            "adaptive_max_size": 4,
        },
        "group_config": {"group_col": "uav_id"},
        "class_names": class_names,
        "class_to_idx": class_to_idx,
        "preprocessor": pre,
        "temperature": 1.0,
        "class_thresholds": np.asarray([0.5], dtype=np.float32),
        "prototype_bank": bank.to_dict(),
        "ood_calibrator": ood_cal.to_dict(),
        "calibration_config": {"q_ood": 0.8, "bank_k": 5, "fusion": "proto", "ood_threshold_mode": "global"},
        "leakage_report": pre.leakage_report(df),
    }


class OnlineDetectorTests(unittest.TestCase):
    def test_online_detector_raises_scores_and_emits_alerts_after_attack(self):
        detector = OnlineDetector(build_test_artifact(), top_records=3)
        benign_outputs = []
        for idx in range(6):
            benign_outputs.extend(detector.consume_record(make_row(idx)))

        attack_rows = [
            make_row(
                6,
                label="jamming_proxy",
                attack_active=True,
                attack_type="jamming_proxy",
                flow_duration_ms=32.0,
                packet_count=180.0,
                byte_rate=9100.0,
                dst_port_entropy=0.92,
                rssi=-92.0,
                snr=3.0,
                latency_ms=145.0,
                loss_rate=0.32,
            ),
            make_row(
                7,
                label="jamming_proxy",
                attack_active=True,
                attack_type="jamming_proxy",
                flow_duration_ms=35.0,
                packet_count=220.0,
                byte_rate=10800.0,
                dst_port_entropy=0.97,
                rssi=-95.0,
                snr=2.0,
                latency_ms=180.0,
                loss_rate=0.40,
            ),
        ]
        attack_outputs = []
        for row in attack_rows:
            attack_outputs.extend(detector.consume_record(row))

        self.assertTrue(benign_outputs)
        self.assertTrue(attack_outputs)
        max_benign = max(row["ood_score"] for row in benign_outputs)
        max_attack = max(row["ood_score"] for row in attack_outputs)
        self.assertGreater(max_attack, max_benign)
        self.assertTrue(any(row["is_ood"] for row in attack_outputs))
        self.assertEqual(detector.model_input_columns, FEATURE_COLS)
        sample = attack_outputs[-1]
        self.assertIn("known_pred_labels", sample)
        self.assertIn("known_pred_probs", sample)
        self.assertIn("ood_score", sample)
        self.assertIn("raw_ood_score", sample)
        self.assertIn("normalized_ood_score", sample)
        self.assertIn("ood_threshold", sample)
        self.assertIn("threshold_source", sample)
        self.assertIn("score_direction", sample)
        self.assertIn("is_ood", sample)
        self.assertIn("alert_level", sample)
        self.assertIn("top_suspicious_records", sample)
        self.assertEqual(sample["ground_truth"]["attack_types"], ["jamming_proxy"])
        self.assertFalse(bool(sample["false_alert"]))
        self.assertEqual(sample["score_mode"], "normalized")
        self.assertEqual(sample["threshold_source"], "normalized_static")
        self.assertEqual(sample["ood_threshold"], sample["normalized_threshold"])
        self.assertIsNotNone(sample["raw_threshold"])
        self.assertNotEqual(sample["raw_ood_score"], sample["normalized_ood_score"])

    def test_online_detector_autogenerates_record_ids_and_ignores_artifact_group_thresholds_without_benign_warmup(self):
        artifact = dict(build_test_artifact())
        artifact["ood_threshold_mode"] = "group"
        artifact["global_ood_threshold"] = 99.0
        artifact["group_ood_thresholds"] = {"u1": -1.0}
        artifact["group_threshold_sources"] = {"u1": "group_test_override"}
        detector = OnlineDetector(artifact, top_records=2)

        outputs = []
        for idx in range(4):
            row = make_row(idx)
            row.pop("record_id")
            row["label"] = "scan"
            row["attack_active"] = True
            row["attack_type"] = "scan"
            outputs.extend(detector.consume_record(row))

        self.assertEqual(len(outputs), 1)
        sample = outputs[0]
        self.assertEqual(sample["group_id"], "u1")
        self.assertEqual(sample["score_mode"], "normalized")
        self.assertEqual(sample["ood_threshold"], detector.normalized_threshold)
        self.assertEqual(sample["normalized_threshold"], detector.normalized_threshold)
        self.assertIsNone(sample["raw_threshold"])
        self.assertEqual(sample["threshold_source"], "normalized_static_no_warmup")
        self.assertEqual(sample["ood_threshold_source"], "normalized_static_no_warmup")
        self.assertEqual(sample["window_valid_count"], 4)
        self.assertEqual(len(sample["record_ids"]), 4)
        self.assertTrue(all(record_id.startswith("u1:") for record_id in sample["record_ids"]))
        self.assertEqual(sample["timestamp_range"], {"start": 0.0, "end": 3.0})

    def test_online_detector_marks_degenerate_windows_as_inference_errors(self):
        detector = OnlineDetector(build_test_artifact(), top_records=2)

        outputs = []
        for idx in range(4):
            row = make_row(idx, uav_id="u2")
            for feature_col in FEATURE_COLS:
                row.pop(feature_col, None)
            outputs.extend(detector.consume_record(row))

        self.assertEqual(len(outputs), 1)
        sample = outputs[0]
        self.assertTrue(bool(sample["inference_error"]))
        self.assertEqual(sample["inference_error_reason"], "degenerate_model_input")
        self.assertIsNone(sample["raw_ood_score"])
        self.assertIsNone(sample["normalized_ood_score"])
        self.assertFalse(bool(sample["has_alert"]))
        self.assertFalse(bool(sample["is_ood"]))

        diagnostics = detector.simulation_diagnostics()
        self.assertEqual(diagnostics["inference_error_count"], 1)
        self.assertEqual(diagnostics["fallback_score_used_count"], 0)
        self.assertEqual(diagnostics["per_uav_inference_error_count"]["u2"], 1)
        self.assertEqual(diagnostics["per_uav_score_unique_count"]["u2"], 0)
        self.assertAlmostEqual(diagnostics["per_uav_missing_feature_ratio"]["u2"], 1.0)
        self.assertGreaterEqual(diagnostics["per_uav_zero_feature_ratio"]["u2"], 1.0)
        self.assertIsNone(diagnostics["per_window_raw_ood_score"][0]["raw_ood_score"])
        self.assertIsNone(diagnostics["per_window_normalized_ood_score"][0]["normalized_ood_score"])


if __name__ == "__main__":
    unittest.main()
