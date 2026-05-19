from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "simulate_live_demo.py"
CONFIG = ROOT / "configs" / "demo_scene_default.yaml"


def run_command(cmd: list[object]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(part) for part in cmd],
        cwd=str(ROOT),
        check=True,
        capture_output=True,
        text=True,
    )


class DemoSmokeTests(unittest.TestCase):
    def test_default_demo_scene_runs_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_json = Path(tmp_dir) / "demo_payload.json"
            completed = run_command(
                [
                    sys.executable,
                    SCRIPT,
                    "--config",
                    CONFIG,
                    "--duration_s",
                    "24",
                    "--bootstrap_duration_s",
                    "18",
                    "--output_json",
                    output_json,
                    "--head",
                    "2",
                ]
            )

            self.assertTrue(output_json.exists(), msg=completed.stdout + completed.stderr)
            payload = json.loads(output_json.read_text(encoding="utf-8"))

        self.assertTrue(payload["online_detection_enabled"])
        self.assertEqual(payload["record_count"], 24 * 6)
        self.assertGreater(payload["attack_record_count"], 0)
        self.assertGreater(payload["trace_summary"]["windows"], 0)
        self.assertGreaterEqual(payload["trace_summary"]["max_post_attack_ood_score"], 0.0)
        self.assertIsInstance(payload["ood_trace"], list)
        self.assertIsInstance(payload["response_events"], list)


if __name__ == "__main__":
    unittest.main()
