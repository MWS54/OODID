from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "launch_dashboard.py"
SPEC = importlib.util.spec_from_file_location("launch_dashboard_script", MODULE_PATH)
launch_dashboard_script = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(launch_dashboard_script)


class DashboardLauncherTests(unittest.TestCase):
    def test_launcher_builds_streamlit_command(self):
        argv = [
            str(MODULE_PATH),
            "--port",
            "8601",
            "--address",
            "0.0.0.0",
            "--headless",
        ]
        with mock.patch.object(launch_dashboard_script.sys, "argv", argv):
            with mock.patch.object(launch_dashboard_script.subprocess, "call", return_value=0) as mock_call:
                with self.assertRaises(SystemExit) as exc:
                    launch_dashboard_script.main()

        self.assertEqual(exc.exception.code, 0)
        command = mock_call.call_args.args[0]
        self.assertEqual(command[:4], [sys.executable, "-m", "streamlit", "run"])
        self.assertIn(str(ROOT / "app" / "streamlit_app.py"), command)
        self.assertIn("--server.port", command)
        self.assertIn("8601", command)
        self.assertIn("--server.address", command)
        self.assertIn("0.0.0.0", command)
        self.assertIn("--server.headless", command)
        self.assertEqual(mock_call.call_args.kwargs["cwd"], str(ROOT))


if __name__ == "__main__":
    unittest.main()
