#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP_PATH = ROOT / "app" / "streamlit_app.py"


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch the Streamlit dashboard.")
    parser.add_argument("--port", type=int, default=8501, help="Port used by Streamlit.")
    parser.add_argument("--address", default="127.0.0.1", help="Bind address for the Streamlit server.")
    parser.add_argument("--headless", action="store_true", help="Run without opening a browser window.")
    args = parser.parse_args()

    if not APP_PATH.exists():
        raise FileNotFoundError(f"Dashboard entrypoint not found: {APP_PATH}")

    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(APP_PATH),
        "--server.port",
        str(args.port),
        "--server.address",
        str(args.address),
    ]
    if args.headless:
        command.extend(["--server.headless", "true"])

    raise SystemExit(subprocess.call(command, cwd=str(ROOT)))


if __name__ == "__main__":
    main()
