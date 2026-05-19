from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import torch


def save_artifact(path: str | Path, artifact: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, path)


def load_artifact(path: str | Path, map_location: str = "cpu") -> Dict[str, Any]:
    # PyTorch >=2.6 may default to weights_only=True; this artifact intentionally
    # stores preprocessing/calibration objects, so full loading is required.
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)
