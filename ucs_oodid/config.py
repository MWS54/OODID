from __future__ import annotations

from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, List

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None


@dataclass
class WindowConfig:
    mode: str = "count"
    size: int = 16
    stride: int = 8
    time_seconds: float = 2.0
    adaptive_min_size: int = 8
    adaptive_max_size: int = 64
    adaptive_target_records: int = 16


@dataclass
class GraphConfig:
    k: int = 8
    tau: float = 0.5
    metric: str = "cosine"  # cosine | rbf | euclidean
    variant: str = "sym_weighted"  # Legacy graph option; transformer_only does not build behavior graphs.
    neighbor_strategy: str = "knn"  # knn | random


@dataclass
class ModelConfig:
    input_dim: int = 0
    num_classes: int = 0
    hidden_dim: int = 64
    num_heads: int = 2
    num_layers: int = 1
    gcn_layers: int = 2  # Legacy GCN depth kept for historical ablations.
    dropout: float = 0.1
    gate: str = "learned"  # learned | temporal_only | graph_only | mean
    encoder_ablation: str = "transformer_only"  # Current recommended mainline; full/gcn_only/random_graph stay for ablations.
    record_head: bool = False


@dataclass
class TrainingConfig:
    epochs: int = 30
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-4
    lambda_record: float = 0.0
    lambda_reg: float = 0.0


@dataclass
class CalibrationConfig:
    q_ood: float = 0.90
    bank_k: int = 5
    fusion: str = "correlation_aware"  # Default onboard fusion; mean remains a simpler supported alternative.
    temperature_grid: tuple[float, ...] = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 5.0)


@dataclass
class ProjectConfig:
    seed: int = 42
    window: WindowConfig = field(default_factory=WindowConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _update_dataclass(obj, values: Dict[str, Any]):
    for k, v in values.items():
        if hasattr(obj, k):
            setattr(obj, k, v)
    return obj


def load_config(path: str | Path | None = None) -> ProjectConfig:
    cfg = ProjectConfig()
    if path is None:
        return cfg
    if yaml is None:
        raise RuntimeError("pyyaml is required to load YAML configs")
    with Path(path).open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if "seed" in raw:
        cfg.seed = int(raw["seed"])
    if "window" in raw:
        _update_dataclass(cfg.window, raw["window"])
    if "graph" in raw:
        _update_dataclass(cfg.graph, raw["graph"])
    if "model" in raw:
        _update_dataclass(cfg.model, raw["model"])
    if "training" in raw:
        _update_dataclass(cfg.training, raw["training"])
    if "calibration" in raw:
        _update_dataclass(cfg.calibration, raw["calibration"])
    return cfg
