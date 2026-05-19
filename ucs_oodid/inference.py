from __future__ import annotations

from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import WindowTorchDataset
from .graph import build_behavior_graph
from .windowing import WindowedData


@torch.no_grad()
def collect_model_outputs(model, windows: WindowedData, graph_cfg: dict, batch_size: int = 256, device: str = "cpu", temperature: float = 1.0, show_progress: bool = False) -> Dict[str, np.ndarray]:
    ds = WindowTorchDataset(windows)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    model.eval()
    logits, probs, embs, attns, masks = [], [], [], [], []
    for batch in tqdm(loader, disable=not show_progress, desc="inference"):
        x = batch["x"].to(device)
        mask = batch.get("mask")
        mask = mask.to(device) if mask is not None else None
        group_index = batch.get("group_index")
        group_index = group_index.to(device) if group_index is not None else None
        graph = None
        if getattr(model, "uses_graph_encoder", True):
            graph = build_behavior_graph(
                x,
                k=graph_cfg.get("k", 8),
                tau=graph_cfg.get("tau", 0.5),
                metric=graph_cfg.get("metric", "cosine"),
                variant=graph_cfg.get("variant", "sym_weighted"),
                mask=mask,
            ).adj
        out = model(x, graph, temperature=temperature, mask=mask, group_index=group_index)
        logits.append(out["logits"].cpu().numpy())
        probs.append(out["probs"].cpu().numpy())
        embs.append(out["embedding"].cpu().numpy())
        attns.append(out["attention"].cpu().numpy())
        masks.append(mask.cpu().numpy() if mask is not None else np.ones(x.shape[:2], dtype=bool))
    if len(logits) == 0:
        c = windows.y.shape[1]
        emb_dim = int(getattr(model, "embedding_dim", model.hidden_dim))
        return {"logits": np.zeros((0, c)), "probs": np.zeros((0, c)), "embeddings": np.zeros((0, emb_dim)), "attention": np.zeros((0, windows.x.shape[1])), "mask": np.zeros((0, windows.x.shape[1]), dtype=bool)}
    return {
        "logits": np.concatenate(logits, axis=0),
        "probs": np.concatenate(probs, axis=0),
        "embeddings": np.concatenate(embs, axis=0),
        "attention": np.concatenate(attns, axis=0),
        "mask": np.concatenate(masks, axis=0).astype(bool),
    }
