from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F


@dataclass
class GraphBuildResult:
    adj: torch.Tensor
    raw_affinity: torch.Tensor


def canonicalize_graph_variant(variant: str) -> str:
    if variant == "no_graph":
        return "identity_graph"
    return variant


def canonicalize_neighbor_strategy(strategy: str | None) -> str:
    mode = str(strategy or "knn").strip().lower()
    if mode not in {"knn", "random"}:
        raise ValueError(f"Unsupported graph neighbor strategy: {strategy}")
    return mode


def build_behavior_graph(
    x: torch.Tensor,
    k: int = 8,
    tau: float = 0.5,
    metric: str = "cosine",
    variant: str = "sym_weighted",
    mask: Optional[torch.Tensor] = None,
    neighbor_strategy: str = "knn",
    eps: float = 1e-8,
) -> GraphBuildResult:
    """Build normalized in-window kNN behavioral graph with optional padding mask.

    x: [B, N, F]. mask: [B, N], where False nodes are padding and cannot be selected as neighbors.
    """
    if x.dim() != 3:
        raise ValueError("x must be [B, N, F]")
    b, n, _ = x.shape
    device = x.device
    variant = canonicalize_graph_variant(variant)
    neighbor_strategy = canonicalize_neighbor_strategy(neighbor_strategy)
    eye = torch.eye(n, device=device).unsqueeze(0).expand(b, n, n)
    if mask is None:
        mask = torch.ones((b, n), dtype=torch.bool, device=device)
    else:
        mask = mask.to(device=device, dtype=torch.bool)
    valid_pair = mask.unsqueeze(1) & mask.unsqueeze(2)

    if variant == "identity_graph" or n <= 1:
        adj = eye * mask.unsqueeze(1).float() * mask.unsqueeze(2).float() + eye * (~mask).unsqueeze(-1).float()
        return GraphBuildResult(adj=adj, raw_affinity=adj)
    k_eff = max(1, min(k, n - 1))
    x_for_sim = torch.where(mask.unsqueeze(-1), x, torch.zeros_like(x))

    if metric == "cosine":
        xn = F.normalize(x_for_sim, dim=-1, eps=eps)
        sim = torch.bmm(xn, xn.transpose(1, 2))
        affinity = torch.exp(sim / max(tau, eps))
    elif metric in {"rbf", "euclidean"}:
        dist2 = torch.cdist(x_for_sim, x_for_sim, p=2).pow(2)
        sim = -dist2
        affinity = torch.exp((-dist2 if metric == "rbf" else sim) / max(tau, eps))
    else:
        raise ValueError(f"Unsupported graph metric: {metric}")

    affinity = affinity.masked_fill(eye.bool() | (~valid_pair), float("-inf"))
    if neighbor_strategy == "random":
        random_scores = torch.rand((b, n, n), device=device)
        random_scores = random_scores.masked_fill(eye.bool() | (~valid_pair), float("-inf"))
        _, nn_idx = torch.topk(random_scores, k=k_eff, dim=-1)
    else:
        _, nn_idx = torch.topk(affinity, k=k_eff, dim=-1)
    knn_mask = torch.zeros((b, n, n), device=device, dtype=torch.bool)
    knn_mask.scatter_(2, nn_idx, True)
    knn_mask = knn_mask & valid_pair
    weights = torch.where(knn_mask, affinity, torch.zeros_like(affinity))
    weights = torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)

    if variant == "binary":
        weights = knn_mask.float()
        weights = 0.5 * (weights + weights.transpose(1, 2))
    elif variant == "directed":
        pass
    elif variant == "mutual":
        mutual_mask = knn_mask & knn_mask.transpose(1, 2)
        weights = torch.where(mutual_mask, weights, torch.zeros_like(weights))
    elif variant == "sym_weighted":
        weights = 0.5 * (weights + weights.transpose(1, 2))
    else:
        raise ValueError(f"Unsupported graph variant: {variant}")

    valid_eye = eye * mask.unsqueeze(1).float() * mask.unsqueeze(2).float()
    pad_eye = eye * (~mask).unsqueeze(-1).float()
    a_tilde = weights + valid_eye + pad_eye
    degree = a_tilde.sum(dim=-1).clamp_min(eps)
    deg_inv_sqrt = degree.pow(-0.5)
    adj = deg_inv_sqrt.unsqueeze(-1) * a_tilde * deg_inv_sqrt.unsqueeze(-2)
    return GraphBuildResult(adj=adj, raw_affinity=weights)
