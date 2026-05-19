from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import nn
import torch.nn.functional as F

VALID_ENCODER_ABLATIONS = ("full", "transformer_only", "gcn_only", "mlp_only", "random_graph")
LEGACY_ENCODER_ABLATION_ALIASES = {
    "temporal_only": "transformer_only",
    "graph_only": "gcn_only",
    "identity_graph": "full",
}


def canonicalize_encoder_ablation(encoder_ablation: str) -> str:
    mode = str(encoder_ablation).strip().lower()
    mode = LEGACY_ENCODER_ABLATION_ALIASES.get(mode, mode)
    if mode not in VALID_ENCODER_ABLATIONS:
        raise ValueError(f"Unsupported encoder ablation: {encoder_ablation}")
    return mode


def masked_mean_pool(h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.unsqueeze(-1).float()
    return (h * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


def masked_max_pool(h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    masked_h = h.masked_fill(~mask.unsqueeze(-1), float("-inf"))
    pooled = masked_h.max(dim=1).values
    pooled = torch.where(mask.any(dim=1, keepdim=True), pooled, torch.zeros_like(pooled))
    return torch.nan_to_num(pooled, nan=0.0, posinf=0.0, neginf=0.0)


def uniform_attention(mask: torch.Tensor) -> torch.Tensor:
    attn = mask.float()
    return attn / attn.sum(dim=-1, keepdim=True).clamp_min(1.0)


class DenseGCNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        h = torch.bmm(adj, h)
        h = self.linear(h)
        h = F.gelu(h)
        h = self.dropout(h)
        return self.norm(h)


class AttentionPool(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.query = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, h: torch.Tensor, mask: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor]:
        score = self.query(torch.tanh(self.proj(h))).squeeze(-1)
        if mask is not None:
            score = score.masked_fill(~mask.bool(), -1e9)
        attn = torch.softmax(score, dim=-1)
        if mask is not None:
            attn = attn * mask.float()
            attn = attn / attn.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        z = torch.sum(h * attn.unsqueeze(-1), dim=1)
        return z, attn


class UCSOODID(nn.Module):
    """Transformer/GCN encoder with strict ablation controls and attention pooling."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        gcn_layers: int = 2,
        dropout: float = 0.1,
        gate: str = "learned",
        encoder_ablation: str = "full",
        record_head: bool = False,
        max_window_size: int = 256,
        num_groups: Optional[int] = None,
        use_group_embedding: bool = False,
        group_embedding_dim: int = 16,
        unknown_group_index: Optional[int] = None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.gate_mode = gate
        # `transformer_only` is the current recommended deployment-oriented mainline;
        # GCN-capable modes remain available for historical ablations and extensions.
        self.encoder_ablation = canonicalize_encoder_ablation(encoder_ablation)
        self.record_head_enabled = record_head
        self.use_group_embedding = bool(use_group_embedding)
        self.group_embedding_dim = int(group_embedding_dim if self.use_group_embedding else 0)
        self.num_groups = None if num_groups is None else int(num_groups)
        self.unknown_group_index = None if unknown_group_index is None else int(unknown_group_index)

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_window_size, hidden_dim))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        self.gcn_in = nn.Linear(input_dim, hidden_dim)
        self.gcn_layers = nn.ModuleList([DenseGCNLayer(hidden_dim, hidden_dim, dropout=dropout) for _ in range(gcn_layers)])
        self.mlp_pool_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.mlp_pool_norm = nn.LayerNorm(hidden_dim)

        self.gate = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fused_norm = nn.LayerNorm(hidden_dim)
        self.pool = AttentionPool(hidden_dim)
        if self.use_group_embedding:
            if self.num_groups is None or self.num_groups <= 0:
                raise ValueError("num_groups must be provided when use_group_embedding=True")
            self.group_embedding = nn.Embedding(self.num_groups, self.group_embedding_dim)
        else:
            self.group_embedding = None
        self.embedding_dim = hidden_dim + self.group_embedding_dim
        self.classifier = nn.Linear(self.embedding_dim, num_classes)
        self.record_classifier = nn.Linear(hidden_dim, num_classes) if record_head else None
        self.dropout = nn.Dropout(dropout)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        if self.group_embedding is not None:
            nn.init.normal_(self.group_embedding.weight, mean=0.0, std=0.02)
            if self.unknown_group_index is not None and 0 <= self.unknown_group_index < self.group_embedding.num_embeddings:
                with torch.no_grad():
                    self.group_embedding.weight[self.unknown_group_index].zero_()

    @property
    def uses_temporal_encoder(self) -> bool:
        return self.encoder_ablation in {"full", "transformer_only", "random_graph"}

    @property
    def uses_graph_encoder(self) -> bool:
        return self.encoder_ablation in {"full", "gcn_only", "random_graph"}

    @property
    def uses_mlp_pool_only(self) -> bool:
        return self.encoder_ablation == "mlp_only"

    def encode(
        self,
        x: torch.Tensor,
        adj: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        group_index: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        b, n, _ = x.shape
        if mask is None:
            mask = torch.ones((b, n), dtype=torch.bool, device=x.device)
        else:
            mask = mask.to(device=x.device, dtype=torch.bool)

        branch_mask = mask.unsqueeze(-1).float()
        h_t = x.new_zeros((b, n, self.hidden_dim))
        h_g = x.new_zeros((b, n, self.hidden_dim))
        h_mlp = x.new_zeros((b, n, self.hidden_dim))

        if self.uses_mlp_pool_only:
            h_mlp = F.gelu(self.input_proj(x))
            h_mlp = self.fused_norm(self.dropout(h_mlp)) * branch_mask
            pooled = torch.cat([masked_mean_pool(h_mlp, mask), masked_max_pool(h_mlp, mask)], dim=-1)
            z = self.mlp_pool_norm(self.dropout(F.gelu(self.mlp_pool_proj(pooled))))
            out = {
                "record_embeddings": h_mlp,
                "embedding": z,
                "attention": uniform_attention(mask),
                "gate": None,
                "h_temporal": h_t,
                "h_graph": h_g,
                "h_mlp": h_mlp,
                "mask": mask,
            }
            if self.use_group_embedding:
                if group_index is None:
                    raise ValueError("group_index is required when use_group_embedding=True")
                group_index = group_index.to(device=x.device, dtype=torch.long).view(b)
                group_emb = self.group_embedding(group_index)
                out["group_index"] = group_index
                out["group_embedding"] = group_emb
                out["embedding"] = torch.cat([z, group_emb], dim=-1)
            return out

        if self.uses_temporal_encoder:
            h0 = self.input_proj(x) + self.pos_embedding[:, :n, :]
            h0 = h0 * branch_mask
            h_t = self.transformer(h0, src_key_padding_mask=~mask)
            h_t = h_t * branch_mask

        if self.uses_graph_encoder:
            if adj is None:
                raise ValueError(f"adj is required when encoder_ablation={self.encoder_ablation}")
            h_g = self.gcn_in(x) * branch_mask
            for layer in self.gcn_layers:
                h_g = layer(h_g, adj) * branch_mask

        if self.encoder_ablation == "transformer_only":
            h = h_t
            gate = torch.ones_like(h_t)
        elif self.encoder_ablation == "gcn_only":
            h = h_g
            gate = torch.zeros_like(h_t)
        else:
            if self.gate_mode == "temporal_only":
                h = h_t
                gate = torch.ones_like(h_t)
            elif self.gate_mode == "graph_only":
                h = h_g
                gate = torch.zeros_like(h_t)
            elif self.gate_mode == "mean":
                gate = torch.full_like(h_t, 0.5)
                h = 0.5 * h_t + 0.5 * h_g
            elif self.gate_mode == "learned":
                gate = torch.sigmoid(self.gate(torch.cat([h_t, h_g], dim=-1)))
                h = gate * h_t + (1.0 - gate) * h_g
            else:
                raise ValueError(f"Unsupported gate mode: {self.gate_mode}")
        h = self.fused_norm(self.dropout(h)) * branch_mask
        z, attn = self.pool(h, mask=mask)
        out = {
            "record_embeddings": h,
            "embedding": z,
            "attention": attn,
            "gate": gate,
            "h_temporal": h_t,
            "h_graph": h_g,
            "h_mlp": h_mlp,
            "mask": mask,
        }
        if self.use_group_embedding:
            if group_index is None:
                raise ValueError("group_index is required when use_group_embedding=True")
            group_index = group_index.to(device=x.device, dtype=torch.long).view(b)
            group_emb = self.group_embedding(group_index)
            out["group_index"] = group_index
            out["group_embedding"] = group_emb
            out["embedding"] = torch.cat([z, group_emb], dim=-1)
        return out

    def forward(
        self,
        x: torch.Tensor,
        adj: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
        mask: Optional[torch.Tensor] = None,
        group_index: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        out = self.encode(x, adj, mask=mask, group_index=group_index)
        logits = self.classifier(out["embedding"])
        probs = torch.sigmoid(logits / max(float(temperature), 1e-6))
        out.update({"logits": logits, "probs": probs})
        if self.record_classifier is not None:
            out["record_logits"] = self.record_classifier(out["record_embeddings"])
            out["record_probs"] = torch.sigmoid(out["record_logits"] / max(float(temperature), 1e-6))
        return out
