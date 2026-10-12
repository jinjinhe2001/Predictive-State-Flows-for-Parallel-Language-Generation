"""Encoder q(z_{1:K} | x_{1:L}): transformer + fixed-grid mean pooling."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from psf.config import EncoderConfig
from psf.models.rope import HybridRoPE


@dataclass
class EncoderOutput:
    block_feats: Tensor   # [B, K, D] one latent per block
    block_valid: Tensor   # [B, K] True when the block holds at least one real token
    # Realization code posterior q(r_k | x_<=k) (prefix encoder, rc_dim > 0):
    rc_mu: Optional[Tensor] = None       # [B, K, R]
    rc_logvar: Optional[Tensor] = None   # [B, K, R]
    rc_valid: Optional[Tensor] = None    # [B, K] block k itself holds real tokens


class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.dropout = dropout
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x: Tensor, rope: HybridRoPE, block_ids: Tensor, mask: Tensor) -> Tensor:
        B, L, D = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(dim=0)   # [B, H, L, d]
        q, k = rope(q, block_ids), rope(k, block_ids)
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=mask[:, None, None, :],   # True = attendable
            dropout_p=self.dropout if self.training else 0.0,
        )
        return self.out_proj(out.transpose(1, 2).reshape(B, L, D))


class EncoderLayer(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SelfAttention(dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, rope: HybridRoPE, block_ids: Tensor, mask: Tensor) -> Tensor:
        x = x + self.dropout(self.attn(self.norm1(x), rope, block_ids, mask))
        return x + self.dropout(self.mlp(self.norm2(x)))


class Encoder(nn.Module):
    def __init__(self, vocab_size: int, latent_dim: int, config: EncoderConfig,
                 num_blocks: int, block_span: int) -> None:
        super().__init__()
        self.config = config
        self.num_blocks = num_blocks
        self.block_span = block_span
        self.latent_dim = latent_dim

        self.token_emb = nn.Embedding(vocab_size, latent_dim)
        nn.init.normal_(self.token_emb.weight, std=latent_dim ** -0.5)
        self.embed_scale = math.sqrt(latent_dim)
        self.rope = HybridRoPE(
            latent_dim // config.num_heads, num_blocks, block_span, config.rope_theta
        )
        self.layers = nn.ModuleList(
            EncoderLayer(latent_dim, config.num_heads, config.mlp_ratio, config.dropout)
            for _ in range(config.depth)
        )
        self.final_norm = nn.LayerNorm(latent_dim)
        # Stand-in latent for blocks that contain no real token.
        self.pad_latent = nn.Parameter(torch.randn(latent_dim) * 0.02)

    def forward(self, tokens: Tensor, mask: Optional[Tensor] = None) -> EncoderOutput:
        B, L = tokens.shape
        if L != self.num_blocks * self.block_span:
            raise ValueError(
                f"expected L={self.num_blocks * self.block_span} tokens; got {L}"
            )
        if mask is None:
            mask = torch.ones(B, L, dtype=torch.bool, device=tokens.device)

        block_ids = torch.arange(L, device=tokens.device) // self.block_span
        h = self.token_emb(tokens) * self.embed_scale
        for layer in self.layers:
            h = layer(h, self.rope, block_ids, mask)
        h = self.final_norm(h)

        # Fixed grid: pooling is a reshape over [K, S].
        K, S, D = self.num_blocks, self.block_span, self.latent_dim
        m = mask.reshape(B, K, S, 1).to(h.dtype)
        counts = m.sum(dim=2)                                    # [B, K, 1]
        feats = (h.reshape(B, K, S, D) * m).sum(dim=2) / counts.clamp(min=1)
        block_valid = counts.squeeze(-1) > 0                     # [B, K]
        feats = torch.where(block_valid.unsqueeze(-1), feats, self.pad_latent.to(feats.dtype))
        return EncoderOutput(block_feats=feats, block_valid=block_valid)
