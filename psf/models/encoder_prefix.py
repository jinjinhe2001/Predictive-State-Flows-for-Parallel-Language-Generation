"""Prefix encoder: S_k summarizes the blocks strictly before block k.

Variant of :mod:`psf.models.encoder`. The transformer runs under a
block-causal attention mask, per-block pooling yields cumulative states
c_j = f(B_{<=j}), and the output is shifted right one block so that
S_k = c_{k-1} = f(B_{<k}); S_1 is a learned begin-of-sequence latent.

Pooling the *unshifted* positions and then shifting is what keeps S_k
leak-free: the residual stream at block k starts from block k's own token
embeddings, so pooling block k directly would expose B_k even under a
strictly-causal attention mask.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from psf.config import EncoderConfig
from psf.models.encoder import EncoderOutput
from psf.models.rope import HybridRoPE


class CausalSelfAttention(nn.Module):
    """Encoder self-attention that takes a full [B, L, L] attendability mask."""

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
            attn_mask=mask[:, None],   # [B, 1, L, L], True = attendable
            dropout_p=self.dropout if self.training else 0.0,
        )
        return self.out_proj(out.transpose(1, 2).reshape(B, L, D))


class PrefixEncoderLayer(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.norm1 = nn.LayerNorm(dim)
        self.attn = CausalSelfAttention(dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, rope: HybridRoPE, block_ids: Tensor, mask: Tensor) -> Tensor:
        x = x + self.dropout(self.attn(self.norm1(x), rope, block_ids, mask))
        return x + self.dropout(self.mlp(self.norm2(x)))


class GlobalReadout(nn.Module):
    """G learned queries cross-attending the full token stream (bidirectional).

    Produces document-level "plan" latents; unlike the shifted prefix states
    these deliberately see the whole document.
    """

    def __init__(self, dim: int, num_heads: int, num_global: int) -> None:
        super().__init__()
        self.queries = nn.Parameter(torch.randn(1, num_global, dim) * 0.02)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, h: Tensor, mask: Tensor) -> Tensor:
        # .clone(): a raw Parameter view as a module input breaks ModuleTracker
        # (FlopCounterMode); contiguous() is a no-op at batch 1, clone always copies.
        q = self.queries.expand(h.shape[0], -1, -1).clone()
        out, _ = self.attn(q, h, h, key_padding_mask=~mask, need_weights=False)
        return self.norm(q + out)


class PrefixEncoder(nn.Module):
    def __init__(self, vocab_size: int, latent_dim: int, config: EncoderConfig,
                 num_blocks: int, block_span: int, num_global: int = 0,
                 t5_name: str | None = None, num_slots: int = 1,
                 rc_dim: int = 0, unshifted: bool = False) -> None:
        super().__init__()
        self.config = config
        self.unshifted = unshifted
        self.num_blocks = num_blocks
        self.block_span = block_span
        self.latent_dim = latent_dim
        self.num_slots = num_slots
        self.rc_dim = rc_dim
        # Realization-code head on the *unshifted* inclusive state c_k =
        # f(B_<=k): the decoder already sees s_k = c_{k-1}, so under a KL
        # budget the code learns to carry what is new in block k.
        self.rc_head = nn.Linear(latent_dim, 2 * rc_dim) if rc_dim > 0 else None
        if num_slots > 1:
            # m addressable slots per block, projected from the same pooled
            # prefix state; the trunk stays latent_dim wide.
            self.slot_proj = nn.Linear(latent_dim, num_slots * latent_dim)
            self.slot_bos = nn.Parameter(torch.randn(num_slots, latent_dim) * 0.02)
            self.slot_pad = nn.Parameter(torch.randn(num_slots, latent_dim) * 0.02)

        self.t5_feats = None
        self.token_emb = None
        if t5_name is not None:
            from psf.models.t5_features import FrozenT5Features

            self.t5_feats = FrozenT5Features(t5_name, block_span)
            self.feat_proj = nn.Linear(self.t5_feats.hidden_size, latent_dim)
        else:
            self.token_emb = nn.Embedding(vocab_size, latent_dim)
            nn.init.normal_(self.token_emb.weight, std=latent_dim ** -0.5)
        self.embed_scale = math.sqrt(latent_dim)
        self.rope = HybridRoPE(
            latent_dim // config.num_heads, num_blocks, block_span, config.rope_theta
        )
        self.layers = nn.ModuleList(
            PrefixEncoderLayer(latent_dim, config.num_heads, config.mlp_ratio, config.dropout)
            for _ in range(config.depth)
        )
        self.final_norm = nn.LayerNorm(latent_dim)
        # Stand-in for blocks that contain no real token, as in the base encoder.
        self.pad_latent = nn.Parameter(torch.randn(latent_dim) * 0.02)
        # S_1: the empty-prefix state.
        self.bos_latent = nn.Parameter(torch.randn(latent_dim) * 0.02)
        self.global_readout = (
            GlobalReadout(latent_dim, config.num_heads, num_global)
            if num_global > 0 else None
        )

    def forward(self, tokens: Tensor, mask: Optional[Tensor] = None) -> EncoderOutput:
        B, L = tokens.shape
        if L != self.num_blocks * self.block_span:
            raise ValueError(
                f"expected L={self.num_blocks * self.block_span} tokens; got {L}"
            )
        if mask is None:
            mask = torch.ones(B, L, dtype=torch.bool, device=tokens.device)

        block_ids = torch.arange(L, device=tokens.device) // self.block_span
        # Attend to real tokens in blocks <= own block.
        attn_mask = (block_ids[None, :] <= block_ids[:, None])[None] & mask[:, None, :]

        if self.t5_feats is not None:
            h = self.feat_proj(self.t5_feats(tokens, mask))
        else:
            h = self.token_emb(tokens) * self.embed_scale
        for layer in self.layers:
            h = layer(h, self.rope, block_ids, attn_mask)
        h = self.final_norm(h)

        # Cumulative states c_j = f(B_{<=j}), pooled exactly like the base encoder.
        K, S, D = self.num_blocks, self.block_span, self.latent_dim
        m = mask.reshape(B, K, S, 1).to(h.dtype)
        counts = m.sum(dim=2)                                    # [B, K, 1]
        c = (h.reshape(B, K, S, D) * m).sum(dim=2) / counts.clamp(min=1)
        block_valid = counts.squeeze(-1) > 0                     # [B, K]
        c = torch.where(block_valid.unsqueeze(-1), c, self.pad_latent.to(c.dtype))

        rc_mu = rc_logvar = rc_valid = None
        if self.rc_head is not None:
            stats = self.rc_head(c)                                # [B, K, 2R]
            rc_mu, rc_logvar = stats.chunk(2, dim=-1)
            rc_logvar = rc_logvar.clamp(-8.0, 4.0)
            rc_valid = block_valid

        # Shift right: S_k = c_{k-1}, S_1 = learned empty-prefix latent.
        if self.unshifted:
            # Content-latent control: block k reads its own inclusive state c_k (no shift, no BOS latent).
            if self.num_slots > 1:
                raise NotImplementedError("prefix_unshifted with num_slots > 1")
            feats, valid = c, block_valid
        elif self.num_slots > 1:
            M = self.num_slots
            slots = self.slot_proj(c).reshape(B, K, M, D)
            slots = torch.where(block_valid.view(B, K, 1, 1), slots,
                                self.slot_pad.to(slots.dtype))
            bos = self.slot_bos.to(slots.dtype).expand(B, 1, M, D)
            feats = torch.cat([bos, slots[:, :-1]], dim=1).reshape(B, K * M, D)
            valid = torch.cat(
                [torch.ones(B, 1, dtype=torch.bool, device=tokens.device),
                 block_valid[:, :-1]],
                dim=1,
            ).repeat_interleave(M, dim=1)
        else:
            bos = self.bos_latent.to(c.dtype).expand(B, 1, D)
            feats = torch.cat([bos, c[:, :-1]], dim=1)
            valid = torch.cat(
                [torch.ones(B, 1, dtype=torch.bool, device=tokens.device),
                 block_valid[:, :-1]],
                dim=1,
            )
        if self.global_readout is not None:
            g = self.global_readout(h, mask)                       # [B, G, D]
            feats = torch.cat([g, feats], dim=1)
            ones = torch.ones(B, g.shape[1], dtype=torch.bool, device=tokens.device)
            valid = torch.cat([ones, valid], dim=1)
        return EncoderOutput(block_feats=feats, block_valid=valid,
                             rc_mu=rc_mu, rc_logvar=rc_logvar, rc_valid=rc_valid)
