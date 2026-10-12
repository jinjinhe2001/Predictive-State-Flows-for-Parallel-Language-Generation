"""Flow-matching denoiser over the K block latents.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from psf.config import DenoiserConfig
from psf.models.rope import RoPE


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        inv = torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return self.weight * (x * inv.to(x.dtype))


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.dropout_p = dropout
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

    def forward(self, x: Tensor, rope: RoPE, positions: Tensor,
                attn_mask: Tensor | None = None) -> Tensor:
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(dim=0)      # [B, H, N, d]
        q = rope.rotate(self.q_norm(q).to(v.dtype), positions)
        k = rope.rotate(self.k_norm(k).to(v.dtype), positions)
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
        )
        return self.proj(out.transpose(1, 2).reshape(B, N, D))


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        eff = int(hidden * 2 / 3)
        self.w12 = nn.Linear(dim, 2 * eff)
        self.w3 = nn.Linear(eff, dim)

    def forward(self, x: Tensor) -> Tensor:
        x1, x2 = self.w12(x).chunk(2, dim=-1)
        return self.w3(F.silu(x1) * x2)


class DenoiserBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = Attention(dim, num_heads, dropout)
        self.norm2 = RMSNorm(dim)
        self.mlp = SwiGLU(dim, int(dim * mlp_ratio))

    def forward(self, x: Tensor, rope: RoPE, positions: Tensor,
                attn_mask: Tensor | None = None) -> Tensor:
        x = x + self.attn(self.norm1(x), rope, positions, attn_mask)
        return x + self.mlp(self.norm2(x))


class TimestepEmbedder(nn.Module):

    def __init__(self, hidden: int, freq_dim: int) -> None:
        super().__init__()
        self.freq_dim = freq_dim
        self.mlp = nn.Sequential(
            nn.Linear(freq_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden)
        )

    def forward(self, t: Tensor) -> Tensor:
        half = self.freq_dim // 2
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, dtype=torch.float32, device=t.device) / half
        )
        args = t.float()[:, None] * 1000.0 * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return self.mlp(emb)


class Denoiser(nn.Module):
    def __init__(self, latent_dim: int, num_blocks: int, config: DenoiserConfig) -> None:
        super().__init__()
        self.config = config
        self.latent_dim = latent_dim
        self.num_blocks = num_blocks
        H = config.hidden_size
        self.num_time_tokens = max(1, config.num_time_tokens)

        self.self_cond_proj = (
            nn.Linear(2 * latent_dim, latent_dim) if config.self_conditioning else None
        )
        # Narrow bottleneck on the way in (ELF-style): D -> bottleneck -> H.
        self.in_proj = nn.Sequential(
            nn.Linear(latent_dim, config.bottleneck_dim, bias=False),
            nn.Linear(config.bottleneck_dim, H),
        )
        self.t_embedder = TimestepEmbedder(H, config.time_embed_dim)
        self.t_tokens = nn.Parameter(torch.randn(1, self.num_time_tokens, H) * 0.02)

        # SC-CFG: in-context guidance-scale tokens (ELF-style), absent by default.
        self.cfg_tokens = None
        self.cfg_embedder = None
        if config.num_cfg_tokens > 0:
            self.cfg_tokens = nn.Parameter(
                torch.randn(1, config.num_cfg_tokens, H) * 0.02
            )
            self.cfg_embedder = TimestepEmbedder(H, config.time_embed_dim)

        self.rope = RoPE(
            H // config.num_heads, self.num_time_tokens + num_blocks, config.rope_theta
        )
        self.blocks = nn.ModuleList(
            DenoiserBlock(H, config.num_heads, config.mlp_ratio, config.dropout)
            for _ in range(config.depth)
        )
        self.out_norm = RMSNorm(H)
        self.out_proj = nn.Linear(H, latent_dim)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

        # Latent standardization, filled from the frozen encoder before stage 2.
        self.register_buffer("latent_mean", torch.zeros(latent_dim))
        self.register_buffer("latent_std", torch.ones(latent_dim))

    def normalize(self, z: Tensor) -> Tensor:
        return (z - self.latent_mean) / self.latent_std

    def denormalize(self, z: Tensor) -> Tensor:
        return z * self.latent_std + self.latent_mean

    @torch.no_grad()
    def set_latent_stats(self, mean: Tensor, std: Tensor, eps: float = 1e-4) -> None:
        self.latent_mean.copy_(mean.to(self.latent_mean))
        self.latent_std.copy_(std.clamp_min(eps).to(self.latent_std))

    def forward(self, z_t: Tensor, t: Tensor, self_cond: Optional[Tensor] = None,
                cfg_scale: Optional[Tensor] = None,
                ctx_drop: Optional[Tensor] = None) -> Tensor:
        """z_t: [B, K, D] noisy latents; t: [B] in [0, 1]. Returns the x0 prediction.

        ``cfg_scale`` is the SC-CFG guidance scale (scalar or [B]); it is only
        consumed when the model was built with num_cfg_tokens > 0, where None
        defaults to the neutral scale 1.

        ``ctx_drop`` is a [B] bool mask (prefix-drop CFG): for those rows each
        latent token attends only the prefix tokens and itself, so the
        prediction is context-free per block.
        """
        B, K, _ = z_t.shape
        if K > self.num_blocks:
            raise ValueError(f"K={K} exceeds num_blocks={self.num_blocks}")

        if self.self_cond_proj is not None:
            if self_cond is None:
                self_cond = torch.zeros_like(z_t)
            z_t = self.self_cond_proj(torch.cat([z_t, self_cond], dim=-1))

        h = self.in_proj(z_t)                                   # [B, K, H]
        if t.dim() == 2:
            # Diffusion forcing: per-block time [B, K]. Each block token receives
            # its own time embedding *relative to the row mean*, and the prefix
            # tokens carry the row mean. A shared-time row therefore reduces
            # EXACTLY to the original network (the relative term is zero), so a
            # plain prior is a valid warm start and the staggered sampler with
            # lambda -> 0 is the ordinary sampler.
            t_row = t.mean(dim=1)
            e_blk = self.t_embedder(t.reshape(-1)).view(B, K, -1)
            h = h + (e_blk - self.t_embedder(t_row).unsqueeze(1)).to(h.dtype)
            t = t_row
        prefix = self.t_tokens.expand(B, -1, -1) + self.t_embedder(t).unsqueeze(1)
        if self.cfg_tokens is not None:
            if cfg_scale is None:
                cfg_scale = torch.ones(B, device=z_t.device)
            elif not torch.is_tensor(cfg_scale):
                cfg_scale = torch.full((B,), float(cfg_scale), device=z_t.device)
            cfg_prefix = (self.cfg_tokens.expand(B, -1, -1)
                          + self.cfg_embedder(cfg_scale).unsqueeze(1))
            prefix = torch.cat([prefix, cfg_prefix.to(prefix.dtype)], dim=1)
        h = torch.cat([prefix.to(h.dtype), h], dim=1)           # [B, p+K, H]

        p = prefix.shape[1]
        positions = torch.cat([
            torch.zeros(p, dtype=torch.long, device=h.device),
            torch.arange(1, K + 1, device=h.device),
        ])
        attn_mask = None
        if ctx_drop is not None and bool(ctx_drop.any()):
            S = h.shape[1]
            eye = torch.eye(S, dtype=torch.bool, device=h.device)
            restricted = eye.clone()
            restricted[:, :p] = True                     # prefix tokens stay visible
            full = torch.ones(S, S, dtype=torch.bool, device=h.device)
            attn_mask = torch.where(
                ctx_drop.view(B, 1, 1, 1), restricted[None, None], full[None, None]
            )
        for block in self.blocks:
            h = block(h, self.rope, positions, attn_mask)
        return self.out_proj(self.out_norm(h[:, p:]))
