from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


def rotate_half(x: Tensor) -> Tensor:
    x = x.unflatten(-1, (-1, 2))
    x1, x2 = x.unbind(dim=-1)
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def rope_tables(length: int, dim: int, theta: float) -> tuple[Tensor, Tensor]:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    angles = torch.outer(torch.arange(length, dtype=torch.float32), freqs)
    angles = angles.repeat_interleave(2, dim=-1)
    return torch.cos(angles), torch.sin(angles)


class HybridRoPE(nn.Module):

    def __init__(self, head_dim: int, num_blocks: int, block_span: int, theta: float) -> None:
        super().__init__()
        if head_dim % 4 != 0:
            raise ValueError(f"hybrid RoPE needs head_dim % 4 == 0; got {head_dim}")
        self.axis_dim = head_dim // 2
        self.block_span = block_span
        pos_cos, pos_sin = rope_tables(num_blocks * block_span, self.axis_dim, theta)
        blk_cos, blk_sin = rope_tables(num_blocks, self.axis_dim, theta)
        self.register_buffer("pos_cos", pos_cos, persistent=False)
        self.register_buffer("pos_sin", pos_sin, persistent=False)
        self.register_buffer("blk_cos", blk_cos, persistent=False)
        self.register_buffer("blk_sin", blk_sin, persistent=False)

    @staticmethod
    def _rotate(x: Tensor, cos: Tensor, sin: Tensor, ids: Tensor) -> Tensor:
        cos = cos[ids].to(x.dtype)[None, None]   # [1, 1, L, axis_dim]
        sin = sin[ids].to(x.dtype)[None, None]
        return x * cos + rotate_half(x) * sin

    def forward(self, x: Tensor, block_ids: Tensor) -> Tensor:
        # x: [B, H, L, head_dim]
        pos_x, blk_x = x.split(self.axis_dim, dim=-1)
        positions = torch.arange(x.shape[-2], device=x.device)
        pos_x = self._rotate(pos_x, self.pos_cos, self.pos_sin, positions)
        blk_x = self._rotate(blk_x, self.blk_cos, self.blk_sin, block_ids)
        return torch.cat([pos_x, blk_x], dim=-1)


class RoPE(nn.Module):
    """Plain 1-D RoPE table applied at explicit positions."""

    def __init__(self, head_dim: int, max_len: int, theta: float) -> None:
        super().__init__()
        cos, sin = rope_tables(max_len, head_dim, theta)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def rotate(self, x: Tensor, positions: Tensor) -> Tensor:
        """x: [..., H, N, d]; positions: [N] indices into the table."""
        cos = self.cos[positions].to(x.dtype)
        sin = self.sin[positions].to(x.dtype)
        return x * cos + rotate_half(x) * sin
