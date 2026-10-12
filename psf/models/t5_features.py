"""Frozen T5 contextual features under a block-causal mask.

ELF showed frozen T5 contextual embeddings are a good diffusion space; here
they replace the from-scratch token embeddings at the *input* of the prefix
encoder. The T5 encoder is bidirectional, so it must run under the same
block-causal mask as the trunk above it — otherwise the contextual embedding
of a token in block j would smuggle information about future blocks into S_k.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class FrozenT5Features(nn.Module):
    def __init__(self, model_name: str, block_span: int) -> None:
        super().__init__()
        from transformers import T5EncoderModel

        self.t5 = T5EncoderModel.from_pretrained(model_name)
        self.t5.requires_grad_(False)
        self.t5.eval()
        self.block_span = block_span
        self.hidden_size = self.t5.config.d_model

    def train(self, mode: bool = True) -> "FrozenT5Features":
        """Stay in eval mode: the trunk is frozen (no dropout ever)."""
        super().train(mode)
        self.t5.eval()
        return self

    @torch.no_grad()
    def forward(self, tokens: Tensor, mask: Tensor) -> Tensor:
        """tokens [B, L], mask [B, L] -> contextual features [B, L, d_model].

        The mask must be a 4-D *additive* float bias: transformers 5.x
        rejects 3-D masks and silently fails to apply 4-D boolean ones
        (verified: boolean leaks future tokens, additive does not).
        """
        B, L = tokens.shape
        block_ids = torch.arange(L, device=tokens.device) // self.block_span
        causal = block_ids[None, :] <= block_ids[:, None]        # [L, L]
        allow = causal[None] & mask[:, None, :]                  # [B, L, L]
        bias = torch.zeros(B, 1, L, L, dtype=torch.float32, device=tokens.device)
        bias.masked_fill_(~allow.unsqueeze(1), -1e9)
        return self.t5(input_ids=tokens, attention_mask=bias).last_hidden_state
