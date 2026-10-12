"""Exponential moving average of the trained parameters (ELF keeps one too).

With the manual all-reduce DDP used here, every rank applies identical
optimizer updates, so per-rank EMA shadows stay identical and rank 0 can save
them without any synchronization.
"""

from __future__ import annotations

import torch
from torch import Tensor


class EMA:
    def __init__(self, params: list[Tensor], decay: float, warmup: bool = True,
                 warmup_steps: int = 2000) -> None:
        """``warmup``: ramp the decay linearly from ~0 to ``decay`` over
        ``warmup_steps`` so the shadow forgets its initialisation quickly and
        then averages with the full horizon. Without it a 0.9999 EMA still
        carries 5% of the initial weights after 30k steps -- for a warm start
        whose new code pathway is zero, that showed up as under-dispersed
        codes.

        History: 2026-09-04 01:20-04:30 UTC (tlow60k, tlow-rollmix60k) used the
        TF-style min(decay, (1+t)/(10+t)), whose decay is still only 0.9997 at
        30k steps (3.3k-step horizon instead of 10k) -- a confound; this linear ramp reaches the full decay at warmup_steps."""
        self.decay = decay
        self.warmup = warmup
        self.warmup_steps = max(1, warmup_steps)
        self.step = 0
        self.params = params
        self.shadow = [p.detach().clone() for p in params]

    def current_decay(self) -> float:
        if not self.warmup:
            return self.decay
        return self.decay * min(1.0, self.step / self.warmup_steps)

    @torch.no_grad()
    def update(self) -> None:
        self.step += 1
        d = self.current_decay()
        for s, p in zip(self.shadow, self.params):
            s.lerp_(p.detach(), 1.0 - d)

    @torch.no_grad()
    def swap(self) -> None:
        """Exchange shadow and live weights (call twice to restore)."""
        for s, p in zip(self.shadow, self.params):
            tmp = p.detach().clone()
            p.copy_(s)
            s.copy_(tmp)

    @torch.no_grad()
    def copy_to(self) -> None:
        """Overwrite the live weights with the EMA (end of training)."""
        for s, p in zip(self.shadow, self.params):
            p.copy_(s)
