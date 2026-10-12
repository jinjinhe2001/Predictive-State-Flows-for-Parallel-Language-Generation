"""Muon: momentum SGD orthogonalized by Newton-Schulz (Jordan et al., 2024).

The speedrun-standard optimizer for hidden 2-D weight matrices; embeddings,
heads, norms and biases stay on AdamW. Gradients are expected to be already
all-reduced (this repo reduces before step), so every rank takes the same
deterministic step.
"""

from __future__ import annotations

import torch


@torch.no_grad()
def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Approximate UV^T for G = USV^T via a quintic Newton-Schulz iteration."""
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.bfloat16()
    transposed = X.shape[0] > X.shape[1]
    if transposed:
        X = X.T
    X = X / (X.norm() + 1e-7)
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X.to(G.dtype)


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr: float = 0.02, momentum: float = 0.95,
                 nesterov: bool = True, ns_steps: int = 5) -> None:
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov,
                        ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None) -> None:
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(group["momentum"]).add_(g)
                if group["nesterov"]:
                    g = g.add(buf, alpha=group["momentum"])
                else:
                    g = buf
                O = zeropower_via_newtonschulz5(g, group["ns_steps"])
                scale = max(1.0, p.shape[0] / p.shape[1]) ** 0.5
                p.add_(O, alpha=-group["lr"] * scale)
