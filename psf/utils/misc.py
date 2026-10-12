"""Small helpers shared by the training stages."""

from __future__ import annotations

import os
from typing import Iterator

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)


def get_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


def cycle(loader: DataLoader) -> Iterator:
    """Repeat a finite loader forever, so training is step- rather than epoch-driven."""
    epoch = 0
    while True:
        if isinstance(getattr(loader, "sampler", None), DistributedSampler):
            loader.sampler.set_epoch(epoch)   # reshuffle each pass over the data
        for batch in loader:
            yield batch
        epoch += 1


def format_metrics(prefix: str, metrics: dict) -> str:
    body = " ".join(f"{k}={v:.4f}" for k, v in metrics.items())
    return f"{prefix} | {body}"


def save_checkpoint(path: str, model: nn.Module, steps: int) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({"model": model.state_dict(), "steps": steps}, path)


def load_checkpoint(path: str, model: nn.Module, device, strict: bool = True) -> int:
    """Load weights in place and return the step count stored in the checkpoint.

    ``strict=False`` tolerates missing/new parameters (e.g. loading a plain AE
    checkpoint into a model whose denoiser has SC-CFG tokens) but still prints
    what was skipped so shape mistakes stay visible.
    """
    ckpt = torch.load(path, map_location=device)
    result = model.load_state_dict(ckpt["model"], strict=strict)
    if not strict and (result.missing_keys or result.unexpected_keys):
        print(f"load_checkpoint({path}): {len(result.missing_keys)} missing, "
              f"{len(result.unexpected_keys)} unexpected keys (non-strict)")
    return int(ckpt.get("steps", 0))
