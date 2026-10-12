"""Distributed helpers for torchrun; every call is a no-op in single-process runs.

The trainers use manual gradient all-reduce instead of DDP because the stage
losses call PSF methods (``reconstruct`` / ``encode``) rather than ``forward``,
which DDP's autograd hooks would not intercept.
"""

from __future__ import annotations

import os
from typing import Iterable, Optional

import torch
import torch.distributed as dist


def setup() -> Optional[torch.device]:
    """Init NCCL from torchrun's env vars; returns the local device, else None."""
    if "RANK" not in os.environ:
        return None
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", device_id=device)
    return device


def cleanup() -> None:
    if is_distributed():
        dist.destroy_process_group()


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def rank() -> int:
    return dist.get_rank() if is_distributed() else 0


def world_size() -> int:
    return dist.get_world_size() if is_distributed() else 1


def is_main() -> bool:
    """True in the process that logs, evaluates, and saves checkpoints."""
    return rank() == 0


def barrier() -> None:
    if is_distributed():
        dist.barrier()


def broadcast_module(module: torch.nn.Module) -> None:
    """Copy rank 0's parameters and buffers into every other process."""
    if not is_distributed():
        return
    for tensor in list(module.parameters()) + list(module.buffers()):
        dist.broadcast(tensor.detach(), src=0)


def allreduce_grads(params: Iterable[torch.nn.Parameter]) -> None:
    """Average gradients across ranks; run after backward, before clipping."""
    if not is_distributed():
        return
    for p in params:
        if p.grad is not None:
            dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)


def allreduce_sum(*tensors: torch.Tensor) -> None:
    """In-place sum across ranks, e.g. for dataset-wide statistics."""
    if not is_distributed():
        return
    for t in tensors:
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
