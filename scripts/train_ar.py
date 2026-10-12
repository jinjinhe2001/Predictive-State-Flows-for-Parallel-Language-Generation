"""Train the plain autoregressive baseline (psf.models.ar_lm.ARLM) on the writer's data.

Same pre-tokenized OpenWebText (T5 tokenizer, 1024-token canvas, PAD after the document), same
optimizer family as the writer (AdamW 3e-4, betas 0.9/0.95, wd 0.01, clip 1.0, warmup 1000, cosine to
0.1x, EMA 0.999, bf16 autocast), manual gradient all-reduce under torchrun.

  torchrun --standalone --nproc_per_node 8 -m scripts.train_ar --steps 103000 --batch-size 64 \
      --output-dir runs/ar-197m > train_ar.log 2>&1
Global batch = per-rank batch x world size (64 x 8 = 512 documents = 0.52M tokens/step as in the
writer's from-scratch stage; 103k steps = 54B tokens = the main line's budget).
Checkpoint: {"model": state_dict (EMA weights), "ar_config": ARConfig fields, "step": int}.
"""
from __future__ import annotations

import argparse
import math
import os
import time
from dataclasses import asdict

import torch

from psf.config import Config
from psf.models.ar_lm import ARConfig, ARLM
from psf.training.ema import EMA
from psf.utils import dist
from psf.utils.data import make_dataloaders
from psf.utils.misc import get_device, set_seed


def parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--dataset-path", default=None)
    p.add_argument("--dim", type=int, default=1024)
    p.add_argument("--depth", type=int, default=13)
    p.add_argument("--heads", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=64, help="per rank")
    p.add_argument("--steps", type=int, default=103_000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup", type=int, default=1000)
    p.add_argument("--wd", type=float, default=0.01)
    p.add_argument("--ema-decay", type=float, default=0.999)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--val-every", type=int, default=2000)
    p.add_argument("--save-every", type=int, default=5000)
    p.add_argument("--resume", default=None, help="checkpoint to continue from (model + step; fresh optimizer)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    return p.parse_args()


def main() -> None:
    a = parse()
    cfg = Config()
    if a.dataset_path:
        cfg.data.dataset_path = a.dataset_path
    cfg.train.batch_size = a.batch_size
    cfg.train.seed = a.seed
    set_seed(a.seed)
    device = dist.setup() or get_device(a.device)
    torch.set_float32_matmul_precision("high")
    train_loader, val_loader = make_dataloaders(cfg)

    ar_cfg = ARConfig(vocab_size=cfg.data.vocab_size, max_seq_len=cfg.max_seq_len, dim=a.dim, depth=a.depth,
                      num_heads=a.heads, bos_id=cfg.data.bob_token_id, eos_id=cfg.data.eos_token_id)
    model = ARLM(ar_cfg).to(device)
    start = 0
    if a.resume:
        ck = torch.load(a.resume, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        start = int(ck.get("step", 0))
    dist.broadcast_module(model)
    params = [p for p in model.parameters() if p.requires_grad]
    decay = [p for p in params if p.ndim >= 2]
    no_decay = [p for p in params if p.ndim < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": a.wd}, {"params": no_decay, "weight_decay": 0.0}],
                            lr=a.lr, betas=(0.9, 0.95))
    ema = EMA(params, a.ema_decay) if a.ema_decay > 0 else None
    if dist.is_main():
        print(f"AR baseline: {model.num_params() / 1e6:.1f}M params | d={a.dim} depth={a.depth} heads={a.heads} | "
              f"global batch={a.batch_size * dist.world_size()} docs | {len(train_loader.dataset)} train docs | "
              f"steps {start}->{a.steps}", flush=True)
        os.makedirs(a.output_dir, exist_ok=True)

    def lr_at(i: int) -> float:
        if i < a.warmup:
            return a.lr * (i + 1) / a.warmup
        f = (i - a.warmup) / max(1, a.steps - a.warmup)
        return a.lr * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * min(1.0, f))))

    def save(step: int) -> None:
        if ema is not None:
            ema.swap()
        if dist.is_main():
            torch.save({"model": model.state_dict(), "ar_config": asdict(ar_cfg), "step": step},
                       os.path.join(a.output_dir, "ar.pt"))
        if ema is not None:
            ema.swap()
        dist.barrier()

    @torch.no_grad()
    def validate() -> float:
        if val_loader is None:
            return float("nan")
        model.eval()
        tot, n = torch.zeros((), device=device), torch.zeros((), device=device)
        for i, batch in enumerate(val_loader):
            if i >= 8:
                break
            batch = batch.to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                nll, cnt = model(batch.tokens, batch.mask)
            tot += nll * cnt
            n += cnt
        dist.allreduce_sum(tot, n)
        model.train()
        return float(tot / n.clamp_min(1))

    model.train()
    it = iter(train_loader)
    epoch = 0
    t0 = time.time()
    run_loss = 0.0
    for step in range(start, a.steps):
        try:
            batch = next(it)
        except StopIteration:
            epoch += 1
            if hasattr(train_loader, "sampler") and hasattr(train_loader.sampler, "set_epoch"):
                train_loader.sampler.set_epoch(epoch)
            it = iter(train_loader)
            batch = next(it)
        batch = batch.to(device)
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, _ = model(batch.tokens, batch.mask)
        loss.backward()
        dist.allreduce_grads(params)
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        if ema is not None:
            ema.update()
        run_loss += float(loss)
        if (step + 1) % a.log_every == 0 and dist.is_main():
            dt = (time.time() - t0) / a.log_every
            print(f"[ar] step {step + 1}/{a.steps} | loss={run_loss / a.log_every:.4f} | gnorm={float(gn):.2f} | "
                  f"lr={lr_at(step):.2e} | s/step={dt:.3f} | tok/s={a.batch_size * dist.world_size() * cfg.max_seq_len / dt:,.0f}",
                  flush=True)
            run_loss, t0 = 0.0, time.time()
        if (step + 1) % a.val_every == 0:
            v = validate()
            if dist.is_main():
                print(f"[ar] step {step + 1} | val nll={v:.4f} nats/token | val ppl(T5)={math.exp(v):.2f}", flush=True)
        if (step + 1) % a.save_every == 0 or step + 1 == a.steps:
            save(step + 1)
    if ema is not None:
        ema.copy_to()
    v = validate()
    if dist.is_main():
        print(f"[ar] final (EMA) val nll={v:.4f} nats/token | val ppl(T5)={math.exp(v):.2f} | saved {a.output_dir}/ar.pt",
              flush=True)
    dist.cleanup()


if __name__ == "__main__":
    main()
