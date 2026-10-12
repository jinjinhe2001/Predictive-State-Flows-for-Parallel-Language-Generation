"""Stage 2 - latent prior.

The autoencoder is frozen. Its latents are standardized once, then a rectified
flow is fitted on them; only the denoiser receives gradients.
"""

from __future__ import annotations

import math
import os
import time
from typing import Optional

import torch
from torch.utils.data import DataLoader

from psf.config import Config
from psf.diffusion.flow import interpolate
from psf.models.psf import PSF
from psf.training.ema import EMA
from psf.training.losses import prior_loss
from psf.utils import dist
from psf.utils.data import Batch
from psf.utils.misc import cycle, format_metrics, save_checkpoint


class PriorTrainer:
    def __init__(self, model: PSF, cfg: Config, train_loader: DataLoader,
                 val_loader: Optional[DataLoader], device) -> None:
        self.model = model
        self.cfg = cfg
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.steps = cfg.train.prior_steps
        self.ckpt_path = os.path.join(cfg.train.output_dir, "prior.pt")

        model.encoder.requires_grad_(False)
        model.decoder.requires_grad_(False)
        dist.broadcast_module(model)   # every rank starts from rank 0's weights
        self.compute_latent_stats(cfg.train.latent_stats_batches)

        self.params = list(model.denoiser.parameters())
        self.optimizer = torch.optim.AdamW(
            self.params, lr=cfg.train.lr, betas=cfg.train.betas,
            weight_decay=cfg.train.weight_decay,
        )
        warmup = max(1, cfg.train.warmup_steps)
        total = max(warmup + 1, cfg.train.prior_steps)
        floor = cfg.train.lr_min_ratio

        def lr_lambda(step: int) -> float:
            if step < warmup:
                return (step + 1) / warmup
            if cfg.train.lr_schedule != "cosine":
                return 1.0
            frac = min(1.0, (step - warmup) / max(1, total - warmup))
            return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * frac))

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        self.ema = (EMA(self.params, cfg.train.ema_decay, warmup=cfg.train.ema_warmup,
                        warmup_steps=cfg.train.ema_warmup_steps)
                    if cfg.train.ema_decay > 0 else None)
        self.autocast = torch.autocast(
            "cuda", torch.bfloat16,
            enabled=cfg.train.bf16 and torch.device(device).type == "cuda",
        )

    @torch.no_grad()
    def compute_latent_stats(self, max_batches: int) -> None:
        """Standardize the encoder latents so z0 matches the N(0, I) noise scale."""
        self.model.eval()
        D = self.model.prior_dim
        total = torch.zeros(D, device=self.device, dtype=torch.float64)
        total_sq = torch.zeros(D, device=self.device, dtype=torch.float64)
        count = torch.zeros((), device=self.device, dtype=torch.float64)
        for i, batch in enumerate(self.train_loader):
            if i >= max_batches:
                break
            batch = batch.to(self.device)
            enc = self.model.encode(batch.tokens, batch.mask)
            valid = enc.block_valid
            if enc.rc_valid is not None:
                valid = valid & enc.rc_valid   # [s_k ; r_k] rows where both are real
            z = self.model.prior_latents(enc)[valid].double()  # only real blocks
            total += z.sum(dim=0)
            total_sq += (z * z).sum(dim=0)
            count += z.shape[0]
        dist.allreduce_sum(total, total_sq, count)   # pool every rank's shard
        if count.item() == 0:
            raise RuntimeError("no valid blocks found while computing latent stats")
        mean = total / count
        std = (total_sq / count - mean * mean).clamp_min(0).sqrt()
        self.model.denoiser.set_latent_stats(mean.float(), std.float())
        if dist.is_main():
            print(
                f"latent stats over {int(count.item())} blocks | "
                f"|mean|={mean.abs().mean():.4f} std={std.mean():.4f}"
            )

    def loss(self, batch: Batch) -> tuple[torch.Tensor, dict]:
        return prior_loss(self.model, batch)

    def train_step(self, batch: Batch, step: int) -> dict:
        with self.autocast:
            loss, metrics = self.loss(batch)
        if not math.isfinite(float(loss.detach())):
            raise RuntimeError(f"[prior] non-finite loss at step {step}")

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        dist.allreduce_grads(self.params)
        torch.nn.utils.clip_grad_norm_(self.params, self.cfg.train.grad_clip)
        self.optimizer.step()
        self.scheduler.step()
        if self.ema is not None:
            self.ema.update()

        metrics["lr"] = self.scheduler.get_last_lr()[0]
        return metrics

    @torch.no_grad()
    def evaluate(self) -> dict:
        if self.val_loader is None:
            return {}
        self.model.eval()
        totals: dict[str, float] = {}
        n = 0
        for i, batch in enumerate(self.val_loader):
            if i >= self.cfg.train.eval_batches:
                break
            _, metrics = self.loss(batch.to(self.device))
            for k, v in metrics.items():
                totals[k] = totals.get(k, 0.0) + v
            n += 1
        self.model.train()
        return {k: v / max(n, 1) for k, v in totals.items()}

    @torch.no_grad()
    def latent_mse(self, max_batches: int = 4, times=(0.25, 0.5, 0.75, 0.9)) -> dict:
        """Decoder-free probe: how well does the denoiser recover z0 at fixed times?"""
        if self.val_loader is None:
            return {}
        self.model.eval()
        sums = {f"x0mse@{t}": 0.0 for t in times}
        n = 0
        for i, batch in enumerate(self.val_loader):
            if i >= max_batches:
                break
            batch = batch.to(self.device)
            enc = self.model.encode(batch.tokens, batch.mask)
            z0 = self.model.denoiser.normalize(self.model.prior_latents(enc))
            eps = torch.randn_like(z0)
            for t in times:
                tt = torch.full((z0.shape[0],), t, device=self.device)
                pred = self.model.denoiser(interpolate(z0, eps, tt), tt, None)
                sums[f"x0mse@{t}"] += float(torch.mean((pred - z0) ** 2))
            n += 1
        return {k: v / max(n, 1) for k, v in sums.items()}

    def train(self) -> None:
        cfg = self.cfg.train
        if dist.is_main():
            n_params = sum(p.numel() for p in self.params)
            print(f"[prior] training {n_params / 1e6:.2f}M params "
                  f"for {self.steps} steps on {self.device} "
                  f"(world={dist.world_size()})")

        self.model.train()
        batches = cycle(self.train_loader)
        start = time.time()
        for step in range(1, self.steps + 1):
            metrics = self.train_step(next(batches).to(self.device), step)

            if dist.is_main() and (step % cfg.log_every == 0 or step == 1):
                metrics["s/step"] = (time.time() - start) / step
                print(format_metrics(f"[prior] step {step}/{self.steps}", metrics))

            if (dist.is_main() and self.val_loader is not None
                    and cfg.eval_every and step % cfg.eval_every == 0):
                print(format_metrics(
                    f"[prior] eval  {step}/{self.steps}", self.evaluate()
                ))
                self.save(step)

        if self.ema is not None:
            self.ema.copy_to()   # every rank: leave the EMA weights in the model
        if dist.is_main():
            save_checkpoint(self.ckpt_path, self.model, self.steps)
            print(f"[prior] saved -> {self.ckpt_path}"
                  + (" (EMA)" if self.ema is not None else ""))

    def save(self, step: int) -> None:
        """Checkpoint the EMA weights when enabled, without touching training."""
        if self.ema is not None:
            self.ema.swap()
        save_checkpoint(self.ckpt_path, self.model, step)
        if self.ema is not None:
            self.ema.swap()
