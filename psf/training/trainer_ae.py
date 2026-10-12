"""Stage 1 - autoencoder.

Encoder and decoder are trained jointly to reconstruct text through the K block
latents. Nothing here touches the denoiser.
"""

from __future__ import annotations

import math
import os
import time
from typing import Optional

import torch
from torch.utils.data import DataLoader

from psf.config import Config
from psf.models.psf import PSF
from psf.training.ema import EMA
from psf.training.losses import ae_loss
from psf.utils import dist
from psf.utils.data import Batch
from psf.utils.misc import cycle, format_metrics, save_checkpoint


class AutoencoderTrainer:
    def __init__(self, model: PSF, cfg: Config, train_loader: DataLoader,
                 val_loader: Optional[DataLoader], device) -> None:
        self.model = model
        self.cfg = cfg
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.steps = cfg.train.ae_steps
        self.ckpt_path = os.path.join(cfg.train.output_dir, "ae.pt")

        dist.broadcast_module(model)   # every rank starts from rank 0's weights
        # Trainer-owned auxiliary heads: feed the encoder a direct gradient,
        # never saved (save() checkpoints the model only).
        self.aux = torch.nn.ModuleDict()
        d_in = cfg.latent_dim * cfg.num_latent_slots
        if cfg.train.ae_aux_bow > 0:
            self.aux["bow"] = torch.nn.Linear(d_in, cfg.data.vocab_size)
        if cfg.train.ae_aux_plan > 0:
            self.aux["plan"] = torch.nn.Linear(d_in, cfg.data.vocab_size)
        self.aux.to(device)
        dist.broadcast_module(self.aux)
        self.params = [
            p for p in
            list(model.encoder.parameters()) + list(model.decoder.parameters())
            + list(self.aux.parameters())
            if p.requires_grad   # skips e.g. the frozen T5 feature trunk
        ]
        # Parameter routing (legacy defaults = one AdamW over everything):
        # Muon takes hidden 2-D matrices, embeddings/head stay on AdamW, and
        # wd_exclude keeps weight decay off norms and biases.
        head_like = {id(model.decoder.token_emb.weight),
                     id(model.decoder.lm_head.weight)}
        enc_emb = getattr(model.encoder, "token_emb", None)
        if enc_emb is not None:
            head_like.add(id(enc_emb.weight))
        decay, no_decay, muon_mats = [], [], []
        for p in self.params:
            if cfg.train.muon and p.ndim == 2 and id(p) not in head_like:
                muon_mats.append(p)
            elif cfg.train.wd_exclude and p.ndim < 2:
                no_decay.append(p)
            else:
                decay.append(p)
        groups = [dict(params=decay, weight_decay=cfg.train.weight_decay)]
        if no_decay:
            groups.append(dict(params=no_decay, weight_decay=0.0))
        self.optimizers = [torch.optim.AdamW(groups, lr=cfg.train.lr,
                                             betas=cfg.train.betas)]
        if muon_mats:
            from psf.training.muon import Muon

            self.optimizers.append(Muon(muon_mats, lr=cfg.train.muon_lr,
                                        momentum=cfg.train.muon_momentum))
        for opt in self.optimizers:
            for g in opt.param_groups:
                g["base_lr"] = g["lr"]
        self._sched_step = 0
        self.ema = EMA(self.params, cfg.train.ema_decay) if cfg.train.ema_decay > 0 else None
        self.autocast = torch.autocast(
            "cuda", torch.bfloat16,
            enabled=cfg.train.bf16 and torch.device(device).type == "cuda",
        )
        # Realization-code KL weight. Fixed at ae_rc_beta, or (kl_target > 0)
        # adapted multiplicatively toward the target nats/block on a 100-step
        # EMA of the KL (GECO-style log-space control, at most x e^0.01 per
        # step so it cannot oscillate); identical on every rank because the
        # update reads the rank-averaged KL.
        self.rc_beta = cfg.train.ae_rc_beta if cfg.rc_dim > 0 else 0.0
        self._kl_ema: Optional[float] = None

    def _rc_beta_eff(self) -> float:
        warm = self.cfg.train.ae_rc_warmup
        if warm <= 0:
            return self.rc_beta
        return self.rc_beta * min(1.0, (self._sched_step + 1) / warm)

    def loss(self, batch: Batch) -> tuple[torch.Tensor, dict]:
        return ae_loss(self.model, batch, self.aux if len(self.aux) else None,
                       rc_beta=self._rc_beta_eff())

    def _adapt_rc_beta(self, kl_local: float) -> None:
        t = self.cfg.train
        if self.cfg.rc_dim == 0 or t.ae_rc_kl_target <= 0:
            return
        kl = torch.tensor(kl_local, device=self.device, dtype=torch.float64)
        dist.allreduce_sum(kl)
        kl_mean = kl.item() / dist.world_size()
        self._kl_ema = (kl_mean if self._kl_ema is None
                        else 0.99 * self._kl_ema + 0.01 * kl_mean)
        if self._sched_step < t.ae_rc_warmup:
            return   # let the zero-init code path switch on before steering it
        drive = max(-1.0, min(1.0, (self._kl_ema - t.ae_rc_kl_target) / t.ae_rc_kl_target))
        self.rc_beta = float(min(100.0, max(1e-4, self.rc_beta * math.exp(0.01 * drive))))

    def _lr_scale(self) -> float:
        """Warmup then constant (legacy) or cosine decay to lr_min_ratio."""
        t = self.cfg.train
        warm = max(1, t.warmup_steps)
        i = self._sched_step
        if i + 1 <= warm or t.lr_schedule != "cosine":
            return min(1.0, (i + 1) / warm)
        progress = min(1.0, (i + 1 - warm) / max(1, self.steps - warm))
        return t.lr_min_ratio + (1.0 - t.lr_min_ratio) * 0.5 * (
            1.0 + math.cos(math.pi * progress))

    def train_step(self, batch: Batch, step: int) -> Optional[dict]:
        """Returns None if this batch should be skipped (any rank saw a
        non-finite loss) -- large-model training occasionally hits a single
        pathological batch/weight state; crashing the whole run over one bad
        step is wasteful when skipping it and moving on costs nothing. The
        allreduce makes ranks agree before anyone touches grads/optimizer, so
        a rank-local non-finite value can never desync the collective calls
        below (every rank must reach them together or none may)."""
        with self.autocast:
            loss, metrics = self.loss(batch)
        bad = torch.tensor(0.0 if math.isfinite(float(loss.detach())) else 1.0,
                           device=self.device)
        dist.allreduce_sum(bad)
        if bad.item() > 0:
            return None

        for opt in self.optimizers:
            opt.zero_grad(set_to_none=True)
        loss.backward()
        dist.allreduce_grads(self.params)
        grad_norm = torch.nn.utils.clip_grad_norm_(self.params, self.cfg.train.grad_clip)
        # A finite loss does not imply finite gradients (a numerically stable
        # forward can still have an unstable backward, e.g. near a division
        # boundary) -- this is what let step 9791 silently corrupt the model
        # with a non-finite update even though its OWN loss printed a normal
        # number, only surfacing as "non-finite loss" from the NEXT step on
        # (the now-corrupted weights). Guard the gradients too, same
        # cross-rank consensus so skipping never desyncs the collective calls.
        bad_grad = torch.tensor(
            0.0 if math.isfinite(float(grad_norm)) else 1.0, device=self.device)
        dist.allreduce_sum(bad_grad)
        if bad_grad.item() > 0:
            return None
        scale = self._lr_scale()
        for opt in self.optimizers:
            for g in opt.param_groups:
                g["lr"] = g["base_lr"] * scale
            opt.step()
        self._sched_step += 1
        if self.ema is not None:
            self.ema.update()
        if "kl" in metrics:
            self._adapt_rc_beta(metrics["kl"])

        metrics["lr"] = self.optimizers[0].param_groups[0]["lr"]
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

    def train(self) -> None:
        cfg = self.cfg.train
        if dist.is_main():
            n_params = sum(p.numel() for p in self.params)
            print(f"[ae] training {n_params / 1e6:.2f}M params "
                  f"for {self.steps} steps on {self.device} "
                  f"(world={dist.world_size()})")

        self.model.train()
        batches = cycle(self.train_loader)
        start = time.time()
        for step in range(1, self.steps + 1):
            metrics = self.train_step(next(batches).to(self.device), step)
            if metrics is None:
                if dist.is_main():
                    print(f"[ae] WARNING: non-finite loss at step {step}, "
                          f"skipping this batch (schedule not advanced)", flush=True)
                continue

            if dist.is_main() and (step % cfg.log_every == 0 or step == 1):
                metrics["s/step"] = (time.time() - start) / step
                print(format_metrics(f"[ae] step {step}/{self.steps}", metrics))

            if (dist.is_main() and self.val_loader is not None
                    and cfg.eval_every and step % cfg.eval_every == 0):
                print(format_metrics(
                    f"[ae] eval  {step}/{self.steps}", self.evaluate()
                ))
                self.save(step)

        if self.ema is not None:
            self.ema.copy_to()   # every rank: leave the EMA weights in the model
        if dist.is_main():
            save_checkpoint(self.ckpt_path, self.model, self.steps)
            print(f"[ae] saved -> {self.ckpt_path}"
                  + (" (EMA)" if self.ema is not None else ""))

    def save(self, step: int) -> None:
        """Checkpoint the EMA weights when enabled, without touching training."""
        if self.ema is not None:
            self.ema.swap()
        save_checkpoint(self.ckpt_path, self.model, step)
        if self.ema is not None:
            self.ema.swap()
