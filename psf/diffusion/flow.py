"""Rectified flow over the block latents.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from psf.config import FlowConfig
from psf.models.denoiser import Denoiser


def sample_time(batch_size: int, cfg: FlowConfig, device) -> Tensor:
    if cfg.time_sampler == "logit_normal":
        normal = torch.randn(batch_size, device=device)
        return torch.sigmoid(normal * cfg.logit_normal_std + cfg.logit_normal_mean)
    return torch.rand(batch_size, device=device)


def sample_cfg_scale(batch_size: int, cfg_min: float, cfg_max: float,
                     dtype, device) -> Tensor:
    """Log-uniform guidance scale in [cfg_min, cfg_max] (ELF's sampler)."""
    u = torch.rand((batch_size,), dtype=dtype, device=device)
    a, b = 1.0 + cfg_min, 1.0 + cfg_max
    return a * torch.exp(u * math.log(b / a)) - 1.0


def interpolate(z0: Tensor, eps: Tensor, t: Tensor) -> Tensor:
    if t.dim() == 2:                       # per-block time [B, K] (diffusion forcing)
        t = t.unsqueeze(-1)
    else:
        t = t.reshape(-1, *([1] * (z0.dim() - 1)))
    return t * z0 + (1.0 - t) * eps


def sample_df_time(B: int, K: int, cfg: FlowConfig, device) -> tuple[Tensor, Tensor]:
    """Per-block times [B, K] plus a [B, K] bool mask of clean-context blocks.
    Independent draws; df_mix_prob rows collapse to one shared time (plain-flow
    rows stay in distribution); df_ordered_prob rows are sorted descending
    (earlier blocks cleaner); in the remaining rows each block is a real clean
    context block (t = 1, excluded from the loss) with prob df_clean_prob."""
    t = sample_time(B * K, cfg, device).view(B, K)
    mix = torch.zeros(B, dtype=torch.bool, device=device)
    if cfg.df_mix_prob > 0:
        mix = torch.rand(B, device=device) < cfg.df_mix_prob
        t = torch.where(mix[:, None], t[:, :1].expand(B, K), t)
    if cfg.df_ordered_prob > 0:
        ordered = torch.rand(B, device=device) < cfg.df_ordered_prob
        t = torch.where(ordered[:, None], t.sort(dim=1, descending=True).values, t)
    clean = torch.zeros(B, K, dtype=torch.bool, device=device)
    if cfg.df_clean_prob > 0:
        clean = (torch.rand(B, K, device=device) < cfg.df_clean_prob) & ~mix[:, None]
        t = torch.where(clean, torch.ones_like(t), t)
    return t, clean


def flow_loss(denoiser: Denoiser, z0: Tensor, cfg: FlowConfig) -> Tensor:
    t = sample_time(z0.shape[0], cfg, z0.device)
    eps = torch.randn_like(z0)
    z_t = interpolate(z0, eps, t)

    self_cond = None
    if denoiser.self_cond_proj is not None and cfg.self_cond_prob > 0:
        if torch.rand(()) < cfg.self_cond_prob:
            with torch.no_grad():
                self_cond = denoiser(z_t, t, None).detach()

    pred = denoiser(z_t, t, self_cond)
    return torch.mean((pred - z0) ** 2)


def flow_loss_improved(denoiser: Denoiser, z0: Tensor, cfg: FlowConfig) -> Tensor:
    """ELF-style objective. With sc_cfg, the guided target is
    z0 + (1 - 1/w) * (x_cond - x_uncond), the x0-space equivalent of ELF's
    v-space target (the map x -> v is affine in x at fixed z_t, t)."""
    B = z0.shape[0]
    device = z0.device
    excluded = None                       # [B, K] bool: context blocks kept out of the loss
    if cfg.diffusion_forcing:
        t, clean = sample_df_time(B, z0.shape[1], cfg, device)    # [B, K]
        if bool(clean.any()):
            excluded = clean
    else:
        t = sample_time(B, cfg, device)
    eps = torch.randn_like(z0)
    z_t = interpolate(z0, eps, t)

    if cfg.diffusion_forcing and cfg.df_rollout_prob > 0:
        K = z0.shape[1]
        rows = torch.rand(B, device=device) < cfg.df_rollout_prob
        ctx = (torch.rand(B, K, device=device) < cfg.df_rollout_frac) & rows[:, None]
        if bool(ctx.any()):
            with torch.no_grad():
                x_self = denoiser(z_t, t, None)          # the model's own clean guess
            c3 = ctx.unsqueeze(-1)
            z_t = torch.where(c3, x_self.to(z_t.dtype), z_t)
            t = torch.where(ctx, torch.ones_like(t), t)  # context blocks sit at time 1
            excluded = ctx if excluded is None else (excluded | ctx)
    loss_mask = None                      # [B, K, 1] float, 1 = block enters the loss
    if excluded is not None:
        loss_mask = (~excluded).to(z0.dtype).unsqueeze(-1)

    use_sc = denoiser.self_cond_proj is not None and cfg.self_cond_prob > 0
    w = None
    if cfg.sc_cfg and use_sc:
        w = sample_cfg_scale(B, cfg.cfg_min, cfg.cfg_max, z0.dtype, device)
    # Prefix-drop CFG: some rows train context-free (each block sees only
    # itself), giving the inference-time guidance a trained "uncond" mode.
    drop = None
    if cfg.ctx_drop_prob > 0:
        drop = torch.rand(B, device=device) < cfg.ctx_drop_prob

    self_cond = None
    target = z0
    if use_sc:
        with torch.no_grad():
            x_uncond = denoiser(z_t, t, None, cfg_scale=w, ctx_drop=drop)
        sc_mask = (torch.rand(B, device=device) < cfg.self_cond_prob)
        sc_mask3 = sc_mask.to(z0.dtype).view(B, 1, 1)
        self_cond = x_uncond * sc_mask3   # zeros = "no self-cond" input
        if w is not None:
            with torch.no_grad():
                x_cond = denoiser(z_t, t, x_uncond, cfg_scale=w, ctx_drop=drop)
            guidance = (1.0 - 1.0 / w).view(B, 1, 1) * (x_cond - x_uncond)
            target = (z0 + guidance * sc_mask3).detach()

    pred = denoiser(z_t, t, self_cond, cfg_scale=w, ctx_drop=drop)
    err = (pred.float() - target.float()) ** 2
    if cfg.objective == "v":
        weight = 1.0 / torch.clamp(1.0 - t.float(), min=cfg.t_eps).pow(2)
        weight = weight.unsqueeze(-1) if t.dim() == 2 else weight.view(B, 1, 1)
        err = err * weight
    if loss_mask is not None:
        return (err * loss_mask).sum() / (loss_mask.sum() * err.shape[-1]).clamp_min(1.0)
    return err.mean()


@torch.no_grad()
def sample(denoiser: Denoiser, num_samples: int, num_blocks: int, latent_dim: int,
           cfg: FlowConfig, device, cfg_scale: float = 1.0,
           sde_gamma: float = 0.0, noise_scale: float = 1.0,
           z0_real: Tensor | None = None, t_start: float = 0.0) -> Tensor:
    """``noise_scale`` < 1 draws the initial noise (and SDE churn noise) from
    N(0, noise_scale^2 I) — a truncation-style quality/diversity knob.

    ``z0_real``/``t_start`` (SDEdit-style partial denoising): corrupt a REAL
    (normalized) latent to time ``t_start`` instead of starting from pure
    noise at t=0, then only integrate the remaining [t_start, 1] stretch.
    t_start=0 is exactly the unconditional default; t_start=1 would be the
    real latent itself (recon, no denoiser call at all). Sweeping t_start
    traces the path between those two endpoints instead of only comparing
    them, isolating how much of the joint structure the denoiser recovers
    given a head start of real signal."""
    was_training = denoiser.training
    denoiser.eval()

    if cfg.df_stagger > 0.0 and z0_real is None:
        z = sample_staggered(denoiser, num_samples, num_blocks, latent_dim, cfg, device,
                             cfg_scale=cfg_scale, sde_gamma=sde_gamma, noise_scale=noise_scale)
        denoiser.train(was_training)
        return z

    if z0_real is not None:
        eps = torch.randn_like(z0_real) * noise_scale
        t0 = torch.full((num_samples,), float(t_start), device=device)
        z = interpolate(z0_real, eps, t0)
        grid = torch.linspace(float(t_start), 1.0, cfg.num_sampling_steps + 1, device=device)
    else:
        z = torch.randn(num_samples, num_blocks, latent_dim, device=device) * noise_scale
        grid = torch.linspace(0.0, 1.0, cfg.num_sampling_steps + 1, device=device)
    w = None
    if getattr(denoiser, "cfg_tokens", None) is not None:
        w = torch.full((num_samples,), float(cfg_scale), device=device)
    self_cond = None
    for i in range(cfg.num_sampling_steps):
        t_cur = float(grid[i])
        t_next = float(grid[i + 1])
        if sde_gamma > 0.0 and i > 0:
            # ELF-style churn: renoise back to alpha * t, then step forward.
            alpha = max(0.0, min(1.0, 1.0 - sde_gamma * (t_next - t_cur)))
            z = alpha * z + (1.0 - alpha) * noise_scale * torch.randn_like(z)
            t_cur = alpha * t_cur
        t = torch.full((num_samples,), t_cur, device=device)
        if cfg_scale != 1.0 and w is None:
            # Prefix-drop CFG (two-pass): guide the full-context prediction
            # away from the per-block context-free one.
            x_full = denoiser(z, t, self_cond)
            x_free = denoiser(z, t, self_cond,
                              ctx_drop=torch.ones(num_samples, dtype=torch.bool,
                                                  device=device))
            x0 = x_free + cfg_scale * (x_full - x_free)
        else:
            x0 = denoiser(z, t, self_cond, cfg_scale=w)
        if denoiser.self_cond_proj is not None:
            self_cond = x0
        if t_next >= 1.0:
            z = x0
        else:
            v = (x0 - z) / max(1.0 - t_cur, 1e-4)
            z = z + v * (t_next - t_cur)

    denoiser.train(was_training)
    return z


@torch.no_grad()
def sample_inpaint(denoiser: Denoiser, z_known: Tensor, known: Tensor, cfg: FlowConfig, device,
                   cfg_scale: float = 1.0, noise_scale: float = 1.0, resample: int = 0) -> Tensor:
    """Conditional sampling by replacement (RePaint-style), no retraining.

    ``z_known`` [B, K, D] holds the NORMALIZED latents that are given (prompt blocks: their prefix
    states and codes, plus the state of the first block to be written, which is a function of the
    prompt alone); ``known`` [B, K, D] bool marks which entries are given. Every Euler step first
    predicts as in ``sample`` and then overwrites the given entries with their exact noisy value
    t*z_known + (1-t)*eps at the new time, so the free entries are denoised in the company of the
    true conditioning at all times. Returns the normalized [B, K, D] plan with the given entries
    set exactly (t = 1)."""
    was_training = denoiser.training
    denoiser.eval()
    B = z_known.shape[0]
    eps = torch.randn_like(z_known) * noise_scale
    z = eps.clone()
    grid = torch.linspace(0.0, 1.0, cfg.num_sampling_steps + 1, device=device)
    w = None
    if getattr(denoiser, "cfg_tokens", None) is not None:
        w = torch.full((B,), float(cfg_scale), device=device)
    self_cond = None
    for i in range(cfg.num_sampling_steps):
        t_cur, t_next = float(grid[i]), float(grid[i + 1])
        # RePaint-style resampling: after a step, re-noise back to t_cur and step again ``resample``
        # times, so the free entries get several chances to harmonize with the given ones.
        for r in range(resample + 1):
            t = torch.full((B,), t_cur, device=device)
            x0 = denoiser(z, t, self_cond, cfg_scale=w)
            if denoiser.self_cond_proj is not None:
                self_cond = x0
            if t_next >= 1.0:
                z_new = x0
            else:
                z_new = z + (x0 - z) / max(1.0 - t_cur, 1e-4) * (t_next - t_cur)
            z_cond = interpolate(z_known, eps, torch.full((B,), t_next, device=device))
            z_new = torch.where(known, z_cond, z_new)
            if r < resample and t_next < 1.0:
                # jump back: z_{t_cur} = (t_cur/t_next) z_{t_next} + fresh noise for the rest of the variance
                a_ = t_cur / t_next
                z = a_ * z_new + math.sqrt(max(0.0, (1.0 - t_cur) ** 2 - (a_ * (1.0 - t_next)) ** 2)) * torch.randn_like(z_new) * noise_scale
            else:
                z = z_new
    denoiser.train(was_training)
    return z


@torch.no_grad()
def sample_staggered(denoiser: Denoiser, num_samples: int, num_blocks: int, latent_dim: int,
                     cfg: FlowConfig, device, cfg_scale: float = 1.0,
                     sde_gamma: float = 0.0, noise_scale: float = 1.0) -> Tensor:
    """Pyramid / staggered reveal (diffusion-forcing sampler): block k's clock is
    t_k = clip(t * (1 + lam) - lam * off_k, 0, 1), off_k in [0, 1] (random
    permutation per sample, or left-to-right), so blocks with a small offset
    finish first and later blocks are denoised next to nearly clean neighbours.
    Per-block SDE churn follows the same pull-back rule as the shared-time path.
    Total denoiser calls are unchanged (num_sampling_steps)."""
    B, K = num_samples, num_blocks
    lam = cfg.df_stagger
    z = torch.randn(B, K, latent_dim, device=device) * noise_scale
    grid = torch.linspace(0.0, 1.0, cfg.num_sampling_steps + 1, device=device)
    if cfg.df_order == "random":
        off = torch.argsort(torch.rand(B, K, device=device), dim=1).float() / max(K - 1, 1)
    else:
        off = (torch.arange(K, device=device, dtype=torch.float32) / max(K - 1, 1)).expand(B, -1)
    w = None
    if getattr(denoiser, "cfg_tokens", None) is not None:
        w = torch.full((B,), float(cfg_scale), device=device)
    self_cond = None
    for i in range(cfg.num_sampling_steps):
        t0k = (grid[i] * (1.0 + lam) - lam * off).clamp(0.0, 1.0)         # [B, K]
        t1k = (grid[i + 1] * (1.0 + lam) - lam * off).clamp(0.0, 1.0)
        dtk = (t1k - t0k).clamp_min(0.0)
        if sde_gamma > 0.0 and i > 0:
            alpha = (1.0 - sde_gamma * dtk).clamp(0.0, 1.0)                # [B, K]
            t0k = alpha * t0k
            a = alpha.unsqueeze(-1)
            z = a * z + (1.0 - a) * noise_scale * torch.randn_like(z)
        x0 = denoiser(z, t0k, self_cond, cfg_scale=w)
        if denoiser.self_cond_proj is not None:
            self_cond = x0
        v = (x0 - z) / (1.0 - t0k).clamp_min(1e-4).unsqueeze(-1)
        z_step = z + v * (t1k - t0k).unsqueeze(-1)
        done = (t1k >= 1.0).unsqueeze(-1)                                   # finished blocks snap to x0
        z = torch.where(done, x0, z_step)
    return z
