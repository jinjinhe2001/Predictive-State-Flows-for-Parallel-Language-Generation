"""A document likelihood bound (ELBO) for PSF, and its components.

  log p(x) >= E_q[log p(x | z~)] + E_q[log p~(z~)] + H(q)

  The generative model whose likelihood is bounded is PSF with the prior's ODE stopped at time t* < 1: the flow is
  integrated from noise (t = 0) to t* and the writer decodes z~ = z_{t*} / t*, a state carrying a residual noise of
  relative scale s = (1 - t*) / t* in the prior's normalized coordinates (the writer is trained under such decoder
  noise, so this is the model as deployed up to the last (1 - t*) of the ODE). Stopping short of t = 1 keeps the
  density well conditioned: the x0-parametrized velocity v = (x0(z_t, t) - z_t) / (1 - t) is singular at t = 1.
    * q(z~ | x) = N(u(x), s^2 I) in normalized coordinates, u(x) = normalize([f(x) ; mu_r(x)]) -- the encoder's states
      and code means with the same noise the prior's path would leave at t*; H(q) = d/2 log(2 pi e s^2);
    * log p(x | z~): the writer's teacher-forced log-likelihood under z~ (landing target from the document; block k's
      first token conditions only on z~, as in generation);
    * log p~(z~) = d log t* + log p_{t*}(t* z~), with log p_{t*} from the probability-flow ODE integrated from t* back to
      0 (M Euler steps, Hutchinson trace by reverse-mode probes); self-conditioning input fixed to zero (first-pass flow).
  Normalization Jacobians cancel between the prior term and H(q), so everything is computed in normalized coordinates.
  Reported in nats per real T5 token (same-data AR held-out NLL: 2.635).

  python -m scripts.elbo_probe --ckpt runs/prior-srpl-rc64-look-k8/prior.pt <arch flags of eval_gen> \
      --num-docs 1000 --tstars 0.8,0.9,0.95 --ode-steps 100 --probes 2 --out outputs/elbo_k8.txt
"""
from __future__ import annotations

import argparse
import math
import sys

import torch
import torch.nn.functional as F

from psf.models.psf import PSF
from psf.utils.data import load_tokenizer, make_dataloaders
from psf.utils.misc import get_device, load_checkpoint, set_seed
from scripts import eval_gen


def own_args():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--num-docs", type=int, default=1000)
    p.add_argument("--tstars", default="0.8,0.9,0.95")
    p.add_argument("--ode-steps", type=int, default=100)
    p.add_argument("--probes", type=int, default=2)
    p.add_argument("--out", default="outputs/elbo.txt")
    mine, rest = p.parse_known_args()
    sys.argv = [sys.argv[0]] + rest
    return mine, eval_gen.parse_args()


@torch.no_grad()
def writer_logp(model: PSF, feats, rc, inputs, targets, block_mask, nf) -> torch.Tensor:
    """[B] sum of log p(x | z) over the real tokens of each document."""
    logits = model.decoder(feats, inputs, next_first=nf, rc=rc)
    V = logits.shape[-1]
    ce = F.cross_entropy(logits.reshape(-1, V).float(), targets.reshape(-1), reduction="none").reshape(targets.shape)
    return -(ce * block_mask.to(ce.dtype)).sum(dim=(1, 2))


def flow_logp(denoiser, w: torch.Tensor, t_star: float, M: int, probes: int, gen: torch.Generator) -> torch.Tensor:
    """[B] log p_{t*}(w): density of the flow's marginal at time t* (normalized coordinates), reverse Euler t* -> 0."""
    B = w.shape[0]
    dev = w.device
    cfg_w = torch.ones(B, device=dev) if getattr(denoiser, "cfg_tokens", None) is not None else None

    def vel(u, t_scalar):
        t = torch.full((B,), float(t_scalar), device=dev)
        x0 = denoiser(u, t, None, cfg_scale=cfg_w)
        return (x0 - u) / (1.0 - float(t_scalar))

    u = w.clone()
    logdet = torch.zeros(B, device=dev)
    dt = t_star / M
    for i in range(M, 0, -1):                              # t from t* down to dt (velocity evaluated at the upper end)
        t_cur = i * dt
        with torch.enable_grad():
            u_ = u.detach().requires_grad_(True)
            v = vel(u_, t_cur)
            div = torch.zeros(B, device=dev)
            for pi in range(probes):                        # Hutchinson via reverse mode: eps . (J^T eps) = eps^T J eps
                eps = torch.randint(0, 2, u.shape, generator=gen, device=dev).float() * 2 - 1
                g = torch.autograd.grad((v * eps).sum(), u_, retain_graph=pi < probes - 1)[0]
                div += (g * eps).flatten(1).sum(1)
        div = (div / probes).detach()
        v = v.detach()
        logdet += div * dt                                  # log p_{t*}(w) = log p_0(u_0) - int_0^{t*} div v dt
        u = u - v * dt
    d = u.flatten(1).shape[1]
    log_p0 = -0.5 * (u.flatten(1) ** 2).sum(1) - 0.5 * d * math.log(2 * math.pi)
    # round trip: integrate the recovered noise forward again with the same Euler grid; the gap to w is the ODE error
    with torch.no_grad():
        uf = u.clone()
        for i in range(0, M):
            uf = uf + vel(uf, i * dt) * dt
        rt = float(((uf - w) ** 2).flatten(1).mean(1).mean())
    flow_logp.diag = dict(w_sq=float((w.flatten(1) ** 2).mean()), u0_sq=float((u.flatten(1) ** 2).mean()),
                          logdet_per_dim=float((logdet / d).mean()), d=d, roundtrip_mse=rt)
    return log_p0 - logdet


def main() -> None:
    mine, args = own_args()
    cfg, _ = eval_gen.config_from_args(args)
    set_seed(args.seed)
    device = get_device(args.device)
    torch.set_float32_matmul_precision("high")
    load_tokenizer(cfg)
    model = PSF(cfg).to(device).eval()
    load_checkpoint(args.ckpt, model, device)
    den = model.denoiser
    K, D, R = cfg.num_blocks, cfg.latent_dim, cfg.rc_dim
    gen = torch.Generator(device=device).manual_seed(args.seed)
    tstars = [float(x) for x in mine.tstars.split(",")]
    lines = [f"# ELBO probe: {args.ckpt} | K={K} D={D} R={R} | {mine.num_docs} docs | ODE steps {mine.ode_steps}, probes {mine.probes}"]
    log = lambda s: (print(s, flush=True), lines.append(s))

    _, val_loader = make_dataloaders(cfg)
    tot = {t: dict(lp_clean=0.0, lp=0.0, prior=0.0, ent=0.0, tok=0.0, docs=0) for t in tstars}
    n = 0
    for batch in val_loader:
        batch = batch.to(device)
        with torch.no_grad():
            enc = model.encode(batch.tokens, batch.mask)
            inputs, targets = model.block_view(batch.tokens)
            nf = model.boundary_targets(targets)
            bm = batch.mask.reshape(targets.shape)
            ntok = bm.float().sum(dim=(1, 2))
            z_clean = model.prior_latents(enc, "mean").float()          # [B, K, D+R]: what the prior models
            u1 = den.normalize(z_clean)
            d = u1.flatten(1).shape[1]
            s_cl, r_cl = model.split_latents(z_clean)
            lp_clean = writer_logp(model, s_cl.to(enc.block_feats.dtype), r_cl, inputs, targets, bm, nf)
        for ts in tstars:
            sc = (1.0 - ts) / ts
            with torch.no_grad():
                eps = torch.randn(u1.shape, generator=gen, device=device)
                zt = u1 + sc * eps                                        # q sample in normalized coordinates
                s_q, r_q = model.split_latents(den.denormalize(zt))
                lp = writer_logp(model, s_q.to(enc.block_feats.dtype), r_q, inputs, targets, bm, nf)
                ent = 0.5 * d * math.log(2 * math.pi * math.e * sc * sc)
            prior = flow_logp(den, ts * zt, ts, mine.ode_steps, mine.probes, gen).detach() + d * math.log(ts)
            t = tot[ts]
            t["lp_clean"] += float(lp_clean.sum()); t["lp"] += float(lp.sum()); t["prior"] += float(prior.sum())
            t["ent"] += ent * u1.shape[0]; t["tok"] += float(ntok.sum()); t["docs"] += u1.shape[0]
            dg = flow_logp.diag
            print(f"  t*={ts:.2f} | mean w^2 {dg['w_sq']:.3f}  mean u0^2 {dg['u0_sq']:.3f} (N(0,I): 1.0)  logdet/dim {dg['logdet_per_dim']:.3f}  round-trip mse/dim {dg['roundtrip_mse']:.4f}", flush=True)
        n += batch.tokens.shape[0]
        print(f"  {n} docs", flush=True)
        if n >= mine.num_docs:
            break

    log(f"\n## bound in nats per real T5 token ({int(tot[tstars[0]]['tok'])} tokens, {tot[tstars[0]]['docs']} docs; latent dims per doc {K * (D + R)})")
    log(f"  {'t*':>5s} {'noise s':>8s} {'writer NLL(clean)':>18s} {'writer NLL(q)':>14s} {'-log p~(z~)':>12s} {'-H(q)':>8s} {'-ELBO':>8s} {'ppl(T5)':>9s}")
    for ts in tstars:
        t = tot[ts]; T = t["tok"]
        w_clean, w_q, pr, h = -t["lp_clean"] / T, -t["lp"] / T, -t["prior"] / T, -t["ent"] / T
        log(f"  {ts:5.2f} {(1 - ts) / ts:8.3f} {w_clean:18.4f} {w_q:14.4f} {pr:12.4f} {h:8.4f} {w_q + pr + h:8.4f} {math.exp(w_q + pr + h):9.2f}")
    log("  reference: same-data AR held-out NLL 2.635 nats/token (Appendix D.1). The bound is for PSF with the prior's ODE "
        "stopped at t*; the writer then reads a state with residual noise s (relative to the prior's per-coordinate std).")
    open(mine.out, "w", encoding="utf-8").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
