"""What the realization codes carry and whether the prior's codes agree with its own states.

Standalone (no existing file changes). One checkpoint per call; run it at every block count K.

  0. Code budget: KL(q(r_k) || N(0,I)) per block and per token on held-out documents, posterior sd, and a check of
     the architectural fact that block k's posterior-mean code is a linear read-out of the NEXT predictive state:
     rc_head is applied to the inclusive state c_k and s_{k+1} = c_k (psf/models/encoder_prefix.py), so
     mu_k = W_mu s_{k+1} + b exactly for k < K (max |difference| is printed).
  1. Teacher-forced writer NLL (nats/token) on the same documents, true states with
       posterior-mean codes / posterior-sampled codes / zero codes / another document's codes /
       codes the PRIOR infers from the true states (flow inpainting with the states given, --no-inpaint to skip),
     and the other-doc / constant states under posterior-mean and zero codes (plan_probe's state sensitivity).
     Derived: code gain = NLL(zero) - NLL(post-mean); the rate the decoder pays for it (KL per token); and the share
     of the gain recovered by the prior's inferred codes, (NLL(zero) - NLL(prior)) / (NLL(zero) - NLL(post-mean)).
  2. Prior self-consistency on --plan-samples plans sampled from the prior (50 Euler steps unless
     --sampling-steps): the prior's code for block k (k < K) against the posterior the encoder's head would assign
     given the prior's OWN next state s_{k+1}: cosine, R^2, RMS error in units of the posterior sd (1.0 = as
     consistent as posterior samples, 0 = the posterior mean). Reference rows: posterior samples of real documents
     (what a perfectly consistent sampler of q would give) and the prior's codes against ANOTHER sample's states.

  python -m scripts.codes_probe --ckpt runs/prior-srpl-rc64-look-k16/prior.pt --num-blocks 16 <arch flags> \
      --out r5_notes/codes_k16.txt
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import torch
import torch.nn.functional as F

from psf.diffusion.flow import sample as flow_sample
from psf.diffusion.flow import sample_inpaint
from psf.models.psf import PSF
from psf.utils.data import make_dataloaders
from psf.utils.misc import get_device, load_checkpoint, set_seed
from scripts import eval_gen


def own_args():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--num-docs", type=int, default=1000)
    p.add_argument("--plan-samples", type=int, default=1000)
    p.add_argument("--out", default="r5_notes/codes.txt")
    p.add_argument("--no-inpaint", action="store_true", help="skip the prior-inferred-codes row")
    p.add_argument("--no-consistency", action="store_true", help="skip section 2 (e.g. states-only priors)")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("-h", "--help", action="store_true")
    mine, rest = p.parse_known_args()
    if mine.help:
        print(__doc__)
        p.print_help()
        print("\nAll remaining flags go to scripts.eval_gen.parse_args (architecture, --gen-batch, --sampling-steps, --seed).")
        sys.exit(0)
    sys.argv = [sys.argv[0]] + rest
    args = eval_gen.parse_args()
    if not any(a == "--gen-batch" or a.startswith("--gen-batch=") for a in rest):
        args.gen_batch = 50
    return mine, args


@torch.no_grad()
def tf_sum(model: PSF, feats, inputs, targets, bm, nf, rc) -> tuple[float, float]:
    logits = model.decoder(feats, inputs, next_first=nf, rc=rc)
    V = logits.shape[-1]
    ce = F.cross_entropy(logits.reshape(-1, V).float(), targets.reshape(-1), reduction="none").reshape(targets.shape)
    m = bm.to(ce.dtype)
    return float((ce * m).sum()), float(m.sum())


def head_stats(model: PSF, s_next: torch.Tensor, R: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Posterior (mu, sd) the encoder's code head assigns to inclusive state s_next ([B, K-1, D])."""
    st = model.encoder.rc_head(s_next.to(model.encoder.rc_head.weight.dtype)).float()
    mu, lv = st[..., :R], st[..., R:].clamp(-8.0, 4.0)
    return mu, torch.exp(0.5 * lv)


def agreement(code: torch.Tensor, mu: torch.Tensor, sd: torch.Tensor) -> tuple[float, float, float, float]:
    """cosine, R^2 (vs the code's own mean), RMS z-error, norm ratio |code|/|mu|, pooled over blocks 1..K-1."""
    c, m = code.float().reshape(-1, code.shape[-1]), mu.float().reshape(-1, mu.shape[-1])
    cos = float(F.cosine_similarity(c, m, dim=-1).mean())
    r2 = float(1.0 - (c - m).pow(2).sum() / (c - c.mean(0, keepdim=True)).pow(2).sum().clamp(min=1e-12))
    z = float(((code.float() - mu) / sd).pow(2).mean().sqrt())
    ratio = float(c.norm(dim=-1).mean() / m.norm(dim=-1).mean().clamp(min=1e-12))
    return cos, r2, z, ratio


def main() -> None:
    mine, args = own_args()
    header = f"# codes_probe: {args.ckpt} | K={args.num_blocks} | {mine.num_docs} docs | {mine.plan_samples} prior plans"
    if mine.dry_run:
        print(header); return
    if not args.ckpt:
        raise SystemExit("--ckpt is required")
    os.makedirs(os.path.dirname(os.path.abspath(mine.out)) or ".", exist_ok=True)
    cfg, _ = eval_gen.config_from_args(args)
    if cfg.rc_dim == 0:
        raise SystemExit("checkpoint has no realization codes (--rc-dim)")
    set_seed(args.seed)
    device = get_device(args.device)
    torch.set_float32_matmul_precision("high")
    model = PSF(cfg).to(device).eval()
    load_checkpoint(args.ckpt, model, device)
    K, R, D = cfg.num_blocks, cfg.rc_dim, cfg.latent_dim
    lines = [header]
    log = lambda s: (print(s, flush=True), lines.append(s))

    _, val_loader = make_dataloaders(cfg)
    batches, n = [], 0
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            batches.append((batch, model.encode(batch.tokens, batch.mask)))
            n += batch.tokens.shape[0]
            if n >= mine.num_docs:
                break
    mean_state = torch.cat([e.block_feats.float() for _, e in batches], 0)[: mine.num_docs].mean(0, keepdim=True)

    # ---------------- 0. code budget and the linear read-out fact ----------------
    kl_sum = kl_blocks = tok = 0.0
    sd_sum = 0.0
    max_diff = 0.0
    post_samples, post_mu, post_sd = [], [], []
    with torch.no_grad():
        for batch, enc in batches:
            valid = enc.rc_valid if getattr(enc, "rc_valid", None) is not None else torch.ones_like(enc.rc_mu[..., 0], dtype=torch.bool)
            kl = model.rc_kl(enc)
            kl_sum += float((kl * valid).sum()); kl_blocks += float(valid.sum())
            tok += float(batch.mask.sum())
            sd_sum += float((torch.exp(0.5 * enc.rc_logvar.float()).mean(-1) * valid).sum())
            mu_hat, sd_hat = head_stats(model, enc.block_feats[:, 1:], R)
            max_diff = max(max_diff, float((mu_hat - enc.rc_mu[:, :-1].float()).abs().max()))
            post_mu.append(mu_hat); post_sd.append(sd_hat)
            post_samples.append(model.sample_rc(enc, "sample")[:, :-1].float())
    log(f"\n## 0. code budget (held-out documents)")
    log(f"  KL per block {kl_sum / kl_blocks:.2f} nats | per token {kl_sum / tok:.4f} nats | mean posterior sd {sd_sum / kl_blocks:.4f}")
    log(f"  linear read-out check: max |rc_head_mu(s_(k+1)) - posterior mean of block k| = {max_diff:.2e} (k < K)")

    # ---------------- 1. teacher-forced NLL ----------------
    rows = ["true | post-mean", "true | post-sample", "true | zero", "true | other-doc codes"]
    if not mine.no_inpaint:
        rows.append("true | prior-inferred")
    rows += ["other-doc | post-mean", "constant | post-mean", "other-doc | zero", "constant | zero"]
    tot = {r: [0.0, 0.0] for r in rows}
    with torch.no_grad():
        for batch, enc in batches:
            inputs, targets = model.block_view(batch.tokens)
            nf = model.boundary_targets(targets)
            bm = batch.mask.reshape(targets.shape)
            f = enc.block_feats
            codes = {"post-mean": enc.rc_mu, "post-sample": model.sample_rc(enc, "sample"),
                     "zero": torch.zeros_like(enc.rc_mu), "other-doc codes": enc.rc_mu.roll(1, dims=0)}
            if not mine.no_inpaint:
                zk = model.denoiser.normalize(model.prior_latents(enc, source="mean"))
                known = torch.zeros_like(zk, dtype=torch.bool)
                known[..., :D] = True
                zf = model.denoiser.denormalize(sample_inpaint(model.denoiser, zk, known, cfg.flow, device,
                                                               cfg_scale=args.cfg_scale))
                codes["prior-inferred"] = model.split_latents(zf)[1].to(enc.rc_mu.dtype)
            states = {"true": f, "other-doc": f.roll(1, dims=0), "constant": mean_state.expand_as(f).to(f.dtype)}
            for r in rows:
                sname, cname = [x.strip() for x in r.split("|")]
                s_, m_ = tf_sum(model, states[sname], inputs, targets, bm, nf, codes[cname])
                tot[r][0] += s_; tot[r][1] += m_
    nll = {r: tot[r][0] / tot[r][1] for r in rows}
    log(f"\n## 1. teacher-forced writer NLL ({int(tot[rows[0]][1])} tokens)")
    for r in rows:
        log(f"  {r:26s} {nll[r]:.4f} nats/token  ppl {math.exp(nll[r]):7.2f}  ({100 * (nll[r] / nll['true | post-mean'] - 1):+.1f}% vs post-mean)")
    gain = nll["true | zero"] - nll["true | post-mean"]
    span = cfg.block_span
    log(f"  code gain NLL(zero)-NLL(post-mean) = {gain:.4f} nats/token = {gain * span:.1f} nats per {span}-token block "
        f"| KL paid {kl_sum / tok:.4f} nats/token")
    log(f"  NLL(post-sample) + KL/token = {nll['true | post-sample'] + kl_sum / tok:.4f}  vs  NLL(zero) {nll['true | zero']:.4f} "
        f"(lower = the code is worth its rate)")
    if not mine.no_inpaint and abs(gain) > 1e-6:
        rec = (nll["true | zero"] - nll["true | prior-inferred"]) / gain
        log(f"  share of the code gain recovered by the prior's inferred codes: {100 * rec:.1f}% of the post-mean gain")
        gain_s = nll["true | zero"] - nll["true | post-sample"]
        if abs(gain_s) > 1e-6:
            rec_s = (nll["true | zero"] - nll["true | prior-inferred"]) / gain_s
            log(f"    ({100 * rec_s:.1f}% of the post-sample gain; like-for-like if the prior was fitted to posterior draws, prior_rc_source=sample, the default)")
    for c in ("post-mean", "zero"):
        base = nll[f"true | {c}"]
        log(f"  state sensitivity with {c} codes: other-doc {100 * (nll[f'other-doc | {c}'] / base - 1):+.1f}%  "
            f"constant {100 * (nll[f'constant | {c}'] / base - 1):+.1f}%")

    # ---------------- 2. prior self-consistency ----------------
    if not mine.no_consistency:
        nlat = K * cfg.num_latent_slots + cfg.num_global_latents
        S, RC = [], []
        with torch.no_grad():
            remaining = mine.plan_samples
            while remaining > 0:
                b = min(args.gen_batch, remaining)
                z = flow_sample(model.denoiser, b, nlat, model.prior_dim, cfg.flow, device,
                                cfg_scale=args.cfg_scale, sde_gamma=args.sde_gamma, noise_scale=args.noise_scale)
                s, rc = model.split_latents(model.denoiser.denormalize(z))
                S.append(s); RC.append(rc)
                remaining -= b
            S, RC = torch.cat(S, 0), torch.cat(RC, 0).float()
            mu_p, sd_p = head_stats(model, S[:, 1:], R)
            mu_o, sd_o = head_stats(model, S.roll(1, dims=0)[:, 1:], R)
        pm, ps, pz = torch.cat(post_mu, 0), torch.cat(post_sd, 0), torch.cat(post_samples, 0)
        log(f"\n## 2. prior self-consistency (blocks 1..K-1; code of block k vs the head's posterior at s_(k+1))")
        log(f"  {'row':54s} {'cos':>6s} {'R^2':>7s} {'rms z':>7s} {'|c|/|mu|':>8s}")
        for name, (c, m, sd) in {
            "reference: posterior samples of real documents": (pz, pm, ps),
            "prior's codes vs head(prior's own next state)": (RC[:, :-1], mu_p, sd_p),
            "prior's codes vs head(another sample's next state)": (RC[:, :-1], mu_o, sd_o),
        }.items():
            cos, r2, z, ratio = agreement(c, m, sd)
            log(f"  {name:54s} {cos:6.3f} {r2:7.3f} {z:7.3f} {ratio:8.3f}")
        log(f"  code rms: prior {RC.pow(2).mean().sqrt():.4f} | posterior samples {pz.pow(2).mean().sqrt():.4f} | "
            f"posterior means {pm.pow(2).mean().sqrt():.4f}")
    open(mine.out, "w", encoding="utf-8").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
