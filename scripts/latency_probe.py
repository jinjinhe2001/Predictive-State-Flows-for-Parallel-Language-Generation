"""Wall-clock latency for one-shot parallel generation across block granularities.

The current ``Decoder.generate`` has no KV cache: every within-block step re-runs a
forward over the whole growing prefix (O(S^2) work per block), so its wall-clock
over-penalises the coarse-block configs (large S). This probe adds a *cached*
incremental decode path -- reusing the trained decoder's own submodules, no
change to the model -- and times all three: prior (flow sampling, NFE serial
forwards over K latents), writer no-cache (current impl), writer cached.

The cache is valid for exactly this decode config (verified by asserting a
greedy-token match against ``Decoder.generate``):
  --causal-latents --boundary-tokens (boundary_window 1) --level-cond
  --rc-dim 64 --rc-lookahead   ; prev_window OFF, landing_state OFF,
  landing_cfg OFF, cross_before_self OFF, num_global 0, num_slots 1.
Everything except the token embedding and the self-attention K/V is a static
per-generation broadcast: state injection, rc_in/rc_next, the noise-level
embedding (z_level constant), the causal-latent cross-attn K/V and mask, and --
because boundary_window is 1 -- the landing target (block k+1's first token is
sampled at step 0, then injected unchanged into every position >= 1). So the
cache is a standard causal incremental decode; ``_landing`` was designed so
position n reads only already-sampled slots, hence no back-injection hazard.

Run (studio venv, one free GPU):
  CUDA_VISIBLE_DEVICES=0 python -m scripts.latency_probe --blocks 16 32 64 --batch 50
"""
from __future__ import annotations

import argparse
import time

import torch
import torch.nn.functional as F

from psf.config import Config
from psf.diffusion.flow import sample as flow_sample
from psf.models.psf import PSF
from psf.utils.misc import get_device, load_checkpoint, set_seed

# The three record checkpoints (prior.pt holds the whole stage-2 model).
CKPT = {
    8: "runs/prior-srpl-rc64-look-k8/prior.pt",     # S=128
    16: "runs/prior-srpl-rc64-look-k16/prior.pt",   # S=64
    32: "runs/prior-srpl-rc64-look-k32/prior.pt",   # S=32
    64: "runs/prior-srpl-rc64-look/prior.pt",       # S=16
}


def build_model(num_blocks: int, device) -> tuple[PSF, Config]:
    """Replicate eval_gen's config assembly for the record ARCH flags."""
    cfg = Config()
    cfg.num_blocks = num_blocks
    cfg.decoder.depth = 18
    cfg.denoiser.bottleneck_dim = 256
    cfg.__post_init__()
    cfg.encoder_type = "prefix"
    cfg.decoder.causal_latents = True
    cfg.decoder.boundary_tokens = True
    cfg.decoder.boundary_window = 1
    cfg.decoder.noise_level_cond = True     # --level-cond
    cfg.rc_dim = 64
    cfg.decoder.rc_lookahead = True         # enables rc_next
    cfg.denoiser.num_cfg_tokens = 4         # --sc-cfg
    cfg.flow.num_sampling_steps = 50
    model = PSF(cfg).to(device).eval()
    load_checkpoint(CKPT[num_blocks], model, device)
    return model, cfg


@torch.no_grad()
def generate_cached(dec, z: torch.Tensor, bob_id: int, rc: torch.Tensor,
                    z_level: float = 0.0, return_logits: bool = False):
    """Cached incremental greedy decode. Returns [B, K, S] token ids.

    Mirrors Decoder._prepare/_run for the record config (G=0, M=1, W=1,
    prev_window off, landing_state off, landing_cfg off, self-then-cross).
    """
    B, K, D = z.shape
    S = dec.block_span
    device = z.device
    layers = dec.layers

    # --- static conditioning (position-independent) -------------------------
    zc = z
    if dec.noise_level_cond:
        levels = torch.full((B, K), float(z_level), device=device, dtype=z.dtype)
        lv = torch.stack([levels, 1.0 - levels], dim=-1).to(z.dtype)
        zc = zc + dec.level_emb(lv)
    if dec.rc_mem is not None:
        zc = zc + dec.rc_mem(rc.to(zc.dtype))
    inject = zc                                             # [B, K, D] state per block
    add = inject.clone()
    if dec.rc_in is not None:
        add = add + dec.rc_in(rc.to(add.dtype))
    if dec.rc_next is not None:
        r_next = torch.cat([rc[:, 1:], torch.zeros_like(rc[:, :1])], dim=1)
        add = add + dec.rc_next(r_next.to(add.dtype))       # [B, K, D]

    # --- static cross-attention K/V + mask (memory = zc) --------------------
    qk_pos = torch.arange(K, device=device)
    cross = []
    for layer in layers:
        kk, vv = layer.cross_kv(zc).chunk(2, dim=-1)        # [B, K, D]
        kk = dec.block_rope.rotate(layer._heads(kk), qk_pos)
        vv = layer._heads(vv)                               # [B, H, K, hd]
        cross.append((kk, vv))
    cmask = (qk_pos[None, :] <= qk_pos[:, None])            # [Kq, Kk] causal_latents

    self_k: list = [None] * len(layers)
    self_v: list = [None] * len(layers)
    landing_add = None
    tok = torch.full((B, K), bob_id, dtype=torch.long, device=device)
    outs = []
    logit_hist = []
    for n in range(S):
        h = dec.token_emb(tok) + add                        # [B, K, D]
        if n >= 1 and landing_add is not None:
            h = h + landing_add
        hbk = h.reshape(B * K, 1, D)
        pos = torch.tensor([n], device=device)
        for li, layer in enumerate(layers):
            # self-attention (causal, cached)
            q, k, v = layer.self_qkv(layer.norm1(hbk)).chunk(3, dim=-1)
            q = dec.within_rope.rotate(layer._heads(q), pos)
            k = dec.within_rope.rotate(layer._heads(k), pos)
            v = layer._heads(v)
            if self_k[li] is None:
                self_k[li], self_v[li] = k, v
            else:
                self_k[li] = torch.cat([self_k[li], k], dim=2)
                self_v[li] = torch.cat([self_v[li], v], dim=2)
            a = F.scaled_dot_product_attention(q, self_k[li], self_v[li])
            hbk = hbk + layer.self_out(layer._merge(a))
            # cross-attention (static memory)
            qc = layer._heads(layer.cross_q(layer.norm2(hbk)).reshape(B, K, D))
            qc = dec.block_rope.rotate(qc, qk_pos)
            kk, vv = cross[li]
            ac = F.scaled_dot_product_attention(qc, kk, vv, attn_mask=cmask)
            hbk = hbk + layer.cross_out(layer._merge(ac).reshape(B * K, 1, D))
            # mlp
            hbk = hbk + layer.mlp(layer.norm3(hbk))
        logits = dec.lm_head(dec.final_norm(hbk)).reshape(B, K, -1)
        nxt = logits.argmax(-1)                             # greedy [B, K]
        outs.append(nxt)
        if return_logits:
            logit_hist.append(logits)
        if n == 0 and dec.boundary_tokens:
            nf = torch.full((B, K), -1, dtype=torch.long, device=device)
            nf[:, :-1] = nxt[:, 1:]                          # block k lands on block k+1's x0
            known = (nf >= 0).unsqueeze(-1)
            v0 = dec.tgt_proj(dec.token_emb(nf.clamp_min(0)))
            landing_add = torch.where(known, v0, dec.no_target.view(1, 1, -1))
        tok = nxt
    toks = torch.stack(outs, dim=2)
    if return_logits:
        return toks, torch.stack(logit_hist, dim=2)         # [B,K,S], [B,K,S,V]
    return toks


def cuda_time(fn, iters: int, warmup: int, device) -> float:
    """Mean wall-clock seconds over ``iters`` calls after ``warmup``."""
    for _ in range(warmup):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--blocks", type=int, nargs="+", default=[16, 32, 64])
    p.add_argument("--batch", type=int, default=50, help="docs per forward (eval used 50)")
    p.add_argument("--iters", type=int, default=5)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    device = get_device(args.device)
    torch.set_float32_matmul_precision("high")
    B = args.batch
    bob = Config().data.bob_token_id

    print(f"device={device}  batch={B}  iters={args.iters} (warmup {args.warmup})\n")
    hdr = (f"{'K':>3} {'S':>3} | {'prior ms':>9} {'writer nc ms':>12} "
           f"{'writer kv ms':>12} | {'total nc':>9} {'total kv':>9} "
           f"{'ms/sample kv':>12} | {'parity':>10}")
    print(hdr)
    print("-" * len(hdr))
    for K in args.blocks:
        set_seed(args.seed)
        model, cfg = build_model(K, device)
        dec = model.decoder
        S = cfg.block_span
        nlat = cfg.num_blocks * cfg.num_latent_slots + cfg.num_global_latents

        # ---- correctness check at a small batch (logits [B,K,S,V] are huge at
        # V=32100, so never materialise them at the timing batch) -------------
        Bc = min(B, 8)
        zc = flow_sample(model.denoiser, Bc, nlat, model.prior_dim, cfg.flow, device)
        sc, rcc = model.split_latents(model.denoiser.denormalize(zc))
        ref = dec.generate(sc, bob, temperature=0.0, z_level=0.0, rc=rcc)
        cac, cac_logits = generate_cached(dec, sc, bob, rcc, z_level=0.0, return_logits=True)
        match = (ref == cac).float().mean().item()          # autoregressive rollout match
        # cached logits vs teacher-forced Decoder.forward on the produced seq
        # (both deterministic; isolates fp noise from a bug -- a real bug shows
        # a large logit gap, ties show ~0).
        prefix = torch.cat([torch.full((Bc, K, 1), bob, dtype=torch.long, device=device),
                            cac[:, :, :-1]], dim=2)
        nf = torch.full((Bc, K), -1, dtype=torch.long, device=device)
        nf[:, :-1] = cac[:, 1:, 0]                           # W=1 landing target
        levels = torch.full((Bc, K), 0.0, device=device)
        ref_logits = dec.forward(sc, prefix, next_first=nf, levels=levels, rc=rcc)
        dmax = (ref_logits - cac_logits).abs().max().item()
        amatch = (ref_logits.argmax(-1) == cac_logits.argmax(-1)).float().mean().item()
        parity = f"{100 * match:.2f}%"
        print(f"  [check K={K} b={Bc}] cached-vs-forward: max|Δlogit|={dmax:.2e}  "
              f"argmax match={100 * amatch:.3f}%  (generate-greedy rollout {parity})")
        del zc, sc, rcc, ref, cac, cac_logits, ref_logits, prefix
        if device.type == "cuda":
            torch.cuda.empty_cache()

        # one real latent draw for timing
        z = flow_sample(model.denoiser, B, nlat, model.prior_dim, cfg.flow, device)
        s, rc = model.split_latents(model.denoiser.denormalize(z))

        # ---- timings --------------------------------------------------------
        t_prior = cuda_time(
            lambda: flow_sample(model.denoiser, B, nlat, model.prior_dim, cfg.flow, device),
            args.iters, args.warmup, device)
        t_nc = cuda_time(
            lambda: dec.generate(s, bob, temperature=0.0, z_level=0.0, rc=rc),
            args.iters, args.warmup, device)
        t_kv = cuda_time(
            lambda: generate_cached(dec, s, bob, rc, z_level=0.0),
            args.iters, args.warmup, device)

        ms = lambda t: t * 1e3
        tot_nc, tot_kv = t_prior + t_nc, t_prior + t_kv
        print(f"{K:>3} {S:>3} | {ms(t_prior):>9.1f} {ms(t_nc):>12.1f} "
              f"{ms(t_kv):>12.1f} | {ms(tot_nc):>9.1f} {ms(tot_kv):>9.1f} "
              f"{ms(tot_kv) / B:>12.2f} | {parity:>10}")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print("\nprior = flow sampling (50 NFE over K latents); writer nc = current "
          "no-cache generate; writer kv = cached incremental decode.")
    print("ms/sample kv = total_kv / batch (K blocks decode in parallel within a batch).")


if __name__ == "__main__":
    main()
