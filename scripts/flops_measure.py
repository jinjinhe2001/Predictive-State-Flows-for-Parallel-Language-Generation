"""Measured FLOPs (torch ``FlopCounterMode``) for one-shot parallel generation.

Counts *real dispatched ops* (matmul/addmm/bmm/scaled_dot_product_attention)
over the ACTUAL generation code path -- prior flow-sampling (NFE forwards over K
latents) + writer decode -- at batch=1, so printed GFLOPs are per sample. It uses the
same tool as the ELF comparison harness, so the two sides are directly comparable.

FlopCounterMode counts a multiply-add as 2 FLOPs for matmul/sdpa (true-FLOP
convention). Example:
  CUDA_VISIBLE_DEVICES=0 python -m scripts.flops_measure --blocks 16 32 64
"""
from __future__ import annotations

import argparse

import torch
from torch.utils.flop_counter import FlopCounterMode

from psf.config import Config
from psf.diffusion.flow import sample as flow_sample
from psf.utils.misc import get_device, set_seed
from scripts.latency_probe import build_model, generate_cached

GF = 1e9


def count(fn) -> float:
    """Total FLOPs (in GF) of one call to ``fn`` under FlopCounterMode."""
    fc = FlopCounterMode(display=False)
    with torch.no_grad(), fc:
        fn()
    return fc.get_total_flops() / GF


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--blocks", type=int, nargs="+", default=[16, 32, 64])
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    device = get_device(args.device)
    torch.set_float32_matmul_precision("high")
    bob = Config().data.bob_token_id
    B = 1  # per-sample FLOPs

    print(f"device={device}  batch={B} (per-sample FLOPs, GF = 1e9 true-FLOPs)\n")
    hdr = (f"{'K':>3} {'S':>3} | {'prior GF':>9} {'writer kv GF':>12} "
           f"{'writer nc GF':>12} | {'TOTAL kv':>9} {'TOTAL nc':>9} "
           f"{'serial':>14}")
    print(hdr)
    print("-" * len(hdr))
    for K in args.blocks:
        set_seed(args.seed)
        model, cfg = build_model(K, device)
        dec = model.decoder
        S = cfg.block_span
        nlat = cfg.num_blocks * cfg.num_latent_slots + cfg.num_global_latents
        z = flow_sample(model.denoiser, B, nlat, model.prior_dim, cfg.flow, device)
        s, rc = model.split_latents(model.denoiser.denormalize(z))

        g_prior = count(lambda: flow_sample(
            model.denoiser, B, nlat, model.prior_dim, cfg.flow, device))
        g_kv = count(lambda: generate_cached(dec, s, bob, rc, z_level=0.0))
        g_nc = count(lambda: dec.generate(s, bob, temperature=0.0, z_level=0.0, rc=rc))

        steps = f"S{S}+NFE{cfg.flow.num_sampling_steps}"
        print(f"{K:>3} {S:>3} | {g_prior:>9.1f} {g_kv:>12.1f} {g_nc:>12.1f} | "
              f"{g_prior + g_kv:>9.1f} {g_prior + g_nc:>9.1f} {steps:>14}")
        del model, z, s, rc
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print("\nprior = flow sampling (NFE denoiser forwards over K latents); "
          "writer kv = cached incremental decode; writer nc = current no-cache generate.")


if __name__ == "__main__":
    main()
