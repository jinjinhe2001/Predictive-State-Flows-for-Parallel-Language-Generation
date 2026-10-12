"""Unified compute benchmark for our models: FLOPs / latency / throughput, one-shot and with refinement.

Protocol (same as the cross-model compute table in the paper): one 1024-token document; FLOPs = torch
FlopCounterMode over the real generation path at batch 1 (prior flow sampling, NFE denoiser
forwards over K latents + cached incremental writer decode); latency = eager, tf32, single GPU,
median-of-``iters`` wall-clock after warmup with cuda.synchronize; throughput = documents per second
at batch ``--batches`` (default 1 and 32). A refinement round = re-encode the generated document
(encoder forward) + decode again; ``R1``/``R2`` = one-shot plus 1 / 2 rounds.

  --arch record : the D=512 record checkpoints (K=64/32/16/8, weights loaded; timings are
                  shape-determined so weights do not matter, but parity is real)
  --arch big    : the D=1024 writer (18 layers) + 308M denoiser (1024x24x16, bottleneck 256),
                  random weights (the Big version's compute before it is trained)

  CUDA_VISIBLE_DEVICES=0 python -m scripts.bench_ours --arch record --blocks 64 32 16 8 --batches 1 32
Writes JSON lines to --out (default bench/ours_<arch>.jsonl) and prints a table.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time

import torch
from torch.utils.flop_counter import FlopCounterMode

from psf.config import Config
from psf.diffusion.flow import sample as flow_sample
from psf.models.psf import PSF
from psf.utils.misc import get_device, load_checkpoint, set_seed
from scripts.latency_probe import CKPT, generate_cached

GF = 1e9


def build(K: int, arch: str, device) -> tuple[PSF, Config]:
    cfg = Config()
    cfg.num_blocks = K
    cfg.decoder.depth = 18
    cfg.denoiser.bottleneck_dim = 256
    if arch == "big":
        cfg.latent_dim = 1024
        cfg.denoiser.hidden_size = 1024
        cfg.denoiser.depth = 24
        cfg.denoiser.num_heads = 16
    cfg.__post_init__()
    cfg.encoder_type = "prefix"
    cfg.decoder.causal_latents = True
    cfg.decoder.boundary_tokens = True
    cfg.decoder.boundary_window = 1
    cfg.decoder.noise_level_cond = True
    cfg.rc_dim = 64
    cfg.decoder.rc_lookahead = True
    cfg.denoiser.num_cfg_tokens = 4
    cfg.flow.num_sampling_steps = 50
    model = PSF(cfg).to(device).eval()
    if arch == "record":
        load_checkpoint(CKPT[K], model, device)
    return model, cfg


def count(fn) -> float:
    fc = FlopCounterMode(display=False)
    with torch.no_grad(), fc:
        fn()
    return fc.get_total_flops() / GF


def timed(fn, iters: int, warmup: int) -> float:
    """Median wall-clock seconds of ``fn`` over ``iters`` calls after ``warmup``."""
    with torch.no_grad():
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        ts = []
        for _ in range(iters):
            t0 = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


def nparams(m) -> float:
    return sum(p.numel() for p in m.parameters()) / 1e6


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arch", choices=["record", "big"], default="record")
    p.add_argument("--blocks", type=int, nargs="+", default=[64, 32, 16, 8])
    p.add_argument("--batches", type=int, nargs="+", default=[1, 32])
    p.add_argument("--iters", type=int, default=5)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--rounds", type=int, default=2, help="refinement rounds to report (R1..R<rounds>)")
    p.add_argument("--out", default=None)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    device = get_device(a.device)
    torch.set_float32_matmul_precision("high")
    out = a.out or f"bench/ours_{a.arch}.jsonl"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    bob = Config().data.bob_token_id
    print(f"device={torch.cuda.get_device_name(0)}  arch={a.arch}  eager tf32  iters={a.iters} (median)  batches={a.batches}")
    rows = []
    for K in a.blocks:
        set_seed(a.seed)
        model, cfg = build(K, a.arch, device)
        dec = model.decoder
        S = cfg.block_span
        L = cfg.max_seq_len
        nlat = cfg.num_blocks * cfg.num_latent_slots + cfg.num_global_latents
        params = {"encoder": nparams(model.encoder), "decoder": nparams(model.decoder), "denoiser": nparams(model.denoiser)}
        params["total"] = sum(params.values())

        def prior(B):
            return flow_sample(model.denoiser, B, nlat, model.prior_dim, cfg.flow, device)

        def latents(B):
            return model.split_latents(model.denoiser.denormalize(prior(B)))

        # ---- FLOPs at batch 1 -------------------------------------------------
        s1, rc1 = latents(1)
        g_prior = count(lambda: prior(1))
        g_dec = count(lambda: generate_cached(dec, s1, bob, rc1, z_level=0.0))
        tok1 = generate_cached(dec, s1, bob, rc1, z_level=0.0).reshape(1, L)
        g_enc = count(lambda: model.encode(tok1))
        flops = {"prior": g_prior, "decode": g_dec, "encode": g_enc, "oneshot": g_prior + g_dec,
                 "refine_round": g_enc + g_dec}
        for r in range(1, a.rounds + 1):
            flops[f"R{r}"] = flops["oneshot"] + r * flops["refine_round"]

        # ---- latency / throughput per batch -----------------------------------
        lat = {}
        thr = {}
        for B in a.batches:
            s, rc = latents(B)
            toks = generate_cached(dec, s, bob, rc, z_level=0.0).reshape(B, L)
            iters = a.iters if B == 1 else max(2, a.iters // 2)
            t_prior = timed(lambda: prior(B), iters, a.warmup)
            t_dec = timed(lambda: generate_cached(dec, s, bob, rc, z_level=0.0), iters, a.warmup)
            t_enc = timed(lambda: model.encode(toks), iters, a.warmup)
            d = {"prior": t_prior, "decode": t_dec, "encode": t_enc, "oneshot": t_prior + t_dec,
                 "refine_round": t_enc + t_dec}
            for r in range(1, a.rounds + 1):
                d[f"R{r}"] = d["oneshot"] + r * d["refine_round"]
            lat[f"b{B}"] = {k: v * 1e3 for k, v in d.items()}                    # ms per batch
            thr[f"b{B}"] = {k: B / d[k] for k in ("oneshot",) + tuple(f"R{r}" for r in range(1, a.rounds + 1))}
            del s, rc, toks
            torch.cuda.empty_cache()

        row = {"system": "ours", "arch": a.arch, "D": cfg.latent_dim, "K": K, "S": S, "params_M": params,
               "flops_GF": flops, "latency_ms": lat, "docs_per_s": thr,
               "serial": {"oneshot": S + cfg.flow.num_sampling_steps, "refine_round": S + 1},
               "precision": "tf32", "mode": "eager, KV-cached writer"}
        rows.append(row)
        b1 = lat["b1"]
        bB = lat[f"b{a.batches[-1]}"] if len(a.batches) > 1 else b1
        print(f"K={K:>2} S={S:>3} | params enc/dec/den = {params['encoder']:.0f}/{params['decoder']:.0f}/{params['denoiser']:.0f} M"
              f" | FLOPs/doc one-shot {flops['oneshot']:.0f} GF (prior {g_prior:.0f} + decode {g_dec:.0f}); +round {flops['refine_round']:.0f} GF"
              f" | b1 one-shot {b1['oneshot']:.0f} ms, R1 {b1['R1']:.0f}, R2 {b1['R2']:.0f}"
              f" | b{a.batches[-1]} one-shot {bB['oneshot'] / a.batches[-1]:.1f} ms/doc = {thr[f'b{a.batches[-1]}']['oneshot']:.1f} docs/s,"
              f" R1 {thr[f'b{a.batches[-1]}']['R1']:.1f} docs/s", flush=True)
        del model
        torch.cuda.empty_cache()
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
