"""Cost of the AR baseline for the paper's compute table: analytic FLOPs per 1024-token document
(2 x params x tokens with a KV cache, attention-score FLOPs added) and measured eager latency /
throughput of psf.models.ar_lm.ARLM.generate (no CUDA graph; the writer's graphed numbers are the
launch-overhead-free comparison point, so the eager number here is an upper bound).

  CUDA_VISIBLE_DEVICES=0 python -m scripts.bench_ar --ckpt runs/ar-197m/ar.pt --batches 1 32 --iters 3
"""
from __future__ import annotations

import argparse
import time

import torch

from psf.models.ar_lm import ARConfig, ARLM


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--batches", type=int, nargs="+", default=[1, 32])
    p.add_argument("--iters", type=int, default=3)
    a = p.parse_args()
    device = torch.device("cuda")
    ck = torch.load(a.ckpt, map_location=device, weights_only=False)
    cfg = ARConfig(**ck["ar_config"])
    model = ARLM(cfg).to(device).eval()
    model.load_state_dict(ck["model"])
    N = model.num_params()
    L = cfg.max_seq_len
    # analytic: matmuls 2*N per token (embedding lookups excluded, output projection included via tied
    # weights) + attention scores/values 2 * 2 * depth * L*(L+1)/2 * dim over the document
    flops = 2 * N * L + 4 * cfg.depth * cfg.dim * L * (L + 1) / 2
    print(f"AR-{N / 1e6:.0f}M: analytic FLOPs/doc = {flops / 1e9:.0f} GF ({2 * N * L / 1e9:.0f} GF matmul + "
          f"{4 * cfg.depth * cfg.dim * L * (L + 1) / 2 / 1e9:.0f} GF attention), {L - 1} dependent steps")
    for B in a.batches:
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            model.generate(B, device, temperature=1.0, top_p=1.0, max_len=64)   # warmup
            torch.cuda.synchronize()
            ts = []
            for _ in range(a.iters):
                t0 = time.perf_counter()
                # temperature 0 with a fixed length so no early stop shortens the run: force full canvas
                model.generate(B, device, temperature=1.0, top_p=1.0)
                torch.cuda.synchronize()
                ts.append(time.perf_counter() - t0)
        ts.sort()
        med = ts[len(ts) // 2]
        print(f"batch {B:3d}: median {med:.2f} s per call = {med / B * 1e3:.0f} ms/doc = {B / med:.1f} docs/s (eager, bf16)")


if __name__ == "__main__":
    main()
