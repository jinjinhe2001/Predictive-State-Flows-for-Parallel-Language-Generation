"""Latency / throughput of the MDLM-paper 110M autoregressive OWT baseline with their own sampler
(`_ar_sampler`, Gumbel-max), same protocol as bench_bd3lm.py: eager, warmup + averaged, cuda.synchronize,
batch 1 and batch 32; FLOPs analytic (flash-attn's Triton kernels are not traceable): 2 * params * tokens
per document if the sampler caches, marked "~".

Run from the mdlm checkout in the bd3lm conda env:
  python bench_ar_mdlm.py --ckpt ../bd3lms/gdrive_ckpts/ar.ckpt --batches 1 32
"""
import argparse
import sys
import time

import torch
from hydra import compose, initialize

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--ckpt", required=True)
p.add_argument("--batches", type=int, nargs="+", default=[1, 32])
p.add_argument("--iters", type=int, default=3)
a = p.parse_args()
sys.argv = [sys.argv[0]]
import main as M  # noqa: E402
import dataloader  # noqa: E402

with initialize(version_base=None, config_path="configs"):
    cfg = compose(config_name="config", overrides=[
        "mode=sample_eval", "model=small-ar", "parameterization=ar", "backbone=ar",
        "data=openwebtext-split", "model.length=1024", "loader.eval_batch_size=1",
        f"eval.checkpoint_path={a.ckpt}", "wandb=null"])
tok = dataloader.get_tokenizer(cfg)
model = M._load_from_checkpoint(cfg, tok).to("cuda").eval()
if getattr(model, "ema", None):
    params = list(model.backbone.parameters()) + list(model.noise.parameters())
    model.ema.store(params)
    model.ema.copy_to(params)
nparam = sum(q.numel() for q in model.backbone.parameters()) / 1e6
L = model.config.model.length
print(f"AR-110M (mdlm repo, `_ar_sampler`): {nparam:.1f}M backbone params, {L} tokens/doc, eager")
print(f"~FLOPs/doc analytic (KV-cached AR): {2 * nparam * 1e6 * L / 1e9:.1f} GF; without cache the sampler "
      f"recomputes the prefix: ~{2 * nparam * 1e6 * L * (L + 1) / 2 / 1e9:.0f} GF")
for B in a.batches:
    with torch.no_grad():
        model._ar_sampler(B)                                   # warmup
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(a.iters):
            model._ar_sampler(B)
        torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / a.iters * 1e3
    print(f"B={B:>3} | {ms:>9.0f} ms/batch | {ms / B:>8.1f} ms/doc | {B / (ms / 1e3):>6.2f} docs/s", flush=True)
