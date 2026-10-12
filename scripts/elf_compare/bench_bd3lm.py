"""Measured FLOPs + batch-1 latency of BD3-LM / MDLM sampling (their code, our tools).

Run from the bd3lms checkout in its conda env:  python bench_bd3lm.py [bd3lm4 bd3lm8 bd3lm16 mdlm]
Same accounting as scripts/flops_measure.py (torch FlopCounterMode over the real sampling
path, batch 1 -> per-document FLOPs) and scripts/latency_probe.py (eager, warmup + averaged,
cuda.synchronize). NFEs per document are taken from the sampling CSVs (gen_nfes column).
FlopCounterMode cannot trace flash-attn's Triton rotary kernel (MDLM's HF code): for such
models the FLOPs fall back to the analytic 2 * params * tokens * NFE and are marked "~".
"""
import sys
import time

import torch
from hydra import compose, initialize
from torch.utils.flop_counter import FlopCounterMode

ARGS = [a for a in sys.argv[1:]]
sys.argv = [sys.argv[0]]
import main as M  # noqa: E402  (registers the omegaconf resolvers)
import dataloader  # noqa: E402

GF = 1e9
COMMON = ["mode=sample_eval", "model=small", "data=openwebtext-split", "wandb=null", "trainer.devices=1",
          "loader.eval_batch_size=1", "sampling.num_sample_batches=1", "model.length=1024",
          "model.attn_backend=sdpa", "sampling.nucleus_p=0.9"]
CONFIGS = {
    "bd3lm4": ("bd3lm bs4", ["algo=bd3lm", "algo.T=5000", "algo.backbone=hf_dit", "block_size=4", "sampling.kv_cache=true",
                             "eval.checkpoint_path=kuleshov-group/bd3lm-owt-block_size4"], 1023),
    "bd3lm8": ("bd3lm bs8", ["algo=bd3lm", "algo.T=5000", "algo.backbone=hf_dit", "block_size=8", "sampling.kv_cache=true",
                             "eval.checkpoint_path=kuleshov-group/bd3lm-owt-block_size8"], 1023),
    "bd3lm16": ("bd3lm bs16", ["algo=bd3lm", "algo.T=5000", "algo.backbone=hf_dit", "block_size=16", "sampling.kv_cache=true",
                               "eval.checkpoint_path=kuleshov-group/bd3lm-owt-block_size16"], 1023),
    "mdlm": ("mdlm T1000", ["algo=mdlm", "algo.T=1000", "algo.backbone=hf_dit", "block_size=1024",
                            "eval.checkpoint_path=kuleshov-group/mdlm-owt"], 1000),
}
SEL = ARGS or list(CONFIGS)


def load(overrides):
    with initialize(version_base=None, config_path="configs"):
        cfg = compose(config_name="config", overrides=COMMON + overrides)
    tok = dataloader.get_tokenizer(cfg)
    model = M._load_from_checkpoint(cfg, tok)
    model.eval()
    if model.ema:
        model.ema.store(model._get_parameters())
        model.ema.copy_to(model._get_parameters())
    return cfg, model


BIG = int([a for a in ARGS if a.startswith("--batch=")][0].split("=")[1]) if any(a.startswith("--batch=") for a in ARGS) else 32
SEL = [a for a in SEL if not a.startswith("--")]

def one(model, cfg, bsz=1):
    model.config.loader.eval_batch_size = bsz          # their KV cache is allocated with this batch size
    with torch.no_grad():
        return model._sample(seqlen=None, batch_size_per_gpu=bsz, num_steps=cfg.algo.T, eps=1e-5)


print(f"{'config':>12} | {'params(M)':>9} | {'FLOPs GF/doc':>13} | {'batch-1 ms/doc':>14} | {'b%d ms/doc' % BIG:>12} {'docs/s':>7}")
print("-" * 88)
for key in SEL:
    name, ov, nfe = CONFIGS[key]
    cfg, model = load(ov)
    nparam = sum(p.numel() for p in model.backbone.parameters()) / 1e6
    one(model, cfg)                                          # warmup
    try:
        fc = FlopCounterMode(display=False)
        with fc:
            one(model, cfg)
        flops = f"{fc.get_total_flops() / GF:13.1f}"
    except Exception as e:                                   # e.g. Triton rotary kernel (flash-attn) not traceable
        flops = f"~{2 * nparam * 1e6 * 1024 * nfe / GF:12.1f}"
        print(f"  [{name}: FlopCounterMode failed ({type(e).__name__}); analytic 2*P*L*NFE]", flush=True)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    iters = 3
    for _ in range(iters):
        one(model, cfg)
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / iters * 1e3
    one(model, cfg, BIG)                                     # warmup at the throughput batch
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(2):
        one(model, cfg, BIG)
    torch.cuda.synchronize()
    msb = (time.perf_counter() - t0) / 2 * 1e3
    print(f"{name:>12} | {nparam:>9.1f} | {flops} | {ms:>14.0f} | {msb / BIG:>12.1f} {BIG / (msb / 1e3):>7.2f}", flush=True)
    del model
    torch.cuda.empty_cache()
