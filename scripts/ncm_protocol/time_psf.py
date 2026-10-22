"""PSF sampling time under NCM's speed protocol: one H100, 1,024 sequences at batch 16, seconds per sequence.

Runs the paper's harness (scripts.latency_r4: CUDA-graph prior + KV-cached block decoder, the evaluated sampler between
replays) unchanged, with two patches: the record checkpoints point at the released HF files, and a sampler spec
tp:<temperature>:<top_p>[:ban<n>] applies temperature and nucleus together (latency_r4's specs take one or the other).
--iters 64 at --batches 16 times 64 back-to-back batches = 1,024 sequences end to end.

    python scripts/ncm_protocol/time_psf.py --fp32-highest --ckpt-dir checkpoints \
        --blocks 8 16 32 --batches 16 --iters 64 --samplers tp:0.95:0.95 --out bench.jsonl
"""

import argparse
import os
import sys

sys.path.insert(0, os.getcwd())   # the PSF repo root

pre = argparse.ArgumentParser(add_help=False)
pre.add_argument("--ckpt-dir", required=True)
pre.add_argument("--prefix", default="psf", help="checkpoints are <ckpt-dir>/<prefix>_k<K>.pt")
pre.add_argument("--fp32-highest", action="store_true",
                 help="full FP32 matmuls (TF32 off) instead of the harness's TF32, to match a TF32-off protocol")
args, rest = pre.parse_known_args()

import torch  # noqa: E402

if args.fp32_highest:   # latency_r4.main sets "high" (TF32); pin it to "highest" instead
    _set = torch.set_float32_matmul_precision
    torch.set_float32_matmul_precision = lambda _p: _set("highest")
    torch.backends.cudnn.allow_tf32 = False

import scripts.latency_probe as probe   # noqa: E402
import scripts.latency_r4 as r4         # noqa: E402

probe.CKPT = {K: os.path.join(args.ckpt_dir, f"{args.prefix}_k{K}.pt") for K in (8, 16, 32, 64)}
_parse = r4.parse_sampler


def parse_sampler(spec):
    parts = spec.split(":")
    if parts[0] != "tp":
        return _parse(spec)
    d = {"name": spec, "temperature": float(parts[1]), "top_p": float(parts[2]), "ban": 0, "greedy": False}
    for p in parts[3:]:
        if p.startswith("ban"):
            d["ban"] = int(p[3:])
    return d


r4.parse_sampler = parse_sampler
sys.argv = [sys.argv[0]] + rest
r4.main()
