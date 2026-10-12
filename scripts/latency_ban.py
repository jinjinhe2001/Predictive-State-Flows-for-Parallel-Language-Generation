"""Latency of the exact Table 1 headline samplers: plain vs states-only+nucleus vs states-only+nucleus+4-gram ban.

Motivation: "the ban costs no FLOPs worth counting" does not establish negligible
latency. Three measurements on one GPU, tf32, median of ``--iters`` after warmup (cuda.synchronize):

  (1) the real eager sampler ``Decoder.generate`` (the code path of scripts.eval_gen; uncached, full-prefix
      forward per step) for the three settings at batch 1 and 32, plus the prior's ``flow_sample`` -- the
      three settings differ only in the sampler, so their ratio is the sampler cost on this path;
  (2) the ban in isolation: ``Decoder._ngram_ban`` + ``masked_fill`` on realistic prefixes, summed over the
      S decoding steps of a document (the ban does not depend on the model, only on B, K, s and V), which is
      the overhead to add to the CUDA-graph numbers of Table 2 / scripts.latency_graphed;
  (3) the prior: ``flow_sample`` with the prior's-codes checkpoint and the states-only checkpoint (same
      architecture, so the same cost; measured to confirm).

  CUDA_VISIBLE_DEVICES=5 python -m scripts.latency_ban --arch record --batches 1 32
  CUDA_VISIBLE_DEVICES=5 python -m scripts.latency_ban --arch big --batches 1 32
Writes JSON lines to bench/latency_ban_<arch>.jsonl and prints a table.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time

import torch

from psf.config import Config
from psf.diffusion.flow import sample as flow_sample
from psf.utils.misc import get_device, load_checkpoint, set_seed
from scripts.bench_ours import build, timed

CK = {
    "record": {"plain": "runs/prior-srpl-rc64-look-k8/prior.pt", "so": "runs/prior-k8-zerocode/prior.pt"},
    "big": {"plain": "runs/big-prior-k8/prior.pt", "so": "runs/big-prior-k8-zerocode/prior.pt"},
}
SETTINGS = {  # name -> (temperature, top_p, rc_mode, no_repeat_ngram)
    "plain t=0.75, nucleus 0.95, prior's codes": (0.75, 0.95, "plan", 0),
    "states-only, nucleus 0.6": (1.0, 0.6, "zero", 0),
    "states-only, nucleus 0.6, 4-gram ban": (1.0, 0.6, "zero", 4),
}


def ban_cost(dec, gen: torch.Tensor, n: int, vocab: int, iters: int) -> float:
    """Seconds per document-batch spent in the ban over all S steps (median of ``iters`` full sweeps)."""
    B, K, S = gen.shape
    logits = torch.randn(B, K, vocab, device=gen.device)
    ts = []
    with torch.no_grad():
        for _ in range(iters + 1):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for s in range(n - 1, S):
                ban = dec._ngram_ban(gen[:, :, :s], n, vocab)
                if ban is not None:
                    logits.masked_fill(ban, float("-inf"))
            torch.cuda.synchronize()
            ts.append(time.perf_counter() - t0)
    return statistics.median(ts[1:])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arch", choices=["record", "big"], default="record")
    p.add_argument("--batches", type=int, nargs="+", default=[1, 32])
    p.add_argument("--iters", type=int, default=5)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--out", default=None)
    p.add_argument("--device", default="auto")
    a = p.parse_args()
    device = get_device(a.device)
    torch.set_float32_matmul_precision("high")
    out = a.out or f"bench/latency_ban_{a.arch}.jsonl"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    bob = Config().data.bob_token_id
    print(f"device={torch.cuda.get_device_name(0)} arch={a.arch} eager tf32, median of {a.iters} (b1) / {max(2, a.iters // 2)} (b>1)", flush=True)
    set_seed(0)
    model, cfg = build(8, a.arch, device)
    dec = model.decoder
    S, L = cfg.block_span, cfg.max_seq_len
    nlat = cfg.num_blocks * cfg.num_latent_slots + cfg.num_global_latents
    V = dec.vocab_size if hasattr(dec, "vocab_size") else cfg.data.vocab_size
    rows = []
    for B in a.batches:
        iters = a.iters if B == 1 else max(2, a.iters // 2)
        rec = {"arch": a.arch, "K": 8, "B": B, "prior_ms": {}, "gen_ms": {}, "ban_ms": None}
        for which in ("plain", "so"):
            load_checkpoint(CK[a.arch][which], model, device)
            model.eval()
            with torch.no_grad():
                z = flow_sample(model.denoiser, B, nlat, model.prior_dim, cfg.flow, device)
                s, rc = model.split_latents(model.denoiser.denormalize(z))
            rec["prior_ms"][which] = timed(lambda: flow_sample(model.denoiser, B, nlat, model.prior_dim, cfg.flow, device), iters, a.warmup) * 1e3
            for name, (temp, top_p, rc_mode, ban) in SETTINGS.items():
                if (which == "plain") != (rc_mode == "plan"):
                    continue
                fn = lambda: dec.generate(s, bob, temperature=temp, top_p=top_p, z_level=0.0, rc=rc,
                                          rc_mode=rc_mode, no_repeat_ngram=ban)
                rec["gen_ms"][name] = timed(fn, iters, a.warmup) * 1e3
                if ban:
                    with torch.no_grad():
                        gen = dec.generate(s, bob, temperature=temp, top_p=top_p, z_level=0.0, rc=rc,
                                           rc_mode=rc_mode, no_repeat_ngram=ban)
                    vocab = int(V)
                    rec["ban_ms"] = ban_cost(dec, gen[:, :, :S].contiguous(), ban, vocab, iters) * 1e3
                print(f"  B={B:>2} {which:>5} | {name:<44} | prior {rec['prior_ms'][which]:8.1f} ms | writer {rec['gen_ms'][name]:9.1f} ms"
                      + (f" | ban alone {rec['ban_ms']:.1f} ms" if ban else ""), flush=True)
            del z, s, rc
            torch.cuda.empty_cache()
        rows.append(rec)
    print()
    print(f"{'setting':<46} {'B':>3} {'ms/doc':>9} {'docs/s':>8} {'vs plain':>9}")
    for rec in rows:
        B = rec["B"]
        base = rec["prior_ms"]["plain"] + rec["gen_ms"][list(SETTINGS)[0]]
        for name in SETTINGS:
            which = "plain" if SETTINGS[name][2] == "plan" else "so"
            tot = rec["prior_ms"][which] + rec["gen_ms"][name]
            print(f"{name:<46} {B:>3} {tot / B:>9.1f} {B / tot * 1e3:>8.2f} {tot / base:>9.3f}")
        print(f"{'ban alone (sum over steps)':<46} {B:>3} {rec['ban_ms'] / B:>9.2f} {'':>8} {rec['ban_ms'] / base:>9.4f}")
        print(f"{'prior: states-only / prior-codes ckpt':<46} {B:>3} {rec['prior_ms']['so'] / B:>9.1f} {'':>8} {rec['prior_ms']['so'] / rec['prior_ms']['plain']:>9.3f}")
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
