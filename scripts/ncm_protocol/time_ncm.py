"""NCM sampling time on OpenWebText under its own speed protocol: one H100, 1,024 sequences at batch 16, s/sequence.

The OWT checkpoint is not released, so -- as NCM's paper does for baselines without checkpoints -- the architecture is
built with random weights and only timed. The timed call is eval_shard.py's run() up to token ids: generate (one
forward per level, 32 levels, temperature 1 / top_k 0 / top_p 1 as config/eval/owt.yaml), idx_to_zhats, the VQ-VAE
decoder and argmax to GPT-2 ids (pick_tokens with no penalties). Text detokenization is excluded on both sides.

  --arch paper     Table 5/6 of the paper: generator 12 x 768 (12 heads), tokenizer codebooks 4,096 per level
  --arch released  config/eval/owt.yaml as shipped: generator 16 x 1024 (16 heads), codebooks 8,192
Modes: fp32 (the released eval path, eager), tf32, bf16 (autocast), bf16+kv (use_kv_cache=True), and with --compile
each of the bf16 modes again with every transformer block torch.compile'd.

    cd <ncm checkout> && python <psf repo>/scripts/ncm_protocol/time_ncm.py --arch paper --batch 16 --iters 64
"""

import argparse
import json
import os
import sys
import time

import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.getcwd())   # the NCM repo root
from models.next_concept import NextConceptTransformer   # noqa: E402
from models.vqvae import VQVAE                            # noqa: E402

EOS = 50256


def build(arch):
    cfg = OmegaConf.load("config/eval/owt.yaml")
    if arch == "paper":
        cfg.model.n_layer, cfg.model.n_embd, cfg.model.n_head = 12, 768, 12
        nlev = len(cfg.model.vocab_size)
        cfg.model.vocab_size = [4096] * nlev
        cfg.tokenizer.vqvae.vqvae_vocab_size = [4096] * len(cfg.tokenizer.vqvae.vqvae_vocab_size)
    vqvae = VQVAE(**cfg.tokenizer.vqvae).cuda().eval()   # as VQVAETokenizer.from_config, minus the weight load
    net = NextConceptTransformer(vqvae_local=vqvae, config=cfg.model).cuda().eval()
    return net, vqvae


def count(m):
    return sum(p.numel() for p in m.parameters()) / 1e6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=["paper", "released"], default="paper")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--iters", type=int, default=64)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--modes", nargs="+", default=["fp32", "tf32", "bf16", "bf16+kv"])
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    torch.manual_seed(0)
    net, vqvae = build(a.arch)
    print(f"NCM-{a.arch}: generator module {count(net):.2f}M params, tokenizer {count(vqvae):.2f}M, "
          f"levels {len(vqvae.vq.levels)} -> {vqvae.vq.levels[-1]} positions", flush=True)
    B, L0 = a.batch, vqvae.vq.levels[0]

    def run(kv):
        l0 = torch.randint(0, vqvae.vqvae_vocab_size[0], (B, L0), device="cuda")
        g = net.generate(l0, torch.full((B, 1), EOS, device="cuda"), temperature=1.0, top_k=0, top_p=1.0,
                         use_kv_cache=kv, cfg_scale=0.0, cfg_null_eos=EOS)
        _, zl = vqvae.vq.idx_to_zhats(g, return_zlevels=True)
        return vqvae.decoder(zl[-1]).argmax(-1)

    modes = list(a.modes)
    if a.compile:
        modes += [m + "+compile" for m in a.modes if m.startswith("bf16")]
    fout = open(a.out, "a", encoding="utf-8") if a.out else None
    compiled = False
    for mode in modes:
        if mode.endswith("+compile") and not compiled:
            torch._dynamo.config.cache_size_limit = 256
            for blk in net.transformer.blocks:
                blk.compile()
            compiled = True
        torch.backends.cuda.matmul.allow_tf32 = mode != "fp32"
        torch.backends.cudnn.allow_tf32 = mode != "fp32"
        use_bf16 = mode.startswith("bf16")
        kv = "+kv" in mode

        def step():
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                return run(kv)

        try:
            out = step()
            assert out.shape == (B, vqvae.vq.levels[-1]), out.shape
            for _ in range(a.warmup):
                step()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(a.iters):
                step()
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
        except Exception as e:   # report and move on: a mode the code path does not support
            print(f"  {mode:18s} FAILED: {type(e).__name__}: {str(e)[:200]}", flush=True)
            continue
        row = dict(model=f"NCM-{a.arch}", mode=mode, B=B, n_seq=B * a.iters, total_s=dt,
                   s_per_seq=dt / (B * a.iters), ms_per_seq=1e3 * dt / (B * a.iters))
        print(f"  {mode:18s} B={B} {B * a.iters} seqs in {dt:7.2f} s -> {row['ms_per_seq']:7.2f} ms/seq", flush=True)
        if fout:
            fout.write(json.dumps(row) + "\n")
            fout.flush()


if __name__ == "__main__":
    main()
