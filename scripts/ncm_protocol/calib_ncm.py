"""Which timing protocol reproduces NCM's own reported numbers?

Anchors from the NCM paper (1,024 sequences, batch 16, one H100 SXM): TinyStories 5.65 s (Table 9, 12 levels,
L=128) and OpenWebText 0.080 s/sequence = 82 s (Figure 5, 32 levels, L=1024). Each run here times one protocol
end to end for 1,024 sequences, random weights (timing depends only on shapes; no checkpoint unpickling):

  fp32            the released eval path as shipped (scripts/eval_shard.py: no autocast, no compile, TF32 off)
  bf16            autocast BF16, eager
  bf16+compile    blocks torch.compile'd; WARM = compile and every level shape run before the clock starts
  bf16+compile-cold   same, but the clock includes compilation (run with a fresh TORCHINDUCTOR_CACHE_DIR)
  +e2e            adds eval_shard's per-batch host work: level-0 codes drawn on the CPU and copied over, ids to
                  host, and for OWT the GPT-2 detokenization of every sample (dec())

    cd <ncm checkout> && python <psf repo>/scripts/ncm_protocol/calib_ncm.py --task owt --mode fp32+compile
"""

import argparse
import json
import os
import sys
import time

import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.getcwd())   # the NCM repo root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.next_concept import NextConceptTransformer   # noqa: E402
from models.vqvae import VQVAE                            # noqa: E402

import time_ncm  # noqa: E402


def build(task):
    if task == "owt":
        return time_ncm.build("paper")
    cfg = OmegaConf.load("config/eval/tinystories.yaml")
    vqvae = VQVAE(**cfg.tokenizer.vqvae).cuda().eval()
    return NextConceptTransformer(vqvae_local=vqvae, config=cfg.model).cuda().eval(), vqvae


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["owt", "tinystories"], required=True)
    ap.add_argument("--mode", required=True, help="fp32 | bf16 | bf16+compile | bf16+compile-cold, optional +e2e")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--n", type=int, default=1024)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    torch.manual_seed(42)
    net, vqvae = build(a.task)
    B, L0, V0 = a.batch, vqvae.vq.levels[0], vqvae.vqvae_vocab_size[0]
    eos = 50256 if a.task == "owt" else 0
    e2e = "+e2e" in a.mode
    use_bf16 = a.mode.startswith("bf16")
    compiled = "compile" in a.mode
    cold = "cold" in a.mode
    torch.backends.cuda.matmul.allow_tf32 = use_bf16    # the shipped fp32 path leaves TF32 at PyTorch's default (off)
    torch.backends.cudnn.allow_tf32 = use_bf16
    tok = None
    if e2e and a.task == "owt":
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("gpt2")
    unigram = torch.full((V0,), 1.0 / V0, dtype=torch.float64)   # stands in for generator.pt:inference.level0_unigram

    if compiled:
        torch._dynamo.config.cache_size_limit = 256
        for blk in net.transformer.blocks:
            blk.compile()

    def batch():
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
            if e2e:
                l0 = torch.multinomial(unigram.float(), B * L0, replacement=True).view(B, L0).cuda()
            else:
                l0 = torch.randint(0, V0, (B, L0), device="cuda")
            g = net.generate(l0, torch.full((B, 1), eos, device="cuda"), temperature=1.0, top_k=0, top_p=1.0,
                             use_kv_cache=False, cfg_scale=0.0, cfg_null_eos=eos)
            _, zl = vqvae.vq.idx_to_zhats(g, return_zlevels=True)
            ids = vqvae.decoder(zl[-1]).argmax(-1)
        if e2e:
            ids = ids.tolist()
            if tok is not None:
                [tok.decode([t for t in r if t != eos], skip_special_tokens=True) for r in ids]
        return ids

    if not cold:   # warm-up: every level shape once (and compilation, if any) before the clock
        for _ in range(3):
            batch()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(a.n // B):
        batch()
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    row = dict(task=a.task, mode=a.mode, B=B, n=a.n, total_s=dt, ms_per_seq=1e3 * dt / a.n)
    print(f"{a.task:12s} {a.mode:24s} {a.n} seqs @ B={B}: {dt:7.2f} s  ({row['ms_per_seq']:.2f} ms/seq)", flush=True)
    if a.out:
        with open(a.out, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
