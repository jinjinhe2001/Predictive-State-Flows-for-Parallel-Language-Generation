"""What the realization code carries: decode the same block from the same plan with
different codes, and with the same code but different sampling noise.

  CUDA_VISIBLE_DEVICES=0 python -m scripts.code_viz --ckpt runs/prior-srpl-rc64-look-k8/prior.pt --arch record \
      --num-blocks 8 --block 3 --plans 100 --temp 0.7 --show 3 --out code_viz_k8.jsonl

For each of ``--plans`` prior samples, block ``--block`` is decoded (a) with its own code and two fresh codes
r ~ N(0, I) under SHARED sampling noise (differences = the code), and (b) three times with its own code under
fresh sampling noise (differences = the sampler). Reported: mean pairwise content-word Jaccard within (a) and
(b), the mean edit distance of the first 20 tokens, and the fraction of first tokens that change; a few
examples are printed and all realizations are saved.
"""
from __future__ import annotations

import argparse
import json
import re

import torch

from psf.utils.data import decode, load_tokenizer
from psf.utils.misc import get_device, set_seed
from scripts.eval_prompted import build_model, content_words

WORD = re.compile(r"[a-z]{4,}")


def jaccard(a: str, b: str) -> float:
    A, B = content_words(a), content_words(b)
    return len(A & B) / max(len(A | B), 1)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--arch", choices=["record", "big"], default="record")
    p.add_argument("--num-blocks", type=int, default=8)
    p.add_argument("--block", type=int, default=3, help="0-based block to vary")
    p.add_argument("--plans", type=int, default=100)
    p.add_argument("--temp", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--show", type=int, default=3)
    p.add_argument("--out", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    a = p.parse_args()
    set_seed(a.seed)
    device = get_device(a.device)
    torch.set_float32_matmul_precision("high")
    model, cfg = build_model(a.arch, a.num_blocks, a.ckpt, device)
    tokenizer = load_tokenizer(cfg)
    K, S, j, bob = cfg.num_blocks, cfg.block_span, a.block, cfg.data.bob_token_id
    from psf.diffusion.flow import sample as flow_sample

    chunk = 25                                   # plans per decode call (x3 rows each)
    blocks_a, blocks_b = [], []
    with torch.no_grad():
        for c0 in range(0, a.plans, chunk):
            n_c = min(chunk, a.plans - c0)
            z = flow_sample(model.denoiser, n_c, K * cfg.num_latent_slots + cfg.num_global_latents, model.prior_dim,
                            cfg.flow, device)
            s, rc = model.split_latents(model.denoiser.denormalize(z))
            # (a) three codes for block j, shared sampling noise: replicate every plan 3x
            s3 = s.repeat_interleave(3, dim=0)
            rc3 = rc.repeat_interleave(3, dim=0).clone()
            fresh = torch.randn_like(rc3[:, j])
            alt = torch.arange(rc3.shape[0], device=device) % 3 != 0
            rc3[alt, j] = fresh[alt]
            torch.manual_seed(a.seed + 1 + c0)
            blocks_a.append(model.decoder.generate(s3, bob, temperature=a.temp, top_p=a.top_p, z_level=0.0, rc=rc3,
                                                   shared_noise=True))
            # (b) own code, three independent noises
            torch.manual_seed(a.seed + 2 + c0)
            blocks_b.append(model.decoder.generate(s3, bob, temperature=a.temp, top_p=a.top_p, z_level=0.0,
                                                   rc=rc.repeat_interleave(3, dim=0)))
        blocks_a = torch.cat(blocks_a, dim=0)
        blocks_b = torch.cat(blocks_b, dim=0)

    def block_text(blocks, i):
        ids = blocks[i, j].tolist()
        if cfg.data.eos_token_id in ids:
            ids = ids[: ids.index(cfg.data.eos_token_id)]
        return decode(tokenizer, ids), blocks[i, j].tolist()

    ja, jb, first_a, first_b, ed_a, ed_b = [], [], [], [], [], []
    records = []
    for n in range(a.plans):
        ta = [block_text(blocks_a, 3 * n + k) for k in range(3)]
        tb = [block_text(blocks_b, 3 * n + k) for k in range(3)]
        for x, y in ((0, 1), (0, 2), (1, 2)):
            ja.append(jaccard(ta[x][0], ta[y][0]))
            jb.append(jaccard(tb[x][0], tb[y][0]))
            first_a.append(ta[x][1][0] != ta[y][1][0])
            first_b.append(tb[x][1][0] != tb[y][1][0])
            ed_a.append(sum(u != v for u, v in zip(ta[x][1][:20], ta[y][1][:20])) / 20)
            ed_b.append(sum(u != v for u, v in zip(tb[x][1][:20], tb[y][1][:20])) / 20)
        records.append({"codes": [t[0] for t in ta], "noises": [t[0] for t in tb]})
    m = lambda v: sum(v) / len(v)
    print(f"block {j} of {K}, {a.plans} plans, t={a.temp}, nucleus {a.top_p}")
    print(f"{'':28s} {'Jaccard':>8s} {'1st tok differs':>16s} {'hamming@20':>11s}")
    print(f"{'different code, same noise':28s} {m(ja):8.3f} {100 * m(first_a):15.1f}% {100 * m(ed_a):10.1f}%")
    print(f"{'same code, different noise':28s} {m(jb):8.3f} {100 * m(first_b):15.1f}% {100 * m(ed_b):10.1f}%")
    for n in range(min(a.show, a.plans)):
        print(f"\n=== plan {n}, block {j}: three codes (shared noise)")
        for k, t in enumerate(records[n]["codes"]):
            print(f"  [code {k}] {t[:260]}")
        print(f"--- same code, three noises")
        for k, t in enumerate(records[n]["noises"]):
            print(f"  [noise {k}] {t[:260]}")
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            f.write(json.dumps({"jaccard_codes": m(ja), "jaccard_noise": m(jb), "first_differs_codes": m(first_a),
                                "first_differs_noise": m(first_b), "hamming20_codes": m(ed_a), "hamming20_noise": m(ed_b)}) + "\n")
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print("wrote", a.out)


if __name__ == "__main__":
    main()
