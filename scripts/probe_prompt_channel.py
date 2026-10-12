"""Did the prompt-aware writer learn anything from its prompt channel?

Two diagnostics on the trained checkpoint: (a) the Frobenius norm of each layer's prompt output
projection (zero-initialised) next to its latent cross-attention output projection; (b) the
teacher-forced NLL of held-out documents' blocks m+1..K under true states and codes, with the first
m blocks fed as prompt tokens versus withheld -- the training objective itself. If (b) barely moves,
the prompt tokens carry nothing the prefix latents do not already give the writer.

  CUDA_VISIBLE_DEVICES=0 python -m scripts.probe_prompt_channel --ckpt runs/prior-k8-prompt/prior.pt --prompt-blocks 2
"""
from __future__ import annotations

import argparse
import math

import torch
import torch.nn.functional as F

from psf.utils.data import make_dataloaders
from psf.utils.misc import get_device, set_seed
from scripts.eval_prompted import build_model


@torch.no_grad()
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--arch", default="record")
    p.add_argument("--num-blocks", type=int, default=8)
    p.add_argument("--prompt-blocks", type=int, default=2)
    p.add_argument("--num-docs", type=int, default=500)
    p.add_argument("--batch", type=int, default=25)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    a = p.parse_args()
    device = get_device(a.device)
    set_seed(a.seed)
    model, cfg = build_model(a.arch, a.num_blocks, a.ckpt, device, prompt_attn=True)
    dec = model.decoder
    print("layer  |p_out|   |cross_out|   |p_q|   |p_kv|")
    for i, l in enumerate(dec.layers):
        print(f"{i:>5} {l.p_out.weight.norm():8.3f} {l.cross_out.weight.norm():12.3f} "
              f"{l.p_q.weight.norm():7.3f} {l.p_kv.weight.norm():7.3f}")
    print(f"prompt_pos |w|={dec.prompt_pos.weight.norm():.3f}  prompt_null |v|={dec.prompt_null.norm():.3f}")

    cfg.train.batch_size = a.batch
    cfg.data.num_workers = 0
    _, val = make_dataloaders(cfg)
    K, S = cfg.num_blocks, cfg.block_span
    m = a.prompt_blocks
    P = m * S
    tot = {"with": 0.0, "without": 0.0, "first_with": 0.0, "first_without": 0.0}
    cnt = {"cont": 0.0, "first": 0.0}
    seen = 0
    for batch in val:
        batch = batch.to(device)
        keep = batch.mask.sum(1) >= 2 * P                      # the same filter as eval_prompted (>= 512 tokens)
        if keep.sum() == 0:
            continue
        tokens, mask = batch.tokens[keep], batch.mask[keep]
        enc = model.encode(tokens, mask)
        inputs, targets = model.block_view(tokens)
        nf = model.boundary_targets(targets)
        rc = model.sample_rc(enc, "mean")
        outs = {
            "with": dec(enc.block_feats, inputs, next_first=nf, rc=rc,
                        prompt=tokens[:, :P], prompt_mask=mask[:, :P]),
            "without": dec(enc.block_feats, inputs, next_first=nf, rc=rc),
        }
        bm = mask.reshape(targets.shape).float()
        bm[:, :m] = 0.0                                          # continuation blocks only
        first = torch.zeros_like(bm)
        first[:, m] = bm[:, m]                                   # the first written block
        for k, logits in outs.items():
            ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), targets.reshape(-1),
                                 reduction="none").reshape(targets.shape)
            tot[k] += float((ce * bm).sum())
            tot["first_" + k] += float((ce * first).sum())
        cnt["cont"] += float(bm.sum())
        cnt["first"] += float(first.sum())
        seen += int(keep.sum())
        if seen >= a.num_docs:
            break
    nw, nwo = tot["with"] / cnt["cont"], tot["without"] / cnt["cont"]
    fw, fwo = tot["first_with"] / cnt["first"], tot["first_without"] / cnt["first"]
    print(f"\n{seen} held-out docs, prompt = first {m} blocks ({P} tokens), true states + codes, teacher forcing")
    print(f"continuation blocks {m + 1}..{K}: NLL with prompt {nw:.4f} | without {nwo:.4f} | "
          f"diff {nw - nwo:+.4f} nats/tok ({100 * (math.exp(nw - nwo) - 1):+.2f}% PPL)")
    print(f"first continuation block only:  NLL with prompt {fw:.4f} | without {fwo:.4f} | "
          f"diff {fw - fwo:+.4f} nats/tok ({100 * (math.exp(fw - fwo) - 1):+.2f}% PPL)")


if __name__ == "__main__":
    main()
