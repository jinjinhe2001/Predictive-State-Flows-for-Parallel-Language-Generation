"""Judge perplexity as a function of the position inside the document.

Does quality drift along the document? A plan over K prefix latents is generated jointly, so the
later blocks could be realized from a worse plan than the early ones (the encoder never saw a
self-generated prefix of 900 tokens at training time). Every document is scored by the judge
(GPT-2 Large) and the per-token NLLs are pooled into absolute 128-token bins of judge tokens
(bin 1 = tokens 1-128, ... bin 8 = tokens 897-1024; the main-line K=8 blocks are 128 T5 tokens,
about the same span). The table gives the corpus-level PPL per bin, so real text gives the
"more context = easier" reference slope and a rising tail would mean drift.

  CUDA_VISIBLE_DEVICES=0 python -m scripts.position_ppl real=real_owt_1000.jsonl ours=texts_....jsonl
"""
from __future__ import annotations

import argparse
import math

import torch
import torch.nn.functional as F

from scripts.topic_metrics import load_texts


@torch.no_grad()
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("files", nargs="+", help="name=path.jsonl")
    p.add_argument("--judge", default="openai-community/gpt2-large")
    p.add_argument("--bin", type=int, default=128)
    p.add_argument("--num", type=int, default=1000)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--out", default=None)
    a = p.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(a.judge)
    tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    judge = AutoModelForCausalLM.from_pretrained(a.judge).to(device).eval()   # fp32, as scripts.eval_gen
    nbins = 1024 // a.bin
    hdr = f"{'system':<22} {'docs':>5} " + " ".join(f"{'bin' + str(b + 1):>7}" for b in range(nbins)) + f" {'tail/bin2':>9} {'all':>7}"
    print(hdr, flush=True)
    lines = [hdr]
    for spec in a.files:
        name, path = spec.split("=", 1)
        texts = [t for t in load_texts(path)[: a.num] if t.strip()]
        nll_sum = torch.zeros(nbins, dtype=torch.float64)
        nll_cnt = torch.zeros(nbins, dtype=torch.float64)
        for i in range(0, len(texts), a.batch):
            enc = tok(texts[i:i + a.batch], return_tensors="pt", padding=True, truncation=True, max_length=1024)
            ids = enc.input_ids.to(device)
            mask = enc.attention_mask.to(device)
            logits = judge(input_ids=ids, attention_mask=mask).logits.float()
            logp = F.log_softmax(logits[:, :-1], dim=-1)
            nll = -logp.gather(-1, ids[:, 1:, None]).squeeze(-1)          # predicts token at position 1..L-1
            m = mask[:, 1:].bool()
            pos = torch.arange(1, ids.shape[1], device=device).expand_as(nll)
            b = (pos // a.bin).clamp(max=nbins - 1)
            nll_sum.index_add_(0, b[m].cpu(), nll[m].double().cpu())
            nll_cnt.index_add_(0, b[m].cpu(), torch.ones(int(m.sum()), dtype=torch.float64))
        ppl = [math.exp(s / c) if c > 0 else float("nan") for s, c in zip(nll_sum.tolist(), nll_cnt.tolist())]
        allppl = math.exp(nll_sum.sum().item() / nll_cnt.sum().item())
        tail = sum(ppl[nbins // 2:]) / (nbins - nbins // 2)
        row = (f"{name:<22} {len(texts):>5} " + " ".join(f"{v:>7.1f}" for v in ppl)
               + f" {tail / ppl[1]:>9.2f} {allppl:>7.1f}")
        print(row, flush=True)
        lines.append(row)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
