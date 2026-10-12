"""Per-document judge scores for the paper's qualitative-sample appendix (ELF-style boxes need a
per-sample "Entropy / Gen. PPL" line).

  python -m scripts.pick_samples LABEL=texts.jsonl [LABEL=texts.jsonl ...] --top 12 --out cands.txt

For every file: per-document GPT-2 Large perplexity over the first 1024 judge tokens (same judge and
truncation as eval_gen.judge_ppl), the document's own unigram entropy in nats over the same tokens
(ELF's per-sample definition), and a repeated-8-gram flag. Candidates = documents with at least
--min-tokens judge tokens, no repeated 8-gram, entropy within --ent-band of the file mean, sorted by
perplexity; the --top of them are printed with their first --chars characters to --out so a human can
pick readable ones. Selection is therefore curated; say so in the paper.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


def per_doc_scores(texts, judge, tok, device, bs=8):
    out = []
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i + bs], return_tensors="pt", padding=True, truncation=True, max_length=1024)
        ids, mask = enc.input_ids.to(device), enc.attention_mask.to(device)
        with torch.no_grad():
            logits = judge(input_ids=ids, attention_mask=mask).logits.float()
        logp = F.log_softmax(logits[:, :-1], dim=-1)
        nll = -logp.gather(-1, ids[:, 1:, None]).squeeze(-1)
        m = mask[:, 1:].float()
        doc_nll = (nll * m).sum(1) / m.sum(1).clamp_min(1)
        for j, (row, rm) in enumerate(zip(ids.tolist(), mask.tolist())):
            kept = [t for t, k in zip(row, rm) if k]
            c = Counter(kept)
            n = len(kept)
            ent = -sum(v / n * math.log(v / n) for v in c.values()) if n else 0.0
            grams = Counter(tuple(kept[k:k + 8]) for k in range(max(0, n - 7)))
            rep8 = max(grams.values()) if grams else 0
            out.append({"ppl": math.exp(float(doc_nll[j])), "entropy": ent, "tokens": n, "rep8": rep8})
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("specs", nargs="+", help="LABEL=path.jsonl")
    p.add_argument("--judge", default="openai-community/gpt2-large")
    p.add_argument("--top", type=int, default=12)
    p.add_argument("--min-tokens", type=int, default=700)
    p.add_argument("--ent-band", type=float, default=0.12)
    p.add_argument("--chars", type=int, default=1500)
    p.add_argument("--out", default="sample_candidates.txt")
    a = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.judge)
    tok.pad_token = tok.eos_token
    judge = AutoModelForCausalLM.from_pretrained(a.judge).to(device).eval()
    with open(a.out, "w", encoding="utf-8") as fo:
        for spec in a.specs:
            label, path = spec.split("=", 1)
            texts = [json.loads(l)["text"] for l in open(path, encoding="utf-8") if l.strip()]
            sc = per_doc_scores(texts, judge, tok, device)
            mean_ent = sum(s["entropy"] for s in sc) / len(sc)
            corpus_ppl = math.exp(sum(math.log(s["ppl"]) * s["tokens"] for s in sc) / sum(s["tokens"] for s in sc))
            cands = [(s["ppl"], i) for i, s in enumerate(sc)
                     if s["tokens"] >= a.min_tokens and s["rep8"] < 2 and abs(s["entropy"] - mean_ent) <= a.ent_band]
            cands.sort()
            head = (f"### {label}  ({path})  docs={len(sc)}  corpus PPL={corpus_ppl:.2f}  mean entropy={mean_ent:.3f}"
                    f"  candidates={len(cands)}")
            print(head, flush=True)
            fo.write(head + "\n")
            for ppl, i in cands[:a.top]:
                s = sc[i]
                line = f"--- {label} doc {i}: PPL={ppl:.2f} entropy={s['entropy']:.3f} tokens={s['tokens']}"
                print(line, flush=True)
                fo.write(line + "\n" + texts[i][:a.chars].replace("\n", " ") + "\n\n")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
