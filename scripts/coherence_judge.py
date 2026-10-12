"""Blinded pairwise LLM-judge coherence evaluation of text dumps.

Generative perplexity at matched entropy cannot see whether a 1024-token document holds together
as ONE document. Here an instruction-tuned judge (default Qwen2.5-7B-Instruct) sees two passages
(the first ``--words`` words of one document from system 1 and one from system 2, in random order)
and answers three questions with a single letter:

  coherent   which passage is more internally coherent (one topic, consistent facts, no contradictions)
  human      which passage reads more like a single article written by one human author
  norepeat   which passage has fewer repeated or looping sentences and phrases

Every document pair is judged TWICE, once in each slot order, so slot bias cancels: we report
system 1's order-averaged win rate (each order counts one half), the number of pairs on which both
orders agree on system 1 / on system 2 / disagree ("split", i.e. the judge followed the slot), a
95% Wilson interval for system 1's share of the consistent pairs, the judge's overall preference
for the A slot, and the fraction of invalid answers.  A "real vs real" pair calibrates the protocol
(expected ~50%).  Lexical overlap with a prompt cannot satisfy this test; it is still an LLM judge
on English OpenWebText with truncated documents.

  CUDA_VISIBLE_DEVICES=6 python -m scripts.coherence_judge \
      --systems real=real_owt_1000.jsonl psf=texts_....jsonl elf=... \
      --pairs psf:real elf:real psf:elf real:real --n 200 --out outputs/coherence.txt
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time

import torch

QUESTIONS = {
    "coherent": ("Which passage is more internally coherent: it stays on one topic, its facts and "
                 "references are consistent with each other, and it contains no contradictions?"),
    "human": "Which passage reads more like a single article written by one human author?",
    "norepeat": "Which passage has fewer repeated or looping sentences and phrases?",
}
SYSTEM_MSG = ("You are a meticulous editor judging text quality. You will read two passages and answer "
              "one question about them with a single letter.")


def load(path: str, n: int) -> list[str]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                t = row if isinstance(row, str) else (row.get("text") or row.get("generated")
                                                     or next((v for v in row.values() if isinstance(v, str)), None))
                if isinstance(t, str) and t.strip():
                    out.append(t)
            if len(out) >= n:
                break
    return out


def truncate(text: str, words: int, skip: int = 0) -> str:
    w = text.split()
    return " ".join(w[skip:skip + words])


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def build_prompt(tok, a: str, b: str, question: str) -> str:
    user = (f"Passage A:\n{a}\n\nPassage B:\n{b}\n\nQuestion: {question}\n"
            "Answer with exactly one letter, A or B, and nothing else.")
    msgs = [{"role": "system", "content": SYSTEM_MSG}, {"role": "user", "content": user}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def judge_batch(model, tok, prompts: list[str], max_new: int = 2) -> list[str]:
    enc = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=4096).to(model.device)
    out = model.generate(**enc, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id)
    gen = out[:, enc["input_ids"].shape[1]:]
    return [tok.decode(g, skip_special_tokens=True) for g in gen]


def parse(ans: str) -> str | None:
    s = ans.strip().upper()
    if not s:
        return None
    c = s[0]
    if c in ("A", "B"):
        return c
    # tolerate "Passage A" / "(A)"
    for ch in s:
        if ch in ("A", "B"):
            return ch
        if ch.isalpha():
            break
    return None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--systems", nargs="+", required=True, help="name=path.jsonl")
    p.add_argument("--pairs", nargs="+", required=True, help="name1:name2 (system 1 vs system 2)")
    p.add_argument("--n", type=int, default=200, help="document pairs per system pair")
    p.add_argument("--words", type=int, default=300, help="truncate each document to its first N words")
    p.add_argument("--skip-words", type=int, default=0, help="start the window after the first N words (0 = document start)")
    p.add_argument("--device-map", action="store_true", help="shard the judge over all visible GPUs (device_map=auto) for 32B-class judges")
    p.add_argument("--min-words", type=int, default=50)
    p.add_argument("--num-docs", type=int, default=1000, help="documents read per dump")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--out", default="outputs/coherence.txt")
    p.add_argument("--raw", default=None, help="jsonl of every judgement (default: <out>.raw.jsonl)")
    p.add_argument("--device", default="auto")
    a = p.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    systems = {}
    for spec in a.systems:
        name, path = spec.split("=", 1)
        docs = [truncate(t, a.words, a.skip_words) for t in load(path, a.num_docs)]
        docs = [d for d in docs if len(d.split()) >= a.min_words]
        if not docs:
            sys.exit(f"no usable documents in {path}")
        systems[name] = docs
        print(f"[load] {name}: {len(docs)} docs from {path}", flush=True)

    dev = "cuda" if a.device == "auto" and torch.cuda.is_available() else a.device
    tok = AutoTokenizer.from_pretrained(a.model)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    if a.device_map:
        model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map="auto").eval()
    else:
        model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16).to(dev).eval()
    print(f"[model] {a.model} on {dev}", flush=True)

    raw_path = a.raw or (a.out + ".raw.jsonl")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    raw = open(raw_path, "w", encoding="utf-8")
    rows = []
    t0 = time.time()
    for pi, pair in enumerate(a.pairs):
        s1, s2 = pair.split(":", 1)
        rng = random.Random(a.seed * 1000 + pi)
        d1, d2 = systems[s1], systems[s2]
        idx1 = [rng.randrange(len(d1)) for _ in range(a.n)]
        idx2 = [rng.randrange(len(d2)) for _ in range(a.n)]
        if s1 == s2:  # calibration: never pair a document with itself
            idx2 = [(j if j != i else (j + 1) % len(d2)) for i, j in zip(idx1, idx2)]
        for q, question in QUESTIONS.items():
            # every document pair in both slot orders: order 0 = system 1 as Passage A, order 1 = as Passage B
            prompts, meta = [], []
            for k in range(a.n):
                x1, x2 = d1[idx1[k]], d2[idx2[k]]
                prompts.append(build_prompt(tok, x1, x2, question)); meta.append((k, 0))
                prompts.append(build_prompt(tok, x2, x1, question)); meta.append((k, 1))
            verdict = [[None, None] for _ in range(a.n)]  # per pair, per order: True = system 1 won, None = invalid
            pickA = 0
            for b0 in range(0, len(prompts), a.batch):
                answers = judge_batch(model, tok, prompts[b0:b0 + a.batch])
                for (k, order), ans in zip(meta[b0:b0 + a.batch], answers):
                    c = parse(ans)
                    raw.write(json.dumps({"pair": pair, "q": q, "k": k, "i1": idx1[k], "i2": idx2[k],
                                          "s1_is": "B" if order else "A", "answer": ans.strip(), "parsed": c}) + "\n")
                    if c is None:
                        continue
                    pickA += int(c == "A")
                    verdict[k][order] = (c == "B") if order else (c == "A")
            raw.flush()
            valid_pairs = [v for v in verdict if v[0] is not None and v[1] is not None]
            n_valid_ans = sum(v is not None for vv in verdict for v in vv)
            cons1 = sum(v == [True, True] for v in valid_pairs)
            cons2 = sum(v == [False, False] for v in valid_pairs)
            split = len(valid_pairs) - cons1 - cons2
            win_avg = sum(0.5 * int(v[0]) + 0.5 * int(v[1]) for v in valid_pairs)
            rate = win_avg / max(len(valid_pairs), 1)
            lo, hi = wilson(cons1, cons1 + cons2)
            rows.append({"pair": pair, "q": q, "n": a.n, "pairs": len(valid_pairs), "rate": rate,
                         "cons1": cons1, "cons2": cons2, "split": split, "lo": lo, "hi": hi,
                         "pickA": pickA / max(n_valid_ans, 1), "invalid": 1 - n_valid_ans / (2 * a.n)})
            r = rows[-1]
            print(f"[{pair:>28s}] {q:9s} rate {rate:.3f}  consistent s1 {cons1:3d} / s2 {cons2:3d} / split {split:3d} "
                  f"CI[{lo:.3f},{hi:.3f}] pickA {r['pickA']:.2f} invalid {r['invalid']:.2f}  ({time.time()-t0:.0f}s)",
                  flush=True)
    raw.close()

    hdr = (f"{'system 1 vs system 2':30s} {'question':9s} {'N':>4s} {'rate':>6s} {'s1':>4s} {'s2':>4s} {'split':>5s} "
           f"{'95% CI (s1 share)':>17s} {'pickA':>6s} {'invalid':>7s}")
    lines = [f"# blinded pairwise judge: {a.model}; words {a.skip_words}-{a.skip_words + a.words} of each document; N={a.n} document pairs "
             f"per row, each judged in both slot orders; seed {a.seed}. rate = system 1's order-averaged win rate; "
             f"s1 / s2 = pairs where both orders prefer system 1 / system 2; split = the two orders disagree "
             f"(slot-driven); CI = 95% Wilson interval of s1 / (s1 + s2); pickA = judge's overall A-slot preference; "
             f"invalid = answers that were not A or B.",
             hdr]
    for r in rows:
        lines.append(f"{r['pair']:30s} {r['q']:9s} {r['n']:4d} {r['rate']:6.3f} {r['cons1']:4d} {r['cons2']:4d} "
                     f"{r['split']:5d} [{r['lo']:.3f},{r['hi']:.3f}]{'':>3s} {r['pickA']:6.2f} {r['invalid']:7.2f}")
    with open(a.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"[done] {len(rows)} rows -> {a.out}; raw judgements -> {raw_path}; {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
