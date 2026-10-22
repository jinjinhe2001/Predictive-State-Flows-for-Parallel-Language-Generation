"""Score unconditional samples with NCM's OpenWebText protocol (HazyResearch/ncm @ c702742, scripts/eval_merge.py
genppl): per seed, corpus perplexity under GPT-2 Large from NCM's own generation_perplexity (each text truncated to
1024 GPT-2 tokens) and the mean per-sample unigram entropy of the GPT-2 encoding; the row is the mean over seeds.

Each run is NAME=a.jsonl,b.jsonl,... with one file per seed.
  --mode doc     the first --n texts of each file, as they are (one PSF document each, ended at its own EOS)
  --mode packed  the documents' GPT-2 ids concatenated with no separator (NCM's decode drops EOS ids), cut into
                 --n consecutive windows of 1024 tokens: the shape of an NCM sample (1024 tokens, cut_at_eos false)
  --data BIN OFFSETS   the Data row instead: --n random 1024-token windows per seed from NCM's tokenized OWT stream

    python scripts/ncm_protocol/ncm_score.py --ncm <ncm checkout> --mode packed psf_k8=s42.jsonl,s0.jsonl,s1.jsonl,s2.jsonl,s3.jsonl
"""

import argparse
import json
import math
import sys
from collections import Counter

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SEEDS = [42, 0, 1, 2, 3]
WINDOW = 1024


def entropy(ids):
    c, m = Counter(ids), len(ids)
    return -sum(v / m * math.log(v / m) for v in c.values()) if m else 0.0


def read_texts(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line)["text"] for line in f if line.strip()]


def pack(texts, tok, n):
    stream = []
    for t in texts:
        stream.extend(tok.encode(t))
        if len(stream) >= n * WINDOW:
            break
    if len(stream) < n * WINDOW:
        raise SystemExit(f"only {len(stream)} GPT-2 tokens, need {n * WINDOW}: generate more documents")
    return [tok.decode(stream[i * WINDOW:(i + 1) * WINDOW]) for i in range(n)]


def data_windows(bin_path, n, seed, tok):
    stream = np.memmap(bin_path, dtype=np.uint16, mode="r")
    rng = np.random.default_rng(seed)
    out = []
    for start in rng.integers(0, len(stream) - 2 * WINDOW, size=n):
        ids = [int(t) for t in stream[start:start + 2 * WINDOW] if t != tok.eos_token_id][:WINDOW]
        out.append(tok.decode(ids))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ncm", required=True, help="NCM checkout (for evaluation.gen_ppl)")
    ap.add_argument("--mode", choices=["doc", "packed"], default="packed")
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--ref", default="gpt2-large")
    ap.add_argument("--data", nargs=2, metavar=("BIN", "OFFSETS"), default=None)
    ap.add_argument("--dump", default=None, help="write the scored texts of each run/seed to DUMP_<name>_s<seed>.jsonl")
    ap.add_argument("runs", nargs="*")
    a = ap.parse_args()

    sys.path.insert(0, a.ncm)
    from evaluation.gen_ppl import generation_perplexity

    rtok = AutoTokenizer.from_pretrained(a.ref)
    rlm = AutoModelForCausalLM.from_pretrained(a.ref).cuda().eval()

    runs = [(r.split("=", 1)[0], r.split("=", 1)[1].split(",")) for r in a.runs]
    if a.data:
        runs = [("data", [None] * len(SEEDS))] + runs
    for name, files in runs:
        per = []
        for seed, path in zip(SEEDS, files):
            if path is None:
                gen = data_windows(a.data[0], a.n, seed, rtok)
            elif a.mode == "doc":
                gen = read_texts(path)[: a.n]
            else:
                gen = pack(read_texts(path), rtok, a.n)
            if a.dump:
                with open(f"{a.dump}_{name}_s{seed}.jsonl", "w", encoding="utf-8") as f:
                    for t in gen:
                        f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")
            with torch.no_grad():
                ppl = generation_perplexity(rlm, rtok, [""] * len(gen), gen, batch_size=32).mean_ppl
            toks = [rtok.encode(h) for h in gen]
            ent = float(np.mean([entropy(t) for t in toks if t]))
            per.append((seed, ppl, ent, float(np.mean([len(t) for t in toks]))))
        g = np.array([p[1] for p in per])
        e = np.array([p[2] for p in per])
        ln = np.mean([p[3] for p in per])
        sd = g.std(ddof=1) if len(g) > 1 else 0.0
        seeds_txt = " ".join(f"{p[0]}:{p[1]:.2f}@{p[2]:.3f}" for p in per)
        print(f"{name:40s} mode={'data' if name == 'data' else a.mode:6s} genPPL={g.mean():8.2f} +/- {sd:5.2f} "
              f"entropy={e.mean():.4f} len={ln:6.0f} | {seeds_txt}", flush=True)


if __name__ == "__main__":
    main()
