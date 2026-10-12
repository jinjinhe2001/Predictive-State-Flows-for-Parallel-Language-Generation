"""Cross-sample diversity and training-set memorization for text dumps.

Per-document unigram entropy (the paper's diversity axis) cannot see two failure modes a
low-temperature sampler can have: (a) every document draws from the same small pool of phrases
(cross-sample collapse) and (b) low perplexity bought by reproducing training text. This script
measures both on the 1000-document dumps behind the tables, with the same tokenizer for every
system (T5, the writer's vocabulary):

  distinct-n   unique n-grams / total n-grams over the whole dump (n = 1..4), corpus level
  Self-BLEU-4  BLEU of each document against all the others of the same dump (300 documents;
               higher = documents resemble each other more)
  memorization fraction of a document's 13-grams that occur verbatim in a shard of the TRAINING
               split (first ``--train-docs`` training documents after the paper's seeded
               train/val split, first 1024 tokens each, the canvas the models saw); documents
               with at least one hit; documents with a run of >= 20 consecutive hits (a shared
               span of >= 32 tokens); longest shared span. Held-out real documents (never in the
               shard) give the natural rate of shared 13-grams (boilerplate, quotations).

  python -m scripts.diversity_memo --train-docs 2000000 --out outputs/diversity_memo.txt \
      real=real_owt_1000.jsonl ours_k8=texts_rev_k8_nfe50_t0.75_p0.95_R0.jsonl ...
"""
from __future__ import annotations

import argparse
import math
import random
import re
import sys
import time
from collections import Counter

import numpy as np

from scripts.topic_metrics import load_texts

NGRAM = 13
SEP = 40000            # never a T5 id (vocab 32100): n-grams spanning two documents never match a query
P = np.uint64(1000003)
WORD = re.compile(r"[a-z0-9]+")


def rolling_hashes(ids: np.ndarray, n: int) -> np.ndarray:
    """uint64 polynomial hash of every n-gram of ``ids`` (len(ids) - n + 1 values)."""
    a = ids.astype(np.uint64) + np.uint64(1)
    m = len(a) - n + 1
    if m <= 0:
        return np.zeros(0, dtype=np.uint64)
    h = np.zeros(m, dtype=np.uint64)
    with np.errstate(over="ignore"):
        for j in range(n):
            h = h * P + a[j:j + m]
    return h


def build_train_index(train_docs: int, canvas: int, seed: int) -> tuple[np.ndarray, int, int]:
    """Sorted hashes of all 13-grams in the first ``train_docs`` training documents."""
    from psf.config import Config
    from psf.utils.data import load_split

    cfg = Config()
    ds = load_split(cfg.data.dataset_path, "train")
    perm = np.random.default_rng(cfg.data.split_seed).permutation(len(ds))
    n_val = min(int(len(perm) * cfg.data.val_frac), cfg.data.val_max_docs, len(perm) - 1)
    train_idx = perm[n_val:n_val + train_docs]
    t0 = time.time()
    chunks, n_tok = [], 0
    step = 20000
    for i in range(0, len(train_idx), step):
        rows = ds[np.sort(train_idx[i:i + step]).tolist()]["input_ids"]
        for ids in rows:
            ids = np.asarray(ids[:canvas], dtype=np.int64)
            chunks.append(ids)
            chunks.append(np.array([SEP], dtype=np.int64))
            n_tok += len(ids)
        if (i // step) % 10 == 0:
            print(f"  read {i + len(rows)}/{len(train_idx)} docs, {n_tok / 1e6:.0f}M tokens, {time.time() - t0:.0f}s",
                  flush=True)
    flat = np.concatenate(chunks)
    del chunks
    print(f"  hashing {len(flat) / 1e6:.0f}M positions ...", flush=True)
    h = rolling_hashes(flat, NGRAM)
    del flat
    print(f"  sorting {len(h) / 1e6:.0f}M hashes ...", flush=True)
    h.sort()
    print(f"  index ready in {time.time() - t0:.0f}s", flush=True)
    return h, len(train_idx), n_tok


def lookup(index: np.ndarray, q: np.ndarray) -> np.ndarray:
    if len(q) == 0:
        return np.zeros(0, dtype=bool)
    pos = np.searchsorted(index, q)
    pos = np.minimum(pos, len(index) - 1)
    return index[pos] == q


def longest_run(hits: np.ndarray) -> int:
    best = cur = 0
    for v in hits:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


def distinct_n(id_lists: list[list[int]], n: int) -> float:
    total, seen = 0, set()
    for ids in id_lists:
        for i in range(len(ids) - n + 1):
            seen.add(tuple(ids[i:i + n]))
            total += 1
    return len(seen) / max(total, 1)


def self_bleu(docs: list[list[str]], max_n: int = 4) -> float:
    """Corpus-style Self-BLEU: every document scored against all the others as references."""
    # top-2 counts per n-gram across documents (and the document holding the max) -> max over refs != self in O(1)
    per_doc = [[Counter(tuple(d[i:i + n]) for i in range(len(d) - n + 1)) for n in range(1, max_n + 1)] for d in docs]
    top: list[dict] = [dict() for _ in range(max_n)]
    for di, counters in enumerate(per_doc):
        for n, c in enumerate(counters):
            t = top[n]
            for g, v in c.items():
                cur = t.get(g)
                if cur is None:
                    t[g] = (v, di, 0)
                elif v > cur[0]:
                    t[g] = (v, di, cur[0])
                elif v > cur[2]:
                    t[g] = (cur[0], cur[1], v)
    lengths = np.array([len(d) for d in docs])
    match = np.zeros(max_n)
    total = np.zeros(max_n)
    hyp_len = ref_len = 0
    for di, counters in enumerate(per_doc):
        for n, c in enumerate(counters):
            for g, v in c.items():
                m1, d1, m2 = top[n][g]
                ref_max = m2 if d1 == di else m1
                match[n] += min(v, ref_max)
                total[n] += v
        hyp_len += lengths[di]
        others = np.delete(lengths, di)
        ref_len += int(others[np.argmin(np.abs(others - lengths[di]))])
    precisions = match / np.maximum(total, 1)
    if np.any(precisions == 0):
        return 0.0
    bp = 1.0 if hyp_len > ref_len else math.exp(1 - ref_len / max(hyp_len, 1))
    return bp * math.exp(np.mean(np.log(precisions)))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("files", nargs="+", help="name=path.jsonl")
    p.add_argument("--train-docs", type=int, default=2_000_000)
    p.add_argument("--canvas", type=int, default=1024)
    p.add_argument("--bleu-docs", type=int, default=300)
    p.add_argument("--num", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    p.add_argument("--no-memo", action="store_true", help="skip the training-set index (diversity only)")
    a = p.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("google-t5/t5-small")

    index = None
    if not a.no_memo:
        print(f"building training index over {a.train_docs} documents ...", flush=True)
        index, n_docs, n_tok = build_train_index(a.train_docs, a.canvas, a.seed)
        print(f"index: {n_docs} training documents, {n_tok / 1e9:.2f}B tokens, {len(index) / 1e9:.2f}B 13-gram hashes",
              flush=True)

    lines = []
    hdr = (f"{'system':<22} {'docs':>5} {'dist1':>6} {'dist2':>6} {'dist3':>6} {'dist4':>6} {'selfBLEU4':>9} | "
           f"{'13g hit%':>8} {'docs>=1':>7} {'docs>=32tok':>11} {'longest':>7}")
    print(hdr, flush=True)
    lines.append(hdr)
    for spec in a.files:
        name, path = spec.split("=", 1)
        texts = load_texts(path)[: a.num]
        enc = tok(texts, add_special_tokens=False, truncation=True, max_length=a.canvas)["input_ids"]
        id_lists = [list(e) for e in enc]
        d = [distinct_n(id_lists, n) for n in (1, 2, 3, 4)]
        rng = random.Random(a.seed)
        sub = rng.sample(range(len(texts)), min(a.bleu_docs, len(texts)))
        words = [WORD.findall(texts[i].lower())[: a.canvas] for i in sub]
        sb = self_bleu(words)
        memo = ""
        if index is not None:
            n_q = n_hit = docs_hit = docs_long = 0
            longest = 0
            for ids in id_lists:
                q = rolling_hashes(np.asarray(ids, dtype=np.int64), NGRAM)
                hits = lookup(index, q)
                n_q += len(q)
                n_hit += int(hits.sum())
                if hits.any():
                    docs_hit += 1
                    run = longest_run(hits)
                    longest = max(longest, run)
                    if run >= 20:
                        docs_long += 1
            memo = (f"{100 * n_hit / max(n_q, 1):>8.2f} {docs_hit / len(id_lists):>7.1%} "
                    f"{docs_long / len(id_lists):>11.1%} {longest + NGRAM - 1 if longest else 0:>7d}")
        row = (f"{name:<22} {len(texts):>5} {d[0]:>6.3f} {d[1]:>6.3f} {d[2]:>6.3f} {d[3]:>6.3f} {sb:>9.3f} | {memo}")
        print(row, flush=True)
        lines.append(row)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
