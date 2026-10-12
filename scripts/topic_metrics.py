"""Topic persistence along a document, same ruler for every system.

Motivation: K=16 samples read as a chain of unrelated articles (a hard topic switch every 3-4
blocks) while ELF-B switches about half as often; the judge PPL cannot see this. Here every
document is cut into fixed-length chunks of ``--chunk`` content words (default 45 ~ one 64-token
block) and we measure how much vocabulary chunk i shares with chunk i+d:

  J(d)   mean Jaccard of content-word sets between chunks d apart (d = 1, 2, 4, 8)
  floor  same statistic between chunks of two DIFFERENT documents (topic-unrelated baseline)
  ratio  J(d) / floor  -- 1.0 = as unrelated as another document
  switch fraction of adjacent chunk pairs whose overlap is <= 1.5 x floor (hard-switch proxy)

Content words = lowercase \\w+ tokens of >= 3 chars minus the file's 150 most frequent words
(stopword proxy). Real held-out text (scripts/dump_real_texts.py) gives the reference band.
  python -m scripts.topic_metrics a.jsonl b.jsonl --labels ours,elf
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter

WORD = re.compile(r"[a-z]+")


def load_texts(path: str) -> list[str]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            t = row if isinstance(row, str) else row.get("text")
            if t is None:
                t = next((v for v in row.values() if isinstance(v, str)), "")
            if t.strip():
                out.append(t)
    return out


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def analyse(texts: list[str], chunk: int, gaps: tuple[int, ...], rng: random.Random) -> dict:
    docs = [[w for w in WORD.findall(t.lower()) if len(w) >= 3] for t in texts]
    freq = Counter(w for d in docs for w in d)
    stop = {w for w, _ in freq.most_common(150)}
    chunks = []
    for d in docs:
        ws = [w for w in d if w not in stop]
        cs = [set(ws[i:i + chunk]) for i in range(0, len(ws) - chunk + 1, chunk)]
        if len(cs) >= 2:
            chunks.append(cs)
    res = {"docs": len(chunks), "chunks/doc": sum(map(len, chunks)) / max(1, len(chunks))}
    for d in gaps:
        vals = [jaccard(cs[i], cs[i + d]) for cs in chunks for i in range(len(cs) - d)]
        res[f"J{d}"] = sum(vals) / max(1, len(vals))
    # cross-document floor: random chunk pairs from different documents
    floor_vals = []
    for _ in range(20000):
        a, b = rng.sample(range(len(chunks)), 2)
        floor_vals.append(jaccard(rng.choice(chunks[a]), rng.choice(chunks[b])))
    res["floor"] = sum(floor_vals) / len(floor_vals)
    thr = 1.5 * res["floor"]
    adj = [jaccard(cs[i], cs[i + 1]) for cs in chunks for i in range(len(cs) - 1)]
    res["switch"] = sum(v <= thr for v in adj) / max(1, len(adj))
    return res


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("files", nargs="+")
    p.add_argument("--labels", default=None)
    p.add_argument("--chunk", type=int, default=45, help="content words per chunk (45 ~ one 64-token block)")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    labels = a.labels.split(",") if a.labels else a.files
    gaps = (1, 2, 4, 8)
    print(f"chunk = {a.chunk} content words; ratio = J(d)/floor; switch = share of adjacent pairs <= 1.5*floor")
    print(f"{'system':>22} | {'docs':>4} {'ch/doc':>6} | {'J1':>6} {'J2':>6} {'J4':>6} {'J8':>6} {'floor':>6} | "
          f"{'r1':>5} {'r2':>5} {'r4':>5} {'r8':>5} | {'switch':>6}")
    print("-" * 118)
    for path, label in zip(a.files, labels):
        r = analyse(load_texts(path), a.chunk, gaps, random.Random(a.seed))
        print(f"{label:>22} | {r['docs']:>4} {r['chunks/doc']:>6.1f} | "
              + " ".join(f"{r[f'J{d}']:>6.3f}" for d in gaps) + f" {r['floor']:>6.3f} | "
              + " ".join(f"{r[f'J{d}'] / max(1e-9, r['floor']):>5.2f}" for d in gaps) + f" | {r['switch']:>6.3f}")


if __name__ == "__main__":
    main()
