"""Document-wide hygiene statistics for text dumps.

For each dump: the paper's loop rate (fraction of documents whose first 256 T5 tokens contain an 8-gram
three or more times) next to the same statistic over the whole document (all T5 tokens, up to 1024), the
mean number of distinct looping 8-grams per looping document, and corpus-level distinct-4
(unique 4-grams / total 4-grams over the dump, as in diversity_memo.py).

  python -m scripts.loop_full LABEL=path.jsonl [LABEL=path.jsonl ...] [--num 1000]
"""
from __future__ import annotations

import argparse
import json
from collections import Counter

from transformers import AutoTokenizer


def load(path: str, n: int) -> list[str]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            t = row if isinstance(row, str) else (row.get("text") or row.get("generated")
                                                 or next((v for v in row.values() if isinstance(v, str)), None))
            if isinstance(t, str) and t.strip():
                out.append(t)
            if len(out) >= n:
                break
    return out


def loops(ids: list[int], prefix: int, order: int = 8, min_count: int = 3) -> int:
    ids = ids[:prefix]
    c = Counter(tuple(ids[i:i + order]) for i in range(len(ids) - order + 1))
    return sum(1 for v in c.values() if v >= min_count)


def distinct_n(id_lists: list[list[int]], n: int) -> float:
    seen, total = set(), 0
    for ids in id_lists:
        for i in range(len(ids) - n + 1):
            seen.add(tuple(ids[i:i + n])); total += 1
    return len(seen) / max(total, 1)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("specs", nargs="+", help="LABEL=path.jsonl")
    p.add_argument("--num", type=int, default=1000)
    p.add_argument("--canvas", type=int, default=1024)
    a = p.parse_args()
    tok = AutoTokenizer.from_pretrained("google-t5/t5-small")
    print(f"{'system':34s} {'loop@256':>9s} {'loop@doc':>9s} {'loops/doc':>10s} {'len':>6s} {'distinct4':>10s}")
    for spec in a.specs:
        label, path = spec.split("=", 1)
        texts = load(path, a.num)
        ids = [tok(t, add_special_tokens=False).input_ids[: a.canvas] for t in texts]
        l256 = [loops(x, 256) for x in ids]
        ldoc = [loops(x, a.canvas) for x in ids]
        n = len(ids)
        r256 = sum(1 for v in l256 if v > 0) / n
        rdoc = sum(1 for v in ldoc if v > 0) / n
        per = sum(v for v in ldoc if v > 0) / max(sum(1 for v in ldoc if v > 0), 1)
        mean_len = sum(len(x) for x in ids) / n
        print(f"{label:34s} {100 * r256:8.1f}% {100 * rdoc:8.1f}% {per:10.1f} {mean_len:6.0f} {distinct_n(ids, 4):10.3f}",
              flush=True)


if __name__ == "__main__":
    main()
