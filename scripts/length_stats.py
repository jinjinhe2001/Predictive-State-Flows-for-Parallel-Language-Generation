"""Document-length statistics of text dumps: how the fixed 1024-token canvas shows
up in the generated documents versus real ones.

  python -m scripts.length_stats LABEL=path.jsonl [...]
Per file: mean / median T5-token length, fraction of documents that end before the canvas (an EOS was
produced -- length < 1000 tokens), fraction at the cap (>= 1000), and the fraction of documents that
end in a full stop / quote (a loose "ends on a sentence boundary" check).
"""
from __future__ import annotations

import argparse
import json
import statistics

from transformers import AutoTokenizer


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("specs", nargs="+")
    p.add_argument("--num", type=int, default=1000)
    a = p.parse_args()
    tok = AutoTokenizer.from_pretrained("google-t5/t5-small")
    print(f"{'system':26s} {'docs':>5s} {'mean':>6s} {'median':>6s} {'<1000':>6s} {'>=1000':>6s} {'sent.end':>8s}")
    for spec in a.specs:
        label, path = spec.split("=", 1)
        texts = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    t = row if isinstance(row, str) else (row.get("text") or row.get("generated")
                                                         or next((v for v in row.values() if isinstance(v, str)), None))
                    if isinstance(t, str):
                        texts.append(t)
                if len(texts) >= a.num:
                    break
        lens = [len(tok(t, add_special_tokens=False).input_ids) for t in texts]
        short = sum(l < 1000 for l in lens) / len(lens)
        ends = sum(t.rstrip()[-1:] in ".!?\"'”’)" for t in texts if t.strip()) / len(texts)
        print(f"{label:26s} {len(lens):5d} {statistics.mean(lens):6.0f} {statistics.median(lens):6.0f} "
              f"{100 * short:5.1f}% {100 * (1 - short):5.1f}% {100 * ends:7.1f}%", flush=True)


if __name__ == "__main__":
    main()
