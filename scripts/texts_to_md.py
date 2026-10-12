"""Turn a JSONL of generated texts into a readable sample dossier (markdown).

Accepts our ``eval_gen --save-texts`` dumps ('text' field) and ELF's
``all_generated_*.jsonl`` ('generated' field; the first string field is used).
  python -m scripts.texts_to_md texts.jsonl --num 20 --label "K=16 | cfg 4" --out samples_compare/x.md
"""
from __future__ import annotations

import argparse
import json


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("jsonl")
    p.add_argument("--num", type=int, default=20)
    p.add_argument("--max-chars", type=int, default=3000)
    p.add_argument("--label", default=None)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    texts = []
    with open(args.jsonl, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            t = row if isinstance(row, str) else row.get("text")
            if t is None:
                t = next((v for v in row.values() if isinstance(v, str)), "")
            if t.strip():
                texts.append(t)
            if len(texts) >= args.num:
                break

    with open(args.out, "w", encoding="utf-8") as f:
        f.write(f"# {args.label or args.jsonl}\n\n")
        f.write(f"source: `{args.jsonl}` — first {len(texts)} documents, each truncated to "
                f"{args.max_chars} chars. Unedited.\n\n")
        for i, t in enumerate(texts, 1):
            body = t[:args.max_chars] + (" […]" if len(t) > args.max_chars else "")
            f.write(f"## {i}\n\n{body}\n\n---\n\n")
    print(f"wrote {len(texts)} samples -> {args.out}")


if __name__ == "__main__":
    main()
