"""Dump held-out OWT documents (our T5 canvas, detokenized) to JSONL as the real-text reference
for scripts.topic_metrics and scripts.mauve_eval.  python -m scripts.dump_real_texts --num 1000 --out real_owt_1000.jsonl
"""
from __future__ import annotations

import argparse
import json

from psf.config import Config
from psf.utils.data import decode, load_tokenizer, make_dataloaders


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--num", type=int, default=1000)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    cfg = Config()
    cfg.num_blocks = 16
    cfg.train.batch_size = 50
    cfg.data.num_workers = 0
    cfg.__post_init__()
    tok = load_tokenizer(cfg)
    _, val = make_dataloaders(cfg)
    n = 0
    with open(a.out, "w", encoding="utf-8") as f:
        for batch in val:
            for i in range(batch.tokens.shape[0]):
                ids = batch.tokens[i][batch.mask[i]].tolist()
                f.write(json.dumps({"text": decode(tok, ids)}, ensure_ascii=False) + "\n")
                n += 1
                if n >= a.num:
                    break
            if n >= a.num:
                break
    print(f"wrote {n} -> {a.out}")


if __name__ == "__main__":
    main()
