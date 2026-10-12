"""Unconditional GPT-2 samples as an autoregressive reference point for the S-axis.

Writes a JSONL ('text' field) that `scripts.eval_gen --texts` scores with our judge.
  python -m scripts.sample_gpt2 --model gpt2 --temperature 1.0 --top-p 0.95 --out gpt2_t1.0_p0.95.jsonl
"""
from __future__ import annotations

import argparse
import json

import torch


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="gpt2")
    p.add_argument("--num", type=int, default=1000)
    p.add_argument("--batch", type=int, default=50)
    p.add_argument("--max-new-tokens", type=int, default=1023)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-repeat-ngram-size", type=int, default=0, help="HF no_repeat_ngram_size (document-wide n-gram ban)")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model).to(device).eval()
    bos = tok.eos_token_id
    n = 0
    with open(args.out, "w", encoding="utf-8") as f, torch.no_grad():
        while n < args.num:
            b = min(args.batch, args.num - n)
            ids = torch.full((b, 1), bos, dtype=torch.long, device=device)
            # min_new_tokens = max: forbid EOS so every document is exactly max_new_tokens
            # long, like every other system in the comparison (fixed 1024-token canvas).
            out = model.generate(ids, do_sample=True, temperature=args.temperature, top_p=args.top_p,
                                 max_new_tokens=args.max_new_tokens, min_new_tokens=args.max_new_tokens,
                                 pad_token_id=bos, no_repeat_ngram_size=args.no_repeat_ngram_size)
            for row in out[:, 1:]:
                f.write(json.dumps({"text": tok.decode(row, skip_special_tokens=True)},
                                   ensure_ascii=False) + "\n")
            n += b
            print(f"  {n}/{args.num}", flush=True)
    print(f"wrote {n} -> {args.out}")


if __name__ == "__main__":
    main()
