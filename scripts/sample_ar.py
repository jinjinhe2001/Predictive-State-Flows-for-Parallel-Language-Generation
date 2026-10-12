"""Sample the AR baseline (scripts/train_ar.py checkpoint) into the eval_gen text-dump format and
report its held-out T5-token perplexity.

  python -m scripts.sample_ar --ckpt runs/ar-197m/ar.pt --num-samples 1000 --gen-batch 100 \
      --settings 0.9:1.0,1.0:1.0,1.1:1.0,1.0:0.95 --save-texts texts_ar197m
Each setting temperature:top_p writes <save-texts>_t{temp}_p{top_p}.jsonl ({"text": ...} per line,
document cut at its first EOS) -- score them with `python -m scripts.eval_gen --texts FILE`.
"""
from __future__ import annotations

import argparse
import json
import math

import torch

from psf.config import Config
from psf.models.ar_lm import ARConfig, ARLM
from psf.utils.data import decode, load_tokenizer, make_dataloaders
from psf.utils.misc import get_device, set_seed


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--num-samples", type=int, default=1000)
    p.add_argument("--gen-batch", type=int, default=100)
    p.add_argument("--settings", default="1.0:1.0")
    p.add_argument("--save-texts", default=None)
    p.add_argument("--val-batches", type=int, default=20, help="held-out batches for the T5-token perplexity (0 = skip)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-repeat-ngram", type=int, default=0, help="document-wide n-gram ban (n > 0), as for the writer")
    p.add_argument("--device", default="auto")
    a = p.parse_args()
    set_seed(a.seed)
    device = get_device(a.device)
    torch.set_float32_matmul_precision("high")
    cfg = Config()
    ck = torch.load(a.ckpt, map_location=device, weights_only=False)
    model = ARLM(ARConfig(**ck["ar_config"])).to(device).eval()
    model.load_state_dict(ck["model"])
    print(f"AR baseline {model.num_params() / 1e6:.1f}M params, step {ck.get('step')}", flush=True)
    tokenizer = load_tokenizer(cfg)

    if a.val_batches > 0:
        cfg.train.batch_size = 32
        _, val_loader = make_dataloaders(cfg)
        tot = n = 0.0
        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                if i >= a.val_batches:
                    break
                batch = batch.to(device)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    nll, cnt = model(batch.tokens, batch.mask)
                tot += float(nll) * float(cnt)
                n += float(cnt)
        print(f"  held-out: nll={tot / n:.4f} nats/token | ppl(T5 tokens)={math.exp(tot / n):.2f} over {int(n)} tokens",
              flush=True)

    for spec in a.settings.split(","):
        temp, top_p = (float(x) for x in spec.split(":"))
        texts = []
        for i in range(0, a.num_samples, a.gen_batch):
            b = min(a.gen_batch, a.num_samples - i)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                toks = model.generate(b, device, temperature=temp, top_p=top_p, no_repeat_ngram=a.no_repeat_ngram)
            for row in toks.tolist():
                if cfg.data.eos_token_id in row:
                    row = row[: row.index(cfg.data.eos_token_id)]
                texts.append(decode(tokenizer, row))
        lens = [len(t) for t in texts]
        print(f"  t={temp:.2f} p={top_p:.2f}: {len(texts)} docs, mean {sum(lens) / len(lens):.0f} chars", flush=True)
        if a.save_texts:
            path = f"{a.save_texts}_t{temp:.2f}_p{top_p:.2f}{f'_ban{a.no_repeat_ngram}' if a.no_repeat_ngram else ''}.jsonl"
            with open(path, "w", encoding="utf-8") as f:
                for t in texts:
                    f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")
            print(f"  wrote {path}", flush=True)


if __name__ == "__main__":
    main()
