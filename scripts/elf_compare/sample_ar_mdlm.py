"""Sample the MDLM-paper 110M autoregressive OWT baseline (Google-Drive `ar.ckpt`) with the mdlm
repo's own `_ar_sampler` (Gumbel-max = temperature 1, no nucleus), writing JSONL for our judge.

Run from the mdlm checkout in the bd3lm conda env (flash-attn required by their AR backbone):
  python sample_ar_mdlm.py --ckpt ../bd3lms/gdrive_ckpts/ar.ckpt --num 1024 --batch 16 --out ar_owt_110m.jsonl
Writes <out> (text cut at the first EOS, document-faithful) and <out stem>_fulleos.jsonl (whole canvas).
"""
import argparse
import json
import os
import sys

import torch
from hydra import compose, initialize

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--ckpt", required=True)
p.add_argument("--num", type=int, default=1024)
p.add_argument("--batch", type=int, default=16)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--top-p", type=float, default=1.0,
               help="nucleus truncation on the AR logits (their gen-ppl protocol uses 0.9); 1.0 = their plain sampler")
p.add_argument("--out", required=True)
a = p.parse_args()
sys.argv = [sys.argv[0]]
import main as M  # noqa: E402
import dataloader  # noqa: E402

with initialize(version_base=None, config_path="configs"):
    cfg = compose(config_name="config", overrides=[
        "mode=sample_eval", "model=small-ar", "parameterization=ar", "backbone=ar",
        "data=openwebtext-split", "model.length=1024", f"loader.eval_batch_size={a.batch}",
        f"eval.checkpoint_path={a.ckpt}", "wandb=null", f"seed={a.seed}"])
tok = dataloader.get_tokenizer(cfg)
model = M._load_from_checkpoint(cfg, tok).to("cuda").eval()
if getattr(model, "ema", None):
    # mdlm's EMA wrapper works on a parameter iterable (bd3lms added _get_parameters later)
    params = list(model.backbone.parameters()) + list(model.noise.parameters())
    model.ema.store(params)
    model.ema.copy_to(params)
torch.manual_seed(a.seed)


@torch.no_grad()
def ar_sample(bsz: int, top_p: float):
    """Their `_ar_sampler` (Gumbel-max, T=1) with optional nucleus truncation of the logits."""
    if top_p >= 1.0:
        return model._ar_sampler(bsz)
    L = model.config.model.length
    x = torch.zeros((bsz, L), dtype=torch.long, device=model.device)
    x[:, 0] = tok.bos_token_id
    for i in range(L - 1):
        logits = model.forward(x[:, :i + 1], None)[:, -1].float()
        probs = torch.softmax(logits, dim=-1)
        sp, si = probs.sort(dim=-1, descending=True)
        keep = (sp.cumsum(-1) - sp) < top_p                      # nucleus: smallest set with mass >= p
        sp = sp * keep
        nxt = si.gather(-1, torch.multinomial(sp / sp.sum(-1, keepdim=True), 1)).squeeze(-1)
        x[:, i + 1] = nxt
    return x


stem, ext = os.path.splitext(a.out)
n = 0
with open(a.out, "w", encoding="utf-8") as f_doc, open(f"{stem}_fulleos{ext}", "w", encoding="utf-8") as f_full, \
        torch.no_grad():
    while n < a.num:
        b = min(a.batch, a.num - n)
        x = ar_sample(b, a.top_p)                               # [b, 1024] incl. BOS
        for row in x[:, 1:].tolist():
            text = tok.decode(row, skip_special_tokens=False)
            full = text.replace(tok.bos_token or "<|endoftext|>", "<|endoftext|>")
            while full.startswith("<|endoftext|>"):
                full = full[len("<|endoftext|>"):]
            f_full.write(json.dumps({"text": full.strip()}, ensure_ascii=False) + "\n")
            doc = full.split("<|endoftext|>")[0].strip()
            if doc:
                f_doc.write(json.dumps({"text": doc}, ensure_ascii=False) + "\n")
        n += b
        print(f"  {n}/{a.num}", flush=True)
print(f"wrote {n} -> {a.out} (+ _fulleos)")
