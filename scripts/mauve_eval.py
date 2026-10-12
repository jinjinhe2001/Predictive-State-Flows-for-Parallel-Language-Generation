"""MAUVE (Pillutla et al. 2021) of generated text dumps against held-out OpenWebText.

  python -m scripts.mauve_eval --real real_owt_1000.jsonl LABEL=path.jsonl [LABEL=path.jsonl ...]
Features: GPT-2 Large final-layer hidden state at the last token of each document (the MAUVE default
featurization), documents truncated to 1024 tokens, both sides 1000 documents; k-means quantization with the
package defaults, 5 quantization seeds averaged. Higher is better (1 = same distribution). The Frontier
Integral (lower is better) is printed as well. Text field: "text" or "generated" (ELF dumps).
"""
from __future__ import annotations

import argparse
import json

import mauve
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load(path: str, n: int) -> list[str]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                t = row if isinstance(row, str) else (row.get("text") or row.get("generated")
                                                     or next((v for v in row.values() if isinstance(v, str)), None))
                if isinstance(t, str) and t.strip():
                    out.append(t)
            if len(out) >= n:
                break
    return out


@torch.no_grad()
def featurize(texts: list[str], model, tok, device, bs: int = 8, max_len: int = 1024) -> np.ndarray:
    feats = []
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i + bs], return_tensors="pt", padding=True, truncation=True, max_length=max_len)
        ids, mask = enc.input_ids.to(device), enc.attention_mask.to(device)
        h = model(input_ids=ids, attention_mask=mask, output_hidden_states=True).hidden_states[-1]
        last = mask.sum(1) - 1                                        # right padding: last real token
        feats.append(h[torch.arange(h.shape[0], device=device), last].float().cpu().numpy())
    return np.concatenate(feats, 0)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("specs", nargs="+", help="LABEL=path.jsonl")
    p.add_argument("--real", default="real_owt_1000.jsonl")
    p.add_argument("--num", type=int, default=1000)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--featurize", default="openai-community/gpt2-large")
    p.add_argument("--max-len", type=int, default=1024,
                   help="judge tokens kept per document on BOTH sides; 512 removes the document-ending effect "
                        "(systems that never emit EOS fill the whole 1024-token canvas, real documents end early)")
    a = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.featurize)
    tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(a.featurize).to(device).eval()
    real = load(a.real, a.num)
    real_feats = featurize(real, model, tok, device, max_len=a.max_len)
    print(f"{'system':28s} {'MAUVE':>7s} {'FI':>7s}  ({len(real)} real docs, {a.seeds} quantization seeds, "
          f"first {a.max_len} judge tokens)", flush=True)
    for spec in a.specs:
        label, path = spec.split("=", 1)
        gen = load(path, a.num)
        gen_feats = featurize(gen, model, tok, device, max_len=a.max_len)
        ms, fis = [], []
        for s in range(a.seeds):
            out = mauve.compute_mauve(p_features=real_feats, q_features=gen_feats, seed=s, verbose=False)
            ms.append(out.mauve)
            fis.append(out.frontier_integral)
        m = sum(ms) / len(ms)
        sd = (sum((x - m) ** 2 for x in ms) / max(len(ms) - 1, 1)) ** 0.5
        print(f"{label:28s} {m:7.4f} {sum(fis) / len(fis):7.4f}  (+-{sd:.4f}, n={len(gen)})", flush=True)


if __name__ == "__main__":
    main()
