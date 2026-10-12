"""Prompted continuation without retraining.

Given the first P tokens of a held-out document (P = m blocks), the prefix encoder yields the exact
prefix states s_1..s_{m+1} and codes r_1..r_m; the prior fills in the remaining plan by replacement
inpainting (flow.sample_inpaint); the writer decodes all blocks in parallel with the prompt blocks
teacher-forced. Baselines from the same prompts: GPT-2 small (HF generate) and, if given, our AR
baseline (scripts/train_ar.py). The real continuation is the reference row.

Metrics on the continuation only (judge = GPT-2 Large): perplexity given the prompt, per-document
unigram entropy, repeated-8-gram rate, and prompt relevance = fraction of continuation content words
that occur in the prompt, minus the same quantity for mismatched prompt/continuation pairs (chance).

  CUDA_VISIBLE_DEVICES=0 python -m scripts.eval_prompted --ckpt runs/prior-srpl-rc64-look-k8/prior.pt \
      --arch record --num-blocks 8 --prompt-tokens 256 --num-docs 500 --settings 0.75:0.95,1.0:0.62 \
      --gpt2 1.0:1.0,0.9:0.95 --ar-ckpt runs/ar-197m/ar.pt --ar-settings 1.0:1.0 --out prompted_k8.jsonl
"""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from psf.config import Config
from psf.diffusion.flow import sample_inpaint
from psf.models.psf import PSF
from psf.utils.data import decode, load_tokenizer, make_dataloaders
from psf.utils.misc import get_device, load_checkpoint, set_seed

STOP = set("""the a an and or but if then of to in on at by for with from as is are was were be been being it its this
that these those he she they we you i his her their our your my me him them us who whom which what when where why how
not no nor so than too very can will just do does did done have has had having would should could may might must shall
also into over under about after before between through during without within up down out off again further once here
there all any both each few more most other some such only own same s t don now said says say one two new first last
also like get got make made many much well back even still way because while against""".split())
WORD = re.compile(r"[a-z]{4,}")


def content_words(text: str) -> set[str]:
    return {w for w in WORD.findall(text.lower()) if w not in STOP}


def build_model(arch: str, K: int, ckpt: str, device, prompt_attn: bool = False) -> tuple[PSF, Config]:
    cfg = Config()
    cfg.num_blocks = K
    cfg.decoder.prompt_attn = prompt_attn
    cfg.decoder.depth = 18
    cfg.denoiser.bottleneck_dim = 256
    if arch == "big":
        cfg.latent_dim = 1024
        cfg.denoiser.hidden_size, cfg.denoiser.depth, cfg.denoiser.num_heads = 1024, 24, 16
    cfg.__post_init__()
    cfg.encoder_type = "prefix"
    cfg.decoder.causal_latents = True
    cfg.decoder.boundary_tokens = True
    cfg.decoder.boundary_window = 1
    cfg.decoder.noise_level_cond = True
    cfg.rc_dim = 64
    cfg.decoder.rc_lookahead = True
    cfg.denoiser.num_cfg_tokens = 4
    cfg.flow.num_sampling_steps = 50
    model = PSF(cfg).to(device).eval()
    load_checkpoint(ckpt, model, device)
    return model, cfg


@torch.no_grad()
def judge_continuations(pairs: list[tuple[str, str]], judge, tok, device, bs: int = 8) -> dict:
    """pairs of (prompt_text, continuation_text). Returns corpus PPL of the continuation given the prompt,
    mean per-doc unigram entropy of the continuation, repeated-8-gram rate, token count."""
    tot_nll = tot_tok = 0.0
    ents, loops = [], 0
    for i in range(0, len(pairs), bs):
        chunk = pairs[i:i + bs]
        fulls = [p + c for p, c in chunk]
        bounds = [len(tok(p).input_ids) for p, _ in chunk]
        enc = tok(fulls, return_tensors="pt", padding=True, truncation=True, max_length=1024)
        ids, mask = enc.input_ids.to(device), enc.attention_mask.to(device)
        logits = judge(input_ids=ids, attention_mask=mask).logits.float()
        logp = F.log_softmax(logits[:, :-1], dim=-1)
        nll = -logp.gather(-1, ids[:, 1:, None]).squeeze(-1)           # predicts token j+1
        for j, b in enumerate(bounds):
            m = mask[j, 1:].clone()
            m[: max(b - 1, 0)] = 0                                      # keep predictions of tokens >= b
            tot_nll += float((nll[j] * m).sum())
            tot_tok += float(m.sum())
            cont = [t for t, k in zip(ids[j, 1:].tolist(), m.tolist()) if k]
            if cont:
                c = Counter(cont)
                n = len(cont)
                ents.append(-sum(v / n * math.log(v / n) for v in c.values()))
                grams = Counter(tuple(cont[k:k + 8]) for k in range(max(0, n - 7)))
                loops += int(bool(grams) and max(grams.values()) >= 2)
    return {"ppl": math.exp(tot_nll / max(tot_tok, 1)), "entropy": sum(ents) / max(len(ents), 1),
            "rep8_rate": loops / max(len(pairs), 1), "tokens": int(tot_tok)}


def relevance(prompts: list[str], conts: list[str]) -> tuple[float, float]:
    """(matched overlap, mismatched overlap): fraction of continuation content words present in the prompt."""
    P = [content_words(p) for p in prompts]
    C = [content_words(c) for c in conts]

    def frac(pi, ci):
        return len(C[ci] & P[pi]) / max(len(C[ci]), 1)

    n = len(P)
    matched = sum(frac(i, i) for i in range(n)) / n
    mism = sum(frac((i + 1) % n, i) for i in range(n)) / n
    return matched, mism


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--arch", choices=["record", "big"], default="record")
    p.add_argument("--num-blocks", type=int, default=8)
    p.add_argument("--prompt-tokens", type=int, default=256)
    p.add_argument("--prompt-attn", action="store_true", help="checkpoint is a prompt-aware writer")
    p.add_argument("--no-feed-prompt", action="store_true",
                   help="control: build the prompt-aware writer but withhold the prompt memory")
    p.add_argument("--num-docs", type=int, default=500)
    p.add_argument("--batch", type=int, default=50)
    p.add_argument("--settings", default="0.75:0.95", help="ours: temperature:top_p list")
    p.add_argument("--cfg-scale", type=float, default=1.0)
    p.add_argument("--resample", default="0", help="RePaint resampling counts to try for the inpainted plan, e.g. 0,4")
    p.add_argument("--lex-bonus", default="0", help="decode-time lexical coupling bonuses to try (after the nucleus cut), e.g. 0,3")
    p.add_argument("--oracle", action="store_true",
                   help="add the ceiling row: plan encoded from the full real document, only the prompt blocks forced")
    p.add_argument("--gpt2", default="1.0:1.0", help="GPT-2 small settings temperature:top_p list ('' = skip)")
    p.add_argument("--gpt2-name", default="openai-community/gpt2")
    p.add_argument("--ar-ckpt", default=None)
    p.add_argument("--ar-settings", default="1.0:1.0")
    p.add_argument("--judge", default="openai-community/gpt2-large")
    p.add_argument("--out", default=None, help="jsonl with prompt / real / generated continuations")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    a = p.parse_args()
    set_seed(a.seed)
    device = get_device(a.device)
    torch.set_float32_matmul_precision("high")

    model, cfg = build_model(a.arch, a.num_blocks, a.ckpt, device, prompt_attn=a.prompt_attn)
    tokenizer = load_tokenizer(cfg)
    K, S, L = cfg.num_blocks, cfg.block_span, cfg.max_seq_len
    P = a.prompt_tokens
    assert P % S == 0 and 0 < P < L, "prompt must be a whole number of blocks"
    m = P // S
    D, R = cfg.latent_dim, cfg.rc_dim
    bob = cfg.data.bob_token_id

    # held-out documents with at least P + 256 real tokens
    cfg.train.batch_size = a.batch
    _, val_loader = make_dataloaders(cfg)
    docs = []
    for batch in val_loader:
        for row, msk in zip(batch.tokens, batch.mask):
            if int(msk.sum()) >= P + 256:
                docs.append(row)
        if len(docs) >= a.num_docs:
            break
    docs = torch.stack(docs[: a.num_docs]).to(device)
    n_docs = docs.shape[0]
    prompts = [decode(tokenizer, r[:P].tolist()) for r in docs]
    reals = []
    for r in docs:
        ids = r.tolist()
        n = ids.index(cfg.data.eos_token_id) if cfg.data.eos_token_id in ids else L
        full = decode(tokenizer, ids[:n])
        pr = decode(tokenizer, ids[:P])
        reals.append(full[len(pr):] if full.startswith(pr) else " " + decode(tokenizer, ids[P:n]))
    print(f"{n_docs} held-out prompts of {P} tokens ({m} of {K} blocks); real continuation = rest of the document",
          flush=True)

    systems: dict[str, list[str]] = {"real": reals}

    # ---- ours: inpaint the plan (or take the oracle plan), teacher-force the prompt blocks ----
    def ours(temp: float, top_p: float, resample: int, oracle: bool, lex_bonus: float = 0.0) -> list[str]:
        conts = []
        for i in range(0, n_docs, a.batch):
            tok = docs[i:i + a.batch]
            B = tok.shape[0]
            with torch.no_grad():
                if oracle:
                    enc = model.encode(tok, tok != cfg.data.pad_token_id)
                    zf = model.prior_latents(enc, source="mean")
                else:
                    prompt_only = tok.clone()
                    prompt_only[:, P:] = cfg.data.pad_token_id
                    pmask = torch.zeros_like(tok, dtype=torch.bool)
                    pmask[:, :P] = True
                    enc = model.encode(prompt_only, pmask)
                    z_known = model.denoiser.normalize(model.prior_latents(enc, source="mean"))
                    known = torch.zeros_like(z_known, dtype=torch.bool)
                    known[:, :m, :] = True        # prompt blocks: state + code
                    known[:, m, :D] = True        # first written block: its prefix state is a function of the prompt
                    z = sample_inpaint(model.denoiser, z_known, known, cfg.flow, device, cfg_scale=a.cfg_scale,
                                       resample=resample)
                    zf = model.denoiser.denormalize(z)
                s, rc = model.split_latents(zf)
                force = torch.full((B, K, S), -1, dtype=torch.long, device=device)
                force[:, :m] = tok.reshape(B, K, S)[:, :m]
                pkw = {}
                if a.prompt_attn and not a.no_feed_prompt:
                    pkw = dict(prompt=tok[:, :P], prompt_mask=tok[:, :P] != cfg.data.pad_token_id)
                blocks = model.decoder.generate(s, bob, temperature=temp, top_p=top_p, z_level=0.0 if not oracle else 1.0,
                                                rc=rc, force=force, lex_bonus=lex_bonus, lex_post=lex_bonus > 0.0, **pkw)
                flat, lens = model.truncate_at_eos(blocks)
            for row, n in zip(flat.tolist(), lens.tolist()):
                n = max(n, P)
                full = decode(tokenizer, row[:n])
                pr = decode(tokenizer, row[:P])
                conts.append(full[len(pr):] if full.startswith(pr) else " " + decode(tokenizer, row[P:n]))
        return conts

    for spec in [x for x in a.settings.split(",") if x]:
        temp, top_p = (float(x) for x in spec.split(":"))
        for rs in (int(x) for x in a.resample.split(",")):
            for lb in (float(x) for x in a.lex_bonus.split(",")):
                tag = (f"ours t={temp:.2f} p={top_p:.2f}" + (f" resample {rs}" if rs else "")
                       + (f" lex {lb:g}" if lb > 0 else ""))
                systems[tag] = ours(temp, top_p, rs, False, lb)
                print(f"  {tag}: done", flush=True)
        if a.oracle:
            systems[f"ours oracle plan t={temp:.2f} p={top_p:.2f}"] = ours(temp, top_p, 0, True)
            print(f"  oracle {spec}: done", flush=True)

    # ---- AR baseline (ours, same data) ----
    if a.ar_ckpt:
        from psf.models.ar_lm import ARConfig, ARLM

        ck = torch.load(a.ar_ckpt, map_location=device, weights_only=False)
        ar = ARLM(ARConfig(**ck["ar_config"])).to(device).eval()
        ar.load_state_dict(ck["model"])
        for spec in a.ar_settings.split(","):
            temp, top_p = (float(x) for x in spec.split(":"))
            conts = []
            for i in range(0, n_docs, a.batch):
                tok = docs[i:i + a.batch]
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out = ar.generate(tok.shape[0], device, temperature=temp, top_p=top_p, prefix=tok[:, :P])
                for row in out.tolist():
                    n = row.index(cfg.data.eos_token_id) if cfg.data.eos_token_id in row[P:] else L
                    n = max(n, P)
                    full = decode(tokenizer, row[:n])
                    pr = decode(tokenizer, row[:P])
                    conts.append(full[len(pr):] if full.startswith(pr) else " " + decode(tokenizer, row[P:n]))
            systems[f"AR-197M t={temp:.2f} p={top_p:.2f}"] = conts
            print(f"  AR {spec}: done", flush=True)
        del ar
        torch.cuda.empty_cache()

    # ---- GPT-2 small from the same prompts (text) ----
    if a.gpt2:
        gtok = AutoTokenizer.from_pretrained(a.gpt2_name)
        gtok.pad_token = gtok.eos_token
        gtok.padding_side = "left"
        g = AutoModelForCausalLM.from_pretrained(a.gpt2_name).to(device).eval()
        for spec in a.gpt2.split(","):
            temp, top_p = (float(x) for x in spec.split(":"))
            conts = []
            for i in range(0, n_docs, a.batch):
                enc = gtok(prompts[i:i + a.batch], return_tensors="pt", padding=True, truncation=True, max_length=400)
                ids, mask = enc.input_ids.to(device), enc.attention_mask.to(device)
                with torch.no_grad():
                    out = g.generate(input_ids=ids, attention_mask=mask, do_sample=True, temperature=temp, top_p=top_p,
                                     max_new_tokens=1024 - ids.shape[1], pad_token_id=gtok.eos_token_id)
                for row in out[:, ids.shape[1]:].tolist():
                    if gtok.eos_token_id in row:
                        row = row[: row.index(gtok.eos_token_id)]
                    conts.append(gtok.decode(row))
            systems[f"GPT-2 small t={temp:.2f} p={top_p:.2f}"] = conts
            print(f"  GPT-2 {spec}: done", flush=True)
        del g
        torch.cuda.empty_cache()

    # ---- judge ----
    jtok = AutoTokenizer.from_pretrained(a.judge)
    jtok.pad_token = jtok.eos_token
    judge = AutoModelForCausalLM.from_pretrained(a.judge).to(device).eval()
    print(f"\n{'system':34s} {'cont. PPL':>9s} {'entropy':>8s} {'rep8':>6s} {'relev.':>7s} {'chance':>7s} {'chars':>6s}")
    rows = {}
    for name, conts in systems.items():
        met = judge_continuations(list(zip(prompts, conts)), judge, jtok, device)
        rel, chance = relevance(prompts, conts)
        chars = sum(len(c) for c in conts) / len(conts)
        rows[name] = {**met, "relevance": rel, "chance": chance, "chars": chars}
        print(f"{name:34s} {met['ppl']:9.2f} {met['entropy']:8.3f} {100 * met['rep8_rate']:5.1f}% {100 * rel:6.1f}% "
              f"{100 * chance:6.1f}% {chars:6.0f}", flush=True)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            f.write(json.dumps({"metrics": rows, "prompt_tokens": P, "num_docs": n_docs}) + "\n")
            for i in range(n_docs):
                f.write(json.dumps({"prompt": prompts[i], **{k: v[i] for k, v in systems.items()}},
                                   ensure_ascii=False) + "\n")
        print("wrote", a.out)


if __name__ == "__main__":
    main()
