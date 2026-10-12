"""Plan-usage and plan-adherence diagnostics.

Does the writer use the plan, and does the text it writes follow the plan it was given?

  1. Teacher-forced writer NLL on held-out documents under four state conditions: the document's own states
     (true), the states of another held-out document (other-doc), the document's own states with block positions
     2..K permuted (block-shuffled; position 1 is the learned empty-prefix latent), and the dataset-mean state at
     every position (constant). With the prior's-mean codes and with zero codes.
  2. Generation from the prior's sampled plans under the same manipulations (plan / block-shuffled plan /
     mean plan) at the given sampler settings: generative perplexity, entropy and loop rate under GPT-2 Large;
     texts are saved for topic_metrics.
  3. Plan adherence: the generated documents are re-encoded and their prefix states compared with the sampled
     plan they were written from (cosine similarity per block, against the plan of a different sample).
  4. Sampled-vs-encoded plan distributions: Frechet distance and k-NN precision / recall (k = 5) in a 64-d PCA
     space fitted on the encoder states of held-out documents.

  python -m scripts.plan_probe --ckpt runs/prior-srpl-rc64-look-k8/prior.pt <arch flags of eval_gen> \
      --num-docs 1000 --gen-batch 50 --settings 0.75:0.95,1.0:0.62 --out outputs/plan_probe_k8.txt --save-prefix texts_probe_k8
"""
from __future__ import annotations

import argparse
import json
import math
import sys

import numpy as np
import torch
import torch.nn.functional as F

from psf.diffusion.flow import sample as flow_sample
from psf.models.psf import PSF
from psf.utils.data import decode, load_tokenizer, make_dataloaders
from psf.utils.gen_metrics import repetition_metrics
from psf.utils.misc import get_device, load_checkpoint, set_seed
from scripts import eval_gen


def own_args():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--num-docs", type=int, default=1000)
    p.add_argument("--gen-batch", type=int, default=50)
    p.add_argument("--settings", default="0.75:0.95,1.0:0.62")
    p.add_argument("--out", default="outputs/plan_probe.txt")
    p.add_argument("--save-prefix", default=None)
    p.add_argument("--skip-gen", action="store_true")
    p.add_argument("--z-level", type=float, default=0.0)
    p.add_argument("--judge-batch", type=int, default=16)
    mine, rest = p.parse_known_args()
    sys.argv = [sys.argv[0]] + rest
    return mine, eval_gen.parse_args()


def shuffle_blocks(s: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    """Permute positions 1..K-1 of every row independently (position 0 kept)."""
    B, K, _ = s.shape
    out = s.clone()
    for b in range(B):
        perm = torch.randperm(K - 1, generator=gen) + 1
        out[b, 1:] = s[b, perm.to(s.device)]
    return out


@torch.no_grad()
def tf_nll(model: PSF, feats, inputs, targets, block_mask, nf, rc) -> tuple[float, float]:
    logits = model.decoder(feats, inputs, next_first=nf, rc=rc)
    V = logits.shape[-1]
    ce = F.cross_entropy(logits.reshape(-1, V).float(), targets.reshape(-1), reduction="none").reshape(targets.shape)
    m = block_mask.to(ce.dtype)
    return float((ce * m).sum()), float(m.sum())


def frechet(a: np.ndarray, b: np.ndarray) -> float:
    mu1, mu2 = a.mean(0), b.mean(0)
    c1, c2 = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    w, v = np.linalg.eigh(c1)
    s1 = (v * np.sqrt(np.clip(w, 0, None))) @ v.T
    m = s1 @ c2 @ s1
    w2 = np.linalg.eigvalsh((m + m.T) / 2)
    tr_sqrt = float(np.sqrt(np.clip(w2, 0, None)).sum())
    return float(((mu1 - mu2) ** 2).sum() + np.trace(c1) + np.trace(c2) - 2 * tr_sqrt)


def knn_prec_rec(real: torch.Tensor, gen: torch.Tensor, k: int = 5) -> tuple[float, float]:
    def radii(x):
        d = torch.cdist(x, x)
        return d.kthvalue(k + 1, dim=1).values          # k-th neighbour excluding self
    rr, rg = radii(real), radii(gen)
    d_gr = torch.cdist(gen, real)                        # [G, R]
    precision = float((d_gr <= rr[None, :]).any(dim=1).float().mean())
    recall = float((d_gr.T <= rg[None, :]).any(dim=1).float().mean())
    return precision, recall


def main() -> None:
    mine, args = own_args()
    cfg, _ = eval_gen.config_from_args(args)
    set_seed(args.seed)
    device = get_device(args.device)
    torch.set_float32_matmul_precision("high")
    tokenizer = load_tokenizer(cfg)
    model = PSF(cfg).to(device).eval()
    load_checkpoint(args.ckpt, model, device)
    K = cfg.num_blocks
    lines = [f"# plan probe: {args.ckpt} | K={K} | {mine.num_docs} docs | settings {mine.settings}"]
    log = lambda s: (print(s, flush=True), lines.append(s))

    # ---------------- 1. teacher-forced NLL under state manipulations ----------------
    _, val_loader = make_dataloaders(cfg)
    gen = torch.Generator().manual_seed(args.seed)
    batches, feats_all = [], []
    n = 0
    for batch in val_loader:
        batch = batch.to(device)
        with torch.no_grad():
            enc = model.encode(batch.tokens, batch.mask)
        batches.append((batch, enc))
        feats_all.append(enc.block_feats.float().cpu())
        n += batch.tokens.shape[0]
        if n >= mine.num_docs:
            break
    real_states = torch.cat(feats_all, 0)[: mine.num_docs]          # [N, K, D]
    mean_state = real_states.mean(0, keepdim=True).to(device)      # [1, K, D]
    conds = ["true", "other-doc", "block-shuffled", "constant"]
    for rc_name in (["prior-mean codes", "zero codes"] if cfg.rc_dim > 0 else ["no codes"]):
        tot = {c: [0.0, 0.0] for c in conds}
        for batch, enc in batches:
            inputs, targets = model.block_view(batch.tokens)
            nf = model.boundary_targets(targets)
            bm = batch.mask.reshape(targets.shape)
            rc = None
            if cfg.rc_dim > 0:
                rc = model.sample_rc(enc, "mean") if rc_name.startswith("prior") else torch.zeros_like(enc.rc_mu)
            f = enc.block_feats
            variants = {"true": f, "other-doc": f.roll(1, dims=0), "block-shuffled": shuffle_blocks(f, gen),
                        "constant": mean_state.expand_as(f).to(f.dtype)}
            for c, fc in variants.items():
                s_, m_ = tf_nll(model, fc, inputs, targets, bm, nf, rc)
                tot[c][0] += s_; tot[c][1] += m_
        log(f"\n## 1. teacher-forced writer NLL, {rc_name} ({int(tot['true'][1])} tokens)")
        base = tot["true"][0] / tot["true"][1]
        for c in conds:
            nll = tot[c][0] / tot[c][1]
            log(f"  {c:16s} {nll:.4f} nats/token  ppl {math.exp(nll):7.2f}  ({100 * (nll / base - 1):+.1f}% vs true)")

    if mine.skip_gen:
        open(mine.out, "w", encoding="utf-8").write("\n".join(lines) + "\n"); return

    # ---------------- 2. generation from manipulated sampled plans ----------------
    from transformers import AutoModelForCausalLM, AutoTokenizer
    judge_tok = AutoTokenizer.from_pretrained(args.judge); judge_tok.pad_token = judge_tok.eos_token
    judge = AutoModelForCausalLM.from_pretrained(args.judge).to(device).eval()
    settings = [tuple(float(x) for x in s.split(":")) for s in mine.settings.split(",")]
    nlat = K * cfg.num_latent_slots + cfg.num_global_latents
    # one fixed set of sampled plans, reused for every manipulation and setting
    plans, codes = [], []
    with torch.no_grad():
        remaining = mine.num_docs
        while remaining > 0:
            b = min(mine.gen_batch, remaining)
            z = flow_sample(model.denoiser, b, nlat, model.prior_dim, cfg.flow, device,
                            cfg_scale=args.cfg_scale, sde_gamma=args.sde_gamma, noise_scale=args.noise_scale)
            s, rc = model.split_latents(model.denoiser.denormalize(z))
            plans.append(s); codes.append(rc)
            remaining -= b
    S_gen = torch.cat(plans, 0)                                       # [N, K, D] (device)
    RC_gen = torch.cat(codes, 0) if codes[0] is not None else None
    mean_plan = S_gen.mean(0, keepdim=True)
    if mine.save_prefix:
        torch.save({"plans": S_gen.cpu(), "codes": None if RC_gen is None else RC_gen.cpu()}, f"{mine.save_prefix}_plans.pt")
    gen_variants = [("plan, prior's codes", "plan", "plan"), ("plan, zero codes", "plan", "zero"),
                    ("block-shuffled plan, zero codes", "shuffle", "zero"), ("mean plan, zero codes", "mean", "zero"),
                    ("other sample's plan for the codes, prior's codes", "roll", "plan")]
    adherence_texts = None
    for temperature, top_p in settings:
        log(f"\n## 2. generation from sampled plans, t={temperature:.2f} p={top_p:.2f}")
        for label, smode, rcmode in gen_variants:
            texts, id_lists, toks_all, lens_all = [], [], [], []
            with torch.no_grad():
                for i in range(0, mine.num_docs, mine.gen_batch):
                    s = S_gen[i:i + mine.gen_batch]
                    rc = None if RC_gen is None else RC_gen[i:i + mine.gen_batch]
                    if smode == "shuffle":
                        s = shuffle_blocks(s, gen)
                    elif smode == "mean":
                        s = mean_plan.expand_as(s).contiguous()
                    elif smode == "roll":
                        s = s.roll(1, dims=0)
                    tokens, lens = model.refine_decode(
                        s, [(temperature, top_p)], None, "random", z_level=mine.z_level, landing_lambda=1.0,
                        n_cands=1, shared_noise=False, rc=rc, rc_mode=rcmode)
                    toks_all.append(tokens.cpu()); lens_all.append(lens.cpu())
                    for j in range(tokens.shape[0]):
                        ids = tokens[j][: int(lens[j])].tolist()
                        id_lists.append(ids); texts.append(decode(tokenizer, ids))
            ppl, n_tok, ent_c, ent_s = eval_gen.judge_ppl(texts, judge, judge_tok, device, mine.judge_batch)
            rep = repetition_metrics(id_lists)
            log(f"  {label:48s} PPL {ppl:7.2f}  entropy/sample {ent_s:.3f}  loop {100 * rep['loop8_sample_rate']:.1f}%  "
                f"mean len {sum(len(x) for x in id_lists) / len(id_lists):.0f}")
            if mine.save_prefix:
                tag = label.replace(" ", "_").replace(",", "").replace("'", "")
                with open(f"{mine.save_prefix}_{tag}_t{temperature:.2f}_p{top_p:.2f}.jsonl", "w", encoding="utf-8") as f:
                    for t in texts:
                        f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")
            if smode == "plan" and rcmode == "zero" and adherence_texts is None:
                adherence_texts = (torch.cat(toks_all, 0), torch.cat(lens_all, 0), temperature, top_p)

    # ---------------- 3. plan adherence: re-encode the generated documents ----------------
    if adherence_texts is not None:
        toks, lens, t_, p_ = adherence_texts
        own, other = [], []
        with torch.no_grad():
            for i in range(0, mine.num_docs, mine.gen_batch):
                tb = toks[i:i + mine.gen_batch].to(device); lb = lens[i:i + mine.gen_batch].to(device)
                mask = torch.arange(tb.shape[1], device=device)[None, :] < lb[:, None]
                enc2 = model.encode(tb, mask)
                s_re = enc2.block_feats.float()[:, 1:]                     # re-encoded prefix states, blocks 2..K
                s_pl = S_gen[i:i + mine.gen_batch].float()[:, 1:]          # the plan they were written from
                own.append(F.cosine_similarity(s_re, s_pl, dim=-1).cpu())
                other.append(F.cosine_similarity(s_re, s_pl.roll(1, dims=0), dim=-1).cpu())
        own, other = torch.cat(own, 0), torch.cat(other, 0)                # [N, K-1]
        log(f"\n## 3. plan adherence (plan, zero codes, t={t_:.2f} p={p_:.2f}): cosine(re-encoded s_k, sampled s_k), blocks 2..K")
        log(f"  own plan   mean {own.mean():.3f}   other sample's plan mean {other.mean():.3f}   "
            f"P(own > other) = {(own.mean(1) > other.mean(1)).float().mean():.3f}")
        log("  per block (own / other): " + "  ".join(f"k{k + 2}: {own[:, k].mean():.2f}/{other[:, k].mean():.2f}" for k in range(K - 1)))

    # ---------------- 4. sampled vs encoded plan distributions ----------------
    real = real_states[:, 1:].reshape(-1, real_states.shape[-1]).numpy()
    genp = S_gen.float().cpu()[:, 1:].reshape(-1, S_gen.shape[-1]).numpy()
    mu = real.mean(0); u, sv, vt = np.linalg.svd(real - mu, full_matrices=False)
    P = vt[:64].T
    r64, g64 = (real - mu) @ P, (genp - mu) @ P
    log("\n## 4. sampled plans vs encoder plans of held-out documents (blocks 2..K pooled, PCA-64 fitted on real)")
    log(f"  Frechet distance {frechet(r64, g64):.2f}   (real vs a second half of real: "
        f"{frechet(r64[: len(r64) // 2], r64[len(r64) // 2:]):.2f})")
    rng = np.random.default_rng(0)
    ir = rng.choice(len(r64), min(5000, len(r64)), replace=False); ig = rng.choice(len(g64), min(5000, len(g64)), replace=False)
    pr, rc_ = knn_prec_rec(torch.tensor(r64[ir], device=device), torch.tensor(g64[ig], device=device))
    log(f"  k-NN (k=5) precision {pr:.3f}  recall {rc_:.3f}")
    log(f"  per-coordinate sd: real {real.std(0).mean():.3f}  sampled {genp.std(0).mean():.3f}")
    open(mine.out, "w", encoding="utf-8").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
