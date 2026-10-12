"""Realization-mismatch probes: where inside a block does one-shot pay, and
how far can a better *state source* alone take parallel decoding?

    python -m scripts.probe_realization --ckpt runs/prior-XXX/prior.pt $ARCH \
        --modes real,recon,parallel,rollout,bank,bankrecon,reenc \
        --bank runs/rollout_bank/bank_d1024_200k.pt --num 500 --temps 0.8

Three questions, one script (all zero-training, GPT-2-large judge):

P1  per-position judge PPL inside a block (positions 0..S-1, blocks >= 2):
    An earlier analysis only split "first word" vs "rest". If the rest-of-block penalty
    decays with depth into the block, the parallel penalty is a *seam
    adaptation* cost and longer blocks would dilute it; if it is flat, the
    whole block is written about a phantom prefix and block length will not
    help.

P2  ``bankrecon``: encode ROLLOUT-generated documents (the offline bank that
    rollmix trains the prior on) and decode them in parallel exactly like
    recon does for real docs. That is the ceiling a prior trained purely on
    rollout states (rollmix p -> 1) could reach -- compare against real-doc
    ``recon`` (perfect prior on real states) and ``parallel`` (actual
    one-shot). ``bank`` judges the bank text itself (rollout quality).

P3  ``reenc``: one-shot decoding with mid-decode re-encoding. At decode step
    i every block has i realized tokens; the half-filled canvas is pushed
    through the (masked-pooling) prefix encoder and the refreshed states
    replace the prior's plan for the remaining steps. Still one decode pass
    (S serial steps) plus a few encoder forwards. Zero-training here, so the
    encoder sees half-filled blocks it never trained on -- a directional
    signal only. Spec ``steps:level`` e.g. ``8:1`` or ``4-8-12:0``.
"""

from __future__ import annotations

import argparse
import bisect
import json
import math
import time

import torch
import torch.nn.functional as F

from psf.config import Config
from psf.diffusion.flow import sample as flow_sample
from psf.models.psf import PSF
from psf.models.decoder import sample_tokens
from psf.utils.data import decode, load_tokenizer, make_dataloaders
from psf.utils.misc import get_device, load_checkpoint, set_seed
from scripts.eval_gen import degeneration_line, mean_sample_entropy, unigram_entropy


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="realization-mismatch probes")
    p.add_argument("--ckpt", required=True, help="stage-2 checkpoint (AE + prior)")
    # Architecture flags -- must match the checkpoint (same names as eval_gen).
    p.add_argument("--encoder", default="prefix", choices=["block", "prefix"])
    p.add_argument("--causal-latents", action="store_true")
    p.add_argument("--boundary-tokens", action="store_true")
    p.add_argument("--boundary-window", type=int, default=1)
    p.add_argument("--wave-lead", type=int, default=None,
                   help="wavefront decode lead; must match the checkpoint (see train_ae)")
    p.add_argument("--lex-bag", action="store_true", help="must match the checkpoint")
    p.add_argument("--lex-gate", action="store_true",
                   help="checkpoint has the learned lexical bonus (decoder.gate_proj / lex_scale)")
    p.add_argument("--sc-cfg", action="store_true")
    p.add_argument("--level-cond", action="store_true")
    p.add_argument("--latent-dim", type=int, default=None)
    p.add_argument("--decoder-depth", type=int, default=None)
    p.add_argument("--denoiser-hidden", type=int, default=None)
    p.add_argument("--denoiser-depth", type=int, default=None)
    p.add_argument("--denoiser-heads", type=int, default=None)
    p.add_argument("--denoiser-bottleneck", type=int, default=None)
    p.add_argument("--cross-before-self", action="store_true")
    p.add_argument("--abs-pos-emb", action="store_true")
    p.add_argument("--rc-dim", type=int, default=None)
    p.add_argument("--no-rc-memory", action="store_true")
    p.add_argument("--rc-lookahead", action="store_true",
                   help="block k also reads the next block's code r_{k+1} (must match the checkpoint)")
    # Probe controls.
    p.add_argument("--modes", default="real,recon,parallel",
                   help="comma list of real,recon,parallel,rollout,bank,bankrecon,reenc,"
                        "nolanding (one-shot without landing tokens), codestats, "
                        "planstats (latent collapse + per-segment text diagnostics), "
                        "codemod:<plain|shuffle|indep|zero|noise<t>|scale<a>|sscale<a>>")
    p.add_argument("--num", type=int, default=500, help="documents per mode")
    p.add_argument("--temps", default="0.8", help="comma list of decoder temperatures")
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--z-level", type=float, default=0.0,
                   help="label fed with the prior's plan (parallel / reenc start)")
    p.add_argument("--recon-levels", default="1,0",
                   help="z_level values to decode recon/bankrecon with")
    p.add_argument("--rc-source", default="sample", choices=["sample", "mean"],
                   help="recon/bankrecon: hand the decoder a posterior sample of the "
                        "realization code (what it trained on) or the posterior mean")
    p.add_argument("--bank", default=None, help="rollout bank .pt for bank/bankrecon")
    p.add_argument("--reenc", default="8:1,4-8-12:1,4-8-12:0",
                   help="comma list of re-encode specs steps:level (steps '-'-joined)")
    p.add_argument("--gen-batch", type=int, default=50)
    p.add_argument("--sampling-steps", type=int, default=50)
    p.add_argument("--num-blocks", type=int, default=None,
                   help="K (block span = 1024 / K); must match the checkpoint")
    p.add_argument("--seam-start", type=int, default=3,
                   help="seamoracle: number of block-initial positions forced to the real tokens")
    p.add_argument("--seam-end", type=int, default=2,
                   help="seamoracle: number of block-final positions forced to the real tokens")
    p.add_argument("--df-stagger", type=float, default=0.0,
                   help="staggered (pyramid) reveal strength lambda for the flow sampler; 0 = off")
    p.add_argument("--df-order", default="random", choices=["random", "ltr"])
    p.add_argument("--stitch", action="store_true",
                   help="checkpoint has the seam-stitch mode (must match); enables mode 'stitch'")
    p.add_argument("--stitch-start", type=int, default=3)
    p.add_argument("--stitch-end", type=int, default=2)
    p.add_argument("--stitch-prev", type=int, default=3)
    p.add_argument("--stitch-next", type=int, default=3)
    p.add_argument("--stitch-ckpt", default=None,
                   help="two-decoder inference: stitch-mode decoder from this checkpoint")
    p.add_argument("--stitch-redecode", type=int, default=0,
                   help="main decoder re-decodes this many positions after each rewritten opening")
    p.add_argument("--judge", default="openai-community/gpt2-large")
    p.add_argument("--judge-batch", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--out", default=None, help="JSON file to append results to")
    return p.parse_args()


def build_cfg(args) -> Config:
    cfg = Config()
    if args.latent_dim is not None:
        cfg.latent_dim = args.latent_dim
    if args.decoder_depth is not None:
        cfg.decoder.depth = args.decoder_depth
    if args.denoiser_hidden is not None:
        cfg.denoiser.hidden_size = args.denoiser_hidden
    if args.denoiser_depth is not None:
        cfg.denoiser.depth = args.denoiser_depth
    if args.denoiser_heads is not None:
        cfg.denoiser.num_heads = args.denoiser_heads
    if args.denoiser_bottleneck is not None:
        cfg.denoiser.bottleneck_dim = args.denoiser_bottleneck
    cfg.__post_init__()
    cfg.encoder_type = args.encoder
    cfg.decoder.causal_latents = args.causal_latents
    cfg.decoder.boundary_tokens = args.boundary_tokens
    cfg.decoder.boundary_window = args.boundary_window
    if args.wave_lead is not None:
        cfg.decoder.wave_lead = args.wave_lead
    cfg.decoder.lex_bag = args.lex_bag
    cfg.decoder.lex_gate = args.lex_gate
    cfg.decoder.noise_level_cond = args.level_cond
    cfg.decoder.cross_before_self = args.cross_before_self
    cfg.decoder.abs_pos_emb = args.abs_pos_emb
    if args.rc_dim is not None:
        cfg.rc_dim = args.rc_dim
    if args.no_rc_memory:
        cfg.decoder.rc_memory = False
    if args.rc_lookahead:
        cfg.decoder.rc_lookahead = True
    if args.sc_cfg:
        cfg.denoiser.num_cfg_tokens = 4
    if args.num_blocks is not None:
        cfg.num_blocks = args.num_blocks
    if args.stitch and not args.stitch_ckpt:
        cfg.decoder.stitch = True
    cfg.flow.num_sampling_steps = args.sampling_steps
    cfg.flow.df_stagger = args.df_stagger
    cfg.flow.df_order = args.df_order
    cfg.data.num_workers = 0
    return cfg


# --------------------------------------------------------------------------- #
# Judge with per-block / per-position aggregation
# --------------------------------------------------------------------------- #
class Judge:
    def __init__(self, name: str, device, tok, K: int, S: int, batch: int) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.jt = AutoTokenizer.from_pretrained(name)
        self.jt.pad_token = self.jt.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(name).to(device).eval()
        self.device, self.tok, self.K, self.S, self.batch = device, tok, K, S, batch

    @torch.no_grad()
    def run(self, docs: list[list[int]]) -> dict:
        K, S = self.K, self.S
        nll_k = torch.zeros(K); cnt_k = torch.zeros(K)
        nll_pos = torch.zeros(S); cnt_pos = torch.zeros(S)        # blocks >= 2
        nll_pos1 = torch.zeros(S); cnt_pos1 = torch.zeros(S)      # block 1 only
        total_nll = 0.0; total_tok = 0
        gpt2_rows: list[list[int]] = []
        for s in range(0, len(docs), self.batch):
            chunk = docs[s:s + self.batch]
            texts = [decode(self.tok, ids) for ids in chunk]
            # Char offset of every T5 token boundary (prefix decode, like the
            # block bounds but at token granularity).
            cums = [[0] + [len(decode(self.tok, ids[:j])) for j in range(1, len(ids) + 1)]
                    for ids in chunk]
            enc = self.jt(texts, return_tensors="pt", padding=True, truncation=True,
                          max_length=1024, return_offsets_mapping=True)
            ids_j = enc.input_ids.to(self.device); mask_j = enc.attention_mask.to(self.device)
            logits = self.model(input_ids=ids_j, attention_mask=mask_j).logits.float()
            logp = F.log_softmax(logits[:, :-1], dim=-1)
            nll = -logp.gather(-1, ids_j[:, 1:, None]).squeeze(-1).cpu()
            offs = enc.offset_mapping
            for bi in range(len(chunk)):
                valid = mask_j[bi, 1:].cpu().bool()
                row = [t for t, keep in zip(ids_j[bi].tolist(), mask_j[bi].tolist()) if keep]
                gpt2_rows.append(row)
                cum = cums[bi]; n_t5 = len(chunk[bi])
                for ti in range(nll.shape[1]):
                    if not valid[ti]:
                        break
                    c = int(offs[bi, ti + 1, 0])
                    j = min(max(bisect.bisect_right(cum, c) - 1, 0), n_t5 - 1)
                    blk, pos = j // S, j % S
                    blk = min(blk, K - 1)
                    v = float(nll[bi, ti])
                    total_nll += v; total_tok += 1
                    nll_k[blk] += v; cnt_k[blk] += 1
                    if blk >= 1:
                        nll_pos[pos] += v; cnt_pos[pos] += 1
                    else:
                        nll_pos1[pos] += v; cnt_pos1[pos] += 1
        ppl = lambda a, b: torch.exp(a / b.clamp(min=1))
        first = float(ppl(nll_pos[0], cnt_pos[0]))
        rest = float(ppl(nll_pos[1:].sum(), cnt_pos[1:].sum()))
        buckets = {}
        for name, lo, hi in (("p0", 0, 1), ("p1-3", 1, 4), ("p4-7", 4, 8),
                             ("p8-15", 8, S)):
            buckets[name] = float(ppl(nll_pos[lo:hi].sum(), cnt_pos[lo:hi].sum()))
        return {
            "overall": math.exp(total_nll / max(total_tok, 1)),
            "first": first, "rest": rest, "buckets": buckets,
            "per_pos": ppl(nll_pos, cnt_pos).tolist(),
            "per_pos_block1": ppl(nll_pos1, cnt_pos1).tolist(),
            "per_block": ppl(nll_k, cnt_k).tolist(),
            "gpt2_entropy_sample": mean_sample_entropy(gpt2_rows),
            "gpt2_entropy_corpus": unigram_entropy(gpt2_rows),
            "t5_entropy_sample": mean_sample_entropy(docs),
            "mean_len": sum(len(d) for d in docs) / max(len(docs), 1),
            "n_docs": len(docs),
        }


# --------------------------------------------------------------------------- #
# Document sources
# --------------------------------------------------------------------------- #
def to_docs(toks: torch.Tensor, lens: torch.Tensor) -> list[list[int]]:
    out = []
    for i in range(toks.shape[0]):
        ids = toks[i][: int(lens[i])].tolist()
        if ids:
            out.append(ids)
    return out


def load_bank(path: str, n: int, L: int, pad_id: int, device):
    bank = torch.load(path, map_location="cpu", mmap=True)
    tokens = bank["tokens"][:n].to(torch.int64)
    lengths = bank["lengths"][:n].to(torch.int64)
    if tokens.shape[1] != L:
        raise ValueError(f"bank canvas {tokens.shape[1]} != model L {L}")
    mask = torch.arange(L)[None, :] < lengths[:, None]
    tokens = torch.where(mask, tokens, torch.full_like(tokens, pad_id))
    return tokens.to(device), mask.to(device), lengths


@torch.no_grad()
def partial_canvas(prefix_tokens: torch.Tensor, K: int, S: int, pad_id: int,
                   eos_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    """[B, K, i] realized tokens -> encoder input [B, L] + mask.

    Every block has its first ``i`` positions filled; like ``recanvas`` we
    keep everything through the first EOS (flat order) and mask the rest.
    """
    B, K_, i = prefix_tokens.shape
    canvas = torch.full((B, K, S), pad_id, dtype=torch.long, device=prefix_tokens.device)
    filled = torch.zeros(B, K, S, dtype=torch.bool, device=prefix_tokens.device)
    canvas[:, :, :i] = prefix_tokens
    filled[:, :, :i] = True
    flat, fm = canvas.reshape(B, K * S), filled.reshape(B, K * S)
    is_eos = (flat == eos_id) & fm
    end = torch.where(is_eos.any(dim=1), is_eos.long().argmax(dim=1) + 1,
                      torch.full((B,), K * S, dtype=torch.long, device=flat.device))
    keep = torch.arange(K * S, device=flat.device)[None, :] < end[:, None]
    mask = fm & keep
    return torch.where(mask, flat, torch.full_like(flat, pad_id)), mask


@torch.no_grad()
def generate_reencode(m: PSF, n: int, temperature: float, top_p: float,
                      z_level: float, steps: list[int], reenc_level: float,
                      device, use_landing: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    """One-shot decode with the prefix encoder refreshing the states mid-way.

    ``use_landing=False`` (with ``steps=[]``) is the plain one-shot decode
    without landing tokens: every block runs on the learned no-target
    embedding (``boundary_drop`` keeps it in distribution), so it tests
    whether the seam anchor still pays once a realization code pins each
    block's content."""
    cfg = m.config
    dec = m.decoder
    K, S, D = cfg.num_blocks, cfg.block_span, cfg.latent_dim
    bob, pad, eos = cfg.data.bob_token_id, cfg.data.pad_token_id, cfg.data.eos_token_id
    z = flow_sample(m.denoiser, n, K, m.prior_dim, cfg.flow, device)
    z, rc = m.split_latents(m.denoiser.denormalize(z))
    levels = z.new_full((n, K), float(z_level)) if dec.noise_level_cond else None
    W = dec.boundary_window
    prefix = torch.full((n, K, 1), bob, dtype=torch.long, device=device)
    next_first = None
    for step in range(S):
        if step in steps and step > 0:
            tokens, mask = partial_canvas(prefix[:, :, 1:], K, S, pad, eos)
            # Only the states are refreshed; each block keeps the prior's
            # realization code (its content is only half realized).
            z = m.encode(tokens, mask).block_feats
            if dec.noise_level_cond:
                levels = z.new_full((n, K), float(reenc_level))
        logits = dec.forward(z, prefix, next_first=next_first, levels=levels,
                             rc=rc)[:, :, -1]
        nxt = sample_tokens(logits, temperature, top_p)
        prefix = torch.cat([prefix, nxt.unsqueeze(-1)], dim=2)
        if dec.boundary_tokens and use_landing and step < W:
            if next_first is None:
                next_first = torch.full((n, K, W), -1, dtype=torch.long, device=device)
            next_first[:, :-1, step] = nxt[:, 1:]
    return m.truncate_at_eos(prefix[:, :, 1:])


@torch.no_grad()
def generate_codemod(m: PSF, n: int, temperature: float, top_p: float,
                     z_level: float, spec: str, device) -> tuple[torch.Tensor, torch.Tensor]:
    """One-shot decode with the prior's realization codes tampered with.

    Separates the two things a code can buy: (i) coordination -- r_k agrees
    with what s_{k+1} says block k realized -- and (ii) block-coherent
    randomness -- all 16 tokens of block k follow the same draw. ``shuffle``
    hands every document another document's codes (marginal of r kept, the
    pairing with s destroyed) and ``indep`` draws r ~ N(0, I): if these score
    as well as the untouched codes, (i) does not matter and the gain is pure
    (ii). ``noise<t>`` adds t * N(0, I) on top of the prior's code (more
    latent-side randomness), ``scale<a>`` shrinks it, ``zero`` removes it.
    """
    cfg = m.config
    K = cfg.num_blocks
    z = flow_sample(m.denoiser, n, K, m.prior_dim, cfg.flow, device)
    s, rc = m.split_latents(m.denoiser.denormalize(z))
    if rc is None:
        raise SystemExit("codemod needs an rc_dim > 0 checkpoint")
    if spec == "shuffle":
        rc = rc[torch.randperm(n, device=device)]
    elif spec == "indep":
        rc = torch.randn_like(rc)
    elif spec == "zero":
        rc = torch.zeros_like(rc)
    elif spec.startswith("noise"):
        rc = rc + float(spec[5:]) * torch.randn_like(rc)
    elif spec.startswith("scale"):
        rc = rc * float(spec[5:])
    elif spec.startswith("sscale"):
        # State dispersion knob: shrink/stretch s_hat around the latent mean
        # (the sloppy bottleneck-128 prior under-dispersed s to 93% of real).
        mu_s = m.denoiser.latent_mean[: cfg.latent_dim].to(s.dtype)
        s = mu_s + float(spec[6:]) * (s - mu_s)
    elif spec != "plain":
        raise SystemExit(f"unknown codemod spec {spec}")
    blocks = m.decoder.generate(s, cfg.data.bob_token_id, temperature, top_p,
                                z_level=z_level, rc=rc)
    return m.truncate_at_eos(blocks)


@torch.no_grad()
def code_stats(m: PSF, n: int, device, val_batches) -> None:
    """Magnitude / spread of the prior's codes vs. posterior samples on real docs."""
    cfg = m.config
    z = flow_sample(m.denoiser, n, cfg.num_blocks, m.prior_dim, cfg.flow, device)
    s_prior, rc_prior = m.split_latents(m.denoiser.denormalize(z))
    s_prior = s_prior[:, 1:].reshape(-1, cfg.latent_dim).float()   # skip the BOS slot
    rc_prior = rc_prior.reshape(-1, cfg.rc_dim).float()
    posts, mus, sigmas, states = [], [], [], []
    got = 0
    for b in val_batches():
        b = b.to(device)
        enc = m.encode(b.tokens, b.mask)
        v = enc.rc_valid
        posts.append(m.sample_rc(enc)[v].float())
        mus.append(enc.rc_mu[v].float())
        sigmas.append(torch.exp(0.5 * enc.rc_logvar[v]).float())
        sv = enc.block_valid.clone(); sv[:, 0] = False
        states.append(enc.block_feats[sv].float())
        got += int(v.sum())
        if got >= rc_prior.shape[0]:
            break
    post = torch.cat(posts)[: rc_prior.shape[0]]
    mu = torch.cat(mus)[: rc_prior.shape[0]]
    sigma = torch.cat(sigmas)[: rc_prior.shape[0]]
    s_real = torch.cat(states)[: s_prior.shape[0]]
    def line(name, x):
        print(f"   {name:18s} mean|x| {x.abs().mean():.3f} | per-dim std {x.std(dim=0).mean():.3f} "
              f"| mean ||x|| {x.norm(dim=-1).mean():.2f}", flush=True)
    print("code stats (prior samples vs posterior samples on real docs):", flush=True)
    line("prior r_hat", rc_prior)
    line("posterior sample", post)
    line("posterior mean", mu)
    print(f"   posterior sigma    mean {sigma.mean():.3f} (per-dim noise the decoder was trained with)", flush=True)
    print("state stats (same check on s, for reference):", flush=True)
    line("prior s_hat", s_prior)
    line("real s", s_real)


def _seg_slices(K: int, n_seg: int = 4) -> list[slice]:
    edges = [round(i * K / n_seg) for i in range(n_seg + 1)]
    return [slice(edges[i], edges[i + 1]) for i in range(n_seg)]


@torch.no_grad()
def latent_collapse_stats(z: torch.Tensor, ref_mean: torch.Tensor, name: str,
                          hi: float = 0.9) -> None:
    """Plan-collapse readout: per segment
    of block indices, the de-meaned norm (dispersion), adjacent-block cosine
    (does block k+1 copy block k?), the share of adjacent pairs above ``hi``
    (near duplicates) and the same-index cross-sample cosine (are samples
    alike?). Compare a prior's samples against real encodings of the same K.

    z: [N, K, d] latents in raw space; ref_mean: [d] latent mean to de-mean with
    (use the REAL latents' mean for both so the two are comparable).
    """
    N, K, _ = z.shape
    zc = (z.float() - ref_mean.float()).reshape(N, K, -1)
    norm = zc.norm(dim=-1)                                              # [N, K]
    zn = torch.nn.functional.normalize(zc, dim=-1)
    adj = (zn[:, :-1] * zn[:, 1:]).sum(-1)                              # [N, K-1]
    # same block index, different samples: mean over off-diagonal pairs
    cross = torch.einsum("nkd,mkd->knm", zn, zn)                        # [K, N, N]
    eye = torch.eye(N, dtype=torch.bool, device=z.device)
    cross = cross.masked_fill(eye[None], 0.0).sum((1, 2)) / (N * (N - 1))   # [K]
    segs = _seg_slices(K)
    cells = []
    for s in segs:
        a = adj[:, s.start:max(s.start, s.stop - 1)]
        cells.append(f"norm {norm[:, s].mean():.2f} adj {a.mean():.3f} "
                     f"dup {(a > hi).float().mean():.3f} cross {cross[s].mean():.3f}")
    print(f"   {name:14s} | " + " | ".join(f"seg{i}: {c}" for i, c in enumerate(cells)), flush=True)


def text_segment_stats(docs: list[list[int]], K: int, S: int, name: str) -> None:
    """Per-quarter distinct-token ratio + loop8 coverage, and two content proxies
    over the whole corpus (share of tokens taken by the 20 most frequent
    trigrams; share of capitalised or numeric words as a cheap entity proxy)."""
    from collections import Counter

    segs = _seg_slices(K)
    dist, loop = [], []
    for s in segs:
        lo, hi_ = s.start * S, s.stop * S
        toks = [d[lo:hi_] for d in docs if len(d) > lo]
        n = sum(len(t) for t in toks)
        uniq = sum(len(set(t)) for t in toks)
        dist.append(uniq / max(n, 1))
        cov = 0
        for t in toks:
            grams = Counter(tuple(t[i:i + 8]) for i in range(len(t) - 7))
            rep = sum(c for c in grams.values() if c >= 3)
            cov += rep * 8
        loop.append(cov / max(n, 1))
    tri = Counter()
    for d in docs:
        tri.update(tuple(d[i:i + 3]) for i in range(len(d) - 2))
    total_tri = sum(tri.values())
    top20 = sum(c for _, c in tri.most_common(20)) / max(total_tri, 1)
    print(f"   {name:14s} | distinct/seg " + " ".join(f"{x:.3f}" for x in dist)
          + " | loop8 cov/seg " + " ".join(f"{x:.3f}" for x in loop)
          + f" | top20-trigram share {top20:.4f}", flush=True)


def parse_reenc(spec: str) -> list[tuple[list[int], float]]:
    out = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        steps_s, level_s = item.split(":")
        out.append(([int(x) for x in steps_s.split("-")], float(level_s)))
    return out


def main() -> None:
    args = parse_args()
    cfg = build_cfg(args)
    set_seed(args.seed)
    device = get_device(args.device)
    torch.set_float32_matmul_precision("high")
    tok = load_tokenizer(cfg)
    m = PSF(cfg).to(device).eval()
    load_checkpoint(args.ckpt, m, device)
    if args.stitch_ckpt:
        m.attach_stitch_decoder(args.stitch_ckpt)
    K, S, L = cfg.num_blocks, cfg.block_span, cfg.max_seq_len
    pad, eos, bob = cfg.data.pad_token_id, cfg.data.eos_token_id, cfg.data.bob_token_id
    judge = Judge(args.judge, device, tok, K, S, args.judge_batch)
    temps = [float(t) for t in args.temps.split(",")]
    recon_levels = [float(x) for x in args.recon_levels.split(",")]
    modes = [x.strip() for x in args.modes.split(",") if x.strip()]
    print(f"# ckpt={args.ckpt} num={args.num} temps={temps} z_level={args.z_level} "
          f"top_p={args.top_p} sampling_steps={args.sampling_steps}", flush=True)

    val_loader = None
    def val_batches():
        nonlocal val_loader
        if val_loader is None:
            _, val_loader = make_dataloaders(cfg)
        return val_loader

    def report(label: str, docs: list[list[int]], extra: dict | None = None) -> None:
        t0 = time.time()
        r = judge.run(docs)
        r["mode"] = label
        r.update(extra or {})
        b = r["buckets"]
        print(f"{label}: overall {r['overall']:.2f} | first {r['first']:.1f} rest {r['rest']:.2f}"
              f" | pos0 {b['p0']:.1f} pos1-3 {b['p1-3']:.2f} pos4-7 {b['p4-7']:.2f} "
              f"pos8-15 {b['p8-15']:.2f} | ent gpt2/sample {r['gpt2_entropy_sample']:.3f} "
              f"t5/sample {r['t5_entropy_sample']:.3f} | len {r['mean_len']:.0f} "
              f"| n={r['n_docs']} ({time.time() - t0:.0f}s judge)", flush=True)
        print("   per-pos (blocks>=2): " + " ".join(f"{v:6.1f}" for v in r["per_pos"]), flush=True)
        print("   per-pos (block 1):   " + " ".join(f"{v:6.1f}" for v in r["per_pos_block1"]), flush=True)
        pb = r["per_block"]
        for row in range(0, K, 16):
            print("   k=%2d..%2d: " % (row + 1, row + 16)
                  + " ".join(f"{v:6.0f}" for v in pb[row:row + 16]), flush=True)
        print(degeneration_line(docs, prefix="   "), flush=True)
        if args.out:
            with open(args.out, "a", encoding="utf-8") as f:
                f.write(json.dumps({"ckpt": args.ckpt, **r}) + "\n")

    @torch.no_grad()
    def real_or_recon(mode: str, temperature: float, level: float) -> list[list[int]]:
        set_seed(args.seed)
        docs = []
        for b in val_batches():
            b = b.to(device)
            if mode == "real":
                toks, lens = b.tokens, b.mask.sum(1)
            else:
                enc = m.encode(b.tokens, b.mask)
                bt = m.decoder.generate(enc.block_feats, bob, temperature, args.top_p,
                                        z_level=level, rc=m.sample_rc(enc, args.rc_source))
                toks, lens = m.truncate_at_eos(bt)
            docs.extend(to_docs(toks, lens))
            if len(docs) >= args.num:
                return docs[: args.num]
        return docs

    @torch.no_grad()
    def recon_forced(temperature: float, level: float, start_n: int, end_n: int) -> list[list[int]]:
        """Seam oracle: parallel recon from REAL states, but the first ``start_n``
        and/or last ``end_n`` positions of every block are teacher-forced to the
        real tokens. Bounds what a perfect boundary stitch could recover: the
        forced positions cost what real words cost in generated context, the
        free positions show how much the body benefits from consistent seams."""
        set_seed(args.seed)
        docs = []
        for b in val_batches():
            b = b.to(device)
            enc = m.encode(b.tokens, b.mask)
            _, targets = m.block_view(b.tokens)                 # [B, K, S] real block tokens
            force = torch.full_like(targets, -1)
            if start_n > 0:
                force[:, :, :start_n] = targets[:, :, :start_n]
            if end_n > 0:
                force[:, :, S - end_n:] = targets[:, :, S - end_n:]
            force = torch.where(targets == pad, torch.full_like(force, -1), force)
            bt = m.decoder.generate(enc.block_feats, bob, temperature, args.top_p,
                                    z_level=level, rc=m.sample_rc(enc, args.rc_source),
                                    force=force)
            toks, lens = m.truncate_at_eos(bt)
            docs.extend(to_docs(toks, lens))
            if len(docs) >= args.num:
                return docs[: args.num]
        return docs

    @torch.no_grad()
    def sampled(fn) -> list[list[int]]:
        set_seed(args.seed)
        docs = []
        while len(docs) < args.num:
            n = min(args.gen_batch, args.num - len(docs))
            toks, lens = fn(n)
            docs.extend(to_docs(toks, lens))
        return docs[: args.num]

    for mode in modes:
        if mode == "real":
            report("real", real_or_recon("real", 0.0, 1.0))
            continue
        if mode == "codestats":
            set_seed(args.seed)
            code_stats(m, args.gen_batch * 4, device, val_batches)
            continue
        if mode == "planstats":
            # Latent-side collapse readout: prior samples vs real encodings, s and r.
            set_seed(args.seed)
            n = min(args.num, 256)
            with torch.no_grad():
                z = flow_sample(m.denoiser, n, K, m.prior_dim, cfg.flow, device)
                s_hat, rc_hat = m.split_latents(m.denoiser.denormalize(z))
                reals, rcs, texts_real = [], [], []
                got = 0
                for b in val_batches():
                    b = b.to(device)
                    enc = m.encode(b.tokens, b.mask)
                    reals.append(enc.block_feats)
                    if rc_hat is not None:
                        rcs.append(m.sample_rc(enc, "mean"))
                    texts_real.extend(to_docs(b.tokens, b.mask.sum(1)))
                    got += b.tokens.shape[0]
                    if got >= n:
                        break
                s_real = torch.cat(reals)[:n]
                mean_s = s_real[:, 1:].reshape(-1, s_real.shape[-1]).mean(0)   # skip the BOS slot
                print("plan-collapse stats (de-meaned with the REAL latent mean; K split in 4 segments):", flush=True)
                latent_collapse_stats(s_real[:, 1:], mean_s, "real s")
                latent_collapse_stats(s_hat[:, 1:], mean_s, "prior s_hat")
                if rc_hat is not None:
                    r_real = torch.cat(rcs)[:n]
                    mean_r = r_real.reshape(-1, r_real.shape[-1]).mean(0)
                    latent_collapse_stats(r_real, mean_r, "real r (mean)")
                    latent_collapse_stats(rc_hat, mean_r, "prior r_hat")
                print("text segment stats (real held-out vs one-shot z0 at each temperature):", flush=True)
                text_segment_stats(texts_real[:n], K, S, "real")
                for temperature in temps:
                    tp = args.top_p if temperature > 0 else 1.0
                    docs = sampled(lambda nn: m.generate(nn, temperature=temperature, top_p=tp,
                                                         z_level=args.z_level))
                    text_segment_stats(docs, K, S, f"one-shot t={temperature:.2f}")
            continue
        if mode == "bank":
            if not args.bank:
                raise SystemExit("--bank required for bank/bankrecon")
            toks, mask, lens = load_bank(args.bank, args.num, L, pad, device)
            report("bank(rollout text)", to_docs(toks.cpu(), lens))
            continue
        for temperature in temps:
            tp = args.top_p if temperature > 0 else 1.0
            if mode == "recon":
                for lv in recon_levels:
                    tag = "" if args.rc_source == "sample" else f" rc={args.rc_source}"
                    report(f"recon t={temperature:.2f} z{lv:g}{tag}",
                           real_or_recon("recon", temperature, lv),
                           {"temperature": temperature, "z_level": lv,
                            "rc_source": args.rc_source})
            elif mode == "stitch":
                if not args.stitch:
                    raise SystemExit("--stitch required for mode stitch")
                docs = sampled(lambda n: m.generate_stitched(
                    n, temperature=temperature, top_p=tp, z_level=args.z_level,
                    start_n=args.stitch_start, end_n=args.stitch_end,
                    prev_n=args.stitch_prev, next_n=args.stitch_next,
                    redecode=args.stitch_redecode))
                report(f"stitch[s{args.stitch_start} e{args.stitch_end} r{args.stitch_redecode}] "
                       f"t={temperature:.2f} z{args.z_level:g}",
                       docs, {"temperature": temperature, "z_level": args.z_level,
                              "stitch_start": args.stitch_start, "stitch_end": args.stitch_end,
                              "stitch_redecode": args.stitch_redecode})
            elif mode == "seamoracle":
                lv = recon_levels[0]
                for name, sn, en in [("start", args.seam_start, 0),
                                     ("end", 0, args.seam_end),
                                     ("both", args.seam_start, args.seam_end)]:
                    report(f"seamoracle[{name} s{sn} e{en}] t={temperature:.2f} z{lv:g}",
                           recon_forced(temperature, lv, sn, en),
                           {"temperature": temperature, "z_level": lv,
                            "seam_start": sn, "seam_end": en})
            elif mode == "bankrecon":
                if not args.bank:
                    raise SystemExit("--bank required for bank/bankrecon")
                toks, mask, lens = load_bank(args.bank, args.num, L, pad, device)
                for lv in recon_levels:
                    set_seed(args.seed)
                    docs = []
                    for s in range(0, toks.shape[0], args.gen_batch):
                        enc = m.encode(toks[s:s + args.gen_batch], mask[s:s + args.gen_batch])
                        bt = m.decoder.generate(enc.block_feats, bob, temperature, tp,
                                                z_level=lv, rc=m.sample_rc(enc, args.rc_source))
                        o, ln = m.truncate_at_eos(bt)
                        docs.extend(to_docs(o, ln))
                    report(f"bankrecon t={temperature:.2f} z{lv:g}", docs,
                           {"temperature": temperature, "z_level": lv})
            elif mode == "parallel":
                docs = sampled(lambda n: m.generate(n, temperature=temperature, top_p=tp,
                                                    z_level=args.z_level))
                report(f"parallel t={temperature:.2f} z{args.z_level:g}", docs,
                       {"temperature": temperature, "z_level": args.z_level})
            elif mode == "rollout":
                docs = sampled(lambda n: m.rollout_generate(n, temperature=temperature, top_p=tp))
                report(f"rollout t={temperature:.2f}", docs, {"temperature": temperature})
            elif mode == "reenc":
                for steps, lv in parse_reenc(args.reenc):
                    docs = sampled(lambda n: generate_reencode(
                        m, n, temperature, tp, args.z_level, steps, lv, device))
                    report(f"reenc{steps} t={temperature:.2f} L{lv:g}", docs,
                           {"temperature": temperature, "reenc_steps": steps,
                            "reenc_level": lv})
            elif mode == "nolanding":
                docs = sampled(lambda n: generate_reencode(
                    m, n, temperature, tp, args.z_level, [], args.z_level, device,
                    use_landing=False))
                report(f"nolanding t={temperature:.2f} z{args.z_level:g}", docs,
                       {"temperature": temperature, "z_level": args.z_level,
                        "landing": False})
            elif mode.startswith("codemod:"):
                spec = mode.split(":", 1)[1]
                docs = sampled(lambda n: generate_codemod(
                    m, n, temperature, tp, args.z_level, spec, device))
                report(f"codemod[{spec}] t={temperature:.2f} z{args.z_level:g}", docs,
                       {"temperature": temperature, "z_level": args.z_level,
                        "codemod": spec})
            else:
                raise SystemExit(f"unknown mode {mode}")


if __name__ == "__main__":
    main()
