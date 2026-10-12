"""Evaluate unconditional generation: GPT-2 PPL, unigram entropy, and FLOPs.

    python -m scripts.eval_gen --ckpt runs/prior/prior.pt --num-samples 1000 --reference

Follows the MDLM/ELF protocol: generate N documents, score them with GPT-2
Large (generative perplexity = exp of the judge's mean NLL per token), and
measure diversity as the entropy of the empirical unigram distribution.
``--reference`` also scores held-out test documents (passed through the same
T5 detokenizer) as a floor. FLOPs come from torch's FlopCounterMode.

Each result also gets a second line of degeneration stats measured on the
first 256 T5 tokens of every sample: distinct-token ratio, repeated-n-gram
fractions, and how much text sits inside a hard loop (an 8-gram that recurs
at least 3 times).
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from typing import Optional

import torch
import torch.nn.functional as F

from psf.config import Config
from psf.models.psf import PSF, refine_schedule
from psf.utils.data import decode, load_tokenizer, make_dataloaders
from psf.utils.gen_metrics import repetition_metrics
from psf.utils.misc import get_device, load_checkpoint, set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PSF unconditional generation eval")
    p.add_argument("--ckpt", default=None, help="stage-2 checkpoint (holds AE + prior)")
    p.add_argument("--texts", default=None,
                   help="JSONL of pre-generated texts to score instead of sampling; "
                        "uses the 'text' field (or the first string field) per line")
    p.add_argument("--encoder", default="block", choices=["block", "prefix"],
                   help="must match the checkpoint")
    p.add_argument("--max-seq-len", type=int, default=None,
                   help="L; must match the checkpoint")
    p.add_argument("--num-blocks", type=int, default=None,
                   help="K; must match the checkpoint")
    p.add_argument("--latent-dim", type=int, default=None,
                   help="D; must match the checkpoint")
    p.add_argument("--latent-slots", type=int, default=None,
                   help="m latent vectors per block; must match the checkpoint")
    p.add_argument("--decoder-depth", type=int, default=None,
                   help="decoder layers; must match the checkpoint")
    p.add_argument("--prev-window", action="store_true",
                   help="must match the checkpoint")
    p.add_argument("--wave-lead", type=int, default=None,
                   help="wavefront decode lead (see train_ae --wave-lead); should match "
                        "the checkpoint's training lead, 0 forces the plain decode")
    p.add_argument("--lex-bag", action="store_true", help="must match the checkpoint (trained lexical bag)")
    p.add_argument("--denoiser-hidden", type=int, default=None,
                   help="must match the checkpoint")
    p.add_argument("--denoiser-depth", type=int, default=None,
                   help="must match the checkpoint")
    p.add_argument("--denoiser-heads", type=int, default=None,
                   help="must match the checkpoint")
    p.add_argument("--denoiser-bottleneck", type=int, default=None,
                   help="must match the checkpoint")
    p.add_argument("--prefix-unshifted", action="store_true", help="must match the checkpoint (content-latent control)")
    p.add_argument("--causal-latents", action="store_true",
                   help="must match the checkpoint")
    p.add_argument("--null-latent", action="store_true",
                   help="must match the checkpoint")
    p.add_argument("--latent-window", type=int, default=0,
                   help="must match the checkpoint")
    p.add_argument("--global-latents", type=int, default=0,
                   help="must match the checkpoint")
    p.add_argument("--t5-features", action="store_true",
                   help="must match the checkpoint")
    p.add_argument("--boundary-tokens", action="store_true",
                   help="must match the checkpoint")
    p.add_argument("--boundary-window", type=int, default=1,
                   help="must match the checkpoint")
    p.add_argument("--landing-state", action="store_true",
                   help="must match the checkpoint")
    p.add_argument("--landing-dim", type=int, default=0,
                   help="must match the checkpoint")
    p.add_argument("--landing-attn", action="store_true",
                   help="must match the checkpoint")
    p.add_argument("--landing-residual", action="store_true",
                   help="must match the checkpoint")
    p.add_argument("--landing-cfg", action="store_true",
                   help="must match the checkpoint")
    p.add_argument("--landing-lambda", type=float, default=1.0,
                   help="landing CFG strength at decode (needs --landing-cfg; "
                        "0 = no landing, 1 = plain landing, >1 extrapolates; "
                        "!=1 doubles decode cost)")
    p.add_argument("--level-cond", action="store_true",
                   help="must match the checkpoint")
    p.add_argument("--cross-before-self", action="store_true",
                   help="must match the checkpoint")
    p.add_argument("--abs-pos-emb", action="store_true",
                   help="must match the checkpoint")
    p.add_argument("--rc-dim", type=int, default=None,
                   help="realization-code dims; must match the checkpoint")
    p.add_argument("--no-rc-memory", action="store_true",
                   help="must match the checkpoint")
    p.add_argument("--rc-lookahead", action="store_true",
                   help="block k also reads the next block's code r_{k+1} (must match the checkpoint)")
    p.add_argument("--lex-bonus", type=float, default=0.0,
                   help="decode-time cross-block lexical coupling: logit bonus for content tokens "
                        "(id >= --lex-min-id) already used by another block (0 = off)")
    p.add_argument("--lex-min-id", type=int, default=2000)
    p.add_argument("--lex-post", action="store_true",
                   help="apply --lex-bonus after the nucleus cut (re-rank inside the top-p set only)")
    p.add_argument("--lex-gate", action="store_true",
                   help="checkpoint has the learned cross-block lexical bonus (decoder.gate_proj/lex_scale)")
    p.add_argument("--lex-gate-mult", type=float, default=1.0,
                   help="inference multiplier on the learned lexical bonus (1 = as trained)")
    p.add_argument("--no-repeat-ngram", type=int, default=0,
                   help="inference-time only: ban any token that would complete an n-gram already "
                        "present anywhere in the document (all blocks); 0 = off")
    p.add_argument("--landing-off", action="store_true",
                   help="inference-time ablation: never feed the landing target (no_target vector everywhere)")
    p.add_argument("--rc-mode", default="plan",
                   help="inference-time ablation of the realization codes: plan (prior's codes) / random (fresh "
                        "N(0,I)) / zero / shuffle (prior's codes permuted across documents) / scale<f> (codes x f)")
    p.add_argument("--rc-lookahead-off", action="store_true",
                   help="inference-time ablation: zero the next-block code channel r_{k+1}")
    p.add_argument("--prompt-attn", action="store_true",
                   help="checkpoint has the prompt cross-attention channel (must match; unconditional "
                        "generation feeds no prompt)")
    p.add_argument("--shared-noise", action="store_true",
                   help="inference-time only: the K blocks reuse the same "
                        "vocabulary-indexed Gumbel noise per decode step "
                        "instead of each drawing independent randomness "
                        "(correlates which word each block reaches for)")
    p.add_argument("--z-level", type=float, default=1.0,
                   help="noise level told to the decoder for the prior's "
                        "latents (needs --level-cond; 1 = trust fully)")
    p.add_argument("--sc-cfg", action="store_true",
                   help="checkpoint has SC-CFG guidance tokens (must match)")
    p.add_argument("--cfg-scale", type=float, default=1.0,
                   help="guidance scale at sampling (needs --sc-cfg ckpt)")
    p.add_argument("--sde-gamma", type=float, default=0.0,
                   help="per-step SDE churn; 0 = pure ODE")
    p.add_argument("--noise-scale", type=float, default=1.0,
                   help="initial/churn noise stddev (truncation trick when < 1)")
    p.add_argument("--df-stagger", type=float, default=0.0,
                   help="staggered (pyramid) reveal strength lambda: block k runs on "
                        "clock clip(t(1+lambda) - lambda*off_k); 0 = shared time")
    p.add_argument("--df-order", default="random", choices=["random", "ltr"],
                   help="with --df-stagger: block reveal order")
    p.add_argument("--stitch", action="store_true",
                   help="checkpoint has the seam-stitch mode (must match) AND run the "
                        "stitch pass after the one-shot decode (start_n + end_n extra steps)")
    p.add_argument("--stitch-start", type=int, default=3)
    p.add_argument("--stitch-end", type=int, default=2)
    p.add_argument("--stitch-prev", type=int, default=3)
    p.add_argument("--stitch-next", type=int, default=3)
    p.add_argument("--stitch-temp", type=float, default=None,
                   help="temperature of the stitch pass (default: the decode temperature)")
    p.add_argument("--stitch-ckpt", default=None,
                   help="two-decoder inference: take the stitch-mode decoder from this checkpoint "
                        "and keep --ckpt's own decoder for the one-shot pass")
    p.add_argument("--stitch-redecode", type=int, default=0,
                   help="after the stitch pass, the main decoder re-decodes this many positions "
                        "right after each rewritten opening (smooths the new junction)")
    p.add_argument("--prior-fwd-mult", type=float, default=3.5,
                   help="denoiser fwds per train step for the FLOPs report "
                        "(3.5 baseline, 5.0 with SC-CFG)")
    p.add_argument("--num-samples", type=int, default=1000)
    p.add_argument("--gen-batch", type=int, default=125)
    p.add_argument("--settings", default="0.0:1.0,0.8:0.95,1.0:1.0",
                   help="comma list of temperature:top_p decoder settings")
    p.add_argument("--save-texts", default=None,
                   help="path prefix: dump each setting's generated texts to "
                        "<prefix>_nfe{steps}_t{temp}_p{top_p}.jsonl ('text' field)")
    p.add_argument("--sampling-steps", default="50",
                   help="comma list of flow Euler step counts to sweep")
    p.add_argument("--judge", default="openai-community/gpt2-large")
    p.add_argument("--judge-batch", type=int, default=16)
    p.add_argument("--judge-window", type=int, default=1024,
                   help="--texts scoring only: judge tokens per document (fixed-window protocol)")
    p.add_argument("--judge-drop-short", action="store_true",
                   help="--texts scoring only: keep only documents with at least --judge-window judge tokens")
    p.add_argument("--reference", action="store_true",
                   help="also score held-out test docs as a floor")
    p.add_argument("--rollout", action="store_true",
                   help="block-AR rollout generation (decode -> re-encode -> "
                        "decode; the prior never enters)")
    p.add_argument("--rollout-prior-s1", action="store_true",
                   help="with --rollout: draw S_1 from the flow prior (a "
                        "marginal single-block sample) instead of the fixed "
                        "learned BOS latent, so every document gets a "
                        "distinct topic seed; the rest of rollout is unchanged")
    p.add_argument("--rollout-gt", action="store_true",
                   help="with --rollout: source every S_k from real held-out "
                        "documents (same states as --recon) but decode them "
                        "through rollout's SEQUENTIAL block-AR loop instead "
                        "of recon's parallel pass -- isolates decode "
                        "scheduling from state quality. Mutually exclusive "
                        "with --rollout-prior-s1/--rollout-seed-block1; "
                        "ignores --num-samples/--gen-batch (iterates the val "
                        "set like --recon)")
    p.add_argument("--rollout-seed-block1", action="store_true",
                   help="with --rollout: seed block 1 with a REAL held-out "
                        "document's verbatim text (never sampled), so S_2 "
                        "is grounded in genuine content; every block from 2 "
                        "on is plain self-referential rollout (sampled, "
                        "re-encoded). Isolates whether a real, concrete "
                        "opening changes the model's own subsequent "
                        "writing. Mutually exclusive with --rollout-gt/"
                        "--rollout-prior-s1; ignores --num-samples/"
                        "--gen-batch (iterates the val set like --recon)")
    p.add_argument("--refine-rounds", default="0",
                   help="comma list of latent-refinement rounds to sweep; each "
                        "round re-encodes the decoded text and decodes it again "
                        "(0 = plain one-shot parallel decode)")
    p.add_argument("--refine-anneal", action="store_true",
                   help="cool the decoder temperature linearly to 0 across the "
                        "refinement rounds so the iteration can settle")
    p.add_argument("--refine-temps", default=None,
                   help="explicit per-pass temperature schedules to sweep, "
                        "commas within a schedule and ';' between them "
                        "(e.g. '0.8,0.6;0.8,0.4'); overrides --refine-rounds. "
                        "Lets the last pass land on a chosen entropy, which is "
                        "the only way to compare PPL against the R=0 curve.")
    p.add_argument("--refine-commit", default=None,
                   help="comma list of per-round block re-decode probabilities "
                        "(Mask-Predict partial commit); the last entry repeats. "
                        "Default: rewrite every block every round.")
    p.add_argument("--refine-cands", type=int, default=1,
                   help=">1: each refine pass decodes this many candidates per "
                        "block and keeps the one the decoder itself scores "
                        "highest (best-of-n; multiplies decode cost)")
    p.add_argument("--refine-order", default="random",
                   choices=["random", "ltr", "conf"],
                   help="which blocks accept their rewrite each round: random "
                        "i.i.d. (default), ltr = left-to-right wavefront "
                        "(block-AR in R waves; ignores --refine-commit), conf "
                        "= rewrite the --refine-commit fraction with the "
                        "lowest decoder self-score")
    p.add_argument("--recon", action="store_true",
                   help="score AE reconstructions of held-out docs (true latents "
                        "+ decoder sampling) instead of generating; isolates the "
                        "decoder ceiling from the prior")
    p.add_argument("--ae-steps", type=int, default=50_000, help="for the train-FLOPs report")
    p.add_argument("--prior-steps", type=int, default=100_000, help="for the train-FLOPs report")
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


@torch.no_grad()
def judge_ppl(texts: list[str], judge, judge_tok, device, batch_size: int,
              window: int = 1024, drop_short: bool = False) -> tuple[float, int, float, float]:
    """Corpus-level exp(mean NLL) under the judge, over the first ``window`` tokens each (1024 = the paper's protocol).

    ``drop_short``: score only documents with at least ``window`` judge tokens (length-controlled variant).
    Returns (ppl, scored_tokens, corpus_entropy, mean_sample_entropy) in
    judge-token space; the last matches ELF's per-sample "mean_entropy".
    """
    texts = [t for t in texts if t.strip()]
    if drop_short:
        texts = [t for t in texts if len(judge_tok(t, truncation=True, max_length=window + 1).input_ids) >= window]
    total_nll = 0.0
    total_tok = 0
    rows: list[list[int]] = []
    for i in range(0, len(texts), batch_size):
        enc = judge_tok(texts[i:i + batch_size], return_tensors="pt", padding=True,
                        truncation=True, max_length=window)
        ids = enc.input_ids.to(device)
        mask = enc.attention_mask.to(device)
        logits = judge(input_ids=ids, attention_mask=mask).logits.float()
        logp = F.log_softmax(logits[:, :-1], dim=-1)
        nll = -logp.gather(-1, ids[:, 1:, None]).squeeze(-1)
        m = mask[:, 1:].float()
        total_nll += float((nll * m).sum())
        total_tok += int(m.sum())
        for row, row_mask in zip(ids.tolist(), mask.tolist()):
            rows.append([t for t, keep in zip(row, row_mask) if keep])
    ppl = math.exp(total_nll / max(total_tok, 1))
    return ppl, total_tok, unigram_entropy(rows), mean_sample_entropy(rows)


def _entropy(counts: Counter) -> float:
    n = sum(counts.values())
    return -sum(c / n * math.log(c / n) for c in counts.values()) if n else 0.0


def unigram_entropy(token_lists: list[list[int]]) -> float:
    """Entropy (nats) of the pooled (corpus-level) unigram distribution."""
    counts = Counter()
    for ids in token_lists:
        counts.update(ids)
    return _entropy(counts)


def mean_sample_entropy(token_lists: list[list[int]]) -> float:
    """Mean over samples of each sample's own unigram entropy (ELF's definition)."""
    ents = [_entropy(Counter(ids)) for ids in token_lists if ids]
    return sum(ents) / max(len(ents), 1)


def degeneration_line(id_lists: list[list[int]], prefix: str = " " * 27) -> str:
    """Repetition / loop stats over the first 256 T5 tokens of each sample."""
    m = repetition_metrics(id_lists)
    return (f"{prefix}distinct={m['distinct']:.3f} | rep2={m['rep2']:.3f} "
            f"rep4={m['rep4']:.3f} rep8={m['rep8']:.3f} | loop8 coverage="
            f"{m['loop8_coverage']:.3f} samples={m['loop8_sample_rate']:.3f}")


def schedules_for(args, temperature: float, top_p: float,
                  rounds_list: list[int]) -> list[tuple[str, list[tuple[float, float]]]]:
    """(label, decode schedule) pairs to sweep for one temperature setting."""
    if args.refine_temps:
        out = []
        for spec in args.refine_temps.split(";"):
            temps = [float(x) for x in spec.split(",")]
            out.append((spec, [(t, top_p if t > 0.0 else 1.0) for t in temps]))
        return out
    tag = "a" if args.refine_anneal else " "
    return [(f"R={r}{tag}", refine_schedule(temperature, top_p, r, args.refine_anneal))
            for r in rounds_list]


def commit_list(args) -> Optional[list[float]]:
    """Per-round block re-decode probabilities, or None to rewrite everything."""
    if not args.refine_commit:
        return None
    return [float(x) for x in args.refine_commit.split(",")]


@torch.no_grad()
def generate_corpus(model: PSF, tokenizer, args, device,
                    schedule) -> tuple[list[str], list[list[int]], float]:
    """Sample documents; returns (texts, t5 id lists, mean length)."""
    texts, id_lists, lengths = [], [], []
    remaining = args.num_samples
    while remaining > 0:
        n = min(args.gen_batch, remaining)
        if args.stitch:
            temperature, top_p = schedule[0]
            tokens, lens = model.generate_stitched(
                n, device=device, temperature=temperature, top_p=top_p,
                cfg_scale=args.cfg_scale, sde_gamma=args.sde_gamma,
                noise_scale=args.noise_scale, z_level=args.z_level,
                start_n=args.stitch_start, end_n=args.stitch_end,
                prev_n=args.stitch_prev, next_n=args.stitch_next,
                stitch_temperature=args.stitch_temp, redecode=args.stitch_redecode,
            )
        else:
            tokens, lens = model.generate(
                n, device=device, cfg_scale=args.cfg_scale, sde_gamma=args.sde_gamma,
                noise_scale=args.noise_scale, schedule=schedule,
                commit=commit_list(args), order=args.refine_order,
                z_level=args.z_level, landing_lambda=args.landing_lambda,
                n_cands=args.refine_cands, shared_noise=args.shared_noise,
                lex_bonus=args.lex_bonus, lex_min_id=args.lex_min_id, lex_post=args.lex_post,
                no_repeat_ngram=args.no_repeat_ngram, landing_off=args.landing_off,
                rc_mode=args.rc_mode, rc_lookahead_off=args.rc_lookahead_off,
            )
        for row, length in zip(tokens, lens):
            ids = row[: int(length)].tolist()
            id_lists.append(ids)
            texts.append(decode(tokenizer, ids))
            lengths.append(int(length))
        remaining -= n
    return texts, id_lists, sum(lengths) / max(len(lengths), 1)


@torch.no_grad()
def component_flops(model: PSF, cfg: Config, device) -> dict[str, int]:
    """Forward FLOPs of each component at batch size 1, via FlopCounterMode."""
    from torch.utils.flop_counter import FlopCounterMode

    tokens = torch.randint(0, cfg.data.vocab_size, (1, cfg.max_seq_len), device=device)
    n_lat = cfg.num_blocks * cfg.num_latent_slots + cfg.num_global_latents
    z = torch.randn(1, n_lat, cfg.latent_dim, device=device)
    zp = torch.randn(1, n_lat, model.prior_dim, device=device)
    rc = torch.randn(1, cfg.num_blocks, cfg.rc_dim, device=device) if cfg.rc_dim > 0 else None
    t = torch.rand(1, device=device)
    inputs, _ = model.block_view(tokens)

    def measure(fn) -> int:
        counter = FlopCounterMode(display=False)
        with counter:
            fn()
        return counter.get_total_flops()

    return {
        "encoder_fwd": measure(lambda: model.encode(tokens)),
        "decoder_tf_fwd": measure(lambda: model.decoder(z, inputs, rc=rc)),
        "decoder_generate": measure(
            lambda: model.decoder.generate(z, cfg.data.bob_token_id, rc=rc)
        ),
        "denoiser_fwd": measure(lambda: model.denoiser(zp, t)),
    }


def report_flops(f: dict[str, int], cfg: Config, ae_steps: int, prior_steps: int,
                 global_batch: int = 512, prior_fwd_mult: float = 3.5) -> None:
    L = cfg.max_seq_len
    T = lambda x: f"{x / 1e12:.3f} TFLOPs"
    nfe = cfg.flow.num_sampling_steps
    sample = nfe * f["denoiser_fwd"] + f["decoder_generate"]
    refine = f["encoder_fwd"] + f["decoder_generate"]
    ae_step_seq = 3 * (f["encoder_fwd"] + f["decoder_tf_fwd"])          # fwd+bwd ~ 3x fwd
    prior_step_seq = f["encoder_fwd"] + prior_fwd_mult * f["denoiser_fwd"]
    ae_total = ae_step_seq * global_batch * ae_steps
    prior_total = prior_step_seq * global_batch * prior_steps

    print("\n--- FLOPs (measured forward, batch=1) ---")
    print(f"  encoder fwd            {T(f['encoder_fwd'])}")
    print(f"  decoder fwd (teacher)  {T(f['decoder_tf_fwd'])}")
    print(f"  decoder generate       {T(f['decoder_generate'])}   ({cfg.block_span} steps, no KV cache)")
    print(f"  denoiser fwd (1 NFE)   {T(f['denoiser_fwd'])}")
    print(f"  sample 1 doc ({L} tok) {T(sample)}   = {nfe} NFE + decode "
          f"({sample / L / 1e9:.3f} GFLOPs/token)")
    print(f"  + 1 refine round       {T(refine)}   = re-encode + decode "
          f"(+{100 * refine / sample:.0f}% per round)")
    print(f"  stage-1 train total    {ae_total / 1e18:.2f} EFLOPs  "
          f"(3x fwd, batch {global_batch} x {ae_steps})")
    print(f"  stage-2 train total    {prior_total / 1e18:.2f} EFLOPs  "
          f"(frozen enc fwd + {prior_fwd_mult}x denoiser, batch {global_batch} x {prior_steps})")
    print(f"  train total            {(ae_total + prior_total) / 1e18:.2f} EFLOPs")


def score_texts_file(args, device) -> None:
    """Judge a JSONL of pre-generated texts with the same protocol as sampling."""
    import json

    from transformers import AutoModelForCausalLM, AutoTokenizer

    texts = []
    with open(args.texts, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            value = row.get("text") if isinstance(row, dict) else row
            if value is None and isinstance(row, dict):
                value = next((v for v in row.values() if isinstance(v, str)), None)
            if isinstance(value, str):
                texts.append(value)
    texts = texts[: args.num_samples]

    judge_tok = AutoTokenizer.from_pretrained(args.judge)
    judge_tok.pad_token = judge_tok.eos_token
    judge = AutoModelForCausalLM.from_pretrained(args.judge).to(device).eval()
    ppl, n_tok, ent_corpus, ent_sample = judge_ppl(
        texts, judge, judge_tok, device, args.judge_batch,
        window=args.judge_window, drop_short=args.judge_drop_short,
    )
    t5_tok = AutoTokenizer.from_pretrained("google-t5/t5-small")
    t5_ids = [t5_tok(t, add_special_tokens=False).input_ids for t in texts]
    if args.judge_window != 1024 or args.judge_drop_short:
        print(f"  [window={args.judge_window} judge tokens, drop_short={args.judge_drop_short}, "
              f"scored docs={n_tok // max(args.judge_window - 1, 1) if args.judge_drop_short else len(texts)}]")
    print(f"  texts={args.texts} n={len(texts)} | gen PPL={ppl:8.2f} | "
          f"gpt2 entropy corpus={ent_corpus:.3f} /sample={ent_sample:.3f} | "
          f"t5 corpus={unigram_entropy(t5_ids):.3f} "
          f"/sample={mean_sample_entropy(t5_ids):.3f} | scored {n_tok} judge tok")
    print(degeneration_line(t5_ids))


def main() -> None:
    args = parse_args()
    if args.texts:
        set_seed(args.seed)
        device = get_device(args.device)
        torch.set_float32_matmul_precision("high")
        score_texts_file(args, device)
        return
    if not args.ckpt:
        raise SystemExit("either --ckpt or --texts is required")
    cfg, steps_list = config_from_args(args)
    set_seed(args.seed)
    device = get_device(args.device)
    torch.set_float32_matmul_precision("high")

    tokenizer = load_tokenizer(cfg)
    model = PSF(cfg).to(device).eval()
    load_checkpoint(args.ckpt, model, device)
    if args.lex_gate:
        model.decoder.lex_gate_mult = args.lex_gate_mult   # inference-side amplification of the learned bonus
    if args.stitch_ckpt:
        model.attach_stitch_decoder(args.stitch_ckpt)
    print(f"loaded {args.ckpt} | sampling steps sweep={steps_list}")

    report_flops(component_flops(model, cfg, device), cfg, args.ae_steps, args.prior_steps,
                 prior_fwd_mult=args.prior_fwd_mult)
    run_main(args, cfg, steps_list, model, tokenizer, device)


def config_from_args(args) -> tuple[Config, list[int]]:
    """The Config behind a checkpoint's architecture flags (shared with the probes)."""
    cfg = Config()
    if args.max_seq_len is not None:
        cfg.max_seq_len = args.max_seq_len
    if args.num_blocks is not None:
        cfg.num_blocks = args.num_blocks
    if args.latent_dim is not None:
        cfg.latent_dim = args.latent_dim
    if args.latent_slots is not None:
        cfg.num_latent_slots = args.latent_slots
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
    cfg.__post_init__()   # re-check L % K and the RoPE head split
    cfg.encoder_type = args.encoder
    cfg.prefix_unshifted = args.prefix_unshifted
    cfg.decoder.causal_latents = args.causal_latents
    cfg.decoder.null_latent = args.null_latent
    cfg.decoder.latent_window = args.latent_window
    cfg.decoder.boundary_tokens = args.boundary_tokens
    cfg.decoder.boundary_window = args.boundary_window
    cfg.decoder.landing_state = args.landing_state
    cfg.decoder.landing_dim = args.landing_dim
    cfg.decoder.landing_attn = args.landing_attn
    cfg.decoder.landing_residual = args.landing_residual
    cfg.decoder.landing_cfg = args.landing_cfg
    cfg.decoder.noise_level_cond = args.level_cond
    cfg.decoder.cross_before_self = args.cross_before_self
    cfg.decoder.abs_pos_emb = args.abs_pos_emb
    cfg.decoder.prev_window = args.prev_window
    if args.wave_lead is not None:
        cfg.decoder.wave_lead = args.wave_lead
    cfg.decoder.lex_bag = args.lex_bag
    cfg.decoder.lex_gate = args.lex_gate
    cfg.num_global_latents = args.global_latents
    cfg.encoder.t5_features = args.t5_features
    if args.rc_dim is not None:
        cfg.rc_dim = args.rc_dim
    if args.no_rc_memory:
        cfg.decoder.rc_memory = False
    if args.rc_lookahead:
        cfg.decoder.rc_lookahead = True
    if getattr(args, "prompt_attn", False):
        cfg.decoder.prompt_attn = True
    if args.stitch and not args.stitch_ckpt:
        cfg.decoder.stitch = True
    if args.sc_cfg:
        cfg.denoiser.num_cfg_tokens = 4
    steps_list = [int(x) for x in args.sampling_steps.split(",")]
    cfg.flow.num_sampling_steps = steps_list[0]
    cfg.flow.df_stagger = args.df_stagger
    cfg.flow.df_order = args.df_order
    return cfg, steps_list


def run_main(args, cfg: Config, steps_list: list[int], model: PSF, tokenizer, device) -> None:
    """Everything after the model is loaded (sweeps, rollout, recon, refinement)."""

    from transformers import AutoModelForCausalLM, AutoTokenizer

    judge_tok = AutoTokenizer.from_pretrained(args.judge)
    judge_tok.pad_token = judge_tok.eos_token
    judge = AutoModelForCausalLM.from_pretrained(args.judge).to(device).eval()

    settings = [tuple(float(x) for x in s.split(":")) for s in args.settings.split(",")]
    rounds_list = [int(x) for x in args.refine_rounds.split(",")]
    commit = commit_list(args)
    for temperature, top_p in settings:
        for label, sched in schedules_for(args, temperature, top_p, rounds_list):
            if len(sched) > 1:
                tail = "" if commit is None else "  commit=" + ",".join(
                    f"{commit[min(i, len(commit) - 1)]:.2f}"
                    for i in range(len(sched) - 1))
                print(f"  refine {label:14s} " + " -> ".join(
                    f"t={t:.2f}/p={p:.2f}" for t, p in sched) + tail)

    if args.rollout:
        gt_driven = args.rollout_gt or args.rollout_seed_block1
        if args.rollout_gt:
            tag = "S_k = real doc (recon's states, rollout's decode order)"
        elif args.rollout_seed_block1:
            tag = "block 1 = real doc verbatim, block 2+ self-referential"
        elif args.rollout_prior_s1:
            tag = "S_1 ~ prior"
        else:
            tag = "S_1 = BOS latent"
        gt_loader = make_dataloaders(cfg)[1] if gt_driven else None
        Sp = cfg.block_span
        print(f"\n--- block-AR rollout PPL under {args.judge} | "
              f"{args.num_samples} samples ({tag}; joint prior otherwise never enters) ---")
        for temperature, top_p in settings:
            set_seed(args.seed)
            texts, id_lists, lengths = [], [], []
            if gt_driven:
                for batch in gt_loader:
                    batch = batch.to(device)
                    if args.rollout_gt:
                        gen_kwargs = dict(gt_tokens=batch.tokens, gt_mask=batch.mask)
                    else:
                        gen_kwargs = dict(gt_block1=batch.tokens[:, :Sp],
                                          gt_block1_mask=batch.mask[:, :Sp])
                    tokens, lens = model.rollout_generate(
                        batch.tokens.shape[0], device=device,
                        temperature=temperature, top_p=top_p, **gen_kwargs,
                    )
                    for i in range(tokens.shape[0]):
                        ids = tokens[i][: int(lens[i])].tolist()
                        id_lists.append(ids)
                        texts.append(decode(tokenizer, ids))
                        lengths.append(int(lens[i]))
                        if len(texts) >= args.num_samples:
                            break
                    if len(texts) >= args.num_samples:
                        break
            else:
                remaining = args.num_samples
                while remaining > 0:
                    n = min(args.gen_batch, remaining)
                    tokens, lens = model.rollout_generate(
                        n, device=device, temperature=temperature, top_p=top_p,
                        prior_s1=args.rollout_prior_s1, cfg_scale=args.cfg_scale,
                        sde_gamma=args.sde_gamma, noise_scale=args.noise_scale,
                    )
                    for i in range(n):
                        ids = tokens[i][: int(lens[i])].tolist()
                        id_lists.append(ids)
                        texts.append(decode(tokenizer, ids))
                        lengths.append(int(lens[i]))
                    remaining -= n
            if args.save_texts:
                path = f"{args.save_texts}_rollout_t{temperature:.2f}_p{top_p:.2f}.jsonl"
                with open(path, "w", encoding="utf-8") as f:
                    for t in texts:
                        f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")
            ppl, n_tok, ent_corpus, ent_sample = judge_ppl(
                texts, judge, judge_tok, device, args.judge_batch
            )
            mean_len = sum(lengths) / max(len(lengths), 1)
            print(f"  rollout temp={temperature:.2f} top_p={top_p:.2f} | "
                  f"gen PPL={ppl:8.2f} | gpt2 entropy corpus={ent_corpus:.3f} "
                  f"/sample={ent_sample:.3f} | t5 corpus={unigram_entropy(id_lists):.3f} "
                  f"/sample={mean_sample_entropy(id_lists):.3f} | "
                  f"mean len={mean_len:.0f} t5 tok (scored {n_tok} judge tok)", flush=True)
            print(degeneration_line(id_lists), flush=True)
        return

    if args.recon:
        _, val_loader = make_dataloaders(cfg)
        print(f"\n--- AE reconstruction PPL under {args.judge} | "
              f"{args.num_samples} held-out docs ---")
        for temperature, top_p in settings:
            for label, sched in schedules_for(args, temperature, top_p, rounds_list):
                set_seed(args.seed)
                texts, id_lists, lengths = [], [], []
                with torch.no_grad():
                    for batch in val_loader:
                        batch = batch.to(device)
                        enc = model.encode(batch.tokens, batch.mask)
                        tokens, lens = model.refine_decode(
                            enc.block_feats, sched, commit_list(args),
                            order=args.refine_order,
                            landing_lambda=args.landing_lambda,
                            rc=model.sample_rc(enc),
                            rc_mode=args.rc_mode, landing_off=args.landing_off,
                            rc_lookahead_off=args.rc_lookahead_off,
                            no_repeat_ngram=args.no_repeat_ngram,
                        )
                        for i in range(tokens.shape[0]):
                            ids = tokens[i][: int(lens[i])].tolist()
                            id_lists.append(ids)
                            texts.append(decode(tokenizer, ids))
                            lengths.append(int(lens[i]))
                            if len(texts) >= args.num_samples:
                                break
                        if len(texts) >= args.num_samples:
                            break
                if args.save_texts:
                    tag = "".join(c for c in label if c.isalnum())
                    path = (f"{args.save_texts}_recon_t{temperature:.2f}_p{top_p:.2f}"
                            f"{'_' + tag if tag else ''}.jsonl")
                    with open(path, "w", encoding="utf-8") as f:
                        for t in texts:
                            f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")
                ppl, n_tok, ent_corpus, ent_sample = judge_ppl(
                    texts, judge, judge_tok, device, args.judge_batch
                )
                mean_len = sum(lengths) / max(len(lengths), 1)
                print(f"  recon {label:14s} temp={temperature:.2f} top_p={top_p:.2f} | "
                      f"gen PPL={ppl:8.2f} | gpt2 entropy corpus={ent_corpus:.3f} "
                      f"/sample={ent_sample:.3f} | t5 corpus={unigram_entropy(id_lists):.3f} "
                      f"/sample={mean_sample_entropy(id_lists):.3f} | "
                      f"mean len={mean_len:.0f} t5 tok (scored {n_tok} judge tok)", flush=True)
                print(degeneration_line(id_lists), flush=True)
        return

    print(f"\n--- generative PPL under {args.judge} | {args.num_samples} samples ---")
    for steps in steps_list:
        cfg.flow.num_sampling_steps = steps   # model.generate reads this live
        for temperature, top_p in settings:
            for label, sched in schedules_for(args, temperature, top_p, rounds_list):
                set_seed(args.seed)   # same initial noise across step counts
                texts, id_lists, mean_len = generate_corpus(
                    model, tokenizer, args, device, sched
                )
                if args.save_texts:
                    tag = "".join(c for c in label if c.isalnum())    # e.g. R2a for a refine schedule
                    path = (f"{args.save_texts}_nfe{steps}_t{temperature:.2f}"
                            f"_p{top_p:.2f}{'_' + tag if tag else ''}.jsonl")
                    with open(path, "w", encoding="utf-8") as f:
                        for t in texts:
                            f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")
                ppl, n_tok, ent_corpus, ent_sample = judge_ppl(
                    texts, judge, judge_tok, device, args.judge_batch
                )
                print(f"  NFE={steps:4d} {label:14s} "
                      f"temp={temperature:.2f} top_p={top_p:.2f} "
                      f"cfg={args.cfg_scale:.2f} sde={args.sde_gamma:.2f} "
                      f"ns={args.noise_scale:.2f} | "
                      f"gen PPL={ppl:8.2f} | gpt2 entropy corpus={ent_corpus:.3f} "
                      f"/sample={ent_sample:.3f} | t5 corpus={unigram_entropy(id_lists):.3f} "
                      f"/sample={mean_sample_entropy(id_lists):.3f} | "
                      f"mean len={mean_len:.0f} t5 tok (scored {n_tok} judge tok)", flush=True)
                print(degeneration_line(id_lists), flush=True)

    if args.reference:
        _, val_loader = make_dataloaders(cfg)
        texts, id_lists = [], []
        for batch in val_loader:
            for i in range(batch.tokens.shape[0]):
                ids = batch.tokens[i][batch.mask[i]].tolist()
                id_lists.append(ids)
                texts.append(decode(tokenizer, ids))
                if len(texts) >= args.num_samples:
                    break
            if len(texts) >= args.num_samples:
                break
        ppl, n_tok, ent_corpus, ent_sample = judge_ppl(
            texts, judge, judge_tok, device, args.judge_batch
        )
        mean_len = sum(len(x) for x in id_lists) / max(len(id_lists), 1)
        print(f"  reference (held-out)     | gen PPL={ppl:8.2f} | "
              f"gpt2 entropy corpus={ent_corpus:.3f} /sample={ent_sample:.3f} | "
              f"t5 corpus={unigram_entropy(id_lists):.3f} "
              f"/sample={mean_sample_entropy(id_lists):.3f} | "
              f"mean len={mean_len:.0f} t5 tok (scored {n_tok} judge tok)")
        print(degeneration_line(id_lists))


if __name__ == "__main__":
    main()
