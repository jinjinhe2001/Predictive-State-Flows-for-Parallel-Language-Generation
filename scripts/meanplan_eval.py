"""Mean-plan / no-prior generation sweep at matched entropy.

Standalone companion of scripts/plan_probe.py (which is left untouched). plan_probe's generation section runs every
variant at one or two sampler settings and cannot pass the 4-gram ban through, so a mean-plan point can never be
read at the paper's entropy anchor. This script:

  * samples ONE fixed set of --num-docs plans (states + codes) from the prior and reuses it for every setting and
    variant. The held-out documents are ALWAYS encoded first (the DataLoader iterator draws from the global RNG), so
    every run on the same checkpoint with the same --seed / --gen-batch / --num-docs / --sampling-steps draws the
    identical plan set, whichever --variants it decodes (e.g. a mean_meancode-only run shares its plans and mean plan
    with a plan_prior run; up to GPU non-determinism). Not guaranteed to be the round-4 plan_probe plans (plan_probe
    runs its teacher-forced section and loads the judge in between);
  * decodes selected plan variants (--variants) at every --settings point, re-seeding before each (setting, variant)
    so the writer's sampling noise is paired across variants;
  * passes --no-repeat-ngram / --landing-off / --z-level through to the writer exactly as eval_gen does
    (defaults follow plan_probe, i.e. the paper's one-shot protocol: --z-level 0.0 and --gen-batch 50 unless given);
  * scores every dump with the paper's judge at the 1024-token protocol AND at a fixed 512-token window (all
    documents truncated; the round-4 length-controlled protocol, anchor 4.845), in one process;
  * writes one eval_gen-style log per (variant, window) that scripts/isoentropy.py can read, a summary with the
    iso-entropy value of every variant (log-linear between the nearest bracketing pair; extrapolation flagged),
    and a LABEL=path list of the two dumps bracketing the anchor per variant (input for mauve_eval / topic_metrics).

Variants (short keys):
  plan_prior     the prior's plan, the prior's codes                    (plain / nucleus rows of the paper)
  plan_zero      the prior's plan, zero codes                           (zero-code / states-only rows)
  mean_zero      every document gets the mean of the sampled plans, zero codes     (= plan_probe "mean plan")
  mean_meancode  mean of the sampled plans AND mean of the sampled codes (no per-document information at all)
  encmean_zero   every document gets the mean ENCODER plan of the held-out documents, zero codes
                 (strictly prior-free: the flow prior is never used)
  true_zero      each held-out document's own encoder plan, zero codes (true-plan decode; upper reference)

  python -m scripts.meanplan_eval --ckpt runs/prior-k8-zerocode/prior.pt --num-blocks 8 <arch flags of eval_gen> \
      --no-repeat-ngram 4 --variants plan_zero,mean_zero,encmean_zero --settings 1.0:0.55,1.0:0.6,1.0:0.65 \
      --out r5_notes/meanplan_so_k8_ban --save-prefix texts_r5_notes_meanplan_so_k8_ban
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import torch

from psf.diffusion.flow import sample as flow_sample
from psf.models.psf import PSF
from psf.utils.data import decode, load_tokenizer, make_dataloaders
from psf.utils.gen_metrics import repetition_metrics
from psf.utils.misc import get_device, load_checkpoint, set_seed
from scripts import eval_gen
from scripts.isoentropy import at_entropy

VARIANTS = ("plan_prior", "plan_zero", "mean_zero", "mean_meancode", "encmean_zero", "true_zero")


def own_args():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--num-docs", type=int, default=1000)
    p.add_argument("--variants", default="plan_prior,plan_zero,mean_zero,encmean_zero",
                   help="comma list of " + ",".join(VARIANTS))
    p.add_argument("--out", default="r5_notes/meanplan",
                   help="prefix for <out>.txt (summary), <out>_<variant>_w<W>.log, <out>_bracket_w<W>.txt")
    p.add_argument("--save-prefix", default=None, help="texts: <prefix>_<variant>_t<T>_p<P>.jsonl")
    p.add_argument("--windows", default="1024:5.16,512:4.845",
                   help="judge window:anchor entropy pairs (1024 = paper protocol; 512 = round-4 fixed window)")
    p.add_argument("--decode-seed-offset", type=int, default=1,
                   help="writer noise seed = --seed + offset, reset before every (setting, variant)")
    p.add_argument("--dry-run", action="store_true", help="parse arguments, print the plan and exit")
    p.add_argument("-h", "--help", action="store_true")
    mine, rest = p.parse_known_args()
    if mine.help:
        print(__doc__)
        p.print_help()
        print("\nAll remaining flags go to scripts.eval_gen.parse_args (architecture, --settings, --gen-batch, "
              "--sampling-steps, --no-repeat-ngram, --landing-off, --z-level, --judge, --judge-batch, --seed).")
        sys.exit(0)
    sys.argv = [sys.argv[0]] + rest
    args = eval_gen.parse_args()
    # plan_probe's defaults (the paper's one-shot protocol), not eval_gen's: z-level 0, generation batch 50
    if not any(a == "--z-level" or a.startswith("--z-level=") for a in rest):
        args.z_level = 0.0
    if not any(a == "--gen-batch" or a.startswith("--gen-batch=") for a in rest):
        args.gen_batch = 50
    return mine, args


def main() -> None:
    mine, args = own_args()
    variants = [v.strip() for v in mine.variants.split(",") if v.strip()]
    bad = [v for v in variants if v not in VARIANTS]
    if bad:
        raise SystemExit(f"unknown variants {bad}; choose from {VARIANTS}")
    settings = [tuple(float(x) for x in s.split(":")) for s in args.settings.split(",")]
    windows = [(int(w.split(":")[0]), float(w.split(":")[1])) for w in mine.windows.split(",")]
    header = (f"# meanplan_eval: {args.ckpt} | K={args.num_blocks} | {mine.num_docs} docs | gen-batch {args.gen_batch} "
              f"| steps {args.sampling_steps} | z-level {args.z_level} | ban {args.no_repeat_ngram} | "
              f"landing-off {args.landing_off} | variants {variants} | settings {settings} | windows {windows}")
    if mine.dry_run:
        print(header)
        return
    if not args.ckpt:
        raise SystemExit("--ckpt is required")
    os.makedirs(os.path.dirname(os.path.abspath(mine.out)), exist_ok=True)
    cfg, _ = eval_gen.config_from_args(args)
    set_seed(args.seed)
    device = get_device(args.device)
    torch.set_float32_matmul_precision("high")
    tokenizer = load_tokenizer(cfg)
    model = PSF(cfg).to(device).eval()
    load_checkpoint(args.ckpt, model, device)
    K = cfg.num_blocks
    lines = [header]
    log = lambda s: (print(s, flush=True), lines.append(s))

    # held-out encoder plans (encmean / true). Encoded unconditionally, before the prior is sampled: iterating the
    # DataLoader consumes the global RNG, so doing it only for some --variants would give different plan sets to
    # runs on the same checkpoint (and break the pairing of e.g. mean_meancode with plan_prior across jobs).
    _, val_loader = make_dataloaders(cfg)
    feats, n = [], 0
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            feats.append(model.encode(batch.tokens, batch.mask).block_feats)
            n += batch.tokens.shape[0]
            if n >= mine.num_docs:
                break
    enc_states = torch.cat(feats, 0)[: mine.num_docs]
    if enc_states.shape[0] < mine.num_docs:
        raise SystemExit(f"only {enc_states.shape[0]} held-out documents")
    enc_mean = enc_states.float().mean(0, keepdim=True)

    nlat = K * cfg.num_latent_slots + cfg.num_global_latents
    plans, codes = [], []
    with torch.no_grad():
        remaining = mine.num_docs
        while remaining > 0:
            b = min(args.gen_batch, remaining)
            z = flow_sample(model.denoiser, b, nlat, model.prior_dim, cfg.flow, device,
                            cfg_scale=args.cfg_scale, sde_gamma=args.sde_gamma, noise_scale=args.noise_scale)
            s, rc = model.split_latents(model.denoiser.denormalize(z))
            plans.append(s); codes.append(rc)
            remaining -= b
    S_gen = torch.cat(plans, 0)
    RC_gen = torch.cat(codes, 0) if codes[0] is not None else None
    mean_plan = S_gen.mean(0, keepdim=True)
    mean_code = None if RC_gen is None else RC_gen.mean(0, keepdim=True)
    if RC_gen is not None:
        log(f"# sampled codes: rms {RC_gen.float().pow(2).mean().sqrt():.4f} | mean-code rms {mean_code.float().pow(2).mean().sqrt():.4f}")
    if enc_mean is not None:
        d = (mean_plan.float() - enc_mean).pow(2).mean().sqrt() / enc_states.float().std()
        log(f"# |mean sampled plan - mean encoder plan| / sd(encoder states) = {d:.4f}")
    if mine.save_prefix:
        torch.save({"plans": S_gen.cpu(), "codes": None if RC_gen is None else RC_gen.cpu()},
                   f"{mine.save_prefix}_plans.pt")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    judge_tok = AutoTokenizer.from_pretrained(args.judge); judge_tok.pad_token = judge_tok.eos_token
    judge = AutoModelForCausalLM.from_pretrained(args.judge).to(device).eval()

    points = {(v, w): [] for v in variants for w, _ in windows}      # -> (entropy, ppl, temp, top_p, path)
    vlogs = {(v, w): f"{mine.out}_{v}_w{w}.log" for v in variants for w, _ in windows}
    for pth in vlogs.values():
        open(pth, "w", encoding="utf-8").write(header + "\n")
    for temperature, top_p in settings:
        log(f"\n## t={temperature:.2f} p={top_p:.2f}")
        for v in variants:
            set_seed(args.seed + mine.decode_seed_offset)
            texts, id_lists = [], []
            with torch.no_grad():
                for i in range(0, mine.num_docs, args.gen_batch):
                    s = S_gen[i:i + args.gen_batch]
                    rc = None if RC_gen is None else RC_gen[i:i + args.gen_batch]
                    rc_mode = "zero"
                    if v == "plan_prior":
                        rc_mode = "plan"
                    elif v == "mean_zero":
                        s = mean_plan.expand_as(s).contiguous()
                    elif v == "mean_meancode":
                        s = mean_plan.expand_as(s).contiguous()
                        rc = None if rc is None else mean_code.expand_as(rc).contiguous()
                        rc_mode = "plan"
                    elif v == "encmean_zero":
                        s = enc_mean.to(s.dtype).expand_as(s).contiguous()
                    elif v == "true_zero":
                        s = enc_states[i:i + args.gen_batch].to(s.dtype)
                    tokens, lens = model.refine_decode(
                        s, [(temperature, top_p)], None, "random", z_level=args.z_level,
                        landing_lambda=args.landing_lambda, n_cands=1, shared_noise=False, rc=rc,
                        no_repeat_ngram=args.no_repeat_ngram, landing_off=args.landing_off, rc_mode=rc_mode)
                    for j in range(tokens.shape[0]):
                        ids = tokens[j][: int(lens[j])].tolist()
                        id_lists.append(ids); texts.append(decode(tokenizer, ids))
            path = None
            if mine.save_prefix:
                path = f"{mine.save_prefix}_{v}_t{temperature:.2f}_p{top_p:.2f}.jsonl"
                with open(path, "w", encoding="utf-8") as f:
                    for t in texts:
                        f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")
            rep = repetition_metrics(id_lists)
            mean_len = sum(len(x) for x in id_lists) / len(id_lists)
            for w, _ in windows:
                ppl, n_tok, ent_c, ent_s = eval_gen.judge_ppl(texts, judge, judge_tok, device, args.judge_batch,
                                                              window=w)
                row = (f"  {v:14s} temp={temperature:.2f} top_p={top_p:.2f} | gen PPL={ppl:8.2f} | "
                       f"gpt2 entropy corpus={ent_c:.3f} /sample={ent_s:.3f} | window {w} | "
                       f"loop8 {100 * rep['loop8_sample_rate']:.1f}% | mean len={mean_len:.0f} t5 tok "
                       f"(scored {n_tok} judge tok)")
                log(row)
                with open(vlogs[(v, w)], "a", encoding="utf-8") as f:
                    f.write(row + "\n")
                points[(v, w)].append((ent_s, ppl, temperature, top_p, path))

    # ---------------- iso-entropy summary ----------------
    for w, anchor in windows:
        log(f"\n## iso-entropy, window {w}, anchor {anchor} (log-linear between the nearest bracketing pair)")
        ref = {}
        for v in variants:
            pts = sorted(points[(v, w)])
            if len(pts) < 2:
                log(f"  {v:14s} fewer than two points"); continue
            val, extrap, how = at_entropy([(h, p, t if t < 0.999 else tp) for h, p, t, tp, _ in pts], anchor)
            ref[v] = val
            log(f"  {v:14s} {val:7.2f}{'  EXTRAPOLATED' if extrap else ''}  ({how}; entropy range "
                f"{pts[0][0]:.3f}..{pts[-1][0]:.3f})")
        for base in ("plan_zero", "plan_prior"):
            if base in ref:
                log(f"  relative to {base}: " + "  ".join(f"{v} {100 * (ref[v] / ref[base] - 1):+.1f}%"
                                                         for v in ref if v != base))
        # the two dumps bracketing the anchor (nearest below / above; else the two nearest) -> MAUVE / topic inputs
        specs = []
        for v in variants:
            pts = sorted(points[(v, w)])
            if len(pts) < 2 or pts[0][4] is None:
                continue
            lo = [p for p in pts if p[0] <= anchor]
            hi = [p for p in pts if p[0] >= anchor]
            pair = [lo[-1], hi[0]] if lo and hi else sorted(pts, key=lambda p: abs(p[0] - anchor))[:2]
            for h, p, t, tp, path in pair:
                tag = os.path.basename(path)[:-len(".jsonl")]
                specs.append(f"{tag}={path}")
        with open(f"{mine.out}_bracket_w{w}.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(dict.fromkeys(specs)) + ("\n" if specs else ""))
    open(f"{mine.out}.txt", "w", encoding="utf-8").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
