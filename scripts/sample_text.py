"""Sample documents from a trained PSF checkpoint into a markdown file.

    python -m scripts.sample_text --ckpt runs/prior-prefix-causal/prior.pt \
        --encoder prefix --causal-latents --num 10 --out samples/run.md

Appends one section per invocation, so several settings can be collected into
the same file. Also prints a cross-sample diversity stat (distinct 20-token
prefixes) — useful for spotting mean-collapse at low step counts.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from psf.config import Config
from psf.models.psf import PSF, refine_schedule
from psf.utils.data import decode, load_tokenizer
from psf.utils.misc import get_device, load_checkpoint, set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PSF text sampling -> markdown")
    p.add_argument("--ckpt", required=True, help="stage-2 checkpoint (holds AE + prior)")
    p.add_argument("--encoder", default="block", choices=["block", "prefix"])
    p.add_argument("--causal-latents", action="store_true")
    p.add_argument("--null-latent", action="store_true")
    p.add_argument("--latent-window", type=int, default=0)
    p.add_argument("--boundary-tokens", action="store_true")
    p.add_argument("--boundary-window", type=int, default=1)
    p.add_argument("--landing-state", action="store_true",
                   help="must match the checkpoint")
    p.add_argument("--landing-dim", type=int, default=0,
                   help="must match the checkpoint")
    p.add_argument("--landing-attn", action="store_true",
                   help="must match the checkpoint")
    p.add_argument("--level-cond", action="store_true",
                   help="must match the checkpoint")
    p.add_argument("--cross-before-self", action="store_true",
                   help="must match the checkpoint")
    p.add_argument("--abs-pos-emb", action="store_true",
                   help="must match the checkpoint")
    p.add_argument("--rc-dim", type=int, default=None,
                   help="must match the checkpoint")
    p.add_argument("--no-rc-memory", action="store_true",
                   help="must match the checkpoint")
    p.add_argument("--rc-lookahead", action="store_true",
                   help="block k also reads the next block's code r_{k+1} (must match the checkpoint)")
    p.add_argument("--z-level", type=float, default=1.0)
    p.add_argument("--landing-residual", action="store_true",
                   help="must match the checkpoint")
    p.add_argument("--landing-cfg", action="store_true",
                   help="must match the checkpoint")
    p.add_argument("--landing-lambda", type=float, default=1.0)
    p.add_argument("--prev-window", action="store_true",
                   help="must match the checkpoint")
    p.add_argument("--decoder-depth", type=int, default=None,
                   help="decoder layers; must match the checkpoint")
    p.add_argument("--denoiser-hidden", type=int, default=None)
    p.add_argument("--denoiser-depth", type=int, default=None)
    p.add_argument("--denoiser-heads", type=int, default=None)
    p.add_argument("--denoiser-bottleneck", type=int, default=None)
    p.add_argument("--global-latents", type=int, default=0)
    p.add_argument("--t5-features", action="store_true")
    p.add_argument("--num-blocks", type=int, default=None,
                   help="K (block span = 1024 / K); must match the checkpoint")
    p.add_argument("--num", type=int, default=10)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--sampling-steps", type=int, default=50)
    p.add_argument("--sc-cfg", action="store_true",
                   help="checkpoint has SC-CFG guidance tokens (must match)")
    p.add_argument("--cfg-scale", type=float, default=1.0)
    p.add_argument("--sde-gamma", type=float, default=0.0)
    p.add_argument("--df-stagger", type=float, default=0.0,
                   help="staggered (pyramid) reveal strength lambda for the flow sampler; 0 = off")
    p.add_argument("--df-order", default="random", choices=["random", "ltr"])
    p.add_argument("--stitch", action="store_true",
                   help="checkpoint has the seam-stitch mode; run the stitch pass after decoding")
    p.add_argument("--stitch-start", type=int, default=3)
    p.add_argument("--stitch-end", type=int, default=2)
    p.add_argument("--stitch-prev", type=int, default=3)
    p.add_argument("--stitch-next", type=int, default=3)
    p.add_argument("--stitch-ckpt", default=None,
                   help="two-decoder inference: stitch-mode decoder from this checkpoint")
    p.add_argument("--stitch-redecode", type=int, default=0)
    p.add_argument("--rollout", action="store_true",
                   help="block-AR rollout generation (no prior)")
    p.add_argument("--refine-rounds", type=int, default=0,
                   help="latent-refinement rounds: re-encode the decoded text "
                        "and decode again this many times")
    p.add_argument("--refine-anneal", action="store_true",
                   help="cool the temperature linearly to 0 across the rounds")
    p.add_argument("--max-chars", type=int, default=1500, help="truncate each sample in the md")
    p.add_argument("--out", default=None,
                   help="markdown file to append to; keep generations under samples/")
    p.add_argument("--label", default=None, help="section heading; auto if omitted")
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Config()
    cfg.encoder_type = args.encoder
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
    if args.rc_dim is not None:
        cfg.rc_dim = args.rc_dim
    if args.no_rc_memory:
        cfg.decoder.rc_memory = False
    if args.rc_lookahead:
        cfg.decoder.rc_lookahead = True
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
    cfg.num_global_latents = args.global_latents
    cfg.encoder.t5_features = args.t5_features
    if args.sc_cfg:
        cfg.denoiser.num_cfg_tokens = 4
    if args.num_blocks is not None:
        cfg.num_blocks = args.num_blocks
    if args.stitch and not args.stitch_ckpt:
        cfg.decoder.stitch = True
    cfg.flow.num_sampling_steps = args.sampling_steps
    cfg.flow.df_stagger = args.df_stagger
    cfg.flow.df_order = args.df_order
    set_seed(args.seed)
    device = get_device(args.device)
    torch.set_float32_matmul_precision("high")

    tokenizer = load_tokenizer(cfg)
    model = PSF(cfg).to(device).eval()
    load_checkpoint(args.ckpt, model, device)
    if args.stitch_ckpt:
        model.attach_stitch_decoder(args.stitch_ckpt)

    if args.rollout:
        tokens, lengths = model.rollout_generate(
            args.num, device=device, temperature=args.temperature, top_p=args.top_p
        )
    elif args.stitch:
        tokens, lengths = model.generate_stitched(
            args.num, device=device, temperature=args.temperature, top_p=args.top_p,
            cfg_scale=args.cfg_scale, sde_gamma=args.sde_gamma, z_level=args.z_level,
            start_n=args.stitch_start, end_n=args.stitch_end,
            prev_n=args.stitch_prev, next_n=args.stitch_next,
            redecode=args.stitch_redecode,
        )
    else:
        tokens, lengths = model.generate(
            args.num, device=device, cfg_scale=args.cfg_scale,
            sde_gamma=args.sde_gamma,
            schedule=refine_schedule(args.temperature, args.top_p,
                                     args.refine_rounds, args.refine_anneal),
            z_level=args.z_level, landing_lambda=args.landing_lambda,
        )
    texts = [decode(tokenizer, tokens[i][: int(lengths[i])]) for i in range(args.num)]

    prefixes = {tuple(tokens[i][:20].tolist()) for i in range(args.num)}
    stat = f"distinct 20-token prefixes: {len(prefixes)}/{args.num}"

    label = args.label or (
        f"NFE={args.sampling_steps}, temperature={args.temperature}, top_p={args.top_p}"
        f", cfg={args.cfg_scale}, sde={args.sde_gamma}"
    )
    lines = [f"## {label}", "", f"`{args.ckpt}` | seed {args.seed} | {stat}", ""]
    for i, text in enumerate(texts):
        shown = text[: args.max_chars] + ("…" if len(text) > args.max_chars else "")
        lines += [f"**[{i + 1}]** ({int(lengths[i])} tokens)", "", f"> {shown}", ""]
    section = "\n".join(lines)

    print(section)
    print(stat)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "a", encoding="utf-8") as f:
            f.write(section + "\n")
        print(f"appended -> {args.out}")


if __name__ == "__main__":
    main()
