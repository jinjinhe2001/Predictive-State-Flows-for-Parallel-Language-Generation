"""Offline bank of rollout-generated ("gold", sequential-quality) documents.

    python -m scripts.sample_rollout_bank --ckpt runs/ae-srpl/ae.pt \
        --encoder prefix --causal-latents --boundary-tokens --sc-cfg \
        --decoder-depth 18 --level-cond --num 20000 --out runs/rollout_bank/bank.pt

Rollout (block-AR, re-encode after every block) eliminates cross-block
decoherence by construction (about 25 @ 5.1, close to real text) at 64x the
decode cost of one-shot -- feasible offline, once, as a training-data source,
never at inference. This bank feeds ``train_ae.py --rollout-bank`` so the
one-shot decoder gets extra exposure to reconstructing sequential-quality
("gold") text under the exact same srp-style state-corruption robustness
training, instead of only ever seeing real text.

Needs only encoder + decoder (no denoiser/prior): rollout never samples z.
"""

from __future__ import annotations

import argparse

import torch

from psf.config import Config, toy_config
from psf.models.psf import PSF
from psf.utils.data import load_tokenizer
from psf.utils.misc import get_device, load_checkpoint, set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="build an offline rollout-generated document bank")
    p.add_argument("--ckpt", required=True, help="ae checkpoint (encoder+decoder)")
    p.add_argument("--toy", action="store_true")
    p.add_argument("--smoke", action="store_true", help="random weights")
    # Architecture flags -- must match the checkpoint.
    p.add_argument("--encoder", default="prefix", choices=["block", "prefix"])
    p.add_argument("--causal-latents", action="store_true")
    p.add_argument("--boundary-tokens", action="store_true")
    p.add_argument("--latent-dim", type=int, default=None)
    p.add_argument("--decoder-depth", type=int, default=None)
    p.add_argument("--level-cond", action="store_true")
    p.add_argument("--cross-before-self", action="store_true")
    p.add_argument("--abs-pos-emb", action="store_true")
    p.add_argument("--rc-dim", type=int, default=None)
    p.add_argument("--no-rc-memory", action="store_true")
    p.add_argument("--rc-lookahead", action="store_true",
                   help="block k also reads the next block's code r_{k+1} (must match the checkpoint)")
    p.add_argument("--sc-cfg", action="store_true",
                   help="unused by rollout but kept for a uniform ARCH string")
    p.add_argument("--num", type=int, default=20_000)
    p.add_argument("--batch", type=int, default=250)
    p.add_argument("--temperature", type=float, default=0.9,
                   help="rollout ceiling sweet spot (t=0.9 matches "
                        "one-shot's t=0.8 entropy region best)")
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--out", default="runs/rollout_bank/bank.pt")
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = toy_config() if args.toy else Config()
    if args.latent_dim is not None:
        cfg.latent_dim = args.latent_dim
    if args.decoder_depth is not None:
        cfg.decoder.depth = args.decoder_depth
    cfg.__post_init__()
    cfg.encoder_type = args.encoder
    cfg.decoder.causal_latents = args.causal_latents
    cfg.decoder.boundary_tokens = args.boundary_tokens
    cfg.decoder.noise_level_cond = args.level_cond
    cfg.decoder.cross_before_self = args.cross_before_self
    cfg.decoder.abs_pos_emb = args.abs_pos_emb
    if args.rc_dim is not None:
        cfg.rc_dim = args.rc_dim
    if args.no_rc_memory:
        cfg.decoder.rc_memory = False
    if args.rc_lookahead:
        cfg.decoder.rc_lookahead = True
    set_seed(args.seed)
    device = get_device(args.device)
    torch.set_float32_matmul_precision("high")

    load_tokenizer(cfg)
    model = PSF(cfg).to(device).eval()
    if not args.smoke:
        load_checkpoint(args.ckpt, model, device, strict=False)
    print(f"rolling out {args.num} docs from {args.ckpt}"
          f"{' (SMOKE)' if args.smoke else ''} | t={args.temperature} p={args.top_p}")

    L = cfg.max_seq_len
    all_tokens = torch.full((args.num, L), cfg.data.pad_token_id, dtype=torch.int16)
    all_lengths = torch.zeros(args.num, dtype=torch.int32)
    done = 0
    with torch.no_grad():
        while done < args.num:
            n = min(args.batch, args.num - done)
            tokens, lengths = model.rollout_generate(
                n, device=device, temperature=args.temperature, top_p=args.top_p)
            all_tokens[done:done + n] = tokens.to("cpu", torch.int16)
            all_lengths[done:done + n] = lengths.to("cpu", torch.int32)
            done += n
            print(f"  ... {done}/{args.num}", flush=True)

    import os
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save({"tokens": all_tokens, "lengths": all_lengths,
                "src": args.ckpt, "temperature": args.temperature,
                "top_p": args.top_p, "seed": args.seed}, args.out)
    print(f"saved {tuple(all_tokens.shape)} int16 "
          f"({all_tokens.numel() * 2 / 1e9:.2f} GB) -> {args.out}")


if __name__ == "__main__":
    main()
