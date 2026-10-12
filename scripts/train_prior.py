"""Stage 2: fit the rectified-flow prior on a frozen autoencoder's latents.

    python -m scripts.train_prior --ae-ckpt runs/ae/ae.pt --steps 100000

The model shape flags must match the AE checkpoint, otherwise its weights will
not load.

Multi-GPU (batch size is per process):

    torchrun --standalone --nproc-per-node 8 -m scripts.train_prior ...
"""

from __future__ import annotations

import argparse

import torch

from psf.config import Config, toy_config
from psf.models.psf import PSF
from psf.training.trainer_prior import PriorTrainer
from psf.utils import dist
from psf.utils.data import decode, load_tokenizer, make_dataloaders
from psf.utils.misc import get_device, load_checkpoint, set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PSF stage-2 prior training")
    p.add_argument("--ae-ckpt", required=True, help="frozen stage-1 checkpoint")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--toy", action="store_true",
                   help="built-in synthetic corpus + small model; no download")
    p.add_argument("--dataset-path", default=None, help="pre-tokenized dataset dir")
    p.add_argument("--tokenizer", default=None)
    p.add_argument("--rollout-bank", default=None,
                   help="path to a scripts.sample_rollout_bank.py bank; when "
                        "set, this fraction of training batches (--rollout-"
                        "mix-p) fit the flow on rollout-generated ('self-"
                        "referential', sequential-quality) documents' latents "
                        "instead of real text's -- the real loader stays as "
                        "an anchor")
    p.add_argument("--rollout-mix-p", type=float, default=0.5,
                   help="P(a training batch is drawn from --rollout-bank)")
    # Must match the AE checkpoint.
    p.add_argument("--max-seq-len", type=int, default=None, help="L")
    p.add_argument("--num-blocks", type=int, default=None, help="K")
    p.add_argument("--latent-dim", type=int, default=None, help="D")
    p.add_argument("--latent-slots", type=int, default=None,
                   help="m latent vectors per block; must match the AE checkpoint")
    p.add_argument("--encoder", default=None, choices=["block", "prefix"],
                   help="must match the AE checkpoint")
    p.add_argument("--causal-latents", action="store_true",
                   help="must match the AE checkpoint")
    p.add_argument("--prefix-unshifted", action="store_true", help="must match stage 1")
    p.add_argument("--null-latent", action="store_true",
                   help="must match the AE checkpoint")
    p.add_argument("--latent-window", type=int, default=None,
                   help="must match the AE checkpoint")
    p.add_argument("--global-latents", type=int, default=None,
                   help="must match the AE checkpoint")
    p.add_argument("--t5-features", action="store_true",
                   help="must match the AE checkpoint")
    p.add_argument("--boundary-tokens", action="store_true",
                   help="must match the AE checkpoint")
    p.add_argument("--landing-attn", action="store_true",
                   help="must match the AE checkpoint")
    p.add_argument("--level-cond", action="store_true",
                   help="must match the AE checkpoint")
    p.add_argument("--decoder-depth", type=int, default=None,
                   help="decoder layers; must match the AE checkpoint")
    p.add_argument("--lex-bag", action="store_true",
                   help="AE checkpoint has the cross-block lexical-bag channel (decoder.bag_proj)")
    p.add_argument("--lex-gate", action="store_true",
                   help="AE checkpoint has the learned lexical bonus (decoder.gate_proj / lex_scale)")
    p.add_argument("--rc-dim", type=int, default=None,
                   help="realization-code dims; must match the AE checkpoint. "
                        "The prior then models [s_k ; r_k] per block")
    p.add_argument("--no-rc-memory", action="store_true",
                   help="must match the AE checkpoint")
    p.add_argument("--rc-lookahead", action="store_true",
                   help="block k also reads the next block's code r_{k+1} (must match the checkpoint)")
    p.add_argument("--rc-source", default=None, choices=["sample", "mean", "zero"],
                   help="which code the flow fits: posterior sample (default), mean, or the constant zero "
                        "(states-only prior)")
    # Prior-side knobs.
    p.add_argument("--denoiser-hidden", type=int, default=None)
    p.add_argument("--denoiser-depth", type=int, default=None)
    p.add_argument("--denoiser-heads", type=int, default=None)
    p.add_argument("--denoiser-bottleneck", type=int, default=None,
                   help="input bottleneck width D -> b -> H (default 128; the "
                        "realization code adds 64 near-independent dims, which "
                        "a 128-wide bottleneck shared with the 512-d state throttles)")
    p.add_argument("--no-self-cond", action="store_true", help="disable self-conditioning")
    p.add_argument("--time-sampler", default=None, choices=["uniform", "logit_normal"])
    p.add_argument("--sampling-steps", type=int, default=None, help="Euler steps at inference")
    # Improved-variant training tricks (ELF-style).
    p.add_argument("--objective", default=None, choices=["x0", "v"],
                   help="v: x0 MSE weighted by 1/(1-t)^2 (ELF's v-space loss)")
    p.add_argument("--sc-cfg", action="store_true",
                   help="train-time self-conditioning CFG with guidance-scale tokens")
    p.add_argument("--cfg-min", type=float, default=None)
    p.add_argument("--cfg-max", type=float, default=None)
    p.add_argument("--ctx-drop", type=float, default=None,
                   help=">0: prefix-drop CFG training (context-free row prob)")
    p.add_argument("--time-mean", type=float, default=None, help="logit-normal t mean")
    p.add_argument("--time-std", type=float, default=None, help="logit-normal t std")
    p.add_argument("--df", action="store_true",
                   help="diffusion forcing: an independent flow time per block (the "
                        "denoiser gets a [B,K] time; same weights/shapes as a plain prior)")
    p.add_argument("--df-mix-prob", type=float, default=None,
                   help="with --df: P(row keeps one shared time) (default 0.3)")
    p.add_argument("--df-clean-prob", type=float, default=None,
                   help="with --df: per-block P(real clean context block: t=1, out of the "
                        "loss) in independent-time rows (default 0)")
    p.add_argument("--df-rollout-prob", type=float, default=None,
                   help="with --df: P(row gets on-policy context: a random half of its "
                        "blocks is replaced by the denoiser's own clean prediction at "
                        "time 1 and dropped from the loss) (default 0)")
    p.add_argument("--df-rollout-frac", type=float, default=None,
                   help="with --df-rollout-prob: share of blocks turned into context (default 0.5)")
    p.add_argument("--df-ordered-prob", type=float, default=None,
                   help="with --df: P(row's per-block times sorted descending, "
                        "earlier blocks cleaner) (default 0)")
    p.add_argument("--ema-decay", type=float, default=None,
                   help=">0 keeps and saves an EMA of the denoiser")
    p.add_argument("--no-ema-warmup", action="store_true",
                   help="constant EMA decay from step 0 (pre-2026-09-04 behaviour; the "
                        "default warms the decay up so the shadow forgets its init fast)")
    p.add_argument("--bf16", action="store_true", help="bfloat16 autocast training")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--lr-schedule", default=None, choices=["const", "cosine"],
                   help="const (default) or cosine decay to lr_min_ratio after warmup")
    p.add_argument("--device", default=None, help="auto | cpu | cuda")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--resume", default=None, help="continue from a prior checkpoint")
    p.add_argument("--samples", type=int, default=4, help="#samples to print at the end")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-p", type=float, default=0.95)
    return p.parse_args()


def build_config(args: argparse.Namespace) -> Config:
    cfg = toy_config() if args.toy else Config()
    if args.output_dir is not None:
        cfg.train.output_dir = args.output_dir
    elif not args.toy:
        cfg.train.output_dir = "runs/prior"
    for target, name, value in [
        (cfg, "max_seq_len", args.max_seq_len),
        (cfg, "num_blocks", args.num_blocks),
        (cfg, "latent_dim", args.latent_dim),
        (cfg, "num_latent_slots", args.latent_slots),
        (cfg, "encoder_type", args.encoder),
        (cfg, "num_global_latents", args.global_latents),
        (cfg.data, "dataset_path", args.dataset_path),
        (cfg.data, "tokenizer_name", args.tokenizer),
        (cfg.denoiser, "hidden_size", args.denoiser_hidden),
        (cfg.denoiser, "depth", args.denoiser_depth),
        (cfg.denoiser, "num_heads", args.denoiser_heads),
        (cfg.denoiser, "bottleneck_dim", args.denoiser_bottleneck),
        (cfg.flow, "time_sampler", args.time_sampler),
        (cfg.flow, "num_sampling_steps", args.sampling_steps),
        (cfg.flow, "objective", args.objective),
        (cfg.flow, "cfg_min", args.cfg_min),
        (cfg.flow, "cfg_max", args.cfg_max),
        (cfg.flow, "ctx_drop_prob", args.ctx_drop),
        (cfg.flow, "logit_normal_mean", args.time_mean),
        (cfg.flow, "logit_normal_std", args.time_std),
        (cfg.flow, "df_mix_prob", args.df_mix_prob),
        (cfg.flow, "df_ordered_prob", args.df_ordered_prob),
        (cfg.flow, "df_clean_prob", args.df_clean_prob),
        (cfg.flow, "df_rollout_prob", args.df_rollout_prob),
        (cfg.flow, "df_rollout_frac", args.df_rollout_frac),
        (cfg.train, "batch_size", args.batch_size),
        (cfg.train, "prior_steps", args.steps),
        (cfg.train, "lr", args.lr),
        (cfg.train, "lr_schedule", args.lr_schedule),
        (cfg.train, "device", args.device),
        (cfg.train, "seed", args.seed),
        (cfg.train, "ema_decay", args.ema_decay),
        (cfg.decoder, "latent_window", args.latent_window),
        (cfg.decoder, "depth", args.decoder_depth),
        (cfg, "rc_dim", args.rc_dim),
        (cfg.train, "prior_rc_source", args.rc_source),
    ]:
        if value is not None:
            setattr(target, name, value)
    if args.no_rc_memory:
        cfg.decoder.rc_memory = False
    if args.rc_lookahead:
        cfg.decoder.rc_lookahead = True
    if args.df:
        cfg.flow.diffusion_forcing = True
    if args.no_ema_warmup:
        cfg.train.ema_warmup = False
    if args.no_self_cond:
        cfg.denoiser.self_conditioning = False
    if args.causal_latents:
        cfg.decoder.causal_latents = True
    if args.prefix_unshifted:
        cfg.prefix_unshifted = True
    if args.null_latent:
        cfg.decoder.null_latent = True
    if args.boundary_tokens:
        cfg.decoder.boundary_tokens = True
    if args.landing_attn:
        cfg.decoder.landing_attn = True
    if args.level_cond:
        cfg.decoder.noise_level_cond = True
    if args.lex_bag:
        cfg.decoder.lex_bag = True
    if args.lex_gate:
        cfg.decoder.lex_gate = True
    if args.sc_cfg:
        cfg.flow.sc_cfg = True
        cfg.denoiser.num_cfg_tokens = 4
    if args.bf16:
        cfg.train.bf16 = True
    if args.t5_features:
        cfg.encoder.t5_features = True
    cfg.__post_init__()   # re-check L % K and the hybrid-RoPE head split
    return cfg


@torch.no_grad()
def load_denoiser_resume(model: PSF, path: str, device) -> None:
    """``--resume`` that tolerates a changed per-block latent width.

    When realization codes are added (prior_dim = D + R) the old prior's
    transformer trunk still applies -- only the latent-facing projections
    change shape. Copy those tensors slice-wise: old latent columns/rows keep
    their weights, the new code columns/rows keep their fresh init (out_proj
    rows are zero-init, so the resumed prior initially predicts r = 0 and
    learns it from there). ``self_cond_proj`` reads [z_t ; self_cond], so its
    two column halves are remapped separately. Latent stats are recomputed
    by the trainer regardless.
    """
    sd = torch.load(path, map_location="cpu", weights_only=False)["model"]
    own = model.state_dict()
    D_new = model.prior_dim
    merged, resized = {}, []
    for k, v in sd.items():
        if not k.startswith("denoiser.") or k not in own:
            continue
        tgt = own[k]
        if v.shape == tgt.shape:
            merged[k] = v
            continue
        if k.endswith("latent_mean") or k.endswith("latent_std"):
            continue   # recomputed from the new latents anyway
        out = tgt.clone()
        if k.endswith("self_cond_proj.weight"):
            D_old = v.shape[0]
            out[:D_old, :D_old] = v[:, :D_old]
            out[:D_old, D_new:D_new + D_old] = v[:, D_old:2 * D_old]
        else:
            sl = tuple(slice(0, min(a, b)) for a, b in zip(v.shape, tgt.shape))
            out[sl] = v[sl]
            if k.endswith("in_proj.1.weight") and tgt.shape[1] > v.shape[1]:
                # A widened bottleneck: the new bottleneck units start silent
                # so step 0 reproduces the old prior exactly on the old dims.
                out[:, v.shape[1]:] = 0.0
        merged[k] = out
        resized.append(k)
    missing, unexpected = model.load_state_dict(merged, strict=False)
    bad = [k for k in missing if k.startswith("denoiser.")
           and not (k.endswith("latent_mean") or k.endswith("latent_std"))]
    if bad or unexpected:
        raise SystemExit(f"resume mismatch: missing={bad[:3]} unexpected={list(unexpected)[:3]}")
    if dist.is_main():
        print(f"resumed denoiser weights from {path}"
              + (f" | slice-copied {len(resized)} resized tensors: {resized}"
                 if resized else ""))


@torch.no_grad()
def show_samples(model: PSF, tokenizer, device, args: argparse.Namespace) -> None:
    if args.samples <= 0:
        return
    tokens, lengths = model.generate(
        args.samples, device=device, temperature=args.temperature, top_p=args.top_p,
    )
    print("\n--- unconditional samples (z ~ prior -> text) ---")
    for i in range(tokens.shape[0]):
        print(f"  [{int(lengths[i]):4d}] {decode(tokenizer, tokens[i])[:300]!r}")


def main() -> None:
    args = parse_args()
    cfg = build_config(args)
    set_seed(cfg.train.seed)
    device = dist.setup() or get_device(cfg.train.device)
    torch.set_float32_matmul_precision("high")   # TF32 matmuls on Ampere+

    tokenizer = load_tokenizer(cfg)
    train_loader, val_loader = make_dataloaders(cfg)
    if dist.is_main():
        print(
            f"data: {len(train_loader.dataset)} train docs | vocab={cfg.data.vocab_size} | "
            f"L={cfg.max_seq_len} K={cfg.num_blocks} S={cfg.block_span} | "
            f"global batch={cfg.train.batch_size * dist.world_size()}"
        )
    if args.rollout_bank:
        from psf.utils.data import MixedBatchLoader, make_rollout_loader

        rollout_loader = make_rollout_loader(args.rollout_bank, cfg)
        if dist.is_main():
            print(f"rollout bank: {len(rollout_loader.dataset)} docs from "
                  f"{args.rollout_bank} | mix p={args.rollout_mix_p} "
                  f"(real data stays the anchor)")
        train_loader = MixedBatchLoader(train_loader, rollout_loader,
                                        args.rollout_mix_p, seed=cfg.train.seed)

    model = PSF(cfg).to(device)
    # Load encoder + decoder only: the AE checkpoint's denoiser is untrained
    # by construction, and its shape need not match this stage's denoiser.
    ckpt = torch.load(args.ae_ckpt, map_location="cpu", weights_only=False)
    ae_sd = {k: v for k, v in ckpt["model"].items()
             if not k.startswith("denoiser.")}
    missing, unexpected = model.load_state_dict(ae_sd, strict=False)
    non_denoiser_missing = [k for k in missing if not k.startswith("denoiser.")]
    if non_denoiser_missing or unexpected:
        raise SystemExit(f"AE checkpoint mismatch: missing={non_denoiser_missing[:3]} "
                         f"unexpected={list(unexpected)[:3]}")
    if dist.is_main():
        print(f"loaded frozen autoencoder from {args.ae_ckpt}")
    if args.resume:
        load_denoiser_resume(model, args.resume, device)
    set_seed(cfg.train.seed + dist.rank())   # decorrelate per-rank flow noise

    trainer = PriorTrainer(model, cfg, train_loader, val_loader, device)
    trainer.train()
    if dist.is_main():
        print("\n--- denoiser probe (x0 MSE at fixed times, lower is better) ---")
        print("  " + " ".join(f"{k}={v:.4f}" for k, v in trainer.latent_mse().items()))
        show_samples(model, tokenizer, device, args)
    dist.barrier()
    dist.cleanup()


if __name__ == "__main__":
    main()
