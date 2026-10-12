"""Stage 1: train the encoder + decoder as an autoencoder.

    python -m scripts.download_data                  # once
    python -m scripts.train_ae --steps 50000 --output-dir runs/ae

Multi-GPU (batch size is per process):

    torchrun --standalone --nproc-per-node 8 -m scripts.train_ae ...
"""

from __future__ import annotations

import argparse

import torch

from psf.config import Config, toy_config
from psf.models.psf import PSF
from psf.training.trainer_ae import AutoencoderTrainer
from psf.utils import dist
from psf.utils.data import decode, load_tokenizer, make_dataloaders
from psf.utils.misc import get_device, load_checkpoint, set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PSF stage-1 autoencoder training")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--toy", action="store_true",
                   help="built-in synthetic corpus + small model; no download")
    p.add_argument("--dataset-path", default=None, help="pre-tokenized dataset dir")
    p.add_argument("--tokenizer", default=None)
    p.add_argument("--max-seq-len", type=int, default=None, help="L; multiple of K")
    p.add_argument("--num-blocks", type=int, default=None, help="K; L/K = compression ratio")
    p.add_argument("--latent-dim", type=int, default=None, help="D")
    p.add_argument("--latent-slots", type=int, default=None,
                   help="m latent vectors per block (addressable capacity, no extra seams)")
    p.add_argument("--decoder-depth", type=int, default=None,
                   help="decoder layers (default 6); a warm start loads whatever "
                        "layers the init checkpoint has, extra layers train fresh")
    p.add_argument("--prev-window", action="store_true",
                   help="decoder self-attention also reads the previous block's "
                        "realized tokens (position-synchronous sliding window)")
    p.add_argument("--wave-lead", type=int, default=None,
                   help="wavefront decoding: block k starts this many steps after block "
                        "k-1 and reads its realized tokens up to position p+lead "
                        "(0 = off; serial depth S + (K-1)*lead)")
    p.add_argument("--wave-drop", type=float, default=None,
                   help="with --wave-lead: P(hide the previous block) during training")
    p.add_argument("--lex-bag", action="store_true",
                   help="cross-block lexical bag channel: block k at position n reads the mean "
                        "embedding of the content tokens the other blocks realized at positions < n")
    p.add_argument("--lex-gate", action="store_true",
                   help="learned cross-block lexical bonus (gate on h x zero-init scale); see config")
    p.add_argument("--lex-bag-drop", type=float, default=None,
                   help="with --lex-bag: P(hide the bag for a block) during training")
    p.add_argument("--lr-schedule", default=None, choices=["const", "cosine"],
                   help="cosine decays to lr-min-ratio after warmup (default const)")
    p.add_argument("--wd-exclude", action="store_true",
                   help="no weight decay on norms and biases")
    p.add_argument("--scaled-init", action="store_true",
                   help="scale residual output projections by 1/sqrt(2*depth) "
                        "(GPT-2 style; applied before any --init-ckpt load)")
    p.add_argument("--muon", action="store_true",
                   help="Muon for hidden 2-D matrices, AdamW for the rest")
    p.add_argument("--muon-lr", type=float, default=None)
    p.add_argument("--encoder", default=None, choices=["block", "prefix"],
                   help="block: z_k = f(B_k); prefix: S_k = f(B_<k). Must match stage 2.")
    p.add_argument("--causal-latents", action="store_true",
                   help="decoder block k cross-attends latents 1..k only")
    p.add_argument("--prefix-unshifted", action="store_true",
                   help="control: block k reads the inclusive content state c_k = f(B_<=k) (prefix encoder only)")
    p.add_argument("--null-latent", action="store_true",
                   help="learned always-visible sink latent in decoder cross-attn")
    p.add_argument("--global-latents", type=int, default=None,
                   help="G document-level latents prepended to the K blocks")
    p.add_argument("--t5-features", action="store_true",
                   help="frozen block-causal T5 features as encoder input")
    p.add_argument("--latent-window", type=int, default=None,
                   help=">0 (with --causal-latents): block k reads S_{k-w+1..k} only")
    p.add_argument("--boundary-tokens", action="store_true",
                   help="block k also conditions on b_{k+1}, block k+1's first "
                        "token, so both sides of a seam read the same symbol")
    p.add_argument("--boundary-drop", type=float, default=None,
                   help="P(hide b_{k+1} behind the learned no-target embedding)")
    p.add_argument("--boundary-window", type=int, default=None,
                   help="W first tokens of block k+1 given to block k (1 = classic bt)")
    p.add_argument("--landing-state", action="store_true",
                   help="block k also conditions on s_{k+1} (D_bridge landing)")
    p.add_argument("--landing-dim", type=int, default=None,
                   help=">0: bottleneck the landing state to this many dims")
    p.add_argument("--landing-drop", type=float, default=None,
                   help="P(hide the landing behind the learned no-landing)")
    p.add_argument("--landing-attn", action="store_true",
                   help="D_bridge via attention: block k reads s_{k+1} at RoPE "
                        "relative position +1 (one-step mask relaxation)")
    p.add_argument("--landing-residual", action="store_true",
                   help="inject (s_{k+1} - s_k) instead of s_{k+1}")
    p.add_argument("--landing-cfg", action="store_true",
                   help="trained landing CFG: reconstruct through the "
                        "lambda-combined logits (2x decoder fwds per step)")
    p.add_argument("--level-cond", action="store_true",
                   help="decoder conditions on each latent's noise level")
    p.add_argument("--cross-before-self", action="store_true",
                   help="each decoder layer cross-attends to z before its "
                        "own causal self-attention (default: self then cross)")
    p.add_argument("--abs-pos-emb", action="store_true",
                   help="learnable per-block absolute position bias added "
                        "to z, on top of the relative-only RoPE scheme")
    p.add_argument("--diffusion-forcing", action="store_true",
                   help="stage 1: per-entry flow-style latent corruption "
                        "(pairs with --level-cond)")
    p.add_argument("--consistency", type=float, default=None,
                   help=">0: distill clean-conditioned logits into the "
                        "noised decode (needs --diffusion-forcing)")
    p.add_argument("--srp-p", type=float, default=None,
                   help=">0: self-realized prefix training -- swap each latent "
                        "row to the re-encoding of the model's own realization "
                        "with this probability")
    p.add_argument("--srp-temp", type=float, default=None,
                   help="temperature of the internal realization (default 0.65)")
    p.add_argument("--srp-top-p", type=float, default=None)
    p.add_argument("--srp-rounds", type=int, default=None,
                   help=">1: chain this many annealed realize+re-encode "
                        "rounds and label by graduated depth (srpg) instead "
                        "of the binary true/realized flag (default 1)")
    # Realization code (rc): a KL-regularized Gaussian code of each block's
    # own content, read by the decoder and sampled jointly with the states
    # by the prior -- an explicit shared noise epsilon_k.
    p.add_argument("--rc-dim", type=int, default=None,
                   help=">0: per-block realization code of this many dims")
    p.add_argument("--rc-beta", type=float, default=None,
                   help="KL weight in per-token units (1 = exact ELBO)")
    p.add_argument("--rc-kl-target", type=float, default=None,
                   help=">0: adapt beta to hit this many KL nats per block")
    p.add_argument("--rc-warmup", type=int, default=None,
                   help="steps over which beta ramps from 0")
    p.add_argument("--no-rc-noise", action="store_true",
                   help="do not mix the code with N(0,I) under decoder-noise")
    p.add_argument("--rc-zero", action="store_true",
                   help="ablation: feed zero codes to the writer in training (code channel off; "
                        "evaluate with --rc-mode zero)")
    p.add_argument("--prompt-attn", action="store_true",
                   help="prompt-aware writer: cross-attention to given prompt tokens in every layer")
    p.add_argument("--prompt-p", type=float, default=None,
                   help="P(a training row hands its first blocks to the writer as a prompt)")
    p.add_argument("--prompt-blocks", default=None,
                   help="comma list of how many blocks a prompt may span, e.g. 1,2")
    p.add_argument("--no-rc-memory", action="store_true",
                   help="feed r_k to block k's own stream only, not memory slot k")
    p.add_argument("--stitch-w", type=float, default=None,
                   help="seam-stitch (FIM) loss weight; >0 enables the decoder's stitch "
                        "mode (two marker embeddings) and one extra decoder pass per step")
    p.add_argument("--stitch-start", type=int, default=None, help="block-initial tokens rewritten (default 3)")
    p.add_argument("--stitch-end", type=int, default=None, help="block-final tokens rewritten (default 2)")
    p.add_argument("--stitch-prev", type=int, default=None, help="previous-block tail tokens shown (default 3)")
    p.add_argument("--stitch-next", type=int, default=None, help="next-block head tokens shown (default 3)")
    p.add_argument("--stitch-blocks", type=int, default=None,
                   help="stitch loss on a random subset of this many blocks per step (memory knob; default all)")
    p.add_argument("--rc-lookahead", action="store_true",
                   help="block k also reads the next block's code r_{k+1} (must match the checkpoint)")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--device", default=None, help="auto | cpu | cuda")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--init-ckpt", default=None, help="start from these weights")
    p.add_argument("--freeze-rc-head", action="store_true",
                   help="with --freeze-encoder: also freeze the realization-code head, so "
                        "the code posterior (and hence an existing prior) stays valid")
    p.add_argument("--freeze-encoder", action="store_true",
                   help="train the decoder only; S stays bit-identical to "
                        "--init-ckpt, so an existing prior remains valid")
    p.add_argument("--init-encoder-only", action="store_true",
                   help="with --init-ckpt: load encoder.* weights only, "
                        "leaving the decoder at random init -- for a fair "
                        "from-scratch decoder-architecture probe on top of "
                        "an already-pretrained (and, with --freeze-encoder, "
                        "frozen) encoder")
    p.add_argument("--show", type=int, default=4, help="#reconstructions to print at the end")
    p.add_argument("--rollout-bank", default=None,
                   help="path to a scripts.sample_rollout_bank.py bank; when "
                        "set, this fraction of training batches (--rollout-"
                        "mix-p) reconstruct rollout-generated ('gold', "
                        "sequential-quality) documents instead of real text "
                        "-- the real loader stays as an anchor")
    p.add_argument("--rollout-mix-p", type=float, default=0.5,
                   help="P(a training batch is drawn from --rollout-bank)")
    # Improved-variant training tricks (ELF-style).
    p.add_argument("--decoder-noise", action="store_true",
                   help="decoder reads noised latents (robust to imperfect priors)")
    p.add_argument("--noise-p-mean", type=float, default=None,
                   help="lambda = sigmoid(N(mean, std^2)) mixing weight")
    p.add_argument("--noise-p-std", type=float, default=None)
    p.add_argument("--ema-decay", type=float, default=None,
                   help=">0 keeps and saves an EMA of the weights")
    p.add_argument("--bf16", action="store_true", help="bfloat16 autocast training")
    p.add_argument("--aux-bow", type=float, default=None,
                   help=">0: S_k predicts the prefix bag-of-words (weight)")
    p.add_argument("--aux-plan", type=float, default=None,
                   help=">0: S_k predicts block k's bag-of-words (weight)")
    return p.parse_args()


def build_config(args: argparse.Namespace) -> Config:
    cfg = toy_config() if args.toy else Config()
    if args.output_dir is not None:
        cfg.train.output_dir = args.output_dir
    elif not args.toy:
        cfg.train.output_dir = "runs/ae"
    for target, name, value in [
        (cfg, "max_seq_len", args.max_seq_len),
        (cfg, "num_blocks", args.num_blocks),
        (cfg, "latent_dim", args.latent_dim),
        (cfg, "num_latent_slots", args.latent_slots),
        (cfg, "encoder_type", args.encoder),
        (cfg, "num_global_latents", args.global_latents),
        (cfg.data, "dataset_path", args.dataset_path),
        (cfg.data, "tokenizer_name", args.tokenizer),
        (cfg.train, "batch_size", args.batch_size),
        (cfg.train, "ae_steps", args.steps),
        (cfg.train, "lr", args.lr),
        (cfg.train, "device", args.device),
        (cfg.train, "seed", args.seed),
        (cfg.train, "ae_noise_p_mean", args.noise_p_mean),
        (cfg.train, "ae_noise_p_std", args.noise_p_std),
        (cfg.train, "ae_aux_bow", args.aux_bow),
        (cfg.train, "ae_aux_plan", args.aux_plan),
        (cfg.train, "lr_schedule", args.lr_schedule),
        (cfg.train, "muon_lr", args.muon_lr),
        (cfg.train, "ema_decay", args.ema_decay),
        (cfg.decoder, "depth", args.decoder_depth),
        (cfg.decoder, "latent_window", args.latent_window),
        (cfg.decoder, "boundary_drop", args.boundary_drop),
        (cfg.decoder, "boundary_window", args.boundary_window),
        (cfg.decoder, "landing_dim", args.landing_dim),
        (cfg.decoder, "landing_drop", args.landing_drop),
        (cfg.train, "ae_consistency", args.consistency),
        (cfg.train, "ae_srp_p", args.srp_p),
        (cfg.train, "ae_srp_temp", args.srp_temp),
        (cfg.train, "ae_srp_top_p", args.srp_top_p),
        (cfg.train, "ae_srp_rounds", args.srp_rounds),
        (cfg, "rc_dim", args.rc_dim),
        (cfg.train, "ae_rc_beta", args.rc_beta),
        (cfg.train, "ae_rc_kl_target", args.rc_kl_target),
        (cfg.train, "ae_rc_warmup", args.rc_warmup),
    ]:
        if value is not None:
            setattr(target, name, value)
    if args.no_rc_noise:
        cfg.train.ae_rc_noise = False
    if args.rc_zero:
        cfg.train.ae_rc_zero = True
        cfg.train.ae_rc_noise = False
    if args.prompt_attn:
        cfg.decoder.prompt_attn = True
    if args.prompt_p is not None:
        cfg.train.ae_prompt_p = args.prompt_p
    if args.prompt_blocks is not None:
        cfg.train.ae_prompt_blocks = args.prompt_blocks
    if args.no_rc_memory:
        cfg.decoder.rc_memory = False
    if args.rc_lookahead:
        cfg.decoder.rc_lookahead = True
    for name, value in [("ae_stitch_w", args.stitch_w), ("ae_stitch_start", args.stitch_start),
                        ("ae_stitch_end", args.stitch_end), ("ae_stitch_prev", args.stitch_prev),
                        ("ae_stitch_next", args.stitch_next), ("ae_stitch_blocks", args.stitch_blocks)]:
        if value is not None:
            setattr(cfg.train, name, value)
    if cfg.train.ae_stitch_w > 0:
        cfg.decoder.stitch = True
    if args.causal_latents:
        cfg.decoder.causal_latents = True
    if args.prefix_unshifted:
        cfg.prefix_unshifted = True
    if args.boundary_tokens:
        cfg.decoder.boundary_tokens = True
    if args.landing_state:
        cfg.decoder.landing_state = True
    if args.landing_attn:
        cfg.decoder.landing_attn = True
    if args.landing_residual:
        cfg.decoder.landing_residual = True
    if args.landing_cfg:
        cfg.decoder.landing_cfg = True
    if args.prev_window:
        cfg.decoder.prev_window = True
    if args.wave_lead is not None:
        cfg.decoder.wave_lead = args.wave_lead
    if args.wave_drop is not None:
        cfg.decoder.wave_drop = args.wave_drop
    if args.lex_bag:
        cfg.decoder.lex_bag = True
    if args.lex_gate:
        cfg.decoder.lex_gate = True
    if args.lex_bag_drop is not None:
        cfg.decoder.lex_bag_drop = args.lex_bag_drop
    if args.wd_exclude:
        cfg.train.wd_exclude = True
    if args.muon:
        cfg.train.muon = True
    if args.level_cond:
        cfg.decoder.noise_level_cond = True
    if args.cross_before_self:
        cfg.decoder.cross_before_self = True
    if args.abs_pos_emb:
        cfg.decoder.abs_pos_emb = True
    if args.diffusion_forcing:
        cfg.train.ae_diffusion_forcing = True
    if args.null_latent:
        cfg.decoder.null_latent = True
    if args.decoder_noise:
        cfg.train.ae_decoder_noise = True
    if args.bf16:
        cfg.train.bf16 = True
    if args.t5_features:
        cfg.encoder.t5_features = True
    cfg.__post_init__()   # re-check L % K and the hybrid-RoPE head split
    return cfg


@torch.no_grad()
def show_reconstructions(model: PSF, loader, tokenizer, device, n: int) -> None:
    """Encode real text, decode it back from the latents only (no teacher forcing)."""
    if loader is None or n <= 0:
        return
    model.eval()
    batch = next(iter(loader)).to(device)
    enc = model.encode(batch.tokens, batch.mask)
    tokens, _ = model.decode_text(enc.block_feats, rc=model.sample_rc(enc))
    print("\n--- reconstructions (encode -> z -> decode) ---")
    for i in range(min(n, tokens.shape[0])):
        print(f"  in : {decode(tokenizer, batch.tokens[i])[:200]!r}")
        print(f"  out: {decode(tokenizer, tokens[i])[:200]!r}")


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
    if args.scaled_init:
        # GPT-2-style: residual output projections scaled by 1/sqrt(2*depth).
        # Applied before any warm start, so loaded layers keep their weights
        # and only fresh layers carry the scaled init.
        s = (2 * cfg.decoder.depth) ** -0.5
        with torch.no_grad():
            for layer in model.decoder.layers:
                for lin in (layer.self_out, layer.cross_out, layer.mlp[-1]):
                    lin.weight.mul_(s)
        if dist.is_main():
            print(f"scaled residual init: x{s:.3f} on decoder projections")
    if args.init_ckpt and args.init_encoder_only:
        ckpt = torch.load(args.init_ckpt, map_location=device, weights_only=False)
        enc_sd = {k: v for k, v in ckpt["model"].items() if k.startswith("encoder.")}
        missing, unexpected = model.load_state_dict(enc_sd, strict=False)
        bad = [k for k in unexpected if k.startswith("encoder.")]
        if bad:
            raise SystemExit(f"encoder-only load mismatch: unexpected {bad[:3]}")
        if dist.is_main():
            print(f"initialized encoder only from {args.init_ckpt} "
                  f"(decoder stays random init)")
    elif args.init_ckpt:
        # Non-strict: warm-starting a variant (e.g. --boundary-tokens) leaves
        # its new parameters at init, and the load prints what it skipped.
        # Tensors whose shape changed (the AE checkpoint's untrained denoiser
        # when --rc-dim widens the prior's latent) are dropped: stage 1 never
        # touches the denoiser, so nothing is lost.
        ckpt = torch.load(args.init_ckpt, map_location=device, weights_only=False)
        own = model.state_dict()
        sd = {k: v for k, v in ckpt["model"].items()
              if k not in own or v.shape == own[k].shape}
        skipped = sorted(set(ckpt["model"]) - set(sd))
        result = model.load_state_dict(sd, strict=False)
        if dist.is_main():
            print(f"initialized from {args.init_ckpt}: "
                  f"{len(result.missing_keys)} missing, "
                  f"{len(result.unexpected_keys)} unexpected, "
                  f"{len(skipped)} shape-mismatched tensors dropped"
                  + (f" (e.g. {skipped[:2]})" if skipped else ""))
    if args.freeze_encoder:
        if not args.init_ckpt:
            raise SystemExit("--freeze-encoder is only meaningful with --init-ckpt")
        # The realization-code head is a new read-out on the frozen trunk:
        # it stays trainable (the states S themselves remain bit-identical).
        for name, p in model.encoder.named_parameters():
            p.requires_grad_(name.startswith("rc_head.") and not args.freeze_rc_head)
        for m in model.encoder.modules():
            if isinstance(m, torch.nn.Dropout):
                m.p = 0.0     # the trainer's model.train() would re-enable it
        model.encoder.eval()
        if dist.is_main():
            print("encoder frozen: latents unchanged, decoder-only training"
                  + (" (+ realization-code head)" if cfg.rc_dim > 0 else ""))
    set_seed(cfg.train.seed + dist.rank())   # decorrelate per-rank sampling noise

    AutoencoderTrainer(model, cfg, train_loader, val_loader, device).train()
    if dist.is_main():
        show_reconstructions(model, val_loader, tokenizer, device, args.show)
    dist.barrier()
    dist.cleanup()


if __name__ == "__main__":
    main()
