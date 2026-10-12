"""Configuration for every PSF component."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class DataConfig:
    """Pre-tokenized HF dataset; see scripts/download_data.py."""

    source: Literal["hf", "toy"] = "hf"   # "toy" = built-in corpus, no download
    dataset_path: str = "data/openwebtext-t5"
    tokenizer_name: str = "google-t5/t5-small"
    vocab_size: int = 32100          # refreshed from the tokenizer when it loads
    pad_token_id: int = 0            # T5: pad=0, eos=1
    eos_token_id: int = 1
    bob_token_id: int = 0            # begin-of-block; reuses the pad slot
    num_workers: int = 4
    val_frac: float = 0.10           # random held-out fraction when there is no val split
    val_max_docs: int = 10_000       # absolute cap on the held-out set regardless
                                      # of val_frac -- diagnostic probes (recon,
                                      # plateau ruler, fdyn) only ever draw a few
                                      # hundred/thousand docs, so 10% of a ~9.7M-doc
                                      # pool (~970k) is far more than needed and
                                      # costs real training signal for nothing.
                                      # The held-out set is a fixed-seed permutation
                                      # prefix, so shrinking the cap only returns
                                      # documents to training -- every past and
                                      # future checkpoint's held-out docs stay a
                                      # subset of the largest cap ever used, so
                                      # cross-checkpoint comparisons remain valid.
    split_seed: int = 0              # seed for the train/val permutation
    toy_train_docs: int = 20000      # source == "toy" only
    toy_val_docs: int = 512
    toy_seed: int = 0


@dataclass
class EncoderConfig:
    """Transformer encoder with hybrid (token-position + block-index) RoPE."""

    depth: int = 6
    num_heads: int = 8          # latent_dim / num_heads must be divisible by 4
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    rope_theta: float = 10000.0
    t5_features: bool = False   # frozen block-causal T5 features as encoder input


@dataclass
class DecoderConfig:
    """AR decoder: K parallel spans of S tokens, cross-attending to z."""

    depth: int = 6
    num_heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    rope_theta: float = 10000.0
    causal_latents: bool = False   # block k cross-attends latents 1..k only (prefix variant)
    null_latent: bool = False      # learned always-visible sink latent in cross-attn
    latent_window: int = 0         # >0 (with causal_latents): block k reads S_{k-w+1..k} only
    boundary_tokens: bool = False  # block k also conditions on b_{k+1}, block k+1's first token
    boundary_drop: float = 0.1     # P(hide b_{k+1} behind the learned no-target embedding)
    boundary_window: int = 1       # W first tokens of block k+1 given to block k (1 = classic bt)
    stitch: bool = False           # seam-stitch (fill-in-the-middle) mode: adds two marker
                                   # embeddings; the decoder can rewrite a block's first /
                                   # last tokens given the neighbour's real tokens + own body
    prev_window: bool = False      # self-attention also reads the previous block's
                                   # realized tokens (position-synchronous: query at
                                   # position p sees prev positions <= p; identical
                                   # mask at train and decode)
    wave_lead: int = 0             # >0: wavefront decoding -- block k starts `lead`
                                   # steps after block k-1, so position p of block k
                                   # reads the previous block's realized tokens up to
                                   # position p+lead (same prev-block channel as
                                   # prev_window, diagonal mask; serial depth becomes
                                   # S + (K-1)*lead). The landing target of block k
                                   # (block k+1's first token) only exists from
                                   # position lead+1 on, and is gated accordingly.
    lex_bag: bool = False          # cross-block LEXICAL channel: position n of block k also
                                   # reads the mean token embedding of the content tokens
                                   # (id >= lex_bag_min_id) the OTHER blocks have realized at
                                   # positions < n (position-synchronous: exactly what the
                                   # parallel decode has available). Zero-init projection.
                                   # Motivation: topic persistence is capped by
                                   # independent per-block sampling, not plan capacity; the
                                   # decode-time logit bonus (--lex-bonus) reaches real-text
                                   # persistence but costs PPL -- this is its trained form.
    lex_bag_drop: float = 0.15     # training: P(hide the bag for a block)
    lex_gate: bool = False         # learned form of the decode-time --lex-bonus: content tokens
                                   # some OTHER block has already realized get
                                   # + lex_scale * sigmoid(gate(h)) on their logits, in teacher
                                   # forcing and generation alike (same code path). Zero-init
                                   # scale. Motivation: the fixed bonus reaches real-text
                                   # topic persistence but pays +13~15% iso-entropy PPL because
                                   # it cannot judge WHEN reuse is appropriate; the mean-embedding
                                   # bag (lex_bag) trains fine but carries no token identity.
    lex_bag_min_id: int = 2000     # content-token filter (T5 ids are ~frequency-ordered)
    prompt_attn: bool = False      # prompt-aware writer: every layer also cross-attends to the
                                   # token embeddings of a given prompt (the first blocks of the document);
                                   # zero-init output projection, so a warm start is identical without a prompt
    prompt_max_len: int = 256      # longest prompt (tokens) the learned prompt positions cover
    wave_drop: float = 0.15        # training: P(hide the previous block entirely) so
                                   # the block also learns to write without neighbour
                                   # text (the ae-pw lesson: neighbour text is an
                                   # error channel at generation time -- hedge it)
    landing_state: bool = False    # block k also conditions on s_{k+1}, the prior's
                                   # plan for what block k should realize (D_bridge)
    landing_dim: int = 0           # >0: bottleneck the landing state to this many
                                   # dims (capacity knob); 0 = full latent_dim
    landing_drop: float = 0.15     # P(hide the landing behind the learned no-landing)
    landing_attn: bool = False     # D_bridge via attention: the causal latent mask
                                   # opens one step (block k reads s_{k+1} at RoPE
                                   # relative position +1 -- no extra channel needed)
    landing_residual: bool = False # inject (s_{k+1} - s_k) instead of s_{k+1}:
                                   # cross-attn keeps s_{<=k} as the main signal,
                                   # the landing only supplies the increment
    landing_cfg: bool = False      # trained landing CFG: reconstruct through
                                   # l_off + lambda*(l_on - l_off) with random
                                   # lambda (told to the model); inference knob
    noise_level_cond: bool = False # memory entries carry a noise-level embedding
                                   # (diffusion-forcing conditioning)
    cross_before_self: bool = False  # each layer: cross-attend to z first,
                                      # then self-attend within the block
                                      # (default: self then cross)
    abs_pos_emb: bool = False        # learnable per-block absolute position
                                      # bias added to z, on top of the
                                      # otherwise fully relative (RoPE)
                                      # position scheme (zero-init)
    rc_memory: bool = True           # realization code (Config.rc_dim > 0):
                                      # besides feeding r_k into block k's own
                                      # token stream, also add a projection of
                                      # r_k to memory slot k so neighbours
                                      # (block k+1 ...) can read what block k
                                      # actually realized
    rc_lookahead: bool = False       # block k also reads r_{k+1}, the planned
                                      # code of the NEXT block (zero for the last
                                      # block): a planned landing that comes from
                                      # the same joint prior sample, so the block
                                      # can end into what its neighbour will open
                                      # with (targets the block-end cost)


@dataclass
class DenoiserConfig:
    """Flow-matching denoiser over the K latents."""

    hidden_size: int = 768
    depth: int = 12
    num_heads: int = 12
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    bottleneck_dim: int = 128   # latent_dim -> bottleneck -> hidden_size
    num_time_tokens: int = 4    # learned time-prefix tokens
    time_embed_dim: int = 256
    self_conditioning: bool = True
    rope_theta: float = 10000.0
    num_cfg_tokens: int = 0     # >0: learned guidance-scale prefix tokens (SC-CFG)


@dataclass
class FlowConfig:
    """Rectified flow, x0-parameterized: z_t = t*z0 + (1-t)*eps, t: 0 -> 1."""

    time_sampler: Literal["uniform", "logit_normal"] = "logit_normal"
    logit_normal_mean: float = 0.0
    logit_normal_std: float = 1.0
    self_cond_prob: float = 0.5   # probability of a self-conditioning pass
    num_sampling_steps: int = 50
    # --- improved objective (ELF-style), all off by default ---------------- #
    objective: Literal["x0", "v"] = "x0"   # "v": x0 MSE weighted by 1/(1-t)^2
    t_eps: float = 0.05                    # clamp for the 1/(1-t) factors
    sc_cfg: bool = False                   # train-time self-conditioning CFG
    cfg_min: float = 0.5                   # log-uniform guidance-scale range
    cfg_max: float = 5.0
    ctx_drop_prob: float = 0.0             # prefix-drop CFG: P(train a row context-free)
    # --- diffusion forcing over the K blocks (experimental, off by default) --- #
    diffusion_forcing: bool = False        # train with an independent time per block
    df_mix_prob: float = 0.3               # P(a row keeps one shared time: plain flow rows)
    df_ordered_prob: float = 0.0           # P(a row's per-block times are sorted descending:
                                           # earlier blocks cleaner, an ltr-shaped reveal)
    df_clean_prob: float = 0.0             # per-block P(real clean context: t = 1, out of the
                                           # loss) in independent-time rows -- the staggered
                                           # sampler shows the net many finished (t = 1) blocks
    # Inference: staggered ("pyramid") reveal. Block k's clock runs as
    # t_k = clip(t * (1 + lam) - lam * off_k, 0, 1) with off_k in [0, 1], so blocks
    # with a small offset finish early and later blocks are denoised while their
    # neighbours are already nearly clean -- a latent-space semi-AR at no decoder
    # cost. lam = 0 recovers the plain shared-time sampler.
    df_stagger: float = 0.0
    df_order: Literal["ltr", "random"] = "random"
    # On-policy context (df_rollout): in a share of DF rows, a random
    # subset of blocks is replaced by the denoiser's OWN clean prediction (time 1)
    # and dropped from the loss, so the remaining blocks learn to denoise next to
    # the imperfect neighbours they will actually see under a staggered reveal.
    df_rollout_prob: float = 0.0           # P(row gets on-policy context)
    df_rollout_frac: float = 0.5           # share of blocks turned into context


@dataclass
class TrainConfig:
    batch_size: int = 32
    ae_steps: int = 50_000
    prior_steps: int = 100_000
    lr: float = 3e-4
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.95)
    grad_clip: float = 1.0
    warmup_steps: int = 1000
    # Modern-recipe knobs (all off by default = legacy behavior):
    lr_schedule: Literal["const", "cosine"] = "const"
    lr_min_ratio: float = 0.1        # cosine floor as a fraction of peak lr
    wd_exclude: bool = False         # no weight decay on norms / biases (ndim < 2)
    muon: bool = False               # Muon for 2-D hidden matrices, AdamW for the rest
    muon_lr: float = 0.02
    muon_momentum: float = 0.95
    log_every: int = 100
    eval_every: int = 2000
    eval_batches: int = 8
    latent_stats_batches: int = 20   # batches used to standardize z before the prior
    # --- improved-variant training tricks (ELF-style), off by default ------ #
    ema_decay: float = 0.0           # >0: keep an EMA of the trained weights
    ema_warmup: bool = True          # ramp decay 0 -> ema_decay over ema_warmup_steps: forget the init fast
    ema_warmup_steps: int = 2000
    bf16: bool = False               # bfloat16 autocast for the training forward
    ae_decoder_noise: bool = False   # stage 1: decoder reads noised latents
    ae_noise_p_mean: float = 0.8     # lambda = sigmoid(N(mean, std^2)) mixing
    ae_noise_p_std: float = 0.8
    # Latent auxiliary supervision (trainer-owned heads, not saved in the ckpt):
    ae_aux_bow: float = 0.0          # weight: S_k predicts the prefix bag-of-words
    ae_aux_plan: float = 0.0         # weight: S_k predicts block k's bag-of-words
    ae_aux_ks: int = 8               # block indices sampled per step for the aux loss
    ae_aux_pos_weight: float = 50.0  # BCE up-weight for the rare positive types
    # Diffusion forcing: per-entry flow-style corruption of the latents, with
    # the level fed to the decoder (needs decoder.noise_level_cond).
    ae_diffusion_forcing: bool = False
    ae_df_clean_p: float = 0.3       # P(an entry stays clean, t=1)
    ae_df_tmin: float = 0.3          # otherwise t ~ U(tmin, 1)
    ae_consistency: float = 0.0      # weight: distill clean-conditioned logits
                                     # into the noised-conditioned decode
    ae_cons_blocks: int = 8          # blocks sampled per step for that KL term
    ae_landing_lam_max: float = 2.0  # landing CFG: lambda ~ U(0, this) per sample
    # Self-realized prefix training (srp): realize the blocks from the true
    # latents each step (no grad), re-encode, and swap latent rows to their
    # realized versions with probability srp_p -- trains the decoder under the
    # state mismatch it meets at one-shot generation. 0 = off (legacy).
    # --- seam stitch (route A, 2026-09-04): FIM loss on block seams ------- #
    ae_stitch_w: float = 0.0         # weight of the stitch CE (0 = off); one extra decoder
                                     # pass per step (start- or end-stitch, alternating)
    ae_stitch_start: int = 3         # block-initial tokens rewritten given prev tail + own body
    ae_stitch_end: int = 2           # block-final tokens rewritten given own body + next head
    ae_stitch_prev: int = 3          # previous-block tail tokens shown as prefix
    ae_stitch_next: int = 3          # next-block head tokens shown as suffix
    ae_stitch_blocks: int = 0        # >0: stitch loss on a random subset of this many blocks
                                     # per step (memory: the FIM pass is ~1.1x a main pass)
    ae_srp_p: float = 0.0
    ae_srp_temp: float = 0.65        # internal realization temperature
    ae_srp_top_p: float = 0.95
    # Graduated extension (srpg): simulate `ae_srp_rounds` chained
    # realize-then-re-encode passes (annealed toward greedy, like
    # refine_schedule) instead of one. Each block independently swaps to a
    # depth d in {0..rounds} with probability ae_srp_p spread over d>=1
    # (d=0 unchanged); the noise-level channel is told 1 - d/rounds, so the
    # model learns a graduated "how many refinement rounds deep is this
    # state" rather than a binary true/realized flag. rounds=1 reduces
    # exactly to the original srp/srpl behaviour.
    ae_srp_rounds: int = 1
    # Prompt-aware training (decoder.prompt_attn): with probability ae_prompt_p a
    # row's first m blocks (m drawn from ae_prompt_blocks) are handed to the
    # writer as prompt tokens; those blocks are excluded from the loss.
    ae_prompt_p: float = 0.0
    ae_prompt_blocks: str = "1,2"
    # Realization code (Config.rc_dim > 0): KL(q(r_k | x_<=k) || N(0, I)) is
    # added to the stage-1 loss as beta * KL_per_block / S (per-token units,
    # beta = 1 is the exact ELBO). kl_target > 0 adapts beta multiplicatively
    # to hit that many nats per block (the semantically meaningful knob: how
    # much of a block's realization the code pins down); warmup ramps beta
    # from 0 so the zero-init decoder projections pick the code up first.
    ae_rc_beta: float = 1.0
    ae_rc_kl_target: float = 0.0
    ae_rc_warmup: int = 0
    ae_rc_noise: bool = True         # mix r with N(0, I) noise under the same
                                     # lambda as the decoder-noise on s (the
                                     # prior will err on r just as it errs on s)
    prior_rc_source: Literal["sample", "mean", "zero"] = "sample"   # stage-2 target for r
    ae_rc_zero: bool = False         # ablation: the writer reads zero codes in
                                     # training -- the code channel exists but carries nothing
    output_dir: str = "runs/psf"
    device: str = "auto"             # "auto" | "cpu" | "cuda"
    seed: int = 0


@dataclass
class Config:
    max_seq_len: int = 1024     # L, the fixed canvas (must be a multiple of K)
    num_blocks: int = 64        # K latents -> S = 16 tokens per block
    latent_dim: int = 512       # D, shared by encoder output / decoder / denoiser
    encoder_type: Literal["block", "prefix"] = "block"   # prefix: S_k = f(B_{<k})
    prefix_unshifted: bool = False  # control: block k reads the inclusive
                                    # content state c_k = f(B_{<=k}) instead of s_k = c_{k-1}
    num_global_latents: int = 0  # G document-level latents prepended to the K blocks
    num_latent_slots: int = 1    # m latent vectors per block: addressable capacity
                                 # without extra seams (slots share the block's
                                 # RoPE position and causal-mask index)
    rc_dim: int = 0              # realization code: a KL-regularized Gaussian
                                 # code r_k of block k's own content (the
                                 # explicit shared noise epsilon_k). Decoder
                                 # reads (s_<=k, r_k); the prior samples
                                 # [s_k ; r_k] jointly, so one draw commits to
                                 # both the plan and its realization. 0 = off.

    data: DataConfig = field(default_factory=DataConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)
    denoiser: DenoiserConfig = field(default_factory=DenoiserConfig)
    flow: FlowConfig = field(default_factory=FlowConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def __post_init__(self) -> None:
        if self.max_seq_len % self.num_blocks != 0:
            raise ValueError(
                f"max_seq_len={self.max_seq_len} must be divisible by "
                f"num_blocks={self.num_blocks}"
            )
        head_dim = self.latent_dim // self.encoder.num_heads
        if head_dim % 4 != 0:
            raise ValueError(
                "hybrid encoder RoPE needs latent_dim/num_heads divisible by 4; "
                f"got {head_dim}"
            )

    @property
    def block_span(self) -> int:
        """S = L / K, the fixed number of tokens per block."""
        return self.max_seq_len // self.num_blocks


def toy_config() -> Config:
    """Small end-to-end preset: synthetic corpus, ~15 min on a laptop CPU."""
    cfg = Config(max_seq_len=128, num_blocks=16, latent_dim=128)
    cfg.data.source = "toy"
    cfg.encoder.depth = cfg.decoder.depth = 4
    cfg.encoder.num_heads = cfg.decoder.num_heads = 4
    cfg.denoiser.hidden_size, cfg.denoiser.depth, cfg.denoiser.num_heads = 192, 6, 6
    cfg.denoiser.bottleneck_dim, cfg.denoiser.num_time_tokens = 64, 2
    cfg.denoiser.time_embed_dim = 128
    cfg.train.ae_steps, cfg.train.prior_steps = 1000, 3000
    cfg.train.warmup_steps, cfg.train.log_every, cfg.train.eval_every = 100, 50, 250
    cfg.train.output_dir = "runs/toy"
    return cfg
