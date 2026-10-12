"""The PSF model: prefix encoder + block decoder (stages I-II) and a flow prior over the states (stage III)."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from psf.config import Config
from psf.models.decoder import Decoder
from psf.models.denoiser import Denoiser
from psf.models.encoder import Encoder, EncoderOutput
from psf.diffusion.flow import sample as flow_sample


def refine_schedule(temperature: float, top_p: float, rounds: int,
                    anneal: bool) -> list[tuple[float, float]]:
    """(temperature, top_p) for each of the ``rounds + 1`` decode passes.

    Annealing drives the last pass to greedy: resampling at temperature t
    re-injects H[x | S] nats per token every round, so a constant-temperature
    iteration cannot contract onto a fixed point no matter how many rounds it
    gets -- it just wanders. Cooling to 0 lets it settle.
    """
    if rounds <= 0:
        return [(temperature, top_p)]
    if not anneal:
        return [(temperature, top_p)] * (rounds + 1)
    out = []
    for i in range(rounds + 1):
        t = temperature * (rounds - i) / rounds
        out.append((t, top_p if t > 0.0 else 1.0))
    return out


class PSF(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config
        V = config.data.vocab_size
        if V <= 0:
            raise ValueError("config.data.vocab_size must be positive")
        D, K, S = config.latent_dim, config.num_blocks, config.block_span
        G = config.num_global_latents
        M = config.num_latent_slots
        R = config.rc_dim
        if R > 0 and (config.encoder_type != "prefix" or M > 1 or G > 0):
            raise ValueError("rc_dim > 0 needs the prefix encoder, one slot, no globals")
        # The prior's per-block vector: [s_k ; r_k] when realization codes are on.
        self.prior_dim = D + R

        if config.encoder_type == "prefix":
            from psf.models.encoder_prefix import PrefixEncoder

            self.encoder = PrefixEncoder(
                V, D, config.encoder, K, S, num_global=G,
                t5_name=(config.data.tokenizer_name
                         if config.encoder.t5_features else None),
                num_slots=M, rc_dim=R, unshifted=config.prefix_unshifted,
            )
        elif M > 1:
            raise ValueError("num_latent_slots > 1 requires encoder_type='prefix'")
        elif G > 0:
            raise ValueError("num_global_latents > 0 requires encoder_type='prefix'")
        else:
            self.encoder = Encoder(V, D, config.encoder, K, S)
        self.decoder = Decoder(V, D, config.decoder, K, S, num_global=G, num_slots=M,
                               rc_dim=R)
        self.denoiser = Denoiser(self.prior_dim, K * M + G, config.denoiser)

    # --- realization code helpers ------------------------------------------- #
    def sample_rc(self, enc: EncoderOutput, source: str = "sample") -> Optional[Tensor]:
        """[B, K, R] code from the posterior q(r_k | x_<=k); None when off."""
        if self.config.rc_dim == 0:
            return None
        if source == "mean":
            return enc.rc_mu
        if source == "zero":
            # states-only prior: the flow's code target is the constant 0,
            # so the writer is fed zero codes at inference without changing any shape
            return torch.zeros_like(enc.rc_mu)
        return enc.rc_mu + torch.exp(0.5 * enc.rc_logvar) * torch.randn_like(enc.rc_mu)

    def rc_kl(self, enc: EncoderOutput) -> Tensor:
        """[B, K] KL(q(r_k) || N(0, I)) in nats per block."""
        mu, lv = enc.rc_mu.float(), enc.rc_logvar.float()
        return 0.5 * (mu.pow(2) + lv.exp() - 1.0 - lv).sum(dim=-1)

    def prior_latents(self, enc: EncoderOutput, source: Optional[str] = None) -> Tensor:
        """What stage 2 models: [B, K, D] states, or [B, K, D+R] with the code."""
        if self.config.rc_dim == 0:
            return enc.block_feats
        rc = self.sample_rc(enc, source or self.config.train.prior_rc_source)
        return torch.cat([enc.block_feats, rc.to(enc.block_feats.dtype)], dim=-1)

    def split_latents(self, z: Tensor) -> tuple[Tensor, Optional[Tensor]]:
        """Undo ``prior_latents``: (states [B, K, D], codes [B, K, R] or None)."""
        if self.config.rc_dim == 0:
            return z, None
        D = self.config.latent_dim
        return z[..., :D], z[..., D:]

    # --- stage 1: reconstruction ------------------------------------------- #
    def encode(self, tokens: Tensor, mask: Optional[Tensor] = None) -> EncoderOutput:
        return self.encoder(tokens, mask)

    def block_view(self, tokens: Tensor) -> tuple[Tensor, Tensor]:
        """Split a flat [B, L] sequence into decoder inputs and targets.

        Returns ``(inputs, targets)`` of shape [B, K, S]; each block is fed
        ``[BOB, x_0, ..., x_{S-2}]`` and supervised on ``[x_0, ..., x_{S-1}]``.
        """
        B = tokens.shape[0]
        targets = tokens.reshape(B, self.config.num_blocks, self.config.block_span)
        bob = torch.full_like(targets[:, :, :1], self.config.data.bob_token_id)
        return torch.cat([bob, targets[:, :, :-1]], dim=2), targets

    def boundary_targets(self, targets: Tensor) -> Optional[Tensor]:
        """[B, K, W] landing window: block k+1's first W tokens, -1 for the last block.

        Teacher forcing takes it straight from the data, never from the
        encoder -- so the decoder learns to aim at a real next word rather
        than to read block k's own content back out of a latent.
        """
        if not self.config.decoder.boundary_tokens:
            return None
        W = self.config.decoder.boundary_window
        nxt = targets[:, 1:, :W]
        none = torch.full_like(nxt[:, :1], -1)
        return torch.cat([nxt, none], dim=1)

    def reconstruct(self, tokens: Tensor, mask: Optional[Tensor] = None,
                    rc_source: str = "sample") -> tuple[Tensor, Tensor, EncoderOutput]:
        """Teacher-forced pass: returns (logits [B,K,S,V], targets [B,K,S], enc)."""
        enc = self.encode(tokens, mask)
        inputs, targets = self.block_view(tokens)
        logits = self.decoder(enc.block_feats, inputs,
                              next_first=self.boundary_targets(targets),
                              rc=self.sample_rc(enc, rc_source))
        return logits, targets, enc

    # --- decoding ----------------------------------------------------------- #
    @torch.no_grad()
    def decode_text(self, z: Tensor, temperature: float = 0.0,
                    top_p: float = 1.0, rc: Optional[Tensor] = None) -> tuple[Tensor, Tensor]:
        """Free-running decode from latents. Returns (tokens [B, L], lengths [B])."""
        block_tokens = self.decoder.generate(
            z, self.config.data.bob_token_id, temperature, top_p, rc=rc
        )
        return self.truncate_at_eos(block_tokens)

    def recanvas(self, block_tokens: Tensor) -> tuple[Tensor, Tensor]:
        """Self-generated blocks reshaped into an encoder input [B, L] + mask.

        Keeps everything through the first EOS and masks off the rest, which
        is exactly what ``collate`` hands the encoder for a short document --
        so re-reading our own text stays in distribution.
        """
        B = block_tokens.shape[0]
        flat = block_tokens.reshape(B, -1)
        L = flat.shape[1]
        is_eos = flat == self.config.data.eos_token_id
        end = torch.where(
            is_eos.any(dim=1),
            is_eos.long().argmax(dim=1) + 1,
            torch.full((B,), L, dtype=torch.long, device=flat.device),
        )
        mask = torch.arange(L, device=flat.device)[None, :] < end[:, None]
        pad = torch.full_like(flat, self.config.data.pad_token_id)
        return torch.where(mask, flat, pad), mask

    @torch.no_grad()
    def block_scores(self, z: Tensor, blocks: Tensor,
                     rc: Optional[Tensor] = None) -> Tensor:
        """[B, K] mean log-prob of the current text under the current latents.

        Teacher-forces ``blocks`` through the decoder (with realized landing
        targets) -- one parallel forward, ~1/S of a decode pass. Low score =
        this block disagrees with the plan and its own context; the natural
        rewrite priority for confidence-ordered partial commit.
        """
        bob = torch.full_like(blocks[:, :, :1], self.config.data.bob_token_id)
        inputs = torch.cat([bob, blocks[:, :, :-1]], dim=2)
        logits = self.decoder(z, inputs, next_first=self.boundary_targets(blocks), rc=rc)
        logp = torch.log_softmax(logits.float(), dim=-1)
        return logp.gather(-1, blocks.unsqueeze(-1)).squeeze(-1).mean(dim=2)

    @torch.no_grad()
    def refine_decode(self, z: Tensor, schedule: list[tuple[float, float]],
                      commit: Optional[list[float]] = None,
                      order: str = "random",
                      z_level: float = 1.0,
                      landing_lambda: float = 1.0,
                      n_cands: int = 1,
                      shared_noise: bool = False,
                      rc: Optional[Tensor] = None,
                      rc_source: str = "mean",
                      lex_bonus: float = 0.0, lex_min_id: int = 2000,
                      lex_post: bool = False,
                      no_repeat_ngram: int = 0, landing_off: bool = False,
                      rc_mode: str = "plan", rc_lookahead_off: bool = False) -> tuple[Tensor, Tensor]:
        """Decode, re-encode, decode again: iterate toward x = Dec(Enc(x)).

        Blocks are decoded in parallel from correct *marginals* p(x_k | S_<=k)
        but their joint is not: S came from the prior, so it describes some
        document, not the one the other blocks happen to write. Feeding the
        text back through the encoder closes that loop -- round r+1 conditions
        on a prefix summary of what was actually written in round r, and any
        fixed point is self-consistent by construction.

        Only the encoder makes latents here; the prior produced z once and
        never runs again. Cost is ``len(schedule) - 1`` extra
        (encoder fwd + block-parallel decode) passes.

        Args:
            z:        [B, G+K, D] initial latents, from the prior or the encoder.
            schedule: (temperature, top_p) per decode pass; see
                      ``refine_schedule``.
            commit:   per-round probability that a block accepts its rewrite
                      (Mask-Predict's partial commit; the last entry repeats).
                      Redecoding every block every round lets the sample walk
                      away from where the prior put it -- keeping a random
                      subset frozen bounds the drift while its neighbours
                      adapt to it. None means commit everything.
            order:    which blocks accept their rewrite each round.
                      "random": i.i.d. with probability ``commit`` (default).
                      "ltr": left-to-right wavefront -- round i freezes the
                      first (i-1)/R of the blocks and rewrites the rest, so a
                      block's final text is always conditioned on an already
                      final prefix (block-AR in R waves; ignores ``commit``).
                      "conf": rewrite the ``commit`` fraction of blocks whose
                      current text scores lowest under the current latents
                      (Mask-Predict's confidence order).
            rc:       [B, K, R] realization codes paired with ``z`` (from the
                      prior or the encoder); required when rc_dim > 0. Later
                      rounds re-derive them from the re-encoded text
                      (``rc_source``: posterior "mean" or a "sample").
        """
        blocks = None
        for i, (temperature, top_p) in enumerate(schedule):
            fresh = self.decoder.generate(
                z, self.config.data.bob_token_id, temperature, top_p,
                z_level=z_level if i == 0 else 1.0,
                landing_lambda=landing_lambda,
                shared_noise=shared_noise, rc=rc,
                lex_bonus=lex_bonus, lex_min_id=lex_min_id, lex_post=lex_post,
                no_repeat_ngram=no_repeat_ngram, landing_off=landing_off,
                rc_mode=rc_mode, rc_lookahead_off=rc_lookahead_off,
            )
            if n_cands > 1:
                # Best-of-n per block by the decoder's own likelihood: the
                # single biggest fixable per-block error is sampling luck.
                best_score = self.block_scores(z, fresh, rc=rc)
                for _ in range(n_cands - 1):
                    alt = self.decoder.generate(
                        z, self.config.data.bob_token_id, temperature, top_p,
                        z_level=z_level if i == 0 else 1.0,
                        landing_lambda=landing_lambda, rc=rc,
                    )
                    alt_score = self.block_scores(z, alt, rc=rc)
                    take = alt_score > best_score
                    fresh = torch.where(take.unsqueeze(-1), alt, fresh)
                    best_score = torch.where(take, alt_score, best_score)
            if blocks is not None and (commit or order != "random"):
                K = fresh.shape[1]
                if order == "ltr":
                    frontier = (i - 1) * K // max(len(schedule) - 1, 1)
                    redo = torch.arange(K, device=fresh.device)[None, :] >= frontier
                    redo = redo.expand(fresh.shape[0], K)
                elif order == "conf":
                    if not commit:
                        raise ValueError("order='conf' needs a commit list "
                                         "(the rewrite fraction per round)")
                    p = commit[min(i - 1, len(commit) - 1)]
                    scores = self.block_scores(z, blocks, rc=rc)
                    n_redo = max(1, round(p * K))
                    kth = scores.kthvalue(n_redo, dim=1, keepdim=True).values
                    redo = scores <= kth
                else:
                    p = commit[min(i - 1, len(commit) - 1)]
                    redo = torch.rand(fresh.shape[:2], device=fresh.device) < p
                fresh = torch.where(redo.unsqueeze(-1), fresh, blocks)
            blocks = fresh
            if i + 1 < len(schedule):
                enc = self.encode(*self.recanvas(blocks))
                z = enc.block_feats
                rc = self.sample_rc(enc, rc_source)
        return self.truncate_at_eos(blocks)

    # --- stage 2: sampling -------------------------------------------------- #
    @torch.no_grad()
    def generate(self, num_samples: int, device=None, temperature: float = 0.0,
                 top_p: float = 1.0, cfg_scale: float = 1.0,
                 sde_gamma: float = 0.0, noise_scale: float = 1.0,
                 schedule: Optional[list[tuple[float, float]]] = None,
                 commit: Optional[list[float]] = None,
                 order: str = "random",
                 z_level: float = 1.0,
                 landing_lambda: float = 1.0,
                 n_cands: int = 1,
                 shared_noise: bool = False,
                 lex_bonus: float = 0.0, lex_min_id: int = 2000,
                 lex_post: bool = False,
                 no_repeat_ngram: int = 0, landing_off: bool = False,
                 rc_mode: str = "plan", rc_lookahead_off: bool = False) -> tuple[Tensor, Tensor]:
        """z ~ p(z) then x ~ p(x | z). Returns (tokens [N, L], lengths [N]).

        A ``schedule`` longer than one pass turns on latent refinement (see
        ``refine_decode``) and overrides ``temperature``/``top_p``; the prior
        still runs exactly once either way.
        """
        was_training = self.training
        self.eval()
        device = device or next(self.parameters()).device

        z = flow_sample(
            self.denoiser, num_samples,
            self.config.num_blocks * self.config.num_latent_slots
            + self.config.num_global_latents,
            self.prior_dim, self.config.flow, device,
            cfg_scale=cfg_scale, sde_gamma=sde_gamma, noise_scale=noise_scale,
        )
        s, rc = self.split_latents(self.denoiser.denormalize(z))
        out = self.refine_decode(s, schedule or [(temperature, top_p)], commit, order,
                                 z_level=z_level, landing_lambda=landing_lambda,
                                 n_cands=n_cands, shared_noise=shared_noise, rc=rc,
                                 lex_bonus=lex_bonus, lex_min_id=lex_min_id,
                                 lex_post=lex_post,
                                 no_repeat_ngram=no_repeat_ngram, landing_off=landing_off,
                                 rc_mode=rc_mode, rc_lookahead_off=rc_lookahead_off)
        self.train(was_training)
        return out

    def attach_stitch_decoder(self, path: str) -> None:
        """Two-decoder inference: keep a seam-stitch-trained decoder (from another
        checkpoint) beside the main one. The one-shot pass keeps using
        ``self.decoder`` (untouched by the stitch fine-tune), only the stitch
        pass runs on the attached module."""
        import copy
        cfg = self.config
        dcfg = copy.deepcopy(cfg.decoder)
        dcfg.stitch = True
        dec = Decoder(cfg.data.vocab_size, cfg.latent_dim, dcfg, cfg.num_blocks, cfg.block_span,
                      num_global=cfg.num_global_latents, num_slots=cfg.num_latent_slots,
                      rc_dim=cfg.rc_dim)
        ckpt = torch.load(path, map_location="cpu")
        sd = {k[len("decoder."):]: v for k, v in ckpt["model"].items() if k.startswith("decoder.")}
        dec.load_state_dict(sd, strict=True)
        self.stitch_decoder = dec.to(next(self.parameters()).device).eval()

    @torch.no_grad()
    def generate_stitched(self, num_samples: int, device=None, temperature: float = 0.0,
                          top_p: float = 1.0, cfg_scale: float = 1.0,
                          sde_gamma: float = 0.0, noise_scale: float = 1.0,
                          z_level: float = 1.0, start_n: int = 3, end_n: int = 2,
                          prev_n: int = 3, next_n: int = 3,
                          stitch_temperature: Optional[float] = None,
                          redecode: int = 0) -> tuple[Tensor, Tensor]:
        """One-shot parallel decode followed by the seam-stitch pass
        (``Decoder.stitch``): start_n + end_n extra parallel decoder steps that
        rewrite every block boundary given its neighbours' actual text. With an
        attached stitch decoder the pass runs on it; ``redecode`` > 0 then lets
        the MAIN decoder re-decode the ``redecode`` positions right after the
        rewritten opening (all blocks in parallel, teacher-forcing the rest),
        so the junction between the new opening and the old body is smoothed."""
        was_training = self.training
        self.eval()
        device = device or next(self.parameters()).device
        z = flow_sample(
            self.denoiser, num_samples,
            self.config.num_blocks * self.config.num_latent_slots
            + self.config.num_global_latents,
            self.prior_dim, self.config.flow, device,
            cfg_scale=cfg_scale, sde_gamma=sde_gamma, noise_scale=noise_scale,
        )
        s, rc = self.split_latents(self.denoiser.denormalize(z))
        blocks = self.decoder.generate(s, self.config.data.bob_token_id, temperature,
                                       top_p, z_level=z_level, rc=rc)
        st_temp = temperature if stitch_temperature is None else stitch_temperature
        stitcher = getattr(self, "stitch_decoder", None) or self.decoder
        blocks = stitcher.stitch(s, blocks, rc, st_temp, top_p, z_level=z_level,
                                 start_n=start_n, end_n=end_n, prev_n=prev_n,
                                 next_n=next_n)
        if redecode > 0 and start_n > 0:
            force = blocks.clone()
            force[:, 1:, start_n:start_n + redecode] = -1
            blocks = self.decoder.generate(s, self.config.data.bob_token_id, temperature,
                                           top_p, z_level=z_level, rc=rc, force=force)
        out = self.truncate_at_eos(blocks)
        self.train(was_training)
        return out

    @torch.no_grad()
    def partial_denoise_generate(self, tokens: Tensor, mask: Tensor, t_start: float,
                                 device=None, temperature: float = 0.0,
                                 top_p: float = 1.0, cfg_scale: float = 1.0,
                                 sde_gamma: float = 0.0, noise_scale: float = 1.0,
                                 z_level: float = 1.0) -> tuple[Tensor, Tensor]:
        """SDEdit-style diagnostic: corrupt the REAL encoded state of a held-
        out document to time ``t_start`` (0 = pure noise, exactly recovering
        unconditional one-shot generation; 1 = the real state itself,
        exactly recovering recon -- no denoiser call at all) and run the
        prior over only the remaining [t_start, 1] stretch before decoding
        in parallel exactly like one-shot generation does.

        Sweeping t_start traces a continuous path between recon and full
        one-shot generation instead of only ever comparing those two
        endpoints -- isolates how much of the joint 64-block structure the
        prior can recover given a head start of real signal, rather than
        lumping "does the decoder read s_k right" and "can the prior
        jointly sample s_1..s_K from nothing" into the same two data
        points.
        """
        was_training = self.training
        self.eval()
        device = device or next(self.parameters()).device
        enc = self.encode(tokens, mask)
        z0_real = self.denoiser.normalize(self.prior_latents(enc))
        z = flow_sample(
            self.denoiser, tokens.shape[0],
            self.config.num_blocks * self.config.num_latent_slots
            + self.config.num_global_latents,
            self.prior_dim, self.config.flow, device,
            cfg_scale=cfg_scale, sde_gamma=sde_gamma, noise_scale=noise_scale,
            z0_real=z0_real, t_start=t_start,
        )
        s, rc = self.split_latents(self.denoiser.denormalize(z))
        out = self.refine_decode(s, [(temperature, top_p)], z_level=z_level, rc=rc)
        self.train(was_training)
        return out

    @torch.no_grad()
    def rollout_generate(self, num_samples: int, device=None, temperature: float = 0.0,
                         top_p: float = 1.0, prior_s1: bool = False,
                         cfg_scale: float = 1.0, sde_gamma: float = 0.0,
                         noise_scale: float = 1.0,
                         gt_tokens: Optional[Tensor] = None,
                         gt_mask: Optional[Tensor] = None,
                         gt_block1: Optional[Tensor] = None,
                         gt_block1_mask: Optional[Tensor] = None) -> tuple[Tensor, Tensor]:
        """Block-AR rollout: s1 = BOS latent, decode block k, re-encode, repeat.

        Every S_k encodes the *actually sampled* prefix, so the prior never
        enters and cross-block decoherence is eliminated by construction. The
        prefix encoder is required (S_1 is its learned empty-prefix latent);
        with temperature 0 the whole rollout is deterministic — one document.

        ``prior_s1=True`` swaps that fixed, shared-across-the-batch S_1 for a
        per-sample draw from the flow-matching prior instead -- a MARGINAL
        single-block sample; the joint 64-block sampling that the one-shot
        path needs (and that collapses) is never invoked. This is the only
        place the prior enters the rollout path, and it isolates whether the
        prior's collapse is specific to jointly sampling all K blocks, or
        already present in its single-block marginal.

        ``gt_tokens``/``gt_mask`` (real held-out documents, mutually exclusive
        with ``prior_s1``): source every S_k from the REAL document -- the
        causal prefix encoder gives the identical S_k whether computed from
        the whole document at once or incrementally, so this is exactly
        recon's states, just fed through the SEQUENTIAL block-AR decode loop
        below instead of recon's single parallel decode pass. Each block is
        still freshly sampled (scored under the judge, never read back as
        truth); nothing about a block's own sample feeds S_{k+1} here, since
        S is fixed by the real text regardless. Isolates "are the states
        right" from "does parallel vs. sequential decode scheduling itself
        matter" -- if this matches recon, the two decode paths are
        equivalent; if not, decode order has its own real effect.

        ``gt_block1`` (real block-1 tokens, [N, S], mutually exclusive with
        the above): a THIRD, different thing from either -- S_1 is still
        whatever the self-referential path would use, but block 1's CONTENT
        is the real text verbatim (never sampled) rather than something the
        model wrote itself, so S_2 = Enc(real block 1) is a genuinely
        grounded state instead of one built on the model's own first guess.
        From block 2 on this is exactly the plain self-referential rollout
        (sample, re-encode, repeat) -- isolates whether a real, concrete
        opening (as opposed to a cold BOS start) changes how well the
        model's OWN subsequent writing goes.

        Boundary tokens do not apply to the self-referential (default) path:
        block k+1's latent does not exist until block k has been written, so
        no landing target is available and every block decodes under the
        learned no-target embedding (which ``decoder.boundary_drop`` keeps in
        distribution). Same omission in the gt_tokens/gt_block1 paths, for a
        fair like-for-like comparison against recon (whose decode call also
        never supplies the true landing word).
        """
        from psf.models.decoder import sample_tokens

        if self.config.encoder_type != "prefix" or self.config.num_global_latents > 0:
            raise NotImplementedError("rollout needs the prefix encoder, no globals")
        if self.config.num_latent_slots > 1:
            raise NotImplementedError("rollout not implemented for multi-slot latents")
        was_training = self.training
        self.eval()
        device = device or next(self.parameters()).device
        cfg = self.config
        K, Sp, D = cfg.num_blocks, cfg.block_span, cfg.latent_dim
        R = cfg.rc_dim

        rc_gt = None
        if gt_tokens is not None:
            num_samples = gt_tokens.shape[0]
            # Causal encoder: one pass over the whole real document already
            # gives every S_k exactly what incremental re-encoding would --
            # no need to recompute it 64 times.
            enc_gt = self.encode(gt_tokens, gt_mask)
            z_gt = enc_gt.block_feats
            rc_gt = self.sample_rc(enc_gt)
        elif gt_block1 is not None:
            num_samples = gt_block1.shape[0]
        elif prior_s1:
            z0 = flow_sample(
                self.denoiser, num_samples, K, self.prior_dim, cfg.flow, device,
                cfg_scale=cfg_scale, sde_gamma=sde_gamma, noise_scale=noise_scale,
            )
            s_prior, rc_prior = self.split_latents(self.denoiser.denormalize(z0))
            s1 = s_prior[:, 0]

        canvas = torch.full((num_samples, K * Sp), cfg.data.pad_token_id,
                            dtype=torch.long, device=device)
        filled = torch.zeros(num_samples, K * Sp, dtype=torch.bool, device=device)
        # Realization codes: the block being written draws its code from the
        # prior marginal N(0, I) (what the KL term shapes q toward); blocks
        # already written get the posterior mean of their realized text on
        # every re-encode, so neighbours read what was actually realized.
        rc_draw = torch.randn(num_samples, K, R, device=device) if R > 0 else None
        if R > 0 and prior_s1 and gt_tokens is None and gt_block1 is None:
            rc_draw[:, 0] = rc_prior[:, 0]   # block 1's code comes with its prior state
        if gt_block1 is not None:
            # Seed block 1 with the real text verbatim -- nothing to sample,
            # nothing to decode; block 2's re-encode below will see this as
            # genuine (masked-)real content, same as any other filled block.
            canvas[:, :Sp] = gt_block1
            filled[:, :Sp] = (gt_block1_mask if gt_block1_mask is not None
                              else torch.ones_like(gt_block1, dtype=torch.bool))
        for k in range(K):
            if gt_block1 is not None and k == 0:
                continue   # already seeded above, no decode for this block
            rc = rc_draw
            if gt_tokens is not None:
                z = z_gt
                rc = rc_gt
            elif k == 0:
                # Empty canvas: S_1 is the learned BOS latent (or a prior draw).
                z = self.encoder.pad_latent.view(1, 1, D).expand(num_samples, K, D).clone()
                z[:, 0] = s1 if prior_s1 else self.encoder.bos_latent
            else:
                enc = self.encode(canvas, filled)
                z = enc.block_feats
                if R > 0:
                    rc = rc_draw.clone()
                    rc[:, :k] = enc.rc_mu[:, :k].to(rc.dtype)
            bid = torch.tensor([k], device=device)
            prefix = torch.full((num_samples, 1, 1), cfg.data.bob_token_id,
                                dtype=torch.long, device=device)
            for _ in range(Sp):
                logits = self.decoder(z, prefix, branch_ids=bid, rc=rc)[:, :, -1]
                nxt = sample_tokens(logits, temperature, top_p)
                prefix = torch.cat([prefix, nxt.unsqueeze(-1)], dim=2)
            lo = k * Sp
            canvas[:, lo:lo + Sp] = prefix[:, 0, 1:]
            filled[:, lo:lo + Sp] = True

        out = self.truncate_at_eos(canvas.reshape(num_samples, K, Sp))
        self.train(was_training)
        return out

    def truncate_at_eos(self, block_tokens: Tensor) -> tuple[Tensor, Tensor]:
        """Flatten [B, K, S] blocks and cut each row at its first EOS."""
        B = block_tokens.shape[0]
        flat = block_tokens.reshape(B, -1)
        L = flat.shape[1]
        is_eos = flat == self.config.data.eos_token_id
        length = torch.where(
            is_eos.any(dim=1),
            is_eos.long().argmax(dim=1),
            torch.full((B,), L, dtype=torch.long, device=flat.device),
        )
        keep = torch.arange(L, device=flat.device)[None, :] < length[:, None]
        pad = torch.full_like(flat, self.config.data.pad_token_id)
        return torch.where(keep, flat, pad), length
