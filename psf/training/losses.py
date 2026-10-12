"""Stage objectives.

Both stages expose the same signature, ``(model, batch) -> (loss, metrics)``,
so the two trainers differ only in what they optimize, not in how they step.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from psf.diffusion.flow import flow_loss, flow_loss_improved
from psf.models.psf import PSF, refine_schedule
from psf.utils.data import Batch


def masked_cross_entropy(logits: Tensor, targets: Tensor, mask: Tensor) -> Tensor:
    """Mean CE over the real tokens of [B, K, S] blocks."""
    V = logits.shape[-1]
    ce = F.cross_entropy(
        logits.reshape(-1, V), targets.reshape(-1), reduction="none"
    ).reshape(targets.shape)
    return (ce * mask).sum() / mask.sum().clamp_min(1.0)


def masked_accuracy(logits: Tensor, targets: Tensor, mask: Tensor) -> float:
    correct = (logits.argmax(dim=-1) == targets) & mask.bool()
    return float(correct.sum() / mask.sum().clamp_min(1.0))


def _bag(tokens: Tensor, mask: Tensor, vocab: int, pad_id: int) -> Tensor:
    """[B, V] multi-hot of the real tokens in each row."""
    ids = torch.where(mask, tokens, torch.full_like(tokens, pad_id))
    lab = torch.zeros(tokens.shape[0], vocab, device=tokens.device)
    lab.scatter_(1, ids, 1.0)
    lab[:, pad_id] = 0.0
    return lab


def latent_aux_loss(model: PSF, aux: nn.ModuleDict, enc_feats: Tensor,
                    batch: Batch) -> tuple[Tensor, dict]:
    """BCE from S_k to prefix / next-block bags-of-words on sampled ks.

    The heads live on the trainer, not the model: they exist to feed the
    encoder a direct gradient ("store what the text said / where it goes"),
    and are dropped at checkpoint time. S_k is read as the concat of its m
    slots. k is 0-based here: prefix = tokens[:, :k*S], plan = block k.

    Two guards keep the pressure on *content*: types present in more than
    half the batch rows (function words) are masked out of the loss -- the
    head's bias would absorb them without ever touching S -- and the rare
    positives are up-weighted (``ae_aux_pos_weight``), otherwise their
    gradient share is ~|bag|/V and the encoder never feels it.
    """
    cfg = model.config
    K, S, V = cfg.num_blocks, cfg.block_span, cfg.data.vocab_size
    M, G = cfg.num_latent_slots, cfg.num_global_latents
    pad = cfg.data.pad_token_id
    B = batch.tokens.shape[0]
    feats = enc_feats[:, G:].reshape(B, K, M * cfg.latent_dim)
    n_k = min(cfg.train.ae_aux_ks, K)
    ks = torch.randperm(K, device=feats.device)[:n_k]
    total = feats.new_zeros(())
    metrics: dict[str, float] = {}
    pos_w = feats.new_tensor(cfg.train.ae_aux_pos_weight)
    for name, head in aux.items():
        w = cfg.train.ae_aux_bow if name == "bow" else cfg.train.ae_aux_plan
        loss_k = feats.new_zeros(())
        for k in ks.tolist():
            if name == "bow":
                lab = _bag(batch.tokens[:, : k * S], batch.mask[:, : k * S], V, pad)
            else:
                lab = _bag(batch.tokens[:, k * S: (k + 1) * S],
                           batch.mask[:, k * S: (k + 1) * S], V, pad)
            content = (lab.mean(dim=0) <= 0.5).to(lab.dtype)
            loss_k = loss_k + F.binary_cross_entropy_with_logits(
                head(feats[:, k]), lab,
                weight=content.unsqueeze(0), pos_weight=pos_w,
            )
        loss_k = loss_k / max(n_k, 1)
        total = total + w * loss_k
        metrics[f"aux_{name}"] = float(loss_k.detach())
    return total, metrics


def stitch_loss(model: PSF, z: Tensor, levels: Optional[Tensor], rc: Optional[Tensor],
                targets: Tensor, block_mask: Tensor,
                nf: Optional[Tensor]) -> tuple[Tensor, dict]:
    """Seam-stitch (FIM) objective on real text, one configuration per call
    (start- or end-stitch, chosen at random so the cost is one extra decoder
    pass per step):

    * start: rewrite block k's first ``ae_stitch_start`` tokens given the
      previous block's last ``ae_stitch_prev`` real tokens and block k's own
      remaining body (block 0 is excluded: no previous tail);
    * end:   rewrite block k's last ``ae_stitch_end`` tokens given its own
      body and the next block's first ``ae_stitch_next`` real tokens (last
      block excluded).

    Conditioning is the pass's own (z, levels, rc) and landing targets, so
    the stitch mode is trained under the same plan distribution the main
    decode sees."""
    tcfg = model.config.train
    dec = model.decoder
    B, K, S = targets.shape
    P, Q = tcfg.ae_stitch_prev, tcfg.ae_stitch_next
    M0, M1 = tcfg.ae_stitch_start, tcfg.ae_stitch_end
    do_start = M0 > 0 and (M1 <= 0 or bool(torch.rand(()) < 0.5))
    if do_start:
        absent = torch.full((B, 1, P), -1, dtype=torch.long, device=targets.device)
        prefix = torch.cat([absent, targets[:, :-1, S - P:]], dim=1)
        suffix = targets[:, :, M0:]
        middle = targets[:, :, :M0]
        mask = block_mask[:, :, :M0].clone()
        mask[:, 0] = 0.0
        name = "stitch_start"
    else:
        absent = torch.full((B, 1, Q), -1, dtype=torch.long, device=targets.device)
        prefix = targets[:, :, : S - M1]
        suffix = torch.cat([targets[:, 1:, :Q], absent], dim=1)
        middle = targets[:, :, S - M1:]
        mask = block_mask[:, :, S - M1:].clone()
        mask[:, -1] = 0.0
        name = "stitch_end"
    bid = None
    if 0 < tcfg.ae_stitch_blocks < K:
        # Memory knob: a random block subset per step (cross-attention still
        # sees the full latent set through branch decoding).
        bid = torch.randperm(K, device=targets.device)[: tcfg.ae_stitch_blocks].sort().values
        prefix, suffix, middle, mask = prefix[:, bid], suffix[:, bid], middle[:, bid], mask[:, bid]
        if nf is not None:
            nf = nf[:, bid]
    logits = dec.forward_stitch(z, prefix, suffix, middle[:, :, :-1], rc,
                                next_first=nf, levels=levels, branch_ids=bid)
    loss = masked_cross_entropy(logits, middle, mask)
    return loss, {name: float(loss.detach())}


def ae_loss(model: PSF, batch: Batch,
            aux: Optional[nn.ModuleDict] = None,
            rc_beta: float = 0.0) -> tuple[Tensor, dict]:
    """Stage 1: reconstruct the text through the K block latents.

    With ``train.ae_decoder_noise`` (ELF's decoder branch) the decoder reads
    lambda * z + (1 - lambda) * n, lambda = sigmoid(N(p_mean, p_std^2)) per
    sample and n ~ N(mu_z, sigma_z^2) from batch stats — exactly the shape of
    a denormalized flow state, so decode is calibrated for imperfect priors.
    Eval passes (model.eval()) stay on clean latents.

    With ``Config.rc_dim`` > 0 every branch also hands the decoder a
    realization code r_k ~ q(r_k | x_<=k) (reparameterized sample from the
    encoder's head) and the loss adds ``rc_beta * KL / S`` (nats per block,
    scaled to per-token units). Under decoder-noise the code is mixed with
    N(0, I) noise by the same lambda as the states -- the prior's error on r
    arrives in-distribution just like its error on s.
    """
    tcfg = model.config.train
    cons = None
    enc = model.encode(batch.tokens, batch.mask)
    inputs, targets = model.block_view(batch.tokens)
    nf = model.boundary_targets(targets)

    # Prompt-aware training: with probability ae_prompt_p a row hands its
    # first m blocks to the writer as prompt tokens (cross-attention channel); those
    # blocks are then excluded from the loss. Rows without a prompt see only the
    # writer's learned null entry. The srp realizations below stay prompt-free.
    pk: dict = {}
    prompt_blocks = None
    if (getattr(tcfg, "ae_prompt_p", 0.0) > 0 and model.training
            and getattr(model.decoder, "prompt_attn", False)):
        B_, K_, S_ = targets.shape
        choices = torch.tensor([int(x) for x in str(tcfg.ae_prompt_blocks).split(",")],
                               device=targets.device)
        m = choices[torch.randint(len(choices), (B_,), device=targets.device)]
        m = torch.where(torch.rand(B_, device=targets.device) < tcfg.ae_prompt_p, m,
                        torch.zeros_like(m))
        P_max = min(int(choices.max()) * S_, model.decoder.prompt_pos.num_embeddings)
        pos = torch.arange(P_max, device=targets.device)
        pmask = (pos[None, :] < (m * S_)[:, None]) & batch.mask[:, :P_max]
        pk = dict(prompt=batch.tokens[:, :P_max], prompt_mask=pmask)
        prompt_blocks = torch.arange(K_, device=targets.device)[None, :] < m[:, None]   # [B, K]

    rc = rc_kl = None
    if model.config.rc_dim > 0:
        rc = model.sample_rc(enc)                                 # [B, K, R]
        if getattr(tcfg, "ae_rc_zero", False):
            rc = torch.zeros_like(rc)                             # code channel carries nothing (ablation)
        kl_blocks = model.rc_kl(enc)                              # [B, K]
        m_rc = enc.rc_valid.to(kl_blocks.dtype)
        rc_kl = (kl_blocks * m_rc).sum() / m_rc.sum().clamp_min(1.0)   # nats/block

    def noised_rc(w: Optional[Tensor]) -> Optional[Tensor]:
        if rc is None or w is None or not tcfg.ae_rc_noise or not model.training:
            return rc
        return w * rc + (1.0 - w) * torch.randn_like(rc)

    def latent_noise_w(B: int, device) -> Tensor:
        return torch.sigmoid(
            torch.randn(B, 1, 1, device=device) * tcfg.ae_noise_p_std
            + tcfg.ae_noise_p_mean
        )

    # (z, levels, rc) the decoder actually read -- the seam-stitch loss reuses
    # them so the stitch mode trains under the same state/label distribution.
    stitch_ctx = (enc.block_feats, None, rc)
    if model.config.decoder.landing_cfg and model.training:
        # Trained landing CFG: reconstruct through lambda-combined logits
        # (l_off + lambda * (l_on - l_off)) with lambda ~ U(0, max) told to
        # the model, so the inference-time lambda knob is in-distribution.
        # Costs two decoder forwards per step.
        z = enc.block_feats
        noise_w = None
        if tcfg.ae_decoder_noise:
            stats = z[enc.block_valid].detach()
            mu, sigma = stats.mean(dim=0), stats.std(dim=0)
            noise_w = latent_noise_w(z.shape[0], z.device)
            z = noise_w * z + (1.0 - noise_w) * (mu + sigma * torch.randn_like(z))
        rc_in = noised_rc(noise_w)
        lam = torch.rand(z.shape[0], device=z.device) * tcfg.ae_landing_lam_max
        l_on = model.decoder(z, inputs, next_first=nf, landing_lam=lam, rc=rc_in, **pk)
        l_off = model.decoder(z, inputs, next_first=nf, landing_off=True,
                              landing_lam=lam, rc=rc_in, **pk)
        logits = l_off + lam.view(-1, 1, 1, 1) * (l_on - l_off)
    elif tcfg.ae_srp_p > 0 and model.training:
        # Self-realized prefix training (srp / graduated srpg): chain
        # `ae_srp_rounds` realize-then-re-encode passes from the true
        # latents (no grad, eval-mode decoder = test-time behavior), annealed
        # toward greedy like refine_schedule so deeper links approximate a
        # converged fixed point rather than just another independent sample
        # (constant-temperature iteration cannot contract -- see
        # refine_schedule's docstring). Each block independently swaps to
        # chain depth d in {0..R} with probability srp_p spread uniformly
        # over d>=1 (d=0 = untouched true state). The decoder then
        # reconstructs the *real* text under states at that depth -- the
        # exact condition it meets at one-shot generation (depth R, every
        # block) and at intermediate refinement rounds (depth r, only the
        # rewritten blocks). Boundary tokens stay teacher-forced from the
        # real text (the seam anchor is not corrupted). R=1 reduces exactly
        # to the original binary srp/srpl.
        #
        # Realization codes are NOT swapped: the realization itself is
        # written under the true code (what the prior's code will ask for
        # at one-shot), only the *states* describing the prefix become the
        # realized variant -- the code says "write this", the states say
        # "the past looks like this"; hedging applies to the latter only.
        R = max(1, tcfg.ae_srp_rounds)
        z = enc.block_feats
        chain = [z]
        with torch.no_grad():
            dec_training = model.decoder.training
            model.decoder.eval()
            sched = refine_schedule(tcfg.ae_srp_temp, tcfg.ae_srp_top_p,
                                    R - 1, anneal=True)
            cur = z
            for temp, top_p in sched:
                realized = model.decoder.generate(
                    cur, model.config.data.bob_token_id, temp, top_p, rc=rc,
                    use_wave=False)   # plain parallel realization: cheap, states only
                cur = model.encode(*model.recanvas(realized)).block_feats
                chain.append(cur)
            if dec_training:
                model.decoder.train()
        B, J, _ = z.shape
        z_base = z
        noise_w = None
        if tcfg.ae_decoder_noise:
            # Compose with the lineage's decoder-noise: depth-0 rows keep the
            # Gaussian prior-error simulation; realized rows (depth>=1) stay
            # exact re-encodings (refinement-time states arrive uncorrupted).
            stats = z[enc.block_valid].detach()
            mu, sigma = stats.mean(dim=0), stats.std(dim=0)
            noise_w = latent_noise_w(B, z.device)
            z_base = noise_w * z + (1.0 - noise_w) * (
                mu + sigma * torch.randn_like(z))
        chain[0] = z_base
        stacked = torch.stack(chain, dim=0).detach()          # [R+1, B, J, D]
        probs = torch.full((R + 1,), tcfg.ae_srp_p / R, device=z.device)
        probs[0] = 1.0 - tcfg.ae_srp_p
        depth = torch.multinomial(probs, B * J, replacement=True).reshape(B, J)
        z_mixed = stacked.gather(
            0, depth.view(1, B, J, 1).expand(1, B, J, stacked.shape[-1])
        ).squeeze(0)
        levels = None
        if model.config.decoder.noise_level_cond:
            # Provenance label through the (zero-init) level channel: 1 =
            # true summary read exactly (matches refinement states at
            # inference); graduated down to 0 = fully realized/converged
            # (matches the prior's plan state at one-shot, fed --z-level 0).
            levels = 1.0 - depth.to(z.dtype) / R
        rc_used = noised_rc(noise_w)
        stitch_ctx = (z_mixed, levels, rc_used)
        bag_tokens = None
        if getattr(model.decoder, "lex_bag", False) or getattr(model.decoder, "lex_gate", False):
            # Lexical bag / gate from the REALIZED streams for realized rows: at one-shot
            # decoding the other blocks' words are the model's own, not the
            # document's -- train the channel on what it will actually read.
            bob = torch.full_like(realized[:, :, :1], model.config.data.bob_token_id)
            realized_inputs = torch.cat([bob, realized[:, :, :-1]], dim=2)
            bag_tokens = torch.where((depth > 0).unsqueeze(-1), realized_inputs, inputs)
        logits = model.decoder(z_mixed, inputs, next_first=nf, levels=levels,
                               rc=rc_used, bag_tokens=bag_tokens, **pk)
    elif tcfg.ae_diffusion_forcing and model.training:
        # Diffusion forcing: every latent entry gets its own corruption level
        # t_j (the flow's own interpolation family), and the decoder is told
        # the levels -- so inference-time prior error arrives in-distribution
        # and the decoder knows how much to trust each entry.
        z = enc.block_feats
        stats = z[enc.block_valid].detach()
        mu, sigma = stats.mean(dim=0), stats.std(dim=0)
        B, J, _ = z.shape
        t = tcfg.ae_df_tmin + (1.0 - tcfg.ae_df_tmin) * torch.rand(B, J, 1, device=z.device)
        t = torch.where(torch.rand(B, J, 1, device=z.device) < tcfg.ae_df_clean_p,
                        torch.ones_like(t), t)
        noise = mu + sigma * torch.randn_like(z)
        logits = model.decoder(t * z + (1.0 - t) * noise, inputs,
                               next_first=nf, levels=t.squeeze(-1), rc=rc, **pk)
        if tcfg.ae_consistency > 0:
            # Distill the clean-conditioned prediction into the noised decode
            # on a block subsample (branch decoding keeps the memory small).
            nb = min(tcfg.ae_cons_blocks, model.config.num_blocks)
            bid = torch.randperm(model.config.num_blocks, device=z.device)[:nb]
            with torch.no_grad():
                cl = model.decoder(z, inputs[:, bid], branch_ids=bid,
                                   next_first=None if nf is None else nf[:, bid],
                                   rc=rc)
                p_clean = torch.softmax(cl.float(), dim=-1)
            lp = torch.log_softmax(logits[:, bid].float(), dim=-1)
            m = batch.mask.reshape(targets.shape)[:, bid].float()
            cons = -((p_clean * lp).sum(-1) * m).sum() / m.sum().clamp(min=1.0)
    elif tcfg.ae_decoder_noise and model.training:
        z = enc.block_feats
        stats = z[enc.block_valid].detach()
        mu, sigma = stats.mean(dim=0), stats.std(dim=0)
        lam = latent_noise_w(z.shape[0], z.device)
        noise = mu + sigma * torch.randn_like(z)
        logits = model.decoder(lam * z + (1.0 - lam) * noise, inputs,
                               next_first=nf, rc=noised_rc(lam), **pk)
    else:
        logits = model.decoder(enc.block_feats, inputs, next_first=nf, rc=rc, **pk)
    block_mask = batch.mask.reshape(targets.shape).to(logits.dtype)
    if prompt_blocks is not None:
        block_mask = block_mask * (~prompt_blocks).unsqueeze(-1).to(block_mask.dtype)
    loss = masked_cross_entropy(logits, targets, block_mask)
    metrics = {
        "recon": float(loss.detach()),
        "acc": masked_accuracy(logits, targets, block_mask),
    }
    if rc_kl is not None:
        # nats/block -> per-token units so beta = 1 is the exact ELBO.
        loss = loss + rc_beta * rc_kl / model.config.block_span
        metrics["kl"] = float(rc_kl.detach())
        metrics["beta"] = float(rc_beta)
    if cons is not None:
        loss = loss + tcfg.ae_consistency * cons
        metrics["cons"] = float(cons.detach())
    if tcfg.ae_stitch_w > 0 and model.training and model.config.decoder.stitch:
        st_loss, st_metrics = stitch_loss(model, *stitch_ctx, targets, block_mask, nf)
        loss = loss + tcfg.ae_stitch_w * st_loss
        metrics.update(st_metrics)
    if aux is not None and len(aux) > 0:
        aux_total, aux_metrics = latent_aux_loss(model, aux, enc.block_feats, batch)
        loss = loss + aux_total
        metrics.update(aux_metrics)
    return loss, metrics


def prior_loss(model: PSF, batch: Batch) -> tuple[Tensor, dict]:
    """Stage 2: rectified flow on the frozen encoder's standardized latents."""
    model.encoder.eval()   # frozen: never let the shared train() flag enable dropout
    with torch.no_grad():
        enc = model.encode(batch.tokens, batch.mask)
        z0 = model.denoiser.normalize(model.prior_latents(enc))
    fcfg = model.config.flow
    if fcfg.objective != "x0" or fcfg.sc_cfg or fcfg.ctx_drop_prob > 0:
        loss = flow_loss_improved(model.denoiser, z0, fcfg)
    else:
        loss = flow_loss(model.denoiser, z0, fcfg)
    return loss, {"flow": float(loss.detach())}
