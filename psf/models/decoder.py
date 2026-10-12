"""Decoder p(x | z): K parallel AR spans that cross-attend to z_{1:K}.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from psf.config import DecoderConfig
from psf.models.rope import RoPE


class DecoderLayer(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float, dropout: float,
                 prompt_attn: bool = False) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.dropout_p = dropout
        # Prompt-aware writer: a third sub-layer that cross-attends to the
        # token embeddings of a given prompt. Zero-init output: identical to the
        # prompt-free model at warm start.
        self.p_out = None
        if prompt_attn:
            self.pnorm = nn.LayerNorm(dim)
            self.p_q = nn.Linear(dim, dim)
            self.p_kv = nn.Linear(dim, 2 * dim)
            self.p_out = nn.Linear(dim, dim)
            nn.init.zeros_(self.p_out.weight)
            nn.init.zeros_(self.p_out.bias)

        self.norm1 = nn.LayerNorm(dim)
        self.self_qkv = nn.Linear(dim, 3 * dim)
        self.self_out = nn.Linear(dim, dim)

        self.norm2 = nn.LayerNorm(dim)
        self.cross_q = nn.Linear(dim, dim)
        self.cross_kv = nn.Linear(dim, 2 * dim)
        self.cross_out = nn.Linear(dim, dim)

        hidden = int(dim * mlp_ratio)
        self.norm3 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, dim)
        )
        self.dropout = nn.Dropout(dropout)

    def _heads(self, x: Tensor) -> Tensor:
        # [B, N, D] -> [B, H, N, d]
        return x.unflatten(-1, (self.num_heads, self.head_dim)).transpose(1, 2)

    def _merge(self, x: Tensor) -> Tensor:
        # [B, H, N, d] -> [B, N, D]
        return x.transpose(1, 2).flatten(2)

    def _self_attention(self, x: Tensor, within_rope: RoPE,
                        prev_mask: Tensor | None = None) -> Tensor:
        """x: [B, K, N, D]; each block attends causally within itself.

        With ``prev_mask`` the keys/values extend to the previous block's
        hidden states (positions 0..N-1, own block at N..2N-1): a sliding
        window over *realized* neighbour tokens. The mask encodes exactly
        what the position-synchronous parallel decode has available (query
        at position p sees prev positions <= p), so one teacher-forced
        training pass matches decoding bit for bit.
        """
        B, K, N, D = x.shape
        if prev_mask is None:
            flat = x.reshape(B * K, N, D)
            q, k, v = (self._heads(p) for p in self.self_qkv(flat).chunk(3, dim=-1))
            positions = torch.arange(N, device=x.device)
            q, k = within_rope.rotate(q, positions), within_rope.rotate(k, positions)
            out = F.scaled_dot_product_attention(
                q, k, v, is_causal=True,
                dropout_p=self.dropout_p if self.training else 0.0,
            )
            return self.self_out(self._merge(out)).reshape(B, K, N, D)
        prev = torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], dim=1)
        kv = torch.cat([prev, x], dim=2).reshape(B * K, 2 * N, D)
        q_all, k, v = (self._heads(p) for p in self.self_qkv(kv).chunk(3, dim=-1))
        positions = torch.arange(2 * N, device=x.device)
        q = within_rope.rotate(q_all, positions)[:, :, N:]      # own positions only
        k = within_rope.rotate(k, positions)
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=prev_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
        )
        return self.self_out(self._merge(out)).reshape(B, K, N, D)

    def _cross_attention(self, x: Tensor, memory: Tensor, block_rope: RoPE,
                         causal_latents: bool, window: int = 0,
                         null: Tensor | None = None, num_global: int = 0,
                         branch_ids: Tensor | None = None,
                         num_slots: int = 1, lookahead: int = 0) -> Tensor:
        """x: [B, K, N, D]; memory: [B, G+J, D] shared by all K branches.

        ``num_global`` memory entries are document-level latents: visible to
        every branch, keyed at rotary position 0. ``window`` > 0 (with
        causal_latents) limits block k to latents k-window+1 .. k. ``null``
        is a learned always-visible sink latent; its key is not rotated (a
        sink should be position-free). With ``num_slots`` > 1, each block
        owns m consecutive memory entries that share its rotary position and
        causal-mask index.
        """
        B, K, N, D = x.shape
        J = memory.shape[1]
        # Fold the K branches into the query axis: one memory, K*N queries.
        q = self._heads(self.cross_q(x).reshape(B, K * N, D))
        k, v = (self._heads(p) for p in self.cross_kv(memory).chunk(2, dim=-1))
        if branch_ids is None:
            branch_ids = torch.arange(K, device=x.device)
        q_pos = branch_ids.repeat_interleave(N)                    # block index per query
        # Block index of each memory entry; negative for the global latents.
        blk = torch.div(torch.arange(J, device=x.device) - num_global,
                        num_slots, rounding_mode="floor")
        if num_global > 0:
            # Globals at position 0; block latents and queries shift up by 1.
            key_pos = torch.where(blk < 0, torch.zeros_like(blk), blk + 1)
            q = block_rope.rotate(q, q_pos + 1)
        else:
            key_pos = blk
            q = block_rope.rotate(q, q_pos)
        k = block_rope.rotate(k, key_pos)
        attn_mask = None
        if causal_latents:   # block k reads globals + latents 1..k (+lookahead)
            j = blk[None, :]
            attn_mask = j <= q_pos[:, None] + lookahead
            if window > 0:
                attn_mask &= j >= q_pos[:, None] - (window - 1)
            if num_global > 0:
                attn_mask |= j < 0
        if null is not None:
            # Project the [1, 1, D] parameter first, then expand to the batch:
            # passing a raw Parameter view into a module breaks ModuleTracker
            # (FlopCounterMode) and this order is cheaper anyway.
            nk, nv = (self._heads(p) for p in self.cross_kv(null).chunk(2, dim=-1))
            k = torch.cat([k, nk.expand(B, -1, -1, -1)], dim=2)
            v = torch.cat([v, nv.expand(B, -1, -1, -1)], dim=2)
            if attn_mask is not None:
                attn_mask = torch.cat(
                    [attn_mask, attn_mask.new_ones(attn_mask.shape[0], 1)], dim=1
                )
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
        )
        return self.cross_out(self._merge(out)).reshape(B, K, N, D)

    def forward(self, x: Tensor, memory: Tensor, within_rope: RoPE, block_rope: RoPE,
                causal_latents: bool = False, window: int = 0,
                null: Tensor | None = None, num_global: int = 0,
                branch_ids: Tensor | None = None, num_slots: int = 1,
                lookahead: int = 0, prev_mask: Tensor | None = None,
                cross_before_self: bool = False,
                prompt_mem: Tensor | None = None,
                prompt_mask: Tensor | None = None) -> Tensor:
        """Sub-layer order is a config knob: default self-then-cross (a block
        first organizes its own causal prefix, then reads the plan); with
        ``cross_before_self`` each block reads the plan first and its
        within-block causal pass runs on plan-conditioned hidden states.
        norm1 always feeds self-attention and norm2 always feeds
        cross-attention regardless of order, so warm-starting from an
        opposite-order checkpoint keeps every tensor's role and shape (the
        input *distribution* each norm sees still changes -- this is not a
        zero-init-style identity warm start).
        """
        def self_step(x: Tensor) -> Tensor:
            return x + self.dropout(self._self_attention(self.norm1(x), within_rope,
                                                          prev_mask))

        def cross_step(x: Tensor) -> Tensor:
            return x + self.dropout(
                self._cross_attention(self.norm2(x), memory, block_rope,
                                      causal_latents, window, null, num_global,
                                      branch_ids, num_slots, lookahead)
            )

        if cross_before_self:
            x = self_step(cross_step(x))
        else:
            x = cross_step(self_step(x))
        if self.p_out is not None and prompt_mem is not None:
            x = x + self.dropout(self._prompt_attention(self.pnorm(x), prompt_mem, prompt_mask))
        return x + self.dropout(self.mlp(self.norm3(x)))

    def _prompt_attention(self, x: Tensor, pmem: Tensor, pmask: Tensor | None) -> Tensor:
        """x: [B, K, N, D] queries from every block; pmem: [B, P, D] prompt token
        embeddings (position-embedded, null token first); pmask: [B, P] True on real
        entries. Position-free attention: the prompt is a bag of located tokens the
        writer may quote, its place in the document is already in the latents."""
        B, K, N, D = x.shape
        q = self._heads(self.p_q(x).reshape(B, K * N, D))
        k, v = (self._heads(p) for p in self.p_kv(pmem).chunk(2, dim=-1))
        mask = None if pmask is None else pmask[:, None, None, :]
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask,
                                             dropout_p=self.dropout_p if self.training else 0.0)
        return self.p_out(self._merge(out)).reshape(B, K, N, D)



def nucleus_mask(logits: Tensor, temperature: float, top_p: float) -> Tensor:
    """Bool [B, K, V]: the top-p set of softmax(logits / temperature) (same rule
    as ``sample_tokens``: keep while the cumulative mass BEFORE the token < p)."""
    probs = torch.softmax(logits.float() / temperature, dim=-1)
    sorted_probs, sorted_ids = probs.sort(dim=-1, descending=True)
    keep_sorted = (sorted_probs.cumsum(dim=-1) - sorted_probs) < top_p
    return torch.zeros_like(keep_sorted).scatter_(-1, sorted_ids, keep_sorted)


def sample_tokens(logits: Tensor, temperature: float, top_p: float,
                  shared_noise: bool = False) -> Tensor:
    """Greedy (temperature <= 0) or nucleus sampling over [B, K, V] logits.

    ``shared_noise``: the K blocks independently draw their own dice by
    default (each row of torch.multinomial gets fresh randomness) -- this is
    the diagnosed source of cross-block drift under an otherwise-accurate
    state: two blocks reading a coherent state can each
    validly realize a *different* filling of it, and nothing correlates
    which one they pick. With ``shared_noise``, every block at this step
    reuses the *same* vocabulary-indexed Gumbel perturbation (the exact
    reparameterization of categorical sampling: argmax(logp + Gumbel) ~
    Categorical(softmax(logits))), so if a specific word gets a big random
    boost, every block whose own distribution also ranks it plausible is
    pulled toward it -- correlating *which* realization gets picked without
    touching the state/plan channel at all (the noise is exogenous, drawn
    fresh, never predicted by any network, so it cannot carry the flow
    prior's sampling error the way a learned shared channel would).
    Training-free: a pure inference-time sampling-procedure change.
    """
    if temperature <= 0.0:
        return logits.argmax(dim=-1)
    probs = torch.softmax(logits.float() / temperature, dim=-1)
    B, K, V = probs.shape
    if not shared_noise:
        if top_p < 1.0:
            sorted_probs, sorted_ids = probs.sort(dim=-1, descending=True)
            keep = (sorted_probs.cumsum(dim=-1) - sorted_probs) < top_p
            sorted_probs = sorted_probs * keep
            sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True).clamp_min(1e-9)
            pick = torch.multinomial(sorted_probs.reshape(-1, V), 1).reshape(B, K, 1)
            return sorted_ids.gather(-1, pick).squeeze(-1)
        return torch.multinomial(probs.reshape(-1, V), 1).reshape(B, K)
    # One Gumbel draw per (batch, vocab) -- shared across the K axis, fresh
    # at every decode step (not shared across steps or across batch rows).
    u = torch.rand(B, V, device=logits.device).clamp_(1e-20, 1.0 - 1e-6)
    gumbel = -torch.log(-torch.log(u))                       # [B, V]
    if top_p < 1.0:
        sorted_probs, sorted_ids = probs.sort(dim=-1, descending=True)
        keep = (sorted_probs.cumsum(dim=-1) - sorted_probs) < top_p
        logp = torch.log(sorted_probs.clamp_min(1e-20)).masked_fill(~keep, float("-inf"))
        g_sorted = gumbel[:, None, :].expand(B, K, V).gather(-1, sorted_ids)
        pick = (logp + g_sorted).argmax(dim=-1, keepdim=True)
        return sorted_ids.gather(-1, pick).squeeze(-1)
    logp = torch.log(probs.clamp_min(1e-20))
    return (logp + gumbel[:, None, :]).argmax(dim=-1)



class Decoder(nn.Module):
    def __init__(self, vocab_size: int, latent_dim: int, config: DecoderConfig,
                 num_blocks: int, block_span: int, num_global: int = 0,
                 num_slots: int = 1, rc_dim: int = 0) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.num_blocks = num_blocks
        self.block_span = block_span
        self.num_global = num_global
        self.num_slots = num_slots
        self.rc_dim = rc_dim
        self.rc_in = self.rc_mem = self.rc_next = None
        if rc_dim > 0:
            # Realization code r_k: added to block k's own token stream (the
            # block reads what it is supposed to realize) and, with rc_memory,
            # projected onto memory slot k so later blocks read what block k
            # realized. Both zero-init: a warm start from a code-free
            # checkpoint behaves identically at step 0.
            self.rc_in = nn.Linear(rc_dim, latent_dim)
            nn.init.zeros_(self.rc_in.weight)
            nn.init.zeros_(self.rc_in.bias)
            if config.rc_memory:
                self.rc_mem = nn.Linear(rc_dim, latent_dim)
                nn.init.zeros_(self.rc_mem.weight)
                nn.init.zeros_(self.rc_mem.bias)
            if config.rc_lookahead:
                # Planned landing: block k reads the next block's code r_{k+1}
                # (zeros for the last block). Zero-init as well.
                self.rc_next = nn.Linear(rc_dim, latent_dim)
                nn.init.zeros_(self.rc_next.weight)
                nn.init.zeros_(self.rc_next.bias)

        self.token_emb = nn.Embedding(vocab_size, latent_dim)
        self.lex_bag = bool(getattr(config, "lex_bag", False))
        self.lex_bag_drop = float(getattr(config, "lex_bag_drop", 0.0))
        self.lex_bag_min_id = int(getattr(config, "lex_bag_min_id", 2000))
        if self.lex_bag:
            # Zero-init: a warm start from a bag-free checkpoint is identical at step 0.
            self.bag_proj = nn.Linear(latent_dim, latent_dim)
            nn.init.zeros_(self.bag_proj.weight)
            nn.init.zeros_(self.bag_proj.bias)
        self.lex_gate = bool(getattr(config, "lex_gate", False))
        self.lex_gate_mult = 1.0          # inference knob: scales the learned bonus
        if self.lex_gate:
            # Learned form of the decode-time --lex-bonus: logits of content tokens
            # some OTHER block has already realized get + lex_scale * sigmoid(gate(h)).
            # Scale zero-init -> identical to the warm-start checkpoint at step 0
            # (the gate itself starts at 0.5 and specialises once the scale moves).
            self.gate_proj = nn.Linear(latent_dim, 1)
            nn.init.zeros_(self.gate_proj.weight)
            nn.init.zeros_(self.gate_proj.bias)
            self.lex_scale = nn.Parameter(torch.zeros(()))
        self.causal_latents = config.causal_latents
        self.latent_window = config.latent_window
        self.null_latent = (
            nn.Parameter(torch.randn(1, 1, latent_dim) * 0.02)
            if config.null_latent else None
        )
        self.boundary_tokens = config.boundary_tokens
        self.boundary_drop = config.boundary_drop
        self.boundary_window = config.boundary_window
        if config.boundary_tokens:
            self.tgt_proj = nn.Linear(latent_dim, latent_dim)
            self.no_target = nn.Parameter(torch.randn(latent_dim) * 0.02)
        if config.boundary_tokens and config.boundary_window > 1:
            W = config.boundary_window
            # Fade-in: slot 0 starts at weight 1 and the wider slots silent, so
            # a warm start from a W=1 checkpoint behaves identically at init.
            self.win_gain = nn.Parameter(
                torch.cat([torch.ones(1), torch.zeros(W - 1)])
            )
            self.win_pos = nn.Parameter(torch.zeros(W, latent_dim))
        else:
            self.win_gain = None
            self.win_pos = None
        self.landing_state = config.landing_state
        self.landing_drop = config.landing_drop
        self.landing_attn = config.landing_attn
        self.landing_residual = config.landing_residual
        self.landing_cfg = config.landing_cfg
        self.noise_level_cond = config.noise_level_cond
        if config.landing_cfg:
            # Zero-init: lambda conditioning is silent at warm start.
            self.lam_emb = nn.Linear(1, latent_dim)
            nn.init.zeros_(self.lam_emb.weight)
            nn.init.zeros_(self.lam_emb.bias)
        if config.noise_level_cond:
            # Zero-init: an all-clean (t=1) pass matches a level-blind decoder
            # at init, so warm starts stay well-behaved.
            self.level_emb = nn.Linear(2, latent_dim)
            nn.init.zeros_(self.level_emb.weight)
            nn.init.zeros_(self.level_emb.bias)
        if config.abs_pos_emb:
            # Zero-init parameter directly (no linear layer needed): exactly
            # no-op at warm start, learns a per-block absolute bias on top of
            # the otherwise fully relative (RoPE) position scheme.
            self.block_pos_emb = nn.Parameter(torch.zeros(num_blocks, latent_dim))
        else:
            self.block_pos_emb = None
        if config.landing_state:
            d = config.landing_dim or latent_dim
            self.landing_in = (nn.Linear(latent_dim, d)
                               if config.landing_dim else nn.Identity())
            # Zero-init the output projection: a warm start from a checkpoint
            # without the landing channel behaves identically at step 0.
            self.landing_out = nn.Linear(d, latent_dim)
            nn.init.zeros_(self.landing_out.weight)
            nn.init.zeros_(self.landing_out.bias)
            self.no_landing = nn.Parameter(torch.randn(latent_dim) * 0.02)
        self.prev_window = config.prev_window
        self.wave_lead = int(getattr(config, "wave_lead", 0))
        self.wave_drop = float(getattr(config, "wave_drop", 0.0))
        self.cross_before_self = config.cross_before_self
        head_dim = latent_dim // config.num_heads
        use_prev = config.prev_window or self.wave_lead > 0
        span = 2 * block_span if use_prev else block_span
        self.stitch_mode = config.stitch
        if config.stitch:
            # Seam stitch: [prefix][SUF][suffix][MID][middle...] sequences are
            # longer than a block; the rotary table is parameter-free, so
            # sizing it up changes nothing for ordinary blocks.
            span = max(span, 2 * block_span + 8)
            self.suf_emb = nn.Parameter(torch.randn(latent_dim) * 0.02)
            self.mid_emb = nn.Parameter(torch.randn(latent_dim) * 0.02)
        self.within_rope = RoPE(head_dim, span, config.rope_theta)
        self.block_rope = RoPE(head_dim, num_blocks + 1, config.rope_theta)
        self.prompt_attn = bool(getattr(config, "prompt_attn", False))
        self.layers = nn.ModuleList(
            DecoderLayer(latent_dim, config.num_heads, config.mlp_ratio, config.dropout,
                         prompt_attn=self.prompt_attn)
            for _ in range(config.depth)
        )
        if self.prompt_attn:
            # learned absolute positions of the prompt tokens + a null entry every row
            # attends to (rows without a prompt see only the null entry)
            self.prompt_pos = nn.Embedding(int(getattr(config, "prompt_max_len", 256)), latent_dim)
            nn.init.normal_(self.prompt_pos.weight, std=0.02)
            self.prompt_null = nn.Parameter(torch.randn(latent_dim) * 0.02)
        self.final_norm = nn.LayerNorm(latent_dim)
        self.lm_head = nn.Linear(latent_dim, vocab_size)

    def _landing(self, next_first: Tensor | None, h: Tensor) -> Tensor:
        """Landing-target injection [B, K, N, D]; zero at within-block position 0.

        Block k is told the first ``boundary_window`` tokens of block k+1, so
        the two neighbours read the *same* symbols at the seam instead of each
        guessing them independently. Position n reads only the first
        min(n, W) window slots -- exactly the slots that have already been
        sampled when the parallel decode reaches position n -- so the single
        teacher-forced training pass matches decode-time information exactly,
        and W=1 reduces to the original single-token injection. Position 0 is
        deliberately excluded: it is the position that *predicts* b_k, and
        when it runs at decode time no b is known yet.
        """
        B, K, N, D = h.shape
        W = self.boundary_window
        no_tgt = self.no_target.view(1, 1, 1, D)
        if next_first is None:
            v = no_tgt.expand(B, K, W, D)
        else:
            if next_first.dim() == 2:
                next_first = next_first.unsqueeze(-1)
            v = self.tgt_proj(self.token_emb(next_first.clamp_min(0)))
            known = next_first >= 0
            if self.training and self.boundary_drop > 0.0:
                known = known & (torch.rand_like(v[..., 0]) >= self.boundary_drop)
            v = torch.where(known.unsqueeze(-1), v, no_tgt)
        if self.win_gain is not None:
            v = (v + self.win_pos) * self.win_gain.view(1, 1, W, 1)
        cum = v.cumsum(dim=2)                            # sum of slots 0..w
        pos = torch.arange(N, device=h.device)
        # Wavefront: block k+1 starts `lead` steps after block k, so its first
        # token (slot 0) exists from position lead+1 on; everything shifts by lead.
        lead = self.wave_lead
        eff = (pos - lead).clamp(min=0)
        take = (eff.clamp(max=W) - 1).clamp(min=0)       # slot count per position - 1
        inj = cum.gather(2, take.view(1, 1, N, 1).expand(B, K, N, D))
        gate = (pos >= 1 + lead).to(h.dtype)
        return inj.to(h.dtype) * gate.view(1, 1, N, 1)

    def _bag_features(self, tokens: Tensor) -> Tensor:
        """Cross-block lexical bag [B, K, N, D] from the input streams ``tokens`` [B, K, N].

        Input index i holds x_{i-1} (index 0 is BOB), so at within-block
        position n the other blocks have realized exactly inputs 1..n. The
        feature for (k, n) is the mean embedding of the content tokens
        (id >= lex_bag_min_id) at inputs 1..n of every block j != k, passed
        through the zero-init ``bag_proj``. Training rows may have the bag
        hidden (``lex_bag_drop``) so the block also learns to write without it.
        """
        B, K, N = tokens.shape
        emb = self.token_emb(tokens).detach()                    # [B, K, N, D]
        keep = (tokens >= self.lex_bag_min_id)
        keep[:, :, 0] = False                                     # BOB slot
        keep = keep.to(emb.dtype).unsqueeze(-1)
        cum = (emb * keep).cumsum(dim=2)                          # inputs 1..n, own block
        cnt = keep.cumsum(dim=2)
        others = cum.sum(dim=1, keepdim=True) - cum               # all blocks minus own
        n_oth = (cnt.sum(dim=1, keepdim=True) - cnt).clamp_min(1.0)
        bag = others / n_oth
        if self.training and self.lex_bag_drop > 0.0:
            drop = torch.rand(B, K, 1, 1, device=tokens.device) < self.lex_bag_drop
            bag = bag.masked_fill(drop, 0.0)
        return self.bag_proj(bag)

    def _state_landing(self, z: Tensor, h: Tensor, off: bool = False) -> Tensor:
        """Landing-state injection [B, K, 1, D] from the latents themselves.

        Block k reads s_{k+1} -- the prior's plan for what block k should
        realize (x_k drives the transition s_k -> s_{k+1}; conditioning on
        both ends makes the block a bridge). The landing is available before
        decoding starts, so every position sees it, and it is derived from
        whatever z the decoder was handed: the teacher-forced noise mixing
        and the inference-time prior error corrupt it through the exact same
        channel. The last block has no landing; ``landing_drop`` trains the
        same no-landing fallback for robustness.
        """
        B, K, N, D = h.shape
        no_land = self.no_landing.view(1, 1, -1)
        if off:
            land = no_land.expand(B, K, -1).to(z.dtype)
        else:
            blocks = z[:, self.num_global:]
            M = self.num_slots
            if M > 1:
                blocks = blocks.reshape(B, K, M, -1).mean(dim=2)
            nxt = torch.cat([blocks[:, 1:], blocks[:, :1]], dim=1)  # s_{k+1}; row K-1 junk
            # Residual mode: cross-attn already supplies s_k as the main
            # signal, the injection only carries the increment.
            land = nxt - blocks if self.landing_residual else nxt
            known = torch.arange(K, device=h.device)[None, :] < K - 1
            known = known.expand(B, K)
            if self.training and self.landing_drop > 0.0:
                known = known & (torch.rand(B, K, device=h.device) >= self.landing_drop)
            land = torch.where(known.unsqueeze(-1), land, no_land.to(land.dtype))
        return self.landing_out(self.landing_in(land)).unsqueeze(2).to(h.dtype)

    def forward(self, z: Tensor, block_inputs: Tensor,
                branch_ids: Tensor | None = None,
                next_first: Tensor | None = None,
                levels: Tensor | None = None,
                landing_off: bool = False,
                landing_lam: Tensor | None = None,
                rc: Tensor | None = None,
                bag_tokens: Tensor | None = None,
                prompt: Tensor | None = None,
                prompt_mask: Tensor | None = None) -> Tensor:
        """Teacher forcing.

        Args:
            z:            [B, G+K, D] latents (G globals first, then K blocks).
            block_inputs: [B, K, N] inputs [BOB, x_0, ..., x_{N-2}].
            bag_tokens:   [B, K, N] input streams the lexical bag is built from
                          (``lex_bag``); default = block_inputs. srp training
                          passes the REALIZED streams for realized rows so the
                          bag matches what decoding will see.
            next_first:   [B, K] or [B, K, W] landing targets (block k+1's
                          first tokens), -1 where there is none. Read only
                          when ``boundary_tokens``; None then means "no
                          target known" for every block.
            levels:       [B, G+K] per-entry noise level of z (1 = clean).
                          Read only when ``noise_level_cond``; None means all
                          clean.
            rc:           [B, K, R] realization codes (always the full K
                          blocks, also under ``branch_ids``). Required when
                          the decoder was built with rc_dim > 0.
        Returns:
            [B, K, N, V] logits predicting [x_0, ..., x_{N-1}].
        """
        if self.rc_dim > 0 and rc is None:
            raise ValueError("decoder built with rc_dim > 0 needs realization codes")
        h, z, prev_mask = self._prepare(z, self.token_emb(block_inputs), branch_ids,
                                        next_first, levels, landing_off, landing_lam, rc)
        streams = bag_tokens if bag_tokens is not None else block_inputs
        if self.lex_bag and branch_ids is None:
            h = h + self._bag_features(streams).to(h.dtype)
        pm, pmk = self._embed_prompt(prompt, prompt_mask)
        if self.lex_gate and branch_ids is None:
            hn = self._run_hidden(h, z, branch_ids, prev_mask, pm, pmk)
            logits = self.lm_head(hn)
            self._add_gate_bonus(logits, hn, streams)
            return logits
        return self._run(h, z, branch_ids, prev_mask, pm, pmk)

    def _embed_prompt(self, prompt: Tensor | None,
                      prompt_mask: Tensor | None) -> tuple[Tensor | None, Tensor | None]:
        """[B, P] prompt token ids (+ [B, P] bool mask) -> ([B, 1+P, D], [B, 1+P]) prompt memory
        with the learned null entry first; (None, None) when the writer has no prompt channel
        or no prompt is given."""
        if prompt is None or not self.prompt_attn:
            return None, None
        B, P = prompt.shape
        P = min(P, self.prompt_pos.num_embeddings)
        ids = prompt[:, :P].clamp_min(0)
        mem = self.token_emb(ids) + self.prompt_pos.weight[:P].unsqueeze(0)
        mask = torch.ones(B, P, dtype=torch.bool, device=prompt.device) if prompt_mask is None \
            else prompt_mask[:, :P]
        null = self.prompt_null.view(1, 1, -1).expand(B, 1, -1).to(mem.dtype)
        mem = torch.cat([null, mem], dim=1)
        mask = torch.cat([mask.new_ones(B, 1), mask], dim=1)
        return mem, mask

    def _add_gate_bonus(self, logits: Tensor, hn: Tensor, tokens: Tensor) -> None:
        """In place: logits[b,k,n,v] += mult * lex_scale * sigmoid(gate(h[b,k,n])) for every
        content token v (id >= lex_bag_min_id) that some block j != k has at inputs 1..n of
        ``tokens`` [B, K, N] (input i holds x_{i-1}; at decode step n the other blocks have
        realized exactly inputs 1..n, so teacher forcing and generation see the same set).
        Blocks are processed one at a time to keep the [B, N, V] count tensors small."""
        B, K, N = tokens.shape
        V = logits.shape[-1]
        gate = torch.sigmoid(self.gate_proj(hn).float()) * self.lex_scale.float() * self.lex_gate_mult
        content = (tokens >= self.lex_bag_min_id)
        content[:, :, 0] = False                                   # BOB slot
        idx = tokens.masked_fill(~content, 0).unsqueeze(-1)       # id 0 (pad) never counted below
        cnt_all = torch.zeros(B, N, V, dtype=torch.int16, device=tokens.device)
        for k in range(K):
            cnt_all.scatter_add_(2, idx[:, k], content[:, k].to(torch.int16).unsqueeze(-1).expand(B, N, 1))
        cnt_all[:, :, 0] = 0
        cum_all = cnt_all.cumsum(dim=1)                            # inputs 0..n, all blocks
        for k in range(K):
            own = torch.zeros(B, N, V, dtype=torch.int16, device=tokens.device)
            own.scatter_add_(2, idx[:, k], content[:, k].to(torch.int16).unsqueeze(-1).expand(B, N, 1))
            own[:, :, 0] = 0
            others = (cum_all - own.cumsum(dim=1)) > 0             # [B, N, V]
            logits[:, k] += (gate[:, k] * others.to(gate.dtype)).to(logits.dtype)

    def _run_hidden(self, h: Tensor, z: Tensor, branch_ids: Tensor | None,
                    prev_mask: Tensor | None, pm: Tensor | None = None,
                    pmk: Tensor | None = None) -> Tensor:
        G, M = self.num_global, self.num_slots
        look = 1 if self.landing_attn else 0
        for layer in self.layers:
            h = layer(h, z, self.within_rope, self.block_rope, self.causal_latents,
                      self.latent_window, self.null_latent, G, branch_ids, M, look,
                      prev_mask, self.cross_before_self, pm, pmk)
        return self.final_norm(h)

    def _run(self, h: Tensor, z: Tensor, branch_ids: Tensor | None,
             prev_mask: Tensor | None, pm: Tensor | None = None,
             pmk: Tensor | None = None) -> Tensor:
        return self.lm_head(self._run_hidden(h, z, branch_ids, prev_mask, pm, pmk))

    def _prepare(self, z: Tensor, tok: Tensor, branch_ids: Tensor | None,
                 next_first: Tensor | None, levels: Tensor | None,
                 landing_off: bool, landing_lam: Tensor | None,
                 rc: Tensor | None) -> tuple[Tensor, Tensor, Tensor | None]:
        """All conditioning: returns (h [B,K,N,D], memory z, prev_mask).

        ``tok`` are the token embeddings [B, K, N, D] of the block streams (N
        is the ordinary block span in teacher forcing / decoding and the
        longer FIM sequence in seam-stitch mode -- everything here is
        position-agnostic except the landing injection, which zeroes position
        0 only)."""
        G = self.num_global
        M = self.num_slots
        block_inputs = tok
        if self.noise_level_cond:
            if levels is None:
                levels = z.new_ones(z.shape[0], z.shape[1])
            lv = torch.stack([levels, 1.0 - levels], dim=-1).to(z.dtype)
            z = z + self.level_emb(lv)
        if self.rc_mem is not None:
            mem_add = self.rc_mem(rc.to(z.dtype))                  # [B, K, D]
            if G > 0:
                mem_add = torch.cat([mem_add.new_zeros(mem_add.shape[0], G,
                                                       mem_add.shape[-1]), mem_add], dim=1)
            z = z + mem_add
        if self.block_pos_emb is not None:
            # Learnable absolute block-index bias on top of the otherwise
            # fully relative (RoPE) position scheme. z always holds the full
            # [B, G+K*M, D] latent set (even under branch_ids), so indexing
            # by true block position here is correct regardless of which
            # branch subset gets decoded downstream. Zero-init: identical to
            # no bias at warm start.
            pos = (self.block_pos_emb if M == 1
                  else self.block_pos_emb.repeat_interleave(M, dim=0))
            bias = torch.cat([pos.new_zeros(G, pos.shape[1]), pos], dim=0)
            z = z + bias.unsqueeze(0).to(z.dtype)
        if branch_ids is None:
            if (z.shape[1] != block_inputs.shape[1] * M + G
                    or z.shape[0] != block_inputs.shape[0]):
                raise ValueError(
                    f"z {tuple(z.shape)} and inputs {tuple(block_inputs.shape[:3])} disagree "
                    f"(num_global={G}, num_slots={M})"
                )
            inject = z[:, G:]
            if M > 1:
                B_, KM, D_ = inject.shape
                inject = inject.reshape(B_, KM // M, M, D_).mean(dim=2)
        else:
            # Rollout decoding: x holds a subset of branches; z holds the full
            # latent set and the causal mask hides entries beyond each branch.
            if M > 1:
                raise NotImplementedError("branch decoding with multi-slot latents")
            inject = z[:, G:][:, branch_ids]
        h = block_inputs + inject.unsqueeze(2)
        if self.rc_in is not None:
            r_own = rc if branch_ids is None else rc[:, branch_ids]
            h = h + self.rc_in(r_own.to(h.dtype)).unsqueeze(2)
        if self.rc_next is not None and not getattr(self, "_lookahead_off", False):
            r_all_next = torch.cat([rc[:, 1:], torch.zeros_like(rc[:, :1])], dim=1)
            r_next = r_all_next if branch_ids is None else r_all_next[:, branch_ids]
            h = h + self.rc_next(r_next.to(h.dtype)).unsqueeze(2)
        if self.landing_cfg:
            lam = landing_lam if landing_lam is not None \
                else z.new_ones(z.shape[0])
            h = h + self.lam_emb(lam.view(-1, 1).to(h.dtype)).view(-1, 1, 1, h.shape[-1])
        if self.boundary_tokens:
            h = h + self._landing(next_first, h)
        if self.landing_state:
            if branch_ids is None:
                h = h + self._state_landing(z, h, off=landing_off)
            else:
                # Branch decoding (rollout): the k+1 latents are unfilled
                # placeholders, so every branch runs on the trained
                # no-landing fallback instead.
                no = self.no_landing.view(1, 1, -1).expand(h.shape[0], h.shape[1], -1)
                h = h + self.landing_out(self.landing_in(no)).unsqueeze(2).to(h.dtype)
        prev_mask = None
        if (self.prev_window or self.wave_lead > 0) and branch_ids is None:
            drop = None
            if self.training and self.wave_lead > 0 and self.wave_drop > 0.0:
                drop = torch.rand(h.shape[0], h.shape[1], device=h.device) < self.wave_drop
            prev_mask = self._prev_window_mask(h.shape[0], h.shape[1],
                                               h.shape[2], h.device,
                                               lead=self.wave_lead, drop=drop)
        return h, z, prev_mask

    # --- seam stitch (fill-in-the-middle over a block boundary) ------------ #
    def forward_stitch(self, z: Tensor, prefix: Tensor, suffix: Tensor,
                       middle_prev: Tensor, rc: Tensor | None = None,
                       next_first: Tensor | None = None,
                       levels: Tensor | None = None,
                       branch_ids: Tensor | None = None) -> Tensor:
        """Teacher-forced FIM pass over every block at once.

        Per block the token stream is
            [prefix (P)] [SUF] [suffix (Q)] [MID] [m_0 .. m_{M-2}]
        and the returned logits [B, K, M, V] predict the middle m_0..m_{M-1}
        (the MID marker predicts m_0). ``prefix``/``suffix`` are token ids
        with -1 = absent (block 0 has no previous tail; the last block no next
        head): absent tokens get a zero embedding. Conditioning (state s_k,
        codes, landing target, level label) is exactly the ordinary one.
        ``branch_ids`` (training memory knob): the streams describe only that
        subset of blocks; z / rc / levels stay the full set, ``next_first`` is
        already the subset.
        """
        if not self.stitch_mode:
            raise ValueError("decoder was built without stitch=True")
        B, K, P = prefix.shape
        Q = suffix.shape[2]
        Mp = middle_prev.shape[2]
        marker = torch.zeros(B, K, 1, dtype=torch.long, device=z.device)
        ids = torch.cat([prefix.clamp_min(0), marker, suffix.clamp_min(0), marker,
                         middle_prev.clamp_min(0)], dim=2)
        tok = self.token_emb(ids)
        present = torch.cat([prefix >= 0, torch.ones_like(marker, dtype=torch.bool),
                             suffix >= 0, torch.ones_like(marker, dtype=torch.bool),
                             torch.ones_like(middle_prev, dtype=torch.bool)], dim=2)
        tok = tok * present.unsqueeze(-1).to(tok.dtype)
        tok[:, :, P] = self.suf_emb.to(tok.dtype)
        tok[:, :, P + 1 + Q] = self.mid_emb.to(tok.dtype)
        h, z, _ = self._prepare(z, tok, branch_ids, next_first, levels, False, None, rc)
        logits = self._run(h, z, branch_ids, None)
        return logits[:, :, P + 1 + Q:]

    @torch.no_grad()
    def stitch(self, z: Tensor, blocks: Tensor, rc: Tensor | None,
               temperature: float, top_p: float, z_level: float = 1.0,
               start_n: int = 3, end_n: int = 2, prev_n: int = 3,
               next_n: int = 3) -> Tensor:
        """Boundary refinement after a parallel decode: rewrite the first
        ``start_n`` tokens of every block (k >= 1) given the previous block's
        last ``prev_n`` tokens and the block's own remaining body, then the
        last ``end_n`` tokens of every block (k <= K-2) given its own body and
        the next block's (new) first ``next_n`` tokens. All seams run in
        parallel; start_n + end_n extra decoder steps in total."""
        B, K, S = blocks.shape
        out = blocks.clone()
        W = self.boundary_window
        levels = None
        if self.noise_level_cond:
            levels = z.new_full((B, z.shape[1]), float(z_level))

        def landing_targets(bl: Tensor) -> Tensor | None:
            if not self.boundary_tokens:
                return None
            nf = torch.full((B, K, W), -1, dtype=torch.long, device=bl.device)
            nf[:, :-1] = bl[:, 1:, :W]
            return nf

        def fill(prefix: Tensor, suffix: Tensor, n: int, nf: Tensor | None) -> Tensor:
            mid = blocks.new_zeros(B, K, 0)
            for _ in range(n):
                logits = self.forward_stitch(z, prefix, suffix, mid, rc, nf, levels)[:, :, -1]
                nxt = sample_tokens(logits, temperature, top_p)
                mid = torch.cat([mid, nxt.unsqueeze(-1)], dim=2)
            return mid

        if start_n > 0:
            absent = torch.full((B, 1, prev_n), -1, dtype=torch.long, device=z.device)
            prefix = torch.cat([absent, out[:, :-1, S - prev_n:]], dim=1)
            mid = fill(prefix, out[:, :, start_n:], start_n, landing_targets(out))
            out[:, 1:, :start_n] = mid[:, 1:]
        if end_n > 0:
            absent = torch.full((B, 1, next_n), -1, dtype=torch.long, device=z.device)
            suffix = torch.cat([out[:, 1:, :next_n], absent], dim=1)
            mid = fill(out[:, :, : S - end_n], suffix, end_n, landing_targets(out))
            out[:, :-1, S - end_n:] = mid[:, :-1]
        return out

    def _prev_window_mask(self, B: int, K: int, N: int, device, lead: int = 0,
                          drop: Tensor | None = None) -> Tensor:
        """[B*K, 1, N, 2N] attendability: own causal + prev block up to p+lead.

        ``lead`` = wavefront offset: block k runs ``lead`` steps behind block
        k-1, so when it is at position p the previous block's hidden states
        at positions <= p+lead already exist (position p+lead is computed in
        the same batched forward). lead=0 is the position-synchronous
        prev_window. ``drop`` [B, K] hides the previous block for those rows
        (training-time hedge; a block must still be able to write alone).
        """
        j = torch.arange(2 * N, device=device)[None, :]
        p = torch.arange(N, device=device)[:, None]
        base = ((j >= N) & (j - N <= p)) | ((j < N) & (j <= p + lead))
        m = base.expand(B, K, N, 2 * N).clone()
        m[:, 0, :, :N] = False                   # block 0 has no previous block
        if drop is not None:
            m[:, :, :, :N] &= ~drop.view(B, K, 1, 1)
        return m.reshape(B * K, 1, N, 2 * N)

    @torch.no_grad()
    def generate(self, z: Tensor, bob_id: int, temperature: float = 0.0,
                 top_p: float = 1.0, z_level: float = 1.0,
                 landing_lambda: float = 1.0,
                 shared_noise: bool = False,
                 rc: Tensor | None = None,
                 force: Tensor | None = None,
                 use_wave: bool | None = None,
                 lex_bonus: float = 0.0, lex_min_id: int = 2000,
                 lex_post: bool = False,
                 no_repeat_ngram: int = 0,
                 landing_off: bool = False,
                 rc_mode: str = "plan",
                 rc_lookahead_off: bool = False,
                 prompt: Tensor | None = None,
                 prompt_mask: Tensor | None = None) -> Tensor:
        """Decode all K spans in parallel; returns [B, K, S] token ids.

        ``prompt`` [B, P] (+ ``prompt_mask``): prompt tokens for the prompt-aware writer
        (``prompt_attn``); every block cross-attends to them. Ignored by writers without
        the channel.

        Additional sampling knobs, all inference-only:

        ``no_repeat_ngram`` n > 0: document-wide n-gram blocking. The K blocks
        decode position-synchronously, so at step s every block knows every
        token any block has emitted at steps < s; a token that would complete
        an n-gram already present anywhere in the document (own block or
        another) gets logit -inf before the nucleus cut. This is the usual
        ``no_repeat_ngram_size`` of sequential samplers, applied across the
        parallel blocks.

        ``landing_off``: never feed the landing target (the trained
        ``no_target`` vector is used at every position, as for the last
        block) -- the inference-time ablation of the landing device on a
        model that was trained with it.

        ``rc_mode``: "plan" uses the codes handed in; "random" replaces them by
        fresh N(0, I) draws (the prior's codes are ignored, the writer still
        gets in-distribution codes); "zero" feeds the prior mean 0.

        ``rc_lookahead_off``: zero the next-block code r_{k+1} channel.
        """
        if rc is not None and rc_mode != "plan":
            if rc_mode == "random":
                rc = torch.randn_like(rc)
            elif rc_mode == "zero":
                rc = torch.zeros_like(rc)
            elif rc_mode == "shuffle":
                # in-distribution codes (the prior's own) that belong to ANOTHER document's plan:
                # separates "the code's distribution" from "the code's information about this plan"
                rc = rc[torch.randperm(rc.shape[0], device=rc.device)]
            elif rc_mode.startswith("scale"):
                rc = rc * float(rc_mode[5:])
            else:
                raise ValueError(f"unknown rc_mode {rc_mode!r}")
        self._lookahead_off = bool(rc_lookahead_off)
        try:
            return self._generate_impl(z, bob_id, temperature, top_p, z_level, landing_lambda,
                                       shared_noise, rc, force, use_wave, lex_bonus, lex_min_id,
                                       lex_post, no_repeat_ngram, landing_off, prompt, prompt_mask)
        finally:
            self._lookahead_off = False

    @staticmethod
    def _ngram_ban(gen: Tensor, n: int, vocab: int) -> Tensor | None:
        """[B, K, V] bool: tokens that would complete an n-gram already in the document.

        ``gen`` [B, K, s]: the tokens every block has produced so far (no BOS).
        Every block's trailing (n-1)-gram is matched against every (n-1)-gram
        window of every block of the same document; the token that followed
        each matching window is banned for that block.
        """
        B, K, s = gen.shape
        if s < n:
            return None
        q = gen[:, :, s - (n - 1):]                                  # [B, K, n-1]
        win = gen.unfold(2, n - 1, 1)[:, :, : s - (n - 1)]           # [B, K, s-n+1, n-1]
        nxt = gen[:, :, n - 1:]                                      # [B, K, s-n+1]
        match = (q[:, :, None, None, :] == win[:, None]).all(dim=-1)  # [B, Kq, Kw, s-n+1]
        banned = torch.where(match, nxt[:, None].expand_as(match), torch.full_like(match, vocab, dtype=torch.long))
        ban = torch.zeros(B, K, vocab + 1, dtype=torch.bool, device=gen.device)
        ban.scatter_(2, banned.reshape(B, K, -1), True)
        return ban[:, :, :vocab]

    def _generate_impl(self, z: Tensor, bob_id: int, temperature: float,
                       top_p: float, z_level: float, landing_lambda: float,
                       shared_noise: bool, rc: Tensor | None, force: Tensor | None,
                       use_wave: bool | None, lex_bonus: float, lex_min_id: int,
                       lex_post: bool, no_repeat_ngram: int, landing_off: bool,
                       prompt: Tensor | None = None, prompt_mask: Tensor | None = None) -> Tensor:
        """Body of ``generate`` (see there).

        ``lex_post``: apply ``lex_bonus`` AFTER the nucleus cut -- the top-p set
        is taken on the model's own distribution and the bonus only re-ranks
        inside it, so no tail token is ever forced in (the default adds the
        bonus first, letting a +2 logit pull tail tokens into the nucleus).

        ``use_wave``: with ``wave_lead`` > 0 the default is the wavefront
        schedule (``_generate_wave``); ``False`` forces the plain
        position-synchronous decode (used by srp training, whose realizations
        are only a state-hedging device and must stay cheap).

        ``lex_bonus`` > 0: cross-block lexical coupling at decode time (no
        training). The K blocks decode position-synchronously, so at step n
        every block can see the tokens the OTHER blocks produced at steps
        < n. Tokens with id >= ``lex_min_id`` (T5's sentencepiece ids are
        roughly frequency-ordered, so this skips function words) that some
        other block has already used get ``lex_bonus`` added to their
        logits -- a document-level "these are our words" prior that the
        plan cannot carry (topic persistence is capped at r1~8.5 even from
        real-text plans). Own-block repeats are not rewarded.

        ``force``: optional [B, K, S] long tensor; entries >= 0 replace the
        sampled token at that position (teacher forcing selected positions,
        e.g. the seam-oracle probe), -1 leaves the position free. Forced
        tokens enter the context and the landing targets like sampled ones.

        With ``boundary_tokens`` the first steps double as the boundary pass:
        step s emits position s of every block at once, and from step s+1 on
        block k also sees block k+1's first s+1 tokens (up to
        ``boundary_window``) as its landing target. Step 0's own prediction
        does not depend on the injection (see ``_landing``), so the seam gets
        pinned for free -- still ``block_span`` forwards.

        ``shared_noise``: see ``sample_tokens`` -- correlates which word each
        of the K independently-decoded blocks reaches for, without touching
        the state/plan channel (pure inference-time knob, no training).
        """
        lead = self.wave_lead if (use_wave is None or use_wave) else 0
        if lead > 0:
            return self._generate_wave(z, bob_id, temperature, top_p, z_level,
                                       landing_lambda, shared_noise, rc, force, lead)
        B = z.shape[0]
        K = (z.shape[1] - self.num_global) // self.num_slots
        W = self.boundary_window
        levels = None
        if self.noise_level_cond:
            levels = z.new_full((B, z.shape[1]), float(z_level))
        lam = z.new_full((B,), float(landing_lambda)) if self.landing_cfg else None
        prefix = torch.full((B, K, 1), bob_id, dtype=torch.long, device=z.device)
        next_first = None
        seen = None
        if lex_bonus > 0.0:
            V = self.lm_head.out_features
            seen = torch.zeros(B, K, V, dtype=torch.bool, device=z.device)
            content = (torch.arange(V, device=z.device) >= lex_min_id).view(1, 1, V)
        for step in range(self.block_span):
            if self.landing_cfg and landing_lambda != 1.0:
                # CFG combine on logits; lambda=1 collapses to the on-branch.
                l_on = self.forward(z, prefix, next_first=next_first,
                                    levels=levels, landing_lam=lam, rc=rc,
                                    prompt=prompt, prompt_mask=prompt_mask)[:, :, -1]
                l_off = self.forward(z, prefix, next_first=next_first,
                                     levels=levels, landing_off=True,
                                     landing_lam=lam, rc=rc,
                                     prompt=prompt, prompt_mask=prompt_mask)[:, :, -1]
                logits = l_off + landing_lambda * (l_on - l_off)
            else:
                logits = self.forward(z, prefix, next_first=next_first,
                                      levels=levels, landing_lam=lam, rc=rc,
                                      prompt=prompt, prompt_mask=prompt_mask)[:, :, -1]
            if no_repeat_ngram > 0 and step >= no_repeat_ngram - 1:
                ban = self._ngram_ban(prefix[:, :, 1:], no_repeat_ngram, logits.shape[-1])
                if ban is not None:
                    logits = logits.masked_fill(ban, float("-inf"))
            if seen is not None and step > 0:
                # used by some OTHER block (any-block minus own), content ids only
                others = seen.any(dim=1, keepdim=True) & ~seen & content
                if lex_post and temperature > 0.0 and top_p < 1.0:
                    keep = nucleus_mask(logits, temperature, top_p)
                    logits = (logits + lex_bonus * others.to(logits.dtype)).masked_fill(~keep, float("-inf"))
                    nxt = sample_tokens(logits, temperature, 1.0, shared_noise=shared_noise)
                else:
                    logits = logits + lex_bonus * others.to(logits.dtype)
                    nxt = sample_tokens(logits, temperature, top_p, shared_noise=shared_noise)
            else:
                nxt = sample_tokens(logits, temperature, top_p, shared_noise=shared_noise)
            if force is not None:
                f = force[:, :, step]
                nxt = torch.where(f >= 0, f, nxt)
            if seen is not None:
                seen.scatter_(2, nxt.unsqueeze(-1), True)   # forced (prompt) tokens count as seen too
            prefix = torch.cat([prefix, nxt.unsqueeze(-1)], dim=2)
            if self.boundary_tokens and step < W and not landing_off:
                if next_first is None:
                    # The last block has nowhere to land: stays -1 (no target).
                    next_first = torch.full((B, K, W), -1, dtype=torch.long,
                                            device=z.device)
                # Position-``step`` token of block k+1 is slot ``step`` of block k.
                next_first[:, :-1, step] = nxt[:, 1:]
        return prefix[:, :, 1:]

    @torch.no_grad()
    def _generate_wave(self, z: Tensor, bob_id: int, temperature: float, top_p: float,
                       z_level: float, landing_lambda: float, shared_noise: bool,
                       rc: Tensor | None, force: Tensor | None, lead: int) -> Tensor:
        """Wavefront decode: block k starts ``lead`` steps after block k-1.

        S + (K-1)*lead steps in total; at step t block k is at within-block
        position t - k*lead (inactive before it starts / after it ends). Every
        step is one teacher-forcing-shaped forward over the full [B, K, S]
        input streams (slots not yet reached hold BOB and are never attended:
        own attention is causal, and the prev-block mask only reaches
        positions <= p+lead, all of which the previous block -- ``lead`` steps
        ahead -- has already produced). Landing slot s of block k is block
        k+1's token s, which appears when block k+1 is at position s, i.e.
        when block k is at position lead+s; ``_landing`` gates it in from
        position lead+s+1, so the single teacher-forced training pass matches
        this decode exactly (checked by the greedy self-consistency test).
        """
        B = z.shape[0]
        K = (z.shape[1] - self.num_global) // self.num_slots
        S, W = self.block_span, self.boundary_window
        dev = z.device
        levels = z.new_full((B, z.shape[1]), float(z_level)) if self.noise_level_cond else None
        lam = z.new_full((B,), float(landing_lambda)) if self.landing_cfg else None
        inputs = torch.full((B, K, S), bob_id, dtype=torch.long, device=dev)
        out = torch.full((B, K, S), bob_id, dtype=torch.long, device=dev)
        next_first = (torch.full((B, K, W), -1, dtype=torch.long, device=dev)
                      if self.boundary_tokens else None)
        kpos = torch.arange(K, device=dev)
        for t in range(S + (K - 1) * lead):
            p = t - kpos * lead                                  # [K] position per block
            active = (p >= 0) & (p < S)
            idx = p.clamp(0, S - 1)
            if self.landing_cfg and landing_lambda != 1.0:
                l_on = self.forward(z, inputs, next_first=next_first, levels=levels,
                                    landing_lam=lam, rc=rc)
                l_off = self.forward(z, inputs, next_first=next_first, levels=levels,
                                     landing_off=True, landing_lam=lam, rc=rc)
                logits = l_off + landing_lambda * (l_on - l_off)
            else:
                logits = self.forward(z, inputs, next_first=next_first, levels=levels,
                                      landing_lam=lam, rc=rc)
            nxt = sample_tokens(logits[:, kpos, idx], temperature, top_p,
                                shared_noise=shared_noise)                    # [B, K]
            if force is not None:
                f = force[:, kpos, idx]
                nxt = torch.where(f >= 0, f, nxt)
            act = active[None, :].expand(B, K)
            out[:, kpos, idx] = torch.where(act, nxt, out[:, kpos, idx])
            slot = (idx + 1).clamp(max=S - 1)                    # input slot for token p
            can = act & (p[None, :] + 1 < S)
            inputs[:, kpos, slot] = torch.where(can, nxt, inputs[:, kpos, slot])
            if next_first is not None:
                # block k+1 (k < K-1) at position s < W just emitted block k's landing slot s
                fill = active[1:] & (p[1:] < W)
                if bool(fill.any()):
                    kk = kpos[:-1][fill]
                    ss = p[1:][fill]
                    next_first[:, kk, ss] = nxt[:, kk + 1]
        return out
