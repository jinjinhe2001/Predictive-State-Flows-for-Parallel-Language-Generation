"""Plain autoregressive transformer LM: the same-data, same-tokenizer baseline for the paper.

GPT-style pre-LN decoder over the T5 vocabulary and the 1024-token canvas the writer sees:
rotary positions, causal SDPA attention, GELU MLP, tied input/output embeddings, no biases.
Begin-of-document token = the PAD id (as the writer's begin-of-block token). Generation uses a KV
cache. Sized so that d=1024 x 13 layers ~= 197M parameters = the main-line writer + prior.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class ARConfig:
    vocab_size: int = 32100
    max_seq_len: int = 1024
    dim: int = 1024
    depth: int = 13
    num_heads: int = 16
    mlp_ratio: int = 4
    dropout: float = 0.0
    bos_id: int = 0
    eos_id: int = 1


def rope_cache(head_dim: int, max_len: int, device, base: float = 10000.0) -> tuple[Tensor, Tensor]:
    inv = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(max_len, device=device).float()
    ang = torch.outer(t, inv)                       # [L, hd/2]
    return ang.cos(), ang.sin()


def apply_rope(x: Tensor, cos: Tensor, sin: Tensor, pos: Tensor) -> Tensor:
    """x [B, H, T, hd]; pos [T] absolute positions."""
    c, s = cos[pos][None, None], sin[pos][None, None]      # [1, 1, T, hd/2]
    x1, x2 = x[..., 0::2], x[..., 1::2]
    out = torch.stack([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1)
    return out.flatten(-2).type_as(x)


class Block(nn.Module):
    def __init__(self, cfg: ARConfig) -> None:
        super().__init__()
        self.h = cfg.num_heads
        self.norm1 = nn.LayerNorm(cfg.dim, bias=False)
        self.qkv = nn.Linear(cfg.dim, 3 * cfg.dim, bias=False)
        self.out = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.norm2 = nn.LayerNorm(cfg.dim, bias=False)
        self.mlp = nn.Sequential(nn.Linear(cfg.dim, cfg.mlp_ratio * cfg.dim, bias=False), nn.GELU(),
                                 nn.Linear(cfg.mlp_ratio * cfg.dim, cfg.dim, bias=False))
        self.drop = cfg.dropout

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor, pos: Tensor,
                cache: dict | None = None) -> Tensor:
        B, T, D = x.shape
        q, k, v = self.qkv(self.norm1(x)).split(D, dim=-1)
        q = q.view(B, T, self.h, -1).transpose(1, 2)
        k = k.view(B, T, self.h, -1).transpose(1, 2)
        v = v.view(B, T, self.h, -1).transpose(1, 2)
        q, k = apply_rope(q, cos, sin, pos), apply_rope(k, cos, sin, pos)
        if cache is not None:
            if "k" in cache:
                k = torch.cat([cache["k"], k], dim=2)
                v = torch.cat([cache["v"], v], dim=2)
            cache["k"], cache["v"] = k, v
            a = F.scaled_dot_product_attention(q, k, v, is_causal=(T > 1))
        else:
            a = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=self.drop if self.training else 0.0)
        x = x + self.out(a.transpose(1, 2).reshape(B, T, D))
        return x + self.mlp(self.norm2(x))


class ARLM(nn.Module):
    def __init__(self, cfg: ARConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.emb = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.depth))
        self.norm = nn.LayerNorm(cfg.dim, bias=False)
        cos, sin = rope_cache(cfg.dim // cfg.num_heads, cfg.max_seq_len + 1, torch.device("cpu"))
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.apply(self._init)
        s = (2 * cfg.depth) ** -0.5                     # GPT-2-style scaled residual projections
        with torch.no_grad():
            for b in self.blocks:
                b.out.weight.mul_(s)
                b.mlp[-1].weight.mul_(s)

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def logits(self, h: Tensor) -> Tensor:
        return F.linear(self.norm(h), self.emb.weight)  # tied output embedding

    def forward(self, tokens: Tensor, mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """tokens [B, L] (PAD after the document). Returns (mean NLL over real tokens, token count).
        Input = [BOS, x_1 .. x_{L-1}], target = [x_1 .. x_L]; the first PAD after the document is
        predicted as the end-of-document target only if the data carries an EOS there (T5 data does)."""
        B, L = tokens.shape
        inp = torch.cat([torch.full((B, 1), self.cfg.bos_id, dtype=tokens.dtype, device=tokens.device),
                         tokens[:, :-1]], dim=1)
        pos = torch.arange(L, device=tokens.device)
        h = self.emb(inp)
        for b in self.blocks:
            h = b(h, self.cos, self.sin, pos)
        logits = self.logits(h).float()
        nll = F.cross_entropy(logits.view(-1, logits.shape[-1]), tokens.reshape(-1), reduction="none").view(B, L)
        m = mask.float() if mask is not None else torch.ones_like(nll)
        return (nll * m).sum() / m.sum().clamp_min(1.0), m.sum()

    @torch.no_grad()
    def generate(self, num_samples: int, device, temperature: float = 1.0, top_p: float = 1.0,
                 max_len: int | None = None, prefix: Tensor | None = None, no_repeat_ngram: int = 0) -> Tensor:
        """Returns [N, L] token ids; positions after the first EOS are filled with PAD.
        ``prefix`` [N, P]: given document prefix (prompted continuation); the output starts with it."""
        L = max_len or self.cfg.max_seq_len
        caches = [dict() for _ in self.blocks]
        out = torch.full((num_samples, L), 0, dtype=torch.long, device=device)
        done = torch.zeros(num_samples, dtype=torch.bool, device=device)
        start = 0
        if prefix is not None and prefix.shape[1] > 0:
            P = prefix.shape[1]
            out[:, :P] = prefix
            inp = torch.cat([torch.full((num_samples, 1), self.cfg.bos_id, dtype=torch.long, device=device),
                             prefix[:, :-1]], dim=1)
            h = self.emb(inp)
            pos = torch.arange(P, device=device)
            for b, c in zip(self.blocks, caches):
                h = b(h, self.cos, self.sin, pos, cache=c)
            tok = prefix[:, -1:]
            start = P
        else:
            tok = torch.full((num_samples, 1), self.cfg.bos_id, dtype=torch.long, device=device)
        for n in range(start, L):
            pos = torch.tensor([n], device=device)
            h = self.emb(tok)
            for b, c in zip(self.blocks, caches):
                h = b(h, self.cos, self.sin, pos, cache=c)
            logits = self.logits(h[:, -1]).float()
            if no_repeat_ngram > 0 and n >= no_repeat_ngram - 1:
                ban = _ngram_ban_1d(out[:, :n], no_repeat_ngram, logits.shape[-1])
                if ban is not None:
                    logits = logits.masked_fill(ban, float("-inf"))
            if temperature <= 0.0:
                nxt = logits.argmax(-1)
            else:
                logits = logits / temperature
                if top_p < 1.0:
                    sorted_logits, idx = logits.sort(dim=-1, descending=True)
                    probs = sorted_logits.softmax(-1)
                    cum = probs.cumsum(-1)
                    drop = cum - probs > top_p
                    sorted_logits[drop] = float("-inf")
                    logits = torch.full_like(logits, float("-inf")).scatter(-1, idx, sorted_logits)
                nxt = torch.multinomial(logits.softmax(-1), 1).squeeze(-1)
            nxt = torch.where(done, torch.zeros_like(nxt), nxt)
            out[:, n] = nxt
            done = done | (nxt == self.cfg.eos_id)
            tok = nxt[:, None]
            if bool(done.all()):
                break
        return out


def _ngram_ban_1d(gen: Tensor, n: int, vocab: int) -> Tensor | None:
    """[B, V] bool: tokens that would complete an n-gram already present in the sequence (document-wide ban,
    the sequential counterpart of the writer's Decoder._ngram_ban). ``gen`` [B, s]: tokens produced so far."""
    B, s = gen.shape
    if s < n:
        return None
    q = gen[:, s - (n - 1):]                                  # [B, n-1]
    win = gen.unfold(1, n - 1, 1)[:, : s - (n - 1)]           # [B, s-n+1, n-1]
    nxt = gen[:, n - 1:]                                      # [B, s-n+1]
    match = (q[:, None, :] == win).all(dim=-1)                # [B, s-n+1]
    banned = torch.where(match, nxt, torch.full_like(nxt, vocab))
    ban = torch.zeros(B, vocab + 1, dtype=torch.bool, device=gen.device)
    ban.scatter_(1, banned, True)
    return ban[:, :vocab]
