"""Time the EVALUATED samplers on the captured-graph path, and the same-data AR on a captured graph.

Extends scripts.latency_graphed (whose GraphedWriter/GraphedPrior time a greedy writer step + the prior as CUDA graphs):
  * the writer's per-step token selection is now the evaluated one -- temperature / nucleus (top-p) sampling and the
    optional document-wide n-gram ban (Decoder._ngram_ban) -- run eagerly on the graph's logits between replays, exactly
    as scripts.eval_gen's sampler would (the graph only holds the network step);
  * a GraphedAR: the 197M same-data AR baseline (psf.models.ar_lm) with a static KV cache [B, H, L, hd] per layer and
    one captured graph per decoding step, greedy or nucleus, so its latency is measured on the same footing as
    GPT-2 small in Table 2 instead of the eager 489 ms / 19.5 s.

  CUDA_VISIBLE_DEVICES=0 python -m scripts.latency_r4 --arch record --blocks 8 --batches 1 32 \
      --samplers greedy nucleus:0.55 nucleus:0.55:ban4 temp:0.75 --ar-ckpt runs/ar-197m/ar.pt
Prints a table; one JSON line per row to bench/latency_r4.jsonl.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import torch
import torch.nn.functional as F

from psf.config import Config
from psf.diffusion.flow import sample as flow_sample
from psf.models.decoder import Decoder
from psf.utils.misc import get_device, set_seed
from scripts.latency_graphed import GraphedPrior, GraphedWriter, _sync_time
from scripts.latency_probe import build_model


def parse_sampler(spec: str) -> dict:
    """greedy | nucleus:<p>[:ban<n>] | temp:<t>[:ban<n>]"""
    parts = spec.split(":")
    d = {"name": spec, "temperature": 1.0, "top_p": 1.0, "ban": 0, "greedy": parts[0] == "greedy"}
    if parts[0] == "nucleus":
        d["top_p"] = float(parts[1])
    elif parts[0] == "temp":
        d["temperature"] = float(parts[1])
    for p in parts[2:]:
        if p.startswith("ban"):
            d["ban"] = int(p[3:])
    return d


def select_tokens(logits: torch.Tensor, s: dict) -> torch.Tensor:
    """logits [..., V] -> tokens [...]; the evaluated sampler's arithmetic (temperature, top-p cut, multinomial)."""
    if s["greedy"]:
        return logits.argmax(-1)
    x = logits / s["temperature"]
    if s["top_p"] < 1.0:
        sorted_logits, idx = x.sort(dim=-1, descending=True)
        probs = sorted_logits.softmax(-1)
        drop = probs.cumsum(-1) - probs > s["top_p"]
        sorted_logits = sorted_logits.masked_fill(drop, float("-inf"))
        x = torch.full_like(x, float("-inf")).scatter(-1, idx, sorted_logits)
    shape = x.shape[:-1]
    return torch.multinomial(x.reshape(-1, x.shape[-1]).softmax(-1), 1).reshape(shape)


class SampledGraphedWriter(GraphedWriter):
    """GraphedWriter with the evaluated token selection (and n-gram ban) between graph replays."""

    def __init__(self, dec, B, K, device, sampler: dict):
        super().__init__(dec, B, K, device)
        self.sampler = sampler

    @torch.no_grad()
    def generate(self, z, bob_id: int, rc, z_level: float = 0.0):
        dec, B, K, S = self.dec, self.B, self.K, self.S
        self.prepare(z, rc, z_level)
        if self.graph is None:
            self.tok_s.fill_(bob_id); self.pos_s.zero_(); self.mask_s.zero_(); self.mask_s[..., 0] = True
            self.capture()
            self.prepare(z, rc, z_level)
        slots = torch.arange(S, device=self.dev)
        self.tok_s.fill_(bob_id)
        outs = []
        n_ban = self.sampler["ban"]
        for n in range(S):
            self.pos_s.fill_(n)
            self.mask_s.copy_((slots <= n).view(1, 1, 1, S))
            self.graph.replay()
            logits = self.logits_s
            if n_ban > 0 and n >= n_ban - 1:
                ban = Decoder._ngram_ban(torch.stack(outs, dim=2), n_ban, logits.shape[-1])
                if ban is not None:
                    logits = logits.masked_fill(ban, float("-inf"))
            nxt = select_tokens(logits, self.sampler)
            outs.append(nxt)
            if n == 0 and dec.boundary_tokens:
                nf = torch.full((B, K), -1, dtype=torch.long, device=self.dev)
                nf[:, :-1] = nxt[:, 1:]
                known = (nf >= 0).unsqueeze(-1)
                v0 = dec.tgt_proj(dec.token_emb(nf.clamp_min(0)))
                self.land_s.copy_(torch.where(known, v0, dec.no_target.view(1, 1, -1)))
            self.tok_s.copy_(nxt)
        return torch.stack(outs, dim=2)


class GraphedAR:
    """Static-KV-cache decode of psf.models.ar_lm.ARLM with one CUDA graph per step (mirrors GraphedWriter)."""

    def __init__(self, model, B: int, L: int, device, sampler: dict):
        from psf.models.ar_lm import apply_rope
        self.m, self.B, self.L, self.dev, self.sampler = model, B, L, device, sampler
        self.apply_rope = apply_rope
        cfg = model.cfg
        self.H, self.hd = cfg.num_heads, cfg.dim // cfg.num_heads
        nl = len(model.blocks)
        self.k_cache = [torch.zeros(B, self.H, L, self.hd, device=device) for _ in range(nl)]
        self.v_cache = [torch.zeros(B, self.H, L, self.hd, device=device) for _ in range(nl)]
        self.tok_s = torch.zeros(B, 1, dtype=torch.long, device=device)
        self.pos_s = torch.zeros(1, dtype=torch.long, device=device)
        self.mask_s = torch.zeros(1, 1, 1, L, dtype=torch.bool, device=device)
        self.graph = None
        self.logits_s = None

    def _step_body(self):
        m, B = self.m, self.B
        h = m.emb(self.tok_s)                                                   # [B, 1, D]
        for li, blk in enumerate(m.blocks):
            D = h.shape[-1]
            q, k, v = blk.qkv(blk.norm1(h)).split(D, dim=-1)
            q = q.view(B, 1, self.H, -1).transpose(1, 2)
            k = k.view(B, 1, self.H, -1).transpose(1, 2)
            v = v.view(B, 1, self.H, -1).transpose(1, 2)
            q, k = self.apply_rope(q, m.cos, m.sin, self.pos_s), self.apply_rope(k, m.cos, m.sin, self.pos_s)
            self.k_cache[li].index_copy_(2, self.pos_s, k)
            self.v_cache[li].index_copy_(2, self.pos_s, v)
            a = F.scaled_dot_product_attention(q, self.k_cache[li], self.v_cache[li], attn_mask=self.mask_s)
            h = h + blk.out(a.transpose(1, 2).reshape(B, 1, D))
            h = h + blk.mlp(blk.norm2(h))
        return m.logits(h[:, -1]).float()                                       # [B, V]

    @torch.no_grad()
    def capture(self):
        s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                self._step_body()
        torch.cuda.current_stream().wait_stream(s)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.logits_s = self._step_body()

    @torch.no_grad()
    def generate(self):
        m, B, L = self.m, self.B, self.L
        for kc, vc in zip(self.k_cache, self.v_cache):
            kc.zero_(); vc.zero_()
        self.tok_s.fill_(m.cfg.bos_id); self.pos_s.zero_(); self.mask_s.zero_(); self.mask_s[..., 0] = True
        if self.graph is None:
            self.capture()
            for kc, vc in zip(self.k_cache, self.v_cache):
                kc.zero_(); vc.zero_()
        slots = torch.arange(L, device=self.dev)
        out = torch.zeros(B, L, dtype=torch.long, device=self.dev)
        n_ban = self.sampler["ban"]
        for n in range(L):
            self.pos_s.fill_(n)
            self.mask_s.copy_((slots <= n).view(1, 1, 1, L))
            self.graph.replay()
            logits = self.logits_s
            if n_ban > 0 and n >= n_ban - 1:
                from psf.models.ar_lm import _ngram_ban_1d
                ban = _ngram_ban_1d(out[:, :n], n_ban, logits.shape[-1])
                if ban is not None:
                    logits = logits.masked_fill(ban, float("-inf"))
            nxt = select_tokens(logits, self.sampler)
            out[:, n] = nxt
            self.tok_s.copy_(nxt[:, None])
        return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--blocks", type=int, nargs="+", default=[8])
    p.add_argument("--batches", type=int, nargs="+", default=[1, 32])
    p.add_argument("--samplers", nargs="+", default=["greedy", "nucleus:0.55", "nucleus:0.55:ban4", "temp:0.75"])
    p.add_argument("--iters", type=int, default=5)
    p.add_argument("--arch", choices=["record", "big"], default="record")
    p.add_argument("--ar-ckpt", default=None, help="time the same-data AR (psf.models.ar_lm) on a captured graph")
    p.add_argument("--ar-samplers", nargs="+", default=["greedy", "nucleus:0.88", "nucleus:0.88:ban4"])
    p.add_argument("--out", default="bench/latency_r4.jsonl")
    p.add_argument("--device", default="auto")
    args = p.parse_args()
    device = get_device(args.device)
    torch.set_float32_matmul_precision("high")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fout = open(args.out, "a", encoding="utf-8")
    bob = Config().data.bob_token_id
    print(f"{'model':>10} {'K':>3} {'B':>3} {'sampler':>20} | {'prior ms':>9} {'writer ms':>10} {'total ms':>9} {'ms/doc':>8}", flush=True)
    for K in args.blocks:
        set_seed(0)
        if args.arch == "big":
            from scripts.bench_ours import build as build_arch
            model, cfg = build_arch(K, "big", device)
        else:
            model, cfg = build_model(K, device)
        dec = model.decoder
        nlat = cfg.num_blocks * cfg.num_latent_slots + cfg.num_global_latents
        for B in args.batches:
            gp = GraphedPrior(model.denoiser, B, nlat, model.prior_dim, cfg.flow, 1.0, device)
            torch.manual_seed(1)
            noise = torch.randn(B, nlat, model.prior_dim, device=device)
            z_g = gp.sample(noise)
            s, rc = model.split_latents(model.denoiser.denormalize(z_g))
            t_pg = _sync_time(lambda: gp.sample(noise), args.iters, 2)
            for spec in args.samplers:
                smp = parse_sampler(spec)
                gw = SampledGraphedWriter(dec, B, K, device, smp)
                gw.generate(s, bob, rc)
                t_wg = _sync_time(lambda: gw.generate(s, bob, rc), args.iters, 2)
                row = dict(model=f"PSF-{args.arch}", K=K, B=B, sampler=spec, prior_ms=1e3 * t_pg, writer_ms=1e3 * t_wg,
                           total_ms=1e3 * (t_pg + t_wg), ms_per_doc=1e3 * (t_pg + t_wg) / B)
                print(f"{row['model']:>10} {K:>3} {B:>3} {spec:>20} | {row['prior_ms']:>9.1f} {row['writer_ms']:>10.1f} "
                      f"{row['total_ms']:>9.1f} {row['ms_per_doc']:>8.2f}", flush=True)
                fout.write(json.dumps(row) + "\n"); fout.flush()
                del gw; torch.cuda.empty_cache()
            del gp; torch.cuda.empty_cache()
        del model; torch.cuda.empty_cache()
    if args.ar_ckpt:
        from psf.models.ar_lm import ARConfig, ARLM
        ck = torch.load(args.ar_ckpt, map_location=device, weights_only=False)
        ar = ARLM(ARConfig(**ck["ar_config"])).to(device).eval()
        ar.load_state_dict(ck["model"])
        L = ar.cfg.max_seq_len
        for B in args.batches:
            for spec in args.ar_samplers:
                smp = parse_sampler(spec)
                g = GraphedAR(ar, B, L, device, smp)
                g.generate()
                t = _sync_time(lambda: g.generate(), max(1, args.iters // 2), 1)
                row = dict(model="AR-197M-graph", K=1, B=B, sampler=spec, prior_ms=0.0, writer_ms=1e3 * t,
                           total_ms=1e3 * t, ms_per_doc=1e3 * t / B)
                print(f"{row['model']:>10} {1:>3} {B:>3} {spec:>20} | {0.0:>9.1f} {row['writer_ms']:>10.1f} "
                      f"{row['total_ms']:>9.1f} {row['ms_per_doc']:>8.2f}", flush=True)
                fout.write(json.dumps(row) + "\n"); fout.flush()
                del g; torch.cuda.empty_cache()
        # eager reference for the AR (the number the paper currently prints)
        for B in args.batches:
            t = _sync_time(lambda: ar.generate(B, device, temperature=1.0, top_p=0.88), 1, 1)
            row = dict(model="AR-197M-eager", K=1, B=B, sampler="nucleus:0.88", prior_ms=0.0, writer_ms=1e3 * t,
                       total_ms=1e3 * t, ms_per_doc=1e3 * t / B)
            print(f"{row['model']:>10} {1:>3} {B:>3} {'nucleus:0.88':>20} | {0.0:>9.1f} {row['writer_ms']:>10.1f} "
                  f"{row['total_ms']:>9.1f} {row['ms_per_doc']:>8.2f}", flush=True)
            fout.write(json.dumps(row) + "\n"); fout.flush()


if __name__ == "__main__":
    main()
