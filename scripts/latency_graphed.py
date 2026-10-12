"""CUDA-graph inference latency for one-shot generation (manual capture, no torch.compile).

The eager cached decode is kernel-launch bound (batch 1 == batch 50 wall-clock). Here one writer
step and one denoiser step are each captured as a CUDA graph over STATIC buffers -- preallocated
KV caches [B*K, H, S, hd] written in place at slot n, attention over all S slots under a boolean
mask -- and replayed; only tiny copies into the static inputs happen between replays. Greedy
tokens must match the eager cached decode (parity); the prior's final latents must match
``flow_sample`` given the same initial noise (deterministic ODE path, cfg tokens).

  CUDA_VISIBLE_DEVICES=0 python -m scripts.latency_graphed --blocks 16 8 --batches 1 50
"""
from __future__ import annotations

import argparse
import time

import torch
import torch.nn.functional as F

from psf.config import Config
from psf.diffusion.flow import sample as flow_sample
from psf.utils.misc import get_device, set_seed
from scripts.latency_probe import build_model, cuda_time, generate_cached


def _sync_time(fn, iters, warmup):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


class GraphedWriter:
    """Static-shape cached greedy decode; one CUDA graph for the per-step body."""

    def __init__(self, dec, B: int, K: int, device):
        self.dec, self.B, self.K, self.dev = dec, B, K, device
        self.S = dec.block_span
        L = len(dec.layers)
        D = dec.layers[0].self_qkv.out_features // 3
        probe = dec.layers[0]._heads(torch.zeros(1, 1, D, device=device))
        self.H, self.hd = probe.shape[1], probe.shape[-1]
        self.k_cache = [torch.zeros(B * K, self.H, self.S, self.hd, device=device) for _ in range(L)]
        self.v_cache = [torch.zeros(B * K, self.H, self.S, self.hd, device=device) for _ in range(L)]
        # static inputs / output
        self.tok_s = torch.zeros(B, K, dtype=torch.long, device=device)
        self.pos_s = torch.zeros(1, dtype=torch.long, device=device)
        self.mask_s = torch.zeros(1, 1, 1, self.S, dtype=torch.bool, device=device)
        self.add_s = torch.zeros(B, K, D, device=device)
        self.land_s = torch.zeros(B, K, D, device=device)
        self.cross_k = [torch.zeros(B, self.H, K, self.hd, device=device) for _ in range(L)]
        self.cross_v = [torch.zeros(B, self.H, K, self.hd, device=device) for _ in range(L)]
        qk_pos = torch.arange(K, device=device)
        self.cmask = qk_pos[None, :] <= qk_pos[:, None]
        self.graph = None
        self.logits_s = None

    @torch.no_grad()
    def prepare(self, z, rc, z_level: float):
        dec, B, K = self.dec, self.B, self.K
        zc = z
        if dec.noise_level_cond:
            levels = torch.full((B, K), float(z_level), device=self.dev, dtype=z.dtype)
            zc = zc + dec.level_emb(torch.stack([levels, 1.0 - levels], dim=-1).to(z.dtype))
        if dec.rc_mem is not None:
            zc = zc + dec.rc_mem(rc.to(zc.dtype))
        add = zc.clone()
        if dec.rc_in is not None:
            add = add + dec.rc_in(rc.to(add.dtype))
        if dec.rc_next is not None:
            r_next = torch.cat([rc[:, 1:], torch.zeros_like(rc[:, :1])], dim=1)
            add = add + dec.rc_next(r_next.to(add.dtype))
        qk_pos = torch.arange(K, device=self.dev)
        for li, layer in enumerate(dec.layers):
            kk, vv = layer.cross_kv(zc).chunk(2, dim=-1)
            self.cross_k[li].copy_(dec.block_rope.rotate(layer._heads(kk), qk_pos))
            self.cross_v[li].copy_(layer._heads(vv))
        self.add_s.copy_(add)
        self.land_s.zero_()
        for kc, vc in zip(self.k_cache, self.v_cache):
            kc.zero_(); vc.zero_()

    def _step_body(self):
        dec, B, K = self.dec, self.B, self.K
        D = self.add_s.shape[-1]
        h = dec.token_emb(self.tok_s) + self.add_s + self.land_s
        hbk = h.reshape(B * K, 1, D)
        qk_pos = torch.arange(K, device=self.dev)
        for li, layer in enumerate(dec.layers):
            q, k, v = layer.self_qkv(layer.norm1(hbk)).chunk(3, dim=-1)
            q = dec.within_rope.rotate(layer._heads(q), self.pos_s)
            k = dec.within_rope.rotate(layer._heads(k), self.pos_s)
            v = layer._heads(v)
            self.k_cache[li].index_copy_(2, self.pos_s, k)
            self.v_cache[li].index_copy_(2, self.pos_s, v)
            a = F.scaled_dot_product_attention(q, self.k_cache[li], self.v_cache[li], attn_mask=self.mask_s)
            hbk = hbk + layer.self_out(layer._merge(a))
            qc = dec.block_rope.rotate(layer._heads(layer.cross_q(layer.norm2(hbk)).reshape(B, K, D)), qk_pos)
            ac = F.scaled_dot_product_attention(qc, self.cross_k[li], self.cross_v[li], attn_mask=self.cmask)
            hbk = hbk + layer.cross_out(layer._merge(ac).reshape(B * K, 1, D))
            hbk = hbk + layer.mlp(layer.norm3(hbk))
        return dec.lm_head(dec.final_norm(hbk)).reshape(B, K, -1)

    @torch.no_grad()
    def capture(self):
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):                       # warmup (allocator, cudnn/sdpa plans)
                self._step_body()
        torch.cuda.current_stream().wait_stream(s)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.logits_s = self._step_body()

    @torch.no_grad()
    def generate(self, z, bob_id: int, rc, z_level: float = 0.0):
        dec, B, K, S = self.dec, self.B, self.K, self.S
        self.prepare(z, rc, z_level)
        if self.graph is None:
            self.tok_s.fill_(bob_id); self.pos_s.zero_(); self.mask_s.zero_(); self.mask_s[..., 0] = True
            self.capture()
            self.prepare(z, rc, z_level)             # the capture warmups wrote into the caches
        slots = torch.arange(S, device=self.dev)
        self.tok_s.fill_(bob_id)
        outs = []
        for n in range(S):
            self.pos_s.fill_(n)
            self.mask_s.copy_((slots <= n).view(1, 1, 1, S))
            self.graph.replay()
            nxt = self.logits_s.argmax(-1)
            outs.append(nxt)
            if n == 0 and dec.boundary_tokens:
                nf = torch.full((B, K), -1, dtype=torch.long, device=self.dev)
                nf[:, :-1] = nxt[:, 1:]
                known = (nf >= 0).unsqueeze(-1)
                v0 = dec.tgt_proj(dec.token_emb(nf.clamp_min(0)))
                self.land_s.copy_(torch.where(known, v0, dec.no_target.view(1, 1, -1)))
            self.tok_s.copy_(nxt)
        return torch.stack(outs, dim=2)


class GraphedPrior:
    """flow.sample's deterministic (ODE, cfg-token) path with the denoiser call captured as a graph."""

    def __init__(self, denoiser, B: int, nlat: int, dim: int, flow_cfg, cfg_scale: float, device):
        self.den, self.B, self.nlat, self.dim, self.cfg, self.dev = denoiser, B, nlat, dim, flow_cfg, device
        self.z_s = torch.zeros(B, nlat, dim, device=device)
        self.t_s = torch.zeros(B, device=device)
        self.sc_s = torch.zeros(B, nlat, dim, device=device)
        self.w_s = torch.full((B,), float(cfg_scale), device=device)
        self.graph = None
        self.x0_s = None

    def _body(self):
        return self.den(self.z_s, self.t_s, self.sc_s, cfg_scale=self.w_s)

    @torch.no_grad()
    def capture(self):
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                self._body()
        torch.cuda.current_stream().wait_stream(s)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.x0_s = self._body()

    @torch.no_grad()
    def sample(self, noise):
        if self.graph is None:
            self.capture()
        N = self.cfg.num_sampling_steps
        grid = torch.linspace(0.0, 1.0, N + 1, device=self.dev)
        z = noise.clone()
        self.sc_s.zero_()
        for i in range(N):
            t_cur, t_next = float(grid[i]), float(grid[i + 1])
            self.z_s.copy_(z); self.t_s.fill_(t_cur)
            self.graph.replay()
            x0 = self.x0_s
            self.sc_s.copy_(x0)
            if t_next >= 1.0:
                z = x0.clone()
            else:
                z = z + (x0 - z) / max(1.0 - t_cur, 1e-4) * (t_next - t_cur)
        return z


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--blocks", type=int, nargs="+", default=[16, 8])
    p.add_argument("--batches", type=int, nargs="+", default=[1, 50])
    p.add_argument("--iters", type=int, default=5)
    p.add_argument("--cfg-scale", type=float, default=1.0)
    p.add_argument("--arch", choices=["record", "big"], default="record",
                   help="record = D=512 checkpoints; big = D=1024 writer + 308M denoiser, random weights "
                        "(latency depends on shapes only; parity is eager vs graph on the same weights)")
    p.add_argument("--device", default="auto")
    args = p.parse_args()
    device = get_device(args.device)
    torch.set_float32_matmul_precision("high")
    bob = Config().data.bob_token_id
    print(f"{'K':>3} {'S':>3} {'B':>3} | {'prior eager':>11} {'prior graph':>11} | {'writer eager':>12} {'writer graph':>12} | "
          f"{'total eager':>11} {'total graph':>11} | {'ms/doc graph':>12} | writer parity | prior max|dz|")
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
            gp = GraphedPrior(model.denoiser, B, nlat, model.prior_dim, cfg.flow, args.cfg_scale, device)
            # prior parity: same initial noise through flow_sample and the graphed loop
            torch.manual_seed(1)
            z_ref = flow_sample(model.denoiser, B, nlat, model.prior_dim, cfg.flow, device, cfg_scale=args.cfg_scale)
            torch.manual_seed(1)
            noise = torch.randn(B, nlat, model.prior_dim, device=device)
            z_g = gp.sample(noise)
            dz = (z_ref - z_g).abs().max().item()
            s, rc = model.split_latents(model.denoiser.denormalize(z_ref))
            gw = GraphedWriter(dec, B, K, device)
            ref = generate_cached(dec, s, bob, rc, z_level=0.0)
            got = gw.generate(s, bob, rc)
            parity = (ref == got).float().mean().item()
            t_pe = cuda_time(lambda: flow_sample(model.denoiser, B, nlat, model.prior_dim, cfg.flow, device, cfg_scale=args.cfg_scale), args.iters, 1, device)
            t_pg = _sync_time(lambda: gp.sample(noise), args.iters, 2)
            t_we = cuda_time(lambda: generate_cached(dec, s, bob, rc, z_level=0.0), args.iters, 1, device)
            t_wg = _sync_time(lambda: gw.generate(s, bob, rc), args.iters, 2)
            ms = lambda t: t * 1e3
            print(f"{K:>3} {dec.block_span:>3} {B:>3} | {ms(t_pe):>11.1f} {ms(t_pg):>11.1f} | {ms(t_we):>12.1f} {ms(t_wg):>12.1f} | "
                  f"{ms(t_pe + t_we):>11.1f} {ms(t_pg + t_wg):>11.1f} | {ms(t_pg + t_wg) / B:>12.2f} | {100 * parity:>12.1f}% | {dz:.2e}",
                  flush=True)
            del gw, gp
            torch.cuda.empty_cache()
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
