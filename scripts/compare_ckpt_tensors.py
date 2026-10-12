"""Per-tensor comparison of two checkpoints, grouped by top-level module (CPU only, read-only).

Check behind footnote a of Table tab:stages: was PSF-L's encoder trunk actually updated during the
self-realized-prefix + label stage (runs/ae-d1024-base-v5 -> runs/ae-d1024-srpl)? That stage ran without
--freeze-encoder, but its loss detaches every state, so the encoder should get grad=None and AdamW should skip it
(and the EMA lerp of an unchanged parameter is exact). Bit-identical encoder tensors confirm it.

  python -m scripts.compare_ckpt_tensors runs/ae-d1024-base-v5/ae.pt runs/ae-d1024-srpl/ae.pt [--prefixes encoder,decoder]
"""
from __future__ import annotations

import argparse
from collections import defaultdict

import torch


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("a")
    p.add_argument("b")
    p.add_argument("--prefixes", default="encoder,decoder,denoiser", help="top-level modules to report")
    p.add_argument("--show", type=int, default=5, help="list up to this many differing tensors per module")
    args = p.parse_args()
    A = torch.load(args.a, map_location="cpu", weights_only=False)
    B = torch.load(args.b, map_location="cpu", weights_only=False)
    print(f"a = {args.a}  (steps {A.get('steps')})\nb = {args.b}  (steps {B.get('steps')})")
    ma, mb = A["model"], B["model"]
    groups = defaultdict(lambda: {"n": 0, "params": 0, "identical": 0, "differ": [], "only_a": [], "only_b": [],
                                  "shape": [], "sq_diff": 0.0, "sq_norm": 0.0, "max_abs": 0.0})
    for k in sorted(set(ma) | set(mb)):
        g = groups[k.split(".", 1)[0]]
        if k not in mb:
            g["only_a"].append(k); continue
        if k not in ma:
            g["only_b"].append(k); continue
        x, y = ma[k], mb[k]
        g["n"] += 1
        g["params"] += x.numel()
        if x.shape != y.shape:
            g["shape"].append(k); continue
        if torch.equal(x, y):
            g["identical"] += 1
            g["sq_norm"] += float(x.double().pow(2).sum())
            continue
        d = (x.double() - y.double())
        g["differ"].append((k, float(d.abs().max()), float(d.norm() / x.double().norm().clamp(min=1e-30))))
        g["sq_diff"] += float(d.pow(2).sum())
        g["sq_norm"] += float(x.double().pow(2).sum())
        g["max_abs"] = max(g["max_abs"], float(d.abs().max()))
    wanted = [s for s in args.prefixes.split(",") if s]
    for name in wanted + sorted(k for k in groups if k not in wanted):
        if name not in groups:
            print(f"\n[{name}] not present in either checkpoint"); continue
        g = groups[name]
        rel = (g["sq_diff"] / g["sq_norm"]) ** 0.5 if g["sq_norm"] > 0 else 0.0
        verdict = ("shared tensors BIT-IDENTICAL" if not g["differ"] and not g["shape"] else "CHANGED")
        print(f"\n[{name}] {g['n']} shared tensors ({g['params'] / 1e6:.2f}M params): {g['identical']} identical, "
              f"{len(g['differ'])} differ, {len(g['shape'])} shape mismatch, {len(g['only_a'])} only in a, "
              f"{len(g['only_b'])} only in b -> {verdict}")
        if g["differ"]:
            print(f"  relative L2 change {rel:.3e}, max |diff| {g['max_abs']:.3e}")
            for k, mx, r in sorted(g["differ"], key=lambda t: -t[2])[: args.show]:
                print(f"    {k}: max |diff| {mx:.3e}, rel {r:.3e}")
        for lab in ("shape", "only_a", "only_b"):
            if g[lab]:
                print(f"  {lab}: {g[lab][: args.show]}{' ...' if len(g[lab]) > args.show else ''}")


if __name__ == "__main__":
    main()
