"""Iso-entropy readout of eval_gen logs: interpolate PPL to target entropies.

    python -m scripts.isoentropy eval_srpl_rc64_z0.log [more logs] \
        --targets 5.089,5.16,5.336

Rule: PPL is only comparable at equal per-sample entropy. This reads
every ``temp=... gen PPL=... /sample=...`` result line (one-shot, recon or
rollout), sorts the points by their GPT-2 per-sample entropy and log-linearly
interpolates PPL to each target entropy. Points outside the measured range
are extrapolated from the two nearest points and flagged -- the project's
convention is not to trust those, so prefer re-sweeping the temperature.
"""

from __future__ import annotations

import argparse
import math
import re

LINE = re.compile(r"temp=([\d.]+).*?gen PPL=\s*([\d.]+).*?gpt2 entropy corpus=[\d.]+ /sample=([\d.]+)")


def read_points(path: str) -> list[tuple[float, float, float]]:
    pts = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = LINE.search(line)
            if m:
                pts.append((float(m.group(3)), float(m.group(2)), float(m.group(1))))
    return sorted(pts)   # by entropy


def at_entropy(pts: list[tuple[float, float, float]], h: float) -> tuple[float, bool, str]:
    if len(pts) < 2:
        raise ValueError("need at least two points")
    lo = [p for p in pts if p[0] <= h]
    hi = [p for p in pts if p[0] >= h]
    if lo and hi:
        (h1, p1, t1), (h2, p2, t2) = lo[-1], hi[0]
        extrap = False
    elif hi:
        (h1, p1, t1), (h2, p2, t2) = pts[0], pts[1]
        extrap = True
    else:
        (h1, p1, t1), (h2, p2, t2) = pts[-2], pts[-1]
        extrap = True
    if h2 == h1:
        return p1, extrap, f"t={t1:.2f}"
    s = (math.log(p2) - math.log(p1)) / (h2 - h1)
    return math.exp(math.log(p1) + s * (h - h1)), extrap, f"t={t1:.2f}..{t2:.2f} slope {s:.2f}/nat"


def main() -> None:
    ap = argparse.ArgumentParser(description="iso-entropy PPL readout of eval_gen logs")
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--targets", default="5.089,5.16,5.336",
                    help="comma list of per-sample entropies to interpolate to")
    args = ap.parse_args()
    targets = [float(x) for x in args.targets.split(",")]
    for path in args.logs:
        pts = read_points(path)
        print(f"# {path}: {len(pts)} points")
        for h, p, t in pts:
            print(f"   t={t:.2f}  PPL {p:8.2f} @ {h:.3f}")
        if len(pts) < 2:
            continue
        for h in targets:
            p, ex, how = at_entropy(pts, h)
            print(f"   @{h:.3f}: {p:7.2f}  [{how}]{'  (EXTRAPOLATED)' if ex else ''}")


if __name__ == "__main__":
    main()
