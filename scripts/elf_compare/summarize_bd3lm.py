"""Side-by-side of THEIR metrics and OURS on the same BD3-LM / MDLM samples.

Their CSVs (bd3lms `sample_eval`) carry gen_ppl (GPT-2-large, sliding window), gen_entropy
(their definition) and gen_nfes per sample; our judge logs (`eval_<cfg>_ourjudge.log`) carry
gen PPL @ per-sample GPT-2 unigram entropy + distinct / rep / loop8. For each model the two
nucleus settings (0.9, 1.0) give two (entropy, PPL) points; we log-linearly interpolate to the
paper's entropy targets exactly like scripts/isoentropy.py does for our temperature sweeps
(flagged EXTRAP when the target lies outside the bracket).
  python summarize_bd3lm.py --bd3 /path/to/bd3lms --ours /path/to/generative_language_latents
"""
from __future__ import annotations

import argparse
import csv
import glob
import math
import os
import re
import sys

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))   # C long is 32-bit on Windows
TARGETS = [5.063, 5.089, 5.16, 5.184]
MODELS = [("bd3lm_bs4", "BD3-LM L'=4"), ("bd3lm_bs8", "BD3-LM L'=8"), ("bd3lm_bs16", "BD3-LM L'=16"),
          ("mdlm", "MDLM T=1000")]


def theirs(bd3: str, cfg: str):
    # headerless rows: gen_ppl, gen_nfes, gen_entropy, gen_lengths, samples, seed
    rows = []
    for path in glob.glob(os.path.join(bd3, "sample_logs", cfg, "gpu*.csv")):
        with open(path, newline="", encoding="utf-8") as f:
            rows += [r for r in csv.reader(f) if len(r) >= 5 and r[0] != "gen_ppl"]
    if not rows:
        return None
    def mean(i):
        vals = []
        for r in rows:
            try:
                vals.append(float(r[i]))
            except ValueError:
                pass
        return sum(vals) / len(vals) if vals else float("nan")
    return {"n": len(rows), "their_ppl": mean(0), "their_entropy": mean(2),
            "nfes": mean(1), "len": mean(3)}


def ours(our: str, cfg: str, suffix: str = ""):
    path = os.path.join(our, f"eval_{cfg}{suffix}_ourjudge.log")
    if not os.path.exists(path):
        return None
    txt = open(path, encoding="utf-8").read()
    m = re.search(r"n=(\d+) \| gen PPL=\s*([0-9.]+) \| gpt2 entropy corpus=([0-9.]+) /sample=([0-9.]+).*?scored (\d+) judge tok", txt)
    d = re.search(r"distinct=([0-9.]+) \| rep2=([0-9.]+) rep4=([0-9.]+) rep8=([0-9.]+) \| loop8 coverage=([0-9.]+) samples=([0-9.]+)", txt)
    if not m or float(m.group(4)) <= 0.0 or int(m.group(1)) == 0:
        return None          # empty / failed judge run (e.g. a config whose sampling crashed)
    out = {"ppl": float(m.group(2)), "H_corpus": float(m.group(3)), "H": float(m.group(4)),
           "tok_per_doc": int(m.group(5)) / max(1, int(m.group(1)))}
    if d:
        out.update(distinct=float(d.group(1)), rep2=float(d.group(2)), rep8=float(d.group(4)),
                   loop=float(d.group(6)))
    return out


def interp(p1, p2, targets):
    (h1, y1), (h2, y2) = sorted([p1, p2])
    if h2 == h1:
        return {}
    slope = (math.log(y2) - math.log(y1)) / (h2 - h1)
    out = {}
    for t in targets:
        val = math.exp(math.log(y1) + slope * (t - h1))
        out[t] = (val, not (h1 - 1e-9 <= t <= h2 + 1e-9), slope)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bd3", default="../bd3lms")
    ap.add_argument("--ours", default=".")
    a = ap.parse_args()
    print("their = official CSV columns (whole 1024-token canvas, EOS kept in-sequence); "
          "our/full = our judge on the same full canvas; our/doc = our judge on the first document (cut at EOS)")
    print(f"{'model':>13} {'p':>4} | {'n':>4} {'their PPL':>9} {'their H':>8} {'NFE':>6} | "
          f"{'full PPL':>8} {'full H':>6} | {'doc PPL':>8} {'doc H':>6} {'tok/doc':>7} {'distinct':>8} {'rep2':>6} {'rep8':>6} {'loop':>6}")
    print("-" * 140)
    for key, label in MODELS:
        pts = []
        for p in ("p09", "p095", "p10"):
            cfg = f"{key}_{p}"
            t, o, of = theirs(a.bd3, cfg), ours(a.ours, cfg), ours(a.ours, cfg, "_fulleos")
            if t is None and o is None:
                continue
            ts = (f"{t['n']:>4} {t['their_ppl']:>9.2f} {t['their_entropy']:>8.3f} {t['nfes']:>6.0f}" if t
                  else f"{'-':>4} {'-':>9} {'-':>8} {'-':>6}")
            fs = f"{of['ppl']:>8.2f} {of['H']:>6.3f}" if of else f"{'-':>8} {'-':>6}"
            os_ = (f"{o['ppl']:>8.2f} {o['H']:>6.3f} {o['tok_per_doc']:>7.0f} {o.get('distinct', float('nan')):>8.3f} "
                   f"{o.get('rep2', float('nan')):>6.3f} {o.get('rep8', float('nan')):>6.3f} {o.get('loop', float('nan')):>6.3f}"
                   if o else f"{'-':>8} {'-':>6} {'-':>7} {'-':>8} {'-':>6} {'-':>6} {'-':>6}")
            print(f"{label:>13} {p[1:]:>4} | {ts} | {fs} | {os_}")
            if o:
                pts.append((o["H"], o["ppl"]))
        if len(pts) >= 2:
            pts = sorted(pts)
            # interpolate between the two points bracketing (or nearest to) each target
            parts = []
            for t in TARGETS:
                lo = [q for q in pts if q[0] <= t]
                hi = [q for q in pts if q[0] >= t]
                lo_pt = lo[-1] if lo else pts[0]
                hi_pt = hi[0] if hi else pts[-1]
                if lo_pt == hi_pt:
                    lo_pt, hi_pt = (pts[0], pts[1]) if t < pts[0][0] else (pts[-2], pts[-1])
                res = interp(lo_pt, hi_pt, [t])
                if res:
                    v, ex, s = res[t]
                    parts.append(f"@{t}: {v:.2f}{' (EXTRAP)' if ex else ''} [slope {s:.2f}]")
            print("   iso-entropy (our judge, doc-level, log-linear between nucleus points): " + "  ".join(parts))
    print("\nours for reference @5.063/5.089/5.16/5.184: K=16 cfg1 20.12/21.30/24.94/25.96; K=8 cfg1 18.12/19.13/22.82/24.23;"
          " ELF-B 32-step 23.97 @5.158.")


if __name__ == "__main__":
    main()
