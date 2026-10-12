"""Summarize the fixed-window re-scoring (outputs/win512, outputs/win512drop) and redo the matched-entropy
comparison at the window entropy of ELF-B 32 steps.

  python -m scripts.window_summary --dir outputs/win512 [--anchor-tag elf_b_s32_g1.5_sc3]
Each log is one dump scored with --judge-window 512 (perplexity and per-document entropy over the first 512 GPT-2
Large tokens). Dumps are grouped into systems by tag prefix; within a system the points are sorted by entropy and the
perplexity at the anchor entropy is interpolated log-linearly between the two bracketing points (extrapolation is
flagged). Output: one table of measured points and one table of anchored values.
"""
from __future__ import annotations

import argparse
import glob
import math
import os
import re

GROUPS = [  # (system name, regex on the tag, setting extractor)
    ("real", r"^real_owt_1000$", lambda t: "-"),
    ("GPT-2 small", r"^gpt2full_", lambda t: t.replace("gpt2full_", "")),
    ("AR same data", r"^texts_ar197m_", lambda t: t.replace("texts_ar197m_", "")),
    ("BD3-LM block 4", r"^bd3lm_bs4_", lambda t: t.replace("bd3lm_bs4_", "")),
    ("BD3-LM block 8", r"^bd3lm_bs8_", lambda t: t.replace("bd3lm_bs8_", "")),
    ("BD3-LM block 16", r"^bd3lm_bs16_", lambda t: t.replace("bd3lm_bs16_", "")),
    ("MDLM", r"^mdlm_", lambda t: t.replace("mdlm_", "")),
    ("ELF-B 32 steps (paper point)", r"^elf_b_s32_g1\.5_sc3$", lambda t: "32 steps, gamma 1.5, sccfg 3"),
    ("ELF-B 32 steps, seeds 43/44", r"^elf_b_s32_g1\.5_sc3_seed", lambda t: t.replace("elf_b_", "")),
    ("ELF-B other settings", r"^elf_b_", lambda t: t.replace("elf_b_", "")),
    ("ELF-M", r"^elf_m_", lambda t: t.replace("elf_m_", "")),
    ("ELF-L", r"^elf_l_", lambda t: t.replace("elf_l_", "")),
    ("PSF K=64 (tlow model)", r"^texts_rev_k64tlow_", lambda t: t.split("nfe50_")[-1]),
    ("PSF K=64", r"^texts_rev_k64_", lambda t: t.split("nfe50_")[-1]),
    ("PSF K=32", r"^texts_rev_k32_", lambda t: t.split("nfe50_")[-1]),
    ("PSF K=16", r"^texts_rev_k16_", lambda t: t.split("nfe50_")[-1]),
    ("PSF K=8 plain", r"^texts_rev_k8_|^texts_k8_base_nfe50_t0", lambda t: t.split("nfe50_")[-1]),
    ("PSF K=8 nucleus", r"^texts_k8_base_nfe50_t1", lambda t: t.split("nfe50_")[-1]),
    ("PSF K=8 states-only nucleus", r"^texts_so_k8_nuc_", lambda t: t.split("nfe50_")[-1]),
    ("PSF K=8 states-only + ban", r"^texts_so_k8_nr4_", lambda t: t.split("nfe50_")[-1]),
    ("PSF-L K=8 plain", r"^texts_big_k8_nfe50_", lambda t: t.split("nfe50_")[-1]),
    ("PSF-L K=8 nucleus", r"^texts_big_k8_nuc_", lambda t: t.split("nfe50_")[-1]),
    ("PSF-L K=16", r"^texts_big_k16_", lambda t: t.split("nfe50_")[-1]),
    ("PSF-L K=64", r"^texts_big_k64_nfe50_", lambda t: t.split("nfe50_")[-1]),
    ("PSF-L K=8 states-only nucleus", r"^texts_so_big_nuc_", lambda t: t.split("nfe50_")[-1]),
    ("PSF-L K=8 states-only + ban", r"^texts_so_big_nr4_", lambda t: t.split("nfe50_")[-1]),
]


def parse(path):
    t = open(path, encoding="utf-8", errors="replace").read()
    m = re.search(r"gen PPL=\s*([0-9.]+) \| gpt2 entropy corpus=([0-9.]+) /sample=([0-9.]+)", t)
    n = re.search(r"n=(\d+) \|", t)
    w = re.search(r"scored docs=(\d+)", t)
    if not m:
        return None
    return dict(ppl=float(m.group(1)), ent=float(m.group(3)), n=int(n.group(1)) if n else 0,
                scored=int(w.group(1)) if w else None)


def interp(points, e):
    pts = sorted(points, key=lambda p: p["ent"])
    for p in pts:
        if abs(p["ent"] - e) < 1e-9:
            return p["ppl"], " (measured at the anchor)"
    lo = [p for p in pts if p["ent"] <= e]; hi = [p for p in pts if p["ent"] >= e]
    if lo and hi and lo[-1] is not hi[0]:
        a, b = lo[-1], hi[0]
        f = (e - a["ent"]) / (b["ent"] - a["ent"])
        return math.exp(math.log(a["ppl"]) + f * (math.log(b["ppl"]) - math.log(a["ppl"]))), ""
    if len(pts) >= 2:
        a, b = (pts[0], pts[1]) if e < pts[0]["ent"] else (pts[-2], pts[-1])
        f = (e - a["ent"]) / (b["ent"] - a["ent"])
        return math.exp(math.log(a["ppl"]) + f * (math.log(b["ppl"]) - math.log(a["ppl"]))), " (extrapolated %.3f nats)" % (min(abs(e - pts[0]["ent"]), abs(e - pts[-1]["ent"])))
    return None, ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="outputs/win512")
    ap.add_argument("--anchor-tag", default="elf_b_s32_g1.5_sc3")
    ap.add_argument("--anchor", type=float, default=None, help="override the anchor entropy")
    a = ap.parse_args()
    rows = {}
    for f in sorted(glob.glob(os.path.join(a.dir, "*.log"))):
        tag = os.path.basename(f)[:-4]
        r = parse(f)
        if r is None:
            print(f"# unparsed: {tag}"); continue
        rows[tag] = r
    anchor = a.anchor
    if anchor is None and a.anchor_tag in rows:
        anchor = rows[a.anchor_tag]["ent"]
    print(f"# {a.dir}: {len(rows)} dumps; anchor entropy {anchor:.3f} ({a.anchor_tag})\n")
    print(f"{'system':32s} {'setting':22s} {'PPL':>8s} {'entropy':>8s} {'docs':>6s}")
    grouped = {}
    for tag, r in rows.items():
        for name, rx, fmt in GROUPS:
            if re.search(rx, tag):
                grouped.setdefault(name, []).append(dict(r, setting=fmt(tag), tag=tag)); break
        else:
            grouped.setdefault("other:" + tag, []).append(dict(r, setting="", tag=tag))
    for name, pts in grouped.items():
        for p in sorted(pts, key=lambda p: p["ent"]):
            docs = p["scored"] if p["scored"] is not None else p["n"]
            print(f"{name:32s} {p['setting']:22s} {p['ppl']:8.2f} {p['ent']:8.3f} {docs:6d}")
    print(f"\n## perplexity at the window anchor entropy {anchor:.3f}")
    for name, pts in grouped.items():
        v, note = interp(pts, anchor)
        if v is not None:
            print(f"{name:32s} {v:8.2f}{note}")
        else:
            print(f"{name:32s} {'single point':>12s} {pts[0]['ppl']:.2f} @{pts[0]['ent']:.3f}")


if __name__ == "__main__":
    main()
