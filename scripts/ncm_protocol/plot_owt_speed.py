"""NCM's OpenWebText panel (Gen. PPL vs seconds per sequence, NCM paper Figure 5 right) with PSF added.

Baseline curves and NCM's star are recovered from the paper's own SVG (marker centres mapped through its log axes; the
recovered end points reproduce Table 8: DUO 77.7, MDLM 106, SEDD 101.4, NCM 83.1 @ 0.080 s). PSF: speed from
<results>/time_psf_fp32.jsonl (time_psf.py --fp32-highest), timed the way NCM's numbers were (one H100, batch 16, 1,024
sequences, FP32 with TF32 off, compiled / CUDA graphs: calib_ncm.py shows this protocol reproduces NCM's own 80 ms and
TinyStories 5.65 s); Gen. PPL from NCM's scorer interpolated to H = 5.42 (<results>/score_packed.log, ncm_score.py).
The SVG is https://arxiv.org/html/2610.14162v1/efficiency.svg.

    python scripts/ncm_protocol/plot_owt_speed.py efficiency.svg <results> assets/ncm_owt_comparison
"""

import json
import os
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FixedLocator, NullLocator  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from summarize import LINE, at  # noqa: E402

INK, MUTED, GRID = "#1f1f1f", "#5f6368", "#e3e3e3"
MDLM, SEDD, DUO, NCM, NCM_INK, PSF = "#008300", "#3f88c5", "#4a3aa7", "#eda100", "#7a5200", "#d1495b"
H_TARGET = 5.42


def ncm_panel(svg_path):
    """OWT panel of NCM's Figure 5: {series: [(seconds, ppl), ...]} and NCM's (seconds, ppl)."""
    s = open(svg_path, encoding="utf-8").read()
    sec = lambda x: 10 ** (-1 + (x - 366.2847) / 50.645)            # x ticks: 0.1 s at 366.2847, 1 s at 416.929
    ppl = lambda y: 100 * 10 ** ((84.408 - y) / 73.387)             # y ticks (top-down): 100 at 84.408, 200 at 62.316
    colour = {"#008300": "MDLM", "#6c7a89": "SEDD", "#4a3aa7": "DUO"}
    series = {}
    for cid in range(96, 122):
        m = re.search(r'<clipPath id="clip_%d">\s*<path transform="matrix\(1,0,0,-1,([\d.]+),([\d.]+)\)"' % cid, s)
        use = re.search(r'clip-path="url\(#clip_%d\)"[^>]*>' % cid, s)
        if not m or float(m.group(1)) < 340 or not use:
            continue
        c = re.search(r'fill="(#[0-9a-f]{6})"', s[use.start():use.start() + 800]).group(1)
        series.setdefault(colour[c], []).append((sec(float(m.group(1))), ppl(float(m.group(2)))))
    star = re.findall(r'd="M([\d.]+) ([\d.]+) [\d.]+ [\d.]+H[\d.]+L[\d.]+ [\d.]+ [\d.]+ ([\d.]+) ', s)
    sx, top, bottom = [(float(a), float(b), float(c)) for a, b, c in star if float(a) > 340][0]
    centre = 138.69812 - (0.809 * top + bottom) / 1.809              # star centre from its top and lower tips
    return series, (sec(sx), ppl(centre))


def psf_points(results):
    timing = {}
    for line in open(os.path.join(results, "time_psf_fp32.jsonl"), encoding="utf-8"):
        r = json.loads(line)
        if r["B"] == 16:
            timing[(r["model"], r["K"], r["sampler"])] = r["ms_per_doc"] / 1e3
    sweeps = {}
    for line in open(os.path.join(results, "score_packed.log"), encoding="utf-8"):
        m = LINE.match(line)
        if m:
            model, _, _, _, p, _, ent, _ = m.groups()
            sweeps.setdefault(model, []).append((float(ent), float(p), "", 0.0, 0))
    q = {k: at(sorted(v), H_TARGET)[0] for k, v in sweeps.items()}
    plain = [(timing[("PSF-record", K, "tp:0.95:0.95")], q[f"psf_k{K}"], K) for K in (32, 16, 8)]
    large = (timing[("PSF-big", 8, "nucleus:0.60")], q["psf-l_k8_states-only"])
    return plain, large


def main():
    svg, results, out = sys.argv[1:4]
    series, (ncm_s, ncm_ppl) = ncm_panel(svg)
    plain, large = psf_points(results)

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.edgecolor": "#444444",
                         "axes.linewidth": 1.4, "xtick.color": "#444444", "ytick.color": "#444444"})
    fig, ax = plt.subplots(figsize=(8, 4.9), dpi=200)
    fig.subplots_adjust(left=0.1, right=0.97, top=0.9, bottom=0.12)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(0.025, 9); ax.set_ylim(20, 1000)
    ax.xaxis.set_major_locator(FixedLocator([0.03, 0.05, 0.1, 0.2, 0.5, 1, 2, 5]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels(["0.03", "0.05", "0.1", "0.2", "0.5", "1", "2", "5"])
    ax.yaxis.set_major_locator(FixedLocator([25, 50, 100, 200, 400, 800]))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_yticklabels(["25", "50", "100", "200", "400", "800"])
    ax.grid(True, color=GRID, linestyle=(0, (4, 3)), linewidth=0.9)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_xlabel("Seconds per 1024-token sequence (one H100, batch 16, FP32)", color=INK)
    ax.set_ylabel("Gen. PPL (GPT-2 Large) ↓", color=INK)
    ax.set_title("OpenWebText (1024 tokens), NCM's protocol", color=INK, fontsize=13, pad=10)

    # baselines as reported by NCM
    for name, col in (("MDLM", MDLM), ("SEDD", SEDD), ("DUO", DUO)):
        xs, ys = zip(*sorted(series[name]))
        ax.plot(xs, ys, color=col, lw=2, zorder=2)
        ax.scatter(xs, ys, s=46, color=col, edgecolors="white", linewidths=1.2, zorder=3)

    # NCM as reported
    ax.scatter([ncm_s], [ncm_ppl], marker="*", s=420, color=NCM, edgecolors=NCM_INK, linewidths=1.2, zorder=5)
    ax.text(ncm_s * 0.9, ncm_ppl * 1.04, f"NCM\nPPL {ncm_ppl:.1f}",
            color=INK, fontsize=9.5, ha="right", va="bottom", fontweight="bold", multialignment="right")

    # PSF (ours): K = 32 / 16 / 8 and PSF-L
    xs, ys, ks = zip(*plain)
    ax.plot(xs, ys, color=PSF, lw=2.4, zorder=6)
    ax.scatter(xs, ys, s=70, color=PSF, edgecolors="white", linewidths=1.4, zorder=7)
    label_at = {32: (0.92, 1.0, "right", "center"), 16: (1.06, 1.04, "left", "bottom"), 8: (1.07, 0.95, "left", "top")}
    for x, y, k in plain:
        fx, fy, ha, va = label_at[k]
        ax.text(x * fx, y * fy, f"K={k}", color=INK, fontsize=9.5, ha=ha, va=va)
    ax.scatter([large[0]], [large[1]], marker="s", s=64, color=PSF, edgecolors="white", linewidths=1.4, zorder=7)
    ax.text(large[0] * 1.07, large[1], "PSF-L, K=8", color=INK, fontsize=9.5, ha="left", va="center")
    ax.text(0.0265, 27.5, "PSF (ours)", color=INK, fontsize=11.5, fontweight="bold", ha="left", va="center")

    # NCM -> PSF K=8: faster and lower PPL
    k8_s, k8_ppl, _ = plain[-1]
    ax.annotate("", xy=(k8_s * 1.01, k8_ppl * 1.07), xytext=(ncm_s * 0.98, ncm_ppl * 0.92),
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.4, mutation_scale=12), zorder=4)
    ax.text(0.092, 50, f"{ncm_ppl / k8_ppl:.1f}× lower PPL\n({k8_ppl:.1f} vs {ncm_ppl:.1f})",
            color=INK, fontsize=10, fontweight="bold", ha="left", va="center")

    ax.text(0.985, 0.975, "Data: Gen. PPL 14.15, H=5.44", transform=ax.transAxes, ha="right", va="top", fontsize=9.5,
            color=INK, bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#bdbdbd"))
    ax.legend(handles=[
        plt.Line2D([], [], color=PSF, lw=2.4, marker="o", mfc=PSF, mec="white", ms=7, label="PSF (ours), H=5.42"),
        plt.Line2D([], [], color=NCM, lw=0, marker="*", mfc=NCM, mec=NCM_INK, ms=13, label="NCM (NFE 32), H=5.42"),
        plt.Line2D([], [], color=DUO, lw=2, marker="o", mfc=DUO, mec="white", ms=6, label="DUO, H=5.55"),
        plt.Line2D([], [], color=SEDD, lw=2, marker="o", mfc=SEDD, mec="white", ms=6, label="SEDD, H=5.61"),
        plt.Line2D([], [], color=MDLM, lw=2, marker="o", mfc=MDLM, mec="white", ms=6, label="MDLM, H=5.64"),
    ], loc="lower right", bbox_to_anchor=(1.0, 0.03), fontsize=9.5, frameon=True, framealpha=1, edgecolor="#d0d0d0")

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}.{ext}", facecolor="white")
    print(f"{out}.png", {k: [(round(a, 3), round(b, 1)) for a, b in v[-1:]] for k, v in series.items()},
          "NCM", (round(ncm_s, 4), round(ncm_ppl, 2)), "ours", [(round(a, 4), round(b, 1)) for a, b, _ in plain],
          (round(large[0], 4), round(large[1], 1)))


if __name__ == "__main__":
    main()
