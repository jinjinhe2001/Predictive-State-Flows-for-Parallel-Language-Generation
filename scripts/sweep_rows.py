"""Print the temperature / nucleus sweeps of our eval_gen logs as LaTeX table rows (paper appendix).

  python -m scripts.sweep_rows LABEL=path.log [LABEL=path.log ...]
Each row: setting (temperature t / nucleus p / guidance / noise scale), Gen. PPL, per-document
entropy, loop-sample fraction. One-shot rows only (R=0), sorted by entropy.
"""
import re
import sys

ROW = re.compile(r"temp=([0-9.]+) top_p=([0-9.]+) cfg=([0-9.]+) sde=([0-9.]+) ns=([0-9.]+) \| gen PPL= *([0-9.]+).*?/sample=([0-9.]+)")
LOOP = re.compile(r"loop8 coverage=([0-9.]+) samples=([0-9.]+)")

for spec in sys.argv[1:]:
    label, path = spec.split("=", 1)
    rows = []
    lines = open(path, encoding="utf-8", errors="replace").read().splitlines()
    for i, line in enumerate(lines):
        m = ROW.search(line)
        if not m or " R=0 " not in line and "R=0" not in line:
            continue
        t, p, cfg, sde, ns, ppl, ent = (float(x) for x in m.groups())
        loop = ""
        for j in range(i + 1, min(i + 3, len(lines))):
            ml = LOOP.search(lines[j])
            if ml:
                loop = f"{100 * float(ml.group(2)):.1f}\\%"
                break
        setting = (f"$p{{=}}{p:.2f}$" if t == 1.0 and p < 1.0 else f"$t{{=}}{t:.2f}$" + (f", $p{{=}}{p:.2f}$" if p < 0.95 else ""))
        if cfg != 1.0:
            setting += f", guidance {cfg:g}"
        if ns != 1.0:
            setting += f", noise {ns:g}"
        rows.append((ent, setting, ppl, loop))
    rows.sort()
    print(f"% ---- {label} ({path})")
    for ent, setting, ppl, loop in rows:
        print(f"{label} & {setting} & {ppl:.2f} & {ent:.3f} & {loop} \\\\")
