"""Read ncm_score.py logs and log-linearly interpolate each model's genPPL to target entropies (scripts/isoentropy.py rule).

    python scripts/ncm_protocol/summarize.py score_packed.log score_doc.log --targets 5.42
"""

import argparse
import math
import re
from collections import defaultdict

LINE = re.compile(r"^(\S+)_nfe50_t([\d.]+)_p([\d.]+)_R0\s+mode=(\w+)\s+genPPL=\s*([\d.]+) \+/-\s*([\d.]+) entropy=([\d.]+) len=\s*(\d+)")


def at(pts, h):
    lo = [p for p in pts if p[0] <= h]
    hi = [p for p in pts if p[0] >= h]
    if lo and hi:
        a, b, flag = lo[-1], hi[0], ""
    else:
        a, b = (pts[:2] if not lo else pts[-2:])
        flag = " (extrapolated)"
    if a[0] == b[0]:
        return a[1], flag
    w = (h - a[0]) / (b[0] - a[0])
    return math.exp(math.log(a[1]) + w * (math.log(b[1]) - math.log(a[1]))), flag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--targets", default="5.42")
    a = ap.parse_args()
    targets = [float(t) for t in a.targets.split(",")]
    for path in a.logs:
        groups = defaultdict(list)
        with open(path, encoding="utf-8") as f:
            for line in f:
                m = LINE.match(line)
                if m:
                    model, t, p, mode, ppl, sd, ent, ln = m.groups()
                    groups[(model, mode)].append((float(ent), float(ppl), f"t{t}/p{p}", float(sd), int(ln)))
        for (model, mode), pts in groups.items():
            pts.sort()
            print(f"\n{model}  [{mode}]")
            for ent, ppl, setting, sd, ln in pts:
                print(f"  {setting:12s} genPPL={ppl:7.2f} +/- {sd:4.2f} @ H={ent:.3f}  len={ln}")
            for h in targets:
                v, flag = at(pts, h)
                print(f"  -> genPPL @ H={h:.2f}: {v:.2f}{flag}")


if __name__ == "__main__":
    main()
