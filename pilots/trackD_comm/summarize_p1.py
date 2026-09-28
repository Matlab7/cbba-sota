"""Paired summary of a P1 run: per setting and range, each arm's makespan relative to G (global information,
same planner) and to the best non-O baseline, with bootstrap CIs over (instance, seed) pairs.

Usage: .venv/bin/python pilots/trackD_comm/summarize_p1.py [--tag p1]
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def boot(x, n=4000, seed=0):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    m = rng.choice(x, (n, len(x))).mean(1)
    return float(x.mean()), float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="p1")
    args = ap.parse_args()
    rows = [json.loads(line) for line in (HERE / "out" / f"{args.tag}.jsonl").read_text().splitlines()]
    by = defaultdict(dict)
    for r in rows:
        key = (r["setting"], r["inst"], r["seed"])
        name = "G" if not np.isfinite(r["R"]) else f"{r['arm']}-{r['lf']}"
        R = r["R"] if np.isfinite(r["R"]) else None
        by[key][(R, name)] = r
    settings = sorted({k[0] for k in by})
    radii = sorted({R for v in by.values() for (R, _) in v if R is not None}, reverse=True)
    names = sorted({n for v in by.values() for (R, n) in v if R is not None})
    lines = []
    for s in settings:
        keys = [k for k in by if k[0] == s]
        g = np.array([by[k][(None, "G")]["makespan"] for k in keys])
        lines.append(f"=== {s}  n={len(keys)}  G (global info, same planner) mean makespan {g.mean():.2f}")
        for R in radii:
            lines.append(f"  R={R}")
            ms = {n: np.array([by[k][(R, n)]["makespan"] if (R, n) in by[k] else np.nan for k in keys])
                  for n in names}
            base = [n for n in names if not n.startswith("O")]
            best_base = np.nanmin(np.vstack([ms[n] for n in base]), axis=0) if base else None
            # the best single baseline arm by mean (a fixed choice, not per-instance oracle)
            bb = min(base, key=lambda n: np.nanmean(ms[n]))
            for n in names:
                x = ms[n]
                if np.isnan(x).any():
                    continue
                succ = np.mean([by[k][(R, n)]["success"] for k in keys])
                st = np.mean([by[k][(R, n)]["st_conn"] for k in keys])
                wasted = np.mean([by[k][(R, n)]["wasted"] + by[k][(R, n)]["extras"] for k in keys])
                ab = np.mean([by[k][(R, n)]["abandons"] for k in keys])
                m, lo, hi = boot(x / g)
                m2, lo2, hi2 = boot(x / ms[bb])
                lines.append(f"    {n:10s} ms {x.mean():7.2f}  /G {m:.3f} [{lo:.3f},{hi:.3f}]  /{bb} {m2:.3f} "
                             f"[{lo2:.3f},{hi2:.3f}]  succ {succ:.2f} st_conn {st:.2f} wasted {wasted:5.1f} "
                             f"abandon {ab:5.1f}")
    text = "\n".join(lines)
    print(text)
    (HERE / "out" / f"{args.tag}_summary.txt").write_text(text + "\n")


if __name__ == "__main__":
    main()
