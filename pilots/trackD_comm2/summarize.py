"""Paired summary of the comm-first DAR pilot: every arm relative to FULL (same planner, global instant comm) and
DAR relative to each baseline, per setting and range (geometric means over instance x seed, failure = 200).

Usage: .venv/bin/python pilots/trackD_comm2/summarize.py out/dar.jsonl [out/collapse.jsonl ...]
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
files = sys.argv[1:] or [str(HERE / "out" / "dar.jsonl")]
rows = []
for f in files:
    p = Path(f) if Path(f).is_absolute() else HERE / f
    rows += [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def ms(r):
    return r["makespan"] if r["success"] else 200.0


full = {(r["setting"], r["inst"], r["seed"]): r for r in rows if r["method"] == "FULL"}
cell = defaultdict(dict)
for r in rows:
    if r["method"] != "FULL":
        cell[(r["setting"], r["R"])][(r["inst"], r["seed"], r["method"])] = r
methods = sorted({r["method"] for r in rows if r["method"] != "FULL"},
                 key=lambda m: ["DAR", "DAR-r", "REP", "LOC-F", "CEN-F"].index(m) if m in
                 ["DAR", "DAR-r", "REP", "LOC-F", "CEN-F"] else 10)


def boot(x, B=4000, seed=0):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    m = rng.choice(x, (B, len(x))).mean(1)
    return x.mean(), np.quantile(m, 0.025), np.quantile(m, 0.975)


def tstat(x):
    x = np.asarray(x, float)
    from scipy import stats
    return stats.ttest_1samp(x, 0.0).pvalue


pooled = defaultdict(lambda: defaultdict(list))
for (s, R) in sorted(cell, key=lambda k: (k[0], -k[1])):
    c = cell[(s, R)]
    keys = sorted({(i, sd) for (i, sd, m) in c if (s, i, sd) in full and all((i, sd, mm) in c for mm in methods)})
    if not keys:
        continue
    print(f"\n{s}  R={R}  n={len(keys)}  FULL mean {np.mean([ms(full[(s, i, sd)]) for i, sd in keys]):.2f}")
    for m in methods:
        lr = [np.log(ms(c[(i, sd, m)]) / ms(full[(s, i, sd)])) for i, sd in keys]
        succ = np.mean([c[(i, sd, m)]["success"] for i, sd in keys])
        w = np.mean([c[(i, sd, m)]["wasted"] for i, sd in keys])
        ab = np.mean([c[(i, sd, m)]["abandons"] for i, sd in keys])
        line = f"  {m:6s} /FULL {np.exp(np.mean(lr)):.3f}  succ {succ:.2f}  wasted {w:5.1f}  abandons {ab:5.1f}"
        if m != "DAR" and "DAR" in methods:
            d = [np.log(ms(c[(i, sd, 'DAR')]) / ms(c[(i, sd, m)])) for i, sd in keys]
            mu, lo, hi = boot(d)
            wins = int(np.sum(np.array(d) < 0))
            line += f"   DAR/{m} {np.exp(mu):.3f} [{np.exp(lo):.3f},{np.exp(hi):.3f}] wins {wins}/{len(d)} p={tstat(d):.3g}"
            pooled[R][m] += d
        print(line)
print("\nPOOLED over settings (log-ratio DAR/arm):")
for R in sorted(pooled, reverse=True):
    for m, d in pooled[R].items():
        mu, lo, hi = boot(d)
        print(f"  R={R:<5} DAR/{m:6s} {np.exp(mu):.3f} [{np.exp(lo):.3f},{np.exp(hi):.3f}] wins "
              f"{int(np.sum(np.array(d) < 0))}/{len(d)} p={tstat(d):.3g}")
