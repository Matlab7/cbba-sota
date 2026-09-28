"""Summarize integ_headroom.jsonl: paired ratios vs FULL and vs the best baseline, per stressor and range."""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

path = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).parent / "out" / "integ_headroom.jsonl")
rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
full = {(r["setting"], r["inst"], r["scen"], r["seed"]): r for r in rows if r["method"] == "FULL"}
cell = defaultdict(dict)
for r in rows:
    if r["method"] == "FULL":
        continue
    cell[(r["setting"], r["scen"], r["R"])][(r["inst"], r["seed"], r["method"])] = r


def ms(r):
    return r["makespan"] if r["success"] else 200.0


def boot(x, B=2000, seed=0):
    x = np.asarray(x)
    rng = np.random.default_rng(seed)
    m = rng.choice(x, (B, len(x))).mean(1)
    return x.mean(), np.quantile(m, 0.025), np.quantile(m, 0.975)


methods = ["OPEN", "CEN-F", "CEN-R", "REP", "LOC-F"]
scens = sorted({r["scen"] for r in rows})
settings = sorted({r["setting"] for r in rows})
radii = sorted({r["R"] for r in rows if r["method"] != "FULL"}, reverse=True)
print(f"{len(rows)} rows. Ratio = makespan / FULL (global-comm re-planning, same planner), geometric mean over"
      " instance x seed; fail counts as 200")
for sc in scens:
    print(f"\n=== {sc}")
    print("setting         R     " + "  ".join(f"{m:>12s}" for m in methods) + "   LOC/bestBase [95% CI] wins  off-st")
    agg = defaultdict(lambda: defaultdict(list))
    for s in settings:
        for R in radii:
            c = cell.get((s, sc, R))
            if not c:
                continue
            keys = sorted({(i, sd) for (i, sd, m) in c})
            vals = {m: [] for m in methods}
            best_base, loc = [], []
            offst = []
            for (i, sd) in keys:
                f = full.get((s, i, sc, sd))
                if f is None or any((i, sd, m) not in c for m in methods):
                    continue
                for m in methods:
                    vals[m].append(np.log(ms(c[(i, sd, m)]) / ms(f)))
                bb = min(ms(c[(i, sd, m)]) for m in ("CEN-F", "CEN-R", "REP"))
                best_base.append(bb)
                loc.append(ms(c[(i, sd, "LOC-F")]))
                lr = c[(i, sd, "LOC-F")]
                offst.append(lr["versions_off_station"] / max(lr["versions"], 1))
            if not loc:
                continue
            # per-cell best baseline by mean (fixed per cell, not per instance)
            means = {m: np.mean(vals[m]) for m in ("CEN-F", "CEN-R", "REP")}
            bm = min(means, key=means.get)
            lr = [np.log(c[(i, sd, "LOC-F")]["makespan"] if c[(i, sd, "LOC-F")]["success"] else 200.0) -
                  np.log(ms(c[(i, sd, bm)])) for (i, sd) in keys if (i, sd, bm) in c and (i, sd, "LOC-F") in c]
            mu, lo, hi = boot(lr)
            wins = int(np.sum(np.array(lr) < 0))
            succ = {m: np.mean([c[(i, sd, m)]["success"] for (i, sd) in keys]) for m in methods}
            txt = "  ".join(f"{np.exp(np.mean(vals[m])):6.3f}({succ[m]:.2f})" for m in methods)
            print(f"{s:15s} {R:4.2f}  {txt}   {np.exp(mu):.3f} [{np.exp(lo):.3f},{np.exp(hi):.3f}] vs {bm:5s} "
                  f"{wins}/{len(lr)}  {np.mean(offst):.2f}")
            for m in methods:
                agg[R][m] += vals[m]
            agg[R]["loc_vs_best"] += lr
    print("ALL settings:")
    for R in radii:
        a = agg[R]
        if not a["LOC-F"]:
            continue
        txt = "  ".join(f"{np.exp(np.mean(a[m])):12.3f}" for m in methods)
        mu, lo, hi = boot(a["loc_vs_best"])
        print(f"{'':15s} {R:4.2f}  {txt}   LOC/best-cell-baseline {np.exp(mu):.3f} [{np.exp(lo):.3f},{np.exp(hi):.3f}]")
