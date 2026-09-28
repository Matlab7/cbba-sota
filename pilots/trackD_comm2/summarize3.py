"""Wave-3 summary: DAR-r vs the strongest replicated control REP+ (= FRZ1 code path) and vs the simple-rule
collapse arms (INF-r: never give new work to unreachable robots; CSRc: constant delay c), per setting x range.
Geometric-mean ratios over (instance, seed), failure = 200; bootstrap 95% CI; paired t on log ratios; TOST +-3%."""
import json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy import stats

OUT = Path(__file__).resolve().parent / "out"
rows = []
for f in ["dar3.jsonl", "dar3_repplus.jsonl"] + sys.argv[1:]:
    rows += [json.loads(l) for l in (OUT / f).read_text().splitlines() if l.strip()]
for r in rows:
    if r["method"] == "FRZ1":
        r["method"] = "REP+"
    if r["method"] == "FRZ1+o":
        r["method"] = "REP+o"
def ms(r): return r["makespan"] if r["success"] else 200.0
full = {(r["setting"], r["inst"], r["seed"]): r for r in rows if r["method"] == "FULL"}
cell = defaultdict(dict)
for r in rows:
    if r["method"] != "FULL":
        cell[(r["setting"], r["R"])][(r["inst"], r["seed"], r["method"])] = r
def boot(x, B=4000):
    x = np.asarray(x); rng = np.random.default_rng(0); m = rng.choice(x, (B, len(x))).mean(1)
    return x.mean(), np.quantile(m, .025), np.quantile(m, .975)
def tost(d, marg=np.log(1.03)):
    d = np.asarray(d); n = len(d); se = d.std(ddof=1) / np.sqrt(n)
    p1 = 1 - stats.t.cdf((d.mean() + marg) / se, n - 1); p2 = stats.t.cdf((d.mean() - marg) / se, n - 1)
    return max(p1, p2)
arms = [m for m in ["REP", "REP+", "INF-r", "CSR2", "CSR4", "CSR8", "DAR-r", "REP+o", "DAR-r+o"]
        if any(r["method"] == m for r in rows)]
pooled = defaultdict(lambda: defaultdict(list))
pooledF = defaultdict(lambda: defaultdict(list))
for (s, R), c in sorted(cell.items(), key=lambda kv: (kv[0][0], -kv[0][1])):
    keys = sorted({(i, sd) for (i, sd, m) in c if all((i, sd, a) in c for a in arms if not (a.endswith("o") and R > 0.25))})
    print(f"\n{s} R={R} n={len(keys)} FULL {np.mean([ms(full[(s,i,sd)]) for i,sd in keys]):.2f}")
    for a in arms:
        if not all((i, sd, a) in c for i, sd in keys):
            continue
        v = [np.log(ms(c[(i, sd, a)]) / ms(full[(s, i, sd)])) for i, sd in keys]
        pooledF[R][a] += v
        su = np.mean([c[(i, sd, a)]["success"] for i, sd in keys])
        line = f"  {a:7s} /FULL {np.exp(np.mean(v)):.3f} succ {su:.2f} wasted {np.mean([c[(i,sd,a)]['wasted'] for i,sd in keys]):5.1f} vers {np.mean([c[(i,sd,a)]['versions'] for i,sd in keys]):7.0f}"
        if a != "DAR-r" and not a.endswith("o"):
            d = [np.log(ms(c[(i, sd, 'DAR-r')]) / ms(c[(i, sd, a)])) for i, sd in keys]
            pooled[R][a] += d
            mu, lo, hi = boot(d)
            line += f"   DAR-r/{a} {np.exp(mu):.3f} [{np.exp(lo):.3f},{np.exp(hi):.3f}] w{int(np.sum(np.array(d)<0))}/{len(d)}"
        print(line)
print("\nPOOLED over 4 settings")
for R in sorted(pooledF, reverse=True):
    print(f" R={R} /FULL: " + " ".join(f"{a}={np.exp(np.mean(v)):.3f}" for a, v in pooledF[R].items()))
    for a, d in pooled[R].items():
        mu, lo, hi = boot(d)
        print(f"    DAR-r/{a:6s} {np.exp(mu):.3f} [{np.exp(lo):.3f},{np.exp(hi):.3f}] wins {int(np.sum(np.array(d)<0))}/{len(d)} "
              f"p_sup={stats.ttest_1samp(d,0,alternative='less').pvalue:.3g} p_TOST3%={tost(d):.3g}")
