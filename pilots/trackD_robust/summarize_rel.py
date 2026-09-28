"""Summarize out/rel/rel.jsonl (+ RL(g.) and greedy rows of the trackD pilot, same release/noise seeds)."""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
rows = [json.loads(l) for l in (HERE / "out" / "rel" / "rel.jsonl").read_text().splitlines() if l.strip()]
SC = ["R1-batch", "R2-pois.5", "R3-pois.8", "R2+N3/causal"]
for l in (HERE.parent / "trackD" / "out" / "rows.jsonl").read_text().splitlines():
    r = json.loads(l)
    if r["scenario"] in SC and r["method"] in ("RL(g.)", "greedy"):
        r = dict(r)
        r["method"] = {"RL(g.)": "RLg", "greedy": "greedy"}[r["method"]]
        rows.append(r)
cell = defaultdict(dict)
per = defaultdict(list)
for r in rows:
    cell[(r["setting"], r["scenario"], r["inst"], r["noise_seed"])][r["method"]] = r["makespan"] if r["success"] else 200.0
    if "solver_s" in r and r.get("n_replans"):
        per[r["method"]].append((r["solver_s"] / r["n_replans"], r["n_replans"]))


def ratio(a, b, s=None, sc=None):
    x = [np.log(d[a] / d[b]) for (s_, sc_, i, ns), d in cell.items()
         if (s is None or s_ == s) and (sc is None or sc_ == sc) and a in d and b in d]
    if not x:
        return "-"
    x = np.array(x)
    bm = np.random.default_rng(0).choice(x, (4000, len(x))).mean(1)
    return f"{np.exp(x.mean()):.3f} [{np.exp(np.quantile(bm, .025)):.3f},{np.exp(np.quantile(bm, .975)):.3f}] " \
           f"{int((x < 0).sum()):2d}/{len(x):2d}"


PAIRS = [("RH-ce@3000", "ES-ce@3000", "re-plan at finishes too / release-only (3000 it)"),
         ("ES-ce@0", "ES-ce@3000", "insertion-only repair / 3000-it repair"),
         ("ES-ce@30", "ES-ce@3000", "C2: 30 it / 3000 it (release-only)"),
         ("ES-ce@300", "ES-ce@3000", "C2: 300 it / 3000 it (release-only)"),
         ("RH-ce@30", "ES-ce@30", "nervousness at 30 it: every event / release-only"),
         ("RH-ce@30", "RH-ce@3000", "RH 30 / 3000"),
         ("RLg", "ES-ce@30", "published RL(g.) online / ES 30 it"),
         ("RLg", "RH-ce@3000", "published RL(g.) online / RH 3000 it"),
         ("greedy", "ES-ce@30", "paper greedy (fixed) / ES 30 it")]
settings = sorted({k[0] for k in cell})
print(f"{len(rows)} rows; geometric-mean ratio [95% CI] wins/n (fail = 200)")
for a, b, lab in PAIRS:
    print(f"\n### {a} / {b}   ({lab})")
    print(f"{'setting':15s}" + "".join(f"{sc:>30s}" for sc in SC))
    for s in settings + [None]:
        print(f"{(s or 'ALL'):15s}" + "".join(f"{ratio(a, b, s, sc):>30s}" for sc in SC))
print("\nper re-plan solver ms / re-plans per episode:")
for m in sorted(per):
    v = np.array(per[m])
    print(f"  {m:12s} {1000 * v[:, 0].mean():7.1f} ms  {v[:, 1].mean():6.1f}")
succ = defaultdict(list)
for r in rows:
    succ[r["method"]].append(r["success"])
print("success:", {m: round(float(np.mean(v)), 3) for m, v in succ.items()})
