"""Summarize the robustness-angle RH budget pilot (out/c2/it{30,300,3000}.jsonl) plus RL(g.) rows from the trackD pilot.

All methods face the same realization (common random numbers: dyn_env keys (noise_seed, inst_key)), so ratios are
paired per (setting, inst, scenario, noise_seed). Reported: geometric mean of per-pair ratios with a bootstrap 95% CI
and win counts. Fail counts as makespan 200.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
C2 = HERE / "out" / "c2"


def load(p, tag=None):
    rows = []
    for line in Path(p).read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if tag and r["method"].startswith("RH"):
                r["method"] = f"{r['method']}@{tag}"
            elif r["method"].startswith("RH"):
                r["method"] = f"{r['method']}@{r['iters']}"
            rows.append(r)
    return rows


rows = []
for it in (3000, 300, 30):
    f = C2 / f"it{it}.jsonl"
    if f.exists():
        rows += load(f)
rl = [json.loads(l) for l in (HERE.parent / "trackD" / "out" / "rows.jsonl").read_text().splitlines() if l.strip()]
for r in rl:
    if r["method"] in ("RL(g.)", "ALNS-ol") and r["scenario"] in ("N2-dur.3/causal", "N3-mod/causal", "N4-sev/causal"):
        r = dict(r)
        r["method"] = "RLg" if r["method"] == "RL(g.)" else "OL-5s"
        rows.append(r)

cell = defaultdict(dict)
per = defaultdict(list)
for r in rows:
    ms = r["makespan"] if r["success"] else 200.0
    cell[(r["setting"], r["scenario"], r["inst"], r["noise_seed"])][r["method"]] = ms
    if r["method"].startswith("RH"):
        per[r["method"]].append(r["solver_s"] / max(r["n_replans"], 1))


def ratio(a, b, setting=None, scen=None):
    x = []
    for (s, sc, i, ns), d in cell.items():
        if (setting and s != setting) or (scen and sc != scen):
            continue
        if a in d and b in d:
            x.append(np.log(d[a] / d[b]))
    if not x:
        return None
    x = np.array(x)
    rng = np.random.default_rng(0)
    bm = rng.choice(x, (4000, len(x))).mean(1)
    return np.exp(x.mean()), np.exp(np.quantile(bm, .025)), np.exp(np.quantile(bm, .975)), int((x < 0).sum()), len(x)


def fmt(v):
    return "        -          " if v is None else f"{v[0]:.3f} [{v[1]:.3f},{v[2]:.3f}] {v[3]:2d}/{v[4]:2d}"


PAIRS = [
    ("OL-5s", "RH-ce@3000", "open-loop 5s plan / reactive RH"),
    ("OL-ce", "RH-ce@3000", "open-loop (same t=0 planner) / RH"),
    ("RH-nom@3000", "RH-ce@3000", "RH nominal / RH calibrated"),
    ("RH-ce@3000", "RH-orc@3000", "RH calibrated / RH perfect duration foresight (proactive ceiling)"),
    ("RH-ce@300", "RH-ce@3000", "C2: 300 it / 3000 it per event"),
    ("RH-ce@30", "RH-ce@3000", "C2: 30 it / 3000 it per event"),
    ("RH-ce@30", "RH-ce@300", "C2: 30 it / 300 it"),
    ("RLg", "RH-ce@3000", "published RL(g.) online / RH calibrated 3000"),
    ("RLg", "RH-ce@30", "published RL(g.) online / RH calibrated 30"),
]


def resolve(m, scen):
    # N2 has no -ce rows (kappa = 1): -ce is identical to -nom there
    if scen.startswith("N2") and "-ce" in m:
        return m.replace("-ce", "-nom")
    return m


settings = sorted({k[0] for k in cell})
scens = ["N2-dur.3/causal", "N3-mod/causal", "N4-sev/causal"]
print(f"{len(rows)} rows. geometric-mean ratio [95% boot CI] wins(ratio<1)/n")
for a, b, label in PAIRS:
    print(f"\n### {a} / {b}  ({label})")
    print(f"{'setting':15s} " + "  ".join(f"{sc.split('/')[0]:>28s}" for sc in scens))
    for s in settings + [None]:
        line = f"{(s or 'ALL'):15s} "
        for sc in scens:
            line += "  " + f"{fmt(ratio(resolve(a, sc), resolve(b, sc), s, sc)):>28s}"
        print(line)
print("\nsolver seconds per re-plan (1 core, loaded host), mean:")
for m in sorted(per):
    print(f"  {m:14s} {np.mean(per[m]) * 1000:8.1f} ms  (n={len(per[m])})")
if len(sys.argv) > 1:
    json.dump({"note": "see summarize_c2.py"}, open(sys.argv[1], "w"))
