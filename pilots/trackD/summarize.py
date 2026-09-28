"""Summary of pilots/trackD/out/rows.jsonl: degradation of each method vs its own static run (paired by instance),
cross-method ratios inside each scenario, oracle-vs-causal observation effect, and per-decision cost."""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
rows = [json.loads(line) for line in open(sys.argv[1] if len(sys.argv) > 1 else HERE / "out" / "rows.jsonl")]
B = 4000
rng = np.random.default_rng(0)


def boot(per_inst: dict[int, list[float]]):
    """Mean over instances of the per-instance mean; 95% percentile bootstrap over instances."""
    keys = sorted(per_inst)
    v = np.array([np.mean(per_inst[k]) for k in keys])
    idx = rng.integers(len(v), size=(B, len(v)))
    bs = v[idx].mean(1)
    return v.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5), len(v)


by = defaultdict(dict)  # (setting, scenario, method) -> {(inst, seed): row}
for r in rows:
    by[(r["setting"], r["scenario"], r["method"])][(r["inst"], r["noise_seed"])] = r

settings = sorted({r["setting"] for r in rows})
scen_order = []
for r in rows:
    if r["scenario"] not in scen_order:
        scen_order.append(r["scenario"])
scen_order = ["static"] + sorted(s for s in scen_order if s != "static")
methods = ["RL(g.)", "RL(s.1)", "greedy", "ALNS-ol", "ALNS-cv"]

for s in settings:
    print(f"\n=== {s} ===")
    print(f"{'scenario':18s} {'method':8s} {'mean ms':>8s} {'succ':>5s}  {'ratio vs own static [95% CI]':>32s}"
          f"  {'RL(g.)/this':>22s}")
    for sc in scen_order:
        for m in methods:
            d = by.get((s, sc, m))
            if not d:
                continue
            ms = np.mean([r["makespan"] for r in d.values()])
            succ = np.mean([r["success"] for r in d.values()])
            st = by.get((s, "static", "ALNS-ol" if m == "ALNS-cv" else m), {})
            per = defaultdict(list)
            for (i, ns), r in d.items():
                if (i, 0) in st:
                    per[i].append(r["makespan"] / st[(i, 0)]["makespan"])
            deg = "" if not per else "{:.3f} [{:.3f}, {:.3f}] n={}".format(*boot(per))
            rl = by.get((s, sc, "RL(g.)"), {})
            per2 = defaultdict(list)
            for k, r in d.items():
                if k in rl:
                    per2[k[0]].append(rl[k]["makespan"] / r["makespan"])
            cmp = "" if m == "RL(g.)" or not per2 else "{:.3f} [{:.3f}, {:.3f}]".format(*boot(per2)[:3])
            print(f"{sc:18s} {m:8s} {ms:8.2f} {succ:5.2f}  {deg:>32s}  {cmp:>22s}")

print("\n=== observation leak: RL(g.) oracle / causal (paired by instance x noise seed) ===")
for s in settings:
    for sc in scen_order:
        if not sc.endswith("/oracle"):
            continue
        o, c = by.get((s, sc, "RL(g.)"), {}), by.get((s, sc.replace("/oracle", "/causal"), "RL(g.)"), {})
        per = defaultdict(list)
        for k in o:
            if k in c:
                per[k[0]].append(o[k]["makespan"] / c[k]["makespan"])
        if per:
            print(f"{s:14s} {sc[:-7]:10s} oracle/causal {boot(per)[0]:.3f} [{boot(per)[1]:.3f}, {boot(per)[2]:.3f}]")

print("\n=== per-decision cost of RL(g.) (single thread, loaded host) ===")
for s in settings:
    for sc in scen_order:
        d = by.get((s, sc, "RL(g.)"), {})
        if not d:
            continue
        n = sum(r["n_dec"] for r in d.values())
        to = sum(r["t_obs"] for r in d.values()) / n * 1e3
        tn = sum(r["t_net"] for r in d.values()) / n * 1e3
        polls = np.mean([r["release_polls"] for r in d.values()])
        print(f"{s:14s} {sc:18s} decisions/episode {n / len(d):6.1f}  obs {to:5.2f} ms  net {tn:5.2f} ms  "
              f"release re-polls/episode {polls:5.1f}")
