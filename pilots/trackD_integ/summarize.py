"""Summaries of the Track D integrated-proposer pilots (2026-09-28). Writes out/summary.txt.

All runs share the integ_headroom.py world (pilots/trackD): HeteroMRTA dev instances (4 settings x 10, 2 CRN seeds),
Poisson release (dod 0.5) + lognormal durations sigma 0.3 + LoRR-2026 delays (RN12: example 0.01/1-4 ticks; RN3:
0.05/1-10), unit-disk range R + 20% per-link per-tick Bernoulli loss, station at the centre, same list-scheduling
planner for every method. Ratios are geometric means of per-(instance, seed) makespan ratios (failure = 200),
bootstrap 95% CIs over pairs.

Blocks:
  P2  cas_pilot.jsonl          consistency variants vs REP, default lease (grace 1, max 10)
  P3  decomp_pilot*.jsonl      knowledge oracles (release / task status / teammate state / all)
  P4  pb_pilot.jsonl           optimistic dead reckoning (PB-A, stopped early) ; clamp_pilot.jsonl (CLAMP)
  P5  beacon_pilot.jsonl       low-rate global state beacon (period P, loss q)
  P6  lease_pilot.jsonl        lease patience (grace, max)
  P7  tuned_*.jsonl, sg_pilot  every method with the same patient lease (grace 3, max 30); staleness gate SG<tau>
"""
from __future__ import annotations

import io
import json
from collections import defaultdict
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "out"


def load(name, keep=None):
    p = OUT / name
    if not p.exists():
        return []
    rows = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
    return [r for r in rows if keep is None or r["method"] in keep]


def ms(r):
    return r["makespan"] if r["success"] else 200.0


def boot(x, B=4000):
    x = np.asarray(x, float)
    m = np.random.default_rng(0).choice(x, (B, len(x))).mean(1)
    return float(x.mean()), float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def fmt(v):
    mu, lo, hi = boot(v)
    return f"{np.exp(mu):.3f}[{np.exp(lo):.3f},{np.exp(hi):.3f}]"


def block(title, rows, methods, full_method, ref, radii=(0.3, 0.2, 0.15, 0.1)):
    print(f"\n=== {title}\n    ratio to {full_method} (geo-mean) | paired ratio to {ref} [95% CI] | success")
    full = {(r["setting"], r["scen"], r["inst"], r["seed"]): r for r in rows if r["method"] == full_method}
    cell = defaultdict(dict)
    for r in rows:
        if r["method"] != full_method:
            cell[(r["setting"], r["scen"], r["R"])][(r["inst"], r["seed"], r["method"])] = r
    pool = defaultdict(lambda: defaultdict(list))
    pref = defaultdict(lambda: defaultdict(list))
    succ = defaultdict(lambda: defaultdict(list))
    for (s, sc, R), c in cell.items():
        if R not in radii:
            continue
        keys = sorted({(i, sd) for (i, sd, m) in c
                       if all((i, sd, mm) in c for mm in methods) and (s, sc, i, sd) in full})
        for m in methods:
            for i, sd in keys:
                pool[(sc, R)][m].append(np.log(ms(c[(i, sd, m)]) / ms(full[(s, sc, i, sd)])))
                pref[(sc, R)][m].append(np.log(ms(c[(i, sd, m)]) / ms(c[(i, sd, ref)])))
                succ[(sc, R)][m].append(c[(i, sd, m)]["success"])
    for k in sorted(pool, key=lambda k: (k[0], -k[1])):
        sc, R = k
        n = len(pool[k][ref])
        print(f"  {sc:4s} R={R:4.2f} n={n:3d} " + "  ".join(f"{m}:{np.exp(np.mean(v)):.3f}" for m, v in pool[k].items()))
        print("             vs ref: " + "  ".join(f"{m}:{fmt(v)}" for m, v in pref[k].items() if m != ref))
        print("             succ:   " + "  ".join(f"{m}:{np.mean(v):.2f}" for m, v in succ[k].items()))


def main():
    buf = io.StringIO()
    with redirect_stdout(buf):
        base = load("cas_pilot.jsonl")
        block("P2 consistency variants, default lease (grace 1, max 10)", base,
              ["CEN-F", "REP", "CAS-L", "CAS-INF", "CAS-1"], "FULL", "REP")
        dec = [r for r in base if r["scen"] == "RN12" and r["method"] in ("FULL", "REP", "CEN-F")]
        dec += load("decomp_pilot.jsonl") + load("decomp_pilot2.jsonl")
        block("P3 knowledge oracles (RN12), default lease", dec,
              ["CEN-F", "REP", "REP-RO", "REP-SO", "REP-AO", "REP-KO"], "FULL", "REP", radii=(0.2, 0.15, 0.1))
        pb = [r for r in base if r["method"] in ("FULL", "REP", "CEN-F")] + load("clamp_pilot.jsonl")
        block("P4 clamped teammate snapshot, default lease", pb, ["CEN-F", "REP", "CLAMP"], "FULL", "REP",
              radii=(0.2, 0.15, 0.1))
        pba = [r for r in base if r["method"] in ("FULL", "REP", "CEN-F")] + load("pb_pilot.jsonl", {"PB-A"})
        block("P4 optimistic dead reckoning PB-A (partial, stopped: harmful)", pba, ["CEN-F", "REP", "PB-A"], "FULL",
              "REP", radii=(0.2, 0.15, 0.1))
        bc = [r for r in base if r["method"] in ("FULL", "REP", "CEN-F")] + load("beacon_pilot.jsonl")
        block("P5 low-rate global state beacon REP-B<period>:<loss>", bc,
              ["CEN-F", "REP", "REP-B1:0.2", "REP-B2:0.5", "REP-B5:0.2"], "FULL", "REP", radii=(0.2, 0.15, 0.1))
        ls = [r for r in base if r["method"] in ("FULL", "REP", "CEN-F")] + load("lease_pilot.jsonl")
        block("P6 lease patience REP-G<grace>:<max>", ls, ["CEN-F", "REP", "REP-G0.3:10", "REP-G3:10", "REP-G3:30"],
              "FULL", "REP", radii=(0.2, 0.15, 0.1))
        tuned = load("tuned_pilot.jsonl") + load("tuned_clamp.jsonl") + load("lease_pilot.jsonl", {"REP-G3:30"})
        tuned += [r for r in load("tuned_consistency.jsonl") if "~G" in r["method"]] + load("sg_pilot.jsonl")
        tm = ["CEN-F~G3:30", "REP-G3:30", "CLAMP~G3:30", "LOC-F~G3:30", "CAS-INF~G3:30", "CAS-1~G3:30",
              "OPEN~G3:30"]
        block("P7 equal infrastructure: every method with lease grace 3, max 30", tuned, tm, "FULL~G3:30",
              "CEN-F~G3:30", radii=(0.2, 0.15, 0.1))
        sg = ["CEN-F~G3:30", "REP-G3:30", "CLAMP~G3:30", "SG2~G3:30", "SG5~G3:30", "SG10~G3:30"]
        block("P7 staleness-gated replicated re-planning SG<tau> (lease 3/30)", tuned, sg, "FULL~G3:30",
              "CEN-F~G3:30", radii=(0.2, 0.15, 0.1))
        hy = tuned + load("hyb_pilot.jsonl") + load("tuned_r03.jsonl") + load("tuned_clamp_r03.jsonl")
        hy += load("hyb2_pilot.jsonl") + load("hyb3_pilot.jsonl")
        for r in hy:
            if r["method"] == "REP~G3:30":
                r["method"] = "REP-G3:30"
        block("P8 HYB: station plans its component only, other components optimistic + clamped (lease 3/30)", hy,
              ["CEN-F~G3:30", "CLAMP~G3:30", "HYB~G3:30", "HYB2~G3:30", "HYB30.5~G3:30", "HYB30.7~G3:30"],
              "FULL~G3:30", "CEN-F~G3:30",
              radii=(0.3, 0.2, 0.15, 0.1))
    text = buf.getvalue()
    (OUT / "summary.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
