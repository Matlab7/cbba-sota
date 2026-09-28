"""Track D week-1 memo: independent re-computation of the gate numbers from the raw rows (dev split only).

This script does not import ``cbba_sota.dyn.stats`` or the agents' analysis scripts. It re-reads the JSONL rows
under ``runs/trackD`` and recomputes, with its own pairing and its own bootstrap (numpy only):

    c1     C1-dyn and baseline ratios SPARC-L / method (T1/K1, K2, K3, K4, B6), pooled F1-F3 and per setting
    c3     C3 regret table, DS, K5 (4 settings x rho {2, 1, 0.5}; and the 5-level grid on the first two settings)
    c3bs   exploratory C3 sensitivity: regret on episodes where all six T2 arms succeed (no 200 imputation)
    k0     recount of the K0 comparison rows
    cpu    per-event CPU percentiles of SPARC-L / SPARC-H at 50 and 200 tasks (loaded host, descriptive)
    rlcost cost of dynamics for RL(g.) on the K0-verified env (same policy seed as its static run)

Unit of analysis (spec 9.1): the instance mean of log r over its paired (cell, CRN seed) episodes where both
methods succeed; GM = exp(mean of instance means); 95% CI = bootstrap of instances within settings (10,000
resamples, own RNG, so CI ends can differ from the reports' by about 1e-4).

    .venv/bin/python scripts/trackD_week1_verify.py [c1 c3 c3bs k0 cpu rlcost]
"""
from __future__ import annotations

import collections
import glob
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs" / "trackD"
PRIMARY = ["MA-AT-25-5-50", "MA-AT-50-5-50", "SA-AT-50-5-50", "SA-BT-50-5-50"]
FIXED = ["CEN-F", "HYB", "REP-clamp", "INF-r"]
C3_ARMS = FIXED + ["SPARC", "SPARC-V3"]
C3_CELLS = ["F1-R2", "F3-pf0.2"]


def _rows(pattern: str) -> list[dict]:
    out = []
    for f in sorted(glob.glob(str(RUNS / pattern))):
        with open(f) as fh:
            out += [json.loads(line) for line in fh]
    return out


def _label(r: dict, b4eq: dict) -> str:
    m, tier, coal = r["method"], r.get("tier"), r.get("options", {}).get("coalition")
    if m == "SPARC":
        return "SPARC-L" if tier == "light" else "SPARC-H"
    if m.startswith("B4?restarts=") and "&" not in m:
        k = int(m.split("=")[1])
        return "B4-eq" if b4eq.get(r["setting"]) == k else f"B4-r{k}"
    if m.startswith("ctrl:cbba_sota.dyn.baselines.rl_mpc"):
        return "B2x3" if "scale=3" in m else "B2"
    if m.startswith("ctrl:cbba_sota.dyn.baselines.cpsat_tiers"):
        return f"B3-{tier}"
    if m.startswith("ctrl:cbba_sota.dyn.baselines.cbta"):
        return "B6:" + m.split("?", 1)[1]
    if m in ("RL(g.)", "RL(s.1)", "greedy"):
        return m + ("-arr" if coal == "arrival" else "")
    return m


def _index(rows: list[dict], seeds: set[int], b4eq: dict) -> dict:
    d: dict = {}
    for r in rows:
        if r["seed"] not in seeds or r.get("status") != "completed":
            continue
        key = (_label(r, b4eq), r["setting"], r["instance"], r["cell"], r["seed"])
        val = (r["makespan"], bool(r["success"]), r["family"])
        old = d.get(key)
        if old is not None and old[0] is not None and val[0] is not None and abs(old[0] - val[0]) > 1e-9:
            print("WARNING duplicate case with a different makespan:", key, old[0], val[0])
        d[key] = val
    return d


def _boot_ci(means: np.ndarray, strata: np.ndarray, reps: int = 10_000, seed: int = 20260928) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    tot = np.zeros(reps)
    for s in np.unique(strata):
        idx = np.where(strata == s)[0]
        tot += means[idx][rng.integers(0, idx.size, size=(reps, idx.size))].sum(1)
    lo, hi = np.quantile(tot / means.size, [0.025, 0.975])
    return math.exp(lo), math.exp(hi)


def _ratio(d: dict, a: str, b: str, families=("F1", "F2", "F3"), settings=PRIMARY) -> dict | None:
    obs: dict = collections.defaultdict(list)
    fail_a = fail_b = pairs = 0
    for (lab, s, i, cell, sd), (mk, ok, fam) in d.items():
        if lab != a or s not in settings or fam not in families:
            continue
        other = d.get((b, s, i, cell, sd))
        if other is None:
            continue
        fail_a += not ok
        fail_b += not other[1]
        if ok and other[1]:
            obs[(s, i)].append(math.log(mk) - math.log(other[0]))
            pairs += 1
    if not obs:
        return None
    keys = sorted(obs)
    means = np.array([np.mean(obs[k]) for k in keys])
    lo, hi = _boot_ci(means, np.array([settings.index(k[0]) for k in keys]))
    per = {s: math.exp(np.mean([m for m, k in zip(means, keys) if k[0] == s])) for s in settings
           if any(k[0] == s for k in keys)}
    return {"r": math.exp(means.mean()), "lo": lo, "hi": hi, "pairs": pairs, "clusters": len(keys),
            "wins": int((means < 0).sum()), "fail_a": fail_a, "fail_b": fail_b, "per": per}


def _show(name: str, res: dict | None) -> None:
    if res is None:
        print(f"  {name:38s} no pairs")
        return
    per = " ".join(f"{v:.4f}" for v in res["per"].values())
    print(f"  {name:38s} {res['r']:.4f} [{res['lo']:.4f}, {res['hi']:.4f}]  pairs {res['pairs']:4d}  "
          f"clusters {res['clusters']:2d}  wins {res['wins']:2d}  failures (a, b) {res['fail_a']}, {res['fail_b']}"
          f"  | per setting {per}")


def c1() -> None:
    with open(RUNS / "c1dyn" / "b4_budget_primary_frozen.json") as fh:
        frozen = json.load(fh)["settings"]
    b4eq = {s: frozen[s]["restarts"] for s in PRIMARY}
    rows = _rows("c1dyn/rows_*.jsonl")
    extra = [r for r in _rows("baselines_d/rows_*.jsonl") if not r["method"].startswith(("RL(g.)", "SPARC"))]
    print(f"C1-dyn rows {len(rows)}, status {dict(collections.Counter(r.get('status') for r in rows))}; "
          f"env hashes {dict(collections.Counter(r.get('code_hash') for r in rows + extra))}; B4-eq {b4eq}")
    d01 = _index(rows + extra, {0, 1}, b4eq)
    print("SPARC-L / method, F1-F3 pooled, CRN seeds 0-1 (settings: " + ", ".join(PRIMARY) + ")")
    for b in ("B4-eq", "B4-r12", "B5", "B6:rule=cbta&objective=start", "B6:rule=cbta&objective=makespan",
              "B6:rule=seq", "B3-light", "ins-only", "open-loop", "every-event", "SPARC-H", "RL(g.)",
              "RL(g.)-arr", "RL(s.1)", "RL(s.1)-arr", "greedy"):
        _show(b, _ratio(d01, "SPARC-L", b))
    print("Gate views")
    _show("K2: ins-only on F3", _ratio(d01, "SPARC-L", "ins-only", families=("F3",)))
    _show("K2: B4-eq on F1", _ratio(d01, "SPARC-L", "B4-eq", families=("F1",)))
    _show("T3: open-loop on F0", _ratio(d01, "SPARC-L", "open-loop", families=("F0",)))
    _show("B6(start) / B4-eq", _ratio(d01, "B6:rule=cbta&objective=start", "B4-eq"))
    d0 = _index(rows + extra, {0}, b4eq)
    print("Heavy-tier subsamples (CRN seed 0)")
    _show("K3: B2 RL-MPC Heavy", _ratio(d0, "SPARC-L", "B2"))
    _show("B2 / RL(g.)", _ratio(d0, "B2", "RL(g.)"))
    _show("B2 at 3x budget", _ratio(d0, "SPARC-L", "B2x3"))
    _show("K4: B3 CP-SAT-LNS Heavy", _ratio(d0, "SPARC-L", "B3-heavy"))
    _show("SPARC-H / B3-Heavy", _ratio(d0, "SPARC-H", "B3-heavy"))
    d23 = _index(rows + _rows("baselines_d/rows_*.jsonl"), {2, 3}, b4eq)
    print("Replication on CRN seeds 2-3 (exploratory)")
    _show("B4-eq", _ratio(d23, "SPARC-L", "B4-eq"))
    _show("B6(start)", _ratio(d23, "SPARC-L", "B6:rule=cbta&objective=start"))
    _show("SPARC-rk", _ratio(d23, "SPARC-L", "SPARC-rk"))
    _show("SPARC-rk / B4-eq", _ratio(d23, "SPARC-rk", "B4-eq"))


def _c3_values(rows: list[dict], rhos: list[float], settings: list[str] | None, both_succeed: bool):
    d = {}
    for r in rows:
        if settings and r["setting"] not in settings:
            continue
        if r.get("excluded") or r.get("status") not in (None, "ok"):
            continue
        rho = "inf" if r["arm"] == "FULL" else r["rho"]
        if rho != "inf" and rho not in rhos:
            continue
        d[(r["arm"], r["cell"], rho, r["setting"], r["inst"], r["seed"])] = (bool(r["success"]), r["makespan"])
    vals = collections.defaultdict(lambda: collections.defaultdict(list))
    for c in C3_CELLS:
        for rho in rhos + ["inf"]:
            arms = ["FULL"] if rho == "inf" else C3_ARMS
            eps = {k[3:] for k in d if k[1] == c and k[2] == rho}
            for e in eps:
                if both_succeed and not all(d.get((a, c, rho) + e, (False, 0))[0] for a in arms):
                    continue
                for a in arms:
                    ok, mk = d[(a, c, rho) + e]
                    vals[(a, c, rho)][e[:2]].append(math.log(mk if ok else 200.0))
    return vals


def _c3_table(rows: list[dict], rhos: list[float], settings: list[str] | None, both_succeed: bool = False) -> None:
    vals = _c3_values(rows, rhos, settings, both_succeed)

    def gm(key):
        return math.exp(np.mean([np.mean(v) for v in vals[key].values()]))

    reg = {}
    for c in C3_CELLS:
        for rho in rhos:
            g = {a: gm((a, c, rho)) for a in C3_ARMS}
            best = min(g[a] for a in FIXED)
            reg.update({(a, c, rho): g[a] / best for a in C3_ARMS})
    head = " ".join(f"{c[:5]}@{r:<4}" for c in C3_CELLS for r in rhos)
    print(f"  {'arm':10s} {head}  max regret")
    for a in C3_ARMS:
        xs = [reg[(a, c, r)] for c in C3_CELLS for r in rhos]
        print(f"  {a:10s} " + " ".join(f"{x:10.3f}" for x in xs) + f"  {max(xs):.3f}")
    ds = []
    for rs in sorted(rhos) + [math.inf]:
        m = max(reg[("CEN-F" if r >= rs else "REP-clamp", c, r)] for c in C3_CELLS for r in rhos)
        ds.append(f"rho*={rs}: {m:.3f}")
    print("  DS (CEN-F if rho >= rho* else REP-clamp) max regret: " + ", ".join(ds))
    if not both_succeed:
        lo = min(rhos)
        for c in C3_CELLS:
            full = vals[("FULL", c, "inf")]
            out = []
            for a in FIXED:
                arm = vals[(a, c, lo)]
                diffs = [np.mean(arm[k]) - np.mean(full[k]) for k in arm if k in full]
                out.append(f"{a} {math.exp(np.mean(diffs)):.3f}")
            print(f"  K5 {c} at rho={lo}, GM ratio to FULL: " + ", ".join(out))


def c3(both_succeed: bool = False) -> None:
    rows = _rows("c3/eval_final.jsonl")
    print(f"C3 rows {len(rows)}; c3 hashes {dict(collections.Counter(r['c3_hash'] for r in rows))}; "
          f"env hashes {dict(collections.Counter(r['env_hash'] for r in rows))}")
    tag = " (episodes where all six T2 arms succeed; exploratory)" if both_succeed else " (failures scored 200)"
    print("4 settings x rho {2, 1, 0.5}" + tag)
    _c3_table(rows, [2.0, 1.0, 0.5], None, both_succeed)
    if not both_succeed:
        print("MA-AT-25 + SA-BT-50 x rho {2, 1.25, 1, 0.75, 0.5}" + tag)
        _c3_table(rows, [2.0, 1.25, 1.0, 0.75, 0.5], ["MA-AT-25-5-50", "SA-BT-50-5-50"], both_succeed)
        t2 = [r for r in rows if r["arm"] != "FULL" and r["rho"] in (2.0, 1.0, 0.5)]
        fails = collections.Counter(r["setting"] for r in t2 if not r["success"])
        print(f"  T2 failures (rho 2/1/0.5): {sum(fails.values())} of {len(t2)}: {dict(fails)}")


def k0() -> None:
    rows = _rows("k0/rows.jsonl")
    hashes = collections.Counter((r["code_hash"], r["executor_hash"], r["controller_hash"]) for r in rows)
    fam = collections.Counter((r["family"], r["executor"]["exact"]) for r in rows)
    mini = collections.Counter((r["nonminimal"] == 0, r["decision"]["exact"]) for r in rows)
    print(f"K0 rows {len(rows)}; (env, executor, controller) hashes {dict(hashes)}")
    print(f"  executor == env exact, by family: {dict(sorted(fam.items()))}")
    print(f"  decision == arrival exact, by (all plans minimal covers, exact): {dict(sorted(mini.items()))}")
    rl = _rows("k0/rl_static.jsonl")
    kinds = collections.Counter((r["kind"], r.get("match")) for r in rl)
    runner = collections.Counter(all(m["env"] == m["ref"] for m in r["runner"]) for r in rl if r["kind"] == "runner")
    print(f"  RL static rows: {dict(kinds)}; runner RL(g.)/RL(s.1) equal to rl.rollout: {dict(runner)}")
    samples = collections.defaultdict(list)
    for r in rl:
        if r["kind"] == "sample":
            samples[(r["setting"], r["instance"])].append(r["env"])
    n = same = best = 0
    for s in sorted({k[0] for k in samples}):
        with open(ROOT / "runs" / "rl" / s / "dev.jsonl") as fh:
            for line in fh:
                row = json.loads(line)
                key = (s, row["instance"])
                if row.get("method") != "RL(s.64)" or key not in samples:
                    continue
                n += 1
                same += sorted(samples[key]) == sorted(row["sample_makespans"])
                best += min(samples[key]) == row["makespan"]
    print(f"  stored Phase-1 RL(s.64) dev rows reproduced in DynTaskEnvX: {n} instances, sample multiset equal {same}, "
          f"best equal {best}")


def cpu() -> None:
    def pct(pattern: str, pred) -> str:
        xs, ep = [], []
        for r in _rows(pattern):
            if r.get("status") == "completed" and pred(r) and r.get("cpu_ms_event"):
                xs += r["cpu_ms_event"]
                ep.append(np.percentile(r["cpu_ms_event"], 95))
        return (f"{len(xs)} events, p50 {np.percentile(xs, 50):.1f} ms, p95 {np.percentile(xs, 95):.1f} ms; "
                f"per-episode p95 median {np.median(ep):.1f} ms, max {np.max(ep):.1f} ms")

    def main(r):
        return r["family"] in ("F1", "F2", "F3") and r["seed"] in (0, 1) and r["method"] == "SPARC"

    print("Per structural event CPU (process time, loaded shared host; descriptive only)")
    print("  SPARC-L, 50 tasks:  " + pct("c1dyn/rows_light.jsonl", main))
    print("  SPARC-H, 50 tasks:  " + pct("c1dyn/rows_heavy.jsonl", main))
    print("  SPARC-L, 200 tasks: " + pct("c1dyn/rows_200.jsonl", lambda r: main(r) and r["tier"] == "light"))
    print("  SPARC-H, 200 tasks: " + pct("c1dyn/rows_200.jsonl", lambda r: main(r) and r["tier"] == "heavy"))


def rlcost() -> None:
    static = {}
    for r in _rows("k0/rl_static.jsonl"):
        if r["kind"] == "runner":
            for m in r["runner"]:
                static[(r["setting"], r["instance"], m["method"])] = (m["seed"], m["env"])
    by = collections.defaultdict(lambda: collections.defaultdict(list))
    same_seed = collections.Counter()
    for r in _rows("c1dyn/rows_native.jsonl"):
        if r["method"] != "RL(g.)" or r["seed"] != 0 or not r["success"]:
            continue
        seed, mk0 = static[(r["setting"], r["instance"], "RL(g.)")]
        same_seed[seed == r["method_seed"]] += 1
        by[r["cell"]][(r["setting"], r["instance"])].append(math.log(r["makespan"]) - math.log(mk0))
    print(f"RL(g.) makespan(cell) / makespan(static), CRN seed 0, same policy seed: {dict(same_seed)}")
    print("  " + ", ".join(f"{c} {math.exp(np.mean([np.mean(v) for v in by[c].values()])):.3f}"
                           for c in sorted(by)))


if __name__ == "__main__":
    todo = sys.argv[1:] or ["c1", "c3", "c3bs", "k0", "cpu", "rlcost"]
    for part in todo:
        print(f"== {part}")
        {"c1": c1, "c3": c3, "c3bs": lambda: c3(True), "k0": k0, "cpu": cpu, "rlcost": rlcost}[part]()
