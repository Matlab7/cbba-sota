"""Track D C1-dyn pilot (docs/trackD-spec.md Section 8 days 2-3; Section 7 T1-T3; gates K1, K2). Dev split only.

    calibrate  B4 equal-CPU restart count per setting, from SPARC-Light episodes in DynTaskEnvX in which every
               re-plan also times B4 on the same belief (``cbba_sota.dyn.c1dyn.ProbeSPARC``). Writes
               runs/trackD/c1dyn/calib.jsonl and b4_budget.json (frozen before the campaign is analysed).
    analyze    reads the campaign rows (scripts/trackD_run.py --out runs/trackD/c1dyn) and writes the gate tables
               and CSVs to runs/trackD/c1dyn/analysis/ (and --copy-to, e.g. docs/results/trackD-week1).

    .venv/bin/python scripts/trackD_c1dyn.py calibrate --instances 0-9 --procs 20
    .venv/bin/python scripts/trackD_c1dyn.py analyze --copy-to docs/results/trackD-week1

The campaign itself is run by scripts/trackD_run.py (see runs/trackD/c1dyn/campaign.sh). Every number here comes
from DynTaskEnvX (the ground truth world); rows record the world and the env code hash, and ``analyze`` refuses
mixed env code unless ``--hash`` picks one.
"""
from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
import re
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "trackD" / "c1dyn"
CALIB_CELLS = ("F1-R2", "F2-R2N3", "F3-pf0.2")
_ONE = {k: "1" for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS")}


def _parse_range(spec: str) -> list[int]:
    out: list[int] = []
    for chunk in spec.split(","):
        if "-" in chunk:
            a, b = chunk.split("-")
            out += list(range(int(a), int(b) + 1))
        elif chunk:
            out.append(int(chunk))
    return sorted(set(out))


# --- calibrate ---------------------------------------------------------------------------------------------------------
def _init_calib() -> None:
    os.environ.update(_ONE)
    from cbba_sota.bench import configs
    from cbba_sota.dyn.planner import PlanState, RHPlanner
    from cbba_sota.hetero import Instance

    # load / compile every kernel once, so that no probed event pays JIT or cache loading
    inst = Instance.from_pickle(configs.get("MA-AT-25-5-50").instance_path("dev", 0))
    st = PlanState.static(inst)
    pl = RHPlanner()
    p0 = pl.plan(st, None, None, 1, iters=60)
    pl.plan(st, p0, None, 2, iters=60)
    pl.plan(st, None, None, 3, iters=0, warm=False, restarts=3)


def calib_job(job: dict) -> dict | None:
    from cbba_sota.bench import configs
    from cbba_sota.dyn import perturb
    from cbba_sota.dyn.c1dyn import ProbeSPARC
    from cbba_sota.dyn.controller import PlanController
    from cbba_sota.dyn.env import make_env

    s = configs.get(job["setting"])
    rz = perturb.realize_instance(job["setting"], "dev", job["instance"], job["seed"], job["cell"])
    if rz.excluded:
        return None
    env = make_env(s.instance_path("dev", job["instance"]), rz)
    pol = ProbeSPARC(iters=300, probe_restarts=job["probe"])
    ep = env.run_plan(PlanController(pol, env.nominal_instance(), rz.kappa()))
    return {**job, "world": ep.world, "makespan": ep.makespan, "success": ep.success, "replans": ep.replans,
            "probes": [{**p, "b4_s": {str(k): v for k, v in p["b4_s"].items()}} for p in pol.probes]}


def cmd_calibrate(a) -> None:
    import numpy as np

    from cbba_sota.dyn.c1dyn import restarts_equal_cpu

    OUT.mkdir(parents=True, exist_ok=True)
    probe = tuple(int(x) for x in a.probe.split(","))
    jobs = [{"setting": s, "instance": i, "seed": sd, "cell": c, "probe": probe}
            for s in a.settings for i in _parse_range(a.instances) for sd in _parse_range(a.seeds) for c in a.cells]
    os.environ.update(_ONE)
    t0 = time.time()
    with mp.get_context("spawn").Pool(min(a.procs, 22), initializer=_init_calib) as pool:
        res = [r for r in pool.imap_unordered(calib_job, jobs) if r]
    path = OUT / ("calib.jsonl" if not a.tag else f"calib_{a.tag}.jsonl")
    with path.open("w") as f:
        for r in res:
            f.write(json.dumps(r) + "\n")
    bpath = OUT / "b4_budget.json"
    budget = json.loads(bpath.read_text()) if bpath.exists() else {"settings": {}}  # merge: other settings kept
    meta = {"probe_restarts": list(probe), "cells": list(a.cells), "instances": a.instances, "seeds": a.seeds,
            "worlds": sorted({r["world"] for r in res}), "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "host_load": os.getloadavg()}
    budget.setdefault("method", "equal mean CPU per re-plan on SPARC-Light belief states (ratio of totals); "
                                "linear fit of B4 CPU over probed restart counts")
    for k, v in meta.items():
        budget.setdefault(k, v)  # the first calibration's metadata stays at the top level
    budget["charging"] = ("restarts = round(r*) with B4 charged for its constructions only: the per-restart "
                          "Problem rebuild of the implementation (about 70% of its per-restart CPU) is not counted, "
                          "which is generous to B4; r*_impl charges the implementation as it runs")
    lines = []
    for s in a.settings:
        ev = [p for r in res if r["setting"] == s for p in r["probes"]]
        sp = [p["sparc_s"] for p in ev]
        b4 = {r: [p["b4_s"][str(r)] for p in ev] for r in probe}
        ov = [p["problem_s"] for p in ev]
        impl = restarts_equal_cpu(sp, b4)
        fair = restarts_equal_cpu(sp, b4, overhead_s=ov)
        later = [p for p in ev if not p["first"]]
        fair_later = restarts_equal_cpu([p["sparc_s"] for p in later],
                                        {r: [p["b4_s"][str(r)] for p in later] for r in probe},
                                        overhead_s=[p["problem_s"] for p in later])
        fit = {"construction_only": fair, "implemented": impl, "r_star_excl_first": fair_later["r_star"],
               "problem_ms_mean": 1e3 * float(np.mean(ov)), "n_open_mean": float(np.mean([p["n_open"] for p in ev])),
               "restarts": max(1, round(fair["r_star"])), "restarts_impl": max(1, round(impl["r_star"])),
               "calibration": meta}
        budget["settings"][s] = fit
        lines.append(f"{s}: {fair['n_events']} events, open tasks {fit['n_open_mean']:.1f}; SPARC-L 300 it mean "
                     f"{fair['sparc_mean_ms']:.2f} ms (median {fair['sparc_median_ms']:.2f}); B4 as implemented "
                     f"{impl['a_ms']:.2f} + {impl['b_ms']:.3f} r ms -> r* {impl['r_star']:.1f}; construction-only "
                     f"{fair['a_ms']:.2f} + {fair['b_ms']:.3f} r ms (Problem rebuild {fit['problem_ms_mean']:.3f} "
                     f"ms) -> r* {fair['r_star']:.1f} (median-based {fair['r_star_median']:.1f}, excl. t=0 "
                     f"{fair_later['r_star']:.1f}) -> B4-eq restarts {fit['restarts']}")
    bpath.write_text(json.dumps(budget, indent=1))
    print("\n".join(lines))
    print(f"{len(res)} episodes, {time.time() - t0:.0f} s; worlds {meta['worlds']}; wrote {path} and {bpath}")


# --- analyze -----------------------------------------------------------------------------------------------------------

# SPARC-Light against every other arm; B4-eq is the pre-specified T1 comparator (runs/trackD/c1dyn/campaign.sh).
COMPARATORS = ("B4-eq", "B4-2eq", "B4-4eq", "B4-impl", "B4-r12", "ins-only", "every-event", "open-loop", "B5",
               "SPARC-H", "SPARC-rk", "RL(g.)", "RL(g.)-arr", "RL(s.1)", "RL(s.1)-arr", "greedy", "greedy-arr",
               "SPARC-L>B4-eq", "B4-eq>SPARC-L")
RATIO_FIELDS = ("a", "b", "scope", "n_clusters", "n_obs", "ratio", "lo", "hi", "median_ratio", "hl_ratio", "p_less",
                "p_two", "sd_log", "wins", "losses", "ties", "obs_wins", "obs_losses", "pairs", "dropped_pairs",
                "success_a", "success_b", "mcnemar_b10", "mcnemar_b01", "mcnemar_p_two")
METHOD_FIELDS = ("label", "setting", "family", "n", "success", "success_rate", "mean_makespan", "gm_makespan",
                 "replans", "route_versions", "structural_events", "travel", "wasted_trips", "abandons", "restarts",
                 "cpu_ep_p50_med", "cpu_ep_p95_med", "cpu_ep_p95_max", "cpu_p50", "cpu_p95", "cpu_max",
                 "cpu_episode_s")


def load_rows(rows_dir: Path, want_hash: str | None) -> tuple[list[dict], dict]:
    """Completed rows of one env code hash, deduplicated by case id (last row wins), plus bookkeeping."""
    raw = []
    for f in sorted(rows_dir.glob("rows*.jsonl")):
        raw += [json.loads(line) for line in f.read_text().splitlines() if line.strip()]
    status: dict = {}
    for r in raw:
        status[r["status"]] = status.get(r["status"], 0) + 1
    hashes = sorted({r.get("code_hash") for r in raw if r["status"] == "completed"})
    if want_hash is None:
        if len(hashes) != 1:
            raise SystemExit(f"rows from several env code hashes {hashes}; pick one with --hash")
        want_hash = hashes[0]
    by_id: dict = {}
    for r in raw:
        if r["status"] == "completed" and r.get("code_hash") == want_hash:
            by_id[r["case_id"]] = r
    excluded = {(r["setting"], r["instance"], r["seed"], r["cell"]) for r in raw if r["status"] == "excluded"}
    errors = [r for r in raw if r["status"] == "error"]
    return list(by_id.values()), {"status": status, "hash": want_hash, "hashes": hashes, "excluded": excluded,
                                  "errors": errors, "worlds": sorted({r["world"] for r in by_id.values()}),
                                  "git": sorted({(r.get("git"), r.get("git_dirty_dyn")) for r in by_id.values()})}


def cmd_analyze(a) -> None:
    import numpy as np

    from cbba_sota.dyn import perturb
    from cbba_sota.dyn import stats as S
    from cbba_sota.dyn.c1dyn import (
        CLUSTER_KEYS,
        PAIR_KEYS,
        b4_levels,
        episode_summary,
        family_balanced,
        k1_decision,
        k2_decision,
        labels,
    )

    rows_dir = Path(a.rows)
    budget = json.loads((rows_dir / "b4_budget.json").read_text())
    b4 = b4_levels(budget)
    base, info = load_rows(rows_dir, a.hash)
    rows = [{**r, "label": lab} for r in base for lab in labels(r, b4)]
    fam_of = {c: perturb.cell(c).family for c in perturb.CELLS}
    prim = set(perturb.PRIMARY_SETTINGS)
    seeds_main = set(_parse_range(a.seeds_main))
    repl = [r for r in rows if r["setting"] in prim and r["family"] in ("F1", "F2", "F3") and r["seed"] not in seeds_main]
    rows = [r for r in rows if r["seed"] in seeds_main]  # the pre-specified analysis uses CRN seeds 0-1 only
    main = [r for r in rows if r["setting"] in prim and r["family"] in ("F1", "F2", "F3")]
    f0 = [r for r in rows if r["setting"] in prim and r["family"] == "F0"]
    out = rows_dir / "analysis"
    out.mkdir(exist_ok=True)
    kw = {"pair_keys": PAIR_KEYS, "cluster_keys": CLUSTER_KEYS, "method_key": "label"}
    txt: list[str] = []

    def say(line: str = "") -> None:
        txt.append(line)
        print(line, flush=True)

    say("Track D C1-dyn pilot (dev split) -- scripts/trackD_c1dyn.py analyze")
    say(f"env code hash {info['hash']}; worlds: {info['worlds']}")
    say(f"git {info['git']}; row status counts {info['status']}; errors {len(info['errors'])}")
    say(f"excluded F3 realizations (all methods, paired): {len(info['excluded'])}")
    say(f"B4 levels (restarts per setting): {b4}")

    # ---- episode summaries -------------------------------------------------------------------------------------
    labs = list(dict.fromkeys(["SPARC-L", *COMPARATORS, *sorted({r["label"] for r in rows})]))
    mrows = []
    for lab in labs:
        for scope_s in [None, *sorted(prim)]:
            for fam in [None, "F0", "F1", "F2", "F3"]:
                sel = [r for r in rows if r["label"] == lab and r["setting"] in prim
                       and (scope_s is None or r["setting"] == scope_s)
                       and (r["family"] == fam if fam else r["family"] in ("F1", "F2", "F3"))]
                if sel:
                    mrows.append({"label": lab, "setting": scope_s or "ALL", "family": fam or "F1-F3",
                                  **episode_summary(sel)})
    say("\nEpisodes per method, F1-F3 pooled over the 4 primary settings (CPU = controller call per structural "
        "event, process time on the loaded shared host, descriptive only):")
    say("  (B4-r<k> rows: per-setting restart counts, in methods.csv only; the first-event CPU max includes one-time "
        "numba cache loading in a fresh worker)")
    say(f"  {'method':12s} {'n':>5s} {'succ':>9s} {'mean mk':>8s} {'replans':>8s} {'versions':>8s} {'wasted':>7s} "
        f"{'cpu p50':>8s} {'p95':>7s} {'ep-p95 med':>10s} {'max':>8s} {'cpu/ep s':>8s}")
    for m in mrows:
        if m["setting"] == "ALL" and m["family"] == "F1-F3" and not re.fullmatch(r"B4-r\d+(\+t0)?", m["label"]):
            say(f"  {m['label']:12s} {m['n']:5d} {m['success']:4d}/{m['n']:<4d} {m['mean_makespan']:8.2f} "
                f"{m['replans']:8.1f} {m['route_versions']:8.1f} {m['wasted_trips']:7.2f} {m['cpu_p50']:8.2f} "
                f"{m['cpu_p95']:7.2f} {m['cpu_ep_p95_med']:10.2f} {m['cpu_max']:8.1f} {m['cpu_episode_s']:8.3f}")

    # ---- paired ratios -------------------------------------------------------------------------------------------
    rrows = []

    def ratio(b: str, scope: str, where, data=main, a_: str = "SPARC-L"):
        d = S.pair_rows(data, a_, b, where=where, **kw)
        n_pairs = len(d["keys"])
        if n_pairs == 0 or d["keep"].sum() < 2 or len({c for c in d["clusters"]}) < 2:
            return None
        g = S.gm_ratio(d["log_r"], d["clusters"], strata=d["strata"])
        mc = S.exact_mcnemar(d["success_a"], d["success_b"])
        rec = {"a": a_, "b": b, "scope": scope, **g.as_dict(), "pairs": n_pairs,
               "dropped_pairs": int((~d["keep"]).sum()), "success_a": int(d["success_a"].sum()),
               "success_b": int(d["success_b"].sum()), "mcnemar_b10": mc["b10"], "mcnemar_b01": mc["b01"],
               "mcnemar_p_two": mc["p_two"]}
        rrows.append(rec)
        return g

    cells = [c for c in perturb.CELLS if fam_of[c] in ("F1", "F2", "F3")]
    results: dict = {}
    for b in COMPARATORS:
        if not any(r["label"] == b for r in main):
            continue
        results[(b, "ALL")] = ratio(b, "F1-F3", None)
        for fam in ("F1", "F2", "F3"):
            results[(b, fam)] = ratio(b, fam, lambda r, fam=fam: r["family"] == fam)
        for s_ in sorted(prim):
            results[(b, s_)] = ratio(b, s_, lambda r, s_=s_: r["setting"] == s_)
            for fam in ("F1", "F2", "F3"):
                ratio(b, f"{s_}|{fam}", lambda r, s_=s_, fam=fam: r["setting"] == s_ and r["family"] == fam)
        for c in cells:
            results[(b, c)] = ratio(b, c, lambda r, c=c: r["cell"] == c)
        d = S.pair_rows(main, "SPARC-L", b, **kw)
        if d["keep"].sum() >= 2:
            v, cl = family_balanced(d, fam_of)
            g = S.gm_ratio(v, cl, strata=[c[0] for c in cl])
            results[(b, "F1-F3 family-balanced")] = g
            rrows.append({"a": "SPARC-L", "b": b, "scope": "F1-F3 family-balanced", **g.as_dict(),
                          "pairs": len(d["keys"]), "dropped_pairs": int((~d["keep"]).sum())})
        if not d["keep"].all():
            di = S.pair_rows(main, "SPARC-L", b, failure="impute", **kw)
            g = S.gm_ratio(di["log_r"], di["clusters"], strata=di["strata"])
            rrows.append({"a": "SPARC-L", "b": b, "scope": "F1-F3 failures=200", **g.as_dict(),
                          "pairs": len(di["keys"]), "dropped_pairs": 0})
    # descriptive extras
    for a_, b in (("RL(g.)-arr", "RL(g.)"), ("RL(s.1)", "RL(g.)"), ("greedy", "RL(g.)"), ("B5", "B4-eq"),
                  ("SPARC-H", "B4-eq"), ("every-event", "B4-eq"), ("B4-4eq", "B4-eq"), ("ins-only", "B4-eq"),
                  ("SPARC-L>B4-eq", "B4-eq"), ("B4-eq>SPARC-L", "B4-eq"), ("SPARC-rk", "SPARC-L"),
                  ("SPARC-L+t0", "SPARC-L"), ("B4-eq+t0", "B4-eq"), ("SPARC-L+t0", "SPARC-H")):
        if any(r["label"] == a_ for r in main) and any(r["label"] == b for r in main):
            ratio(b, "F1-F3", None, a_=a_)

    say("\nPaired GM ratio SPARC-L / method (pairs where both succeed; cluster bootstrap over 80 instances, strata "
        "= settings, 10,000 resamples, seed 0):")
    say(f"  {'method':12s} {'F1-F3 pooled':>26s} {'F1':>8s} {'F2':>8s} {'F3':>8s} | "
        + " ".join(f"{s_[:9]:>9s}" for s_ in sorted(prim)) + " | wins/80  p_less")
    for b in COMPARATORS:
        g = results.get((b, "ALL"))
        if g is None:
            continue
        fams = " ".join(f"{results[(b, f)].ratio:8.4f}" if results.get((b, f)) else f"{'-':>8s}"
                        for f in ("F1", "F2", "F3"))
        sets = " ".join(f"{results[(b, s_)].ratio:9.4f}" if results.get((b, s_)) else f"{'-':>9s}"
                        for s_ in sorted(prim))
        say(f"  {b:12s} {g.ratio:.4f} [{g.lo:.4f}, {g.hi:.4f}] {fams} | {sets} | {g.wins:2d}/{g.n_clusters}  "
            f"{g.p_less:.1e}")
    say("\nPer cell (SPARC-L / method):")
    say(f"  {'method':12s} " + " ".join(f"{c:>12s}" for c in cells))
    for b in COMPARATORS:
        if results.get((b, "ALL")) is None:
            continue
        say(f"  {b:12s} " + " ".join(f"{results[(b, c)].ratio:12.4f}" if results.get((b, c)) else f"{'-':>12s}"
                                     for c in cells))

    # ---- gates --------------------------------------------------------------------------------------------------
    grows = []
    say("\nGates (spec Section 7 T1-T3 and Section 8 K1, K2; previews of E2a and K3 for the other agents):")
    comp = S.compare(main, "SPARC-L", "B4-eq", **kw)
    t1 = k1_decision(comp.pooled, comp.per_setting)
    say(f"  T1 (SPARC-L vs B4-eq, pre-specified): {'PASS' if t1.passed else 'FAIL'} -- {t1.detail}")
    say(f"     stats.compare t1_pass={comp.t1_pass} clear_margin={comp.clear_margin}; per setting "
        + ", ".join(f"{k} {v.ratio:.4f} [{v.lo:.4f}, {v.hi:.4f}]" for k, v in comp.per_setting.items()))
    say(f"  K1 (planner collapse): {'FIRES (kill)' if not t1.passed else 'does not fire'}")
    grows.append({"gate": "T1/K1", "comparator": "B4-eq", "decision": "pass" if t1.passed else "fail",
                  "detail": t1.detail})
    for alt in ("B4-2eq", "B4-4eq", "B4-impl", "B4-r12"):
        if results.get((alt, "ALL")) is None:
            continue
        c2 = S.compare(main, "SPARC-L", alt, **kw)
        g2 = k1_decision(c2.pooled, c2.per_setting)
        say(f"     sensitivity T1 vs {alt}: {'pass' if g2.passed else 'fail'} -- {g2.detail}")
        grows.append({"gate": "T1 sensitivity", "comparator": alt, "decision": "pass" if g2.passed else "fail",
                      "detail": g2.detail})
    for a_, b_ in (("SPARC-H", "B4-eq"), ("SPARC-rk", "B4-eq"), ("SPARC-L+t0", "B4-eq+t0")):  # exploratory
        if any(r["label"] == a_ for r in main) and any(r["label"] == b_ for r in main):
            c3 = S.compare(main, a_, b_, **kw)
            g3 = k1_decision(c3.pooled, c3.per_setting)
            say(f"     exploratory T1 rule for {a_} vs {b_}: {'pass' if g3.passed else 'fail'} -- {g3.detail}")
            grows.append({"gate": "T1 exploratory", "comparator": f"{a_} vs {b_}",
                          "decision": "pass" if g3.passed else "fail", "detail": g3.detail})
    fb = results.get(("B4-eq", "F1-F3 family-balanced"))
    if fb is not None:
        say(f"     sensitivity T1 family-balanced weighting (F1, F2, F3 equal): {fb.ratio:.4f} [{fb.lo:.4f}, "
            f"{fb.hi:.4f}]")
    k2 = k2_decision(results[("ins-only", "F3")], results[("B4-eq", "F1")])
    say(f"  K2 (no dynamic separation): {'FIRES (kill)' if k2.passed else 'does not fire'} -- {k2.detail}")
    grows.append({"gate": "K2", "comparator": "ins-only (F3), B4-eq (F1)",
                  "decision": "fires" if k2.passed else "does not fire", "detail": k2.detail})
    g = results.get(("every-event", "ALL"))
    if g is not None:
        say(f"  T2 (trigger rule, descriptive): SPARC-L / every-event {g.ratio:.4f} [{g.lo:.4f}, {g.hi:.4f}]")
        grows.append({"gate": "T2", "comparator": "every-event", "decision": "descriptive",
                      "detail": f"{g.ratio:.4f} [{g.lo:.4f}, {g.hi:.4f}]"})
    g = results.get(("ins-only", "ALL"))
    if g is not None:
        say(f"  T1b (vs insertion-only, descriptive): {g.ratio:.4f} [{g.lo:.4f}, {g.hi:.4f}]")
    if f0:
        for b in ("open-loop", "every-event"):
            d = S.pair_rows(f0, "SPARC-L", b, **kw)
            if not len(d["keys"]):
                continue
            tt = S.tost(d["log_r"], d["clusters"], strata=d["strata"])
            gg = S.gm_ratio(d["log_r"], d["clusters"], strata=d["strata"])
            ident = float(np.mean(np.abs(d["log_r"]) < 1e-12))
            rp = [r["replans"] for r in f0 if r["label"] == "SPARC-L"]
            detail = (f"GM {gg.ratio:.4f} [{gg.lo:.4f}, {gg.hi:.4f}], TOST +-2% p={tt.p:.2e} (t), bootstrap 90% CI "
                      f"of mean log r [{tt.ci90_lo:.4f}, {tt.ci90_hi:.4f}], equivalent {tt.equivalent_t}; identical "
                      f"makespans {ident:.1%} of {len(d['keys'])} pairs; SPARC-L re-plans per F0 episode "
                      f"{min(rp)}-{max(rp)}")
            tag = "T3 (F0 noise control)" if b == "open-loop" else "F0 re-planning on noise (S2 analogue, descr.)"
            say(f"  {tag} SPARC-L vs {b}: {detail}")
            grows.append({"gate": "T3" if b == "open-loop" else "F0 S2 analogue (descriptive)", "comparator": b,
                          "decision": "pass" if (tt.equivalent_t and b == "open-loop") else "see detail",
                          "detail": detail})
            rrows.append({"a": "SPARC-L", "b": b, "scope": "F0", **gg.as_dict(), "pairs": len(d["keys"]),
                          "dropped_pairs": int((~d["keep"]).sum())})
    d = S.pair_rows(main, "SPARC-L", "SPARC-H", **kw)
    if len(d["keys"]):
        tt = S.tost(d["log_r"], d["clusters"], strata=d["strata"])
        detail = (f"mean log r {tt.mean_log:+.4f} (ratio {np.exp(tt.mean_log):.4f}); TOST +-2% p={tt.p:.2e}; "
                  f"bootstrap 90% CI [{np.exp(tt.ci90_lo):.4f}, {np.exp(tt.ci90_hi):.4f}]; equivalent (t) "
                  f"{tt.equivalent_t}, (bootstrap) {tt.equivalent_boot}")
        say(f"  E2a preview (SPARC-L vs SPARC-H, TOST +-2%): {detail}")
        grows.append({"gate": "E2a preview", "comparator": "SPARC-H", "decision":
                      "equivalent" if tt.equivalent_t else "not shown", "detail": detail})
    for b in ("RL(g.)", "RL(g.)-arr", "RL(s.1)", "RL(s.1)-arr"):
        g = results.get((b, "ALL"))
        if g is not None:
            fire = g.ratio > 1 / 1.05
            say(f"  K3 preview (B1 {b} within 5% of SPARC?): SPARC-L/{b} {g.ratio:.4f} [{g.lo:.4f}, {g.hi:.4f}] -> "
                f"{'FIRES' if fire else 'does not fire'}")
            grows.append({"gate": "K3 preview", "comparator": b, "decision": "fires" if fire else "does not fire",
                          "detail": f"{g.ratio:.4f} [{g.lo:.4f}, {g.hi:.4f}]"})
    say("\nSuccess (all F1-F3 episodes; exact McNemar vs SPARC-L on the paired episodes):")
    for rr in rrows:
        if rr["a"] == "SPARC-L" and rr["scope"] == "F1-F3" and rr.get("mcnemar_b10") is not None:
            say(f"  {rr['b']:12s} SPARC-L {rr['success_a']}/{rr['pairs']}, {rr['b']} {rr['success_b']}/{rr['pairs']}"
                f"; discordant {rr['mcnemar_b10']}/{rr['mcnemar_b01']}, p = {rr['mcnemar_p_two']:.3g}")

    # ---- exploratory replication on fresh CRN seeds -----------------------------------------------------------------
    if repl:
        say(f"\nEXPLORATORY replication on CRN seeds {sorted({r['seed'] for r in repl})} (same dev instances, new "
            "realizations; decided after the seeds 0-1 snapshot; no gate uses it):")
        for a_, b in (("SPARC-L", "B4-eq"), ("SPARC-rk", "B4-eq"), ("SPARC-L", "SPARC-rk")):
            if not (any(r["label"] == a_ for r in repl) and any(r["label"] == b for r in repl)):
                continue
            c4 = S.compare(repl, a_, b, **kw)
            g4 = k1_decision(c4.pooled, c4.per_setting)
            fam = " ".join(f"{f} {ratio(b, f'replication {a_} {f}', lambda r, f=f: r['family'] == f, data=repl, a_=a_).ratio:.4f}"
                           for f in ("F1", "F2", "F3"))
            ratio(b, "replication F1-F3", None, data=repl, a_=a_)
            say(f"  {a_}/{b}: {c4.pooled.ratio:.4f} [{c4.pooled.lo:.4f}, {c4.pooled.hi:.4f}] ({fam}); settings "
                + ", ".join(f"{k[:9]} {v.ratio:.4f}" for k, v in c4.per_setting.items())
                + (f"; T1 rule {'pass' if g4.passed else 'fail'}" if b == "B4-eq" else ""))
            grows.append({"gate": "replication (exploratory)", "comparator": f"{a_} vs {b} (seeds 2-3)",
                          "decision": ("T1 rule pass" if g4.passed else "T1 rule fail") if b == "B4-eq" else
                          "descriptive", "detail": f"{c4.pooled.ratio:.4f} [{c4.pooled.lo:.4f}, {c4.pooled.hi:.4f}]"})

    # ---- secondary setting ---------------------------------------------------------------------------------------
    sec = [r for r in rows if r["setting"] not in prim and r["family"] in ("F1", "F2", "F3")]
    if sec:
        say("\nSecondary setting(s) (descriptive):")
        for s_ in sorted({r["setting"] for r in sec}):
            for b in COMPARATORS:
                g = ratio(b, f"secondary {s_}", lambda r, s_=s_: r["setting"] == s_, data=sec)
                if g is not None:
                    say(f"  {s_} SPARC-L/{b}: {g.ratio:.4f} [{g.lo:.4f}, {g.hi:.4f}] clusters {g.n_clusters} "
                        f"obs {g.n_obs}")
            for lab in sorted({r["label"] for r in sec if r["setting"] == s_}):
                m = episode_summary([r for r in sec if r["setting"] == s_ and r["label"] == lab])
                mrows.append({"label": lab, "setting": s_, "family": "F1-F3", **m})
                say(f"    {lab:12s} n {m['n']} success {m['success']} mean {m['mean_makespan']:.2f} replans "
                    f"{m['replans']:.1f} cpu p50 {m['cpu_p50']:.1f} p95 {m['cpu_p95']:.1f} ms")

    # ---- write -------------------------------------------------------------------------------------------------
    def write(name: str, fields, recs) -> None:
        with (out / name).open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(fields), extrasaction="ignore")
            w.writeheader()
            for rec in recs:
                w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in rec.items()})

    write("ratios.csv", RATIO_FIELDS, rrows)
    write("methods.csv", METHOD_FIELDS, mrows)
    write("gates.csv", ("gate", "comparator", "decision", "detail"), grows)
    (out / "summary.txt").write_text("\n".join(txt) + "\n")
    if a.copy_to:
        dst = ROOT / a.copy_to
        dst.mkdir(parents=True, exist_ok=True)
        for name in ("ratios.csv", "methods.csv", "gates.csv", "summary.txt"):
            shutil.copyfile(out / name, dst / f"c1dyn_{name}")  # byte-exact (the csv module writes CRLF)
        shutil.copyfile(rows_dir / "b4_budget.json", dst / "c1dyn_b4_budget.json")
    print(f"wrote {out}" + (f" and {a.copy_to}" if a.copy_to else ""))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("calibrate", "analyze"))
    ap.add_argument("--settings", nargs="+", default=["MA-AT-25-5-50", "MA-AT-50-5-50", "SA-AT-50-5-50",
                                                      "SA-BT-50-5-50"])
    ap.add_argument("--instances", default="0-9")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--cells", nargs="+", default=list(CALIB_CELLS))
    ap.add_argument("--probe", default="0,24", help="B4 restart counts timed at every probed event")
    ap.add_argument("--procs", type=int, default=20)
    ap.add_argument("--rows", default=str(OUT))
    ap.add_argument("--hash", default=None, help="env code hash to analyse (default: the only one present)")
    ap.add_argument("--copy-to", default=None)
    ap.add_argument("--seeds-main", default="0-1", help="analyze: CRN seeds of the pre-specified analysis")
    ap.add_argument("--tag", default=None, help="calibrate: suffix of the calib jsonl (the budget file is merged)")
    a = ap.parse_args()
    if a.cmd == "calibrate":
        cmd_calibrate(a)
    else:
        cmd_analyze(a)


if __name__ == "__main__":
    sys.exit(main())
