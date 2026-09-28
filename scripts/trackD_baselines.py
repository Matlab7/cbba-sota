"""Track D week-1 baselines (agent D, days 3-4): B2 calibration and the K3 / K4 report.

    calibrate-b2  CPU per policy decision of the RL-MPC rollouts (lockstep, ``--batch`` rollouts, forward passes on
                  ``--device``) on nominal clones taken at the structural events of dev episodes; prints the Heavy
                  budget ``8 CPU-s / (CPU per decision)`` (``rl_mpc.HEAVY_DECISIONS`` is set from it by hand).
    fidelity-b6   spec 5.2 fidelity check of B6 on static dev instances: average start time of the CBTA-style auction
                  vs the CBGA-style grouping auction (and the makespans of every auction rule).
    report        paired comparisons SPARC-Light / method (``--ref``) on dev F1-F3 (``--families``) from ``scripts/trackD_run.py`` rows (the C1-dyn
                  pilot rows in ``runs/trackD/day2_preview`` plus this workflow's rows), pooled, per family and per
                  setting (cluster bootstrap over instances, strata = settings; ``cbba_sota.dyn.stats``), success,
                  per-event CPU, and the K3 / K4 verdicts (docs/trackD-spec.md Section 8).

    .venv/bin/python scripts/trackD_baselines.py calibrate-b2 --procs 4
    .venv/bin/python scripts/trackD_baselines.py fidelity-b6 --instances 5
    .venv/bin/python scripts/trackD_baselines.py report runs/trackD/c1dyn runs/trackD/baselines_d
"""
from __future__ import annotations

import argparse
import glob
import json
import multiprocessing as mp
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

_ONE = {k: "1" for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS")}
CAL_CASES = [("MA-AT-25-5-50", 7, 1, "F1-R2"), ("MA-AT-50-5-50", 8, 1, "F3-pf0.2-R2"),
             ("SA-AT-50-5-50", 9, 1, "F2-R2N3"), ("SA-BT-50-5-50", 10, 1, "F1-R3")]


# --- calibrate-b2 ----------------------------------------------------------------------------------------------------
def _cal_one(args) -> dict:
    case, device, batch, n_clones = args
    os.environ.update(_ONE)
    import pickle
    import time

    import numpy as np
    import torch

    torch.set_num_threads(1)
    from cbba_sota.bench import configs
    from cbba_sota.dyn import perturb
    from cbba_sota.dyn.baselines import rl_mpc
    from cbba_sota.dyn.env import make_env

    setting, inst, seed, cell = case
    clones: list[bytes] = []
    orig = rl_mpc.build_clone

    def spy(*a, **k):
        c = orig(*a, **k)
        clones.append(pickle.dumps(c))
        return c

    rl_mpc.build_clone = spy
    try:
        s = configs.get(setting)
        rz = perturb.realize_instance(setting, "dev", inst, seed, cell)
        env = make_env(s.instance_path("dev", inst), rz)
        env.run_plan(rl_mpc.make_controller(env, rz, "heavy", 0, decisions=1, batch=1, device=device))
    finally:
        rl_mpc.build_clone = orig
    net = rl_mpc._net(device)
    pick = np.linspace(0, len(clones) - 1, min(n_clones, len(clones))).round().astype(int)
    cpu = dec = wall = 0.0
    rl_mpc.lockstep(clones[0], net, [(0, 1, True)], device=device)  # warm-up (CUDA context, kernels)
    for k in pick:
        specs = [(b, 1000 + b, True) for b in range(batch)]
        c0, w0 = time.process_time(), time.perf_counter()
        res = rl_mpc.lockstep(clones[k], net, specs, device=device)
        cpu += time.process_time() - c0
        wall += time.perf_counter() - w0
        dec += sum(r.decisions for r in res)
    return {"case": case, "device": device, "clones": len(pick), "decisions": int(dec),
            "cpu_ms_per_decision": 1e3 * cpu / dec, "wall_ms_per_decision": 1e3 * wall / dec}


def calibrate_b2(a) -> None:
    devices = [f"cuda:{k % 2}" for k in range(len(CAL_CASES))] if a.device == "auto" else [a.device] * len(CAL_CASES)
    jobs = [(c, d, a.batch, a.clones) for c, d in zip(CAL_CASES, devices)]
    os.environ.update(_ONE)
    with ProcessPoolExecutor(min(a.procs, len(jobs)), mp_context=mp.get_context("spawn")) as pool:
        out = list(pool.map(_cal_one, jobs))
    for r in out:
        print(json.dumps(r))
    import numpy as np

    per = float(np.sum([r["cpu_ms_per_decision"] * r["decisions"] for r in out]) / np.sum([r["decisions"] for r in out]))
    print(f"CPU per decision (pooled over {sum(r['decisions'] for r in out)} decisions): {per:.3f} ms; "
          f"Heavy budget (8 CPU-s per event) = {8000 / per:.0f} decisions")


# --- fidelity-b6 -----------------------------------------------------------------------------------------------------
def fidelity_b6(a) -> None:
    """Spec 5.2 fidelity check of B6 (week-1 form): on static dev instances, the CBTA-style auction's average task
    start time vs the CBGA-style grouping auction's (planner-model schedule of each auction's t = 0 plan)."""
    import numpy as np

    from cbba_sota.bench import configs
    from cbba_sota.dyn.baselines import cbta
    from cbba_sota.dyn.planner import PlanState, Problem, check_plan
    from cbba_sota.hetero import Instance

    def schedule(state, plan):
        prob = Problem(state)
        sol = prob.new_sol()
        xs = sorted((x for x in range(prob.Tk) if plan.members[prob.gid[x]]),
                    key=lambda x: (plan.keys[prob.gid[x]], prob.gid[x]))
        prob.load(sol, xs, [tuple(plan.members[prob.gid[x]]) for x in range(prob.Tk)], 0.1)
        return float(sol[4][:prob.Tk].mean()), float(sol[10][0])

    rules = [("cbta", "start"), ("cbta", "makespan"), ("cbga", "start"), ("seq", "start")]
    out = {r: [] for r in rules}
    for setting in PRIMARY:
        for i in range(a.instances):
            st = PlanState.static(Instance.from_pickle(configs.get(setting).instance_path("dev", i)))
            for rule in rules:
                plan = cbta.AuctionRH(*rule).replan(st)
                errs = check_plan(st, plan)
                if errs:
                    raise AssertionError(f"{rule} {setting} {i}: {errs[:3]}")
                out[rule].append(schedule(st, plan))
    for rule, v in out.items():
        v = np.array(v)
        print(f"  {rule[0]:5s} {rule[1]:9s} mean start {v[:, 0].mean():7.3f}  mean makespan {v[:, 1].mean():7.3f}")
    wins = int((np.array(out[("cbta", "start")])[:, 0] < np.array(out[("cbga", "start")])[:, 0]).sum())
    print(f"CBTA-style (start) has the lower average start time than CBGA-style on {wins} of "
          f"{len(out[('cbga', 'start')])} static dev instances")


# --- report ------------------------------------------------------------------------------------------------------------
PRIMARY = ("MA-AT-25-5-50", "MA-AT-50-5-50", "SA-AT-50-5-50", "SA-BT-50-5-50")


def load_rows(dirs: list[str], settings=PRIMARY, families=("F1", "F2", "F3")) -> list[dict]:
    """Completed F1-F3 rows of ``settings``; a case id seen twice (the same case run by two runners) is kept once,
    from the first directory, after checking that both runs gave the same makespan (determinism)."""
    rows, seen = [], {}
    for d in dirs:
        for f in sorted(glob.glob(os.path.join(d, "rows*.jsonl"))):
            with open(f) as fh:
                for line in fh:
                    if line.strip():
                        r = json.loads(line)
                        if r.get("status") != "completed" or r.get("family") not in families \
                                or r.get("setting") not in settings:
                            continue
                        if r["case_id"] in seen:
                            if seen[r["case_id"]] != r["makespan"]:
                                raise ValueError(f"case {r['case_id']} ran twice with different makespans")
                            continue
                        seen[r["case_id"]] = r["makespan"]
                        rows.append(r)
    return rows


def label(r: dict) -> str:
    """Method label: runner method name, the tier, and the coalition rule when it is not the default."""
    m = r["method"]
    short = {"ctrl:cbba_sota.dyn.baselines.rl_mpc:make_controller": "B2",
             "ctrl:cbba_sota.dyn.baselines.cpsat_tiers:make_controller": "B3",
             "ctrl:cbba_sota.dyn.baselines.cbta:make_controller": "B6"}
    name, _, q = m.partition("?")
    name = short.get(name, name)
    if q:
        name += "?" + q
    tier = r.get("tier", "native")
    if tier != "native" and not name.startswith("B6"):
        name += f"[{tier}]"
    coal = (r.get("options") or {}).get("coalition", "auto")
    if coal != "auto":
        name += f"[{coal}]"
    return name


def report(a) -> None:
    import numpy as np

    from cbba_sota.dyn.stats import gm_ratio, pair_rows

    rows = [r for r in load_rows(a.dirs, families=tuple(a.families)) if r["seed"] in set(a.seeds)]
    for r in rows:
        r["label"] = label(r)
    worlds = sorted({r["world"] for r in rows})
    print(f"{len(rows)} completed rows (families {a.families}, primary settings, CRN seeds {a.seeds}) from {a.dirs}; "
          f"worlds: {worlds}")
    if a.methods:
        keep = set(a.methods) | {a.ref}
        rows = [r for r in rows if r["label"] in keep]
    by = defaultdict(list)
    for r in rows:
        by[r["label"]].append(r)
    ref = a.ref
    print("\n== rows per method: success; mean makespan (successful episodes); per-event CPU ms (median over episodes "
          "of the episode p50 / p95; process time, loaded host); CPU per episode s (method calls); re-plans, wasted "
          "trips per episode")
    for m, rr in sorted(by.items()):
        p50 = [r["cpu_ms_p50"] for r in rr if r.get("cpu_ms_p50") is not None]
        p95 = [r["cpu_ms_p95"] for r in rr if r.get("cpu_ms_p95") is not None]
        ms = [r["makespan"] for r in rr if r["success"]]
        print(f"  {m:40s} n={len(rr):5d} success {sum(r['success'] for r in rr)}/{len(rr)} ms {np.mean(ms):6.2f} "
              f"cpu p50 {np.median(p50) if p50 else float('nan'):8.1f} p95 {np.median(p95) if p95 else float('nan'):8.1f}"
              f" cpu/ep {np.mean([r.get('cpu_method_s') or 0.0 for r in rr]):7.2f} "
              f"replans {np.mean([r['replans'] for r in rr]):5.1f} wasted {np.mean([r['wasted_trips'] for r in rr]):4.2f}"
              f" settings {len({r['setting'] for r in rr})} instances {len({(r['setting'], r['instance']) for r in rr})}")
    pk = ("setting", "instance", "cell", "seed")
    out = {}

    def show(b: str, where=None, tag="ALL", verbose=True):
        p = pair_rows(rows, ref, b, pair_keys=pk, cluster_keys=("setting", "instance"), method_key="label",
                      where=where)
        if len(p["log_r"]) < 2:
            return None
        g = gm_ratio(p["log_r"], p["clusters"], strata=p["strata"])
        if verbose:
            print(f"  {ref}/{b} [{tag}] {g.ratio:.4f} [{g.lo:.4f}, {g.hi:.4f}] p_less {g.p_less:.2g} "
                  f"clusters {g.n_clusters} obs {g.n_obs} wins {g.obs_wins}/{g.n_obs} "
                  f"success ref {int(p['success_a'].sum())}/{len(p['success_a'])} "
                  f"method {int(p['success_b'].sum())}/{len(p['success_b'])} dropped {int((~p['keep']).sum())}")
        return g

    for b in sorted(by):
        if b == ref:
            continue
        print(f"\n== {ref} / {b}")
        out[b] = show(b)
        for fam in a.families:
            show(b, lambda r, fam=fam: r["family"] == fam, fam)
        for s in sorted({r["setting"] for r in by[b]}):
            show(b, lambda r, s=s: r["setting"] == s, s)
        for c in sorted({r["cell"] for r in by[b]}):
            show(b, lambda r, c=c: r["cell"] == c, c)
    if ref != "SPARC[light]" or set(a.families) != {"F1", "F2", "F3"}:
        return  # the gates are defined on SPARC-Light vs a competitor, pooled over F1-F3
    print("\n== gates (dev pilot; docs/trackD-spec.md Section 8)")
    for b, g in out.items():
        if g is None:
            continue
        if b.startswith(("RL(", "B2")):
            hit = g.ratio > 0.95
            print(f"  K3 {b}: {ref}/{b} = {g.ratio:.4f} [{g.lo:.4f}, {g.hi:.4f}] -> "
                  f"{'TRIGGERED (within 5%)' if hit else 'not triggered'} (threshold: ratio > 0.95, or > "
                  f"{1 / 1.05:.4f} reading 'within 5%' as competitor / SPARC < 1.05; CI upper {g.hi:.4f})")
        if b.startswith("B3") and "heavy" in b:
            hit = g.ratio > 1.02 and g.lo > 1.0
            print(f"  K4 {b}: {ref}/{b} = {g.ratio:.4f} [{g.lo:.4f}, {g.hi:.4f}] -> "
                  f"{'TRIGGERED (heavy beats light by > 2%, CI excludes 0)' if hit else 'not triggered'}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("calibrate-b2")
    c.add_argument("--device", default="auto")
    c.add_argument("--batch", type=int, default=16)
    c.add_argument("--clones", type=int, default=6)
    c.add_argument("--procs", type=int, default=4)
    f = sub.add_parser("fidelity-b6")
    f.add_argument("--instances", type=int, default=5)
    r = sub.add_parser("report")
    r.add_argument("dirs", nargs="+")
    r.add_argument("--ref", default="SPARC[light]")
    r.add_argument("--methods", nargs="*", default=None, help="labels to compare (default: every label)")
    r.add_argument("--seeds", nargs="+", type=int, default=[0, 1], help="CRN seeds (default: the pilot's 0 1)")
    r.add_argument("--families", nargs="+", default=["F1", "F2", "F3"], help="families (F0 diagnostic: F0 static)")
    a = ap.parse_args(argv)
    if a.cmd == "calibrate-b2":
        calibrate_b2(a)
    elif a.cmd == "fidelity-b6":
        fidelity_b6(a)
    else:
        report(a)


if __name__ == "__main__":
    sys.exit(main())
