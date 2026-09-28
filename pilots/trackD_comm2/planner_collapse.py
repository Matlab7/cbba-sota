"""Planner-collapse check for Track D (comm-first proposer, 2026-09-28).

Question: in rolling-horizon execution under noise (global information), does anytime coalition ALNS in the loop
beat our cheap regret-insertion constructor re-run at every event? If not, the dynamic contribution collapses to
a simple dispatch rule whatever the communication layer does.

World and RH logic: pilots/trackD_robust/rh_pilot.py, unchanged (plan following, LoRR delays, lognormal
durations, a task's coalition is frozen once its first member departs, re-plan of every uncommitted task at each
finish event from the nominal predicted state, key order kept). Arms (all nominal predictor):
  OL-A   t=0 ALNS (iters0) then open loop
  RH-A   t=0 ALNS (iters0), re-plans = ALNS warm-started from the previous plan (iters)
  OL-C   t=0 constructor portfolio only (max_iters=0), open loop
  RH-C   t=0 constructor, re-plans = cold constructor (max_iters=0, no warm start)
  RH-CA  t=0 ALNS (iters0), re-plans = cold constructor (separates t=0 quality from re-plan quality)

Usage: .venv/bin/python pilots/trackD_comm2/planner_collapse.py --n 10 --procs 24
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROB = HERE.parent / "trackD_robust"
TD = HERE.parent / "trackD"


def _setup():
    sys.path.insert(0, str(TD / "_snap"))
    sys.path.insert(0, str(TD))
    sys.path.insert(0, str(ROB))
    import rh_pilot as RP

    return RP


def make_cold_constructor(RP, seed):
    A_ = RP.alns_mod()
    from cbba_sota.hetero import Instance

    cfg = A_.ALNSConfig.v1(verify=False)
    calls = [0]
    import numpy as np

    def planner(ep, t):
        F = np.flatnonzero(~ep.committed)
        if F.size == 0:
            return None
        ready, pos = RP.predicted_state(ep, t, "nom")
        inst = ep.inst
        sub = Instance(req=inst.req[F], loc=inst.loc[F], dur=ep.dur_nom[F], ab=inst.ab, depot=inst.depot,
                       species=inst.species, speed=inst.speed)
        dout = np.empty((ep.A, F.size))
        for i in range(ep.A):
            for f, j in enumerate(F):
                dout[i, f] = ready[i] + ep.travel(i, pos[i], j)
        object.__setattr__(sub, "_dout", dout)
        plan, _ = A_.solve(sub, 60.0, seed=seed + calls[0], init_plan=None, config=cfg, max_iters=0)
        calls[0] += 1
        order = [int(F[f]) for f in plan.order() if plan.members[f]]
        return [int(j) for j in F], [plan.members[f] for f in range(F.size)], order

    return planner


def job(j):
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ[k] = "1"
    RP = _setup()
    inst, dur_real, delays, kap = RP.build(j)
    m = j["method"]
    t0 = time.perf_counter()
    ep = RP.Episode(inst, dur_real, delays, kap)
    s0 = j["noise_seed"]
    if m in ("OL-A", "RH-A", "RH-CA"):
        init = RP.make_planner("nom", j["iters0"], seed=s0)
    else:
        init = make_cold_constructor(RP, seed=s0)
    ep.set_plan(*init(ep, 0.0))
    if m == "RH-A":
        pl = RP.make_planner("nom", j["iters"], seed=1000 * s0 + 7)
    elif m in ("RH-C", "RH-CA"):
        pl = make_cold_constructor(RP, seed=1000 * s0 + 7)
    else:
        pl = None
    res = ep.run(pl)
    return {k: j[k] for k in ("setting", "inst", "scenario", "noise_seed", "method", "iters", "iters0")} | res | {
        "wall_s": time.perf_counter() - t0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--settings", nargs="+", default=["MA-AT-25-5-50", "MA-AT-50-5-50", "SA-AT-50-5-50",
                                                      "SA-BT-50-5-50"])
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--procs", type=int, default=24)
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--iters0", type=int, default=6000)
    ap.add_argument("--methods", nargs="+", default=["OL-A", "RH-A", "OL-C", "RH-C", "RH-CA"])
    ap.add_argument("--scenarios", nargs="+", default=["N2-dur.3/causal", "N3-mod/causal"])
    ap.add_argument("--out", default=str(HERE / "out" / "planner_collapse.jsonl"))
    args = ap.parse_args()
    RP = _setup()
    from cbba_sota.bench.configs import get

    jobs = []
    for s in args.settings:
        st = get(s)
        for i in range(args.n):
            for scn in args.scenarios:
                for ns in range(args.seeds):
                    for m in args.methods:
                        jobs.append({"setting": s, "inst": i, "path": str(st.instance_path("dev", i)),
                                     "inst_key": st.seed("dev", i), "scenario": scn, "noise_seed": ns,
                                     "method": m, "iters": args.iters, "iters0": args.iters0})
    jobs.sort(key=lambda j: j["method"] != "RH-A")
    print(len(jobs), "jobs", flush=True)
    t0 = time.time()
    with mp.get_context("spawn").Pool(args.procs) as pool, open(args.out, "w") as f:
        for k, r in enumerate(pool.imap_unordered(job, jobs)):
            f.write(json.dumps(r) + "\n")
            f.flush()
            if (k + 1) % 50 == 0:
                print(f"{k + 1}/{len(jobs)} {time.time() - t0:.0f}s", flush=True)
    print("done", time.time() - t0, flush=True)


if __name__ == "__main__":
    main()
