"""Track D / robustness angle, pilot P3: stability-gated re-planning under execution noise.

out/c2 showed: re-planning at every task finish ties open loop at 3000 iterations and is 3-5% worse at 30 iterations
(nervousness), but gains 2-4% under severe delays (N4). A stability gate adopts a re-plan only when its predicted
makespan (same state, same predictor) beats the warm plan's by more than eps. Question: does the gate keep the N4 gain
without the N2/N3 loss, i.e. is it more than the fixed rule "never re-plan on noise"?

Methods (rh_pilot executor and predictors; t = 0 plan 20000 iterations as in out/c2):
  GT-ce@N/eps   at every finish: ALNS N iterations from the warm plan; adopt iff ms_new < (1 - eps) * ms_warm
Reference rows (OL-ce, RH-ce@N) come from out/c2.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rh_pilot as R  # noqa: E402


def make_gated(pred, iters, eps, seed):
    A_ = R.alns_mod()
    from cbba_sota.hetero import Instance, Plan

    cfg = A_.ALNSConfig.v1(verify=False)
    calls = [0]
    stats = {"adopt": 0, "reject": 0}

    def planner(ep, t):
        F = np.flatnonzero(~ep.committed)
        if F.size == 0:
            return None
        kap = 1.0 if pred == "nom" else ep.kappa
        ready, pos = R.predicted_state(ep, t, pred)
        inst = ep.inst
        sub = Instance(req=inst.req[F], loc=inst.loc[F], dur=ep.dur_nom[F], ab=inst.ab, depot=inst.depot,
                       species=inst.species, speed=inst.speed / kap)
        dout = np.empty((ep.A, F.size))
        for i in range(ep.A):
            for f, j in enumerate(F):
                dout[i, f] = ready[i] + ep.travel(i, pos[i], j) * kap
        object.__setattr__(sub, "_dout", dout)
        warm = Plan([ep.plan_mem[j] for j in F], ep.key[F], ep.A)
        p0, s0 = A_.solve(sub, 60.0, seed=seed + calls[0], init_plan=warm, config=cfg, max_iters=0)
        p1, s1 = A_.solve(sub, 60.0, seed=seed + calls[0], init_plan=warm, config=cfg, max_iters=iters)
        calls[0] += 1
        plan = p1 if s1.makespan < (1.0 - eps) * s0.makespan else p0
        stats["adopt" if plan is p1 else "reject"] += 1
        order = [int(F[f]) for f in plan.order() if plan.members[f]]
        return [int(j) for j in F], [plan.members[f] for f in range(F.size)], order

    planner.stats = stats
    return planner


def run_job(job):
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ[k] = "1"
    inst, dur_real, delays, kap = R.build(job)
    _, rest = job["method"].split("-", 1)
    pred, rest = rest.split("@")
    iters, eps = rest.split("/")
    iters, eps = int(iters), float(eps)
    t0 = time.perf_counter()
    ep = R.Episode(inst, dur_real, delays, kap)
    tasks, members, order = R.make_planner(pred, job["iters0"], seed=job["noise_seed"])(ep, 0.0)
    ep.set_plan(tasks, members, order)
    pl = make_gated(pred, iters, eps, seed=1000 * job["noise_seed"] + 7)
    res = ep.run(pl)
    return {**{k: job[k] for k in ("setting", "inst", "scenario", "noise_seed", "method")}, **res,
            "wall_s": time.perf_counter() - t0, "iters": iters, "iters0": job["iters0"], **pl.stats}


def main():
    from cbba_sota.bench.configs import get

    procs = int(sys.argv[1]) if len(sys.argv) > 1 else 32
    out = HERE / "out" / "c2" / "gate.jsonl"
    methods = ["GT-ce@300/0.005", "GT-ce@300/0.02", "GT-ce@3000/0.005"]
    jobs = []
    for s in ["MA-AT-25-5-50", "MA-AT-50-5-50", "SA-AT-50-5-50", "SA-BT-50-5-50"]:
        st = get(s)
        for i in range(10):
            for scn in R.SCEN:
                for ns in range(2):
                    for m in methods:
                        mm = m.replace("-ce", "-nom") if scn.startswith("N2") else m
                        jobs.append({"setting": s, "inst": i, "path": str(st.instance_path("dev", i)),
                                     "inst_key": st.seed("dev", i), "scenario": scn, "noise_seed": ns,
                                     "method": mm, "iters0": 20000})
    print(f"{len(jobs)} jobs", flush=True)
    t0 = time.time()
    with mp.get_context("spawn").Pool(procs) as pool, out.open("w") as f:
        for k, r in enumerate(pool.imap_unordered(run_job, jobs)):
            f.write(json.dumps(r) + "\n")
            f.flush()
            if k % 100 == 0:
                print(f"{k}/{len(jobs)} {time.time() - t0:.0f}s", flush=True)
    print("done", time.time() - t0, flush=True)


if __name__ == "__main__":
    main()
