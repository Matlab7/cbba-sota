"""Track D day-1 gate, planner stress: noise-free static runs of SPARC vs open-loop ALNS (dev split only).

Every task is released at 0 and nothing is perturbed, so predictions are exact. Open loop executes the first plan
(``--iters0`` ALNS iterations); SPARC starts from the same plan and, with ``--idle eager``, re-plans (300 iterations)
whenever a robot runs out of work while open tasks exist. World: ``cbba_sota.dyn.executor`` (surrogate; its static
open-loop makespan equals env replay bit for bit, tested). With ``--idle env`` (the trigger semantics of
``DynTaskEnvX``) SPARC never re-plans after t = 0 and equals open loop exactly.

Usage: .venv/bin/python scripts/trackD_static_gate.py --n 5 --iters0 2000 --procs 10
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
from collections import defaultdict

import numpy as np

SETTINGS = ["MA-AT-25-5-50", "MA-AT-50-5-50", "SA-AT-50-5-50", "SA-BT-50-5-50"]


def _threads() -> None:
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ[k] = "1"


def job(a: tuple) -> tuple:
    setting, i, keys, iters0, idle = a
    _threads()
    from cbba_sota.bench.configs import get
    from cbba_sota.dyn.executor import Executor, make_realization
    from cbba_sota.dyn.planner import SearchConfig
    from cbba_sota.dyn.sparc import SPARC, OpenLoop, SPARCConfig
    from cbba_sota.hetero import Instance

    st = get(setting)
    inst = Instance.from_pickle(st.instance_path("dev", i))
    world = make_realization(inst, st.seed("dev", i), 0)
    r_ol = Executor(inst, world).run(OpenLoop(iters0=iters0))
    sp = SPARC(SPARCConfig(iters=300, iters0=iters0, search=SearchConfig(keys=keys)))
    r_sp = Executor(inst, world, idle=idle).run(sp)
    return setting, i, keys, r_ol["makespan"], r_sp["makespan"], r_sp["n_replans"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--iters0", type=int, default=2000)
    ap.add_argument("--idle", choices=("env", "eager"), default="eager")
    ap.add_argument("--procs", type=int, default=10)
    a = ap.parse_args()
    jobs = [(s, i, k, a.iters0, a.idle) for s in SETTINGS for i in range(a.n) for k in ("monotone", "relaxed")]
    with mp.get_context("spawn").Pool(a.procs, initializer=_threads) as pool:
        rows = pool.map(job, jobs)
    by = defaultdict(list)
    for s, i, k, ol, sp, nr in rows:
        by[k].append(np.log(sp / ol))
        by[(k, s)].append(np.log(sp / ol))
        print(f"{s} dev {i} {k:8s} open-loop {ol:.3f} SPARC {sp:.3f} ratio {sp / ol:.4f} re-plans {nr}")
    for k, v in by.items():
        v = np.asarray(v)
        print(f"{k}: GM SPARC/open-loop {np.exp(v.mean()):.4f}, max {np.exp(v.max()):.4f}, min {np.exp(v.min()):.4f}, "
              f"n={len(v)} (world: executor surrogate, idle={a.idle}, iters0={a.iters0})")


if __name__ == "__main__":
    main()
