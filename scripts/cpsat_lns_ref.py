"""CP-SAT LNS reference runs on the dev split at the paper's budgets, as CPSAT-8 in scripts/compare_dev.py.

Usage: cpsat_lns_ref.py --settings MA-AT-150-5-500 MA-AT-150-10-500 [--n 20] [--time B1] [--workers 8] [--lanes 2]

Each job builds the ``greedy.construct`` hint for min(10% of the budget, 3 s) and runs ``cpsat.solve_lns`` with
``--workers`` threads for the rest of the budget (the clock starts before the hint, after the instance and its
travel matrices are loaded). ``--lanes`` jobs run at once, so at most lanes x workers threads compute. Rows (env
replay of the plan) go to runs/alns/cpsat_ref/<setting>-dev.jsonl, keyed by (instance, budget, workers); resumable.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

from cbba_sota.bench import configs, runtime

THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS")


def _job(name: str, split: str, i: int, budget: float, workers: int) -> dict:
    from cbba_sota.hetero import Instance, evaluate, replay
    from cbba_sota.solvers import cpsat, greedy

    inst = Instance.from_pickle(configs.get(name).instance_path(split, i))
    inst.tt, inst.da  # noqa: B018  (loading, outside the budget)
    t0, c0 = time.perf_counter(), time.process_time()
    hint = greedy.construct(inst, time_limit=min(0.1 * budget, 3.0))
    hint_ms = evaluate(inst, hint).makespan
    res = cpsat.solve_lns(inst, budget, hint, workers=workers, t0=t0)
    wall, cpu = time.perf_counter() - t0, time.process_time() - c0
    rep = replay(inst, res.plan)
    return {"setting": name, "split": split, "index": i, "method": f"CPSAT-{workers}", "budget_s": budget,
            "workers": workers, "makespan": rep["makespan"], "success": rep["success"],
            "eval_makespan": evaluate(inst, res.plan).makespan, "skipped": rep["skipped"], "wall_s": wall,
            "cpu_s": cpu, "hint_makespan": hint_ms, "iterations": res.iterations,
            "trace": [(t, ms) for t, ms, _ in res.trajectory], "routes": res.plan.to_env_routes(),
            "fingerprint": runtime.fingerprint(inst), "load1": os.getloadavg()[0]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--settings", nargs="+", required=True)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--time", default="B1", help="seconds, or B1 / B2 per setting")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--lanes", type=int, default=2)
    args = ap.parse_args()
    if args.split == "test":
        raise SystemExit("reference runs for tuning use dev (or validation) splits only")
    for var in THREAD_VARS:
        os.environ[var] = "1"
    from run_alns import budget

    root = configs.RUNS_DIR / "alns" / "cpsat_ref"
    root.mkdir(parents=True, exist_ok=True)
    jobs = []
    for name in args.settings:
        s = configs.get(name)
        b = budget(s, args.time)
        out = root / f"{name}-{args.split}.jsonl"
        done = {(r["index"], r["budget_s"], r["workers"]) for r in runtime.read_rows(out) if runtime.is_current(r)}
        jobs += [(out, name, i, b) for i in range(min(args.n, s.n_instances(args.split)))
                 if (i, b, args.workers) not in done]
    print(f"{len(jobs)} jobs", flush=True)
    version = runtime.code_version()
    with ProcessPoolExecutor(args.lanes, mp_context=mp.get_context("spawn")) as ex:
        futures = {ex.submit(_job, name, args.split, i, b, args.workers): out for out, name, i, b in jobs}
        for fut in as_completed(futures):
            row = fut.result() | {"git": version}
            with futures[fut].open("a") as f:
                f.write(json.dumps(row) + "\n")
            print(f"{row['setting']} {row['index']} {row['makespan']:.3f} (hint {row['hint_makespan']:.3f})",
                  flush=True)


if __name__ == "__main__":
    main()
