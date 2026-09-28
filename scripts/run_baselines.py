"""Resumable classical-baseline campaign: runs/baselines/<setting>/<split>.jsonl, one row per instance x method x
budget.

Usage: run_baselines.py --split dev --methods greedy greedy_repo construct [--settings NAME ...]
       run_baselines.py --split dev --methods cpsat cpsat_hint --budgets 5 30 120 --workers 8 --cap 16

Methods (``makespan``/``success``/``awt`` always come from the env):
- greedy, greedy_repo: the env's own greedy loops (solvers.greedy.greedy_nearest / greedy_repo); single-threaded,
  no budget. ``replay_makespan`` re-runs the extracted routes through execute_by_route for reference only.
- construct: plan-level constructive heuristic for ``budget_s`` seconds (single-threaded), env-replayed.
- cpsat: monolithic CP-SAT model, no hint; cpsat_hint: the same with a constructive hint (``construct`` for
  min(10% of the budget, 3 s), counted in the budget); cpsat_knn: hinted, arcs restricted to the 10 nearest
  candidates; cpsat_lns: LNS in CP from the hint. CP-SAT jobs use ``--workers`` threads each; plans are pruned to
  minimal covers and env-replayed. Rows named *_r8 come from an earlier run whose hint was ``construct`` with 8
  restarts.
Jobs are packed so that the summed threads never exceed ``--cap``, and each job is pinned to its own disjoint CPUs
out of ``--cpus`` (default: the ``--cap`` least busy physical cores at start). Budgets start after the instance and
its travel matrices are loaded. ``routes`` are per agent, 0-based task ids. Every row records the instance
``fingerprint``, the ``git`` code version, its CPU ``affinity``, the load average at start and end and the change of
the cgroup's throttling counters; resuming skips only rows whose fingerprint matches the current instance.

Summary: run_baselines.py --split dev --summary [--settings ...] [--instances k] [--methods ...] [--budgets ...]:
per setting, method and budget the number of rows, rows with a plan, success rate (all tasks finished and makespan
< 200), mean env makespan over rows with a plan (failed env runs enter at the env's value) and mean wall/CPU
seconds, over the instances that every listed (method, budget) has.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path
from typing import NamedTuple

from cbba_sota.bench import runtime

_ONE = {k: "1" for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS")}
GREEDY = ("greedy", "greedy_repo")
CPSAT = ("cpsat", "cpsat_hint", "cpsat_knn", "cpsat_lns")
METHODS = (*GREEDY, "construct", *CPSAT)
HINT_SHARE, HINT_MAX = 0.1, 3.0  # constructive warm start: min(10% of the budget, 3 s), inside the CP-SAT budget


def hint_seconds(budget: float) -> float:
    return min(HINT_SHARE * budget, HINT_MAX)


class Job(NamedTuple):
    setting: str
    split: str
    instance: int
    method: str
    budget: float | None
    workers: int
    cpus: tuple[int, ...] = ()  # pinned CPUs, assigned at submission

    @property
    def threads(self) -> int:
        return self.workers if self.method in CPSAT else 1

    @property
    def key(self) -> tuple:
        return self.instance, self.method, self.budget


def _run(job: Job) -> dict:
    """One job in a pool process, pinned to ``job.cpus`` (if any) for its duration."""
    if job.cpus:
        runtime.pin(job.cpus)
    probe = runtime.Probe()
    return _solve(job) | probe.fields()


def _solve(job: Job) -> dict:
    from cbba_sota.bench import configs
    from cbba_sota.hetero import Instance, evaluate, replay, replay_routes
    from cbba_sota.solvers import cpsat, greedy

    setting = configs.get(job.setting)
    path = setting.instance_path(job.split, job.instance)
    inst = Instance.from_pickle(path)
    row = {"setting": job.setting, "split": job.split, "instance": job.instance,
           "seed": None if setting.source is not None else setting.seed(job.split, job.instance),
           "method": job.method, "budget_s": job.budget, "workers": job.threads,
           "fingerprint": runtime.fingerprint(inst)}
    if job.method in GREEDY:
        c0 = time.process_time()
        res = greedy.greedy_nearest(path) if job.method == "greedy" else greedy.greedy_repo(path)
        cpu = time.process_time() - c0
        routes_1 = [[j + 1 for j in r] for r in res.routes]
        try:
            again = replay_routes(path, routes_1)["makespan"]
        except ValueError:  # a task nobody reached (failed run)
            again = None
        return row | {"makespan": res.makespan, "success": res.success, "env_finished": res.env_finished,
                      "awt": res.awt, "wall_s": res.wall_s, "cpu_s": cpu, "replay_makespan": again,
                      "depot_revisits": res.depot_revisits, "routes": res.routes}

    inst.tt, inst.da  # noqa: B018  (instance loading, outside the budget)
    t0, c0 = time.perf_counter(), time.process_time()
    extra = {}
    if job.method == "construct":
        plan = greedy.construct(inst, time_limit=job.budget)
    else:
        hint = None if job.method == "cpsat" else greedy.construct(inst, time_limit=hint_seconds(job.budget))
        if hint is not None:
            extra |= {"hint_makespan": evaluate(inst, hint).makespan, "hint_s": time.perf_counter() - t0}
        if job.method == "cpsat_lns":
            res = cpsat.solve_lns(inst, job.budget, hint, workers=job.workers, t0=t0)
        else:
            res = cpsat.solve_full(inst, job.budget, workers=job.workers, hint=hint, t0=t0,
                                   knn=10 if job.method == "cpsat_knn" else None)
        plan = res.plan
        extra |= {"status": res.status, "objective": res.objective, "bound": res.bound,
                  "iterations": res.iterations, "trajectory": [(round(t, 3), m) for t, m, _ in res.trajectory]}
    wall, cpu = time.perf_counter() - t0, time.process_time() - c0
    if plan is None:  # no solution within the budget
        return row | {"makespan": None, "success": False, "awt": None, "wall_s": wall, "cpu_s": cpu,
                      "eval_makespan": None, "routes": None} | extra
    rep = replay(inst, plan)
    return row | {"makespan": rep["makespan"], "success": rep["success"], "env_finished": rep["env_finished"],
                  "awt": rep["awt"], "wall_s": wall, "cpu_s": cpu, "eval_makespan": evaluate(inst, plan).makespan,
                  "skipped": rep["skipped"], "routes": plan.routes()} | extra


def out_path(root: Path, setting: str, split: str) -> Path:
    return root / setting / f"{split}.jsonl"


def done_keys(path: Path) -> set[tuple]:
    """(instance, method, budget) of the rows computed on the current instance (matching fingerprint)."""
    return {(row["instance"], row["method"], row["budget_s"]) for row in runtime.read_rows(path)
            if runtime.is_current(row)}


def _write(root: Path, row: dict) -> None:
    path = out_path(root, row["setting"], row["split"])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(row) + "\n")


def summarize(root: Path, settings: list[str], split: str, instances: int | None,
              methods: list[str] | None = None, budgets: list[float] | None = None) -> None:
    import numpy as np

    from cbba_sota.hetero.replay import succeeded

    for name in settings:
        path = out_path(root, name, split)
        if not path.exists():
            continue
        rows: dict[tuple, dict[int, dict]] = {}
        for row in runtime.read_rows(path):
            if runtime.is_stale(row):
                continue
            if ((instances is None or row["instance"] < instances) and (methods is None or row["method"] in methods)
                    and (budgets is None or row["budget_s"] is None or row["budget_s"] in budgets)):
                rows.setdefault((row["method"], row["budget_s"]), {})[row["instance"]] = row
        if not rows:
            continue
        common = set.intersection(*(set(v) for v in rows.values()))
        print(f"{name} {split}: {len(common)} common instances")
        for (method, budget), by_inst in sorted(rows.items(), key=lambda kv: (kv[0][0], kv[0][1] or 0)):
            sel = [by_inst[i] for i in sorted(common)]
            ms = np.array([np.nan if r["makespan"] is None else r["makespan"] for r in sel], float)
            mean = np.mean(ms[np.isfinite(ms)]) if np.isfinite(ms).any() else np.nan
            print(f"  {method:12s} {budget!s:>6} n={len(sel):3d} with_plan={np.isfinite(ms).sum():3d} "
                  f"success={np.mean([succeeded(r['success'], r['makespan']) for r in sel]):.2f} "
                  f"makespan={mean:8.3f} "
                  f"wall={np.mean([r['wall_s'] for r in sel]):7.2f}s cpu={np.mean([r['cpu_s'] for r in sel]):7.2f}s")


def main() -> None:
    from cbba_sota.bench import configs

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--settings", nargs="*", default=[s.name for s in configs.SETTINGS])
    ap.add_argument("--split", required=True, choices=list(configs.SPLITS))
    ap.add_argument("--methods", nargs="+", default=None, choices=METHODS, help="default: greedy greedy_repo")
    ap.add_argument("--budgets", nargs="*", type=float, default=None, help="seconds (construct, CP-SAT); default 30")
    ap.add_argument("--workers", type=int, default=8, help="CP-SAT threads per solve")
    ap.add_argument("--cap", type=int, default=16, help="max threads in use at once")
    ap.add_argument("--cpus", default=None, help="CPU list to pin jobs to, e.g. 0-15 (default: least busy cores)")
    ap.add_argument("--instances", type=int, default=None, help="only the first k instances")
    ap.add_argument("--out", type=Path, default=configs.RUNS_DIR / "baselines")
    ap.add_argument("--summary", action="store_true", help="print the summary table and exit")
    ap.add_argument("--allow-dirty", action="store_true", help="run on uncommitted source (smoke tests only)")
    args = ap.parse_args()
    if args.summary:
        return summarize(args.out, args.settings, args.split, args.instances, args.methods, args.budgets)
    args.methods = args.methods or list(GREEDY)
    args.budgets = args.budgets or [30.0]
    if args.workers > args.cap:
        ap.error("--workers exceeds --cap")

    jobs: list[Job] = []
    seen: set[Job] = set()
    for name in args.settings:
        setting = configs.get(name)
        if args.split not in setting.splits():
            continue
        done = done_keys(out_path(args.out, name, args.split))
        n = setting.n_instances(args.split) if args.instances is None else args.instances
        for budget in sorted(args.budgets):  # short budgets first so that complete tables land early
            for i in range(n):
                for m in args.methods:
                    b = None if m in GREEDY else budget
                    job = Job(name, args.split, i, m, b, args.workers)
                    if job.key not in done and job not in seen:
                        seen.add(job)
                        jobs.append(job)
    size = {name: configs.get(name).n_tasks * configs.get(name).n_agents for name in args.settings}
    jobs.sort(key=lambda j: (j.budget or 0.0, -size[j.setting]))  # per budget, largest instances first
    print(f"{len(jobs)} jobs", flush=True)
    if not jobs:
        return

    os.environ.update(_ONE)  # single-threaded BLAS/numba in the workers; CP-SAT threads are set per solve
    free = runtime.parse_cpus(args.cpus) if args.cpus else runtime.choose_cpus(args.cap)
    if len(free) < args.cap:
        ap.error(f"--cpus has {len(free)} CPUs for --cap {args.cap}")
    free, version = free[:args.cap], runtime.require_clean(args.allow_dirty)
    print(f"pinning to CPUs {runtime.format_cpus(free)}, code {version}", flush=True)
    queue, running = list(jobs), {}
    t0 = time.time()
    with ProcessPoolExecutor(args.cap, mp_context=mp.get_context("spawn")) as ex:
        while queue or running:
            k = 0
            while k < len(queue):  # first queued job that fits
                if queue[k].threads <= len(free):
                    job = queue.pop(k)
                    job = job._replace(cpus=tuple(free[:job.threads]))
                    del free[:job.threads]
                    running[ex.submit(_run, job)] = job
                else:
                    k += 1
            finished, _ = wait(running, return_when=FIRST_COMPLETED)
            for fut in finished:
                job = running.pop(fut)
                free = sorted(free + list(job.cpus))
                row = fut.result() | {"git": version}
                _write(args.out, row)
                ms = row["makespan"]
                print(f"[{time.time() - t0:7.0f}s] {job.setting} {job.split} {job.instance} {job.method} "
                      f"{job.budget} -> {ms if ms is None else round(ms, 3)} ok={row['success']}", flush=True)


if __name__ == "__main__":
    main()
