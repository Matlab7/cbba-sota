"""Anytime curves on the validation split: quality vs wall-clock budget on 1 and 8 cores, and the headroom report.

Usage: anytime.py run [--grid phase1b|c1] [--settings NAME ...] [--methods M ...] [--n 10] [--lanes 4] [--cpus LIST]
                      [--out DIR]
       anytime.py report [--grid phase1b|c1] [--settings NAME ...] [--csv-dir DIR] [--png FILE] [--out DIR]
       anytime.py tune --variants PCPSAT:sub_time=0.5 ... [--refs ALNS2 CPSAT] [--settings ...] [--n 10] [--out DIR]
       anytime.py tune-report [--out DIR]

Methods (one row per method, cores and budget; budgets 0.5, 1, 2, 5, 10 s, B1 and 2B1 = 2 x B1, where B1 is the
paper's RL(s.10) time of the setting):
- ALNS2: ALNS v2 (the default ``ALNSConfig``), 1 worker or 8 forked workers, seed 0.
- ALNS1: ALNS v1 (``ALNSConfig.v1()``), 1 worker at every budget, 8 workers at B1.
- CPSAT: CP-SAT LNS with 1 or 8 threads from the ``greedy.construct`` hint built in min(10% of the budget, 3 s),
  sub-solves of ``bks.sub_time(setting)`` seconds (the better variant on dev).
- RL: RL(s.N) with the released policy on 1 or 8 single-threaded CPU processes, each sampling lockstep batches
  until the budget is used (``rl.sample_until``; the first batch has 1 sample at budgets up to ``RL_FIRST_ONE``
  = 10 s, else 4), scored by the best rollout's own env run.
- CONSTRUCT: the regret-insertion constructor with randomized restarts. Restarts are independent of the budget,
  so one job runs 8 restart streams (seeds 0-7, the sequence of ``greedy.construct``) on 8 cores up to the largest
  budget and reads every budget off them: CONSTRUCT-1 at b is stream 0's best plan finished by b, CONSTRUCT-8 the
  best over the 8 streams (the first plan if none finished by b; ``t_found`` says when it did).
- PCPSAT: parallel CP-SAT LNS (``cpsat.solve_lns_parallel``): 1 or 8 forked processes with one CP-SAT thread each
  improving one shared incumbent, started from the best of their ``greedy.construct`` restart streams built in
  min(10% of the budget, 3 s); sub-solves of at most ``bks.pcpsat_sub_time`` seconds (the dev choice, ``tune``).
- CPFULL: the monolithic CP-SAT model (``cpsat.solve_full``, CP-SAT's own portfolio of search and LNS workers) on 1
  or 8 threads, hinted with the best of 1 or 8 forked ``greedy.construct`` restart streams built in min(10% of the
  budget, 3 s); arcs to the ``bks.cpfull_knn`` nearest candidate tasks only on large instances (the dev choice).
- CTAS: CTAS-D, the benchmark paper's exact MILP (``cbba_sota.solvers.ctas``, verified against its shipped Gurobi
  runs), solved by CP-SAT as a MIP backend on 1 or 8 threads within the budget (model building included), no warm
  start. A budget without a converted plan is a failure (makespan 200).
Rows of the multi-process methods (ALNS2/ALNS1 with 8 workers, PCPSAT, CPFULL, RL, CONSTRUCT) record the CPU
seconds of all their processes and threads (``cpu_s``; CONSTRUCT: ``stream_cpu_s``).
Grid ``c1`` (the same-host C1/C2 campaign of 2026-09-28 on the second server): on every instance ALNS2 on 1 core at
0.5, 1, 2 s and B1; ALNS2, CPSAT, PCPSAT, CPFULL, CTAS and RL on 8 cores at B1 and the 8 restart streams; ALNS2,
PCPSAT and CPFULL on 8 cores at 2B1; on instances 0-4 also ALNS2 on 1 core at 5 and 10 s and ALNS2, PCPSAT, CPFULL and
RL on 8 cores at 0.5-10 s. 500 tasks: ALNS2 on 1 core at 0.5-10 s and B1, the 8-core B1 points except CTAS, and the 8
restart streams up to B1. CTAS-D found no incumbent within B1 on either 500-task dev pilot instance and its solver
held 55-87 GB each (docs/results/ctas/ctas500_dev_pilot.jsonl, scripts/ctas500_pilot.py), so several such jobs at once
would exceed the container's 330 GB; it counts as failed there. 20-task settings are not in this grid.
Grid ``phase1b`` (``grid``; cut to the 6-hour compute window on a shared host, see docs/headroom-2026-09.md): on
instances 0-4
every method at every budget on 1 core, and ALNS2, CPSAT and RL at 0.5-10 s on 8 cores; on every instance ALNS2,
ALNS1, CPSAT and RL on 1 core at 0.5, 1, 2 s and B1 and on 8 cores at B1, ALNS2 on 8 cores at 2B1, CPSAT on 8 cores
at 2B1 up to 50 tasks, and the 8 restart streams (up to 2B1, 200 tasks up to B1). 20-task settings run 8 cores at B1
only. 500 tasks: ALNS2 and ALNS1 on 1 core up to 10 s, ALNS2 on 1 core at B1, ALNS2, CPSAT and RL on 8 cores at B1,
and one restart stream up to B1.

Execution reuses scripts/compare_dev.py: ``--lanes`` lanes of 8 warm single-threaded processes pinned to their own
8 CPUs; a lane runs one 8-core job or a bundle of up to 8 one-core jobs of the same budget. Budgets start after
the instance and its travel matrices are loaded. Every makespan is the env replay of the plan (RL: the rollout's
own env run). Rows go to <out>/<setting>.jsonl (default runs/anytime) with 0-based routes, fingerprint, code version
(a dirty source tree is refused unless ``--allow-dirty``), host, CPU affinity, load and throttling counters
(resumable per setting, instance, method, cores and budget). The container shares a CPU
quota with other tenants, and a saturated quota stalls every process in it: a lane starts a unit only when at most
``--quiet`` (40%) of the cgroup's CPU periods were throttled over 1 s, and rows whose job saw more than 40%
throttled periods (``disturbed``) are re-run on resume and left out of reports. A throttled period costs a job
only the rest of that period after the quota ran out: on this host, one-process jobs with at most 40% throttled
periods still got 87-100% of their core (mean 96%), the same for every method.

``report`` reads the BKS table of scripts/bks.py (``bks.py collect``) and writes, per setting: the mean gap to the
BKS of every method, core count and budget; the share of the constructor-to-BKS gap each method closes (against
the constructor at the same budget and cores, as a ratio of means); paired ratios ALNS2 / competitor at B1 and
2B1 on 8 cores and ALNS2 on 1 core at 0.5, 1 and 2 s against every competitor at B1 on 8 cores (bootstrap 95% CI,
10,000 resamples, seed 0, wins-losses, one-sided paired t-test of mean log ratio < 0). A run whose wall time
exceeds its budget by more than 10% + 0.25 s is late: it has no solution within the budget and is left out of the
means (the count of on-time runs is printed). A failed plan counts as 200.

``tune`` runs competitor variants (``METHOD:param=value``, e.g. ``PCPSAT:sub_time=0.5``, ``CPFULL:knn=10``;
``knn=0`` is the full arc set) and reference methods at B1 on 8 cores on the dev split (the tuning split), rows to
``--out`` (default runs/tune_dev) with ``variant``; ``tune-report`` prints per setting each variant's mean makespan,
its paired ratio to the best variant of its method, and the CPU share used.
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import random
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bks
import compare_dev as cd

from cbba_sota.bench import configs, runtime
from cbba_sota.bench.configs import RUNS_DIR
from cbba_sota.hetero.replay import succeeded

OUT = RUNS_DIR / "anytime"
SPLIT = "val"
LANE = cd.LANE
LOW = (0.5, 1.0, 2.0, 5.0, 10.0)
METHODS = ("ALNS2", "ALNS1", "CPSAT", "PCPSAT", "CPFULL", "CTAS", "RL", "CONSTRUCT")
COMPETITORS = ("CPSAT", "PCPSAT", "CPFULL", "CTAS", "RL", "CONSTRUCT")
EXTENDED = 5  # instances below this get the full budget grid (see ``grid``)
FAIL = cd.FAIL
QUIET = 0.4  # a lane starts a unit only when at most this share of the cgroup's CPU periods was throttled
RL_FIRST_ONE = 10.0  # RL budgets up to this many seconds start with a lockstep batch of 1 sample (else 4)
DISTURBED = 0.4  # rows whose job saw a larger throttled share are re-run on resume and left out of reports
LATE_RULES = {"phase1b": "drop", "c1": "score", "test": "score"}  # how reports treat late runs (``at_budget``)
LATE_RULE = "drop"


def budgets(setting) -> dict[str, float]:
    b1 = setting.paper["RL(s.10)"].time_s
    return {**{f"{b:g}": b for b in LOW}, "B1": b1, "2B1": 2 * b1}


def grid(setting, i: int = 0) -> list[tuple[str, int, str]]:
    """(method, cores, budget label) of the jobs of instance ``i``; CONSTRUCT jobs carry the label "stream" (their
    rows are read off at every budget). Instances from ``EXTENDED`` on get the core points only: 1 core at 0.5, 1,
    2 s and B1, 8 cores at B1 (and 2B1 for ALNS2 and, up to 50 tasks, CPSAT); 20-task settings run 8 cores at B1
    only."""
    labels = list(budgets(setting))
    T = setting.n_tasks
    if T >= 500:
        jobs = [("ALNS2", 1, b) for b in (*labels[:5], "B1")] + [("ALNS1", 1, b) for b in labels[:5]]
        return jobs + [(m, LANE, "B1") for m in ("ALNS2", "CPSAT", "RL")] + [("CONSTRUCT", 1, "stream")]
    extended = i < EXTENDED
    ones = labels if extended else ["0.5", "1", "2", "B1"]
    jobs = [(m, 1, b) for m in ("ALNS2", "ALNS1", "CPSAT", "RL") for b in ones]
    jobs += [(m, LANE, "B1") for m in ("ALNS2", "ALNS1", "CPSAT", "RL")]
    if T > 20:
        jobs += [("ALNS2", LANE, "2B1")] + ([("CPSAT", LANE, "2B1")] if T <= 50 else [])
        if extended:
            jobs += [(m, LANE, b) for m in ("ALNS2", "CPSAT", "RL") for b in labels[:5]]
    return jobs + [("CONSTRUCT", LANE, "stream")]


def grid_c1(setting, i: int = 0) -> list[tuple[str, int, str]]:
    """(method, cores, budget label) of instance ``i`` in the same-host C1/C2 campaign (see the module docstring)."""
    labels = list(budgets(setting))
    eight = [(m, LANE, "B1") for m in ("ALNS2", "CPSAT", "PCPSAT", "CPFULL", "CTAS", "RL")]
    eight.append(("CONSTRUCT", LANE, "stream"))
    if setting.n_tasks >= 500:  # CTAS-D: no incumbent within B1, 55-87 GB per solve (module docstring)
        return [("ALNS2", 1, b) for b in (*labels[:5], "B1")] + [job for job in eight if job[0] != "CTAS"]
    if setting.n_tasks <= 20:
        return []
    jobs = [("ALNS2", 1, b) for b in ("0.5", "1", "2", "B1")] + eight + [(m, LANE, "2B1")
                                                                           for m in ("ALNS2", "PCPSAT", "CPFULL")]
    if i < EXTENDED:
        jobs += [("ALNS2", 1, b) for b in ("5", "10")]
        jobs += [(m, LANE, b) for m in ("ALNS2", "PCPSAT", "CPFULL", "RL") for b in labels[:5]]
    return jobs


def grid_test(setting, i: int = 0) -> list[tuple[str, int, str]]:
    """The confirmatory test-split grid of the frozen pre-registration: the ``c1`` points of every instance (C1 at B1
    and 2B1 on 8 cores, C2 with ALNS2 on 1 core at 0.5, 1, 2 s and B1, the 8 restart streams) without the anytime
    curves of instances 0-4, which are descriptive and come from val."""
    return grid_c1(setting, EXTENDED)


GRIDS = {"phase1b": grid, "c1": grid_c1, "test": grid_test}
GRID = grid  # --grid


def stream_budget(setting) -> float:
    """Length of the CONSTRUCT restart streams: the largest budget read off them."""
    b = budgets(setting)
    return b["B1"] if setting.n_tasks >= 200 else b["2B1"]


# --- jobs (run in the lane processes) --------------------------------------------------------------------------


def run_alns(name: str, split: str, i: int, workers: int, budget: float, preset: str) -> dict:
    from cbba_sota.solvers import alns

    inst, _, fp = cd._load(name, split, i)
    cfg = alns.ALNSConfig.v1() if preset == "ALNS1" else alns.ALNSConfig()
    c0, k0 = time.process_time(), cd._children_cpu()
    t0 = time.monotonic()
    plan, st = alns.solve(inst, budget, seed=0, n_workers=workers, config=cfg)
    wall = time.monotonic() - t0
    cpu = time.process_time() - c0 + cd._children_cpu() - k0
    return cd._plan_row(inst, plan, wall, cpu) | {"iterations": st.iterations, "init_makespan": st.init_makespan,
                                                  "trace": st.trace, "fingerprint": fp}


def run_cpsat(name: str, split: str, i: int, budget: float, workers: int) -> dict:
    from cbba_sota.hetero import evaluate
    from cbba_sota.solvers import cpsat, greedy

    inst, _, fp = cd._load(name, split, i)
    t0, c0 = time.perf_counter(), time.process_time()
    hint = greedy.construct(inst, time_limit=min(0.1 * budget, 3.0))
    res = cpsat.solve_lns(inst, budget, hint, workers=workers, sub_time=bks.sub_time(name), t0=t0)
    wall, cpu = time.perf_counter() - t0, time.process_time() - c0
    return cd._plan_row(inst, res.plan, wall, cpu) | {
        "iterations": res.iterations, "init_makespan": evaluate(inst, hint).makespan,
        "trace": [(t, ms) for t, ms, _ in res.trajectory], "sub_time": bks.sub_time(name), "fingerprint": fp}


def run_pcpsat(name: str, split: str, i: int, budget: float, workers: int, sub_time: float | None = None) -> dict:
    from cbba_sota.solvers import cpsat

    inst, _, fp = cd._load(name, split, i)
    sub_time = bks.pcpsat_sub_time(name) if sub_time is None else sub_time
    t0 = time.perf_counter()
    res = cpsat.solve_lns_parallel(inst, budget, workers=workers, init_time=min(0.1 * budget, 3.0),
                                   sub_time=sub_time, t0=t0)
    return cd._plan_row(inst, res.plan, time.perf_counter() - t0, res.cpu_s) | {
        "iterations": res.iterations, "improvements": res.improvements, "init_makespan": res.init_makespan,
        "trace": [(t, ms) for t, ms, _ in res.trajectory], "sub_time": sub_time, "fingerprint": fp}


def run_cpfull(name: str, split: str, i: int, budget: float, workers: int, knn: int | None = None) -> dict:
    from cbba_sota.hetero import evaluate
    from cbba_sota.solvers import cpsat

    inst, _, fp = cd._load(name, split, i)
    knn = bks.cpfull_knn(name) if knn is None else knn
    t0, c0, k0 = time.perf_counter(), time.process_time(), cd._children_cpu()
    hint = cpsat.construct_parallel(inst, workers, t0 + min(0.1 * budget, 3.0))
    res = cpsat.solve_full(inst, budget, workers=workers, hint=hint, knn=knn or None, t0=t0)
    wall, cpu = time.perf_counter() - t0, time.process_time() - c0 + cd._children_cpu() - k0
    return cd._plan_row(inst, res.plan, wall, cpu) | {
        "status": res.status, "bound": res.bound, "knn": knn, "init_makespan": evaluate(inst, hint).makespan,
        "trace": [(t, ms) for t, ms, _ in res.trajectory], "fingerprint": fp}


def run_ctas(name: str, split: str, i: int, budget: float, workers: int) -> dict:
    from cbba_sota.solvers import ctas

    inst, _, fp = cd._load(name, split, i)
    t0 = time.perf_counter()
    res = ctas.solve(inst, budget, backend="CP_SAT", threads=workers, t0=t0)
    wall = time.perf_counter() - t0
    extra = {"status": res.status, "objective": res.objective, "bound": res.bound, "qmax": res.qmax,
             "build_s": res.build_s, "fingerprint": fp}
    if res.plan is None:
        return {"makespan": FAIL, "success": False, "env_finished": False, "eval_makespan": None, "skipped": None,
                "wall_s": wall, "cpu_s": res.cpu_s, "routes": None, "routes_base": 0} | extra
    return cd._plan_row(inst, res.plan, wall, res.cpu_s) | extra


def run_stream(name: str, split: str, i: int, seed: int, checkpoints: list[float]) -> dict:
    """One restart stream of ``greedy.construct`` (same plan sequence for ``seed``) until the last checkpoint.
    Returns, per checkpoint, the best plan finished by then (or the first plan) with its env replay."""
    from cbba_sota.hetero import evaluate, replay
    from cbba_sota.solvers import greedy

    inst, _, fp = cd._load(name, split, i)
    rng = np.random.default_rng(seed)
    t0, c0, end = time.perf_counter(), time.process_time(), max(checkpoints)
    found: list[tuple[float, float, object]] = []  # improvements: (finish time, makespan, plan)
    k = 0
    while True:
        if k == 0:
            plan = greedy.dispatch(inst)
        elif k % 2:
            plan = greedy.dispatch(inst, noise=0.3, seed=int(rng.integers(2**31)))
        else:
            plan = greedy.insertion(inst, rng)
        ms = evaluate(inst, plan).makespan
        t = time.perf_counter() - t0
        k += 1
        if not found or ms < found[-1][1]:
            found.append((t, ms, plan))
        if t >= end:
            break
    cpu = time.process_time() - c0
    out, replays = {}, {}
    for b in checkpoints:
        t, ms, plan = next((f for f in reversed(found) if f[0] <= b), found[0])
        if id(plan) not in replays:
            replays[id(plan)] = replay(inst, plan) | {"eval_makespan": ms, "routes": plan.routes()}
        rep = replays[id(plan)]
        out[b] = {"makespan": rep["makespan"], "success": rep["success"], "env_finished": rep["env_finished"],
                  "eval_makespan": ms, "skipped": rep["skipped"], "routes": rep["routes"], "t_found": t,
                  "constructions": sum(1 for f in found if f[0] <= b)}
    return {"seed": seed, "checkpoints": out, "constructions": k, "cpu_s": cpu, "fingerprint": fp}


# --- lanes -----------------------------------------------------------------------------------------------------


def _stream_rows(job: dict, parts: list[dict]) -> list[dict]:
    """CONSTRUCT-1 (stream 0) and, with 8 streams, CONSTRUCT-8 rows at every budget the streams were read at."""
    out = []
    for label, b in budgets(configs.get(job["setting"])).items():
        if b not in parts[0]["checkpoints"]:
            continue
        for cores in sorted({1, len(parts)}):
            best = min((p["checkpoints"][b] | {"stream": p["seed"]} for p in parts[:cores]),
                       key=lambda c: (c["eval_makespan"], c["stream"]))
            out.append({"method": "CONSTRUCT", "cores": cores, "budget": label, "budget_s": b,
                        "wall_s": best["t_found"], "stream_cpu_s": sum(p["cpu_s"] for p in parts[:cores]),
                        "fingerprint": parts[0]["fingerprint"], "routes_base": 0,
                        "streams": cores, "constructions": sum(p["checkpoints"][b]["constructions"]
                                                               for p in parts[:cores])} | best)
    return out


def _run_job(ex, job: dict) -> list[dict]:
    name, i, method, cores, budget = job["setting"], job["instance"], job["method"], job["cores"], job["budget_s"]
    params = job.get("params", {})
    if method in ("ALNS2", "ALNS1"):
        return [ex.submit(run_alns, name, SPLIT, i, cores, budget, method).result()]
    if method == "CPSAT":
        return [ex.submit(run_cpsat, name, SPLIT, i, budget, cores).result()]
    if method == "PCPSAT":
        return [ex.submit(run_pcpsat, name, SPLIT, i, budget, cores, **params).result()]
    if method == "CPFULL":
        return [ex.submit(run_cpfull, name, SPLIT, i, budget, cores, **params).result()]
    if method == "CTAS":
        return [ex.submit(run_ctas, name, SPLIT, i, budget, cores).result()]
    if method == "RL":
        first = 1 if budget <= RL_FIRST_ONE else 4
        parts = [f.result() for f in [ex.submit(cd.run_rl, name, SPLIT, i, p, cores, budget, first=first)
                                      for p in range(cores)]]
        return [cd._with_replay(ex, cd._rl_row(parts)[0], name, SPLIT, i)]
    if method == "CONSTRUCT":
        checkpoints = [b for b in budgets(configs.get(name)).values() if b <= budget + 1e-9]
        parts = [f.result() for f in [ex.submit(run_stream, name, SPLIT, i, k, checkpoints) for k in range(cores)]]
        return _stream_rows(job, parts)
    raise ValueError(method)


def throttled_share(seconds: float = 3.0) -> float:
    """Share of the cgroup's CPU periods throttled during the next ``seconds``."""
    a = runtime.cgroup_cpu_stat()
    time.sleep(seconds)
    b = runtime.cgroup_cpu_stat()
    periods = b.get("nr_periods", 0) - a.get("nr_periods", 0)
    return (b.get("nr_throttled", 0) - a.get("nr_throttled", 0)) / periods if periods > 0 else 0.0


def disturbed(r: dict) -> bool:
    """The container was throttled in more than ``DISTURBED`` of the CPU periods while the job ran."""
    return r.get("nr_periods", 0) > 0 and r.get("nr_throttled", 0) > DISTURBED * r["nr_periods"]


class Lane(cd.Lane):
    quiet = QUIET

    def _unit(self, jobs: list[dict]) -> None:
        waited = 0.0
        while (share := throttled_share(1.0)) > self.quiet:
            if waited % 600 < 30:
                print(f"lane {self.k}: host busy ({share:.0%} of CPU periods throttled), waited {waited:.0f}s",
                      flush=True)
            time.sleep(27.0)
            waited += 30.0

        def one(job) -> bool:
            probe, started = runtime.Probe(self.cpus), time.time()
            try:
                rows = _run_job(self.ex, job)
            except BrokenProcessPool:  # a lane process died (e.g. out of memory): requeued below
                return False
            except Exception:  # noqa: BLE001  (logged; the job has no row and reruns on resume)
                print(f"FAILED {job}\n{traceback.format_exc()}", flush=True)
                return True
            periods = runtime.cgroup_cpu_stat().get("nr_periods", 0) - probe.stat.get("nr_periods", 0)
            for row in rows:
                self.sink(job | row | probe.fields() | {"started": started, "lane": self.k, "nr_periods": periods,
                                                        "probe_s": time.time() - started, "waited_s": waited})
            return True

        with ThreadPoolExecutor(len(jobs)) as tp:
            broken = [job for job, ok in zip(jobs, tp.map(one, jobs)) if not ok]
        if broken:  # a broken pool fails every later job at once: stop this lane, leave the jobs to the others
            self.units.put(broken)
            raise RuntimeError(f"lane {self.k}: a lane process died; {len(broken)} jobs requeued")


def _done(version: str | None = None) -> set[tuple]:
    """Keys (setting, instance, method, cores, budget label[, variant]) of the current, undisturbed rows of this split
    and, given ``version``, of that code version (a campaign resumed on other code recomputes its rows)."""
    out = set()
    for path in OUT.glob("*.jsonl"):
        for r in runtime.read_rows(path):
            if (runtime.is_current(r) and not disturbed(r) and r.get("split") == SPLIT
                    and (version is None or r.get("git") == version)):
                label = "stream" if r["method"] == "CONSTRUCT" else r["budget"]
                cores = r.get("streams_job", r["cores"])
                out.add((r["setting"], r["instance"], r["method"], cores, label) + ((r["variant"],)
                                                                                    if r.get("variant") else ()))
    return out


def _units(settings: list[str], n: int, n_large: int, done: set[tuple], rng: random.Random,
           first: int = 5, max_budget: float = float("inf"), methods: tuple[str, ...] | None = None,
           plan=None) -> list[list[dict]]:
    """8-core jobs alone, 1-core jobs bundled by setting and budget. Instances below ``first`` come first (so an
    early stop leaves complete sets), then longest first, ties shuffled. Jobs above ``max_budget`` seconds are left
    out (restart streams are cut to it), and so are methods not in ``methods``. ``plan(setting, i)`` lists the
    (method, cores, budget label, variant, params) of an instance (default: ``GRID`` without variants)."""
    singles, bundles = [], []
    for name in settings:
        s = configs.get(name)
        b = budgets(s) | {"stream": min(stream_budget(s), max_budget)}
        pending: dict[tuple[bool, str], list[dict]] = {}
        for i in range(min(n if s.n_tasks < 500 else n_large, s.n_instances(SPLIT))):
            points = plan(s, i) if plan is not None else [(*point, "", {}) for point in GRID(s, i)]
            for method, cores, label, variant, params in points:
                key = (name, i, method, cores, label) + ((variant,) if variant else ())
                if key in done or b[label] > max_budget or (methods and method not in methods):
                    continue
                job = {"setting": name, "split": SPLIT, "instance": i, "seed": s.seed(SPLIT, i), "method": method,
                       "cores": cores, "budget": label, "budget_s": b[label], "streams_job": cores}
                if variant:
                    job |= {"variant": variant, "params": params}
                if cores == LANE:
                    singles.append([job])
                else:
                    pending.setdefault((i >= first, label), []).append(job)
        for jobs in pending.values():
            rng.shuffle(jobs)
            bundles += [jobs[k:k + LANE] for k in range(0, len(jobs), LANE)]
    units = singles + bundles
    rng.shuffle(units)
    units.sort(key=lambda u: (min(j["instance"] for j in u) >= first, -max(j["budget_s"] for j in u)))
    return units


def run(args, plan=None) -> None:
    os.environ.update(cd._ONE)
    if SPLIT == "test" and (GRID is not grid_test or plan is not None):
        raise SystemExit("the test split runs the frozen grid only: anytime.py run --split test --grid test")
    version = runtime.require_frozen() if SPLIT == "test" else runtime.require_clean(args.allow_dirty)
    OUT.mkdir(parents=True, exist_ok=True)
    units = _units(args.settings, args.n, args.n_large, _done(version), random.Random(0), max_budget=args.max_budget,
                   methods=args.methods, plan=plan)
    print(f"{len(units)} units, {sum(map(len, units))} jobs, {args.lanes} lanes, split {SPLIT}", flush=True)
    if not units:
        return
    cpus = runtime.parse_cpus(args.cpus) if args.cpus else runtime.choose_cpus(LANE * args.lanes)
    lane_cpus = runtime.blocks(cpus, LANE)[:args.lanes]
    if len(lane_cpus) < args.lanes:
        raise SystemExit(f"{len(cpus)} CPUs for {args.lanes} lanes of {LANE}")
    print(f"lanes pinned to {[runtime.format_cpus(c) for c in lane_cpus]}, code {version}", flush=True)
    work: queue.Queue = queue.Queue()
    for u in units:
        work.put(u)
    lock = threading.Lock()
    t0, count = time.time(), [0]

    def sink(row: dict) -> None:
        row["git"] = version
        with lock:
            with (OUT / f"{row['setting']}.jsonl").open("a") as f:
                f.write(json.dumps(row) + "\n")
            count[0] += 1
            print(f"[{time.time() - t0:6.0f}s] {row['setting']} {row['instance']:2d} {row['budget']:>4s} "
                  f"{row['method'] + (':' + row['variant'] if row.get('variant') else ''):9s} x{row['cores']} "
                  f"{row['makespan']:8.3f} ok={row['success']} "
                  f"wall={row['wall_s']:.1f}s load={row['load1']:.0f} thr={row.get('nr_throttled', 0)} "
                  f"({count[0]} rows)", flush=True)

    Lane.quiet = args.quiet
    lanes = [Lane(k, work, sink, c) for k, c in enumerate(lane_cpus)]
    for lane in lanes:
        lane.start()
    for lane in lanes:
        lane.join()
    errors = [lane.error for lane in lanes if lane.error is not None]
    if errors:
        raise errors[0]


# --- report ----------------------------------------------------------------------------------------------------


def late(r: dict) -> bool:
    """No solution within the budget (CONSTRUCT: ``wall_s`` is when its plan was found)."""
    return r["wall_s"] > 1.1 * r["budget_s"] + 0.25


def score(r: dict) -> float:
    return r["makespan"] if succeeded(r["success"], r["makespan"]) else FAIL


def at_budget(r: dict) -> float | None:
    """A run's score at its budget: its ``score`` if it ended on time. A ``late`` run is left out (None) under the
    ``phase1b`` rule (docs/headroom-2026-09.md); under the ``c1`` rule it scores the best plan its trace had reached by
    the budget, or a failure (200) if it had none (RL and CTAS keep no trace)."""
    if not late(r):
        return score(r)
    if LATE_RULE == "drop":
        return None
    reached = [ms for t, ms in r.get("trace") or () if t <= r["budget_s"]]
    return min(min(reached), FAIL) if reached else FAIL


def load_rows(name: str, keep_disturbed: bool = False) -> dict[tuple[str, int, str], dict[int, dict]]:
    """(method, cores, budget label) -> instance -> row (current fingerprints, not ``disturbed``, planned by ``GRID``
    for that instance, no ``tune`` variants; the last row wins)."""
    s = configs.get(name)
    out: dict[tuple[str, int, str], dict[int, dict]] = {}
    rows = [r for r in runtime.read_rows(OUT / f"{name}.jsonl")
            if runtime.is_current(r) and r.get("split", SPLIT) == SPLIT]
    if keep_disturbed:  # an undisturbed row of a cell wins over any disturbed one, else the last one
        rows.sort(key=lambda r: not disturbed(r))
    for r in rows:
        cell = (r["method"], r["cores"], r["budget"])
        planned = (r["method"], r.get("streams_job", r["cores"]), "stream" if r["method"] == "CONSTRUCT" else r["budget"])
        if (keep_disturbed or not disturbed(r)) and not r.get("variant") and planned in GRID(s, r["instance"]):
            out.setdefault(cell, {})[r["instance"]] = r
    return out


def gap_table(name: str, rows, bks_of: dict[int, float]) -> list[dict]:
    """Mean gap to the BKS (%) per method, cores and budget over the runs scored at their budget (``at_budget``;
    ``on_time`` counts them, ``late`` the runs over budget), and the share of the constructor-to-BKS gap closed (ratio
    of means over instances where both are scored)."""
    s = configs.get(name)
    labels = budgets(s)
    streams = next((c for m, c, _ in GRID(s, 0) if m == "CONSTRUCT"), LANE)  # cores of the restart-stream job
    out = []
    for (method, cores, label), by_i in rows.items():
        key = (method, streams if method == "CONSTRUCT" else cores, "stream" if method == "CONSTRUCT" else label)
        planned = sum(key in GRID(s, i) for i in bks_of)
        ran = {i: r for i, r in by_i.items() if i in bks_of}
        ok = {i: v for i, r in ran.items() if (v := at_budget(r)) is not None}
        if not ok:
            continue
        gaps = np.array([100 * (v - bks_of[i]) / bks_of[i] for i, v in ok.items()])
        con = {i: v for i, r in rows.get(("CONSTRUCT", cores, label), {}).items() if (v := at_budget(r)) is not None}
        both = [i for i in ok if i in con]
        closed = None
        if both and method != "CONSTRUCT":
            c = np.mean([con[i] for i in both])
            m = np.mean([ok[i] for i in both])
            b = np.mean([bks_of[i] for i in both])
            closed = float((c - m) / (c - b)) if c - b > 1e-9 else None
        out.append({"setting": name, "method": method, "cores": cores, "budget": label, "budget_s": labels[label],
                    "planned": planned, "n": len(ran), "on_time": len(ok), "late": sum(map(late, ran.values())),
                    "gap_pct": float(gaps.mean()), "gap_sd": float(gaps.std(ddof=1)) if len(gaps) > 1 else 0.0,
                    "mean_makespan": float(np.mean(list(ok.values()))),
                    "success": float(np.mean([v < FAIL for v in ok.values()])),
                    "wall_s": float(np.mean([r["wall_s"] for r in by_i.values()])), "closed": closed,
                    "cpu_share": float(np.mean([cpu_s(r) / (labels[label] * cores) for r in ran.values()]))})
    return sorted(out, key=lambda g: (METHODS.index(g["method"]), g["cores"], g["budget_s"]))


def ratios(name: str, rows) -> list[dict]:
    """ALNS2 / competitor, paired over instances where both are scored at their budget (``at_budget``)."""
    out = []

    def pair(ours, other, tag):
        a = {i: v for i, r in rows.get(ours, {}).items() if (v := at_budget(r)) is not None}
        b = {i: v for i, r in rows.get(other, {}).items() if (v := at_budget(r)) is not None}
        common = sorted(set(a) & set(b))
        if len(common) < 2:
            return
        res = cd._ratio(np.array([a[i] for i in common]), np.array([b[i] for i in common]))
        out.append({"setting": name, "comparison": tag, "ours": "{}-{}@{}".format(*ours),
                    "other": "{}-{}@{}".format(*other), "n": len(common), **res})

    for label in ("B1", "2B1"):  # Holm families: settings x competitors per row type; v1 is an ablation
        for other in COMPETITORS:  # (competitors without rows in this campaign are skipped)
            pair(("ALNS2", LANE, label), (other, LANE, label), f"C1 8 cores {label}")
        pair(("ALNS2", LANE, label), ("ALNS1", LANE, label), f"ablation v2/v1 8 cores {label}")
    pair(("ALNS2", LANE, "B1"), ("ALNS2", 1, "B1"), "ablation 8 workers / 1 worker B1")
    for label in ("0.5", "1", "2"):
        for other in COMPETITORS:
            pair(("ALNS2", 1, label), (other, LANE, "B1"), f"C2 1 core {label} s vs 8 cores B1")
            pair(("ALNS2", 1, label), (other, 1, label), f"1 core {label} s")
    return out


def _fmt(g: dict | None) -> str:
    if g is None:
        return "      -"
    mark = "" if g["on_time"] == g["n"] and not g.get("late") else "*"
    return f"{g['gap_pct']:6.2f}{mark}"


def report(args) -> None:
    from cbba_sota.stats import holm

    bks_all = bks.load_bks(SPLIT)
    gaps, rats, curves = [], [], []
    for name in args.settings:
        if name not in bks_all:
            continue
        rows = load_rows(name, args.keep_disturbed)
        if not rows:
            continue
        g = gap_table(name, rows, bks_all[name])
        gaps += g
        first = {i: b for i, b in bks_all[name].items() if i < EXTENDED}  # every budget ran on these
        curves += gap_table(name, {k: {i: r for i, r in v.items() if i in first} for k, v in rows.items()}, first)
        rats += ratios(name, rows)
        s = configs.get(name)
        labels = budgets(s)
        every = [r for r in runtime.read_rows(OUT / f"{name}.jsonl") if runtime.is_current(r)]
        used = [r for by_i in rows.values() for r in by_i.values()]
        share = [r["nr_throttled"] / r["nr_periods"] for r in used if r.get("nr_periods")]
        got = [r["cpu_s"] / (r["wall_s"] * r["cores"]) for r in used  # CPU received by one-process-per-core jobs
               if r["method"] in ("ALNS2", "ALNS1", "RL", "PCPSAT") and r.get("cpu_s") and r["wall_s"] > 0]
        late_note = "late and left out" if LATE_RULE == "drop" else "late, scored at the budget by their trace or as 200"
        versions = sorted({r.get("git", "?") for r in used})
        print(f"\n{name}  (val, n = {len(bks_all[name])} BKS; B1 = {labels['B1']:g} s)  mean gap to BKS, %"
              f"  (* = some runs {late_note}); code {', '.join(versions)}")
        kept = "kept" if args.keep_disturbed else "left out"
        print(f"  rows used {len(used)}, disturbed {sum(map(disturbed, every))} ({kept}); throttled share of CPU periods "
              f"in used rows: mean {np.mean(share):.3f}, max {np.max(share):.3f}; CPU received by ALNS/RL jobs: mean "
              f"{np.mean(got):.3f}, 5th percentile {np.percentile(got, 5):.3f}")
        print(f"  {'method':12s}" + "".join(f"{lab:>8s}" for lab in labels))
        by = {(x["method"], x["cores"], x["budget"]): x for x in g}
        for m in METHODS:
            for c in (1, LANE):
                if any((m, c, lab) in by for lab in labels):
                    print(f"  {m + '-' + str(c):12s}" + "".join(f"  {_fmt(by.get((m, c, lab)))}" for lab in labels))
        print("  CPU seconds used / (budget x cores), 8 cores at B1: " + ", ".join(
            f"{m} {np.mean([cpu_s(r) / (r['budget_s'] * LANE) for r in rows[m, LANE, 'B1'].values()]):.2f}"
            for m in METHODS if (m, LANE, "B1") in rows))
        print("  share of the constructor-to-BKS gap closed (same budget and cores):")
        for m in ("ALNS2", "ALNS1", "CPSAT", "PCPSAT", "CPFULL", "RL"):
            for c in (1, LANE):
                vals = [by.get((m, c, lab), {}).get("closed") for lab in labels]
                if any(v is not None for v in vals):
                    print(f"  {m + '-' + str(c):12s}" + "".join(
                        f"{'-' if v is None else f'{100 * v:.0f}%':>8s}" for v in vals))
    for tag in sorted({r["comparison"] for r in rats}):
        sel = [r for r in rats if r["comparison"] == tag]
        for r, p in zip(sel, holm([r["p"] for r in sel])):
            r["p_holm"] = float(p)
    print("\nPaired ratios ALNS2 / other (bootstrap 95% CI; wins-losses; Holm over settings x competitors per row "
          "type)")
    for r in sorted(rats, key=lambda r: (r["comparison"], r["setting"], r["other"])):
        print(f"  {r['comparison']:34s} {r['setting']:17s} vs {r['other']:18s} n={r['n']:2d} {r['ratio']:.3f} "
              f"[{r['lo']:.3f}, {r['hi']:.3f}] {r['wins']}-{r['losses']} p_holm {r['p_holm']:.2g}")
    if args.md:
        args.md.write_text((markdown_c1 if GRID is grid_c1 else markdown)(gaps, rats, bks_all))
        print(f"wrote {args.md}")
    if args.csv_dir:
        import csv

        args.csv_dir.mkdir(parents=True, exist_ok=True)
        for fname, table in (("anytime_gaps_val.csv", gaps), ("anytime_ratios_val.csv", rats)):
            with (args.csv_dir / fname).open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(table[0]))
                w.writeheader()
                w.writerows(table)
        print(f"\nwrote {args.csv_dir}/anytime_gaps_val.csv and anytime_ratios_val.csv")
    if args.png:
        plot(curves, args.png)
    if args.compare:
        compare(args)


def compare(args) -> None:
    """Cells (method, cores, budget) of this campaign that another campaign (``--compare DIR``, e.g. the first host's
    runs/anytime) also ran: paired mean makespans over the instances both ran on time, ratio this / other."""
    global OUT
    here = OUT
    print(f"\nSame cells in {args.compare} (paired over instances on time in both; ratio {here.name} / "
          f"{args.compare.name}). RL at budgets up to {RL_FIRST_ONE:g} s is left out: its first lockstep batch "
          "changed from 4 samples to 1 between the campaigns.")
    for name in args.settings:
        OUT = here
        mine = load_rows(name)
        OUT = args.compare
        other = _load_any(name)
        labels = budgets(configs.get(name))
        for cell in sorted(set(mine) & set(other), key=lambda c: (METHODS.index(c[0]), c[1], labels[c[2]])):
            if cell[0] == "RL" and labels[cell[2]] <= RL_FIRST_ONE:
                continue
            a, b = mine[cell], other[cell]
            code = sorted({r.get("git", "?")[:7] for r in (*a.values(), *b.values())})
            common = sorted(i for i in set(a) & set(b) if not late(a[i]) and not late(b[i]))
            if len(common) < 2:
                continue
            res = cd._ratio(np.array([score(a[i]) for i in common]), np.array([score(b[i]) for i in common]))
            print(f"  {name:17s} {cell[0]}-{cell[1]}@{cell[2]:4s} n={len(common):2d} "
                  f"{np.mean([score(a[i]) for i in common]):8.3f} vs {np.mean([score(b[i]) for i in common]):8.3f}"
                  f"  ratio {res['ratio']:.3f} [{res['lo']:.3f}, {res['hi']:.3f}] {res['wins']}-{res['losses']}  "
                  f"code {'/'.join(code)}")
    OUT = here


def _load_any(name: str) -> dict[tuple[str, int, str], dict[int, dict]]:
    """``load_rows`` under whichever grid planned the rows of ``OUT``."""
    global GRID
    keep, out = GRID, {}
    for g in GRIDS.values():
        GRID = g
        for cell, by_i in load_rows(name).items():
            out.setdefault(cell, {}).update(by_i)
    GRID = keep
    return out


def cpu_s(r: dict) -> float:
    """CPU seconds of all processes and threads of a row's job. CONSTRUCT rows are read off restart streams that run
    on to the largest budget: their streams' CPU pro rata up to the row's budget."""
    if "stream_cpu_s" in r:
        return r["stream_cpu_s"] * min(1.0, r["budget_s"] / stream_budget(configs.get(r["setting"])))
    return r.get("cpu_s", 0.0)


def markdown_c1(gaps: list[dict], rats: list[dict], bks_all: dict) -> str:
    """Compact tables of the ``c1`` campaign: gap to BKS (%) and paired ratios ALNS2 / competitor, 8 cores at B1 and
    2B1, and ALNS2 on 1 core at 0.5 / 1 / 2 s against every competitor on 8 cores at B1."""
    g = {(x["setting"], x["method"], x["cores"], x["budget"]): x for x in gaps}
    r = {(x["setting"], x["ours"], x["other"]): x for x in rats}
    names = list(dict.fromkeys(x["setting"] for x in gaps))
    methods = [m for m in METHODS if any(x["method"] == m for x in gaps)]
    others = [m for m in COMPETITORS if m in methods]

    def gap(name, m, c, b):
        x = g.get((name, m, c, b))
        if x is None:
            return "-"
        return (f"{x['gap_pct']:.2f}" + ("" if x["on_time"] == x["planned"] else f" ({x['on_time']}/{x['planned']})")
                + ("" if x["success"] == 1 else f" [solved {100 * x['success']:.0f}%]")
                + (f" [late {x['late']}]" if x.get("late") else ""))

    def ratio(name, ours, other):
        x = r.get((name, ours, other))
        if x is None:
            return "-"
        return f"{x['ratio']:.3f} [{x['lo']:.3f}, {x['hi']:.3f}] {x['wins']}-{x['losses']}"

    def cpu(name, m):
        x = g.get((name, m, LANE, "B1"))
        return "-" if x is None else f"{x['cpu_share']:.2f}"

    out = ["**C1, 8 cores at B1.** Mean gap to the best of our own runs (%).", "",
           "| Setting | n | " + " | ".join(methods) + " |", "|" + " --- |" * (len(methods) + 2)]
    for name in names:
        out.append(f"| {name} | {len(bks_all.get(name, {}))} | "
                   + " | ".join(gap(name, m, LANE, "B1") for m in methods) + " |")
    out += ["", "**CPU used, 8 cores at B1.** CPU seconds of all processes and threads / (B1 x 8), mean over runs.",
            "", "| Setting | " + " | ".join(methods) + " |", "|" + " --- |" * (len(methods) + 1)]
    for name in names:
        out.append(f"| {name} | " + " | ".join(cpu(name, m) for m in methods) + " |")
    for label in ("B1", "2B1"):
        cols = [m for m in others if any(k[0] == name and k[2] == f"{m}-8@{label}" for k in r for name in names)]
        if not cols:
            continue
        out += ["", f"**C1, 8 cores at {label}.** Paired ratio ALNS2 / competitor (bootstrap 95% CI, wins-losses).",
                "", "| Setting | " + " | ".join(f"ALNS2/{m}" for m in cols) + " |", "|" + " --- |" * (len(cols) + 1)]
        for name in names:
            out.append(f"| {name} | " + " | ".join(ratio(name, f"ALNS2-8@{label}", f"{m}-8@{label}") for m in cols)
                       + " |")
    for b in ("0.5", "1", "2"):
        title = (f"**C2, ALNS2 on 1 core at {b} s vs each competitor on 8 cores at B1.** Gap of ALNS2-1 (%) and "
                 "paired ratio.")
        out += ["", title, "", "| Setting | ALNS2-1 gap | " + " | ".join(f"vs {m}-8@B1" for m in others) + " |",
                "|" + " --- |" * (len(others) + 2)]
        for name in names:
            out.append(f"| {name} | {gap(name, 'ALNS2', 1, b)} | "
                       + " | ".join(ratio(name, f"ALNS2-1@{b}", f"{m}-8@B1") for m in others) + " |")
    return "\n".join(out) + "\n"


def markdown(gaps: list[dict], rats: list[dict], bks_all: dict) -> str:
    """Compact tables for docs/headroom-2026-09.md (grid ``phase1b``): gap to BKS (%) and paired ratios, 8 cores at B1
    and 1 core at low budgets."""
    methods, competitors = ("ALNS2", "ALNS1", "CPSAT", "RL", "CONSTRUCT"), ("CPSAT", "RL", "CONSTRUCT")
    g = {(x["setting"], x["method"], x["cores"], x["budget"]): x for x in gaps}
    r = {(x["setting"], x["ours"], x["other"]): x for x in rats}
    names = list(dict.fromkeys(x["setting"] for x in gaps))

    def gap(name, m, c, b):
        x = g.get((name, m, c, b))
        if x is None:
            return "-"
        return f"{x['gap_pct']:.2f}" + ("" if x["on_time"] == x["planned"] else f" ({x['on_time']}/{x['planned']})")

    def closed(name, m, c, b):
        x = g.get((name, m, c, b))
        return "-" if x is None or x["closed"] is None else f"{100 * x['closed']:.0f}%"

    def ratio(name, ours, other):
        x = r.get((name, ours, other))
        if x is None:
            return "-"
        return f"{x['ratio']:.3f} [{x['lo']:.3f}, {x['hi']:.3f}] {x['wins']}-{x['losses']}"

    title = ("**C1 on val, 8 cores at B1.** Mean gap to BKS (%), share of the constructor-to-BKS gap closed (vs "
             "CONSTRUCT-8 at B1), and paired ratios ALNS2 / competitor (bootstrap 95% CI, wins-losses).")
    head = ("| Setting | n | ALNS2 | ALNS1 | CPSAT | RL | CONSTRUCT | ALNS2 closes | CPSAT closes | ALNS2/CPSAT | "
            "ALNS2/RL | ALNS2/CONSTRUCT |")
    out = [title, "", head, "|" + " --- |" * 12]
    for name in names:
        n = len(bks_all.get(name, {}))
        out.append(f"| {name} | {n} | " + " | ".join(gap(name, m, LANE, "B1") for m in methods) + " | "
                   + " | ".join(closed(name, m, LANE, "B1") for m in ("ALNS2", "CPSAT")) + " | "
                   + " | ".join(ratio(name, "ALNS2-8@B1", f"{m}-8@B1") for m in competitors) + " |")
    out += ["", "**C1 at 2B1, 8 cores.** Gap to BKS (%) and ALNS2 / CPSAT.", "",
            "| Setting | ALNS2 | CPSAT | CONSTRUCT | ALNS2/CPSAT |", "|" + " --- |" * 5]
    for name in names:
        out.append(f"| {name} | " + " | ".join(gap(name, m, LANE, "2B1") for m in ("ALNS2", "CPSAT", "CONSTRUCT"))
                   + f" | {ratio(name, 'ALNS2-8@2B1', 'CPSAT-8@2B1')} |")
    title = ("**C2, 1 core.** Mean gap to BKS (%) at 0.5 / 1 / 2 s and B1, and ALNS2 on 1 core at 1 s against each "
             "competitor on 8 cores at B1. In every table `(k/n)`: k of the n planned runs were on time and not "
             "disturbed; late runs have no solution within the budget.")
    head = ("| Setting | ALNS2 0.5/1/2 s | ALNS2 B1 | ALNS1 1 s | CPSAT 1 s | RL 1 s | CONSTRUCT 1 s | "
            "ALNS2-1@1s / CPSAT-8@B1 | ALNS2-1@1s / RL-8@B1 | ALNS2-1@1s / CONSTRUCT-8@B1 |")
    out += ["", title, "", head, "|" + " --- |" * 10]
    for name in names:
        out.append(f"| {name} | " + " / ".join(gap(name, "ALNS2", 1, b) for b in ("0.5", "1", "2")) + " | "
                   + gap(name, "ALNS2", 1, "B1") + " | "
                   + " | ".join(gap(name, m, 1, "1") for m in ("ALNS1", "CPSAT", "RL", "CONSTRUCT")) + " | "
                   + " | ".join(ratio(name, "ALNS2-1@1", f"{m}-8@B1") for m in competitors) + " |")
    out += ["", "**Share of the constructor-to-BKS gap closed, 1 core** (vs CONSTRUCT-1 at the same budget).", "",
            "| Setting | ALNS2 0.5 s | ALNS2 1 s | ALNS2 2 s | ALNS2 B1 | CPSAT 1 s | CPSAT B1 |", "|" + " --- |" * 7]
    for name in names:
        out.append(f"| {name} | " + " | ".join(closed(name, "ALNS2", 1, b) for b in ("0.5", "1", "2", "B1")) + " | "
                   + " | ".join(closed(name, "CPSAT", 1, b) for b in ("1", "B1")) + " |")
    return "\n".join(out) + "\n"


def plot(gaps: list[dict], png: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(dict.fromkeys(g["setting"] for g in gaps))
    cols = 4
    fig, axes = plt.subplots((len(names) + cols - 1) // cols, cols, figsize=(4 * cols, 3.2 * ((len(names) + 3) // 4)),
                             squeeze=False)
    colors = dict(zip(METHODS, ("C0", "C1", "C2", "C4", "C5", "C6", "C3", "C7")))
    for ax, name in zip(axes.flat, names):
        for m in METHODS:
            for c, ls in ((1, "-"), (LANE, "--")):
                pts = sorted((g["budget_s"], g["gap_pct"]) for g in gaps  # budgets where every run had a plan
                             if g["setting"] == name and g["method"] == m and g["cores"] == c
                             and g["on_time"] == g["n"] and g["success"] == 1)
                if pts:
                    ax.plot(*zip(*pts), ls, marker="o", ms=3, color=colors[m], label=f"{m}-{c}")
        ax.set_xscale("log")
        ax.set_yscale("symlog", linthresh=1)
        ax.set_ylim(bottom=0)
        ax.set_title(f"{name} (val 0-{EXTENDED - 1})", fontsize=9)
        ax.set_xlabel("budget (s)", fontsize=8)
        ax.set_ylabel("gap to BKS (%)", fontsize=8)
        ax.grid(alpha=0.3)
    for ax in list(axes.flat)[len(names):]:
        ax.axis("off")
    axes.flat[0].legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(png, dpi=120)
    print(f"wrote {png}")


# --- dev tuning of competitor variants ------------------------------------------------------------------------


VARIANT_PARAMS = {"PCPSAT": ("sub_time",), "CPFULL": ("knn",)}  # what run_pcpsat / run_cpfull take


def parse_variant(spec: str) -> tuple[str, str, dict]:
    """``"PCPSAT:sub_time=0.5"`` -> ``("PCPSAT", "sub_time=0.5", {"sub_time": 0.5})``."""
    method, _, rest = spec.partition(":")
    if method not in VARIANT_PARAMS or not rest:
        raise SystemExit(f"bad variant {spec!r}: expected PCPSAT:sub_time=S or CPFULL:knn=K")
    params = {k: json.loads(v) for k, v in (item.split("=", 1) for item in rest.split(","))}
    if unknown := set(params) - set(VARIANT_PARAMS[method]):
        raise SystemExit(f"bad variant {spec!r}: {method} takes {', '.join(VARIANT_PARAMS[method])}, not "
                         f"{', '.join(sorted(unknown))}")
    return method, rest, params


def tune(args) -> None:
    variants = [parse_variant(v) for v in args.variants]

    def plan(s, i):
        return [(m, LANE, "B1", v, params) for m, v, params in variants] + [(m, LANE, "B1", "", {})
                                                                           for m in args.refs]

    args.methods = None
    run(args, plan)


def tune_report(args) -> None:
    """Per setting: every variant's mean makespan over the instances all of its method's variants ran on time, its
    paired ratio to the best of them, and the CPU share it used; reference methods for scale."""
    for path in sorted(OUT.glob("*.jsonl")):
        name = path.stem
        cells: dict[tuple[str, str], dict[int, dict]] = {}
        for r in runtime.read_rows(path):
            if runtime.is_current(r) and not disturbed(r) and r["budget"] == "B1" and r["cores"] == LANE:
                cells.setdefault((r["method"], r.get("variant", "")), {})[r["instance"]] = r
        print(f"\n{name}  (dev, B1 = {budgets(configs.get(name))['B1']:g} s, 8 cores)")
        for method in dict.fromkeys(m for m, _ in cells):
            mine = {v: by for (m, v), by in cells.items() if m == method}
            common = sorted(set.intersection(*({i for i, r in by.items() if not late(r)} for by in mine.values())))
            if not common:
                continue
            means = {v: np.mean([score(by[i]) for i in common]) for v, by in mine.items()}
            best = min(means, key=means.get)
            for v, by in sorted(mine.items(), key=lambda kv: means[kv[0]]):
                res = cd._ratio(np.array([score(by[i]) for i in common]), np.array([score(mine[best][i])
                                                                                   for i in common]))
                cpu = np.mean([cpu_s(r) / (r["budget_s"] * LANE) for r in by.values()])
                n_late = sum(late(r) for r in by.values())
                print(f"  {method + (':' + v if v else ''):28s} n={len(common):2d} mean {means[v]:8.3f}  / best "
                      f"{res['ratio']:.3f} [{res['lo']:.3f}, {res['hi']:.3f}] {res['wins']}-{res['losses']}  "
                      f"CPU share {cpu:.2f}  late {n_late}" + ("  <- best" if v == best else ""))


def main() -> None:
    global OUT, SPLIT, GRID, LATE_RULE
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["run", "report", "tune", "tune-report"])
    ap.add_argument("--grid", choices=list(GRIDS), default="phase1b", help="run/report: the campaign's grid")
    ap.add_argument("--split", choices=["val", "dev", "test"], default=None,
                    help="default val (tune: dev); test only with --grid test under the frozen pre-registration")
    ap.add_argument("--methods", nargs="+", default=None, choices=METHODS, help="run: only these methods")
    ap.add_argument("--variants", nargs="+", default=[], help="tune: METHOD:param=value[,param=value]")
    ap.add_argument("--refs", nargs="*", default=["ALNS2", "CPSAT"], choices=METHODS, help="tune: reference methods")
    ap.add_argument("--settings", nargs="+", default=list(bks.SETTINGS))
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--n-large", type=int, default=3, help="instances of the 500-task settings")
    ap.add_argument("--lanes", type=int, default=4, help=f"lanes of {LANE} processes")
    ap.add_argument("--cpus", default=None)
    ap.add_argument("--max-budget", type=float, default=float("inf"), help="skip jobs with longer budgets")
    ap.add_argument("--quiet", type=float, default=QUIET, help="max throttled share of CPU periods to start a unit")
    ap.add_argument("--csv-dir", type=Path, default=None)
    ap.add_argument("--png", type=Path, default=None)
    ap.add_argument("--md", type=Path, default=None, help="report: compact markdown tables")
    ap.add_argument("--keep-disturbed", action="store_true", help="report: sensitivity run keeping disturbed rows")
    ap.add_argument("--compare", type=Path, default=None, help="report: rows of another campaign to compare cells")
    ap.add_argument("--out", type=Path, default=None, help="rows (default runs/anytime; tune: runs/tune_dev)")
    ap.add_argument("--allow-dirty", action="store_true", help="run on uncommitted source (smoke tests only)")
    args = ap.parse_args()
    tuning = args.command.startswith("tune")
    if tuning and args.split == "test":
        raise SystemExit("tuning uses dev only")
    OUT = args.out or (RUNS_DIR / "tune_dev" if tuning else OUT)
    SPLIT = args.split or ("dev" if tuning else "val")
    GRID = GRIDS[args.grid]
    LATE_RULE = LATE_RULES[args.grid]
    {"run": run, "report": report, "tune": tune, "tune-report": tune_report}[args.command](args)


if __name__ == "__main__":
    main()
