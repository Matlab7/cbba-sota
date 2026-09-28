"""Anytime curves on the validation split: quality vs wall-clock budget on 1 and 8 cores, and the headroom report.

Usage: anytime.py run [--settings NAME ...] [--n 10] [--lanes 4] [--cpus LIST]
       anytime.py report [--settings NAME ...] [--csv-dir DIR] [--png FILE]

Methods (one row per method, cores and budget; budgets 0.5, 1, 2, 5, 10 s, B1 and 2B1 = 2 x B1, where B1 is the
paper's RL(s.10) time of the setting):
- ALNS2: ALNS v2 (the default ``ALNSConfig``), 1 worker or 8 forked workers, seed 0.
- ALNS1: ALNS v1 (``ALNSConfig.v1()``), 1 worker at every budget, 8 workers at B1.
- CPSAT: CP-SAT LNS with 1 or 8 threads from the ``greedy.construct`` hint built in min(10% of the budget, 3 s),
  sub-solves of ``bks.sub_time(setting)`` seconds (the better variant on dev).
- RL: RL(s.N) with the released policy on 1 or 8 single-threaded CPU processes, each sampling lockstep batches
  until the budget is used (``rl.sample_until``), scored by the best rollout's own env run.
- CONSTRUCT: the regret-insertion constructor with randomized restarts. Restarts are independent of the budget,
  so one job runs 8 restart streams (seeds 0-7, the sequence of ``greedy.construct``) on 8 cores up to the largest
  budget and reads every budget off them: CONSTRUCT-1 at b is stream 0's best plan finished by b, CONSTRUCT-8 the
  best over the 8 streams (the first plan if none finished by b; ``t_found`` says when it did).
Grid (``grid``; cut to the 6-hour compute window on a shared host, see docs/headroom-2026-09.md): on instances 0-4
every method at every budget on 1 core, and ALNS2, CPSAT and RL at 0.5-10 s on 8 cores; on every instance ALNS2,
ALNS1, CPSAT and RL on 1 core at 0.5, 1, 2 s and B1 and on 8 cores at B1, ALNS2 on 8 cores at 2B1, CPSAT on 8 cores
at 2B1 up to 50 tasks, and the 8 restart streams (up to 2B1, 200 tasks up to B1). 20-task settings run 8 cores at B1
only. 500 tasks: ALNS2 and ALNS1 on 1 core up to 10 s, ALNS2 on 1 core at B1, ALNS2, CPSAT and RL on 8 cores at B1,
and one restart stream up to B1.

Execution reuses scripts/compare_dev.py: ``--lanes`` lanes of 8 warm single-threaded processes pinned to their own
8 CPUs; a lane runs one 8-core job or a bundle of up to 8 one-core jobs of the same budget. Budgets start after
the instance and its travel matrices are loaded. Every makespan is the env replay of the plan (RL: the rollout's
own env run). Rows go to runs/anytime/<setting>.jsonl with routes, fingerprint, code version, CPU affinity, load
and throttling counters (resumable per setting, instance, method, cores and budget). The container shares a CPU
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
METHODS = ("ALNS2", "ALNS1", "CPSAT", "RL", "CONSTRUCT")
COMPETITORS = ("CPSAT", "RL", "CONSTRUCT")
EXTENDED = 5  # instances below this get the full budget grid (see ``grid``)
FAIL = cd.FAIL
QUIET = 0.4  # a lane starts a unit only when at most this share of the cgroup's CPU periods was throttled
DISTURBED = 0.4  # rows whose job saw a larger throttled share are re-run on resume and left out of reports


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
                        "fingerprint": parts[0]["fingerprint"],
                        "streams": cores, "constructions": sum(p["checkpoints"][b]["constructions"]
                                                               for p in parts[:cores])} | best)
    return out


def _run_job(ex, job: dict) -> list[dict]:
    name, i, method, cores, budget = job["setting"], job["instance"], job["method"], job["cores"], job["budget_s"]
    if method in ("ALNS2", "ALNS1"):
        return [ex.submit(run_alns, name, SPLIT, i, cores, budget, method).result()]
    if method == "CPSAT":
        return [ex.submit(run_cpsat, name, SPLIT, i, budget, cores).result()]
    if method == "RL":
        parts = [f.result() for f in [ex.submit(cd.run_rl, name, SPLIT, i, p, cores, budget) for p in range(cores)]]
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

        def one(job):
            probe, started = runtime.Probe(self.cpus), time.time()
            try:
                rows = _run_job(self.ex, job)
            except Exception:  # noqa: BLE001  (logged; the job has no row and reruns on resume)
                print(f"FAILED {job}\n{traceback.format_exc()}", flush=True)
                return
            periods = runtime.cgroup_cpu_stat().get("nr_periods", 0) - probe.stat.get("nr_periods", 0)
            for row in rows:
                self.sink(job | row | probe.fields() | {"started": started, "lane": self.k, "nr_periods": periods,
                                                        "probe_s": time.time() - started, "waited_s": waited})

        with ThreadPoolExecutor(len(jobs)) as tp:
            list(tp.map(one, jobs))


def _done() -> set[tuple]:
    out = set()
    for path in OUT.glob("*.jsonl"):
        for r in runtime.read_rows(path):
            if runtime.is_current(r) and not disturbed(r):
                label = "stream" if r["method"] == "CONSTRUCT" else r["budget"]
                cores = r.get("streams_job", r["cores"])
                out.add((r["setting"], r["instance"], r["method"], cores, label))
    return out


def _units(settings: list[str], n: int, n_large: int, done: set[tuple], rng: random.Random,
           first: int = 5, max_budget: float = float("inf")) -> list[list[dict]]:
    """8-core jobs alone, 1-core jobs bundled by setting and budget. Instances below ``first`` come first (so an
    early stop leaves complete sets), then longest first, ties shuffled. Jobs above ``max_budget`` seconds are left
    out (restart streams are cut to it)."""
    singles, bundles = [], []
    for name in settings:
        s = configs.get(name)
        b = budgets(s) | {"stream": min(stream_budget(s), max_budget)}
        pending: dict[tuple[bool, str], list[dict]] = {}
        for i in range(min(n if s.n_tasks < 500 else n_large, s.n_instances(SPLIT))):
            for method, cores, label in grid(s, i):
                if (name, i, method, cores, label) in done or b[label] > max_budget:
                    continue
                job = {"setting": name, "split": SPLIT, "instance": i, "seed": s.seed(SPLIT, i), "method": method,
                       "cores": cores, "budget": label, "budget_s": b[label], "streams_job": cores}
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


def run(args) -> None:
    os.environ.update(cd._ONE)
    OUT.mkdir(parents=True, exist_ok=True)
    units = _units(args.settings, args.n, args.n_large, _done(), random.Random(0), max_budget=args.max_budget)
    print(f"{len(units)} units, {sum(map(len, units))} jobs, {args.lanes} lanes", flush=True)
    if not units:
        return
    cpus = runtime.parse_cpus(args.cpus) if args.cpus else runtime.choose_cpus(LANE * args.lanes)
    lane_cpus = runtime.blocks(cpus, LANE)[:args.lanes]
    if len(lane_cpus) < args.lanes:
        raise SystemExit(f"{len(cpus)} CPUs for {args.lanes} lanes of {LANE}")
    version = runtime.code_version()
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
                  f"{row['method']:9s} x{row['cores']} {row['makespan']:8.3f} ok={row['success']} "
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


def load_rows(name: str, keep_disturbed: bool = False) -> dict[tuple[str, int, str], dict[int, dict]]:
    """(method, cores, budget label) -> instance -> row (current fingerprints, not ``disturbed``, planned by ``grid``
    for that instance; the last row wins)."""
    s = configs.get(name)
    out: dict[tuple[str, int, str], dict[int, dict]] = {}
    rows = [r for r in runtime.read_rows(OUT / f"{name}.jsonl") if runtime.is_current(r)]
    if keep_disturbed:  # an undisturbed row of a cell wins over any disturbed one, else the last one
        rows.sort(key=lambda r: not disturbed(r))
    for r in rows:
        cell = (r["method"], r["cores"], r["budget"])
        planned = (r["method"], r.get("streams_job", r["cores"]), "stream" if r["method"] == "CONSTRUCT" else r["budget"])
        if (keep_disturbed or not disturbed(r)) and planned in grid(s, r["instance"]):
            out.setdefault(cell, {})[r["instance"]] = r
    return out


def gap_table(name: str, rows, bks_of: dict[int, float]) -> list[dict]:
    """Mean gap to the BKS (%) per method, cores and budget over on-time runs, and the share of the
    constructor-to-BKS gap closed (ratio of means over instances where both ran on time)."""
    s = configs.get(name)
    labels = budgets(s)
    out = []
    for (method, cores, label), by_i in rows.items():
        key = (method, LANE if method == "CONSTRUCT" and s.n_tasks < 500 else cores,
               "stream" if method == "CONSTRUCT" else label)
        planned = sum(key in grid(s, i) for i in bks_of)
        ran = {i: r for i, r in by_i.items() if i in bks_of}
        ok = {i: r for i, r in ran.items() if not late(r)}
        if not ok:
            continue
        gaps = np.array([100 * (score(r) - bks_of[i]) / bks_of[i] for i, r in ok.items()])
        con = rows.get(("CONSTRUCT", cores, label), {})
        both = [i for i in ok if i in con]
        closed = None
        if both and method != "CONSTRUCT":
            c = np.mean([score(con[i]) for i in both])
            m = np.mean([score(ok[i]) for i in both])
            b = np.mean([bks_of[i] for i in both])
            closed = float((c - m) / (c - b)) if c - b > 1e-9 else None
        out.append({"setting": name, "method": method, "cores": cores, "budget": label, "budget_s": labels[label],
                    "planned": planned, "n": len(ran), "on_time": len(ok), "gap_pct": float(gaps.mean()),
                    "gap_sd": float(gaps.std(ddof=1)) if len(gaps) > 1 else 0.0,
                    "mean_makespan": float(np.mean([score(r) for r in ok.values()])),
                    "success": float(np.mean([succeeded(r["success"], r["makespan"]) for r in ok.values()])),
                    "wall_s": float(np.mean([r["wall_s"] for r in by_i.values()])), "closed": closed})
    return sorted(out, key=lambda g: (METHODS.index(g["method"]), g["cores"], g["budget_s"]))


def ratios(name: str, rows) -> list[dict]:
    """ALNS2 / competitor, paired over instances where both ran on time."""
    out = []

    def pair(ours, other, tag):
        a, b = rows.get(ours, {}), rows.get(other, {})
        common = sorted(i for i in set(a) & set(b) if not late(a[i]) and not late(b[i]))
        if len(common) < 2:
            return
        res = cd._ratio(np.array([score(a[i]) for i in common]), np.array([score(b[i]) for i in common]))
        out.append({"setting": name, "comparison": tag, "ours": "{}-{}@{}".format(*ours),
                    "other": "{}-{}@{}".format(*other), "n": len(common), **res})

    for label in ("B1", "2B1"):  # Holm families: settings x competitors per row type; v1 is an ablation
        for other in COMPETITORS:
            pair(("ALNS2", LANE, label), (other, LANE, label), f"C1 8 cores {label}")
        pair(("ALNS2", LANE, label), ("ALNS1", LANE, label), f"ablation v2/v1 8 cores {label}")
    for label in ("0.5", "1", "2"):
        for other in COMPETITORS:
            pair(("ALNS2", 1, label), (other, LANE, "B1"), f"C2 1 core {label} s vs 8 cores B1")
            pair(("ALNS2", 1, label), (other, 1, label), f"1 core {label} s")
    return out


def _fmt(g: dict | None) -> str:
    if g is None:
        return "      -"
    mark = "" if g["on_time"] == g["n"] else "*"
    return f"{g['gap_pct']:6.2f}{mark}"


def report(args) -> None:
    from cbba_sota.stats import holm

    bks_all = bks.load_bks()
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
               if r["method"] in ("ALNS2", "ALNS1", "RL") and r.get("cpu_s") and r["wall_s"] > 0]
        print(f"\n{name}  (val, n = {len(bks_all[name])} BKS; B1 = {labels['B1']:g} s)  mean gap to BKS, %"
              "  (* = some runs late and left out)")
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
        print("  share of the constructor-to-BKS gap closed (same budget and cores):")
        for m in ("ALNS2", "ALNS1", "CPSAT", "RL"):
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
        args.md.write_text(markdown(gaps, rats, bks_all))
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


def markdown(gaps: list[dict], rats: list[dict], bks_all: dict) -> str:
    """Compact tables for docs/headroom-2026-09.md: gap to BKS (%) and paired ratios, 8 cores at B1 and 1 core at
    low budgets."""
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
        out.append(f"| {name} | {n} | " + " | ".join(gap(name, m, LANE, "B1") for m in METHODS) + " | "
                   + " | ".join(closed(name, m, LANE, "B1") for m in ("ALNS2", "CPSAT")) + " | "
                   + " | ".join(ratio(name, "ALNS2-8@B1", f"{m}-8@B1") for m in COMPETITORS) + " |")
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
                   + " | ".join(ratio(name, "ALNS2-1@1", f"{m}-8@B1") for m in COMPETITORS) + " |")
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
    colors = dict(zip(METHODS, ("C0", "C1", "C2", "C3", "C7")))
    for ax, name in zip(axes.flat, names):
        for m in METHODS:
            for c, ls in ((1, "-"), (LANE, "--")):
                pts = sorted((g["budget_s"], g["gap_pct"]) for g in gaps
                             if g["setting"] == name and g["method"] == m and g["cores"] == c
                             and g["on_time"] == g["n"])
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


def main() -> None:
    global OUT
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["run", "report"])
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
    ap.add_argument("--out", type=Path, default=OUT, help="rows (smoke tests)")
    args = ap.parse_args()
    OUT = args.out
    (run if args.command == "run" else report)(args)


if __name__ == "__main__":
    main()
