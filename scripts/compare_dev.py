"""Matched wall-clock comparison on the dev split (dry run of the Phase 1 kill gate).

Usage: compare_dev.py run [--settings NAME ...] [--instances 20] [--lanes 8] [--out runs/compare_dev]
       compare_dev.py report [--out runs/compare_dev] [--csv FILE]

Budgets per setting: B1 = the paper's RL(s.10) time for that setting, B2 = 2 x B1. Methods at each budget:
- ALNS-1, ALNS-8: coalition ALNS with 1 or 8 forked workers (seed 0).
- CPSAT-8: CP-SAT with 8 workers, hinted: ``construct`` for min(10% of B, 3 s), then LNS in CP (``solve_lns``), the
  best CP-SAT variant on dev in the baseline campaign (runs/baselines).
- RL-1, RL-8: RL(s.N) with the released policy in 1 or 8 single-threaded processes; each process samples lockstep
  batches until the budget is used (``rl.sample_until``), so N fills the budget on those cores.
- RL64+ALNS-8: RL(s.64) on 8 processes (8 lockstep samples each), then ALNS-8 from the best rollout's coalitions
  (``rl.to_plan``) for the rest of the budget; result = better of the two.
- greedy: the paper's fixed nearest-task greedy (``greedy_nearest``), the env's own run, no budget.

Makespans are env ground truth: plans are replayed with pre_set_route + execute_by_route, RL and greedy are scored
by the env's own episodes (their routes are also replayed, ``replay_makespan``, a diagnostic only). Success means all
tasks finished and makespan < 200. Every budget starts after the instance is loaded and its travel matrices are
built, in the process that runs it (RL included); the RL policy, numba kernels and imports are loaded once per
process beforehand.

Execution: ``--lanes`` lanes of 8 persistent single-threaded processes each (spawned, policy and kernels warm), each
lane pinned to its own 8 CPUs (``--cpus``, default the least busy physical cores). A lane runs one 8-core job at a
time (RL uses its 8 processes, ALNS-8 forks 8 workers from one of them, CP-SAT runs 8 threads in one) or a bundle
of up to 8 one-core jobs, so at most 8 x lanes processes compute at once. Rows go to ``<out>/<setting>.jsonl``
with the instance ``fingerprint``, the ``git`` code version, the lane's CPU ``affinity``, load average at start and
end and the change of the cgroup's throttling counters (resumable: rows whose fingerprint matches the current
instance are skipped).
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import queue
import random
import resource
import threading
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path

import numpy as np

from cbba_sota.bench import runtime
from cbba_sota.bench.configs import MAX_TIME
from cbba_sota.hetero.replay import succeeded

SETTINGS = ("MA-AT-25-5-50", "SA-AT-50-5-50", "MA-AT-50-5-50", "SA-BT-50-5-50", "MA-AT-50-5-200")
LANE = 8
CORES = {"ALNS-1": 1, "ALNS-8": 8, "CPSAT-8": 8, "RL-1": 1, "RL-8": 8, "RL64+ALNS-8": 8, "greedy": 1}
OURS = ("ALNS-1", "ALNS-8")
HYBRID = "RL64+ALNS-8"
GATE = 0.92
FAIL = MAX_TIME  # a failed instance enters the ratio at the time cap
_ONE = {k: "1" for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS")}


def budgets(setting, scale: float = 1.0) -> dict[str, float]:
    b1 = scale * setting.paper["RL(s.10)"].time_s
    return {"B1": b1, "B2": 2 * b1}


# --- lane processes --------------------------------------------------------------------------------------------

_net = None


def _init(cpus: list[int]) -> None:
    global _net
    runtime.pin(cpus)
    import importlib

    import torch

    from cbba_sota.hetero import Instance, evaluate
    from cbba_sota.solvers import alns, greedy, rl

    importlib.import_module("cbba_sota.solvers.cpsat")  # import cost outside the budgets
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    _net = rl.load_policy("cpu")
    alns._warmup()
    rng = np.random.default_rng(0)
    toy = Instance(req=np.eye(2)[[0, 1, 0, 1]], loc=rng.random((4, 2)), dur=rng.random(4), ab=np.eye(2)[[0, 0, 1, 1]],
                   depot=np.zeros((4, 2)), species=[0, 0, 1, 1])
    evaluate(toy, greedy.construct(toy, restarts=2))


def _pid() -> tuple[int, str]:
    time.sleep(0.5)
    return os.getpid(), runtime.affinity()


@lru_cache(maxsize=2)
def _load(name: str, i: int):
    """(instance with its travel matrices, env pickle bytes, fingerprint), cached per process; loading is outside
    every budget."""
    from cbba_sota.bench import configs
    from cbba_sota.hetero import Instance

    path = configs.get(name).instance_path("dev", i)
    inst = Instance.from_pickle(path)
    inst.tt, inst.da  # noqa: B018
    return inst, path.read_bytes(), runtime.fingerprint(inst)


def _children_cpu() -> float:
    ru = resource.getrusage(resource.RUSAGE_CHILDREN)
    return ru.ru_utime + ru.ru_stime


def _plan_row(inst, plan, wall: float, cpu: float) -> dict:
    from cbba_sota.hetero import evaluate, replay

    rep = replay(inst, plan)
    return {"makespan": rep["makespan"], "success": rep["success"], "env_finished": rep["env_finished"],
            "eval_makespan": evaluate(inst, plan).makespan, "skipped": rep["skipped"], "wall_s": wall, "cpu_s": cpu,
            "routes": plan.routes()}


def run_alns(name: str, i: int, workers: int, deadline: float | None = None, budget: float | None = None,
             init=None) -> dict:
    """ALNS until ``deadline`` (monotonic) or for ``budget`` seconds; ``init``: warm-start plan. A deadline is
    moved by the time this process spends loading the instance (outside every budget)."""
    from cbba_sota.solvers import alns

    loading = time.monotonic()
    inst, _, fp = _load(name, i)
    c0, k0 = time.process_time(), _children_cpu()
    t0 = time.monotonic()
    limit = budget if deadline is None else deadline - loading  # = time left + loading time
    plan, st = alns.solve(inst, limit, seed=0, n_workers=workers, init_plan=init)
    end = time.monotonic()
    wall = end - t0
    cpu = time.process_time() - c0 + _children_cpu() - k0
    return _plan_row(inst, plan, wall, cpu) | {"iterations": st.iterations, "init_makespan": st.init_makespan,
                                               "last_improvement_s": st.trace[-1][0], "end": end,
                                               "fingerprint": fp}


def run_cpsat(name: str, i: int, budget: float, workers: int = LANE) -> dict:
    from cbba_sota.hetero import evaluate
    from cbba_sota.solvers import cpsat, greedy

    inst, _, fp = _load(name, i)
    t0, c0 = time.perf_counter(), time.process_time()
    hint = greedy.construct(inst, time_limit=min(0.1 * budget, 3.0))
    hint_ms = evaluate(inst, hint).makespan
    res = cpsat.solve_lns(inst, budget, hint, workers=workers, t0=t0)
    wall, cpu = time.perf_counter() - t0, time.process_time() - c0
    return _plan_row(inst, res.plan, wall, cpu) | {"iterations": res.iterations, "init_makespan": hint_ms,
                                                   "last_improvement_s": res.trajectory[-1][0], "fingerprint": fp}


def run_rl(name: str, i: int, index: int, stride: int, budget: float | None = None,
           samples: int | None = None) -> dict:
    """RL samples ``index``, ``index + stride``, ... of the instance: for ``budget`` seconds after this process has
    loaded it, or exactly ``samples`` in one lockstep batch. Returns the best rollout (with coalitions) and timing."""
    from dataclasses import asdict

    from cbba_sota.bench import configs
    from cbba_sota.solvers import rl

    _, data, fp = _load(name, i)
    seed = configs.get(name).seed("dev", i)

    def seed_of(n: int) -> int:
        return rl.sample_seed(seed, index + stride * n)

    start = time.monotonic()
    if samples is None:
        res = rl.sample_until(data, _net, seed_of, start + budget)
    else:
        res = rl.sample_lockstep(data, _net, [seed_of(n) for n in range(samples)], sample=True)
    return {"best": asdict(res.best), "makespans": res.makespans, "cpu_s": res.cpu_s, "start": start,
            "end": time.monotonic(), "fingerprint": fp}


def run_replay(name: str, i: int, routes) -> tuple[float, bool]:
    from cbba_sota.solvers import rl

    return rl.replay(_load(name, i)[1], routes)


def run_greedy(name: str, i: int) -> dict:
    from cbba_sota.bench import configs
    from cbba_sota.solvers import greedy

    c0 = time.process_time()
    res = greedy.greedy_nearest(configs.get(name).instance_path("dev", i))
    return {"makespan": res.makespan, "success": res.success, "env_finished": res.env_finished,
            "wall_s": res.wall_s, "cpu_s": time.process_time() - c0, "routes": res.routes,
            "fingerprint": _load(name, i)[2]}


# --- lane controller (threads of the main process) -------------------------------------------------------------


def _rl_row(parts: list[dict]):
    """Row fields of RL sample blocks that ran concurrently (wall = the longest block, CPU = sum) and the best
    rollout, whose own env run is the RL score."""
    from cbba_sota.solvers import rl

    if len({p["fingerprint"] for p in parts}) > 1:
        raise RuntimeError("RL blocks ran on different instances")
    best = rl.best_of([rl.Result(**p["best"]) for p in parts])
    return {"makespan": best.makespan, "success": best.success, "env_finished": best.env_finished,
            "wall_s": max(p["end"] - p["start"] for p in parts), "cpu_s": sum(p["cpu_s"] for p in parts),
            "n_samples": sum(len(p["makespans"]) for p in parts), "routes": best.routes,
            "fingerprint": parts[0]["fingerprint"]}, best


def _with_replay(ex: ProcessPoolExecutor, row: dict, name: str, i: int) -> dict:
    """Replay of the rollout's routes (after the budget, for reference; the rollout is the env's own run)."""
    rep = ex.submit(run_replay, name, i, row["routes"]).result()
    return row | {"replay_makespan": rep[0], "replay_success": rep[1]}


def _run_job(ex: ProcessPoolExecutor, job: dict) -> dict:
    name, i, method, budget = job["setting"], job["instance"], job["method"], job["budget_s"]
    if method in ("ALNS-1", "ALNS-8"):
        return ex.submit(run_alns, name, i, CORES[method], budget=budget).result()
    if method == "CPSAT-8":
        return ex.submit(run_cpsat, name, i, budget).result()
    if method == "greedy":
        return ex.submit(run_greedy, name, i).result()
    if method in ("RL-1", "RL-8"):
        n = CORES[method]
        parts = [f.result() for f in [ex.submit(run_rl, name, i, p, n, budget) for p in range(n)]]
        return _with_replay(ex, _rl_row(parts)[0], name, i)
    if method == HYBRID:  # the budget starts when the first RL block has loaded the instance
        from cbba_sota.solvers import rl

        parts = [f.result() for f in [ex.submit(run_rl, name, i, p, LANE, samples=8) for p in range(LANE)]]
        rl_row, best = _rl_row(parts)
        t0 = min(p["start"] for p in parts)
        deadline = t0 + budget
        rl_ms = best.makespan if best.success else FAIL
        out = {f"rl_{k}": v for k, v in rl_row.items() if k not in ("routes", "fingerprint")}
        if deadline - time.monotonic() < 0.5:  # RL(s.64) used the whole budget
            return _with_replay(ex, rl_row, name, i) | out | {"polished": False}
        init = rl.to_plan(best, len(best.routes)) if best.success else None
        polish = ex.submit(run_alns, name, i, LANE, deadline=deadline, init=init).result()
        row = polish if polish["success"] and polish["makespan"] <= rl_ms else _with_replay(ex, rl_row, name, i)
        return row | out | {"polished": True, "alns_makespan": polish["makespan"], "wall_s": polish["end"] - t0,
                            "cpu_s": rl_row["cpu_s"] + polish["cpu_s"]}
    raise ValueError(method)


class Lane(threading.Thread):
    """One lane: a pool of ``LANE`` warm processes that works through units from the shared queue."""

    def __init__(self, k: int, units: queue.Queue, sink, cpus: list[int]):
        super().__init__(daemon=True)
        self.k, self.units, self.sink, self.cpus = k, units, sink, cpus
        self.ex = ProcessPoolExecutor(LANE, mp_context=mp.get_context("spawn"), initializer=_init, initargs=(cpus,))
        self.error: BaseException | None = None

    def run(self) -> None:
        try:
            pids: dict[int, str] = {}
            while len(pids) < LANE:  # force all processes up (the executor spawns them on demand)
                pids |= dict(f.result() for f in [self.ex.submit(_pid) for _ in range(LANE)])
            if set(pids.values()) != {runtime.format_cpus(self.cpus)}:
                raise RuntimeError(f"lane {self.k} processes are not pinned to {self.cpus}: {pids}")
            while True:
                try:
                    unit = self.units.get_nowait()
                except queue.Empty:
                    return
                self._unit(unit)
        except BaseException as e:  # noqa: BLE001  (re-raised by the main thread)
            self.error = e
        finally:
            self.ex.shutdown()

    def _unit(self, jobs: list[dict]) -> None:
        def one(job):
            probe, started = runtime.Probe(self.cpus), time.time()
            try:
                row = _run_job(self.ex, job)
            except Exception:  # noqa: BLE001  (logged; the job has no row and reruns on resume)
                print(f"FAILED {job}\n{traceback.format_exc()}", flush=True)
                return
            self.sink(job | row | probe.fields() | {"started": started, "lane": self.k})

        with ThreadPoolExecutor(len(jobs)) as tp:
            list(tp.map(one, jobs))


def _units(settings: list[str], n: int, done: set[tuple], rng: random.Random, scale: float) -> list[list[dict]]:
    """8-core jobs alone, 1-core jobs bundled by (setting, budget); longest first, ties shuffled."""
    from cbba_sota.bench import configs

    singles, bundles = [], []
    for name in settings:
        s = configs.get(name)
        pending: dict[str, list[dict]] = {}
        for label, b in [*budgets(s, scale).items(), ("none", None)]:
            for i in range(min(n, s.n_instances("dev"))):
                for method, cores in CORES.items():
                    if (method == "greedy") != (b is None) or (name, i, label, method) in done:
                        continue
                    job = {"setting": name, "split": "dev", "instance": i, "seed": s.seed("dev", i),
                           "budget": label, "budget_s": b, "method": method, "cores": cores}
                    if cores == LANE:
                        singles.append([job])
                    else:
                        pending.setdefault(label, []).append(job)
        for jobs in pending.values():
            bundles += [jobs[k:k + LANE] for k in range(0, len(jobs), LANE)]
    units = singles + bundles
    rng.shuffle(units)
    units.sort(key=lambda u: -max(j["budget_s"] or 0.0 for j in u))
    return units


def _done(out: Path) -> set[tuple]:
    """Keys of the rows computed on the current instances (matching fingerprint)."""
    return {(r["setting"], r["instance"], r["budget"], r["method"]) for path in out.glob("*.jsonl")
            for r in runtime.read_rows(path) if runtime.is_current(r)}


def run(args) -> None:
    os.environ.update(_ONE)
    args.out.mkdir(parents=True, exist_ok=True)
    units = _units(args.settings, args.instances, _done(args.out), random.Random(0), args.budget_scale)
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
            with (args.out / f"{row['setting']}.jsonl").open("a") as f:
                f.write(json.dumps(row) + "\n")
            count[0] += 1
            print(f"[{time.time() - t0:6.0f}s] {row['setting']} {row['instance']:2d} {row['budget']} "
                  f"{row['method']:12s} {row['makespan']:8.3f} ok={row['success']} wall={row['wall_s']:.1f}s "
                  f"load={row['load1']:.0f} ({count[0]} rows)", flush=True)

    lanes = [Lane(k, work, sink, c) for k, c in enumerate(lane_cpus)]
    for lane in lanes:
        lane.start()
    for lane in lanes:
        lane.join()
    errors = [lane.error for lane in lanes if lane.error is not None]
    if errors:
        raise errors[0]


# --- report ----------------------------------------------------------------------------------------------------


def _load_rows(out: Path) -> dict[tuple[str, str], dict[str, dict[int, dict]]]:
    """(setting, budget label) -> method -> instance -> row (budget-free greedy rows are copied to every budget).
    Only rows computed on the current instances (matching fingerprint) enter."""
    from cbba_sota.bench import configs

    rows: dict[tuple[str, str], dict[str, dict[int, dict]]] = {}
    for path in sorted(out.glob("*.jsonl")):
        name = path.stem
        labels = list(budgets(configs.get(name)))
        stale = 0
        for r in runtime.read_rows(path):
            if not runtime.is_current(r):
                stale += 1
                continue
            for label in labels if r["budget"] == "none" else [r["budget"]]:
                rows.setdefault((name, label), {}).setdefault(r["method"], {})[r["instance"]] = r
        if stale:
            print(f"{path.name}: {stale} rows skipped (no fingerprint, or not the current instance's)")
    return rows


def _score(r: dict) -> float:
    return r["makespan"] if succeeded(r["success"], r["makespan"]) else FAIL


def _boot(x: np.ndarray, reps: int = 10_000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), (reps, len(x)))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _ratio(ours: np.ndarray, other: np.ndarray) -> dict:
    """Paired ratio, bootstrap CI and the one-sided paired t-test of log ratio < log GATE."""
    from scipy import stats as st

    r = ours / other
    lo, hi = _boot(r)
    d = np.log(r) - np.log(GATE)
    p = float(st.ttest_1samp(d, 0.0, alternative="less").pvalue) if d.std() > 0 else float(d.mean() >= 0)
    return {"ratio": float(r.mean()), "lo": lo, "hi": hi, "wins": int((r < 1 - 1e-9).sum()),
            "losses": int((r > 1 + 1e-9).sum()), "p_gate": p}


def report(args) -> None:
    from cbba_sota.stats import holm

    data = _load_rows(args.out)
    table, gates = [], []
    for (name, label), by_method in sorted(data.items(), key=lambda kv: (SETTINGS.index(kv[0][0]), kv[0][1])):
        common = sorted(set.intersection(*(set(v) for v in by_method.values())))
        if len(common) < 2:
            continue
        budget = next(r["budget_s"] for m in by_method.values() for r in m.values() if r["budget_s"])
        print(f"\n{name}  {label} = {budget:g} s  ({len(common)} instances with every method)")
        print(f"  {'method':12s} {'cores':>5s} {'mean ms':>8s} {'succ':>5s} {'wall s':>7s} {'CPU s':>8s} "
              f"{'N':>6s}  extra")
        ms = {m: np.array([_score(rows[i]) for i in common]) for m, rows in by_method.items()}
        for m, cores in CORES.items():
            if m not in by_method:
                continue
            rows = [by_method[m][i] for i in common]
            n = np.mean([r.get("rl_n_samples", r.get("n_samples", np.nan)) for r in rows])  # RL samples
            extra = ""
            if m == HYBRID:
                extra = (f"RL(s.64) {np.mean([_score({'makespan': r['rl_makespan'], 'success': r['rl_success']}) for r in rows]):.3f}"
                         f" in {np.mean([r['rl_wall_s'] for r in rows]):.1f}s; polished {sum(r['polished'] for r in rows)}")
            over = np.mean([r["wall_s"] > r["budget_s"] + 1.0 for r in rows]) if rows[0]["budget_s"] else 0.0
            if over:
                extra += f" overrun>1s {over:.0%}"
            ok = np.mean([succeeded(r["success"], r["makespan"]) for r in rows])
            print(f"  {m:12s} {cores:5d} {ms[m].mean():8.3f} {ok:5.2f} "
                  f"{np.mean([r['wall_s'] for r in rows]):7.1f} {np.mean([r['cpu_s'] for r in rows]):8.1f} "
                  f"{n:6.1f}  {extra}")
            table.append({"setting": name, "budget": label, "budget_s": budget, "method": m, "cores": cores,
                          "n": len(rows), "mean_makespan": float(ms[m].mean()),
                          "success": float(ok),
                          "wall_s": float(np.mean([r["wall_s"] for r in rows])),
                          "cpu_s": float(np.mean([r["cpu_s"] for r in rows])), "n_samples": float(n)})
        others = [m for m in ms if m not in OURS]
        comparisons = [("ALNS-8", "all others", others),
                       ("ALNS-8", "others w/o hybrid", [m for m in others if m != HYBRID]),
                       ("ALNS-8", "RL, same cores", [m for m in others if m == "RL-8"]),
                       ("ALNS-1", "all others", others),
                       ("ALNS-1", "1-core others", [m for m in others if CORES[m] == 1])]
        for ours, scope, pool in comparisons:
            if ours not in ms or not pool:
                continue
            best = min(pool, key=lambda m: ms[m].mean())
            res = _ratio(ms[ours], ms[best])
            oracle = _ratio(ms[ours], np.min([ms[m] for m in pool], axis=0))
            verdict = "PASS" if res["hi"] <= GATE else ("mean<=0.92" if res["ratio"] <= GATE else "FAIL")
            print(f"  {ours}/{best:12s} ({scope:17s}) ratio {res['ratio']:.3f} [{res['lo']:.3f}, {res['hi']:.3f}] "
                  f"wins {res['wins']}/{len(common)}  p(<0.92) {res['p_gate']:.2g}  gate {verdict}   "
                  f"vs per-instance best {oracle['ratio']:.3f} [{oracle['lo']:.3f}, {oracle['hi']:.3f}]")
            gates.append({"setting": name, "budget": label, "ours": ours, "scope": scope, "best": best, **res,
                          "oracle_ratio": oracle["ratio"], "oracle_hi": oracle["hi"]})
    for label in ("B1", "B2"):
        for ours, scope in (("ALNS-8", "all others"), ("ALNS-8", "others w/o hybrid")):
            sel = [g for g in gates if g["budget"] == label and g["ours"] == ours and g["scope"] == scope]
            if not sel:
                continue
            adj = holm([g["p_gate"] for g in sel])
            print(f"\nHolm over settings, {ours} vs best of {scope} at {label}: " + ", ".join(
                f"{g['setting']} {g['ratio']:.3f} p_adj={a:.2g}" for g, a in zip(sel, adj)))
            for g, a in zip(sel, adj):
                g["p_holm"] = float(a)
    if args.csv:
        import csv

        for path, rows in ((args.csv, table), (args.csv.with_name(args.csv.stem + "_ratios.csv"), gates)):
            with path.open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r}))
                w.writeheader()
                w.writerows(rows)
        print(f"\nwrote {args.csv} and {args.csv.with_name(args.csv.stem + '_ratios.csv')}")


def main() -> None:
    from cbba_sota.bench import configs

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["run", "report"])
    ap.add_argument("--settings", nargs="+", default=list(SETTINGS), choices=SETTINGS)
    ap.add_argument("--instances", type=int, default=20)
    ap.add_argument("--lanes", type=int, default=8, help=f"lanes of {LANE} processes")
    ap.add_argument("--out", type=Path, default=configs.RUNS_DIR / "compare_dev")
    ap.add_argument("--csv", type=Path, default=None, help="report: write the table (and *_ratios.csv) here")
    ap.add_argument("--budget-scale", type=float, default=1.0, help="multiply B1/B2 (smoke tests only)")
    ap.add_argument("--cpus", default=None, help="CPUs for the lanes, e.g. 0-63 (default: least busy cores)")
    args = ap.parse_args()
    (run if args.command == "run" else report)(args)


if __name__ == "__main__":
    main()
