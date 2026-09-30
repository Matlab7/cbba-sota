"""Best-known solutions (BKS) on the validation split: long portfolio runs, CP-SAT bounds and the BKS table.

Usage: bks.py run --kinds ALNS2 [CPLNS CPFULL] [--settings NAME ...] [--n 10] [--width 16] [--lanes 1]
                  [--cpus 0-15] [--seconds S]
       bks.py collect [--settings NAME ...] [--csv FILE]

Kinds (each job uses ``--width`` CPUs, one job per lane at a time):
- ALNS2: ALNS v2 (the default ``ALNSConfig``) with ``width`` forked workers (seeds 1000 + k) for T seconds.
- ALNS2i: the same with independent workers (no incumbent exchange; seeds 2000 + k): the best of ``width`` runs.
  A check (``CHECKS``): run on a subset, compared with the BKS and not part of it.
- CPLNS: CP-SAT LNS (``cpsat.solve_lns``, ``width`` threads) from the ``greedy.construct`` hint built in
  min(10% of T, 3 s), for T seconds; sub-solves of ``SUB_TIME`` seconds (the better variant per setting on dev).
- CPFULL: the monolithic CP-SAT model (``cpsat.solve_full``, ``width`` threads) hinted with the best plan found so
  far for the instance (``collect`` rows), until proven optimal or T seconds. Its integer model rounds each of the
  at most T + 1 arc weights on a critical chain up by at most 1 / SCALE, so ``bound - (n_tasks + 1) / SCALE`` is a
  certified lower bound on the true optimal makespan (``lb``).
- Independent references (``INDEPENDENT``; no ALNS component and no ALNS plan as a hint):
  - PLNS: parallel CP-SAT LNS (``cpsat.solve_lns_parallel``, ``width`` single-thread workers, seeds 3000 + k) from
    the best of their ``greedy.construct`` restart streams built in min(10% of T, 3 s), sub-solves of
    ``pcpsat_sub_time`` seconds;
  - CPFULLc: the full CP-SAT model (all arcs, ``width`` threads, seed 4000) hinted with the best of ``width``
    ``greedy.construct`` restart streams built in min(10% of T, 3 s); certified lower bound as for CPFULL. Up to 200
    tasks only: at 500 tasks the full arc set has about 30 million arc literals.
  They enter the BKS like every other run. ``collect`` also reports, per setting, the best plan of the runs without
  any ALNS component (``alns_free``: the independent references, CPLNS and every competitor of the timed campaigns)
  next to the best plan of the ALNS runs (and of CPFULL, which is hinted with the best plan so far), so that "gap to
  the BKS" has a reference that ALNS did not produce.
T (``--seconds``, default ``SECONDS`` by task count) is wall time from after instance loading, construction
included. Every makespan is the env replay of the returned plan. Rows go to runs/bks/<setting>.jsonl with 0-based
routes, the best-so-far trace, the instance fingerprint, the code version and host conditions; resumable per
(instance, kind, width, T).

``--check`` marks runs that test the BKS budget (e.g. the plan's 10 min x 16 cores on a subset): they are compared
with the BKS and left out of it, so every instance's BKS comes from the same portfolio. ``collect`` takes, per
instance, the best successful env-replayed plan over runs/bks and runs/anytime (current fingerprints only) as the
BKS, with the best certified lower bound, and writes runs/bks/<split>_bks.json (0-based routes
included) and a CSV summary. Budgets are cut from the Phase 1b plan (>= 10 min x 16 cores) to fit the 6-hour
compute window; see docs/headroom-2026-09.md.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import queue
import sys
import threading
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import compare_dev as cd

from cbba_sota.bench import configs, runtime
from cbba_sota.bench.configs import RUNS_DIR
from cbba_sota.hetero.replay import succeeded

OUT = RUNS_DIR / "bks"
ANYTIME = (RUNS_DIR / "anytime", RUNS_DIR / "anytime_c1", RUNS_DIR / "anytime_test")  # timed campaigns
SPLIT = "val"
SETTINGS = ("SA-BT-25-5-20", "SA-AT-25-5-20", "MA-AT-25-5-20", *cd.SETTINGS)
SECONDS = {20: 30.0, 50: 90.0, 200: 240.0, 500: 600.0}  # T per task count (see the module docstring)
SUB_TIME = {"MA-AT-150-10-500": 10.0}  # CP-SAT LNS seconds per sub-solve (default 2 s; dev choice)
SEEDS = {"ALNS2": 1000, "ALNS2i": 2000, "CPLNS": 1, "CPFULL": 0, "PLNS": 3000, "CPFULLc": 4000}
CHECKS = ("ALNS2i",)  # run on a subset only: compared with the BKS, not part of it
INDEPENDENT = ("PLNS", "CPFULLc")  # long reference runs without any ALNS component
# Dev choices of ``anytime.py tune`` (2026-09-28, 8 cores at B1, dev 0-9, 500 tasks dev 0-2): per setting the variant
# with the lowest mean makespan (docs/results/val-c1/tune_dev.txt). Parallel CP-SAT LNS sub-solve seconds (default 2 s):
PCPSAT_SUB_TIME: dict[str, float] = {"SA-BT-25-5-50": 0.5, "SA-BT-50-5-50": 5.0, "SA-AT-50-5-50": 0.5,
                                     "MA-AT-25-5-50": 0.5, "MA-AT-50-5-50": 0.5, "MA-AT-50-5-200": 2.0,
                                     "MA-AT-150-10-500": 5.0, "MA-AT-150-5-500": 2.0}
# Full CP-SAT model: arcs to the k nearest candidate tasks (0: all arcs):
CPFULL_KNN: dict[str, int] = {"SA-BT-25-5-50": 0, "SA-BT-50-5-50": 10, "SA-AT-50-5-50": 0, "MA-AT-25-5-50": 0,
                              "MA-AT-50-5-50": 10, "MA-AT-50-5-200": 20, "MA-AT-150-10-500": 10,
                              "MA-AT-150-5-500": 5}


def sub_time(name: str) -> float:
    return SUB_TIME.get(name, 2.0)


def pcpsat_sub_time(name: str) -> float:
    return PCPSAT_SUB_TIME.get(name, 2.0)


def cpfull_knn(name: str) -> int:
    """Nearest candidate tasks per task in the full CP-SAT model of the CPFULL competitor (0: all arcs)."""
    return CPFULL_KNN.get(name, 0 if configs.get(name).n_tasks <= 50 else 10)


# --- jobs (run in the lane process) ----------------------------------------------------------------------------


def _init(cpus: list[int]) -> None:
    import importlib

    runtime.pin(cpus)
    importlib.import_module("cbba_sota.solvers.cpsat")  # import cost outside the budgets
    from cbba_sota.solvers import alns

    alns._warmup()


def _row(inst, plan, t0: float, c0: float, k0: float) -> dict:
    wall, cpu = time.perf_counter() - t0, time.process_time() - c0 + cd._children_cpu() - k0
    return cd._plan_row(inst, plan, wall, cpu)


def run_job(name: str, split: str, i: int, kind: str, width: int, seconds: float, hint_routes=None) -> dict:
    from cbba_sota.hetero import Plan, evaluate
    from cbba_sota.solvers import alns, cpsat, greedy

    inst, _, fp = cd._load(name, split, i)
    t0, c0, k0 = time.perf_counter(), time.process_time(), cd._children_cpu()
    if kind in ("ALNS2", "ALNS2i"):
        cfg = alns.ALNSConfig(exchange_s=None) if kind == "ALNS2i" else None
        plan, st = alns.solve(inst, seconds, seed=SEEDS[kind], n_workers=width, config=cfg)
        return _row(inst, plan, t0, c0, k0) | {"trace": st.trace, "iterations": st.iterations,
                                               "init_makespan": st.init_makespan, "fingerprint": fp}
    if kind == "CPLNS":
        hint = greedy.construct(inst, time_limit=min(0.1 * seconds, 3.0))
        res = cpsat.solve_lns(inst, seconds, hint, workers=width, seed=SEEDS[kind], sub_time=sub_time(name), t0=t0)
        return _row(inst, res.plan, t0, c0, k0) | {
            "trace": [(t, ms) for t, ms, _ in res.trajectory], "iterations": res.iterations,
            "init_makespan": evaluate(inst, hint).makespan, "sub_time": sub_time(name), "fingerprint": fp}
    if kind == "CPFULL":
        hint = Plan.from_routes(hint_routes, inst.n_tasks)
        res = cpsat.solve_full(inst, seconds, workers=width, hint=hint, seed=SEEDS[kind], t0=t0)
        lb = res.bound - (inst.n_tasks + 1) / cpsat.SCALE if res.bound == res.bound else None
        return _row(inst, res.plan, t0, c0, k0) | {
            "trace": [(t, ms) for t, ms, _ in res.trajectory], "status": res.status, "bound": res.bound,
            "objective": res.objective, "lb": lb, "hint_makespan": evaluate(inst, hint).makespan, "fingerprint": fp}
    if kind == "PLNS":
        res = cpsat.solve_lns_parallel(inst, seconds, workers=width, init_time=min(0.1 * seconds, 3.0),
                                       seed=SEEDS[kind], sub_time=pcpsat_sub_time(name), t0=t0)
        return _row(inst, res.plan, t0, c0, k0) | {
            "trace": [(t, ms) for t, ms, _ in res.trajectory], "iterations": res.iterations,
            "init_makespan": res.init_makespan, "sub_time": pcpsat_sub_time(name), "fingerprint": fp}
    if kind == "CPFULLc":
        if inst.n_tasks > 200:
            raise ValueError("CPFULLc builds every arc: up to 200 tasks only")
        hint = cpsat.construct_parallel(inst, width, t0 + min(0.1 * seconds, 3.0), seed=SEEDS[kind])
        res = cpsat.solve_full(inst, seconds, workers=width, hint=hint, seed=SEEDS[kind], t0=t0)
        lb = res.bound - (inst.n_tasks + 1) / cpsat.SCALE if res.bound == res.bound else None
        return _row(inst, res.plan, t0, c0, k0) | {
            "trace": [(t, ms) for t, ms, _ in res.trajectory], "status": res.status, "bound": res.bound,
            "objective": res.objective, "lb": lb, "hint_makespan": evaluate(inst, hint).makespan, "fingerprint": fp}
    raise ValueError(kind)


# --- collection ------------------------------------------------------------------------------------------------


def rows_of(name: str, split: str | None = None) -> list[dict]:
    """Current-fingerprint rows of an instance set from runs/bks and the timed campaigns (``source`` names the
    directory); ``split`` defaults to the current ``SPLIT`` (set by --split), read at call time."""
    split = SPLIT if split is None else split
    out = []
    for root in (OUT, *ANYTIME):
        path = root / f"{name}.jsonl"
        out += [r | {"source": root.name} for r in runtime.read_rows(path)
                if r.get("split", split) == split and runtime.is_current(r)]
    return out


def alns_free(r: dict) -> bool:
    """The row's plan has no ALNS component: not an ALNS run, not a CPFULL BKS run (hinted with the best plan so
    far), not an ALNS polish."""
    return r.get("kind", "") not in ("ALNS2", "ALNS2i", "CPFULL") and r.get("method", "") not in ("ALNS2", "ALNS1")


def best_known(rows: list[dict]) -> dict[int, dict]:
    """Per instance: the best successful row with routes (``bks``) and the best certified lower bound (``lb``)."""
    out: dict[int, dict] = {}
    for r in rows:
        i = r["instance"]
        e = out.setdefault(i, {"instance": i, "bks": None, "lb": None, "row": None})
        if r.get("check") or r.get("kind") in CHECKS:  # subset checks are compared with the BKS, not part of it
            continue
        if r.get("lb") is not None and (e["lb"] is None or r["lb"] > e["lb"]):
            e["lb"] = r["lb"]
        if r.get("routes") and succeeded(r["success"], r["makespan"]) and (
                e["bks"] is None or r["makespan"] < e["bks"]):
            e["bks"], e["row"] = r["makespan"], r
    return out


def load_bks(split: str | None = None) -> dict[str, dict[int, float]]:
    """setting -> instance -> BKS makespan, from runs/bks/<split>_bks.json (default: the current ``SPLIT``)."""
    split = SPLIT if split is None else split
    data = json.loads((OUT / f"{split}_bks.json").read_text())
    return {name: {int(i): e["bks"] for i, e in by_i.items()} for name, by_i in data.items()}


def label(r: dict) -> str:
    """Short name of the run a row comes from: ``ALNS2-16x90s`` (BKS runs) or ``ALNS2-8@B1`` (anytime runs)."""
    if "kind" in r:
        return f"{r['kind']}-{r['workers']}x{r['time_s']:g}s" + ("-check" if r.get("check") or r["kind"] in CHECKS
                                                                 else "")
    return f"{r['method']}-{r['cores']}@{r['budget']}"


def at_time(trace: list, t: float) -> float | None:
    """Best-so-far makespan of a (seconds, makespan) trace at ``t`` seconds (None before the first point)."""
    vals = [ms for s, ms in trace if s <= t]
    return min(vals) if vals else None


def diagnostics(name: str, rows: list[dict], best: dict[int, dict]) -> list[dict]:
    """Per BKS run type: instances, mean gap to the BKS (%), how often it found the BKS, and for the long runs the
    improvement of their own best-so-far between T / 2 and T (%); CPFULL: proven optima and the BKS-to-bound gap."""
    out = []
    groups: dict[str, dict[int, dict]] = {}
    for r in rows:
        if "kind" in r and r["instance"] in best:
            groups.setdefault(label(r), {})[r["instance"]] = r
    for lab, by_i in sorted(groups.items()):
        ok = [i for i in by_i if best[i]["bks"] is not None]
        gaps = [100 * (by_i[i]["makespan"] - best[i]["bks"]) / best[i]["bks"] for i in ok]
        half = [100 * (h - r["makespan"]) / r["makespan"] for r in by_i.values()
                if r.get("trace") and (h := at_time(r["trace"], r["time_s"] / 2)) is not None]
        d = {"setting": name, "run": lab, "n": len(by_i), "gap_to_bks_pct": float(np.mean(gaps)),
             "found_bks": sum(label(best[i]["row"]) == lab for i in ok),
             "second_half_gain_pct": float(np.mean(half)) if half else None}
        if lab.startswith("CPFULL"):  # CPFULL and CPFULLc
            d["proven_optimal"] = sum(r["status"] == "OPTIMAL" for r in by_i.values())
            lbs = [100 * (best[i]["bks"] - best[i]["lb"]) / best[i]["bks"] for i in ok if best[i]["lb"] is not None]
            d["bks_to_lb_pct"] = float(np.mean(lbs)) if lbs else None
        out.append(d)
    return out


def collect(args) -> None:
    table, diag, out = [], [], {}
    for name in args.settings:
        rows = rows_of(name)
        best = best_known(rows)
        if not best:
            continue
        indep = best_known([r for r in rows if alns_free(r)])
        own = best_known([r for r in rows if not alns_free(r)])
        out[name] = {}
        for i in sorted(best):
            e = best[i]
            if e["bks"] is None:
                continue
            r, lb = e["row"], e["lb"]
            ind = indep.get(i, {}).get("bks")
            out[name][i] = {"bks": e["bks"], "routes": r["routes"], "routes_base": 0, "found_by": label(r),
                            "file": r["source"], "fingerprint": r["fingerprint"], "lb": lb}
            table.append({"setting": name, "instance": i, "bks": e["bks"], "found_by": label(r), "lb": lb,
                          "gap_to_lb_pct": None if lb is None else 100 * (e["bks"] - lb) / e["bks"],
                          "alns_best": own.get(i, {}).get("bks"), "alns_free_best": ind,
                          "alns_free_by": None if ind is None else label(indep[i]["row"])})
        both = [i for i in out[name] if indep.get(i, {}).get("bks") is not None
                and own.get(i, {}).get("bks") is not None]
        if both:
            rel = np.array([100 * (own[i]["bks"] - indep[i]["bks"]) / indep[i]["bks"] for i in both])
            print(f"\n{name}: on {len(both)} instances, best ALNS-derived plan {np.mean([own[i]['bks'] for i in both]):.3f}"
                  f" vs best plan without ALNS {np.mean([indep[i]['bks'] for i in both]):.3f}: ALNS-derived "
                  f"{rel.mean():+.2f}% (min {rel.min():+.2f}%, max {rel.max():+.2f}%); without ALNS better on "
                  f"{int((rel > 1e-7).sum())}")
        d = diagnostics(name, rows, best)
        diag += d
        found = {}
        for e in out[name].values():
            found[e["found_by"]] = found.get(e["found_by"], 0) + 1
        print(f"\n{name}: {len(out[name])} BKS, mean {np.mean([e['bks'] for e in out[name].values()]):.3f}; found by "
              + ", ".join(f"{k} {v}" for k, v in sorted(found.items(), key=lambda kv: -kv[1])))
        for x in d:
            extra = "" if x["second_half_gain_pct"] is None else f", own gain T/2 -> T {x['second_half_gain_pct']:.2f}%"
            if "proven_optimal" in x:
                extra += f", proven optimal {x['proven_optimal']}/{x['n']}"
                if x["bks_to_lb_pct"] is not None:
                    extra += f", BKS to certified bound {x['bks_to_lb_pct']:.2f}%"
            print(f"  {x['run']:18s} n={x['n']:2d} gap to BKS {x['gap_to_bks_pct']:.2f}%, found BKS {x['found_bks']}{extra}")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{SPLIT}_bks.json").write_text(json.dumps(out))  # SPLIT: --split
    print(f"\nwrote {OUT / f'{SPLIT}_bks.json'}: {sum(map(len, out.values()))} instances")
    if args.csv:
        import csv

        for path, rows in ((args.csv, table), (args.csv.with_name(args.csv.stem + "_runs.csv"), diag)):
            with path.open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
                w.writeheader()
                w.writerows(rows)
            print(f"wrote {path}")


# --- runner ----------------------------------------------------------------------------------------------------


def _done(kind: str, width: int, seconds_of, check: bool) -> set[tuple]:
    return {(r["setting"], r["instance"]) for path in OUT.glob("*.jsonl") for r in runtime.read_rows(path)
            if r.get("kind") == kind and r.get("workers") == width and r.get("time_s") == seconds_of(r["setting"])
            and r.get("check", False) == check and runtime.is_current(r)}


def run(args) -> None:
    os.environ.update(cd._ONE)
    OUT.mkdir(parents=True, exist_ok=True)

    def seconds_of(name: str) -> float:
        return args.seconds or SECONDS[configs.get(name).n_tasks]

    jobs = []
    for kind in args.kinds:
        done = _done(kind, args.width, seconds_of, args.check)
        for name in args.settings:
            if kind == "CPFULLc" and configs.get(name).n_tasks > 200:
                print(f"skip CPFULLc on {name}: every arc of 500 tasks (up to 200 tasks only)", flush=True)
                continue
            n = min(args.n, configs.get(name).n_instances(SPLIT))
            jobs += [{"setting": name, "split": SPLIT, "instance": i, "kind": kind, "workers": args.width,
                      "time_s": seconds_of(name)} | ({"check": True} if args.check else {})
                     for i in range(n) if (name, i) not in done]
    print(f"{len(jobs)} jobs, {args.lanes} lanes of {args.width} CPUs", flush=True)
    if not jobs:
        return
    cpus = runtime.parse_cpus(args.cpus) if args.cpus else runtime.choose_cpus(args.width * args.lanes)
    lane_cpus = runtime.blocks(cpus, args.width)[:args.lanes]
    version = runtime.require_clean(args.allow_dirty)
    work: queue.Queue = queue.Queue()
    for job in jobs:
        work.put(job)
    lock, t_start = threading.Lock(), time.time()

    def lane(k: int, cpus: list[int]) -> None:
        with ProcessPoolExecutor(1, mp_context=mp.get_context("spawn"), initializer=_init, initargs=(cpus,)) as ex:
            while True:
                try:
                    job = work.get_nowait()
                except queue.Empty:
                    return
                hint = None
                if job["kind"] == "CPFULL":
                    best = best_known(rows_of(job["setting"])).get(job["instance"])
                    if best is None or best["row"] is None:
                        print(f"skip {job}: no plan to hint", flush=True)
                        continue
                    hint = best["row"]["routes"]
                probe, started = runtime.Probe(cpus), time.time()
                try:
                    row = ex.submit(run_job, job["setting"], SPLIT, job["instance"], job["kind"], job["workers"],
                                    job["time_s"], hint).result()
                except Exception:  # noqa: BLE001  (logged; the job reruns on resume)
                    print(f"FAILED {job}\n{traceback.format_exc()}", flush=True)
                    continue
                row = job | row | probe.fields() | {"started": started, "lane": k, "git": version}
                with lock:
                    with (OUT / f"{job['setting']}.jsonl").open("a") as f:
                        f.write(json.dumps(row) + "\n")
                    extra = f" lb={row['lb']:.3f} {row['status']}" if job["kind"] == "CPFULL" else ""
                    print(f"[{time.time() - t_start:6.0f}s] {job['setting']} {job['instance']:2d} {job['kind']} "
                          f"{row['makespan']:.3f} ok={row['success']} wall={row['wall_s']:.1f}s{extra} "
                          f"load={row['load1']:.0f}", flush=True)

    threads = [threading.Thread(target=lane, args=(k, c), daemon=True) for k, c in enumerate(lane_cpus)]
    print(f"lanes pinned to {[runtime.format_cpus(c) for c in lane_cpus]}, code {version}", flush=True)
    for t in threads:
        t.start()
    for t in threads:
        t.join()


def main() -> None:
    global OUT, SPLIT
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["run", "collect"])
    ap.add_argument("--settings", nargs="+", default=list(SETTINGS))
    ap.add_argument("--kinds", nargs="+", default=["ALNS2"], choices=list(SEEDS))
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--width", type=int, default=16)
    ap.add_argument("--lanes", type=int, default=1)
    ap.add_argument("--seconds", type=float, default=None)
    ap.add_argument("--cpus", default=None)
    ap.add_argument("--csv", type=Path, default=None)
    ap.add_argument("--check", action="store_true", help="BKS-budget check runs: compared with the BKS, not in it")
    ap.add_argument("--out", type=Path, default=OUT, help="rows and BKS table (smoke tests)")
    ap.add_argument("--allow-dirty", action="store_true", help="run on uncommitted source (smoke tests only)")
    ap.add_argument("--split", choices=["val", "test"], default="val",
                    help="collect: the best of every run on this split (test: the timed campaign's rows only)")
    args = ap.parse_args()
    SPLIT = args.split
    if args.command == "run" and SPLIT != "val":
        raise SystemExit("long BKS and reference runs use val only")
    OUT = args.out
    (run if args.command == "run" else collect)(args)


if __name__ == "__main__":
    main()
