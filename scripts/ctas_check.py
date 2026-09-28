"""Checks of the CTAS-D port (``cbba_sota.solvers.ctas``) on the shipped RALTestSet (non-blind) and on one generated
50-task instance. Writes the report (default docs/results/ctas/ralt_check.txt) and the per-run rows, routes
included, next to it (.json).

Usage: ctas_check.py [--seconds 60] [--threads 8] [--jobs 2] [--instances 10] [--large-seconds 30] [--out FILE]

1. The benchmark's Gurobi CTAS-D runs (RALTestSet env_<i>/results.yaml, all 50; 600 s): (a) the env replay of their
   routes against qMax = timeCost / 100, once exactly as the benchmark's baseline/CTAS-D.py (vehicle paths of type
   k + 1 popped onto species k's agents, routes with their trailing depot, ``execute_by_route``) and once through
   ``replay_routes``; (b) the shipped values plugged into ``ctas.Model`` (``ctas.check_shipped``).
2. CTAS-D with each backend, ``--threads`` threads, ``--seconds`` s from model building, on RALTestSet env_0 ..
   env_<instances - 1>; at most ``--jobs`` solves at a time, each in its own process pinned to its own ``--threads``
   idle cores (``runtime.pinned_pool``). Per run: status, env replay of the plan, time to the first incumbent (the
   backend log's time of its first incumbent plus model building), objective and bound (from the log if the
   backend reports none).
3. The same on SA-BT-50-5-50 val 0 with ``--large-seconds`` s: whether any backend finds an incumbent.

Per RALTestSet instance, the Gurobi makespan is the env replay of the shipped solution and the BKS the best
successful env replay among it, the ALNS runs in runs/alns/final-ral-{4s,30s}, the pilot CP-SAT 30 s x 8 routes
(pilots/heteromrta/redteam/cpsat_30s_8w.json; all stored routes are replayed here) and the runs of section 2.
Failures (no plan, or the env does not finish every task below 200) count as makespan 200.
"""
from __future__ import annotations

import argparse
import contextlib
import ctypes
import io
import json
import math
import os
import re
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import ortools

from cbba_sota.bench import configs, runtime
from cbba_sota.bench.configs import HETEROMRTA_DIR, MAX_TIME, ROOT, RUNS_DIR
from cbba_sota.hetero import Instance, make_env, replay, replay_routes
from cbba_sota.solvers import ctas

RAL = HETEROMRTA_DIR / "RALTestSet"
ALNS = [RUNS_DIR / "alns" / f"final-ral-{s}" / "RALTestSet-test.jsonl" for s in ("4s", "30s")]
PILOT_CP = ROOT / "pilots" / "heteromrta" / "redteam" / "cpsat_30s_8w.json"
LARGE = ("SA-BT-50-5-50", "val", 0)
_ONE = {k: "1" for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS")}


def _score(rep: dict | None) -> float:
    return rep["makespan"] if rep is not None and rep["success"] else MAX_TIME


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def _fmt(x: float | None, spec: str = ".3f") -> str:
    return "-" if x is None or (isinstance(x, float) and math.isnan(x)) else format(x, spec)


# --- 1. shipped Gurobi runs ------------------------------------------------------------------------------------


def _ctasd_py(i: int) -> float:
    """baseline/CTAS-D.py on env_<i>: ``baseline`` then ``execute_by_route``; the makespan it records."""
    import yaml

    env = make_env(RAL / f"env_{i}.pkl")
    res = yaml.load((RAL / f"env_{i}" / "results.yaml").read_text(), Loader=yaml.CSafeLoader)
    nodes: dict[int, list] = {k: [] for k in range(len(res["vehNumPerType"]))}
    for v in res["vehicle"].values():
        nodes[v["type"] - 1].append(v["node"])
    for species, routes in nodes.items():
        for route in routes:
            env.pre_set_route(route[1:], env.species_dict[species].pop(0))
    env.force_wait = True
    with contextlib.redirect_stdout(io.StringIO()):
        env.execute_by_route(plot_figure=False)
    return float(env.current_time)


def shipped_checks() -> tuple[list[dict], dict[int, float]]:
    """Section 1 rows, and per instance the env replay of the shipped solution (CTAS-D.py's makespan)."""
    rows, gurobi = [], {}
    for i in range(50):
        inst = Instance.from_pickle(RAL / f"env_{i}.pkl")
        shipped = ctas.shipped_solution(inst, RAL / f"env_{i}")
        if shipped is None:
            rows.append({"i": i, "missing": True})
            continue
        rep = replay_routes(inst.source, [[j + 1 for j in r] for r in shipped.routes])
        gurobi[i] = _score(rep)
        res = shipped.result
        rows.append({"i": i, "qmax": shipped.qmax, "timeCost": res["timeCost"], "replay": rep["makespan"],
                     "success": rep["success"], "skipped": rep["skipped"], "ctasd_py": _ctasd_py(i),
                     "objVal": res["objVal"], "energyCost": res["energyCost"], "MIPGap": res["MIPGap"],
                     **ctas.check_shipped(inst, shipped)})
    return rows, gurobi


def report_shipped(rows: list[dict]) -> list[str]:
    ok = [r for r in rows if not r.get("missing")]
    n = len(ok)
    rep = np.array([abs(r["replay"] - r["qmax"]) for r in ok])
    py = np.array([abs(r["ctasd_py"] - r["replay"]) for r in ok])
    obj = np.array([abs(r["objective"] - r["objVal"]) for r in ok])
    time_cost = sum(abs(r["timeCost"] - ctas.TIME_PENALTY * r["qmax"]) < 1e-4 for r in ok)
    out = ["== 1. Shipped Gurobi CTAS-D runs (RALTestSet results.yaml, 600 s)",
           f"instances with a solution: {n}/{len(rows)}"]
    out.append(f"(a) env replay (replay_routes) = qMax within 1e-5: {(rep < 1e-5).sum()}/{n} (max |diff| "
               f"{rep.max():.1e}); env success {sum(r['success'] for r in ok)}/{n}, skipped visits "
               f"{sum(r['skipped'] for r in ok)}; timeCost = 100 qMax within 1e-4: {time_cost}/{n}")
    out.append(f"    CTAS-D.py's own protocol (trailing depot, execute_by_route) = replay_routes exactly: "
               f"{(py == 0).sum()}/{n} (max |diff| {py.max():.1e})")
    out.append(f"(b) shipped values in ctas.Model: graph.yaml vs our arcs max |diff| "
               f"{max(r['graph_diff'] for r in ok):.1e}; largest violation without energy rows "
               f"{max(r['violation'] for r in ok):.1e} (6-decimal printing; big-M x xr within Gurobi's integrality "
               f"tolerance of 1), with them {max(r['eng_violation'] for r in ok):.2f} (EngEdge, the same with big-M "
               f"1e8); shipped values of variables we do not create max {max(r['dropped'] for r in ok):.1e}")
    out.append(f"    our objective (energy=True) = objVal within 1e-4: {(obj < 1e-4).sum()}/{n} (max |diff| "
               f"{obj.max():.1e}); mean objVal {np.mean([r['objVal'] for r in ok]):.3f}, mean qMax "
               f"{np.mean([r['qmax'] for r in ok]):.3f}, mean MIPGap {np.mean([r['MIPGap'] for r in ok]):.3f}")
    out.append("")
    out.append(f"{'i':>3} {'qMax':>10} {'replay':>10} {'CTAS-D.py':>10} {'objVal':>12} {'ours':>12} {'viol':>8} "
               f"{'eng_viol':>8}")
    for r in ok:
        out.append(f"{r['i']:>3} {r['qmax']:>10.6f} {r['replay']:>10.6f} {r['ctasd_py']:>10.6f} "
                   f"{r['objVal']:>12.6f} {r['objective']:>12.6f} {r['violation']:>8.1e} {r['eng_violation']:>8.1e}")
    return out


# --- 2./3. solves ------------------------------------------------------------------------------------------------


@contextlib.contextmanager
def _captured(path: str):
    """Send this process's C-level stdout and stderr (the backend log) to ``path``."""
    sys.stdout.flush()
    sys.stderr.flush()
    saved = os.dup(1), os.dup(2)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    os.dup2(fd, 1)
    os.dup2(fd, 2)
    os.close(fd)
    try:
        yield
    finally:
        ctypes.CDLL(None).fflush(None)
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(saved[0], 1)
        os.dup2(saved[1], 2)
        os.close(saved[0])
        os.close(saved[1])


def first_incumbent(backend: str, log: str) -> float | None:
    """Seconds from the backend's start to its first incumbent, read from its log; None if it found none."""
    if backend == "CP_SAT":  # "#1   0.18s best:... next:[...] <worker>"
        m = re.search(r"^#1\s+([\d.]+)s ", log, re.MULTILINE)
        return float(m.group(1)) if m else None
    if backend == "HIGHS":  # B&B rows end with "... BestBound BestSol Gap Cuts InLp Confl. LpIters Time"
        for line in log.splitlines():
            tok = line.split()
            if len(tok) >= 12 and re.fullmatch(r"[\d.]+s", tok[-1]):
                with contextlib.suppress(ValueError):
                    if math.isfinite(float(tok[-7])):
                        return float(tok[-1][:-1])
        return None
    col = None  # SCIP: display rows under a header with a primalbound column, "--" before the first incumbent
    for line in log.splitlines():
        cells = [c.strip() for c in line.split("|")]
        if "primalbound" in cells:
            col = cells.index("primalbound")
        elif col is not None and len(cells) > col and cells[col] not in ("", "--"):
            m = re.search(r"([\d.]+)s$", cells[0])
            if m:
                return float(m.group(1))
    return None


def log_bound(backend: str, log: str) -> float:
    """The dual bound a backend's log reports at its end (nan if none)."""
    pattern = {"CP_SAT": r"^best_bound: (\S+)", "HIGHS": r"^\s*Dual bound\s+(\S+)",
               "SCIP": r"^Dual Bound\s+:\s+(\S+)"}[backend]
    found = re.findall(pattern, log, re.MULTILINE)
    with contextlib.suppress(ValueError, IndexError):
        return float(found[-1])
    return math.nan


def run_job(job: dict) -> dict:
    """One CTAS-D solve in a pool process (pinned by the pool), its backend log captured to ``job['log']``."""
    probe = runtime.Probe()
    inst = Instance.from_pickle(job["path"])
    inst.tt, inst.da  # noqa: B018  (instance loading, outside the budget)
    t0 = time.perf_counter()
    with _captured(job["log"]):
        res = ctas.solve(inst, job["seconds"], backend=job["backend"], threads=job["threads"], log=True, t0=t0)
    log = Path(job["log"]).read_text(errors="replace")
    first = first_incumbent(job["backend"], log)
    rep = replay(inst, res.plan) if res.plan is not None else None
    bound = res.bound if math.isfinite(res.bound) else log_bound(job["backend"], log)
    return {**{k: v for k, v in job.items() if k != "log"}, "status": res.status,
            "makespan": rep["makespan"] if rep else None, "success": bool(rep and rep["success"]),
            "skipped": rep["skipped"] if rep else None, "eval_makespan": res.makespan if rep else None,
            "objective": res.objective, "bound": bound, "qmax": res.qmax, "build_s": res.build_s,
            "first_s": None if first is None else res.build_s + first, "wall_s": res.wall_s, "cpu_s": res.cpu_s,
            "routes": res.plan.to_env_routes() if res.plan is not None else None,
            "fingerprint": runtime.fingerprint(inst)} | probe.fields()


def references(n: int) -> dict[int, dict[str, float]]:
    """Per instance < n: env replays of the stored ALNS runs and of the pilot CP-SAT 30 s x 8 routes."""
    out: dict[int, dict[str, float]] = {i: {} for i in range(n)}
    for path in ALNS:
        if path.exists():
            for row in runtime.read_rows(path):
                if row["index"] < n and row.get("routes"):
                    rep = replay_routes(RAL / f"env_{row['index']}.pkl", row["routes"])
                    out[row["index"]][path.parent.name] = _score(rep)
    if PILOT_CP.exists():
        for row in json.loads(PILOT_CP.read_text()):
            i = int(row[0].split("env_")[1].split(".")[0])
            if i < n:
                out[i]["pilot-cpsat-30s-x8"] = _score(replay_routes(RAL / f"env_{i}.pkl", row[-1]))
    return out


def report_runs(runs: list[dict], shipped: dict[int, dict], gurobi: dict[int, float],
                refs: dict[int, dict[str, float]], seconds: float, threads: int, backends: tuple[str, ...]) -> list[str]:
    by = {(r["backend"], r["i"]): r for r in runs}
    ids = sorted({r["i"] for r in runs})
    ours = {i: [by[b, i]["makespan"] for b in backends if by[b, i]["success"]] for i in ids}
    bks = {i: min([gurobi[i], *refs[i].values(), *ours[i]]) for i in ids}
    out = [f"== 2. CTAS-D port, {seconds:g} s from model building, {threads} threads, RALTestSet env_0..{max(ids)}",
           ("per backend: env-replayed makespan (x ratio to the shipped Gurobi 600 s run), time to the first "
            "incumbent [s], status"), "",
           f"{'i':>3} {'Gurobi':>8} {'BKS':>8} " + " ".join(f"{b:>30}" for b in backends)]
    for i in ids:
        cells = []
        for b in backends:
            r = by[b, i]
            ms = f"{r['makespan']:.3f} (x{r['makespan'] / gurobi[i]:.3f})" if r["success"] else "fail"
            cells.append(f"{ms} {_fmt(r['first_s'], '.2f')} {r['status'][:4]}".rjust(30))
        out.append(f"{i:>3} {gurobi[i]:>8.3f} {bks[i]:>8.3f} " + " ".join(cells))
    out.append("")
    out.append(f"{'backend':>8} {'solved':>6} {'mean ms':>8} {'Gurobi':>8} {'ratio':>6} {'ratio200':>8} {'vsBKS':>6} "
               f"{'first_s mean/med/max':>21} {'obj/Gurobi':>10} {'build_s':>7} {'cpu_s':>6}")
    for b in backends:
        rs = [by[b, i] for i in ids]
        solved = [r for r in rs if r["success"]]
        score = [r["makespan"] if r["success"] else MAX_TIME for r in rs]
        firsts = [r["first_s"] for r in rs if r["first_s"] is not None]
        ratio = [r["makespan"] / gurobi[r["i"]] for r in solved]
        # Gurobi's objective without its 1e-4 sum(g) term, as our energy=False objective
        objr = [r["objective"] / (shipped[r["i"]]["energyCost"] + shipped[r["i"]]["timeCost"]) for r in solved]
        first = (f"{np.mean(firsts):.2f}/{np.median(firsts):.2f}/{max(firsts):.2f}" if firsts else "-")
        out.append(f"{b:>8} {len(solved):>3}/{len(rs):<2} {np.mean(score):>8.3f} "
                   f"{np.mean([gurobi[i] for i in ids]):>8.3f} {_fmt(float(np.mean(ratio)) if ratio else None):>6} "
                   f"{np.mean([s / gurobi[r['i']] for s, r in zip(score, rs)]):>8.3f} "
                   f"{np.mean([s / bks[r['i']] for s, r in zip(score, rs)]):>6.3f} {first:>21} "
                   f"{_fmt(float(np.mean(objr)) if objr else None):>10} "
                   f"{np.nanmean([r['build_s'] for r in rs]):>7.3f} {np.mean([r['cpu_s'] for r in rs]):>6.1f}")
    out.append("ratio: mean over solved instances of makespan / Gurobi; ratio200 and vsBKS: over all instances, "
               "failures as 200; first_s over runs with an incumbent; obj/Gurobi: our CTAS objective / "
               "(energyCost + timeCost) of the shipped run, solved instances.")
    out.append("BKS sources: " + "; ".join(f"{i}: " + ", ".join(f"{k} {v:.3f}" for k, v in sorted(refs[i].items()))
                                          for i in ids))
    return out


def report_large(runs: list[dict], seconds: float, threads: int) -> list[str]:
    name, split, i = LARGE
    out = [(f"== 3. {name} {split} {i}, {seconds:g} s from model building, {threads} threads (paper: Gurobi "
            f"CTAS-D solved 96% of this setting in 3600 s)"),
           (f"{'backend':>8} {'status':>10} {'makespan':>9} {'first_s':>8} {'build_s':>8} {'objective':>11} "
            f"{'bound':>10} {'wall_s':>7}")]
    for r in runs:
        out.append(f"{r['backend']:>8} {r['status']:>10} {_fmt(r['makespan'])!s:>9} {_fmt(r['first_s'], '.2f'):>8} "
                   f"{_fmt(r['build_s'], '.2f'):>8} {_fmt(r['objective'], '.2f'):>11} {_fmt(r['bound'], '.2f'):>10} "
                   f"{r['wall_s']:>7.1f}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--instances", type=int, default=10)
    ap.add_argument("--large-seconds", type=float, default=30.0)
    ap.add_argument("--backends", nargs="+", default=list(ctas.BACKENDS), choices=ctas.BACKENDS)
    ap.add_argument("--out", type=Path, default=ROOT / "docs" / "results" / "ctas" / "ralt_check.txt")
    args = ap.parse_args()
    os.environ.update(_ONE)  # inherited by the pool
    started = _now()
    backends = tuple(args.backends)

    rows, gurobi = shipped_checks()
    refs = references(args.instances)

    logs = Path(tempfile.mkdtemp(prefix="ctas_check_"))
    large = configs.get(LARGE[0]).instance_path(LARGE[1], LARGE[2])
    jobs = [{"backend": b, "i": i, "path": str(RAL / f"env_{i}.pkl"), "seconds": args.seconds,
             "threads": args.threads, "log": str(logs / f"{b}_{i}.log")} for i in range(args.instances) for b in backends]
    jobs += [{"backend": b, "i": "large", "path": str(large), "seconds": args.large_seconds, "threads": args.threads,
              "log": str(logs / f"{b}_large.log")} for b in backends]
    with runtime.pinned_pool(args.jobs, args.threads) as pool:
        runs = list(pool.map(run_job, jobs))
    ral = [r for r in runs if r["i"] != "large"]
    big = [r for r in runs if r["i"] == "large"]

    lines = [(f"CTAS-D port check (cbba_sota.solvers.ctas); started {started}, finished {_now()}; code "
              f"{runtime.code_version()}; OR-Tools {ortools.__version__}; host {runs[0]['host']}; command: "
              f"{' '.join(sys.argv)}"),
             (f"jobs: {args.jobs} at a time, pinned to CPUs {sorted({r['affinity'] for r in runs})}; load1 at job "
              f"starts {min(r['load1'] for r in runs):.1f}..{max(r['load1'] for r in runs):.1f}"), ""]
    lines += report_shipped(rows) + [""]
    shipped = {r["i"]: r for r in rows if not r.get("missing")}
    lines += report_runs(ral, shipped, gurobi, refs, args.seconds, args.threads, backends) + [""]
    lines += report_large(big, args.large_seconds, args.threads)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n")
    args.out.with_suffix(".json").write_text(json.dumps({"shipped": rows, "runs": runs}, default=float) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
