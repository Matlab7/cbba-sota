"""Track D planner pilot on the ground truth (``DynTaskEnvX.run_plan`` via ``PlanController``), dev split only.

Methods (agent B, spec 5.2 and references):
  SPARC-L        SPARC, 300 ALNS iterations per structural event (spec key monotonicity)
  SPARC-L-rk     the same with relaxed keys (committed order + head-first; see cbba_sota/dyn/planner.py)
  SPARC-H        SPARC, 3000 iterations
  ins-only       insertion floor only (T1b)
  every-L        every-event RH (R-b), 300 iterations, also re-plans at task finishes
  B4-L           rolling constructor with ``--restarts`` restarts (T1 collapse control)
  B5-L           D-ITAGS-style targeted repair, 300 iterations
  B3-L           state-start CP-SAT-LNS, ``--cp-dtime`` deterministic time per event, 1 worker

Rows are appended to a JSONL file (resumable; key = setting, instance, CRN seed, cell, method). Every row records
the world that produced it (``Episode.world``) and the per-event planner CPU (process time, loaded shared host:
descriptive only; budgets are iteration / restart / deterministic-time counts).

Usage:
  .venv/bin/python scripts/trackD_pilot.py run --n 5 --seeds 2 --procs 10 --out runs/trackD/pilot_b.jsonl
  .venv/bin/python scripts/trackD_pilot.py summary --out runs/trackD/pilot_b.jsonl
  .venv/bin/python scripts/trackD_pilot.py calibrate --n 2 --procs 8
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SETTINGS = ["MA-AT-25-5-50", "MA-AT-50-5-50", "SA-AT-50-5-50", "SA-BT-50-5-50"]
CELLS = ["F1-R2", "F2-R2N3", "F3-pf0.2"]
METHODS = ["SPARC-L", "SPARC-L-rk", "SPARC-H", "ins-only", "every-L", "B4-L", "B5-L"]


def _threads() -> None:
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ[k] = "1"


def make_policy(method: str, args: dict):
    from cbba_sota.dyn.baselines.constructor_rh import ConstructorRH
    from cbba_sota.dyn.baselines.cpsat_rh import CPSATRH, CPSATConfig
    from cbba_sota.dyn.baselines.ditags import DITAGS
    from cbba_sota.dyn.planner import SearchConfig
    from cbba_sota.dyn.sparc import SPARC, SPARCConfig, every_event_rh, insertion_only

    return {
        "SPARC-L": lambda: SPARC(SPARCConfig(iters=300)),
        "SPARC-L-rk": lambda: SPARC(SPARCConfig(iters=300, search=SearchConfig(keys="relaxed"))),
        "SPARC-H": lambda: SPARC(SPARCConfig(iters=3000)),
        "ins-only": lambda: insertion_only(),
        "every-L": lambda: every_event_rh(iters=300),
        "B4-L": lambda: ConstructorRH(restarts=args["restarts"]),
        "B5-L": lambda: DITAGS(iters=300),
        "B3-L": lambda: CPSATRH(CPSATConfig(dtime=args["cp_dtime"], sub_dtime=args["cp_dtime"])),
    }[method]()


def run_job(job: dict) -> dict:
    try:
        return _run_job(job)
    except Exception as e:  # noqa: BLE001  (recorded, not fatal: the row carries the traceback)
        import traceback

        key = {k: job[k] for k in ("setting", "inst", "crn", "cell", "method")}
        return {**key, "excluded": False, "error": repr(e), "traceback": traceback.format_exc()[-3000:]}


def _run_job(job: dict) -> dict:
    _threads()
    import numpy as np

    from cbba_sota.bench.configs import get
    from cbba_sota.dyn.controller import run_env
    from cbba_sota.dyn.perturb import realize
    from cbba_sota.hetero import Instance

    st = get(job["setting"])
    inst = Instance.from_pickle(st.instance_path("dev", job["inst"]))
    real = realize(job["setting"], st.seed("dev", job["inst"]), job["crn"], job["cell"], inst.req, inst.ab, inst.dur)
    key = {k: job[k] for k in ("setting", "inst", "crn", "cell", "method")}
    if real.excluded:
        return {**key, "excluded": True}
    pol = make_policy(job["method"], job)
    violations: list[str] = []
    if job.get("check"):
        from cbba_sota.dyn.planner import check_plan

        orig = pol.replan
        relaxed = job["method"].endswith("-rk")

        def replan(state, scope=None, event_index=None):
            plan = orig(state, scope, event_index)
            violations.extend(check_plan(state, plan, scope, relaxed=relaxed))
            return plan

        pol.replan = replan
    t0 = time.perf_counter()
    ep, ctl = run_env(pol, inst, real)
    cpu = np.array(ctl.cpu) if ctl.cpu else np.zeros(1)
    return {**key, "excluded": False, "makespan": ep.makespan, "success": ep.success, "completion": ep.completion,
            "replans": ep.replans, "structural_events": ep.structural_events, "travel": ep.travel,
            "wasted": ep.wasted_trips, "abandons": ep.abandons, "restarts": ep.restarts, "failures": ep.failures,
            "route_versions": ep.route_versions, "cpu_event_p50": float(np.median(cpu)),
            "cpu_event_p95": float(np.percentile(cpu, 95)), "cpu_event_mean": float(cpu.mean()),
            "cpu_sum": float(cpu.sum()), "wall_s": time.perf_counter() - t0, "world": ep.world,
            "violations": violations[:5] if job.get("check") else None,
            "code": job["code"], "b4_restarts": job["restarts"] if job["method"] == "B4-L" else None,
            "cp_dtime": job["cp_dtime"] if job["method"] == "B3-L" else None}


def _code() -> str:
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                          check=False).stdout
    return f"HEAD {head.strip()} + uncommitted cbba_sota/dyn (agent B day 1)"


def cmd_run(a) -> None:
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            r = json.loads(line)
            done.add((r["setting"], r["inst"], r["crn"], r["cell"], r["method"]))
    code = _code()
    jobs = [{"setting": s, "inst": i, "crn": c, "cell": cell, "method": m, "restarts": a.restarts,
             "cp_dtime": a.cp_dtime, "code": code, "check": a.check}
            for s in a.settings for i in range(a.n) for c in range(a.seeds) for cell in a.cells for m in a.methods
            if (s, i, c, cell, m) not in done]
    jobs.sort(key=lambda j: j["method"] not in ("SPARC-H", "B3-L"))
    print(f"{len(jobs)} jobs on {a.procs} processes", flush=True)
    t0 = time.time()
    with mp.get_context("spawn").Pool(a.procs, initializer=_threads) as pool, out.open("a") as f:
        for k, r in enumerate(pool.imap_unordered(run_job, jobs)):
            f.write(json.dumps(r) + "\n")
            f.flush()
            if (k + 1) % 50 == 0:
                print(f"{k + 1}/{len(jobs)} {time.time() - t0:.0f}s", flush=True)
    print(f"done in {time.time() - t0:.0f}s", flush=True)


def cmd_summary(a) -> None:
    import numpy as np

    rows = [json.loads(line) for line in Path(a.out).read_text().splitlines()]
    bad = [r for r in rows if r.get("violations")]
    print(f"plans with contract violations in {len(bad)} episodes", *(f"\n  {r['method']} {r['violations'][:2]}"
                                                                     for r in bad[:5]))
    errors = [r for r in rows if r.get("error")]
    for r in errors:
        print("ERROR", {k: r[k] for k in ("setting", "inst", "crn", "cell", "method")}, r["error"])
    rows = [r for r in rows if not r.get("excluded") and not r.get("error")]
    by = {}
    for r in rows:
        by[(r["setting"], r["inst"], r["crn"], r["cell"], r["method"])] = r
    methods = sorted({r["method"] for r in rows}, key=lambda m: (METHODS + ["B3-L"]).index(m))
    ref = a.ref
    worlds = sorted({r["world"] for r in rows})
    print(f"{len(rows)} rows; world(s): {worlds}")
    print("success: " + ", ".join(f"{m} {np.mean([r['success'] for r in rows if r['method'] == m]):.3f}"
                                   for m in methods))
    rng = np.random.default_rng(0)
    for cell in sorted({r["cell"] for r in rows}) + ["ALL"]:
        print(f"\n== {cell}: geometric-mean ratio {ref} / method, instance-cluster bootstrap 95% CI (10000), "
              f"wins of {ref} / n pairs (both successful)")
        for m in methods:
            if m == ref:
                continue
            per_inst: dict = {}
            wins = n = 0
            for (s, i, c, ce, mm), r in by.items():
                if mm != ref or (cell != "ALL" and ce != cell):
                    continue
                o = by.get((s, i, c, ce, m))
                if o is None or not (r["success"] and o["success"]):
                    continue
                lr = np.log(r["makespan"] / o["makespan"])
                per_inst.setdefault((s, i), []).append(lr)
                wins += lr < -1e-12
                n += 1
            if not per_inst:
                continue
            x = np.array([np.mean(v) for v in per_inst.values()])
            boots = np.exp(rng.choice(x, (10000, len(x))).mean(axis=1))
            lo, hi = np.percentile(boots, [2.5, 97.5])
            print(f"  vs {m:11s} {np.exp(x.mean()):.4f} [{lo:.4f}, {hi:.4f}]  wins {wins}/{n}  clusters {len(x)}")
    print("\nper-event planner CPU (process time, loaded shared host; descriptive): p50 / p95 of per-episode p50, "
          "mean re-plans per episode")
    for m in methods:
        rr = [r for r in rows if r["method"] == m]
        p50 = np.array([r["cpu_event_p50"] for r in rr]) * 1e3
        p95 = np.array([r["cpu_event_p95"] for r in rr]) * 1e3
        print(f"  {m:11s} {np.median(p50):7.2f} ms / {np.median(p95):7.2f} ms  replans {np.mean([r['replans'] for r in rr]):.1f}")


def calib_job(job: dict) -> dict:
    """CPU per event of SPARC-L and of one cold construction on the same belief states (a SPARC-L episode's)."""
    _threads()
    import numpy as np

    from cbba_sota.bench.configs import get
    from cbba_sota.dyn.controller import PlanController
    from cbba_sota.dyn.env import make_env
    from cbba_sota.dyn.perturb import realize
    from cbba_sota.dyn.planner import RHPlanner
    from cbba_sota.dyn.sparc import SPARC, SPARCConfig
    from cbba_sota.hetero import Instance

    st = get(job["setting"])
    inst = Instance.from_pickle(st.instance_path("dev", job["inst"]))
    real = realize(job["setting"], st.seed("dev", job["inst"]), 0, job["cell"], inst.req, inst.ab, inst.dur)
    if real.excluded:
        return {}
    sp = SPARC(SPARCConfig(iters=300))
    states = []
    orig = sp.replan

    def replan(state, scope=None, event_index=None):
        states.append(state)
        return orig(state, scope, event_index)

    sp.replan = replan
    env = make_env(inst.source, real)
    ctl = PlanController(sp, inst, real.kappa())
    env.run_plan(ctl)
    pl = RHPlanner()
    out = {"sparc": [], "construct": [], "floor": [], "n_open": []}
    for s in states[1:]:  # skip the t = 0 plan
        for _ in range(3):
            c0 = time.process_time()
            pl.plan(s, None, None, 1, iters=300, warm=False)
            out["sparc"].append(time.process_time() - c0)
            c0 = time.process_time()
            pl.plan(s, None, None, 1, iters=0, warm=False, restarts=8)
            out["construct"].append((time.process_time() - c0) / 9)
            c0 = time.process_time()
            pl.plan(s, None, None, 1, iters=0, warm=False)
            out["floor"].append(time.process_time() - c0)
        out["n_open"].append(int(s.open_tasks().size))
    return {k: float(np.mean(v)) if v else 0.0 for k, v in out.items()} | {"setting": job["setting"]}


def cmd_calibrate(a) -> None:
    import numpy as np

    jobs = [{"setting": s, "inst": i, "cell": c} for s in a.settings for i in range(a.n) for c in ("F1-R2",)]
    with mp.get_context("spawn").Pool(a.procs, initializer=_threads) as pool:
        res = [r for r in pool.map(calib_job, jobs) if r]
    for s in a.settings:
        rr = [r for r in res if r["setting"] == s]
        sp, co, fl = (np.mean([r[k] for r in rr]) for k in ("sparc", "construct", "floor"))
        print(f"{s}: SPARC 300 it {sp * 1e3:.2f} ms/event, one construction ~{co * 1e3:.2f} ms, regret floor "
              f"{fl * 1e3:.2f} ms, open tasks {np.mean([r['n_open'] for r in rr]):.1f} -> restarts for equal CPU "
              f"~{(sp - fl) / max(co, 1e-9):.0f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("run", "summary", "calibrate"))
    ap.add_argument("--settings", nargs="+", default=SETTINGS)
    ap.add_argument("--cells", nargs="+", default=CELLS)
    ap.add_argument("--methods", nargs="+", default=METHODS)
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--procs", type=int, default=10)
    ap.add_argument("--restarts", type=int, default=8)
    ap.add_argument("--cp-dtime", type=float, default=0.02)
    ap.add_argument("--ref", default="SPARC-L")
    ap.add_argument("--check", action="store_true", help="validate every plan (check_plan) and record violations")
    ap.add_argument("--out", default=str(ROOT / "runs" / "trackD" / "pilot_b.jsonl"))
    a = ap.parse_args()
    {"run": cmd_run, "summary": cmd_summary, "calibrate": cmd_calibrate}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
