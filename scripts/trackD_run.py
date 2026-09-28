"""Track D campaign runner: resumable JSONL rows on DynTaskEnvX (pattern of cbja scripts/run_v5.py).

A case is (setting, split, instance, CRN seed, cell, method, budget tier, world options). Its ``case_id`` hashes all
of these; a case is done when some ``rows*.jsonl`` in the output directory holds a row with that id, status
``completed`` or ``excluded`` and the current env code hash (``--any-code`` accepts rows from older env code).
Each row says which world produced it (``world``: DynTaskEnvX, coalition rule, observations, lease, env code hash)
and carries the git commit (and whether ``cbba_sota/dyn`` or this script had uncommitted changes).

Methods:
  RL(g.) / RL(s.1)          the released checkpoint online (``RLPolicy``), policy seed keyed by (instance, seed)
  greedy                    the paper greedy, distance bug fixed, online (B7, ``NearestPolicy``)
  OL-construct              descriptive smoke reference, only for cells without release: our constructor plan
                            (8 restarts) on the nominal instance, followed open loop (no re-planning)
  SPARC, SPARC-rk, ins-only, every-event, open-loop, B3, B4, B5
                            rolling planners (``cbba_sota.dyn.methods``) through ``PlanController`` in ``run_plan``;
                            need ``--tier light`` or ``heavy`` (budgets in ``methods.TIERS``, overridable by
                            parameters, e.g. ``'B4?restarts=15'``); planner seeds come from the belief (spec 4.1)
  ctrl:<module>:<factory>   a plan-following controller: ``factory(env, realization, tier, seed, **params)``
                            returns an object with ``on_events(env, t, events)`` (``run_plan``)
  pol:<module>:<factory>    a policy: ``factory(env, realization, tier, seed, **params)`` returns an object with
                            ``selectable(env, i)`` / ``act(env, i)`` (``run_policy``)
  Parameters follow after '?' as k=v pairs joined by '&' (numbers parsed), e.g.
  ``ctrl:cbba_sota.dyn.sparc:make_controller?iters=300``.

Budget tiers (docs/trackD-spec.md Section 3.6): native | light | heavy; passed to factories, recorded in rows.
Deterministic budgets only; nothing here reads a wall clock inside an episode (CPU is measured, not used).

Worlds: ``--world env`` (default) is the ground truth ``DynTaskEnvX``. ``--world executor`` runs the rolling
planners in the hand-written executor (``cbba_sota.dyn.executor``), allowed for speed only where gate K0 holds
(``scripts/trackD_k0.py``: executor == env trajectories at the verified env/executor code); its rows say so in
``world`` and carry the executor code hash.

Dev split only (the validation split of the spec is not generated yet; the test split is refused). Workers are
capped at 12, each single-threaded (OMP/MKL/NUMBA/torch = 1). Output: runs/trackD/campaign/rows*.jsonl.

    .venv/bin/python scripts/trackD_run.py --families F1 F2 F3 --methods 'RL(g.)' greedy --seeds 0 1 --workers 8
    .venv/bin/python scripts/trackD_run.py --cells F0-N3 --methods OL-construct --instances 0-4 --dry
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import multiprocessing as mp
import os
import subprocess
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "trackD" / "campaign"
MAX_WORKERS = 12
_ONE_THREAD = {k: "1" for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS")}
BUILTIN = ("RL(g.)", "RL(s.1)", "greedy", "OL-construct")
WORLDS = ("env", "executor")


# --- cases -----------------------------------------------------------------------------------------------------------


def parse_range(spec: list[str]) -> list[int]:
    out: list[int] = []
    for part in spec:
        for chunk in str(part).split(","):
            if "-" in chunk:
                a, b = chunk.split("-")
                out += list(range(int(a), int(b) + 1))
            elif chunk:
                out.append(int(chunk))
    return sorted(set(out))


def parse_method(method: str) -> tuple[str, dict]:
    """``name?k=v&k2=v2`` -> (name, params)."""
    name, _, query = method.partition("?")
    params: dict = {}
    for kv in filter(None, query.split("&")):
        k, _, v = kv.partition("=")
        try:
            params[k] = json.loads(v)
        except json.JSONDecodeError:
            params[k] = v
    return name, params


def world_options(args) -> dict:
    opts = {"coalition": args.coalition, "obs": args.obs, "lease_L": args.lease_L,
            "abandon_on_failure": not args.no_abandon_on_failure}
    if getattr(args, "world", "env") != "env":
        opts["world"] = args.world  # env rows keep their case ids (no key added)
    return opts


def case_id(case: dict) -> str:
    key = {k: case[k] for k in ("setting", "split", "instance", "seed", "cell", "method", "tier", "options")}
    return hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()[:16]


def build_cases(args) -> list[dict]:
    from cbba_sota.dyn import perturb

    cells = list(args.cells or [])
    for fam in args.families or []:
        cells += list(perturb.FAMILIES[fam])
    cells = list(dict.fromkeys(cells)) or list(perturb.FAMILIES["F1"])
    for c in cells:
        perturb.cell(c)  # validates
    opts = world_options(args)
    cases = []
    for s in args.settings:
        for i in parse_range(args.instances):
            for seed in parse_range(args.seeds):
                for c in cells:
                    for m in args.methods:
                        case = {"setting": s, "split": args.split, "instance": i, "seed": seed, "cell": c,
                                "family": perturb.cell(c).family, "method": m, "tier": args.tier, "options": opts}
                        case["case_id"] = case_id(case)
                        cases.append(case)
    return cases


# --- provenance ------------------------------------------------------------------------------------------------------


def git_state() -> dict:
    def git(*a):
        return subprocess.run(["git", "-C", str(ROOT), *a], capture_output=True, text=True, check=False).stdout.strip()

    dirty = git("status", "--porcelain", "--", "cbba_sota/dyn", "scripts/trackD_run.py")
    return {"git": git("rev-parse", "--short=12", "HEAD"), "git_dirty_dyn": bool(dirty)}


# --- worker ----------------------------------------------------------------------------------------------------------

_NET = None


def _init_worker() -> None:
    os.environ.update(_ONE_THREAD)
    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)


def _net():
    global _NET
    if _NET is None:
        from cbba_sota.solvers import rl

        _NET = rl.load_policy("cpu")
    return _NET


def policy_seed(instance_key: int, crn_seed: int, method: str) -> int:
    """Seed of a method's own randomness (shuffle, sampling, search), keyed by (instance, CRN seed, method)."""
    import numpy as np

    tag = int(hashlib.sha256(method.encode()).hexdigest()[:8], 16)
    return int(np.random.SeedSequence([int(instance_key), int(crn_seed), tag]).generate_state(1)[0])


def _factory(spec: str):
    _, module, attr = spec.split(":", 2)
    return getattr(importlib.import_module(module), attr)


def run_case(case: dict, prov: dict) -> dict:
    import numpy as np

    from cbba_sota.bench import configs
    from cbba_sota.dyn import perturb
    from cbba_sota.dyn.env import code_hash
    from cbba_sota.dyn.methods import PLANNERS, make_policy

    row = {k: case[k] for k in ("case_id", "setting", "split", "instance", "seed", "cell", "family", "method",
                                 "tier")}
    row.update(options=case["options"], code_hash=code_hash(), **prov, started=time.strftime("%Y-%m-%dT%H:%M:%S"))
    wall0, cpu0 = time.perf_counter(), time.process_time()
    try:
        if case["split"] == "test":
            raise PermissionError("the test split is frozen until the Track D prereg")
        s = configs.get(case["setting"])
        rz = perturb.realize_instance(case["setting"], case["split"], case["instance"], case["seed"], case["cell"])
        row.update(instance_key=s.seed(case["split"], case["instance"]), H=rz.H, n_tasks=rz.n_tasks,
                   n_agents=rz.n_agents, n_released_at_0=int((rz.release <= 0).sum()),
                   n_fail_draws=int(np.isfinite(rz.fail_onset).sum()), excluded=rz.excluded)
        if rz.excluded:
            row.update(status="excluded", note="failure realization leaves a task uncoverable (excluded, paired)")
            return row
        options = dict(case["options"])
        world = options.pop("world", "env")
        name, params = parse_method(case["method"])
        pseed = policy_seed(row["instance_key"], case["seed"], name)
        row["method_seed"] = pseed
        if world == "executor":
            if name not in PLANNERS:
                raise ValueError(f"the executor world runs the rolling planners only, not {name!r}")
            row.update(_run_executor(s, case, rz, make_policy(name, case["tier"], **params)))
        else:
            row.update(_run_env(s, case, rz, options, name, params, pseed))
    except Exception as e:  # noqa: BLE001  (recorded in the row, not fatal for the campaign)
        row.update(status="error", error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc()[-2000:])
    row.update(wall_s=time.perf_counter() - wall0, cpu_s=time.process_time() - cpu0)
    return row


def _run_env(s, case: dict, rz, options: dict, name: str, params: dict, pseed: int) -> dict:
    """One episode in the ground truth ``DynTaskEnvX``."""
    import numpy as np

    from cbba_sota.dyn.env import NearestPolicy, RLPolicy, make_env
    from cbba_sota.dyn.methods import PLANNERS, make_controller

    env = make_env(s.instance_path(case["split"], case["instance"]), rz, **options)
    if name in PLANNERS:
        ep = env.run_plan(make_controller(env, rz, case["tier"], pseed, method=name, **params))
    elif name in ("RL(g.)", "RL(s.1)"):
        ep = env.run_policy(RLPolicy(_net(), sample=name == "RL(s.1)", seed=pseed), shuffle_seed=pseed)
    elif name == "greedy":
        ep = env.run_policy(NearestPolicy())
    elif name == "OL-construct":
        if rz.cell.release != "none":
            raise ValueError("OL-construct needs every task known at t = 0")
        from cbba_sota.solvers import greedy

        plan = greedy.construct(env.nominal_instance(), restarts=int(params.get("restarts", 8)), seed=pseed)
        ep = env.run_plan(routes=plan.routes())
    elif name.startswith("ctrl:"):
        ctl = _factory(name)(env, rz, case["tier"], pseed, **params)
        ep = env.run_plan(ctl)
    elif name.startswith("pol:"):
        pol = _factory(name)(env, rz, case["tier"], pseed, **params)
        ep = env.run_policy(pol, shuffle_seed=pseed if params.get("shuffle", True) else None)
    else:
        raise ValueError(f"unknown method {case['method']!r}")
    cpu = np.asarray(ep.cpu_event_s, float) * 1e3
    return {"status": "completed", "world": ep.world, "makespan": ep.makespan, "success": ep.success,
            "completion": ep.completion, "env_finished": ep.env_finished,
            "makespan_or_cap": ep.makespan_or_cap,  # P90 input / failure imputation
            "travel": ep.travel, "wait": ep.wait, "wait_per_agent": ep.wait / max(rz.n_agents, 1),
            "wasted_trips": ep.wasted_trips, "abandons": ep.abandons, "restarts": ep.restarts,
            "failures": ep.failures, "detected": ep.detected, "decisions": ep.decisions,
            "controller_calls": ep.controller_calls, "replans": ep.replans, "route_versions": ep.route_versions,
            "structural_events": ep.structural_events, "epochs": ep.epochs, "cpu_events": len(cpu),
            "cpu_ms_p50": float(np.percentile(cpu, 50)) if cpu.size else None,
            "cpu_ms_p95": float(np.percentile(cpu, 95)) if cpu.size else None,
            "cpu_ms_max": float(cpu.max()) if cpu.size else None,
            "cpu_ms_event": [round(float(x), 4) for x in cpu], "cpu_method_s": ep.cpu_total_s}


def _run_executor(s, case: dict, rz, policy) -> dict:
    """A rolling planner in the hand-written executor (K0-verified world; see the module docstring)."""
    import numpy as np

    from cbba_sota.dyn.executor import Executor, Realization
    from cbba_sota.hetero import Instance

    inst = Instance.from_pickle(s.instance_path(case["split"], case["instance"]))
    ex = Executor(inst, Realization.from_perturb(rz), detect_h=rz.detect_after)
    res = ex.run(policy)
    cpu = np.array([d["cpu_s"] for d in policy.decisions if d["replan"]], float) * 1e3
    return {"status": "completed", "world": res["world"], "makespan": res["makespan"] if res["success"] else None,
            "success": res["success"], "completion": res["completion"],
            "makespan_or_cap": res["makespan"] if res["success"] else 200.0, "travel": res["travel"],
            "wasted_trips": res["wasted"], "abandons": res["abandons"], "restarts": res["aborts"],
            "failures": res["n_failed"], "replans": res["n_replans"], "cpu_events": len(cpu),
            "cpu_ms_p50": float(np.percentile(cpu, 50)) if cpu.size else None,
            "cpu_ms_p95": float(np.percentile(cpu, 95)) if cpu.size else None,
            "cpu_ms_max": float(cpu.max()) if cpu.size else None, "cpu_ms_event": [round(float(x), 4) for x in cpu],
            "cpu_method_s": float(cpu.sum() / 1e3), "note": "cpu = planner calls only (executor world)"}


# --- main --------------------------------------------------------------------------------------------------------------


def load_done(out: Path, any_code: bool) -> set[str]:
    from cbba_sota.dyn.env import code_hash

    current = code_hash()
    done = set()
    for f in out.glob("rows*.jsonl"):
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("status") in ("completed", "excluded") and (any_code or row.get("code_hash") == current):
                done.add(row["case_id"])
    return done


def main(argv=None) -> None:
    from cbba_sota.dyn import perturb

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--settings", nargs="+", default=list(perturb.PRIMARY_SETTINGS))
    ap.add_argument("--split", default="dev", choices=["dev"])  # validation: not generated yet (configs.SPLITS)
    ap.add_argument("--instances", nargs="+", default=["0-19"])
    ap.add_argument("--seeds", nargs="+", default=["0-1"])
    ap.add_argument("--cells", nargs="+", default=None, help=f"cells: {', '.join(perturb.CELLS)}")
    ap.add_argument("--families", nargs="+", default=None, help=f"families: {', '.join(perturb.FAMILIES)}")
    ap.add_argument("--methods", nargs="+", default=["RL(g.)", "greedy"])
    ap.add_argument("--tier", default="native", choices=["native", "light", "heavy"])
    ap.add_argument("--coalition", default="auto", choices=["auto", "arrival", "decision"])
    ap.add_argument("--obs", default="causal", choices=["causal", "oracle"])
    ap.add_argument("--lease-L", dest="lease_L", type=float, default=200.0)
    ap.add_argument("--no-abandon-on-failure", action="store_true")
    ap.add_argument("--world", default="env", choices=WORLDS, help="env (ground truth) or executor (K0-verified)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--rows", default="rows.jsonl", help="output file name in --out (one per concurrent runner)")
    ap.add_argument("--any-code", action="store_true", help="count rows from older env code as done")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args(argv)
    if args.split == "test":
        raise SystemExit("the test split is frozen")
    workers = max(1, min(args.workers, MAX_WORKERS))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    done = load_done(out, args.any_code)
    cases = [c for c in build_cases(args) if c["case_id"] not in done]
    if args.limit is not None:
        cases = cases[:args.limit]
    print(f"{len(cases)} pending ({len(done)} done rows in {out}), workers={workers}", flush=True)
    if args.dry or not cases:
        return
    prov = git_state()
    os.environ.update(_ONE_THREAD)  # inherited by the spawned workers
    path = out / args.rows
    finished, t0 = 0, time.time()
    while cases:
        try:
            with path.open("a") as sink, ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn"),
                                                             initializer=_init_worker) as pool:
                futures = {pool.submit(run_case, c, prov): c for c in cases}
                for fut in as_completed(futures):
                    row = fut.result()
                    sink.write(json.dumps(row, default=float) + "\n")
                    sink.flush()
                    cases = [c for c in cases if c["case_id"] != row["case_id"]]
                    finished += 1
                    if finished % 25 == 0 or not cases or row["status"] == "error":
                        print(f"{finished} done, {len(cases)} left, {time.time() - t0:.0f}s: {row['setting']} "
                              f"{row['instance']} {row['cell']} {row['method']} {row['status']} "
                              f"{row.get('makespan')} {row.get('error', '')}", flush=True)
        except BrokenProcessPool:
            print(f"pool broke with {len(cases)} left; restarting", flush=True)


if __name__ == "__main__":
    sys.exit(main())
