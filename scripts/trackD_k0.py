"""Gate K0 (docs/trackD-spec.md Section 8, day 2): executor vs env fidelity for plan-following methods.

For every case (setting, dev instance, CRN seed, cell, planner) the same planner is run three times on the same
``perturb`` realization:

  env        ``DynTaskEnvX.run_plan`` with ``PlanController`` (the ground truth; D2 arrival-order coalitions)
  decision   the same with the env's decision-order coalition rule (check (ii): arrival == decision at ideal comms)
  executor   ``cbba_sota.dyn.executor.Executor`` (the fast hand-written world K0 is about)

and the executor and decision-order trajectories are compared with the env's:
  - the re-plan sequence: times and planner seeds. A seed is ``hash(belief digest, event index)``, so equal seeds mean
    bit-identical beliefs at every planner call;
  - every robot's legs in order: departure time, destination, arrival (inf for a robot that halts on the way);
  - every task's (final) start and finish;
  - makespan, success, completion.
A case *matches* when the structure is identical (same re-plans, seeds, legs, destinations) and every time agrees
to ``--tol`` (spec: 1e-6); the largest absolute time difference is recorded (0.0 = bit-identical). Counters (wasted
trips, abandons, restarts, travel) are compared and reported, not gated.

Rows go to ``runs/trackD/k0/rows.jsonl`` (resumable on (setting, instance, seed, cell, method, tier)); ``summary``
prints the gate table per family. Dev split only.

``rl`` is check (iv), RL online with every perturbation off: on the static realization, ``DynTaskEnvX.run_policy``
with the released checkpoint must reproduce the Phase-1 static RL numbers bit for bit:
  - the stored samples of ``runs/rl/<setting>/dev.jsonl`` (RL(s.64), seeds ``rl.sample_seed(instance seed, k)``,
    k < 64) re-run one by one in the env: the multiset of the 64 makespans and the best must equal the stored ones
    (``sample_makespans`` is stored in block-completion order of the parallel run, not seed order);
  - RL(g.) with seed ``sample_seed(instance seed, 0)`` (the Phase-1 RL(g.) seed) vs ``rl.rollout`` on the plain env;
  - RL(g.) and RL(s.1) through ``scripts/trackD_run.py`` (``run_case``, cell ``static``, its own method seed) vs
    ``rl.rollout`` on the plain env with the same seed.
Rows go to ``runs/trackD/k0/rl_static.jsonl``.

    .venv/bin/python scripts/trackD_k0.py run --procs 12
    .venv/bin/python scripts/trackD_k0.py summary
    .venv/bin/python scripts/trackD_k0.py rl --procs 12
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
OUT = ROOT / "runs" / "trackD" / "k0" / "rows.jsonl"
MAX_PROCS = 16
FAMILY_CELLS = ["static", "F0-N1", "F0-N2", "F0-N3", "F1-R1", "F1-R2", "F1-R3", "F2-R2N3", "F3-pf0.1", "F3-pf0.2",
                "F3-pf0.2-R2"]


def _threads() -> None:
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ[k] = "1"


def _no_op(leg: tuple[float, int, float]) -> bool:
    """A zero-length trip home (an unused robot at its depot sent home at once): no movement, not compared."""
    return leg[1] < 0 and leg[0] == leg[2]


def _legs_env(env) -> list[list[tuple[float, int, float]]]:
    out = []
    for i, a in env.agent_dic.items():
        arr = a["arrival_time"][1:]
        legs = env.legs[i]
        assert len(arr) == len(legs)
        out.append([x for x in ((float(d), int(to) if to >= 0 else -1, float(x))
                                for (_, d, _, to), x in zip(legs, arr)) if not _no_op(x)])
    return out


def trajectory_env(env, ep, policy) -> dict:
    T = len(env.task_dic)
    fin = [bool(env.task_dic[j]["finished"]) for j in range(T)]
    return {"replans": [(float(d["t"]), int(d["seed"])) for d in policy.decisions if d["replan"]],
            "legs": _legs_env(env),
            "start": [float(env.task_dic[j]["time_start"]) if fin[j] else None for j in range(T)],
            "finish": [float(env.task_dic[j]["time_finish"]) if fin[j] else None for j in range(T)],
            "makespan": ep.makespan if ep.success else None, "success": bool(ep.success),
            "completion": float(ep.completion),
            "counters": {"wasted": ep.wasted_trips, "abandons": ep.abandons, "restarts": ep.restarts,
                         "travel": float(ep.travel)}}


def trajectory_executor(ex, res, policy) -> dict:
    T = ex.T
    return {"replans": [(float(d["t"]), int(d["seed"])) for d in policy.decisions if d["replan"]],
            "legs": [[x for x in ((float(d), int(j), float(a)) for d, j, a in legs) if not _no_op(x)]
                     for legs in ex.legs],
            "start": [float(ex.start_t[j]) if ex.done[j] else None for j in range(T)],
            "finish": [float(ex.finish_t[j]) if ex.done[j] else None for j in range(T)],
            "makespan": res["makespan"] if res["success"] else None, "success": bool(res["success"]),
            "completion": float(res["completion"]),
            "counters": {"wasted": res["wasted"], "abandons": res["abandons"], "restarts": res["aborts"],
                         "travel": float(res["travel"])}}


def compare(a: dict, b: dict, tol: float) -> dict:
    """Differences of trajectory ``b`` from the reference ``a``: structure first, then the largest time gap."""
    import math

    diffs: list[float] = []
    first = None

    def gap(x, y, what):
        nonlocal first
        if x is None or y is None:
            if (x is None) != (y is None) and first is None:
                first = f"{what}: {x} vs {y}"
            return
        if math.isinf(x) or math.isinf(y):
            if x != y and first is None:
                first = f"{what}: {x} vs {y}"
            return
        diffs.append(abs(x - y))
        if abs(x - y) > tol and first is None:
            first = f"{what}: {x!r} vs {y!r}"

    ra, rb = a["replans"], b["replans"]
    for k, (x, y) in enumerate(zip(ra, rb)):
        gap(x[0], y[0], f"re-plan {k} time")
        if x[1] != y[1] and first is None:
            first = f"re-plan {k} at t={x[0]:.6f}: seed {x[1]} vs {y[1]} (beliefs differ)"
    if len(ra) != len(rb) and first is None:
        first = f"re-plan count {len(ra)} vs {len(rb)}"
    for i, (la, lb) in enumerate(zip(a["legs"], b["legs"])):
        for k, (x, y) in enumerate(zip(la, lb)):
            if x[1] != y[1] and first is None:
                first = f"robot {i} leg {k}: to {x[1]} vs {y[1]} (t={x[0]:.6f})"
            gap(x[0], y[0], f"robot {i} leg {k} departure")
            gap(x[2], y[2], f"robot {i} leg {k} arrival")
        if len(la) != len(lb) and first is None:
            first = f"robot {i}: {len(la)} vs {len(lb)} legs"
    for key in ("start", "finish"):
        for j, (x, y) in enumerate(zip(a[key], b[key])):
            gap(x, y, f"task {j} {key}")
    gap(a["makespan"], b["makespan"], "makespan")
    if (a["success"], a["completion"]) != (b["success"], b["completion"]) and first is None:
        first = f"success/completion {a['success']}/{a['completion']} vs {b['success']}/{b['completion']}"
    counters = {k: (a["counters"][k], b["counters"][k]) for k in a["counters"]
                if (abs(a["counters"][k] - b["counters"][k]) > tol if k == "travel"
                    else a["counters"][k] != b["counters"][k])}
    return {"match": first is None, "first": first, "max_diff": max(diffs, default=0.0),
            "exact": first is None and max(diffs, default=0.0) == 0.0, "counters_differ": counters}


def run_case(job: dict) -> dict:
    _threads()
    key = {k: job[k] for k in ("setting", "instance", "seed", "cell", "method", "tier")}
    try:
        return {**key, **_run_case(job)}
    except Exception as e:  # noqa: BLE001  (recorded in the row)
        import traceback

        return {**key, "status": "error", "error": repr(e), "traceback": traceback.format_exc()[-3000:]}


def _run_case(job: dict) -> dict:
    from cbba_sota.bench.configs import get
    from cbba_sota.dyn.controller import PlanController
    from cbba_sota.dyn.env import make_env
    from cbba_sota.dyn.executor import Executor, Realization
    from cbba_sota.dyn.methods import make_policy
    from cbba_sota.dyn.perturb import realize_instance
    from cbba_sota.hetero import Instance

    st = get(job["setting"])
    rz = realize_instance(job["setting"], "dev", job["instance"], job["seed"], job["cell"])
    if rz.excluded:
        return {"status": "excluded"}
    path = st.instance_path("dev", job["instance"])
    inst = Instance.from_pickle(path)
    out = {"status": "completed", "family": rz.cell.family}
    trajs, walls, worlds, nonmin = {}, {}, {}, {}
    for world in ("env", "decision", "executor"):
        pol = make_policy(job["method"], job["tier"])
        nonmin[world] = _track_minimality(pol, inst)
        t0 = time.perf_counter()
        try:
            if world == "executor":
                ex = Executor(inst, Realization.from_perturb(rz), detect_h=rz.detect_after)
                res = ex.run(pol)
                trajs[world] = trajectory_executor(ex, res, pol)
                worlds[world] = res["world"]
            else:
                env = make_env(path, rz, coalition="arrival" if world == "env" else "decision")
                ep = env.run_plan(PlanController(pol, inst, rz.kappa()))
                trajs[world] = trajectory_env(env, ep, pol)
                worlds[world] = ep.world
        except Exception as e:
            if world == "env":
                raise
            trajs[world] = {"error": f"{type(e).__name__}: {e}"}
        walls[world] = time.perf_counter() - t0
    ref = trajs["env"]
    out.update(worlds=worlds, wall_s=walls, makespan=ref["makespan"], success=ref["success"],
               n_replans=len(ref["replans"]), n_legs=sum(len(x) for x in ref["legs"]),
               nonminimal=nonmin["env"][0],  # coalitions that are not minimal covers, summed over the env's plans
               makespan_executor=trajs["executor"].get("makespan"),
               makespan_decision=trajs["decision"].get("makespan"))
    for world in ("executor", "decision"):
        t = trajs[world]
        out[world] = compare(ref, t, job["tol"]) if "error" not in t else {
            "match": False, "exact": False, "first": t["error"], "max_diff": None, "counters_differ": {}}
    return out


def _track_minimality(policy, inst) -> list[int]:
    """Count, over every plan ``policy`` returns, the coalitions that are not minimal covers (residual repair keeps
    every frozen member, so a frozen member can become redundant once extras join). D2 (arrival order == decision
    order) is claimed for minimal-cover plans only. Returns a one-element list updated in place."""
    count = [0]
    orig = policy.replan

    def replan(state, scope=None, event_index=None):
        plan = orig(state, scope, event_index)
        for j, m in enumerate(plan.members):
            if m:
                tot = inst.ab[list(m)].sum(axis=0)
                count[0] += any(bool((tot - inst.ab[i] >= inst.req[j]).all()) for i in m)
        return plan

    policy.replan = replan
    return count


def _git() -> dict:
    import hashlib

    def git(*a):
        return subprocess.run(["git", "-C", str(ROOT), *a], capture_output=True, text=True, check=False).stdout.strip()

    from cbba_sota.dyn import executor

    dyn = ROOT / "cbba_sota" / "dyn"
    ctl = hashlib.sha1((dyn / "controller.py").read_bytes()).hexdigest()[:10]
    return {"git": git("rev-parse", "--short=12", "HEAD"),
            "git_dirty_dyn": bool(git("status", "--porcelain", "--", "cbba_sota/dyn")),
            "executor_hash": executor.code_hash(), "controller_hash": ctl}


def _key(r: dict) -> tuple:
    return tuple(r[k] for k in ("setting", "instance", "seed", "cell", "method", "tier"))


def cmd_run(a) -> None:
    from cbba_sota.dyn.env import code_hash

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    chash = code_hash()
    prov = {**_git(), "code_hash": chash}
    world = ("code_hash", "executor_hash", "controller_hash")  # a row counts only for the world code it ran on
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            r = json.loads(line)
            if r.get("status") in ("completed", "excluded") and all(r.get(k) == prov[k] for k in world):
                done.add(_key(r))
    jobs = [{"setting": s, "instance": i, "seed": c, "cell": cell, "method": m, "tier": a.tier, "tol": a.tol}
            for s in a.settings for i in _range(a.instances) for c in _range(a.seeds) for cell in a.cells
            for m in a.methods]
    jobs = [j for j in jobs if _key(j) not in done]
    procs = max(1, min(a.procs, MAX_PROCS))
    print(f"{len(jobs)} cases on {procs} processes (env code {chash})", flush=True)
    t0 = time.time()
    with mp.get_context("spawn").Pool(procs, initializer=_threads) as pool, out.open("a") as f:
        for k, r in enumerate(pool.imap_unordered(run_case, jobs)):
            r.update(prov)
            f.write(json.dumps(r) + "\n")
            f.flush()
            bad = r.get("status") == "error" or (r.get("status") == "completed" and not r["executor"]["match"])
            if bad or (k + 1) % 50 == 0:
                print(f"{k + 1}/{len(jobs)} {time.time() - t0:.0f}s", _key(r), r.get("error") or
                      (r.get("executor") or {}).get("first"), flush=True)
    print(f"done in {time.time() - t0:.0f}s", flush=True)


def _range(spec: list[str]) -> list[int]:
    out: list[int] = []
    for part in spec:
        for chunk in str(part).split(","):
            if "-" in chunk:
                lo, hi = chunk.split("-")
                out += range(int(lo), int(hi) + 1)
            elif chunk:
                out.append(int(chunk))
    return sorted(set(out))


def cmd_summary(a) -> None:
    import numpy as np

    from cbba_sota.dyn.env import code_hash

    chash = code_hash()
    prov = _git()
    rows = [json.loads(line) for line in Path(a.out).read_text().splitlines()]
    rows = [r for r in rows if a.any_code or (r.get("code_hash") == chash and r.get("executor_hash") ==
                                                 prov["executor_hash"] and r.get("controller_hash") ==
                                                 prov["controller_hash"])]
    rows = list({_key(r): r for r in rows}.values())  # the latest row of each case (a rerun supersedes an error)
    errors = [r for r in rows if r.get("status") == "error"]
    rows = [r for r in rows if r.get("status") == "completed"]
    print(f"env code {chash}, executor {prov['executor_hash']}, controller {prov['controller_hash']}: "
          f"{len(rows)} compared cases, {len(errors)} errors")
    for r in errors[:5]:
        print("  ERROR", _key(r), r["error"])
    fams = sorted({r["family"] for r in rows}, key=lambda f: (f != "static", f))
    print("worlds:", sorted({w for r in rows for w in r["worlds"].values()}))
    views = [("executor", "executor vs env", lambda r: True),
             ("decision", "decision vs arrival order, episodes whose plans are all minimal covers (D2 claim)",
              lambda r: r["nonminimal"] == 0),
             ("decision", ("decision vs arrival order, episodes with non-minimal coalitions (residual repair; "
                           "descriptive)"), lambda r: r["nonminimal"] > 0)]
    for other, title, keep in views:
        print(f"\n== {title} (tolerance {a.tol:g}): cases matched / cases, bit-identical, max |dt|, "
              f"counter differences")
        for fam in fams + ["ALL"]:
            rr = [r for r in rows if (fam == "ALL" or r["family"] == fam) and keep(r)]
            m = [r for r in rr if r[other]["match"]]
            ex = [r for r in rr if r[other]["exact"]]
            md = max((r[other]["max_diff"] for r in m), default=0.0)
            cd = sum(bool(r[other]["counters_differ"]) for r in rr)
            print(f"  {fam:9s} {len(m):4d} / {len(rr):4d}   exact {len(ex):4d}   max |dt| {md:.3g}   "
                  f"counters differ in {cd}")
        for r in [r for r in rows if keep(r) and not r[other]["match"]][:a.show]:
            print("   diverged:", _key(r), r[other]["first"])
    by_m = sorted({(r["method"], r["tier"]) for r in rows})
    print("\nexecutor vs env by planner: matched / cases (bit-identical)")
    for m, tier in by_m:
        rr = [r for r in rows if (r["method"], r["tier"]) == (m, tier)]
        print(f"  {m:12s} {tier:5s} {sum(r['executor']['match'] for r in rr):5d} / {len(rr):5d} "
              f"({sum(r['executor']['exact'] for r in rr)})")
    print("\nK0 verdict per family (spec 8: >= 20 episodes per family, executor == env to the tolerance for "
          "minimal covers; decision == arrival order at ideal comms)")
    ok_all = True
    for fam in fams:
        rr = [r for r in rows if r["family"] == fam]
        mc = [r for r in rr if r["nonminimal"] == 0]
        ex_mc = all(r["executor"]["match"] for r in mc)
        ex_all = all(r["executor"]["match"] for r in rr)
        d2 = all(r["decision"]["match"] for r in mc)
        ok = len(mc) >= 20 and ex_mc and d2
        ok_all &= ok
        print(f"  {fam:7s} {'PASS' if ok else 'FAIL'}: minimal-cover episodes {len(mc)} (executor "
              f"{sum(r['executor']['match'] for r in mc)}, decision {sum(r['decision']['match'] for r in mc)}); "
              f"all episodes {len(rr)} (executor {sum(r['executor']['match'] for r in rr)}"
              f"{'' if ex_all else ', NOT all'})")
    print(f"  K0 {'PASS' if ok_all else 'FAIL'} on env {chash}")
    by_m = sorted({r["method"] for r in rows})
    print("\nwall time per episode (median s): env / executor, by method")
    for m in by_m:
        rr = [r for r in rows if r["method"] == m]
        we = np.median([r["wall_s"]["env"] for r in rr])
        wx = np.median([r["wall_s"]["executor"] for r in rr])
        print(f"  {m:12s} {we:.3f} / {wx:.3f}  (x{we / wx:.1f})  n={len(rr)}")


_NET = None


def _rl_net():
    global _NET
    if _NET is None:
        import torch

        from cbba_sota.solvers import rl

        torch.set_num_threads(1)
        _NET = rl.load_policy("cpu")
    return _NET


def rl_job(job: dict) -> dict:
    """One check (iv) job: see the module docstring."""
    _threads()
    from cbba_sota.bench.configs import get
    from cbba_sota.bench.heteromrta import load_env
    from cbba_sota.dyn.env import RLPolicy, make_env
    from cbba_sota.solvers import rl

    st = get(job["setting"])
    path = st.instance_path("dev", job["instance"])
    net = _rl_net()
    out = {k: job[k] for k in ("setting", "instance", "kind", "k")}
    if job["kind"] == "runner":
        sys.path.insert(0, str(ROOT / "scripts"))
        import trackD_run

        rows = []
        for method in ("RL(g.)", "RL(s.1)"):
            case = {"setting": job["setting"], "split": "dev", "instance": job["instance"], "seed": 0,
                    "cell": "static", "family": "static", "method": method, "tier": "native",
                    "options": {"coalition": "auto", "obs": "causal", "lease_L": 200.0, "abandon_on_failure": True}}
            case["case_id"] = trackD_run.case_id(case)
            row = trackD_run.run_case(case, {"git": "", "git_dirty_dyn": None})
            ref = rl.rollout(load_env(path), net, sample=method == "RL(s.1)", seed=row["method_seed"])
            rows.append({"method": method, "seed": row["method_seed"], "env": row.get("makespan"),
                         "ref": ref.makespan if ref.success else None, "world": row.get("world"),
                         "status": row["status"], "error": row.get("error")})
        out["runner"] = rows
        out["match"] = all(r["status"] == "completed" and r["env"] == r["ref"] for r in rows)
        return out
    seed = rl.sample_seed(st.seed("dev", job["instance"]), job["k"])
    env = make_env(path)
    ep = env.run_policy(RLPolicy(net, sample=job["kind"] == "sample", seed=seed), shuffle_seed=seed)
    if job["kind"] == "sample":
        ref = None  # compared per instance as a multiset (cmd_rl)
    else:
        r = rl.rollout(load_env(path), net, sample=False, seed=seed)
        ref = r.makespan if r.success else None
    out.update(seed=seed, env=ep.makespan if ep.success else None, ref=ref, world=ep.world)
    out["match"] = out["env"] == ref if job["kind"] != "sample" else None
    return out


def cmd_rl(a) -> None:
    from cbba_sota.dyn.env import code_hash
    from cbba_sota.dyn.perturb import PRIMARY_SETTINGS

    jobs, stored = [], {}
    for s in PRIMARY_SETTINGS:
        rows = {r["instance"]: r for r in map(json.loads, (ROOT / "runs" / "rl" / s / "dev.jsonl").open())
                if r["method"] == "RL(s.64)"}
        for i in _range(a.instances):
            stored[(s, i)] = rows[i]
            jobs.append({"setting": s, "instance": i, "kind": "greedy", "k": 0})
            jobs.append({"setting": s, "instance": i, "kind": "runner", "k": 0})
            for k in range(len(rows[i]["sample_makespans"])):
                jobs.append({"setting": s, "instance": i, "kind": "sample", "k": k})
    procs = max(1, min(a.procs, MAX_PROCS))
    print(f"{len(jobs)} jobs on {procs} processes (env code {code_hash()})", flush=True)
    t0 = time.time()
    with mp.get_context("spawn").Pool(procs, initializer=_threads) as pool:
        res = pool.map(rl_job, jobs, chunksize=4)
    out = Path(a.out).parent / "rl_static.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for r in res:
            f.write(json.dumps({**r, "code_hash": code_hash()}) + "\n")
    n_ok = 0
    for (s, i), row in stored.items():
        got = [r["env"] for r in res if r["kind"] == "sample" and (r["setting"], r["instance"]) == (s, i)]
        ok = sorted(got, key=lambda x: (x is None, x)) == sorted(row["sample_makespans"]) and \
            min(x for x in got if x is not None) == row["makespan"]
        n_ok += ok
        if not ok:
            print(f"   stored RL(s.64) not reproduced: {s} {i}")
    print(f"sample : {n_ok} / {len(stored)} instances with the 64 stored RL(s.64) sample makespans and best "
          f"reproduced bit for bit ({sum(r['kind'] == 'sample' for r in res)} rollouts)")
    for kind in ("greedy", "runner"):
        rr = [r for r in res if r["kind"] == kind]
        bad = [r for r in rr if not r["match"]]
        print(f"{kind:7s}: {len(rr) - len(bad)} / {len(rr)} bit-identical", *(f"\n   {r}" for r in bad[:5]))
    worlds = sorted({r.get("world") for r in res if r.get("world")} |
                    {x["world"] for r in res for x in r.get("runner", ()) if x.get("world")})
    print("worlds:", worlds, f"({time.time() - t0:.0f}s) ->", out)


def main(argv=None) -> None:
    from cbba_sota.dyn.methods import PLANNERS
    from cbba_sota.dyn.perturb import PRIMARY_SETTINGS

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("run", "summary", "rl"))
    ap.add_argument("--settings", nargs="+", default=list(PRIMARY_SETTINGS))
    ap.add_argument("--instances", nargs="+", default=["0-4"])
    ap.add_argument("--seeds", nargs="+", default=["0"])
    ap.add_argument("--cells", nargs="+", default=FAMILY_CELLS)
    ap.add_argument("--methods", nargs="+", default=["SPARC"], choices=PLANNERS)
    ap.add_argument("--tier", default="light", choices=["light", "heavy"])
    ap.add_argument("--tol", type=float, default=1e-6)
    ap.add_argument("--procs", type=int, default=12)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--show", type=int, default=10, help="diverged cases to list in the summary")
    ap.add_argument("--any-code", action="store_true", help="summary: include rows from older env code")
    a = ap.parse_args(argv)
    if a.cmd == "rl" and a.instances == ["0-4"]:
        a.instances = ["0-19"]
    {"run": cmd_run, "summary": cmd_summary, "rl": cmd_rl}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
