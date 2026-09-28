"""Track D pilot: the released HeteroMRTA policy run online under task release and execution noise.

Usage (repo root): .venv/bin/python pilots/trackD/run_pilot.py [--procs 16] [--seeds 3] [--settings ...]
Writes pilots/trackD/out/rows.jsonl (resumable) and prints a summary (see summarize.py).

Methods (all see the same realization per (instance, noise seed, release seed)):
  RL(g.)   released checkpoint, greedy decode, run online in DynTaskEnv (one rollout: best-of-N is not an online
           policy once the world is stochastic)
  RL(s.1)  one sampled rollout (the stochastic policy as deployed)
  greedy   paper greedy (nearest contributable task, env masks, max-open-task rule), online
  ALNS-ol  ALNS plan on the NOMINAL static instance (5 s, 1 core), executed open loop (pre_set_route) under the
           noise; only for scenarios without release
  ALNS-cv  clairvoyant: ALNS on the realized durations (only scenarios without delays and release) = the value of
           perfect information about durations
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

from dyn_env import Scenario  # noqa: E402

MAX_TIME = 200.0
# release horizon = half the paper's RL(g.) mean makespan of the setting (Table III)
HORIZON = {"MA-AT-25-5-50": 25.0, "MA-AT-50-5-50": 15.0, "SA-AT-50-5-50": 20.0, "SA-BT-50-5-50": 12.0}

LORR = dict(p_delay=0.01, min_delay=1, max_delay=4)  # Start-Kit 2026 example config
MOD = dict(p_delay=0.05, min_delay=1, max_delay=10)
SEV = dict(p_delay=0.05, min_delay=10, max_delay=50)


def scenarios(horizon: float) -> list[Scenario]:
    out = [Scenario("static")]
    for obs in ("oracle", "causal"):
        out += [Scenario(f"N1-lorr/{obs}", obs=obs, **LORR),
                Scenario(f"N2-dur.3/{obs}", obs=obs, dur_sigma=0.3),
                Scenario(f"N3-mod/{obs}", obs=obs, dur_sigma=0.3, **MOD),
                Scenario(f"N4-sev/{obs}", obs=obs, dur_sigma=0.6, **SEV)]
    out += [Scenario("R1-batch", release="batch"),
            Scenario("R2-pois.5", release="poisson", dod=0.5, horizon=horizon),
            Scenario("R3-pois.8", release="poisson", dod=0.8, horizon=horizon)]
    for obs in ("oracle", "causal"):
        out += [Scenario(f"R1+N3/{obs}", release="batch", obs=obs, dur_sigma=0.3, **MOD),
                Scenario(f"R2+N3/{obs}", release="poisson", dod=0.5, horizon=horizon, obs=obs, dur_sigma=0.3, **MOD)]
    return out


_net = None


def _init():
    global _net
    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    from cbba_sota.solvers import rl

    _net = rl.load_policy("cpu")


def _instrumented_rl(env, sample: bool, seed: int):
    """rl.rollout with per-decision timing (same control flow as rl._episode / Worker.run_episode)."""
    import random

    import torch

    from cbba_sota.solvers.rl import _choose

    env.init_state()
    gen = torch.Generator("cpu").manual_seed(seed)
    rng = random.Random(seed)
    tasks = env.task_dic.values()
    t_obs = t_net = 0.0
    n_dec = step = 0
    while not env.finished and env.current_time < MAX_TIME:
        released, env.current_time = env.next_decision()
        rng.shuffle(released[0])
        for aid in released[0] + released[1]:
            a = time.perf_counter()
            task_obs, agent_obs, mask = env.agent_observe(aid, False)
            b = time.perf_counter()
            t_obs += b - a
            if mask[0, 1:].all():
                if not all(t["feasible_assignment"] for t in tasks):
                    env.agent_dic[aid]["no_choice"] = True
                    continue
                if env.agent_dic[aid]["current_task"] < 0:
                    continue
            with torch.no_grad():
                x = [torch.as_tensor(v, dtype=torch.float32) for v in (task_obs, agent_obs, mask)]
                probs, _ = _net(*x, torch.tensor([[[aid]]]))
                action = _choose(probs, sample, gen).item()
            t_net += time.perf_counter() - b
            n_dec += 1
            env.agent_step(aid, action, step)
        env.finished = env.check_finished()
        step += 1
    return {"t_obs": t_obs, "t_net": t_net, "n_dec": n_dec}


def _summary(env) -> dict:
    env.calculate_waiting_time()
    fin = [bool(t["finished"]) for t in env.task_dic.values()]
    ms = float(env.current_time)
    success = bool(env.finished) and all(fin) and ms < MAX_TIME
    return {"makespan": ms if np.isfinite(ms) else MAX_TIME, "success": success, "completion": float(np.mean(fin)),
            "awt": float(np.mean([a["sum_waiting_time"] for a in env.agent_dic.values()])),
            "release_polls": int(getattr(env, "n_release_polls", 0)),
            "depot_revisits": int(sum(sum(t < 0 for t in a["route"][1:-1]) for a in env.agent_dic.values()))}


def run_job(job: dict) -> list[dict]:
    from cbba_sota.solvers import greedy
    from dyn_env import make_dyn_env

    sc = Scenario(**job["sc"])
    key = dict(setting=job["setting"], inst=job["inst"], scenario=sc.name, noise_seed=job["noise_seed"])
    rows = []

    def mk():
        for attempt in range(5):  # transient read failures were seen on the shared filesystem
            try:
                return make_dyn_env(job["path"], sc, inst_key=job["inst_key"], noise_seed=job["noise_seed"],
                                    release_seed=job["noise_seed"])
            except OSError:
                if attempt == 4:
                    raise
                time.sleep(2.0)

    for method, sample, seed in (("RL(g.)", False, 0), ("RL(s.1)", True, 1000 + job["noise_seed"])):
        env = mk()
        t0 = time.perf_counter()
        tim = _instrumented_rl(env, sample, seed)
        rows.append({**key, "method": method, **_summary(env), "wall_s": time.perf_counter() - t0, **tim})
    env = mk()
    t0 = time.perf_counter()
    greedy._run_nearest(env, True, "wait")
    rows.append({**key, "method": "greedy", **_summary(env), "wall_s": time.perf_counter() - t0})
    if sc.release == "none" and job.get("plan_routes") is not None:
        env = mk()
        env.init_state()
        for i, r in enumerate(job["plan_routes"]):
            env.pre_set_route(list(r), i)
        t0 = time.perf_counter()
        with contextlib.redirect_stdout(io.StringIO()):
            env.execute_by_route()
        rows.append({**key, "method": "ALNS-ol", **_summary(env), "wall_s": time.perf_counter() - t0,
                     "plan_nominal": job["plan_ms"]})
    if sc.release == "none" and sc.p_delay == 0 and sc.dur_sigma > 0 and sc.obs == "oracle" and job.get("clairvoyant"):
        from cbba_sota.hetero import Instance, evaluate
        from cbba_sota.solvers.alns import solve

        env = mk()
        inst = Instance.from_env(env)  # task['time'] = realized durations
        plan, st = solve(inst, job["alns_s"], seed=0)
        env.init_state()
        for i, r in enumerate(plan.to_env_routes()):
            env.pre_set_route(list(r), i)
        with contextlib.redirect_stdout(io.StringIO()):
            env.execute_by_route()
        s = _summary(env)
        assert abs(s["makespan"] - evaluate(inst, plan).makespan) < 1e-9
        rows.append({**key, "method": "ALNS-cv", **s, "wall_s": st.time_s})
    return rows


def plan_job(args):
    path, alns_s = args
    from cbba_sota.hetero import Instance
    from cbba_sota.solvers.alns import solve

    inst = Instance.from_pickle(path)
    plan, st = solve(inst, alns_s, seed=0)
    return path, plan.to_env_routes(), st.makespan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--settings", nargs="+", default=["MA-AT-25-5-50", "MA-AT-50-5-50"])
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--procs", type=int, default=16)
    ap.add_argument("--alns_s", type=float, default=5.0)
    ap.add_argument("--out", default=str(HERE / "out" / "rows.jsonl"))
    args = ap.parse_args()
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ[k] = "1"
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            r = json.loads(line)
            done.add((r["setting"], r["inst"], r["scenario"], r["noise_seed"]))
    from cbba_sota.bench.configs import get

    ctx = mp.get_context("spawn")
    # nominal ALNS plans (cached)
    plan_file = out.parent / "alns_plans.json"
    plans = json.loads(plan_file.read_text()) if plan_file.exists() else {}
    paths = [str(get(s).instance_path("dev", i)) for s in args.settings for i in range(args.n)]
    todo = [p for p in paths if p not in plans]
    if todo:
        with ctx.Pool(args.procs) as pool:
            for p, routes, ms in pool.imap_unordered(plan_job, [(p, args.alns_s) for p in todo]):
                plans[p] = {"routes": routes, "makespan": ms}
                plan_file.write_text(json.dumps(plans))
    jobs = []
    for s in args.settings:
        st = get(s)
        for i in range(args.n):
            path = str(st.instance_path("dev", i))
            for sc in scenarios(HORIZON.get(s, 20.0)):
                seeds = [0] if sc.name == "static" else range(args.seeds)
                for ns in seeds:
                    if (s, i, sc.name, ns) in done:
                        continue
                    jobs.append({"setting": s, "inst": i, "path": path, "inst_key": st.seed("dev", i),
                                 "sc": sc.__dict__, "noise_seed": ns, "plan_routes": plans[path]["routes"],
                                 "plan_ms": plans[path]["makespan"], "clairvoyant": True, "alns_s": args.alns_s})
    print(f"{len(jobs)} jobs", flush=True)
    t0 = time.time()
    with ctx.Pool(args.procs, initializer=_init) as pool, out.open("a") as f:
        for k, rows in enumerate(pool.imap_unordered(run_job, jobs)):
            for r in rows:
                f.write(json.dumps(r) + "\n")
            f.flush()
            if k % 50 == 0:
                print(f"{k}/{len(jobs)} {time.time() - t0:.0f}s", flush=True)
    print("done", time.time() - t0)


if __name__ == "__main__":
    main()
