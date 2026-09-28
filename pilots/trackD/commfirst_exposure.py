"""Comm-first pre-pilot: how partitioned is a HeteroMRTA team under a disk (range-R) network?

For each dev instance we execute (i) the cached nominal ALNS plan and (ii) the released RL policy, greedy decode,
in the static env, reconstruct every robot's piecewise-linear trajectory from its route and arrival times
(departure = arrival - nominal travel, exact without noise), and on a time grid compute the multi-hop disk graph
among robots plus a station at the square centre. Reported per range R:

  st_conn    mean fraction of robots in the station's component (multi-hop relay allowed)
  pair_conn  mean fraction of robot pairs that share a component (what gossip can reach eventually)
  deg        mean one-hop degree
  lcc        mean fraction of robots in the largest component (CC-OPI reports 15% at its representative radius)
  coal_conn  for tasks with >= 2 members: fraction of member pairs connected at the first member's departure
  cov_rel    Poisson-release proxy: at 50 random times in [0, H], for a random task location, does the component
             of the nearest robot within R (discovery) cover the task's requirement? (0 if nobody is within R)

Usage: .venv/bin/python pilots/trackD/commfirst_exposure.py [--n 20] [--procs 16]
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
RADII = (0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4)
SETTINGS = ("MA-AT-25-5-50", "MA-AT-50-5-50", "SA-AT-50-5-50", "SA-BT-50-5-50")
HORIZON = {"MA-AT-25-5-50": 25.0, "MA-AT-50-5-50": 15.0, "SA-AT-50-5-50": 20.0, "SA-BT-50-5-50": 12.0}
STATION = np.array([0.5, 0.5])


def _loc(env, node):
    return env.task_dic[node]["location"] if node >= 0 else env.depot_dic[-node - 1]["location"]


def trajectories(env):
    """Per agent a list of legs (dep, arr, p0, p1); the robot is at p1 from arr until the next dep."""
    legs = []
    for a in env.agent_dic.values():
        route, arr = a["route"], a["arrival_time"]
        out, prev = [], np.asarray(a["depot"], float)
        for node, t_arr in zip(route[1:], arr[1:]):
            p1 = np.asarray(_loc(env, node), float)
            tau = float(np.linalg.norm(p1 - prev)) / a["velocity"]
            out.append((t_arr - tau, t_arr, prev, p1))
            prev = p1
        legs.append((np.asarray(a["depot"], float), out))
    return legs


def positions(legs, t):
    P = np.empty((len(legs), 2))
    for i, (p, out) in enumerate(legs):
        pos = p
        for dep, arr, p0, p1 in out:
            if t < dep:
                break
            if t <= arr:
                pos = p0 + (p1 - p0) * ((t - dep) / max(arr - dep, 1e-12))
                break
            pos = p1
        P[i] = pos
    return P


def components(P, R):
    from scipy.sparse.csgraph import connected_components

    D = np.linalg.norm(P[:, None] - P[None], axis=-1)
    _, lab = connected_components(D <= R, directed=False)
    return lab, D


def analyse(env, legs, makespan, horizon, rng):
    A = len(legs)
    ab = np.array([a["abilities"] for a in env.agent_dic.values()], float)
    ts = np.arange(0.0, makespan, 0.25)
    res = {R: {"st_conn": [], "pair_conn": [], "deg": [], "lcc": [], "coal_conn": [], "cov_rel": []} for R in RADII}
    for t in ts:
        P = positions(legs, t)
        Pst = np.vstack([P, STATION])
        for R in RADII:
            lab, D = components(Pst, R)
            res[R]["st_conn"].append(float(np.mean(lab[:A] == lab[A])))
            la = lab[:A]
            same = (la[:, None] == la[None]).sum() - A
            res[R]["pair_conn"].append(same / (A * (A - 1)))
            res[R]["deg"].append(float(((D[:A, :A] <= R).sum() - A) / A))
            res[R]["lcc"].append(float(np.bincount(la).max() / A))
    # coalition connectivity at the first member's departure towards the task
    for j, task in env.task_dic.items():
        mem = list(task["members"])
        if len(mem) < 2:
            continue
        deps = []
        for i in mem:
            route, arr = env.agent_dic[i]["route"], env.agent_dic[i]["arrival_time"]
            k = max(idx for idx, n in enumerate(route) if n == j)
            deps.append(legs[i][1][k - 1][0])
        P = positions(legs, min(deps))
        for R in RADII:
            lab, _ = components(P, R)
            lm = lab[mem]
            res[R]["coal_conn"].append(float(np.mean(lm[:, None] == lm[None])))
    # discovery proxy for online release
    tasks = list(env.task_dic.values())
    for _ in range(50):
        t = rng.uniform(0, horizon)
        task = tasks[rng.integers(len(tasks))]
        P = positions(legs, t)
        d = np.linalg.norm(P - task["location"], axis=1)
        for R in RADII:
            if d.min() > R:
                res[R]["cov_rel"].append(0.0)
                continue
            lab, _ = components(P, R)
            comp = lab == lab[int(np.argmin(d))]
            res[R]["cov_rel"].append(float(np.all(ab[comp].sum(0) >= task["requirements"])))
    return {str(R): {k: float(np.mean(v)) if v else float("nan") for k, v in m.items()} for R, m in res.items()}


_net = None


def job(args):
    setting, i, path, routes = args
    import torch

    torch.set_num_threads(1)
    from cbba_sota.solvers import rl
    from dyn_env import Scenario, make_dyn_env

    global _net
    if _net is None:
        _net = rl.load_policy("cpu")
    rng = np.random.default_rng(1000 + i)
    out = []
    env = make_dyn_env(path, Scenario("static"), inst_key=i)
    env.init_state()
    for k, r in enumerate(routes):
        env.pre_set_route(list(r), k)
    with contextlib.redirect_stdout(io.StringIO()):
        env.execute_by_route()
    out.append({"setting": setting, "inst": i, "method": "ALNS", "makespan": float(env.current_time),
                **analyse(env, trajectories(env), float(env.current_time), HORIZON[setting], rng)})
    env = make_dyn_env(path, Scenario("static"), inst_key=i)
    res = rl.rollout(env, _net, sample=False, seed=0)
    out.append({"setting": setting, "inst": i, "method": "RL(g.)", "makespan": res.makespan,
                **analyse(env, trajectories(env), res.makespan, HORIZON[setting], np.random.default_rng(1000 + i))})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--procs", type=int, default=16)
    args = ap.parse_args()
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ[k] = "1"
    plans = json.loads((HERE / "out" / "alns_plans.json").read_text())
    jobs = []
    for s in SETTINGS:
        for i in range(args.n):
            path = next(p for p in plans if f"/{s}/dev/env_{i}.pkl" in p)
            jobs.append((s, i, path, plans[path]["routes"]))
    rows = []
    with mp.get_context("spawn").Pool(args.procs) as pool:
        for r in pool.imap_unordered(job, jobs):
            rows += r
    (HERE / "out" / "commfirst_exposure.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    keys = ("st_conn", "pair_conn", "deg", "lcc", "coal_conn", "cov_rel")
    for s in SETTINGS:
        for m in ("ALNS", "RL(g.)"):
            sub = [r for r in rows if r["setting"] == s and r["method"] == m]
            print(f"{s:15s} {m:7s} makespan {np.mean([r['makespan'] for r in sub]):6.2f}")
            for R in RADII:
                vals = "  ".join(f"{k} {np.nanmean([r[str(R)][k] for r in sub]):.3f}" for k in keys)
                print(f"    R={R:<5} {vals}")


if __name__ == "__main__":
    main()
