"""Centralized coalition-LNS headroom pilot on HeteroMRTA's shipped RALTestSet (50 instances).

Semantics copied from HeteroMRTA env/task_env.py: additive integer requirements, binary agent
abilities, a task starts when the members' summed abilities cover the requirement (start = last
member arrival), fixed duration, velocity 0.2, Euclidean travel, every agent returns to its species
depot, makespan = last return. Plans are routes consistent with one global task order (keys), which
makes the plan graph acyclic (no deadlock) by construction.

This is a feasibility pilot for the headroom of optimization over the published RL policy,
not the proposed decentralized algorithm.
"""
import glob
import math
import pickle
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, "HeteroMRTA")
from env.task_env import TaskEnv  # noqa: E402,F401  (pickles reference __main__.TaskEnv)
V = 0.2


def load(path):
    env = pickle.load(open(path, "rb"))
    T = len(env.task_dic)
    req = np.array([np.asarray(env.task_dic[j]["requirements"], float) for j in range(T)])
    loc = np.array([env.task_dic[j]["location"] for j in range(T)], float)
    dur = np.array([env.task_dic[j]["time"] for j in range(T)], float)
    A = len(env.agent_dic)
    ab = np.array([np.asarray(env.agent_dic[i]["abilities"], float) for i in range(A)])
    dep = np.array([env.agent_dic[i]["depot"] for i in range(A)], float)
    return env, req, loc, dur, ab, dep


class Inst:
    def __init__(self, req, loc, dur, ab, dep):
        self.req, self.loc, self.dur, self.ab, self.dep = req, loc, dur, ab, dep
        self.T, self.A = len(req), len(ab)
        self.tt = np.linalg.norm(loc[:, None] - loc[None], axis=2) / V
        self.dt = np.linalg.norm(dep[:, None] - loc[None], axis=2) / V  # agent x task


def evaluate(I, members, key):
    """members: list of sets (task -> agents); key: task order. Returns (makespan, start, routes)."""
    order = sorted([t for t in range(I.T) if members[t]], key=lambda j: key[j])
    free_t = np.zeros(I.A)
    pos = [None] * I.A  # last task index
    start = np.zeros(I.T)
    for j in order:
        arr = []
        for i in members[j]:
            p = pos[i]
            a = free_t[i] + (I.dt[i, j] if p is None else I.tt[p, j])
            arr.append(a)
        s = max(arr)
        start[j] = s
        f = s + I.dur[j]
        for i in members[j]:
            free_t[i] = f
            pos[i] = j
    ms = 0.0
    for i in range(I.A):
        if pos[i] is not None:
            ms = max(ms, free_t[i] + I.dt[i, pos[i]])
    return ms, start


def covering(I, j, cand_arrivals, rng, noise=0.0):
    """Greedy cover of task j's requirement by agents sorted by (arrival, -useful ability)."""
    need = I.req[j].copy()
    chosen = []
    order = sorted(range(I.A), key=lambda i: cand_arrivals[i] * (1 + noise * rng.random()))
    for i in order:
        if (need <= 0).all():
            break
        gain = np.minimum(np.maximum(need, 0), I.ab[i]).sum()
        if gain > 0:
            chosen.append(i)
            need = need - I.ab[i]
    return chosen if (need <= 0).all() else None


def arrivals_if_inserted(I, members, key, j, kval):
    """Arrival time of each agent at task j if j is inserted at global key kval (ignores downstream)."""
    order = sorted([t for t in range(I.T) if t != j and members[t]], key=lambda t: key[t])
    free_t = np.zeros(I.A)
    pos = [None] * I.A
    arr_j = np.full(I.A, np.inf)
    done = np.zeros(I.A, bool)
    for t in order + [None]:
        if t is None or key[t] > kval:
            for i in range(I.A):
                if not done[i]:
                    arr_j[i] = free_t[i] + (I.dt[i, j] if pos[i] is None else I.tt[pos[i], j])
                    done[i] = True
            if t is None:
                break
        arr = [free_t[i] + (I.dt[i, t] if pos[i] is None else I.tt[pos[i], t]) for i in members[t]]
        f = max(arr) + I.dur[t]
        for i in members[t]:
            free_t[i] = f
            pos[i] = t
    return arr_j


def insert_best(I, members, key, j, rng, n_slots=8, noise=0.3):
    ks = sorted(key[t] for t in range(I.T) if t != j and members[t])
    cands = [(ks[k - 1] + ks[k]) / 2 for k in range(1, len(ks))] if len(ks) > 1 else []
    cands = [(ks[0] - 1.0) if ks else 0.0] + cands + [(ks[-1] + 1.0) if ks else 1.0]
    if len(cands) > n_slots:
        cands = rng.sample(cands, n_slots)
    best = None
    for kval in cands:
        arr = arrivals_if_inserted(I, members, key, j, kval)
        for trial in range(2):
            ch = covering(I, j, arr, rng, noise if trial else 0.0)
            if ch is None:
                continue
            members[j] = set(ch)
            old = key[j]
            key[j] = kval
            ms, _ = evaluate(I, members, key)
            key[j] = old
            sc = (ms, sum(arr[i] for i in ch))
            if best is None or sc < best[0]:
                best = (sc, set(ch), kval)
    members[j] = best[1]
    key[j] = best[2]


def construct(I, rng):
    members = [set() for _ in range(I.T)]
    key = [0.0] * I.T
    for j in sorted(range(I.T), key=lambda _: rng.random()):
        insert_best(I, members, key, j, rng, n_slots=6)
    return members, key


def lns(I, seconds, seed):
    rng = random.Random(seed)
    members, key = construct(I, rng)
    cur, _ = evaluate(I, members, key)
    best = (cur, [set(m) for m in members], list(key))
    t0 = time.time()
    temp0 = 0.02 * cur
    it = 0
    while time.time() - t0 < seconds:
        it += 1
        frac = (time.time() - t0) / seconds
        temp = temp0 * (1 - frac) + 1e-9
        M = [set(m) for m in members]
        K = list(key)
        q = rng.randint(1, max(2, I.T // 5))
        if rng.random() < 0.5:
            rem = rng.sample(range(I.T), q)
        else:
            s0 = rng.randrange(I.T)
            rem = list(np.argsort(I.tt[s0])[:q])
        for j in rem:
            M[j] = set()
        for j in sorted(rem, key=lambda _: rng.random()):
            insert_best(I, M, K, j, rng)
        # renormalise keys to schedule start times (keeps order consistent)
        ms, st = evaluate(I, M, K)
        if ms < cur or rng.random() < math.exp(-(ms - cur) / temp):
            members, key, cur = M, K, ms
            if ms < best[0]:
                best = (ms, [set(m) for m in M], list(K))
    return best[0], it


def job(args):
    path, secs = args
    env, req, loc, dur, ab, dep = load(path)
    I = Inst(req, loc, dur, ab, dep)
    g_env = pickle.load(open(path, "rb"))
    g_env.init_state()
    import io, contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        g_env.execute_greedy_action("./", "greedy", False)
    _, fin = g_env.get_episode_reward(100)
    greedy = g_env.current_time if np.all(fin) else float("nan")
    ms, it = lns(I, secs, 0)
    return path, greedy, ms, it, I.A, I.T


if __name__ == "__main__":
    secs = float(sys.argv[1]) if len(sys.argv) > 1 else 10
    files = sorted(glob.glob("HeteroMRTA/RALTestSet/env_*.pkl"))
    res = []
    with ProcessPoolExecutor(50) as ex:
        for r in ex.map(job, [(f, secs) for f in files]):
            res.append(r)
    g = np.array([r[1] for r in res])
    l = np.array([r[2] for r in res])
    print(f"n={len(res)} agents={res[0][4]} tasks={res[0][5]} lns_secs={secs} mean_iters={np.mean([r[3] for r in res]):.0f}")
    print(f"HeteroMRTA Greedy (their code) mean makespan = {np.nanmean(g):.3f}  (finished {np.sum(~np.isnan(g))}/{len(g)})")
    print(f"centralized coalition-LNS pilot mean makespan = {l.mean():.3f}")
    print(f"LNS/Greedy ratio (paired mean) = {np.nanmean(l / g):.3f}")
