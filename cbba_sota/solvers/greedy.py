"""Greedy baselines for HeteroMRTA.

Env-loop greedies (ground truth is the env's own execution):
- ``greedy_repo``: the shipped ``TaskEnv.execute_greedy_action`` unchanged. It never updates ``dist`` inside its
  argmin loop, so every released agent takes the *last* selectable index (the highest task id it may choose).
- ``greedy_nearest``: the rule the paper describes (RA-L 2025, Sec. V-A): each released agent takes the closest
  task it can contribute to, "with the same decision constraints as Section IV-B": the env's masks (unfinished
  tasks, ability mask, maximum-open-task rule via ``max_waiting=True``) and the decision order of the official
  ``Worker.run_episode``: an agent with nothing selectable postpones (``no_choice``) and is released again at the
  next decision step; it heads home only once every task is assigned. ``idle="depot"`` instead keeps the repo
  greedy's loop, where such an agent returns to its depot (the only index left). The depot never enters the argmin:
  every agent would pick it at time 0 (distance 0) and the loop would never advance.

Plan-level constructors (fast warm starts for ALNS and CP-SAT; key-ordered minimal covers, so ``evaluate`` equals
the env replay):
- ``dispatch``: list scheduling; repeatedly append the task whose earliest-arrival cover starts first.
- ``insertion``: tasks in random or regret order, each inserted at the global position and with the
  earliest-arrival cover that minimise (makespan, summed depot returns).
"""
from __future__ import annotations

import contextlib
import io
import time
from dataclasses import dataclass

import numpy as np
from numba import njit

from cbba_sota.bench.configs import MAX_TIME
from cbba_sota.bench.heteromrta import TaskEnv
from cbba_sota.hetero.evaluate import evaluate
from cbba_sota.hetero.instance import Instance
from cbba_sota.hetero.plan import Plan
from cbba_sota.hetero.replay import Source, make_env, succeeded

MAX_OPEN = 5  # agent_observe(max_waiting=True): with more open tasks, agents may only join open ones


@dataclass
class EnvResult:
    makespan: float  # env current_time at the end
    success: bool  # every task finished and makespan < MAX_TIME (``succeeded``)
    awt: float
    routes: list[list[int]]  # per agent, 0-based tasks in visiting order (depot visits dropped)
    depot_revisits: int  # intermediate depot visits dropped from ``routes``
    wall_s: float
    env_finished: bool = False  # the env's raw ``finished`` flag


# --- env-loop greedies -----------------------------------------------------------------------------------------


def selectable(env: TaskEnv, agent_id: int, max_waiting: bool = True) -> np.ndarray:
    """[T] bool, tasks the agent may choose: the complement of ``agent_observe``'s task mask."""
    mask = env.get_unfinished_task_mask() | env.get_contributable_task_mask(agent_id)
    if max_waiting:
        not_open, _ = env.get_waiting_tasks()
        if np.sum(~not_open) > MAX_OPEN:
            mask |= not_open
    return ~mask


def _nearest_action(env: TaskEnv, agent_id: int, max_waiting: bool) -> int:
    """1-based task id of the closest selectable task (lowest id on ties), 0 (depot) if none."""
    agent = env.agent_dic[agent_id]
    best, action = np.inf, 0
    for j in np.flatnonzero(selectable(env, agent_id, max_waiting)):
        d = env.calculate_eulidean_distance(agent, env.task_dic[int(j)])
        if d < best:
            best, action = d, int(j) + 1
    return action


def _run_nearest(env: TaskEnv, max_waiting: bool, idle: str) -> None:
    """Nearest-task decisions in the loop of ``Worker.run_episode`` (``idle="wait"``, released agents not shuffled)
    or of ``execute_greedy_action`` (``idle="depot"``)."""
    tasks = env.task_dic.values()
    while not env.finished and env.current_time < MAX_TIME:
        released, env.current_time = env.next_decision()
        for agent_id in released[0] + released[1]:
            action = _nearest_action(env, agent_id, max_waiting)
            if action == 0 and idle == "wait":
                agent = env.agent_dic[agent_id]
                if not all(t["feasible_assignment"] for t in tasks):
                    agent["no_choice"] = True
                    continue
                if agent["current_task"] < 0:
                    continue
            env.agent_step(agent_id, action, 0)
        env.finished = env.check_finished()


def _result(env: TaskEnv, t0: float) -> EnvResult:
    env.calculate_waiting_time()
    agents = [env.agent_dic[i] for i in range(len(env.agent_dic))]
    routes = [[int(j) for j in a["route"] if j >= 0] for a in agents]
    depots = sum(sum(j < 0 for j in a["route"][1:-1]) for a in agents)  # first = start, last = final return
    makespan = float(env.current_time)
    success = succeeded(all(t["finished"] for t in env.task_dic.values()), makespan)
    return EnvResult(makespan, success, float(np.mean([a["sum_waiting_time"] for a in agents])), routes, int(depots),
                     time.perf_counter() - t0, bool(env.finished))


def greedy_nearest(source: Source, max_waiting: bool = True, idle: str = "wait") -> EnvResult:
    """The paper's greedy (nearest contributable task) run inside the env decision loop."""
    if idle not in ("wait", "depot"):
        raise ValueError(f"idle must be 'wait' or 'depot', got {idle!r}")
    t0 = time.perf_counter()
    env = make_env(source)
    _run_nearest(env, max_waiting, idle)
    return _result(env, t0)


def greedy_repo(source: Source) -> EnvResult:
    """The shipped ``TaskEnv.execute_greedy_action`` (with its argmin bug), for reference."""
    t0 = time.perf_counter()
    env = make_env(source)
    with contextlib.redirect_stdout(io.StringIO()):
        env.execute_greedy_action(plot_figure=False)
    return _result(env, t0)


# --- plan-level constructors -----------------------------------------------------------------------------------


@njit(cache=True)
def _cover(j, arr, req, ab, out, scanned=None) -> int:
    """Earliest-arrival cover of task ``j`` given agent arrivals ``arr``, made minimal by dropping redundant
    members latest arrival first (one pass suffices). Writes members to ``out``; returns their count, -1 if the
    agents cannot cover ``j`` at all. ``scanned`` (optional, bool [A]) marks the members before pruning."""
    K = req.shape[1]
    need = req[j].copy()
    n = 0
    for i in np.argsort(arr, kind="mergesort"):
        gain = 0.0
        for k in range(K):
            if need[k] > 0 and ab[i, k] > 0:
                gain += 1.0
        if gain > 0:
            out[n] = i
            n += 1
            for k in range(K):
                need[k] -= ab[i, k]
            done = True
            for k in range(K):
                if need[k] > 0:
                    done = False
            if done:
                break
    for k in range(K):
        if need[k] > 0:
            return -1
    if scanned is not None:
        scanned[:] = False
        for p in range(n):
            scanned[out[p]] = True
    m = n
    for p in range(n - 1, -1, -1):  # need <= 0 is the slack; drop a member if the slack absorbs it
        i = out[p]
        ok = True
        for k in range(K):
            if need[k] + ab[i, k] > 0:
                ok = False
                break
        if ok:
            for k in range(K):
                need[k] += ab[i, k]
            out[p] = -1
            m -= 1
    q = 0
    for p in range(n):
        if out[p] >= 0:
            out[q] = out[p]
            q += 1
    return m


@njit(cache=True)
def _arrivals(j, free, pos, tt, da, out) -> None:
    for i in range(len(free)):
        out[i] = free[i] + (da[i, j] if pos[i] < 0 else tt[pos[i], j])


@njit(cache=True)
def _dispatch(req, ab, dur, tt, da, noise, seed, order, mem, cnt) -> None:
    """Fill ``order``/``mem``/``cnt``: append the task with the earliest (noisy) cover start until all are placed.

    A cached cover stays valid while none of the agents taken by its scan (before pruning) moves: arrivals never
    decrease (triangle inequality), so a moved agent that the scan skipped still adds nothing where it lands."""
    np.random.seed(seed)
    T, A = req.shape[0], ab.shape[0]
    free, pos = np.zeros(A), np.full(A, -1, np.int64)
    arr = np.empty(A)
    stale = np.ones(T, np.bool_)
    scanned = np.zeros((T, A), np.bool_)
    placed = np.zeros(T, np.bool_)
    start = np.empty(T)
    score = np.empty(T)
    buf = np.empty(A, np.int64)
    for step in range(T):
        best = -1
        for j in range(T):
            if placed[j]:
                continue
            if stale[j]:
                _arrivals(j, free, pos, tt, da, arr)
                c = _cover(j, arr, req, ab, buf, scanned[j])
                cnt[j] = c
                s = -np.inf
                for p in range(c):
                    mem[j, p] = buf[p]
                    s = max(s, arr[buf[p]])
                start[j] = s if c > 0 else np.inf
                score[j] = start[j] * (1.0 + noise * np.random.random())
                stale[j] = False
            if best < 0 or score[j] < score[best]:
                best = j
        order[step] = best
        placed[best] = True
        f = start[best] + dur[best]
        for p in range(cnt[best]):
            i = mem[best, p]
            free[i], pos[i] = f, best
        for j in range(T):
            if not placed[j]:
                for q in range(cnt[best]):
                    if scanned[j, mem[best, q]]:
                        stale[j] = True


@njit(cache=True)
def _pass_from(seq, n, mem, cnt, dur, tt, da, free, pos, tot) -> float:
    """Continue a forward pass over ``seq[:n]`` from state ``free``/``pos`` (modified); returns the makespan and
    writes the summed depot returns to ``tot[0]``."""
    for t in range(n):
        j = seq[t]
        s = -np.inf
        for p in range(cnt[j]):
            i = mem[j, p]
            s = max(s, free[i] + (da[i, j] if pos[i] < 0 else tt[pos[i], j]))
        f = s + dur[j]
        for p in range(cnt[j]):
            i = mem[j, p]
            free[i], pos[i] = f, j
    ms, total = 0.0, 0.0
    for i in range(len(free)):
        if pos[i] >= 0:
            r = free[i] + da[i, pos[i]]
            ms = max(ms, r)
            total += r
    tot[0] = total
    return ms


@njit(cache=True)
def _best_insertion(j, seq, n, mem, cnt, req, ab, dur, tt, da, cov, res) -> int:
    """Best position (0..n) of task ``j`` in ``seq[:n]`` with its earliest-arrival minimal cover there.

    Writes the cover to ``cov`` and (makespan, summed returns, 2nd-best makespan) to ``res``; returns the
    position, or -1 if ``j`` cannot be covered."""
    A = ab.shape[0]
    free, pos = np.zeros(A), np.full(A, -1, np.int64)
    f2, p2 = np.empty(A), np.empty(A, np.int64)
    arr, buf, tot = np.empty(A), np.empty(A, np.int64), np.empty(1)
    best_ms, best_tot, second, best_at, best_c = np.inf, np.inf, np.inf, -1, -1
    for at in range(n + 1):
        _arrivals(j, free, pos, tt, da, arr)
        c = _cover(j, arr, req, ab, buf)
        if c < 0:
            return -1
        s = -np.inf
        for p in range(c):
            s = max(s, arr[buf[p]])
        f2[:] = free
        p2[:] = pos
        for p in range(c):
            f2[buf[p]], p2[buf[p]] = s + dur[j], j
        ms = _pass_from(seq[at:], n - at, mem, cnt, dur, tt, da, f2, p2, tot)
        if ms < best_ms - 1e-9 or (ms < best_ms + 1e-9 and tot[0] < best_tot):
            second = min(second, best_ms)
            best_ms, best_tot, best_at, best_c = ms, tot[0], at, c
            cov[:c] = buf[:c]
        else:
            second = min(second, ms)
        if at < n:  # advance the prefix state past seq[at]
            k = seq[at]
            s = -np.inf
            for p in range(cnt[k]):
                i = mem[k, p]
                s = max(s, free[i] + (da[i, k] if pos[i] < 0 else tt[pos[i], k]))
            for p in range(cnt[k]):
                i = mem[k, p]
                free[i], pos[i] = s + dur[k], k
    res[0], res[1], res[2] = best_ms, best_tot, second
    return best_c * (n + 1) + best_at  # packed: count and position


@njit(cache=True)
def _insertion(req, ab, dur, tt, da, tasks, regret, seq, mem, cnt) -> int:
    """Insert ``tasks`` one by one (in the given order, or by maximal regret over the remaining ones if ``regret``
    > 0, which evaluates up to ``regret`` candidates per step). Returns the number of tasks placed."""
    T = len(tasks)
    A = ab.shape[0]
    cov, res = np.empty(A, np.int64), np.empty(3)
    best_cov = np.empty(A, np.int64)
    left = tasks.copy()
    n_left = T
    for n in range(T):
        pick, packed, best_key = 0, -1, -np.inf
        m = min(n_left, regret) if regret > 0 else 1
        for q in range(m):
            j = left[q]
            got = _best_insertion(j, seq, n, mem, cnt, req, ab, dur, tt, da, cov, res)
            if got < 0:
                return -1
            key = res[2] - res[0] if regret > 0 else 0.0
            if key > best_key:
                best_key, pick, packed = key, q, got
                best_cov[:] = cov
        j = left[pick]
        left[pick] = left[n_left - 1]
        n_left -= 1
        c, at = packed // (n + 1), packed % (n + 1)
        seq[at + 1:n + 1] = seq[at:n].copy()
        seq[at] = j
        cnt[j] = c
        mem[j, :c] = best_cov[:c]
    return T


def _width(inst: Instance) -> int:
    """Upper bound on a minimal cover's size: every member covers a unit of requirement nobody else does."""
    return int(min(inst.n_agents, inst.req.sum(axis=1).max()))


def _plan(inst: Instance, order: np.ndarray, mem: np.ndarray, cnt: np.ndarray) -> Plan:
    return Plan.from_arrays(order, mem, cnt, inst.n_agents)


def dispatch(inst: Instance, noise: float = 0.0, seed: int = 0) -> Plan:
    """List scheduling by earliest cover start; ``noise`` > 0 multiplies scores by U[1, 1 + noise)."""
    T = inst.n_tasks
    order, cnt = np.empty(T, np.int64), np.zeros(T, np.int64)
    mem = np.full((T, max(1, _width(inst))), -1, np.int64)
    _dispatch(inst.req, inst.ab, inst.dur, inst.tt, inst.da, float(noise), int(seed), order, mem, cnt)
    if (cnt <= 0).any():
        raise ValueError("some task cannot be covered by all agents together")
    return _plan(inst, order, mem, cnt)


def insertion(inst: Instance, rng: np.random.Generator, regret: int = 0) -> Plan:
    """Cheapest insertion in random order (``regret=0``) or regret order over up to ``regret`` random candidates
    per step (cost O(regret * T^2) per step)."""
    T = inst.n_tasks
    seq, cnt = np.empty(T, np.int64), np.zeros(T, np.int64)
    mem = np.full((T, max(1, _width(inst))), -1, np.int64)
    tasks = rng.permutation(T).astype(np.int64)
    if _insertion(inst.req, inst.ab, inst.dur, inst.tt, inst.da, tasks, int(regret), seq, mem, cnt) < 0:
        raise ValueError("some task cannot be covered by all agents together")
    return _plan(inst, seq, mem, cnt)


def construct(inst: Instance, time_limit: float | None = None, restarts: int | None = None, seed: int = 0) -> Plan:
    """Best of the plain dispatch and randomized constructions (noisy dispatch and random-order insertion
    alternately) until ``restarts`` randomized ones are done or ``time_limit`` seconds have passed, whichever comes
    first (at least one of each kind). With ``restarts`` only, the result is deterministic."""
    if time_limit is None and restarts is None:
        raise ValueError("give time_limit or restarts")
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    best, best_ms = None, np.inf
    k = 0
    while True:
        if k == 0:
            plan = dispatch(inst)
        elif k % 2:
            plan = dispatch(inst, noise=0.3, seed=int(rng.integers(2**31)))
        else:
            plan = insertion(inst, rng)
        ms = evaluate(inst, plan).makespan
        if ms < best_ms:
            best, best_ms = plan, ms
        k += 1
        if k >= 3 and ((restarts is not None and k > restarts)
                       or (time_limit is not None and time.perf_counter() - t0 >= time_limit)):
            return best
