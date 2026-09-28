"""Forward pass of a plan: start/finish times, depot returns, waiting, makespan.

Semantics of ``TaskEnv.execute_by_route``: an agent leaves when its previous task finishes (time 0 from the
depot), a task starts at the latest arrival of its members and lasts ``dur``, and the makespan is the latest depot
return. For minimal covers in key order this equals the env replay bit for bit (see ``plan``); every member listed
in the plan is assumed to take part.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit

from cbba_sota.bench.configs import MAX_TIME
from cbba_sota.hetero.instance import Instance
from cbba_sota.hetero.plan import Plan


@dataclass(frozen=True, eq=False)
class Schedule:
    makespan: float
    start: np.ndarray  # [T], nan for tasks without members
    finish: np.ndarray  # [T]
    ret: np.ndarray  # [A] depot return time (0 for idle agents)
    wait: np.ndarray  # [A] summed waiting at tasks (start - own arrival)
    success: bool  # every task covered and makespan < MAX_TIME

    @property
    def awt(self) -> float:
        """Mean over agents of their waiting time, as the env's ``sum_waiting_time`` average."""
        return float(self.wait.mean()) if self.wait.size else 0.0


@njit(cache=True)
def forward_pass(order, mem, cnt, dur, tt, da, start, finish, ret, wait) -> float:
    """Fill ``start``/``finish`` [T] and ``ret``/``wait`` [A]; return the makespan.

    ``order`` lists task ids in execution order (tasks with ``cnt == 0`` are skipped); ``mem[j, :cnt[j]]`` are the
    members of task ``j``; ``tt`` [T, T] and ``da`` [A, T] are travel times (``Instance.tt``, ``Instance.da``).
    """
    A = da.shape[0]
    free = np.zeros(A)
    pos = np.full(A, -1, np.int64)
    start[:] = np.nan
    finish[:] = np.nan
    wait[:] = 0.0
    for j in order:
        c = cnt[j]
        if c == 0:
            continue
        s = -np.inf
        for k in range(c):
            i = mem[j, k]
            a = free[i] + (da[i, j] if pos[i] < 0 else tt[pos[i], j])
            s = max(s, a)
        f = s + dur[j]
        for k in range(c):
            i = mem[j, k]
            wait[i] += s - (free[i] + (da[i, j] if pos[i] < 0 else tt[pos[i], j]))
            free[i] = f
            pos[i] = j
        start[j] = s
        finish[j] = f
    makespan = 0.0
    for i in range(A):
        ret[i] = free[i] + da[i, pos[i]] if pos[i] >= 0 else 0.0
        makespan = max(makespan, ret[i])
    return makespan


@njit(cache=True)
def covered(req, ab, mem, cnt) -> np.ndarray:
    """[T] bool: the members' summed traits cover each task's requirement."""
    T, K = req.shape
    out = np.empty(T, np.bool_)
    for j in range(T):
        ok = True
        for k in range(K):
            total = 0.0
            for m in range(cnt[j]):
                total += ab[mem[j, m], k]
            if total < req[j, k]:
                ok = False
                break
        out[j] = ok
    return out


def _in_range(ids: np.ndarray, n: int) -> bool:
    return ids.size == 0 or (ids.min() >= 0 and ids.max() < n)


def evaluate_arrays(inst: Instance, order: np.ndarray, mem: np.ndarray, cnt: np.ndarray) -> Schedule:
    """Array entry point (see ``Plan.to_arrays``); ``order`` must list every task once for ``success``.

    Ids are range-checked here; the kernels themselves do no bounds checking."""
    T, A = inst.n_tasks, inst.n_agents
    used = mem[np.arange(mem.shape[1]) < cnt[:, None]]
    if len(cnt) != T or not _in_range(cnt, mem.shape[1] + 1) or not _in_range(order, T) or not _in_range(used, A):
        raise ValueError("task id, agent id or coalition size out of range")
    start, finish, ret, wait = np.empty(T), np.empty(T), np.empty(A), np.empty(A)
    makespan = forward_pass(order, mem, cnt, inst.dur, inst.tt, inst.da, start, finish, ret, wait)
    complete = len(order) == T and bool(np.all(np.bincount(order, minlength=T) == 1))
    success = complete and bool(covered(inst.req, inst.ab, mem, cnt).all()) and makespan < MAX_TIME
    return Schedule(makespan, start, finish, ret, wait, success)


def evaluate(inst: Instance, plan: Plan) -> Schedule:
    return evaluate_arrays(inst, *plan.to_arrays())


def evaluate_reference(inst: Instance, plan: Plan) -> Schedule:
    """Pure-Python forward pass, independent of the kernels and of the cached travel matrices (tests only)."""

    def travel(p: np.ndarray, q: np.ndarray) -> float:
        return float(np.linalg.norm(p - q)) / inst.speed

    T, A = inst.n_tasks, inst.n_agents
    free = [0.0] * A
    where = [inst.depot[i] for i in range(A)]
    busy = [False] * A
    start, finish, wait = [float("nan")] * T, [float("nan")] * T, [0.0] * A
    for j in sorted(range(T), key=lambda j: (plan.keys[j], j)):
        if not plan.members[j]:
            continue
        arrival = {i: free[i] + travel(where[i], inst.loc[j]) for i in plan.members[j]}
        start[j] = max(arrival.values())
        finish[j] = start[j] + float(inst.dur[j])
        for i, a in arrival.items():
            wait[i] += start[j] - a
            free[i], where[i], busy[i] = finish[j], inst.loc[j], True
    ret = [free[i] + travel(where[i], inst.depot[i]) if busy[i] else 0.0 for i in range(A)]
    makespan = max(ret, default=0.0)
    ok = all(m and all(sum(inst.ab[i][k] for i in m) >= inst.req[j][k] for k in range(inst.n_traits))
             for j, m in enumerate(plan.members))
    return Schedule(makespan, np.array(start), np.array(finish), np.array(ret), np.array(wait),
                    ok and makespan < MAX_TIME)
