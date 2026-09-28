"""Coalition plans: per-task member sets plus a global task order (float keys).

Each agent visits its tasks in key order, so every plan is deadlock-free by construction. The env forms coalitions
in decision order and rejects members that join an already covered task, so the forward pass in ``evaluate``
equals the env exactly only for minimal covers (no member removable without breaking coverage); see
``prune_to_minimal``.
"""
from __future__ import annotations

import heapq
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise

import numpy as np

from cbba_sota.hetero.instance import Instance


@dataclass(eq=False)
class Plan:
    members: list[tuple[int, ...]]  # per task, sorted agent ids
    keys: np.ndarray  # [T] global order; ties broken by task id
    n_agents: int

    def __post_init__(self) -> None:
        self.members = [tuple(sorted(set(map(int, m)))) for m in self.members]
        self.keys = np.array(self.keys, dtype=float)
        if self.keys.shape != (len(self.members),):
            raise ValueError("need one key per task")
        if any(m and not 0 <= m[0] <= m[-1] < self.n_agents for m in self.members):
            raise ValueError("agent id out of range")

    @property
    def n_tasks(self) -> int:
        return len(self.members)

    def copy(self) -> Plan:
        return Plan(list(self.members), self.keys.copy(), self.n_agents)

    def order(self) -> np.ndarray:
        """All task ids sorted by (key, id)."""
        return np.argsort(self.keys, kind="stable")

    def routes(self) -> list[list[int]]:
        """Per agent, its tasks (0-based) in key order."""
        out: list[list[int]] = [[] for _ in range(self.n_agents)]
        for j in self.order():
            for i in self.members[j]:
                out[i].append(int(j))
        return out

    def to_env_routes(self) -> list[list[int]]:
        """Per agent, 1-based task ids for ``TaskEnv.pre_set_route`` (0 is the depot)."""
        return [[j + 1 for j in r] for r in self.routes()]

    @classmethod
    def from_routes(cls, routes: Sequence[Sequence[int]], n_tasks: int, one_based: bool = False) -> Plan:
        """Plan from per-agent routes; keys are a topological rank of the route precedences.

        Raises ``ValueError`` if a route repeats a task or the routes contradict every global order (a cycle,
        which deadlocks the env).
        """
        shift = 1 if one_based else 0
        members: list[list[int]] = [[] for _ in range(n_tasks)]
        succ: list[set[int]] = [set() for _ in range(n_tasks)]
        for i, route in enumerate(routes):
            route = [int(j) - shift for j in route]
            if len(set(route)) != len(route) or not all(0 <= j < n_tasks for j in route):
                raise ValueError(f"route of agent {i} repeats a task or leaves the task range: {route}")
            for j in route:
                members[j].append(i)
            for a, b in pairwise(route):
                succ[a].add(b)
        indeg = np.zeros(n_tasks, dtype=int)
        for s in succ:
            for b in s:
                indeg[b] += 1
        ready = [j for j in range(n_tasks) if indeg[j] == 0]
        heapq.heapify(ready)
        keys = np.full(n_tasks, np.nan)
        rank = 0
        while ready:
            j = heapq.heappop(ready)
            keys[j] = rank
            rank += 1
            for b in succ[j]:
                indeg[b] -= 1
                if indeg[b] == 0:
                    heapq.heappush(ready, b)
        if rank < n_tasks:
            stuck = np.flatnonzero(np.isnan(keys)).tolist()
            raise ValueError(f"routes contain a precedence cycle through tasks {stuck}")
        return cls(members, keys, len(routes))

    def to_arrays(self, width: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(order [T], mem [T, W], cnt [T])`` int64 arrays for the numba kernels; ``mem[j, :cnt[j]]`` are the
        members of task ``j``, padded with -1."""
        cnt = np.array([len(m) for m in self.members], dtype=np.int64)
        width = max(1, int(cnt.max(initial=0))) if width is None else width
        if cnt.max(initial=0) > width:
            raise ValueError(f"a coalition has {cnt.max()} members, more than width={width}")
        mem = np.full((self.n_tasks, width), -1, dtype=np.int64)
        for j, m in enumerate(self.members):
            mem[j, :len(m)] = m
        return self.order().astype(np.int64), mem, cnt

    @classmethod
    def from_arrays(cls, order: np.ndarray, mem: np.ndarray, cnt: np.ndarray, n_agents: int) -> Plan:
        """Inverse of ``to_arrays``; ``order`` must be a permutation of the tasks."""
        if sorted(order) != list(range(len(cnt))):
            raise ValueError("order is not a permutation of the tasks")
        keys = np.empty(len(order))
        keys[np.asarray(order)] = np.arange(len(order))
        return cls([tuple(mem[j, :cnt[j]]) for j in range(len(cnt))], keys, n_agents)

    # --- coverage ------------------------------------------------------------------------------------------

    def covers(self, inst: Instance) -> np.ndarray:
        """[T] bool: the members' summed traits cover the requirement."""
        return np.array([(inst.ab[list(m)].sum(axis=0) >= r).all() for m, r in zip(self.members, inst.req)])

    def redundant(self, inst: Instance) -> list[tuple[int, int]]:
        """(task, agent) pairs whose removal keeps the task covered."""
        out = []
        for j, (m, r) in enumerate(zip(self.members, inst.req)):
            total = inst.ab[list(m)].sum(axis=0)
            if (total >= r).all():
                out += [(j, i) for i in m if (total - inst.ab[i] >= r).all()]
        return out

    def is_minimal_cover(self, inst: Instance) -> bool:
        """Every task covered and no member removable."""
        return bool(self.covers(inst).all()) and not self.redundant(inst)

    def prune_to_minimal(self, inst: Instance) -> Plan:
        """Drop redundant members until every covered task is a minimal cover; keys are kept.

        Tasks are visited in key order and, among removable members, the latest arrival goes first. Removing
        member ``i`` of task ``j`` never delays anything: ``j`` starts at the max of fewer arrivals, and ``i``
        travels ``p -> q`` directly instead of ``p -> j -> q`` (triangle inequality, and ``j`` finishes after
        ``i`` arrives there), so by monotonicity of the forward pass (max and +) no start, finish or return time
        increases. Floating-point rounding can break the triangle inequality by an ulp, hence the tolerance.
        """
        tt, da, ab, req = inst.tt, inst.da, inst.ab, inst.req
        free = np.zeros(self.n_agents)
        pos = np.full(self.n_agents, -1)
        members = list(self.members)
        for j in self.order():
            m = list(members[j])
            if not m:
                continue
            arr = {i: free[i] + (da[i, j] if pos[i] < 0 else tt[pos[i], j]) for i in m}
            total = ab[m].sum(axis=0)
            if (total >= req[j]).all():
                for i in sorted(m, key=lambda i: (arr[i], i), reverse=True):
                    if (total - ab[i] >= req[j]).all():
                        total = total - ab[i]
                        m.remove(i)
                members[j] = tuple(m)
            finish = max(arr[i] for i in m) + inst.dur[j]
            for i in m:
                free[i], pos[i] = finish, j
        pruned = Plan(members, self.keys.copy(), self.n_agents)

        from cbba_sota.hetero.evaluate import evaluate

        assert evaluate(inst, pruned).makespan <= evaluate(inst, self).makespan + 1e-9
        return pruned


def random_plan(inst: Instance, rng: np.random.Generator, greedy: float = 0.0) -> Plan:
    """Random key-ordered minimal-cover plan, for tests and fuzzing.

    Tasks get a random global order. Each coalition takes agents that still contribute to the open requirement,
    in uniform random order (``greedy=0``) or by noisy earliest arrival (``greedy>0``; larger is less noisy),
    then drops redundant members in random order (one pass suffices: dropping only shrinks the coverage).
    """
    T, A = inst.n_tasks, inst.n_agents
    keys = rng.permutation(T).astype(float)
    free = np.zeros(A)
    pos = np.full(A, -1)
    members: list[tuple[int, ...]] = [()] * T
    for j in np.argsort(keys):
        arr = free + np.where(pos < 0, inst.da[:, j], inst.tt[pos, j])
        score = rng.random(A) if greedy <= 0 else arr * (1 + rng.random(A) / greedy)
        need = inst.req[j].copy()
        m = []
        for i in np.argsort(score):
            if (need <= 0).all():
                break
            if (np.minimum(inst.ab[i], need) > 0).any():
                m.append(int(i))
                need -= inst.ab[i]
        if (need > 0).any():
            raise ValueError(f"task {j} cannot be covered by all agents together")
        total = inst.ab[m].sum(axis=0)
        for i in rng.permutation(m):
            if (total - inst.ab[i] >= inst.req[j]).all():
                total = total - inst.ab[i]
                m.remove(i)
        members[j] = tuple(m)
        finish = arr[m].max() + inst.dur[j]
        free[m], pos[m] = finish, j
    return Plan(members, keys, A)
