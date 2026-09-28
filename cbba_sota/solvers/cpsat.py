"""OR-Tools CP-SAT for makespan coalition scheduling on HeteroMRTA instances.

Model (``full`` mode):
- ``x[a, j]``: agent ``a`` is a member of task ``j``; only for agents that can reduce ``j``'s requirement.
- One ``AddCircuit`` per agent over its depot (node 0) and its candidate tasks; a skipped task is a self-loop
  (``not x``), an idle agent loops on the depot. Arc ``j -> k`` enforces ``s_k >= s_j + w(j, k)``, depot arcs
  enforce ``s_j >= d0(a, j)`` and ``ms >= s_j + d1(a, j)``.
- Coverage ``sum_a ab[a, k] x[a, j] >= req[j, k]``; objective ``min ms``.
- Valid cuts: per species and task at most ``max_{k in species traits} req[j, k]`` members (a larger coalition has
  a redundant member, and ``Plan.prune_to_minimal`` never increases the makespan); earliest starts and return
  tails from the closest capable agent per required trait.
- Symmetry: agents of one species are identical, so their routes are ordered by the smallest task id they visit
  (``first[a, j] = OR_{k <= j} x[a, k]`` and ``first[a + 1, j] <= first[a, j]``); hints are permuted to match.

Time scaling: all times are integers in units of ``1 / SCALE`` (``SCALE = 1000``). Every arc weight is rounded up
as one sum, ``w(j, k) = max(1, ceil((dur_j + tt_jk) * SCALE))``, ``d0 = ceil(da * SCALE)``,
``d1 = ceil((dur_j + da) * SCALE)``, so any CP schedule is feasible in real time and the real earliest-start
forward pass of the same members and routes is never later: ``evaluate <= objective / SCALE``, at most ~1e-3 per
arc on a route. The floor of one unit on task-to-task arcs (binding only when ``dur_j + tt_jk`` rounds to 0) makes
CP starts strictly increase along every agent's circuit, so the plan keys (CP starts) order each route as its
circuit does. Every plan is converted to a key-ordered minimal cover (``prune_to_minimal``) and must be scored by
env replay.

``lns`` mode (LNS in CP): starting from a plan, repeatedly free a window of tasks (critical chain, spatial,
temporal or random), keep every other coalition fixed, and re-solve with a small CP-SAT model: agents that may
take a freed task keep their fixed tasks as forced visits and get a circuit over those and their candidate freed
tasks; all other agents become fixed precedence chains. The incumbent is always a complete hint.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from itertools import pairwise

import numpy as np
from ortools.sat.python import cp_model

from cbba_sota.bench.configs import MAX_TIME
from cbba_sota.hetero.evaluate import evaluate
from cbba_sota.hetero.instance import Instance
from cbba_sota.hetero.plan import Plan

SCALE = 1000


@dataclass
class Weights:
    """Integer times (units of 1 / SCALE), rounded up; values within 1e-6 of an integer are snapped to it first,
    so exact multiples of 1 / SCALE do not gain a unit from floating-point noise. Task-to-task weights are at least
    one unit, so starts strictly increase along a route (zero durations on coincident points would tie them)."""

    w: np.ndarray  # [T, T] dur_j + travel j -> k, at least 1
    d0: np.ndarray  # [A, T] depot -> j
    d1: np.ndarray  # [A, T] dur_j + j -> depot

    @classmethod
    def of(cls, inst: Instance, scale: int = SCALE) -> Weights:
        up = lambda x: np.ceil(np.round(x * scale, 6)).astype(np.int64)
        return cls(np.maximum(up(inst.dur[:, None] + inst.tt), 1), up(inst.da), up(inst.dur[None, :] + inst.da))

    def schedule(self, plan: Plan) -> tuple[np.ndarray, int]:
        """Integer forward pass of a key-ordered plan: (starts [T], makespan)."""
        pos = np.full(len(self.d0), -1)
        start = np.zeros(len(plan.members), np.int64)
        for j in plan.order():
            m = plan.members[j]
            if m:
                start[j] = max(self.d0[i, j] if pos[i] < 0 else start[pos[i]] + self.w[pos[i], j] for i in m)
                pos[list(m)] = j
        ms = max((int(start[p] + self.d1[i, p]) for i, p in enumerate(pos) if p >= 0), default=0)
        return start, ms


@dataclass
class Result:
    plan: Plan | None  # key-ordered minimal cover; None if nothing was found
    makespan: float  # evaluate() of ``plan`` (equals env replay)
    status: str
    objective: float  # last CP objective / SCALE (full mode), nan otherwise
    bound: float  # CP lower bound / SCALE (full mode), nan otherwise
    wall_s: float
    cpu_s: float
    trajectory: list[tuple[float, float, Plan]] = field(default_factory=list)  # (elapsed s, makespan, plan)
    iterations: int = 0  # LNS sub-solves
    improvements: int = 0

    def at(self, budget: float) -> tuple[float, Plan] | None:
        """Best (makespan, plan) found within ``budget`` seconds of the start, None if nothing yet."""
        best = None
        for t, ms, plan in self.trajectory:
            if t <= budget and (best is None or ms < best[0]):
                best = (ms, plan)
        return best


def useful(inst: Instance) -> np.ndarray:
    """[A, T] bool: the agent can reduce the task's requirement."""
    return (inst.ab[:, None, :] * (inst.req[None, :, :] > 0)).sum(axis=2) > 0


def species_cap(inst: Instance) -> np.ndarray:
    """[S, T] max members of one species in a minimal cover (each member needs a trait where the coalition's
    total does not exceed the requirement)."""
    first = [int(np.flatnonzero(inst.species == s)[0]) for s in range(inst.n_species)]
    has = inst.ab[first] > 0  # [S, K]
    return np.where(has[:, None, :], inst.req[None, :, :], 0).max(axis=2).astype(np.int64)


def canonical(inst: Instance, plan: Plan) -> Plan:
    """Permute identical agents (same species) so that routes are ordered by their smallest task id."""
    routes = plan.routes()
    perm = np.arange(inst.n_agents)
    for s in range(inst.n_species):
        agents = np.flatnonzero(inst.species == s)
        key = [min(routes[a], default=inst.n_tasks) for a in agents]
        perm[agents] = agents[np.argsort(key, kind="stable")]  # perm[new agent] = old agent whose route it takes
    new_of_old = np.empty_like(perm)
    new_of_old[perm] = np.arange(inst.n_agents)
    return Plan([tuple(new_of_old[list(m)]) for m in plan.members], plan.keys.copy(), plan.n_agents)


class _Model:
    """CP-SAT model: circuits for the agents in ``cand`` (candidate tasks, ``forced`` visits among them), fixed
    precedence chains for the agents in ``chains``, coverage for ``cover_tasks``."""

    def __init__(self, inst: Instance, W: Weights, cand: dict[int, list[int]], forced: dict[int, set[int]],
                 chains: dict[int, list[int]], cover_tasks: list[int], arcs: dict[int, set[tuple[int, int]]] | None,
                 symmetry: bool, ub: int, lex: bool = False):
        T = inst.n_tasks
        self.m = m = cp_model.CpModel()
        est, tail = _bounds(inst, W)
        if (est + tail).max() > ub:
            raise ValueError("upper bound below the trivial lower bound")
        self.s = [m.NewIntVar(int(est[j]), int(ub - tail[j]), f"s{j}") for j in range(T)]
        self.ms = m.NewIntVar(int((est + tail).max()), int(ub), "ms")
        for j in range(T):
            m.Add(self.ms >= self.s[j] + int(tail[j]))
        self.x: dict[tuple[int, int], cp_model.IntVar] = {}  # decision literals only (forced visits excluded)
        self.by_task: list[list[tuple[int, cp_model.IntVar]]] = [[] for _ in range(T)]
        self.arc: dict[tuple[int, int, int], cp_model.IntVar] = {}  # (a, j, k) with -1 = depot
        self.idle: dict[int, cp_model.IntVar] = {}
        self.first: dict[tuple[int, int], cp_model.IntVar] = {}
        self.end = {a: m.NewIntVar(0, int(ub), f"end{a}") for a in [*cand, *chains]}
        for v in self.end.values():
            m.Add(self.ms >= v)
        for a, tasks in cand.items():
            self._agent(a, tasks, forced.get(a, set()), None if arcs is None else arcs[a], W)
        for a, route in chains.items():
            if route:
                m.Add(self.s[route[0]] >= int(W.d0[a, route[0]]))
                for j, k in pairwise(route):
                    m.Add(self.s[k] >= self.s[j] + int(W.w[j, k]))
                m.Add(self.end[a] >= self.s[route[-1]] + int(W.d1[a, route[-1]]))
        cap = species_cap(inst)
        for j in cover_tasks:
            for k in np.flatnonzero(inst.req[j] > 0):
                m.Add(sum(int(inst.ab[a, k]) * v for a, v in self.by_task[j] if inst.ab[a, k] > 0)
                      >= int(inst.req[j, k]))
            for sp in range(inst.n_species):
                vs = [v for a, v in self.by_task[j] if inst.species[a] == sp]
                if len(vs) > cap[sp, j]:
                    m.Add(sum(vs) <= int(cap[sp, j]))
        if symmetry:
            self._symmetry(inst)
        if lex:  # (makespan, summed depot returns) lexicographically
            lb = int((est + tail).max())
            m.Minimize((self.ms - lb) * (len(self.end) * int(ub) + 1) + sum(self.end.values()))
        else:
            m.Minimize(self.ms)

    def _agent(self, a: int, tasks: list[int], forced: set[int], arcs: set[tuple[int, int]] | None,
               W: Weights) -> None:
        m, s = self.m, self.s
        node = {j: p + 1 for p, j in enumerate(tasks)}
        self.idle[a] = idle = m.NewBoolVar(f"idle{a}")
        lits = [(0, 0, idle)]
        if forced:
            m.Add(idle == 0)
        for j in tasks:
            if j not in forced:
                x = m.NewBoolVar(f"x{a}_{j}")
                lits.append((node[j], node[j], x.Not()))
                m.AddImplication(x, idle.Not())
                self.x[a, j] = x
                self.by_task[j].append((a, x))
            self.arc[a, -1, j] = lo = m.NewBoolVar("")
            lits.append((0, node[j], lo))
            m.Add(s[j] >= int(W.d0[a, j])).OnlyEnforceIf(lo)
            self.arc[a, j, -1] = hi = m.NewBoolVar("")
            lits.append((node[j], 0, hi))
            m.Add(self.end[a] >= s[j] + int(W.d1[a, j])).OnlyEnforceIf(hi)
        for j in tasks:
            for k in tasks:
                if j != k and (arcs is None or (j, k) in arcs):
                    self.arc[a, j, k] = lit = m.NewBoolVar("")
                    lits.append((node[j], node[k], lit))
                    m.Add(s[k] >= s[j] + int(W.w[j, k])).OnlyEnforceIf(lit)
        m.AddCircuit(lits)

    def _symmetry(self, inst: Instance) -> None:
        """``first[a, j] = OR_{k <= j} x[a, k]``, ordered within each species."""
        m = self.m
        for a in range(inst.n_agents):
            prev = None
            for j in range(inst.n_tasks):
                x = self.x.get((a, j))
                if x is not None:
                    f = m.NewBoolVar("")
                    m.AddImplication(x, f)
                    if prev is None:
                        m.AddImplication(f, x)
                    else:
                        m.AddImplication(prev, f)
                        m.AddBoolOr([prev, x, f.Not()])
                    self.first[a, j] = prev = f
        for sp in range(inst.n_species):
            agents = np.flatnonzero(inst.species == sp)
            for a, b in pairwise(agents):
                for j in range(inst.n_tasks):  # same abilities, so the same candidate tasks
                    if (b, j) in self.first:
                        m.AddImplication(self.first[b, j], self.first[a, j])

    def hint(self, plan: Plan, W: Weights) -> None:
        """Complete hint: the plan's integer schedule, visits, arcs and symmetry literals."""
        start, ms = W.schedule(plan)
        m = self.m
        for j, v in enumerate(self.s):
            m.AddHint(v, int(start[j]))
        m.AddHint(self.ms, ms)
        routes = plan.routes()
        for a, v in self.end.items():
            m.AddHint(v, int(start[routes[a][-1]] + W.d1[a, routes[a][-1]]) if routes[a] else 0)
        used = {(a, j, k) for a, r in enumerate(routes) for j, k in zip([-1, *r], [*r, -1])}
        on = {(a, j) for a, r in enumerate(routes) for j in r}
        for a, v in self.idle.items():
            m.AddHint(v, int(not routes[a]))
        for key, lit in self.arc.items():
            m.AddHint(lit, int(key in used))
        for key, v in self.x.items():
            m.AddHint(v, int(key in on))
        for (a, j), f in self.first.items():
            m.AddHint(f, int(bool(routes[a]) and min(routes[a]) <= j))

    def extract(self, value, fixed: list[tuple[int, ...]] | None = None) -> tuple[list[list[int]], np.ndarray]:
        """(members per task, CP start times) of the current solution; ``fixed`` members are added per task."""
        members: list[list[int]] = [[] for _ in self.s] if fixed is None else [list(m) for m in fixed]
        for (a, j), v in self.x.items():
            if value(v):
                members[j].append(a)
        return members, np.array([value(v) for v in self.s], dtype=float)


def _bounds(inst: Instance, W: Weights) -> tuple[np.ndarray, np.ndarray]:
    """Per task: earliest start and minimal return tail over the closest capable agent of each required trait."""
    T = inst.n_tasks
    est, tail = np.zeros(T, np.int64), np.zeros(T, np.int64)
    for j in range(T):
        for k in np.flatnonzero(inst.req[j] > 0):
            capable = inst.ab[:, k] > 0
            est[j] = max(est[j], W.d0[capable, j].min())
            tail[j] = max(tail[j], W.d1[capable, j].min())
    return est, tail


def _solver(time_limit: float, workers: int, seed: int, log: bool = False) -> cp_model.CpSolver:
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = max(0.01, time_limit)
    solver.parameters.num_workers = workers
    solver.parameters.random_seed = seed
    solver.parameters.log_search_progress = log
    return solver


class _Recorder(cp_model.CpSolverSolutionCallback):
    """Keeps every improving solution with its time since ``t0``."""

    def __init__(self, model: _Model, t0: float):
        super().__init__()
        self.model, self.t0 = model, t0
        self.raw: list[tuple[float, list[list[int]], np.ndarray]] = []

    def on_solution_callback(self) -> None:
        self.raw.append((time.perf_counter() - self.t0, *self.model.extract(self.Value)))


def _plan(inst: Instance, members: list[list[int]], keys: np.ndarray) -> tuple[float, Plan]:
    plan = Plan(members, keys, inst.n_agents).prune_to_minimal(inst)
    return evaluate(inst, plan).makespan, plan


def knn_arcs(inst: Instance, cand: list[int], k: int) -> set[tuple[int, int]]:
    """Arcs from each candidate task to its ``k`` nearest other candidates."""
    c = np.asarray(cand)
    if len(c) <= k + 1:
        return {(j, l) for j in cand for l in cand if j != l}
    d = inst.tt[np.ix_(c, c)] + np.diag(np.full(len(c), np.inf))
    near = np.argsort(d, axis=1)[:, :k]
    return {(int(c[p]), int(c[q])) for p in range(len(c)) for q in near[p]}


def solve_full(inst: Instance, time_limit: float, workers: int = 8, hint: Plan | None = None, seed: int = 0,
               symmetry: bool = True, knn: int | None = None, log: bool = False, t0: float | None = None) -> Result:
    """Monolithic model. ``knn``: keep only arcs to the ``knn`` nearest candidate tasks (plus the hint's arcs), a
    heuristic restriction. ``t0``: start of the budget (default now), so that hint construction counts."""
    t0 = time.perf_counter() if t0 is None else t0
    c0 = time.process_time()
    W = Weights.of(inst)
    ok = useful(inst)
    cand = {a: np.flatnonzero(ok[a]).tolist() for a in range(inst.n_agents)}
    traj = []
    if hint is not None:
        hint = canonical(inst, hint) if symmetry else hint
        traj.append((time.perf_counter() - t0, evaluate(inst, hint).makespan, hint))
    arcs = None
    if knn is not None:
        arcs = {a: knn_arcs(inst, c, knn) for a, c in cand.items()}
        if hint is not None:
            for a, r in enumerate(hint.routes()):
                arcs[a] |= set(pairwise(r))
    ub = W.schedule(hint)[1] if hint is not None else int(MAX_TIME * SCALE)
    model = _Model(inst, W, cand, {}, {}, list(range(inst.n_tasks)), arcs, symmetry, ub)
    if hint is not None:
        model.hint(hint, W)
    solver = _solver(time_limit - (time.perf_counter() - t0), workers, seed, log)
    rec = _Recorder(model, t0)
    status = solver.Solve(model.m, rec)
    for t, members, keys in rec.raw:
        traj.append((t, *_plan(inst, members, keys)))
    wall, cpu = time.perf_counter() - t0, time.process_time() - c0
    feasible = status in (cp_model.OPTIMAL, cp_model.FEASIBLE)
    objective = solver.ObjectiveValue() / SCALE if feasible else math.nan
    bound = solver.BestObjectiveBound() / SCALE if feasible or status == cp_model.UNKNOWN else math.nan
    if not traj:
        return Result(None, math.inf, solver.StatusName(status), objective, bound, wall, cpu)
    best = min(traj, key=lambda r: r[1])
    return Result(best[2], best[1], solver.StatusName(status), objective, bound, wall, cpu, traj)


# --- LNS in CP -----------------------------------------------------------------------------------------------


def critical_chain(inst: Instance, plan: Plan) -> list[int]:
    """Tasks on the makespan-defining chain: the latest-returning agent's last task, then repeatedly the previous
    task of the member that arrived last (the one that set the start), back to a depot departure."""
    sched = evaluate(inst, plan)
    routes = plan.routes()
    prev = {}
    for a, r in enumerate(routes):
        for p, j in enumerate(r):
            prev[a, j] = r[p - 1] if p else -1
    a = int(np.argmax(sched.ret))
    j = routes[a][-1] if routes[a] else -1
    chain = []
    while j >= 0:
        chain.append(j)
        arr = {i: (inst.da[i, j] if prev[i, j] < 0 else sched.finish[prev[i, j]] + inst.tt[prev[i, j], j])
               for i in plan.members[j]}
        j = prev[max(arr, key=arr.get), j]
    return chain[::-1]


def _window(inst: Instance, plan: Plan, q: int, kind: str, rng: np.random.Generator) -> list[int]:
    T = inst.n_tasks
    if q >= T:
        return list(range(T))
    if kind == "random":
        return rng.choice(T, q, replace=False).tolist()
    if kind == "temporal":
        start = evaluate(inst, plan).start
        seed = rng.integers(T)
        return np.argsort(np.abs(start - start[seed]), kind="stable")[:q].tolist()
    if kind == "critical":
        chain = critical_chain(inst, plan)
        if len(chain) > q:
            lo = rng.integers(len(chain) - q + 1)
            chain = chain[lo:lo + q]
        out = list(dict.fromkeys(chain))
        near = [iter(np.argsort(inst.tt[j], kind="stable")) for j in out]
        while len(out) < q:  # round-robin spatial neighbours of the chain
            for it in near:
                for k in it:
                    if k not in out:
                        out.append(int(k))
                        break
                if len(out) >= q:
                    break
        return out
    seed = rng.integers(T)  # spatial
    return np.argsort(inst.tt[seed], kind="stable")[:q].tolist()


def _detour(inst: Instance, route: list[int], a: int, j: int) -> float:
    """Cheapest extra travel time for agent ``a`` to visit ``j`` somewhere in its route."""

    def leg(p: int, q: int) -> float:  # -1 is the agent's depot
        if p < 0 and q < 0:
            return 0.0
        return inst.da[a, max(p, q)] if min(p, q) < 0 else inst.tt[p, q]

    pts = [-1, *route, -1]
    return min(leg(p, j) + leg(j, q) - leg(p, q) for p, q in pairwise(pts))


def _subproblem(inst: Instance, W: Weights, plan: Plan, free: list[int], per_species: int, ub: int,
                lex: bool = True) -> tuple[_Model, list[tuple[int, ...]]]:
    """Model with the ``free`` tasks' coalitions open and every other coalition fixed (returned per task).

    Candidates of a freed task: its current members plus, per useful species, the ``per_species`` agents with the
    cheapest detour to it in their current routes."""
    free_set = set(free)
    routes = plan.routes()
    ok = useful(inst)
    cand_agents: dict[int, set[int]] = {j: set(plan.members[j]) for j in free}
    for j in free:
        for sp in range(inst.n_species):
            agents = [a for a in np.flatnonzero(inst.species == sp) if ok[a, j]]
            agents.sort(key=lambda a: (_detour(inst, routes[a], a, j), a))
            cand_agents[j] |= set(agents[:per_species])
    active = sorted(set().union(*cand_agents.values()))
    cand, forced, arcs = {}, {}, {}
    for a in active:
        fixed = [j for j in routes[a] if j not in free_set]
        opts = [j for j in free if a in cand_agents[j]]
        cand[a], forced[a] = fixed + opts, set(fixed)
        arcs[a] = set(pairwise(fixed)) | {(j, k) for j in cand[a] for k in opts if j != k} | {
            (j, k) for j in opts for k in fixed}
    chains = {a: routes[a] for a in range(inst.n_agents) if a not in cand}
    fixed = [() if j in free_set else plan.members[j] for j in range(inst.n_tasks)]
    return _Model(inst, W, cand, forced, chains, free, arcs, False, ub, lex), fixed


def _score(inst: Instance, plan: Plan) -> tuple[float, float]:
    sched = evaluate(inst, plan)
    return sched.makespan, float(sched.ret.sum())


def solve_lns(inst: Instance, time_limit: float, init: Plan, workers: int = 8, seed: int = 0, q0: int = 12,
              q_min: int = 4, q_max: int = 40, per_species: int = 4, sub_time: float = 2.0,
              kinds: tuple[str, ...] = ("critical", "spatial", "temporal", "random"), lex: bool = False,
              t0: float | None = None) -> Result:
    """LNS in CP from ``init`` until ``time_limit`` (measured from ``t0``, default now).

    Sub-solves minimise (makespan, summed depot returns) lexicographically if ``lex`` (else the makespan only),
    and a result replaces the incumbent unless it is lexicographically worse in real time, so the search keeps
    moving on makespan plateaus. The window size adapts: +1 when a sub-solve is proven optimal, -1 when it times
    out."""
    t0 = time.perf_counter() if t0 is None else t0
    c0 = time.process_time()
    rng = np.random.default_rng(seed)
    W = Weights.of(inst)
    cur = init.prune_to_minimal(inst)
    cur_score = _score(inst, cur)
    best, best_ms = cur, cur_score[0]
    traj = [(time.perf_counter() - t0, best_ms, best)]
    q, it, improved = min(q0, inst.n_tasks), 0, 0
    while (left := time_limit - (time.perf_counter() - t0)) > 0.05:
        kind = kinds[it % len(kinds)]
        free = _window(inst, cur, q, kind, rng)
        model, fixed = _subproblem(inst, W, cur, free, per_species, W.schedule(cur)[1], lex)
        model.hint(cur, W)
        solver = _solver(min(sub_time, left), workers, int(rng.integers(2**31)))
        status = solver.Solve(model.m)
        it += 1
        if status == cp_model.OPTIMAL:
            q = min(q + 1, q_max, inst.n_tasks)
        elif status == cp_model.FEASIBLE:
            q = max(q - 1, q_min)
        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            continue
        _, plan = _plan(inst, *model.extract(solver.Value, fixed))
        score = _score(inst, plan)
        if score[0] < cur_score[0] - 1e-9 or (score[0] < cur_score[0] + 1e-9 and score[1] <= cur_score[1] + 1e-9):
            cur, cur_score = plan, score
            ms = score[0]
            if ms < best_ms - 1e-9:
                best, best_ms = plan, ms
                improved += 1
                traj.append((time.perf_counter() - t0, ms, plan))
    return Result(best, best_ms, "LNS", math.nan, math.nan, time.perf_counter() - t0, time.process_time() - c0,
                  traj, it, improved)
