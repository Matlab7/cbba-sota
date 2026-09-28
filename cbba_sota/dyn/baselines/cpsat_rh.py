"""B3: rolling state-start CP-SAT-LNS (docs/trackD-spec.md Section 5.2).

Pinned copy: the model and the LNS loop follow ``cbba_sota/solvers/cpsat.py`` at commit **4cf5e04** (the Phase-1
CP-SAT-LNS, ``git show 4cf5e04:cbba_sota/solvers/cpsat.py``: free a window of tasks -- critical chain, spatial,
temporal or random -- keep every other coalition fixed, re-solve with a small CP-SAT model in which robots that may
take a freed task get a circuit over their fixed visits and candidate freed tasks, all others are fixed precedence
chains; the window grows after a proven-optimal sub-solve and shrinks after a timed-out one; a result replaces the
incumbent unless it is worse by (makespan, objective)). Adapted to the state start of the rolling horizon:

- robots start at their ready time and position (first-leg weights ``ready + dout``), robots with no task end at
  ``ready + dhome``; task starts respect ``est``;
- anchored (committed, residual-repaired) tasks are never freed, their coalitions and order are fixed; freed tasks
  may not precede an anchored task in a robot's circuit (key monotonicity) and a robot locked to its head task
  visits it first;
- deterministic budgets only: each sub-solve gets ``max_deterministic_time`` and the event stops once the summed
  deterministic time of its sub-solves reaches ``dtime`` (Light / Heavy tiers are calibrated to CPU once, spec
  3.6); ``workers > 1`` uses ``interleave_search`` so multi-worker runs stay deterministic;
- the start plan is SPARC's own warm start plus insertion floor (``RHPlanner`` with 0 iterations), i.e. the
  incumbent restricted to open tasks with new work inserted; plans are pruned to minimal covers (latest arrival
  first) and scored by the planner's forward pass.

Integer times are in units of 1 / ``SCALE`` rounded up, as in the pinned model, so CP schedules are feasible in real
time and the real forward pass is never later.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from itertools import pairwise

import numpy as np
from ortools.sat.python import cp_model

from cbba_sota.dyn import rh_kernels as K
from cbba_sota.dyn.executor import STRUCTURAL
from cbba_sota.dyn.planner import DynPlan, PlanState, Problem, RHPlanner, SearchConfig
from cbba_sota.dyn.sparc import RollingPolicy

__all__ = ["CPSATRH", "CPSATConfig", "lns"]

SCALE = 1000


@dataclass(frozen=True)
class CPSATConfig:
    dtime: float = 0.2  # deterministic time per event, summed over sub-solves (to be calibrated: spec 3.6)
    dtime0: float | None = None  # first plan (None: the same)
    sub_dtime: float = 0.05  # per sub-solve
    workers: int = 1  # 1 = Light; 8 = Heavy (interleaved, deterministic)
    q0: int = 12
    q_min: int = 4
    q_max: int = 40
    per_species: int = 4
    kinds: tuple[str, ...] = ("critical", "spatial", "temporal", "random")
    max_subsolves: int = 10_000


def _up(x: np.ndarray) -> np.ndarray:
    return np.ceil(np.round(x * SCALE, 6))


class _Weights:
    """Integer times of a kernel problem (rounded up, task-to-task at least one unit)."""

    def __init__(self, prob: Problem):
        a = prob.arrays
        dur, tt, da = a[K.I_DUR], a[K.I_TT], a[K.I_DA]
        ready, dout, dhome, est = a[K.I_READY], a[K.I_DOUT], a[K.I_DHOME], a[K.I_EST]
        with np.errstate(invalid="ignore"):
            self.w = np.maximum(_up(dur[:, None] + tt), 1).astype(np.int64)
            d0 = ready[:, None] + dout
            self.d0 = np.where(np.isfinite(d0), _up(np.where(np.isfinite(d0), d0, 0.0)), -1).astype(np.int64)
            self.d1 = _up(dur[None, :] + da).astype(np.int64)
            end = ready + dhome
            self.idle_end = np.where(np.isfinite(end), _up(np.where(np.isfinite(end), end, 0.0)), 0).astype(np.int64)
            self.est = np.where(np.isfinite(est), _up(np.where(np.isfinite(est), est, 0.0)), 0).astype(np.int64)

    def bounds(self, prob: Problem) -> tuple[np.ndarray, np.ndarray]:
        """Per task: earliest start and minimal return tail over the closest robot with each required trait (the
        pinned ``_bounds``, from state-start first legs; valid lower bounds by the triangle inequality)."""
        a = prob.arrays
        ab, req = a[K.I_AB], a[K.I_REQ]
        alive = np.isfinite(a[K.I_DHOME])
        Tk = prob.Tk
        est, tail = self.est.copy(), np.zeros(Tk, np.int64)
        for x in range(Tk):
            for k in np.flatnonzero(req[x] > 0):
                ok = (ab[:, k] > 0) & alive & (self.d0[:, x] >= 0)
                if ok.any():
                    est[x] = max(est[x], int(self.d0[ok, x].min()))
                    tail[x] = max(tail[x], int(self.d1[ok, x].min()))
        return est, tail

    def schedule(self, routes: list[list[int]], Tk: int) -> tuple[np.ndarray, int, np.ndarray]:
        """Integer forward pass over the global order implied by ``routes`` (each robot in order); returns starts,
        makespan and robot ends. ``routes`` must be consistent with one global order (plans from the kernel are)."""
        start = np.zeros(Tk, np.int64)
        pos = [0] * len(routes)
        members: dict[int, list[int]] = {}
        for a, r in enumerate(routes):
            for x in r:
                members.setdefault(x, []).append(a)
        done: set[int] = set()
        prev = [-1] * len(routes)
        remaining = sum(len(r) for r in routes)
        order = []
        while remaining:
            progressed = False
            for a, r in enumerate(routes):
                if pos[a] < len(r):
                    x = r[pos[a]]
                    if x in done:
                        continue
                    if all(pos[b] < len(routes[b]) and routes[b][pos[b]] == x for b in members[x]):
                        s = int(self.est[x])
                        for b in members[x]:
                            p = prev[b]
                            s = max(s, int(self.d0[b, x]) if p < 0 else int(start[p] + self.w[p, x]))
                        start[x] = s
                        done.add(x)
                        order.append(x)
                        for b in members[x]:
                            prev[b] = x
                            pos[b] += 1
                        remaining -= len(members[x])
                        progressed = True
            if not progressed:
                raise ValueError("routes are not consistent with a global order")
        ends = np.array([int(start[prev[a]] + self.d1[a, prev[a]]) if prev[a] >= 0 else int(self.idle_end[a])
                         for a in range(len(routes))], np.int64)
        return start, int(ends.max(initial=0)), ends


def _routes(prob: Problem, order: list[int], members: list[tuple[int, ...]]) -> list[list[int]]:
    out: list[list[int]] = [[] for _ in range(prob.A)]
    for x in order:
        for i in members[x]:
            out[i].append(x)
    return out


def _state(prob: Problem, sol: tuple) -> tuple[list[int], list[tuple[int, ...]]]:
    seq, mem, cnt, ni = sol[:4]
    order = [int(x) for x in seq[:ni[0]]]
    members = [tuple(int(i) for i in mem[x, :cnt[x]]) for x in range(prob.Tk)]
    return order, members


def _window(prob: Problem, sol: tuple, free: list[int], q: int, kind: str, rng: np.random.Generator) -> list[int]:
    if q >= len(free):
        return list(free)
    arr = np.asarray(free)
    tt = prob.arrays[K.I_TT]
    if kind == "random":
        return [int(x) for x in rng.choice(arr, q, replace=False)]
    if kind == "temporal":
        start = sol[4]
        seed = arr[rng.integers(len(arr))]
        return [int(x) for x in arr[np.argsort(np.abs(start[arr] - start[seed]), kind="stable")[:q]]]
    if kind == "critical":
        buf = np.zeros(prob.Tk, np.int64)
        L = K._critical_chain(sol, buf)
        fs = set(free)
        chain = [int(x) for x in buf[:L][::-1] if int(x) in fs]
        if len(chain) > q:
            lo = int(rng.integers(len(chain) - q + 1))
            chain = chain[lo:lo + q]
        out = list(dict.fromkeys(chain))
        if not out:
            out = [int(arr[rng.integers(len(arr))])]
        near = [iter(arr[np.argsort(tt[x, arr], kind="stable")]) for x in out]
        while len(out) < q:
            for it in near:
                for k in it:
                    if int(k) not in out:
                        out.append(int(k))
                        break
                if len(out) >= q:
                    break
        return out
    seed = arr[rng.integers(len(arr))]  # spatial
    return [int(x) for x in arr[np.argsort(tt[seed, arr], kind="stable")[:q]]]


def _detour(prob: Problem, route: list[int], a: int, x: int) -> float:
    """Cheapest extra travel for robot ``a`` to visit ``x`` after its anchored visits."""
    arrs = prob.arrays
    tt, da, dout, dhome = arrs[K.I_TT], arrs[K.I_DA], arrs[K.I_DOUT], arrs[K.I_DHOME]
    mode = prob.mode
    best = math.inf
    pts = [-1, *route, -2]
    for p, q in pairwise(pts):
        if q >= 0 and mode[q] != K.M_FREE:
            continue  # freed tasks go behind the anchored prefix
        leg_px = dout[a, x] if p == -1 else tt[p, x]
        leg_xq = da[a, x] if q == -2 else tt[x, q]
        if p == -1 and q == -2:
            leg_pq = dhome[a]
        elif p == -1:
            leg_pq = dout[a, q]
        elif q == -2:
            leg_pq = da[a, p]
        else:
            leg_pq = tt[p, q]
        best = min(best, leg_px + leg_xq - leg_pq)
    return best


class _Sub:
    """Sub-problem model (see the module docstring)."""

    def __init__(self, prob: Problem, W: _Weights, routes: list[list[int]], window: list[int], per_species: int,
                 ub: int):
        a = prob.arrays
        ab, req = a[K.I_AB], a[K.I_REQ]
        capl, ncap, mode, head = a[K.I_CAPL], a[K.I_NCAP], prob.mode, prob.head
        Tk, A = prob.Tk, prob.A
        wset = set(window)
        m = self.m = cp_model.CpModel()
        est, tail = W.bounds(prob)
        ub = max(int(ub), int((est + tail).max(initial=0)))
        self.s = [m.NewIntVar(int(est[x]), int(ub - tail[x]), f"s{x}") for x in range(Tk)]
        self.ms = m.NewIntVar(int((est + tail).max(initial=0)), int(ub), "ms")
        for x in range(Tk):
            m.Add(self.ms >= self.s[x] + int(tail[x]))
        species = prob.inst.species
        cand: dict[int, set[int]] = {x: set() for x in window}
        for i in range(A):
            for x in routes[i]:
                if x in wset:
                    cand[x].add(i)
        for x in window:
            capable = [int(i) for i in capl[x, :ncap[x]]]
            for sp in np.unique(species[capable]) if capable else []:
                agents = [i for i in capable if species[i] == sp and head[i] < 0 or
                          (species[i] == sp and head[i] >= 0 and mode[head[i]] != K.M_FREE)]
                agents.sort(key=lambda i: (_detour(prob, routes[i], i, x), i))
                cand[x] |= set(agents[:per_species])
        active = sorted(set().union(*cand.values())) if cand else []
        self.x: dict[tuple[int, int], cp_model.IntVar] = {}
        by_task: dict[int, list[tuple[int, cp_model.IntVar]]] = {x: [] for x in window}
        self.arc: dict[tuple[int, int, int], cp_model.IntVar] = {}
        for i in range(A):
            end = m.NewIntVar(0, int(ub), f"end{i}")
            m.Add(self.ms >= end)
            fixed = [x for x in routes[i] if x not in wset]
            if i not in active:  # fixed precedence chain
                if fixed:
                    m.Add(self.s[fixed[0]] >= int(W.d0[i, fixed[0]]))
                    for u, v in pairwise(fixed):
                        m.Add(self.s[v] >= self.s[u] + int(W.w[u, v]))
                    m.Add(end >= self.s[fixed[-1]] + int(W.d1[i, fixed[-1]]))
                elif W.idle_end[i] > 0:
                    m.Add(end >= int(W.idle_end[i]))
                continue
            opts = [x for x in window if i in cand[x]]
            nodes = fixed + opts
            node = {x: k + 1 for k, x in enumerate(nodes)}
            anchored_first = bool(fixed) and mode[fixed[0]] != K.M_FREE
            idle = m.NewBoolVar(f"idle{i}")
            lits = [(0, 0, idle)]
            if fixed:
                m.Add(idle == 0)
            else:
                m.Add(end >= int(W.idle_end[i])).OnlyEnforceIf(idle)
            fixed_set = set(fixed)
            for x in nodes:
                if x not in fixed_set:
                    v = m.NewBoolVar(f"x{i}_{x}")
                    lits.append((node[x], node[x], v.Not()))
                    m.AddImplication(v, idle.Not())
                    self.x[i, x] = v
                    by_task[x].append((i, v))
                if (not anchored_first or x == fixed[0]) and W.d0[i, x] >= 0:
                    lo = m.NewBoolVar("")
                    self.arc[i, -1, x] = lo
                    lits.append((0, node[x], lo))
                    m.Add(self.s[x] >= int(W.d0[i, x])).OnlyEnforceIf(lo)
                hi = m.NewBoolVar("")
                self.arc[i, x, -1] = hi
                lits.append((node[x], 0, hi))
                m.Add(end >= self.s[x] + int(W.d1[i, x])).OnlyEnforceIf(hi)
            arcs = set(pairwise(fixed)) | {(u, v) for u in nodes for v in opts if u != v}
            arcs |= {(v, u) for v in opts for u in fixed if mode[u] == K.M_FREE}
            for u, v in arcs:
                lit = m.NewBoolVar("")
                self.arc[i, u, v] = lit
                lits.append((node[u], node[v], lit))
                m.Add(self.s[v] >= self.s[u] + int(W.w[u, v])).OnlyEnforceIf(lit)
            m.AddCircuit(lits)
        cap = _species_cap(prob)
        for x in window:
            for k in np.flatnonzero(req[x] > 0):
                m.Add(sum(int(ab[i, k]) * v for i, v in by_task[x] if ab[i, k] > 0) >= int(req[x, k]))
            for sp in range(cap.shape[0]):
                vs = [v for i, v in by_task[x] if species[i] == sp]
                if len(vs) > cap[sp, x]:
                    m.Add(sum(vs) <= int(cap[sp, x]))
        m.Minimize(self.ms)
        self.routes, self.window = routes, window

    def hint(self, W: _Weights, Tk: int) -> None:
        start, ms, _ = W.schedule(self.routes, Tk)
        m = self.m
        for x, v in enumerate(self.s):
            m.AddHint(v, int(start[x]))
        m.AddHint(self.ms, ms)
        used = {(i, u, v) for i, r in enumerate(self.routes) for u, v in zip([-1, *r], [*r, -1])}
        on = {(i, x) for i, r in enumerate(self.routes) for x in r}
        for key, lit in self.arc.items():
            m.AddHint(lit, int(key in used))
        for key, v in self.x.items():
            m.AddHint(v, int(key in on))


def _species_cap(prob: Problem) -> np.ndarray:
    inst = prob.inst
    req = prob.arrays[K.I_REQ]
    first = [int(np.flatnonzero(inst.species == s)[0]) for s in range(inst.n_species)]
    has = inst.ab[first] > 0
    return np.where(has[:, None, :], req[None, :, :], 0).max(axis=2).astype(np.int64)


def _prune_minimal(prob: Problem, order: list[int], members: list[tuple[int, ...]], window: list[int],
                   lam: float) -> tuple:
    """Drop redundant members of the window tasks, in order, latest arrival first; returns the scheduled solution."""
    ab, req = prob.arrays[K.I_AB], prob.arrays[K.I_REQ]
    sol = prob.new_sol()
    members = list(members)
    prob.load(sol, order, members, lam)
    wset = set(window)
    for x in order:
        if x not in wset or len(members[x]) < 2:
            continue
        m = list(members[x])
        tot = ab[m].sum(axis=0)
        barr, mem = sol[6], sol[1]
        arr = {int(mem[x, k]): float(barr[x, k]) for k in range(len(m))}
        changed = False
        for i in sorted(m, key=lambda i: (arr.get(i, 0.0), i), reverse=True):
            if (tot - ab[i] >= req[x]).all():
                tot = tot - ab[i]
                m.remove(i)
                changed = True
        if changed:
            members[x] = tuple(sorted(m))
            prob.load(sol, order, members, lam)
    return sol


def lns(prob: Problem, sol: tuple, cfg: CPSATConfig, seed: int, lam: float, dtime: float) -> tuple[tuple, dict]:
    """CP-SAT-LNS over the free tasks of ``prob`` from solution ``sol`` until ``dtime`` deterministic time."""
    rng = np.random.default_rng(seed)
    W = _Weights(prob)
    cur = sol
    free = [x for x in range(prob.Tk) if prob.mode[x] == K.M_FREE and prob.fcnt[x] == 0 and prob.removable[x]]
    stats = {"subsolves": 0, "improvements": 0, "dtime": 0.0, "optimal": 0}
    if not free:
        return cur, stats
    q = min(cfg.q0, len(free))
    it = 0
    while stats["dtime"] < dtime - 1e-12 and it < cfg.max_subsolves:
        kind = cfg.kinds[it % len(cfg.kinds)]
        it += 1
        order, members = _state(prob, cur)
        routes = _routes(prob, order, members)
        window = _window(prob, cur, free, q, kind, rng)
        ub = W.schedule(routes, prob.Tk)[1]
        sub = _Sub(prob, W, routes, window, cfg.per_species, int(ub))
        sub.hint(W, prob.Tk)
        solver = cp_model.CpSolver()
        solver.parameters.max_deterministic_time = max(1e-4, min(cfg.sub_dtime, dtime - stats["dtime"]))
        solver.parameters.num_workers = int(cfg.workers)
        if cfg.workers > 1:
            solver.parameters.interleave_search = True
        solver.parameters.random_seed = int(rng.integers(2**31))
        status = solver.Solve(sub.m)
        stats["subsolves"] += 1
        stats["dtime"] += float(solver.deterministic_time)
        if status == cp_model.OPTIMAL:
            stats["optimal"] += 1
            q = min(q + 1, cfg.q_max, len(free))
        elif status == cp_model.FEASIBLE:
            q = max(q - 1, min(cfg.q_min, len(free)))
        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            continue
        new_members = list(members)
        wset = set(window)
        for x in window:
            new_members[x] = tuple(sorted(i for (i, y), v in sub.x.items() if y == x and solver.Value(v)))
        starts = np.array([solver.Value(v) for v in sub.s], dtype=float)
        anchored = [x for x in order if prob.mode[x] != K.M_FREE]
        rest = sorted((x for x in range(prob.Tk) if prob.mode[x] == K.M_FREE and new_members[x]),
                      key=lambda x: (starts[x], order.index(x) if x in order else prob.Tk))
        cand = _prune_minimal(prob, anchored + rest, new_members, list(wset), lam)
        fc, fu = cand[10], cur[10]
        if fc[3] < fu[3] - 0.5 or (fc[3] < fu[3] + 0.5 and (fc[0] < fu[0] - 1e-9 or
                                                            (fc[0] <= fu[0] + 1e-9 and fc[2] <= fu[2] + 1e-9))):
            if fc[0] < fu[0] - 1e-9:
                stats["improvements"] += 1
            cur = cand
    return cur, stats


class CPSATRH(RollingPolicy):
    """B3 at the Light tier (``workers=1``) or Heavy tier (``workers=8``, larger ``dtime``)."""

    name = "RH-CP-SAT-LNS"

    def __init__(self, config: CPSATConfig | None = None, seed_base: int = 0, search: SearchConfig | None = None,
                 triggers: frozenset[str] = STRUCTURAL):
        super().__init__(triggers, seed_base)
        self.cfg = config or CPSATConfig()
        search = search or SearchConfig()
        if search.keys != "monotone":
            raise NotImplementedError("the CP-SAT sub-model implements the monotone key rule only")
        self.planner = RHPlanner(search)
        self.stats: list[dict] = []

    def solve(self, state: PlanState, scope: np.ndarray | None, seed: int, first: bool) -> DynPlan:
        c0 = time.process_time()
        base, prob, sol = self.planner.plan(state, self.incumbent, scope, seed, iters=0, return_problem=True)
        if sol is None:
            return base
        dtime = self.cfg.dtime0 if first and self.cfg.dtime0 is not None else self.cfg.dtime
        best, st = lns(prob, sol, self.cfg, seed, self.planner.cfg.lam, dtime)
        self.stats.append(st)
        plan = self.planner.extract(prob, best, base, time.process_time() - c0)
        plan.iterations = st["subsolves"]
        return plan
