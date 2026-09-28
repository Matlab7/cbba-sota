"""CTAS-D (Fu et al., "Robust task scheduling for heterogeneous robot teams under capability uncertainty", T-RO 2022)
as the HeteroMRTA benchmark (RA-L 2025) ran it: planner mode TEAMPLANNER_CONDET of the MARMot Lab C++/Gurobi code
(``formVarNameCost`` + ``formCommonModel`` + ``formContinuousModel`` + ``formContinuousDetEnergy``, flows rounded
and split into routes by ``FlowConverter``), rebuilt from an ``Instance`` and solved by an open MIP backend through
OR-Tools ``pywraplp`` (``CP_SAT``, ``HIGHS`` or ``SCIP``).

Graph, per species ``k`` with ``N_k`` agents (as the benchmark's yamlGenerator.py writes graph.yaml): the tasks, a
start node ``s_k = T + k`` and an end node ``u_k = T + S + k`` at the species depot; arcs ``i -> j`` between distinct
tasks, ``s_k -> j`` and ``j -> u_k``; energy cost = euclidean distance, time cost = travel time (``Instance.tt``,
``Instance.da``); node time = task duration (0 at ``s_k`` and ``u_k``).

Model (C++ names; one continuous flow per species instead of one route per agent):
- ``x[k, e]`` in [0, N_k]: flow of species ``k`` on arc ``e``; ``xr[k, e]`` binary with ``xr <= x <= N_k xr``
  (X_Xr1/2), so a used arc carries at least one agent.
- ``y[k, j]`` in [0, N_k] = inflow of ``k`` at task ``j`` (RelationXY); ``yr[k, j]`` binary, ``yr <= y <= N_k yr``.
- FlowInOut (inflow = outflow at every task), FlowLessOne (inflow <= N_k, outflow of ``s_k`` <= N_k), FlowLessStart
  (inflow <= outflow of ``s_k``; redundant).
- ``q[n]`` in [0, MAXTIME]: start time of task ``n``, shared by all species; TimeStart ``q[s_k] = 0``; TimeEdge
  ``q[i] + time(i) + time(i, j) <= q[j] + LARGETIME (1 - xr[k, (i, j)])`` on every arc of every species, so a task
  starts after every used arc into it has arrived (synchronisation).
- Coverage: TaskComplete ``z[j] = 1``; per required trait ``c`` of task ``j`` (one AND clause with one OR literal in
  the benchmark's task_param.yaml) TaskReqAlphaDef ``alpha[j, c] = sum_k cap[k, c] y[k, j]`` in [0, sumCap_c] with
  ``sumCap_c = sum_k cap[k, c] N_k`` (cumulative capability), ``w[j, c]`` binary, TaskReqOrL ``req w <= alpha``,
  TaskReqOrG ``alpha <= req - 1 + max(1, sumCap_c - req + 1) w``, TaskReqAnd ``z[j] <= w[j, c]``.
- TaskReqUnrelavant: ``y[k, j] = 0`` if species ``k`` has none of task ``j``'s required traits.
- Energy (``energy=True``): ``g[k, n]`` in [0, MAXENG], EngEdge ``g[k, i] + energy(i, j) <= g[k, j] + MAXENG
  (1 - xr[k, (i, j)])``, EngNode ``g[k, s_k] = 0`` and ``g[k, n] <= ENG_CAP``.
- Objective ``sum_{k, e} energy(e) x[k, e] + TIME_PENALTY ms (+ G_COST sum g)`` with the makespan ``ms = q[u_k]``
  for every species.

The objective is taken from the 50 shipped RALTestSet ``results.yaml`` (Gurobi, 600 s), not from the cloned C++ code:
on all 50, ``objVal = energyCost + timeCost + 1e-4 sum(gVar)`` (to print precision), ``energyCost = sum xVar * edge
energy``, ``timeCost = 100 qMax`` and ``qVar[u_k] = qMax`` for every species. The clone has no ``qMax`` and puts
``timePenalty`` on every ``q[u_k]`` (the summed species return times); the benchmark authors' makespan version is not
in it, and ``ms`` with ``q[u_k] = ms`` reproduces its objective. ``check_shipped`` plugs a shipped solution into this
model: on all 50 the objective equals ``objVal`` (5e-5) and every row holds up to 1.5e-5 (values printed with 6
decimals; big-M rows whose ``xr`` Gurobi left within its integrality tolerance of 1), EngEdge rows up to 0.5 (the same
with M = 1e8).

Deviations from the C++ model:
- (species, task) pairs excluded by TaskReqUnrelavant get no variables: ``y = 0`` forces ``x = xr = 0`` on every arc
  of the task and leaves its TimeEdge / EngEdge rows slack (same optimum).
- ``energy=False`` (the default) drops ``g`` with EngEdge / EngNode. With engCap = 1e6 they constrain nothing; they
  only add ``1e-4 sum g`` (the longest energy path to each node, ~2e-6 of objVal on RALTestSet), a tie-breaker far
  below one makespan unit (100) or one distance unit (1), for 28-31% more rows with a big-M of 1e8 (an ``xr`` within
  a solver's integrality tolerance of 1 frees ``g`` by up to 1e3). Measured at 30 s x 8 threads: no effect with
  CP_SAT (RALTestSet env_0..4, mean makespan 23.336 without, 23.333 with), worse with HIGHS (env_0..1: 64.1 vs
  71.1). ``objective`` then excludes that term; ``energy=True`` builds the complete model.
- The vehicle ``engCost`` term on ``y`` is 0 in the benchmark inputs and is omitted.
- The time limit covers model building (``t0``); Gurobi's 600 s covered the solve only.
- CP_SAT solves a MIP over integers after multiplying every continuous variable by ``mip_var_scaling``: we set it to
  ``CPSAT_SCALING`` = 1000, i.e. times (and flows) on a 1e-3 grid. Its solutions are feasible for the model above,
  its optimum can be ~1e-3 per arc of the critical chain worse, and its bound holds for the grid model only. (With
  the default scaling of 1, start times are integers.)

Backends (defaults otherwise): CP_SAT runs ``threads`` workers; SCIP with ``threads > 1`` runs its concurrent mode
(independent SCIP runs, whose late incumbents can miss the final synchronisation and be lost); HIGHS gets ``threads``
through its own option (``SetNumThreads`` does not reach it), and the first HIGHS solve of a process fixes it (HiGHS
keeps one global scheduler and refuses to run with another count). OR-Tools 9.15 drops HiGHS's incumbent when the
time limit stops it (status 99), so HiGHS writes every improving solution to a temporary file
(``mip_improving_solution_file``) and the last one is used; no bound is reported then. CP_SAT is the default: on
RALTestSet env_0..9 (60 s, 8 threads; docs/results/ctas/ralt_check.txt) it had an incumbent within 0.25 s on every
instance and ended at 0.97x the shipped Gurobi 600 s makespans, HIGHS needed 4-19 s and ended at 2.2x, SCIP found
incumbents on 5 of 10 after 29-60 s (in 60 s its root cut loop alone does not finish on env_0 with one thread).

Conversion (``printSolution`` -> ``getPathCover``): per species, ``FlowConverter`` keeps the arcs at ``s_k`` or
``u_k`` and those with ``x >= 0.1``, rounds to the least-energy integer flow with ``f >= ceil(x - 1e-4)`` conserved at
every task (an LP with a network matrix: GLOP's simplex vertex is integral, as Gurobi's was) and splits ``f`` into
``F = outflow(s_k)`` start-to-end paths minimising the largest path energy (a small MILP, SCIP here). The benchmark
gives the ``F`` paths of species ``k`` to its agents in id order (CTAS-D.py); ``F > N_k`` has no such assignment (the
benchmark script would raise) and fails here. Plan keys are the MILP start times ``q`` (every task-to-task arc of
a route has ``xr = 1``, so ``q`` increases along it), the plan is pruned to a minimal cover (``prune_to_minimal``; the
benchmark replayed unpruned routes and the env drops redundant members as they come) and scored by ``evaluate``
(= env replay). Ties between optimal roundings or decompositions may resolve differently from Gurobi; the forward-pass
start times, hence the makespan before pruning, do not depend on the decomposition (all members of a task leave it
together).
"""
from __future__ import annotations

import contextlib
import math
import os
import tempfile
import time
from collections import defaultdict
from dataclasses import dataclass
from itertools import pairwise
from os import PathLike
from pathlib import Path

import numpy as np
from ortools.linear_solver import linear_solver_pb2, pywraplp

from cbba_sota.hetero.evaluate import evaluate
from cbba_sota.hetero.instance import Instance
from cbba_sota.hetero.plan import Plan

BACKENDS = ("CP_SAT", "HIGHS", "SCIP")
# planner_param.yaml / vehicle_param.yaml of the benchmark (yamlGenerator.py), hard-coded constants of the C++ code
TIME_PENALTY = 100.0  # timePenalty: objective weight of the makespan
LARGETIME = 1e4  # big-M of TimeEdge
MAXTIME = 1e3  # upper bound of every q
MAXENG = 1e8  # big-M of EngEdge and upper bound of every g
ENG_CAP = 1e6  # engCap of every species
G_COST = 1e-4  # objective weight of every g (formVarNameCost, "TODO, HARD CODE")
FLOW_EPS = 0.1  # getPathCover: task-to-task arcs with less flow are dropped before rounding
ROUND_EPS = 1e-4  # FlowConverter: lower bound ceil(x - ROUND_EPS) of the rounded flow
FLOW_UB = 1e5  # FlowConverter: upper bound of the rounded flow
COVER_LIMIT = 10.0  # seconds for each path-cover MILP (the C++ code gives it solverMaxTime)
CPSAT_SCALING = 1000  # CP-SAT mip_var_scaling: continuous variables on a 1 / CPSAT_SCALING grid

_HIGHS_THREADS: list[int] = []  # HiGHS's global scheduler keeps the thread count of its first run in the process
_STATUS = {pywraplp.Solver.OPTIMAL: "OPTIMAL", pywraplp.Solver.FEASIBLE: "FEASIBLE",
           pywraplp.Solver.INFEASIBLE: "INFEASIBLE", pywraplp.Solver.UNBOUNDED: "UNBOUNDED",
           pywraplp.Solver.ABNORMAL: "ABNORMAL", pywraplp.Solver.MODEL_INVALID: "MODEL_INVALID",
           pywraplp.Solver.NOT_SOLVED: "NOT_SOLVED", 99: "NOT_SOLVED"}  # 99: OR-Tools' HiGHS wrapper after a limit


class _Failed(Exception):
    """A stage after the MILP failed; ``args[0]`` is the status reported."""


@dataclass
class Result:
    plan: Plan | None  # key-ordered minimal cover from the rounded, split flows; None on failure
    makespan: float  # evaluate() of ``plan`` (equals env replay), inf without a plan
    status: str  # MILP status (OPTIMAL, FEASIBLE, INFEASIBLE, NOT_SOLVED, ...), TIMEOUT or a conversion failure
    objective: float  # CTAS objective of the MILP incumbent (see ``energy``), nan without one
    bound: float  # MILP dual bound, nan if the backend reports none
    wall_s: float  # from ``t0``: building, solve and conversion
    cpu_s: float  # process CPU time (all solver threads)
    qmax: float = math.nan  # the MILP's makespan ``ms`` (>= ``makespan`` unless rounding added depot arcs)
    build_s: float = math.nan  # from ``t0`` to the end of model building


class Model:
    """The CONDET MILP of ``inst`` in a fresh ``pywraplp`` solver (see the module docstring).

    ``arcs[k]`` lists species ``k``'s arcs ``(i, j)`` (tasks ``< T``, ``s_k = T + k``, ``u_k = T + S + k``) in the
    order of ``x[k]`` / ``xr[k]``; ``y``, ``yr`` are keyed by (species, task), ``alpha``, ``w`` by (task, trait),
    ``g`` by (species, node). Building stops with ``TimeoutError`` once ``perf_counter`` passes ``deadline``."""

    def __init__(self, inst: Instance, backend: str = "CP_SAT", energy: bool = False, deadline: float = math.inf):
        solver = pywraplp.Solver.CreateSolver(backend) if backend in BACKENDS else None
        if solver is None:
            raise ValueError(f"backend {backend!r} is not one of the available {BACKENDS}")
        self.inst, self.solver, self.backend, self.energy = inst, solver, backend, energy
        T, S = inst.n_tasks, inst.n_species
        first = [np.flatnonzero(inst.species == k) for k in range(S)]
        self.n = np.array([len(a) for a in first])  # N_k
        self.cap = np.array([inst.ab[a[0]] if len(a) else np.zeros(inst.n_traits) for a in first])  # [S, K]
        self.depot_time = np.array([inst.da[a[0]] if len(a) else np.zeros(T) for a in first])  # [S, T]
        self.relevant = ((self.cap[:, None, :] > 0.1) & (inst.req != 0)[None]).any(axis=2)  # [S, T]
        inf = solver.infinity()
        obj = solver.Objective()
        obj.SetMinimization()

        def row(lb: float, ub: float, terms) -> None:
            ct = solver.Constraint(lb, ub)
            for v, c in terms:
                ct.SetCoefficient(v, c)

        self.q = [solver.NumVar(0.0, MAXTIME, f"q{n}") for n in range(T + 2 * S)]
        self.ms = solver.NumVar(0.0, MAXTIME, "ms")
        obj.SetCoefficient(self.ms, TIME_PENALTY)
        self.arcs: list[list[tuple[int, int]]] = []
        self.x: list[list] = []
        self.xr: list[list] = []
        self.y, self.yr, self.g = {}, {}, {}
        for k in range(S):
            N, s, u = float(self.n[k]), T + k, T + S + k
            tasks = np.flatnonzero(self.relevant[k]).tolist() if N > 0 else []
            arcs = [(i, j) for i in tasks for j in tasks if i != j] + [(s, j) for j in tasks] + [(j, u) for j in tasks]
            xs, xrs = [solver.NumVar(0.0, N, "") for _ in arcs], [solver.BoolVar("") for _ in arcs]
            self.arcs.append(arcs)
            self.x.append(xs)
            self.xr.append(xrs)
            if energy:
                for n in [s, *tasks, u]:
                    self.g[k, n] = v = solver.NumVar(0.0, MAXENG, "")
                    obj.SetCoefficient(v, G_COST)
                    row(0.0, 0.0 if n == s else ENG_CAP, [(v, 1.0)])  # EngNode
            for (i, j), x, xr in zip(arcs, xs, xrs):
                t = self.time(k, i, j)
                obj.SetCoefficient(x, t * inst.speed)  # energy = distance
                row(0.0, inf, [(x, 1.0), (xr, -1.0)])  # X_Xr1
                row(-inf, 0.0, [(x, 1.0), (xr, -N)])  # X_Xr2
                node_time = float(inst.dur[i]) if i < T else 0.0
                row(-inf, LARGETIME - node_time - t, [(self.q[i], 1.0), (self.q[j], -1.0), (xr, LARGETIME)])  # TimeEdge
                if energy:
                    row(-inf, MAXENG - t * inst.speed, [(self.g[k, i], 1.0), (self.g[k, j], -1.0), (xr, MAXENG)])
            ins, outs = defaultdict(list), defaultdict(list)
            for (i, j), x in zip(arcs, xs):
                outs[i].append(x)
                ins[j].append(x)
            row(-inf, N, [(v, 1.0) for v in outs[s]])  # FlowLessOne at s_k
            for j in tasks:
                row(0.0, 0.0, [(v, 1.0) for v in ins[j]] + [(v, -1.0) for v in outs[j]])  # FlowInOut
                row(-inf, N, [(v, 1.0) for v in ins[j]])  # FlowLessOne
                row(-inf, 0.0, [(v, 1.0) for v in ins[j]] + [(v, -1.0) for v in outs[s]])  # FlowLessStart
                self.y[k, j] = y = solver.NumVar(0.0, N, "")
                self.yr[k, j] = yr = solver.BoolVar("")
                row(0.0, 0.0, [(v, 1.0) for v in ins[j]] + [(y, -1.0)])  # RelationXY
                row(0.0, inf, [(y, 1.0), (yr, -1.0)])  # Y_Yr1
                row(-inf, 0.0, [(y, 1.0), (yr, -N)])  # Y_Yr2
            row(0.0, 0.0, [(self.q[s], 1.0)])  # TimeStart
            row(0.0, 0.0, [(self.q[u], 1.0), (self.ms, -1.0)])  # makespan (the benchmark's version)
            if time.perf_counter() > deadline:
                raise TimeoutError("model building ran out of time")
        sum_cap = (self.cap * self.n[:, None]).sum(axis=0)
        self.z = [solver.BoolVar("") for _ in range(T)]
        self.alpha, self.w = {}, {}
        for j in range(T):
            row(1.0, 1.0, [(self.z[j], 1.0)])  # TaskComplete
            for c in np.flatnonzero(inst.req[j] != 0).tolist():
                req, big = float(inst.req[j, c]), max(1.0, float(sum_cap[c] - inst.req[j, c] + 1))
                self.alpha[j, c] = alpha = solver.NumVar(0.0, float(sum_cap[c]), "")
                self.w[j, c] = w = solver.BoolVar("")
                row(0.0, 0.0, [(alpha, 1.0)] + [(self.y[k, j], -float(self.cap[k, c])) for k in range(S)
                                                if self.cap[k, c] > 0.1 and (k, j) in self.y])  # TaskReqAlphaDef
                row(-inf, 0.0, [(w, req), (alpha, -1.0)])  # TaskReqOrL
                row(-inf, req - 1.0, [(w, -big), (alpha, 1.0)])  # TaskReqOrG
                row(-inf, 0.0, [(self.z[j], 1.0), (w, -1.0)])  # TaskReqAnd
        if time.perf_counter() > deadline:
            raise TimeoutError("model building ran out of time")

    def time(self, k: int, i: int, j: int) -> float:
        """Travel time of species ``k``'s arc ``i -> j`` (energy = time * speed)."""
        T = self.inst.n_tasks
        if i < T and j < T:
            return float(self.inst.tt[i, j])
        return float(self.depot_time[k, j] if j < T else self.depot_time[k, i])

    def solution(self) -> np.ndarray:
        """Values of all variables (by index) in the solver's solution."""
        return np.array([v.solution_value() for v in self.solver.variables()])

    def split(self, values: np.ndarray) -> tuple[list[np.ndarray], np.ndarray, float]:
        """(per species the flows ``x``, task start times ``q[:T]``, ``ms``) of an assignment by variable index."""
        x = [values[[v.index() for v in xs]] for xs in self.x]
        return x, values[[v.index() for v in self.q[:self.inst.n_tasks]]], float(values[self.ms.index()])

    # --- assignments ------------------------------------------------------------------------------------------

    def assignment(self, plan: Plan) -> dict[int, float]:
        """Values of every model variable (by index) encoding a plan whose members all contribute: one unit of
        species flow per agent and route arc, ``q`` = the plan's forward-pass start times, ``ms`` = its makespan,
        ``g`` = longest energy path from ``s_k``. Raises ``ValueError`` for a route arc the model does not have."""
        inst = self.inst
        T, S = inst.n_tasks, inst.n_species
        sched = evaluate(inst, plan)
        flow = [defaultdict(float) for _ in range(S)]
        for a, route in enumerate(plan.routes()):
            if route:
                k = int(inst.species[a])
                for i, j in pairwise([T + k, *route, T + S + k]):
                    flow[k][i, j] += 1.0
        vals: dict[int, float] = {}
        for n, v in enumerate(self.q):
            vals[v.index()] = float(sched.start[n]) if n < T else 0.0 if n < T + S else sched.makespan
        vals[self.ms.index()] = sched.makespan
        for k in range(S):
            pos = {arc: e for e, arc in enumerate(self.arcs[k])}
            if missing := set(flow[k]) - set(pos):
                raise ValueError(f"species {k} uses arcs outside the model: {sorted(missing)}")
            for arc, f in flow[k].items():
                vals[self.x[k][pos[arc]].index()] = f
                vals[self.xr[k][pos[arc]].index()] = 1.0
            inflow = defaultdict(float)
            for (_, j), f in flow[k].items():
                inflow[j] += f
            for (kk, j), y in self.y.items():
                if kk == k:
                    vals[y.index()] = inflow[j]
                    vals[self.yr[k, j].index()] = float(inflow[j] > 0)
            if self.energy:
                g = _longest(flow[k], T + k, lambda i, j, k=k: self.time(k, i, j) * inst.speed)
                for (kk, n), v in self.g.items():
                    if kk == k:
                        vals[v.index()] = g.get(n, 0.0)
        for z in self.z:
            vals[z.index()] = 1.0
        for (j, c), alpha in self.alpha.items():
            total = sum(self.cap[k, c] * vals[self.y[k, j].index()] for k in range(S)
                        if (k, j) in self.y and self.cap[k, c] > 0.1)
            vals[alpha.index()] = float(total)
            vals[self.w[j, c].index()] = float(total >= inst.req[j, c])
        return vals

    def check(self, values: dict[int, float]) -> tuple[float, float]:
        """(largest violation of a bound, an integrality or a row, objective) of an assignment given by variable
        index (unset variables are 0), from the exported model: independent of the backend."""
        proto = linear_solver_pb2.MPModelProto()
        self.solver.ExportModelToProto(proto)
        v = np.zeros(len(proto.variable))
        for idx, val in values.items():
            v[idx] = val
        lb = np.array([p.lower_bound for p in proto.variable])
        ub = np.array([p.upper_bound for p in proto.variable])
        integer = np.array([p.is_integer for p in proto.variable])
        viol = max(0.0, float((lb - v).max(initial=0)), float((v - ub).max(initial=0)),
                   float(np.abs(v - np.round(v))[integer].max(initial=0)))
        for ct in proto.constraint:
            act = float(v[list(ct.var_index)] @ np.array(ct.coefficient)) if ct.var_index else 0.0
            viol = max(viol, ct.lower_bound - act, act - ct.upper_bound)
        c = np.array([p.objective_coefficient for p in proto.variable])
        return viol, float(proto.objective_offset + c @ v)


def _longest(flow: dict[tuple[int, int], float], source: int, weight) -> dict[int, float]:
    """Longest ``weight`` path from ``source`` to every node of the (acyclic) support of ``flow``."""
    succ, indeg = defaultdict(list), defaultdict(int)
    for (i, j), f in flow.items():
        if f > 0:
            succ[i].append(j)
            indeg[j] += 1
    dist, ready = {source: 0.0}, [source]
    while ready:
        i = ready.pop()
        for j in succ[i]:
            dist[j] = max(dist.get(j, -math.inf), dist[i] + weight(i, j))
            indeg[j] -= 1
            if indeg[j] == 0:
                ready.append(j)
    return dist


# --- FlowConverter ---------------------------------------------------------------------------------------------


def _round(arcs: list[tuple[int, int]], x: np.ndarray, energy: np.ndarray, s: int, u: int) -> np.ndarray:
    """``FlowConverter::formRoundProblem`` + ``optimizeRound``: least-energy flow ``f >= ceil(x - ROUND_EPS)`` on the
    kept arcs (at ``s`` or ``u``, or ``x >= FLOW_EPS``; ``f = 0`` elsewhere), conserved at every other node."""
    keep = np.array([i == s or j == u or xv >= FLOW_EPS for (i, j), xv in zip(arcs, x)], dtype=bool)
    f = np.zeros(len(arcs))
    lp = pywraplp.Solver.CreateSolver("GLOP")
    fv = {e: lp.NumVar(max(0.0, math.ceil(x[e] - ROUND_EPS)), FLOW_UB, "") for e in np.flatnonzero(keep).tolist()}
    balance: dict[int, pywraplp.Constraint] = {}
    for e, v in fv.items():
        for node, sign in ((arcs[e][0], -1.0), (arcs[e][1], 1.0)):
            if node not in (s, u):
                if node not in balance:
                    balance[node] = lp.Constraint(0.0, 0.0)
                balance[node].SetCoefficient(v, sign)
        lp.Objective().SetCoefficient(v, float(energy[e]))
    lp.Objective().SetMinimization()
    if lp.Solve() != pywraplp.Solver.OPTIMAL:
        raise _Failed("ROUNDING")
    for e, v in fv.items():
        f[e] = round(v.solution_value())
    net = defaultdict(float)
    for (i, j), fe in zip(arcs, f):
        net[i] -= fe
        net[j] += fe
    if any(abs(b) > 0.5 for node, b in net.items() if node not in (s, u)):
        raise _Failed("ROUNDING")  # a non-integral LP vertex (cannot happen for a network matrix)
    return f


def _split(arcs: list[tuple[int, int]], f: np.ndarray, energy: np.ndarray, s: int, u: int,
           time_limit: float) -> list[list[int]]:
    """``FlowConverter::formCoverProblem`` + ``optimizeCover`` + ``getPath``: split the integer flow ``f`` into
    ``outflow(s)`` binary ``s``-``u`` paths (conserved per path, arc sums equal to ``f``) with the least largest path
    energy (SCIP; a greedy split if it finds nothing in ``time_limit``). Returns the tasks of each path."""
    support = np.flatnonzero(f > 0.5).tolist()
    n_paths = round(float(sum(f[e] for e in support if arcs[e][0] == s)))
    if n_paths == 0:
        return []
    mip = pywraplp.Solver.CreateSolver("SCIP")
    c = {(p, e): mip.BoolVar("") for p in range(n_paths) for e in support}
    top = mip.NumVar(0.0, 1e10, "")
    for e in support:
        ct = mip.Constraint(f[e], f[e])  # EdgeFlowSum
        for p in range(n_paths):
            ct.SetCoefficient(c[p, e], 1.0)
    nodes = {n for e in support for n in arcs[e]} - {s, u}
    for p in range(n_paths):
        for n in nodes:  # FlowInOut per path
            ct = mip.Constraint(0.0, 0.0)
            for e in support:
                if arcs[e][1] == n:
                    ct.SetCoefficient(c[p, e], 1.0)
                elif arcs[e][0] == n:
                    ct.SetCoefficient(c[p, e], -1.0)
        start = mip.Constraint(1.0, 1.0)  # PathFlow
        load = mip.Constraint(-mip.infinity(), 0.0)  # MaxEng
        load.SetCoefficient(top, -1.0)
        for e in support:
            if arcs[e][0] == s:
                start.SetCoefficient(c[p, e], 1.0)
            load.SetCoefficient(c[p, e], float(energy[e]))
    mip.Objective().SetCoefficient(top, 1.0)
    mip.Objective().SetMinimization()
    mip.SetTimeLimit(max(1, int(1000 * time_limit)))
    if mip.Solve() in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        choice = [[e for e in support if c[p, e].solution_value() > 0.5] for p in range(n_paths)]
    else:  # greedy split of the flow
        left = {e: round(float(f[e])) for e in support}
        choice = []
        for _ in range(n_paths):
            path, node = [], s
            while node != u:
                e = next((e for e in support if arcs[e][0] == node and left[e] > 0), None)
                if e is None or len(path) > len(support):
                    raise _Failed("DECOMPOSITION")
                left[e] -= 1
                path.append(e)
                node = arcs[e][1]
            choice.append(path)
    out = []
    for path in choice:
        nxt = {arcs[e][0]: arcs[e][1] for e in path}
        if len(nxt) != len(path) or s not in nxt:
            raise _Failed("DECOMPOSITION")  # a node left twice: a cycle in the flow
        tasks, node = [], nxt[s]
        while node != u and node in nxt and len(tasks) <= len(path):
            tasks.append(node)
            node = nxt[node]
        if node != u or len(tasks) + 1 != len(path):
            raise _Failed("DECOMPOSITION")
        out.append(tasks)
    return out


def convert(model: Model, x: list[np.ndarray], q: np.ndarray, time_limit: float = COVER_LIMIT) -> Plan:
    """Plan from a MILP solution: rounded and split species flows, paths given to the species' agents in id order,
    keys = MILP start times ``q`` (a topological order of the routes if ``q`` does not order them strictly), pruned
    to a minimal cover. Raises ``_Failed`` with the stage that failed."""
    inst = model.inst
    T, S = inst.n_tasks, inst.n_species
    routes: list[list[int]] = [[] for _ in range(inst.n_agents)]
    for k in range(S):
        arcs = model.arcs[k]
        if not arcs:
            continue
        energy = np.array([model.time(k, i, j) * inst.speed for i, j in arcs])
        f = _round(arcs, x[k], energy, T + k, T + S + k)
        paths = _split(arcs, f, energy, T + k, T + S + k, time_limit)
        agents = np.flatnonzero(inst.species == k)
        if len(paths) > len(agents):
            raise _Failed("ROUNDING_EXCEEDS_AGENTS")
        for a, path in zip(agents, paths):
            routes[a] = path
    members: list[list[int]] = [[] for _ in range(T)]
    for a, route in enumerate(routes):
        for j in route:
            members[j].append(a)
    if all(q[a] < q[b] for r in routes for a, b in pairwise(r)):
        plan = Plan(members, q, inst.n_agents)
    else:
        try:
            plan = Plan.from_routes(routes, T)
        except ValueError:
            raise _Failed("DECOMPOSITION") from None
    if not plan.covers(inst).all():
        raise _Failed("UNCOVERED")
    return plan.prune_to_minimal(inst)


def _highs_incumbent(path: str, n_vars: int) -> tuple[float, np.ndarray] | None:
    """(objective, values by variable index) of the last complete improving solution HiGHS wrote to ``path``
    (blocks of ``Objective <value>``, ``# Columns <n>`` and one ``<name> <value>`` line per column, in variable
    order); None if there is none."""
    text = Path(path).read_text() if os.path.exists(path) else ""
    for block in reversed(text.split("Objective ")[1:]):
        lines = block.splitlines()
        if len(lines) >= 2 + n_vars and lines[1].split()[-1] == str(n_vars):
            with contextlib.suppress(ValueError, IndexError):
                return float(lines[0]), np.array([float(line.rsplit(None, 1)[1]) for line in lines[2:2 + n_vars]])
    return None


def solve(inst: Instance, time_limit: float, backend: str = "CP_SAT", threads: int = 8, hint: Plan | None = None,
          energy: bool = False, log: bool = False, t0: float | None = None) -> Result:
    """Build and solve the CTAS-D model within ``time_limit`` seconds from ``t0`` (default now), building included.
    ``hint``: a complete MILP hint from a plan (CP_SAT and SCIP; ``SetHint`` crashes OR-Tools 9.15's HiGHS
    interface); the original has no warm start. ``log`` prints the backend log to the process stdout. A plan is
    returned only when the MILP has an incumbent and its conversion succeeds; otherwise ``plan`` is None and
    ``status`` says why."""
    t0 = time.perf_counter() if t0 is None else t0
    c0 = time.process_time()
    deadline = t0 + time_limit
    if hint is not None and backend == "HIGHS":
        raise ValueError("HIGHS takes no hint: SetHint crashes OR-Tools 9.15's HiGHS interface")

    def result(status: str, plan: Plan | None = None, objective: float = math.nan, bound: float = math.nan,
               qmax: float = math.nan, build: float = math.nan) -> Result:
        ms = evaluate(inst, plan).makespan if plan is not None else math.inf
        return Result(plan, ms, status, objective, bound, time.perf_counter() - t0, time.process_time() - c0,
                      qmax, build)

    try:
        model = Model(inst, backend, energy, deadline)
    except TimeoutError:
        return result("TIMEOUT")
    solver = model.solver
    if hint is not None:
        vals = model.assignment(hint.prune_to_minimal(inst))
        variables = solver.variables()
        solver.SetHint([variables[i] for i in vals], list(vals.values()))
    build = time.perf_counter() - t0
    left = deadline - time.perf_counter()
    if left < 0.01:
        return result("TIMEOUT", build=build)
    solver.SetTimeLimit(max(1, int(1000 * left)))
    solver.SetNumThreads(threads)
    sol_file = None
    if backend == "CP_SAT":
        solver.SetSolverSpecificParametersAsString(f"mip_var_scaling: {CPSAT_SCALING}")
    elif backend == "HIGHS":  # returns False although HiGHS applies the options
        fd, sol_file = tempfile.mkstemp(prefix="ctas_highs_", suffix=".sol")
        os.close(fd)
        _HIGHS_THREADS[:] = _HIGHS_THREADS or [threads]  # another count would make HiGHS refuse to run
        solver.SetSolverSpecificParametersAsString(f"threads = {_HIGHS_THREADS[0]}\nmip_improving_solution_save = true"
                                                   f"\nmip_improving_solution_file = {sol_file}\n")
    if log:
        solver.EnableOutput()
    try:
        status = _STATUS.get(solver.Solve(), "UNKNOWN")
        if status in ("OPTIMAL", "FEASIBLE"):
            objective, bound, values = solver.Objective().Value(), solver.Objective().BestBound(), model.solution()
        elif sol_file is not None and (inc := _highs_incumbent(sol_file, solver.NumVariables())) is not None:
            status, objective, bound, values = "FEASIBLE", inc[0], math.nan, inc[1]
        else:
            return result(status, build=build)
    finally:
        if sol_file is not None:
            os.unlink(sol_file)
    x, q, qmax = model.split(values)
    try:
        plan = convert(model, x, q, min(COVER_LIMIT, max(0.1, deadline - time.perf_counter())))
    except _Failed as e:
        return result(e.args[0], objective=objective, bound=bound, qmax=qmax, build=build)
    return result(status, plan, objective, bound, qmax, build)


# --- the benchmark's shipped Gurobi runs -------------------------------------------------------------------------


@dataclass
class Shipped:
    """One RALTestSet ``env_<i>/results.yaml`` (the benchmark's Gurobi CTAS-D run) with its graph.yaml."""

    routes: list[list[int]]  # per agent, 0-based tasks: the paths of each vehicle type given to its agents in order
    plan: Plan  # the routes as a plan, keys from ``qVar`` (not pruned)
    result: dict  # the ``result:`` block (objVal, objBound, MIPGap, energyCost, timeCost, ...)
    qmax: float  # qMax (= timeCost / 100)
    variables: dict[str, np.ndarray]  # xVar, xrVar, yVar, yrVar, zVar, qVar, qMax, alphaVar, wVar, gVar as printed
    edges: list[list[tuple[int, int, float, float]]]  # per species, graph.yaml arcs (i, j, energy, time) in order
    node_time: np.ndarray  # [T] graph.yaml node times


def shipped_solution(inst: Instance, folder: str | PathLike) -> Shipped | None:
    """Read ``folder``/results.yaml and graph.yaml of the benchmark's CTAS-D run on ``inst``; None if the run has
    no vehicle paths. Vehicles of type ``k + 1`` go to the agents of species ``k`` in id order, as CTAS-D.py does."""
    import yaml  # PyYAML: only this reader needs it

    def load(name: str) -> dict:
        return yaml.load((Path(folder) / name).read_text(), Loader=getattr(yaml, "CSafeLoader", yaml.SafeLoader))

    res = load("results.yaml")
    if "vehicle" not in res:
        return None
    graph = load("graph.yaml")
    T, S = inst.n_tasks, inst.n_species
    free = {k: np.flatnonzero(inst.species == k).tolist() for k in range(S)}
    routes: list[list[int]] = [[] for _ in range(inst.n_agents)]
    for v in res["vehicle"].values():
        routes[free[int(v["type"]) - 1].pop(0)] = [int(n) - 1 for n in v["node"] if int(n) != 0]
    variables = {k: np.array(res[k], dtype=float) for k in ("xVar", "xrVar", "yVar", "yrVar", "zVar", "qVar",
                                                             "alphaVar", "wVar", "gVar", "qMax")}
    members: list[list[int]] = [[] for _ in range(T)]
    for a, route in enumerate(routes):
        for j in route:
            members[j].append(a)
    edges = []
    for k in range(S):
        g = graph[f"vehicle{k}"]
        n_edges = sum(key.startswith("edge") for key in g)
        edges.append([(int(e[0]), int(e[1]), float(e[3]), float(e[5])) for e in (g[f"edge{n}"] for n in range(n_edges))])
    node_time = np.array([graph["vehicle0"].get(f"node{n}", 0.0) for n in range(T)])
    return Shipped(routes, Plan(members, variables["qVar"][:T], inst.n_agents), res["result"],
                   float(variables["qMax"][0]), variables, edges, node_time)


def shipped_assignment(model: Model, shipped: Shipped) -> tuple[dict[int, float], float]:
    """The shipped variable values mapped onto ``model`` (by variable index), and the largest shipped value of a
    variable the model does not create (pairs excluded by TaskReqUnrelavant; must be 0)."""
    inst = model.inst
    T, S = inst.n_tasks, inst.n_species
    var = shipped.variables
    vals: dict[int, float] = {}
    dropped = 0.0
    E = len(shipped.edges[0])
    for k in range(S):
        pos = {(i, j): e for e, (i, j) in enumerate(model.arcs[k])}
        for e, (i, j, _, _) in enumerate(shipped.edges[k]):
            if (i, j) in pos:
                vals[model.x[k][pos[i, j]].index()] = var["xVar"][k * E + e]
                vals[model.xr[k][pos[i, j]].index()] = var["xrVar"][k * E + e]
            else:
                dropped = max(dropped, abs(var["xVar"][k * E + e]), abs(var["xrVar"][k * E + e]))
        for j in range(T):
            if (k, j) in model.y:
                vals[model.y[k, j].index()] = var["yVar"][k * T + j]
                vals[model.yr[k, j].index()] = var["yrVar"][k * T + j]
            else:
                dropped = max(dropped, abs(var["yVar"][k * T + j]), abs(var["yrVar"][k * T + j]))
        for (kk, n), v in model.g.items():
            if kk == k:  # gId2sub: tasks, then s_k, then u_k
                vals[v.index()] = var["gVar"][k * (T + 2) + (n if n < T else T if n == T + k else T + 1)]
    for n, v in enumerate(model.q):
        vals[v.index()] = var["qVar"][n]
    vals[model.ms.index()] = shipped.qmax
    for j, z in enumerate(model.z):
        vals[z.index()] = var["zVar"][j]
    for idx, key in enumerate(sorted(model.alpha)):  # C++ order: task, then AND clause (= required trait)
        vals[model.alpha[key].index()] = var["alphaVar"][idx]
        vals[model.w[key].index()] = var["wVar"][idx]
    return vals, dropped


def check_shipped(inst: Instance, shipped: Shipped) -> dict:
    """Plug a shipped solution into our model. Returns ``graph_diff`` (largest energy / time / node-time difference
    between graph.yaml and our arcs), ``violation`` (largest bound, integrality or row violation of the model
    without energy rows), ``eng_violation`` (the same with them), ``objective`` (full model, ``energy=True``) with
    the shipped ``objVal``, and ``dropped`` (largest shipped value of a variable we do not create; must be 0).

    Shipped values are printed with 6 decimals, and an ``xr`` printed as 1.000000 can be 1 - 5e-9 (Gurobi accepts
    1e-5 off an integer), which a big-M turns into slack: up to 1.5e-5 on TimeEdge rows (M = 1e4) and 0.5 on EngEdge
    rows (M = 1e8) on RALTestSet."""
    diff = float(np.abs(shipped.node_time - inst.dur).max())
    out = {}
    for energy in (False, True):
        model = Model(inst, "SCIP", energy)
        vals, dropped = shipped_assignment(model, shipped)
        out["eng_violation" if energy else "violation"], objective = model.check(vals)
    for k, edges in enumerate(shipped.edges):
        for i, j, eng, t in edges:
            ours = model.time(k, i, j)
            diff = max(diff, abs(ours - t), abs(ours * inst.speed - eng))
    return {"graph_diff": diff, **out, "objective": objective, "objVal": float(shipped.result["objVal"]),
            "dropped": dropped}
