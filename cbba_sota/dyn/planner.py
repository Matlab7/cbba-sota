"""State-start rolling-horizon coalition planner (Track D spec Section 4.1): belief state -> plan.

A planner call takes a belief state (``PlanState``: what one replica knows at time ``now``) and a scope (the robots
it may assign) and returns a ``DynPlan`` for every open task. It is a pure function of (state, scope, incumbent,
seed, budget), so identical replicas with the same seed compute identical plans (G3); the comm layer decides scope
and seeds.

Problem seen by the kernel (``Problem``):
- anchored tasks: committed (a member departed), not started; their live frozen members are fixed and their keys
  keep the order (prefix of the key order). A committed task whose fixed members no longer cover it (failure,
  abandon) is a *residual*: the planner adds extra members at its rank (spec 4.1 "re-planned with its residual
  requirement").
- open tasks: released, not committed, not started, not finished. They get keys above every committed key (G1).
- robots: free at ``ready`` at ``pos`` (state start); a robot travelling to or waiting at an anchored task is
  locked to it (``head``); a robot working on a started task is free at its predicted finish; failed robots and
  robots outside the scope take no new work.
- predictors (spec 4.1): travel times are nominal x ``kappa``; residual durations are nominal.

Search (``RHPlanner.plan``): warm start = the incumbent's coalitions of open tasks (in its key order) behind the
anchored prefix; floor = regret-2 insertion of new, orphaned and residual tasks (G4: every coverable task gets a
feasible slot); then ``iters`` ALNS iterations (deterministic budget, one kernel batch, no wall clock) with the
lexicographic objective (tasks placed, makespan, objective). ALNS parameters are the pinned v2 defaults (commit
4cf5e04, ``cbba_sota/solvers/alns.py``), copied in ``SearchConfig``.

G4 fallback for residuals: a residual that cannot be covered at its rank is moved behind the committed prefix with
its fixed members (a new key); if that is impossible too (a fixed member is locked to it and has later committed
tasks), its fixed members are released and it is re-planned as an open task. Both are reported in the plan.
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field

import numpy as np

from cbba_sota.dyn import rh_kernels as K
from cbba_sota.hetero.instance import Instance

__all__ = ["PINNED_COMMIT", "DynPlan", "PlanState", "Problem", "RHPlanner", "SearchConfig", "check_plan",
           "q_range", "state_digest", "t_start_rule", "travel_from"]

PINNED_COMMIT = "4cf5e04"
AT_DEPOT = -2  # PlanState.pos_task value of a robot at its depot
AT_POINT = -1  # ... of a robot anywhere else that is not a task location


# --- configuration (pinned v2 defaults) ----------------------------------------------------------------------


@dataclass(frozen=True)
class SearchConfig:
    """ALNS parameters: the pinned v2 defaults of ``ALNSConfig`` at commit 4cf5e04 that matter for a single-worker,
    iteration-budget search (no pool, no exchange, no portfolio)."""

    lam: float = 0.1  # weight of the mean finish time in the search objective
    noise: float = 0.2  # multiplicative arrival noise of the noisy repair
    q: tuple[int, int] | None = None  # removed tasks per iteration (None: ``q_range`` of the removable tasks)
    max_slots: int = 64
    shaw_p: float = 6.0
    worst_p: float = 3.0
    regret_max_q: int = 10  # regret-2 repair inside the search only for up to this many removed tasks
    rho: float = 0.1
    seg_len: int = 100
    sigma: tuple[float, float, float] = (33.0, 9.0, 13.0)
    restart_iters: int = 3000
    t_start: float | None = None  # SA start temperature relative to the initial objective (None: t_start_rule)
    t_end_ratio: float = 0.02
    ops_off: tuple[str, ...] = ("member_swap",)
    q_stag: int = 0
    q_big: int = 12
    floor: str = "regret"  # "regret": regret-2 insertion of the tasks to place; "insert": seeded random order (v1)
    keys: str = "monotone"  # "monotone": repaired tasks behind every committed task (spec 4.1, G1); "relaxed":
    #                         repaired tasks may precede committed ones (committed order and head-first kept)


def q_range(n_tasks: int) -> tuple[int, int]:
    """Pinned default removal size: 1 to 6 tasks at 20 tasks, 4 at 50, 2 from 200 on."""
    return 1, int(np.clip(round(4 * np.sqrt(50 / max(n_tasks, 1))), 2, 6))


def t_start_rule(n_tasks: int) -> float:
    """Pinned default relative start temperature: 0.01 up to 50 tasks, then ~T^-1.5."""
    return 0.01 * min(1.0, (50 / max(n_tasks, 1)) ** 1.5)


def travel_from(point: np.ndarray, locs: np.ndarray, speed: float) -> np.ndarray:
    """Travel times from an arbitrary point to ``locs`` [n, 2], one ``np.linalg.norm`` per pair exactly as the env
    (``TaskEnv.calculate_eulidean_distance``); the executor uses the same function, so predictions and physics
    agree bit for bit."""
    return np.array([np.linalg.norm(point - x) / speed for x in locs])


# --- belief state and plan -------------------------------------------------------------------------------------


@dataclass
class PlanState:
    """What a replica knows at ``now`` (predicted quantities use the declared predictors).

    Task arrays have length T (all tasks of ``inst``; unreleased ones are ignored), robot arrays length A.
    ``members``/``keys`` matter for committed tasks only. ``pos_task[i]`` says where robot ``i`` is at ``ready[i]``:
    a task id (its location), ``AT_DEPOT`` or ``AT_POINT`` (then ``pos[i]`` is used)."""

    inst: Instance
    now: float
    released: np.ndarray  # [T] bool
    done: np.ndarray  # [T] bool, finished
    started: np.ndarray  # [T] bool, in progress
    committed: np.ndarray  # [T] bool, a member departed, not started, not finished
    members: list[tuple[int, ...]]  # [T] frozen coalitions of committed tasks
    keys: np.ndarray  # [T] keys of committed tasks
    alive: np.ndarray  # [A] bool, not known to have failed
    ready: np.ndarray  # [A] predicted time the robot is next free
    pos: np.ndarray  # [A, 2] position at ``ready``
    pos_task: np.ndarray  # [A] int, see above
    head: np.ndarray  # [A] int, committed task the robot travels to or waits at (-1: none)
    kappa: float = 1.0  # travel predictor factor 1 / (1 - expected stall fraction)
    est: np.ndarray | None = None  # [T] earliest starts (hindsight reference only)
    key_floor: float = -1.0  # repaired tasks get keys > max(key_floor, committed keys)

    @classmethod
    def static(cls, inst: Instance) -> PlanState:
        """Time 0 with every task released and every robot at its depot (the static problem)."""
        T, A = inst.n_tasks, inst.n_agents
        return cls(inst=inst, now=0.0, released=np.ones(T, bool), done=np.zeros(T, bool), started=np.zeros(T, bool),
                   committed=np.zeros(T, bool), members=[()] * T, keys=np.zeros(T), alive=np.ones(A, bool),
                   ready=np.zeros(A), pos=np.array(inst.depot, float), pos_task=np.full(A, AT_DEPOT, np.int64),
                   head=np.full(A, -1, np.int64))

    def open_tasks(self) -> np.ndarray:
        """Released tasks that are neither committed, started nor finished."""
        return np.flatnonzero(self.released & ~self.committed & ~self.started & ~self.done)


def state_digest(state: PlanState) -> bytes:
    """Digest of a belief state (replica contents that a planner call reads); equal replicas give equal digests."""
    h = hashlib.blake2b(digest_size=16)
    h.update(np.float64(state.now).tobytes())
    for arr in (state.released, state.done, state.started, state.committed, state.alive):
        h.update(np.packbits(np.asarray(arr, bool)).tobytes())
    com = np.flatnonzero(state.committed)
    h.update(np.asarray(state.keys, float)[com].tobytes())
    for j in com:
        h.update(np.asarray(state.members[j], np.int64).tobytes() + b"|")
    for arr in (state.ready, state.pos, state.pos_task, state.head):
        h.update(np.ascontiguousarray(arr).tobytes())
    h.update(np.float64(state.kappa).tobytes())
    return h.digest()


@dataclass
class DynPlan:
    """A plan for all non-finished, non-started tasks: coalitions and a global key order."""

    members: list[tuple[int, ...]]  # [T]; () for tasks not in the plan
    keys: np.ndarray  # [T]; nan for tasks not in the plan
    makespan: float  # predicted (planner model) makespan
    unplaced: list[int] = field(default_factory=list)  # open tasks no robot in scope can cover
    residual: list[int] = field(default_factory=list)  # committed tasks that received extra members
    demoted: list[int] = field(default_factory=list)  # residuals moved behind the committed prefix (new key)
    released: list[int] = field(default_factory=list)  # residuals whose frozen members were released (G4 fallback)
    iterations: int = 0
    seed: int = 0
    cpu_s: float = 0.0  # process CPU time of the call (reporting only; budgets are iteration counts)
    floor_makespan: float = float("nan")  # after warm start + floor, before search
    n_open: int = 0
    n_anchored: int = 0

    def routes(self) -> list[list[int]]:
        """Per robot, its planned tasks in key order."""
        A = max((max(m) for m in self.members if m), default=-1) + 1
        return self.routes_for(A)

    def routes_for(self, n_agents: int) -> list[list[int]]:
        out: list[list[int]] = [[] for _ in range(n_agents)]
        planned = [j for j in range(len(self.members)) if self.members[j]]
        for j in sorted(planned, key=lambda j: (self.keys[j], j)):
            for i in self.members[j]:
                out[i].append(j)
        return out


# --- kernel problem ---------------------------------------------------------------------------------------------


def _own(a) -> np.ndarray:
    return np.array(a, dtype=float, order="C")


def _new_sol(T: int, A: int, W: int) -> tuple:
    return (np.zeros(T, np.int64), np.full((T, W), -1, np.int64), np.zeros(T, np.int64), np.zeros(1, np.int64),
            np.full(T, np.nan), np.full(T, np.nan), np.zeros((T, W)), np.full((T, W), -1, np.int64),
            np.zeros(A), np.full(A, -1, np.int64), np.zeros(K.N_FL), np.full((T, W), -1, np.int64),
            np.full(A, -1, np.int64), np.zeros(A, np.int64), np.zeros(T))


class Problem:
    """Kernel instance built from a belief state (local task ids 0..Tk-1 map to ``gid``)."""

    def __init__(self, state: PlanState, scope: np.ndarray | None = None, relaxed: bool = False):
        inst = state.inst
        self.relaxed = bool(relaxed)
        T, A = inst.n_tasks, inst.n_agents
        self.state, self.inst = state, inst
        alive = np.asarray(state.alive, bool)
        self.scope = alive.copy() if scope is None else np.asarray(scope, bool) & alive
        kap = float(state.kappa)
        live = state.released & ~state.done & ~state.started
        anchored = []
        for j in np.flatnonzero(state.committed & live):
            fixed = tuple(sorted(int(i) for i in state.members[j] if alive[i]))
            if fixed:
                anchored.append((float(state.keys[j]), int(j), fixed))
        anchored.sort()
        anch = {j for _, j, _ in anchored}
        open_ = [int(j) for j in np.flatnonzero(live) if int(j) not in anch]
        gid = np.array([j for _, j, _ in anchored] + open_, np.int64)
        Tk, na = len(gid), len(anchored)
        self.gid, self.Tk, self.A, self.na = gid, Tk, A, na
        self.lid = np.full(T, -1, np.int64)
        self.lid[gid] = np.arange(Tk)
        req, ab = inst.req[gid], inst.ab
        fixed_sets = [f for _, _, f in anchored] + [()] * len(open_)
        fcnt = np.array([len(f) for f in fixed_sets], np.int64)
        binary = np.isin(ab, (0.0, 1.0)).all() and np.array_equal(req, np.round(req))
        bound = np.ceil(req).sum(axis=1).astype(np.int64) if binary else np.full(Tk, A, np.int64)
        W = int(min(A, max(1, int((fcnt + bound).max(initial=1)))))
        W = max(W, int(fcnt.max(initial=0)))
        self.W = W
        fmem = np.full((Tk, W), -1, np.int64)
        for x, f in enumerate(fixed_sets):
            fmem[x, :len(f)] = f
        mode = np.zeros(Tk, np.int64)
        rank = np.full(Tk, -1, np.int64)
        for x in range(na):
            rank[x] = x
            f = list(fixed_sets[x])
            mode[x] = K.M_ANCHOR if (ab[f].sum(axis=0) >= req[x]).all() else K.M_RESIDUAL
        capable = ((ab[None, :, :] > 0) & (req[:, None, :] > 0)).any(axis=2)  # [Tk, A]
        cand = capable & self.scope[None, :]
        for x, f in enumerate(fixed_sets):
            cand[x, list(f)] = False
        ncap = cand.sum(axis=1).astype(np.int64)
        capl = np.full((Tk, A), -1, np.int64)
        for x in range(Tk):
            capl[x, :ncap[x]] = np.flatnonzero(cand[x])
        capf = capable.copy()
        for x, f in enumerate(fixed_sets):
            capf[x, list(f)] = True
        tt = inst.tt[np.ix_(gid, gid)]
        da = inst.da[:, gid]
        if kap != 1.0:
            tt, da = tt * kap, da * kap
        near = np.argsort(tt + np.diag(np.full(Tk, np.inf)), axis=1, kind="stable")[:, :max(Tk - 1, 0)]
        marker = np.where(self.scope, 0.0, np.arange(A) + 1.0)[:, None]  # robots outside the scope never swap
        _, group = np.unique(np.hstack([ab, marker]), axis=0, return_inverse=True)
        group = group.ravel().astype(np.int64)
        partners = np.argsort(group, kind="stable").astype(np.int64)
        pstart = np.searchsorted(group[partners], np.arange(group.max() + 2)).astype(np.int64)
        ready = np.where(alive, np.asarray(state.ready, float), 0.0)
        dout = np.empty((A, Tk))
        dhome = np.empty(A)
        head = np.full(A, -1, np.int64)
        for i in range(A):
            if not alive[i]:
                dout[i] = np.inf
                dhome[i] = -np.inf
                continue
            p = int(state.pos_task[i])
            if p == AT_DEPOT:
                dout[i] = inst.da[i, gid]
                dhome[i] = 0.0
            elif p >= 0:
                dout[i] = inst.tt[p, gid]
                dhome[i] = inst.da[i, p]
            else:
                dout[i] = travel_from(state.pos[i], inst.loc[gid], inst.speed)
                dhome[i] = np.linalg.norm(state.pos[i] - inst.depot[i]) / inst.speed
            if kap != 1.0:
                dout[i] *= kap
                dhome[i] *= kap
            h = int(state.head[i])
            if h >= 0 and self.lid[h] >= 0 and self.lid[h] < na:
                head[i] = self.lid[h]
                dout[i, head[i]] = 0.0  # ready is the (predicted) arrival there
        est = np.full(Tk, -np.inf) if state.est is None else _own(np.asarray(state.est)[gid])
        self.mode0 = mode.copy()
        self.arrays = (_own(req), _own(ab), _own(inst.dur[gid]), _own(tt), _own(da), capl, ncap,
                       np.ascontiguousarray(near, np.int64), partners, pstart, group, np.ascontiguousarray(capf),
                       _own(ready), _own(dout), _own(dhome), est, fmem, fcnt, mode, rank,
                       np.ascontiguousarray(mode != K.M_ANCHOR), head, np.array([int(self.relaxed)], np.int64))
        assert len(self.arrays) == len(K.INST)

    # views of the mutable arrays
    @property
    def mode(self) -> np.ndarray:
        return self.arrays[K.I_MODE]

    @property
    def fcnt(self) -> np.ndarray:
        return self.arrays[K.I_FCNT]

    @property
    def fmem(self) -> np.ndarray:
        return self.arrays[K.I_FMEM]

    @property
    def head(self) -> np.ndarray:
        return self.arrays[K.I_HEAD]

    @property
    def removable(self) -> np.ndarray:
        return self.arrays[K.I_REM]

    def new_sol(self) -> tuple:
        return _new_sol(self.Tk, self.A, self.W)

    def workspace(self) -> tuple:
        Tk, A, W = self.Tk, self.A, self.W
        return (np.zeros(A), np.zeros(A), np.zeros(self.inst.n_traits), np.zeros(A, np.int64),
                np.zeros(A, np.int64), np.zeros(A), np.full(A, -1, np.int64), np.zeros(len(K.COUNTERS), np.int64),
                np.zeros(Tk + 1, np.int64), np.zeros(Tk + 1), np.zeros(Tk + 1), np.zeros(Tk + 1),
                np.zeros(Tk + 1, np.int64), np.zeros((Tk + 1, W), np.int64),
                np.full(A, -1, np.int64), np.zeros(A, np.int64), np.zeros(A, np.int64))

    def load(self, sol: tuple, order: list[int], members: list[tuple[int, ...]], lam: float) -> None:
        """Put local tasks ``order`` (with ``members[x]`` indexed by local id) into ``sol`` and schedule it."""
        seq, mem, cnt, ni = sol[:4]
        cnt[:] = 0
        for p, x in enumerate(order):
            m = members[x]
            mem[x, :len(m)] = m
            cnt[x] = len(m)
            seq[p] = x
        ni[0] = len(order)
        K.schedule(self.arrays, sol, lam)

    def order_of(self, sol: tuple) -> list[int]:
        return [int(x) for x in sol[0][:sol[3][0]]]


# --- planner ------------------------------------------------------------------------------------------------


def _covers(ab: np.ndarray, req: np.ndarray, m) -> bool:
    return bool(len(m)) and bool((ab[list(m)].sum(axis=0) >= req).all())


def _minimal(ab: np.ndarray, req: np.ndarray, m) -> bool:
    tot = ab[list(m)].sum(axis=0)
    return all(not (tot - ab[i] >= req).all() for i in m)


class RHPlanner:
    """Warm-started state-start coalition ALNS with insertion floor and deterministic iteration budget."""

    def __init__(self, config: SearchConfig | None = None):
        self.cfg = config or SearchConfig()

    # -- main entry --------------------------------------------------------------------------------------------
    def plan(self, state: PlanState, incumbent: DynPlan | None = None, scope: np.ndarray | None = None,
             seed: int = 0, iters: int = 300, removable=None, warm: bool = True, restarts: int = 0,
             restart_noise: float = 0.2, return_problem: bool = False):
        """Plan every open and residual task of ``state`` for the robots in ``scope`` (default: all alive).

        ``incumbent``: previous plan (warm start); ``iters``: ALNS iterations (0 = floor only, insertion-only
        repair); ``removable``: tasks the search may move (targeted repair), a [T] bool array in global ids or a
        callable ``(problem, solution, warm local ids) -> [Tk] bool`` evaluated after the floor; ``warm=False``
        rebuilds every open task by the floor (cold constructor); ``restarts``: additional randomized cold
        constructions (seeded random order, cover noise ``restart_noise`` on every second one), best kept.
        ``return_problem``: return ``(plan, problem, solution)`` (for solvers that continue from the floor)."""
        c0 = time.process_time()
        cfg = self.cfg
        relaxed = cfg.keys == "relaxed"
        prob = Problem(state, scope, relaxed)
        if prob.Tk == 0:
            T = state.inst.n_tasks
            empty = DynPlan(members=[()] * T, keys=np.full(T, np.nan), makespan=float("nan"), seed=int(seed),
                            cpu_s=time.process_time() - c0)
            return (empty, prob, None) if return_problem else empty
        K.seed(int(seed) & 0x7FFFFFFF)
        ctx = self._initial(prob, incumbent, warm, seed, cfg.floor, 0.0)
        for k in range(1, restarts + 1):
            pk = Problem(state, scope, relaxed)
            ck = self._initial(pk, None, False, seed + 7919 * k, "insert", restart_noise if k % 2 == 0 else 0.0)
            if K.better(ck["sol"][10], ctx["sol"][10]):
                prob, ctx = pk, ck
        sol = ctx["sol"]
        floor_ms = float(sol[10][0])
        best = sol
        its = 0
        if removable is not None:
            mask = removable(prob, sol, ctx["warm"]) if callable(removable) else np.asarray(removable, bool)[prob.gid]
            prob.removable[:] &= np.asarray(mask, bool)
        n_rem = int(prob.removable.sum())
        if iters > 0 and n_rem > 0:
            best, its = self._search(prob, sol, ctx["ws"], ctx["slot_ok"], ctx["posn"], ctx["best_c"],
                                     ctx["regret_c"], iters, n_rem)
        plan = self._extract(prob, best, ctx["demoted"], ctx["released"], its, seed, floor_ms,
                             time.process_time() - c0)
        return (plan, prob, best) if return_problem else plan

    def extract(self, prob: Problem, sol: tuple, like: DynPlan, cpu: float) -> DynPlan:
        """Plan from a kernel solution of ``prob`` with the bookkeeping (residual, demoted, released) of ``like``."""
        lid = prob.lid
        plan = self._extract(prob, sol, [int(lid[j]) for j in like.demoted], [int(lid[j]) for j in like.released],
                             like.iterations, like.seed, like.floor_makespan, cpu)
        return plan

    def _initial(self, prob: Problem, incumbent: DynPlan | None, warm: bool, seed: int, floor: str,
                 noise: float) -> dict:
        """Anchored tasks, then the incumbent's valid coalitions of open tasks (``warm``), then the floor."""
        cfg = self.cfg
        state = prob.state
        sol = prob.new_sol()
        ctx = {"sol": sol, "ws": prob.workspace(), "slot_ok": np.zeros(prob.Tk + 1, np.bool_),
               "posn": np.full(prob.Tk, -1, np.int64), "best_c": np.zeros(prob.A, np.int64),
               "regret_c": np.zeros(prob.A, np.int64)}
        members: list[tuple[int, ...]] = [()] * prob.Tk
        order = []
        anchored = []
        for x in range(prob.na):
            if prob.mode[x] == K.M_ANCHOR:
                members[x] = tuple(prob.fmem[x, :prob.fcnt[x]])
                anchored.append((float(state.keys[prob.gid[x]]), x))
                order.append(x)
        warm_tasks = []
        if warm and incumbent is not None:
            ab, req = prob.arrays[K.I_AB], prob.arrays[K.I_REQ]
            for x in range(prob.na, prob.Tk):
                j = int(prob.gid[x])
                m = incumbent.members[j] if j < len(incumbent.members) else ()
                if m and np.isfinite(incumbent.keys[j]) and all(prob.scope[i] for i in m) \
                        and _covers(ab, req[x], m) and _minimal(ab, req[x], m):
                    warm_tasks.append((float(incumbent.keys[j]), x, tuple(int(i) for i in m)))
        warm_tasks.sort()
        for _, x, m in warm_tasks:
            members[x] = m
            order.append(x)
        if prob.relaxed:  # one key space: the incumbent's (the executor adopts every key of the last plan)
            order = [x for _, x in sorted(anchored + [(k, x) for k, x, _ in warm_tasks])]
        prob.load(sol, order, members, cfg.lam)
        placed = set(order)
        todo_res = [x for x in range(prob.na) if prob.mode[x] == K.M_RESIDUAL]
        todo_free = [x for x in range(prob.na, prob.Tk) if x not in placed]
        ctx["warm"] = [x for _, x, _ in warm_tasks]
        ctx["demoted"], ctx["released"] = self._floor(prob, sol, ctx["ws"], ctx["slot_ok"], ctx["posn"],
                                                      ctx["best_c"], ctx["regret_c"], todo_res, todo_free, seed,
                                                      floor, noise)
        return ctx

    # -- floor ---------------------------------------------------------------------------------------------------
    def _floor(self, prob: Problem, sol, ws, slot_ok, posn, best_c, regret_c, todo_res, todo_free, seed,
               floor: str = "regret", noise: float = 0.0):
        cfg = self.cfg
        arrs = prob.arrays
        demoted, released = [], []
        if todo_res:
            rem = np.array(todo_res, np.int64)
            n = K.regret_insert(rem, len(rem), arrs, sol, ws, slot_ok, posn, best_c, regret_c, cfg.lam,
                                cfg.max_slots)
            left = [int(x) for x in rem[:len(rem) - n]]
            for x in left:  # G4 fallback: behind the prefix with the fixed members, else release them
                locked = [int(i) for i in prob.fmem[x, :prob.fcnt[x]] if prob.head[i] == x]
                other = any(prob.fcnt[y] and any(i in prob.fmem[y, :prob.fcnt[y]] for i in locked)
                            for y in range(prob.na) if y != x)
                prob.mode[x] = K.M_FREE
                if other:
                    self._release(prob, x)
                    released.append(x)
                else:
                    demoted.append(x)
                todo_free.append(x)
        if todo_free:
            rem = np.array(todo_free, np.int64)
            if floor == "insert":
                rem = rem[np.random.default_rng(seed).permutation(len(rem))]
                K.construct(rem, arrs, sol, ws, slot_ok, posn, best_c, noise, cfg.lam, cfg.max_slots)
            else:
                K.regret_insert(rem, len(rem), arrs, sol, ws, slot_ok, posn, best_c, regret_c, cfg.lam,
                                cfg.max_slots)
            inplan = set(prob.order_of(sol))
            for x in list(demoted):
                if x not in inplan:  # could not be covered behind the prefix either: release and retry
                    demoted.remove(x)
                    self._release(prob, x)
                    released.append(x)
                    one = np.array([x], np.int64)
                    K.regret_insert(one, 1, arrs, sol, ws, slot_ok, posn, best_c, regret_c, cfg.lam, cfg.max_slots)
        return demoted, released

    @staticmethod
    def _release(prob: Problem, x: int) -> None:
        for i in prob.fmem[x, :prob.fcnt[x]]:
            if prob.head[i] == x:
                prob.head[i] = -1
        prob.fcnt[x] = 0
        prob.fmem[x, :] = -1
        prob.removable[x] = True

    # -- search -------------------------------------------------------------------------------------------------
    def _search(self, prob: Problem, sol, ws, slot_ok, posn, best_c, regret_c, iters: int, n_rem: int):
        cfg = self.cfg
        cur, cand, best = sol, prob.new_sol(), prob.new_sol()
        K.copy_sol(cur, cand)
        K.copy_sol(cur, best)
        q_lo, q_hi = cfg.q or q_range(n_rem)
        q_hi = max(1, min(q_hi, n_rem))
        q_lo = max(1, min(q_lo, q_hi))
        par = np.array([cfg.lam, cfg.noise, q_lo, q_hi, cfg.max_slots, cfg.shaw_p, cfg.worst_p, cfg.regret_max_q,
                        cfg.rho, cfg.seg_len, *cfg.sigma, cfg.restart_iters, cfg.q_stag,
                        max(q_hi, min(cfg.q_big, n_rem)), 0.0], float)
        wd = np.array([0.0 if op in cfg.ops_off else 1.0 for op in K.DESTROY])
        wr = np.ones(len(K.REPAIR))
        sd, sr, ud, ur = np.zeros(len(wd)), np.zeros(len(wr)), np.zeros(len(wd)), np.zeros(len(wr))
        stats = np.zeros(len(K.STATS), np.int64)
        removed, flag, buf = np.zeros(prob.Tk, np.int64), np.zeros(prob.Tk, np.bool_), np.zeros(prob.Tk, np.int64)
        fl = cur[10]
        obj0 = float(fl[0] + cfg.lam * fl[1] / prob.Tk)  # penalty-free objective
        temp0 = (cfg.t_start if cfg.t_start is not None else t_start_rule(prob.Tk)) * obj0
        temp1 = cfg.t_end_ratio * temp0
        K.run_batch(int(iters), temp0, temp1, prob.arrays, cur, cand, best, ws, slot_ok, posn, best_c, regret_c,
                    removed, flag, buf, wd, wr, sd, sr, ud, ur, par, stats)
        return best, int(stats[K.S_IT])

    # -- output -----------------------------------------------------------------------------------------------
    def _extract(self, prob: Problem, sol, demoted, released, its, seed, floor_ms, cpu) -> DynPlan:
        state = prob.state
        T = prob.inst.n_tasks
        members: list[tuple[int, ...]] = [()] * T
        keys = np.full(T, np.nan)
        com = np.zeros(T, bool)
        com[prob.gid[:prob.na]] = True
        base = max(float(state.key_floor), float(np.max(state.keys[com], initial=-1.0))) + 1.0
        seq, mem, cnt, ni = sol[:4]
        nfree = 0
        residual = []
        for p in range(ni[0]):
            x = int(seq[p])
            j = int(prob.gid[x])
            members[j] = tuple(sorted(int(i) for i in mem[x, :cnt[x]]))
            if prob.mode[x] == K.M_RESIDUAL and prob.relaxed:
                residual.append(j)
            if prob.relaxed:  # every task re-keyed in plan order (committed order is kept by construction)
                keys[j] = base + p
            elif prob.mode[x] != K.M_FREE:
                keys[j] = float(state.keys[j])
                if prob.mode[x] == K.M_RESIDUAL:
                    residual.append(j)
            else:
                keys[j] = base + nfree
                nfree += 1
        inplan = {int(x) for x in seq[:ni[0]]}
        unplaced = [int(prob.gid[x]) for x in range(prob.Tk) if x not in inplan]
        return DynPlan(members=members, keys=keys, makespan=float(sol[10][0]), unplaced=unplaced,
                       residual=residual, demoted=[int(prob.gid[x]) for x in demoted],
                       released=[int(prob.gid[x]) for x in released], iterations=its, seed=int(seed), cpu_s=cpu,
                       floor_makespan=floor_ms, n_open=prob.Tk - prob.na, n_anchored=prob.na)

    # -- evaluation of an arbitrary plan ---------------------------------------------------------------------------
    def evaluate(self, state: PlanState, plan: DynPlan, scope: np.ndarray | None = None) -> float:
        """Planner-model makespan of ``plan`` from ``state`` (the forward pass the search optimizes)."""
        prob = Problem(state, scope, self.cfg.keys == "relaxed")
        sol = prob.new_sol()
        xs = [x for x in range(prob.Tk) if plan.members[prob.gid[x]]]
        xs.sort(key=lambda x: (plan.keys[prob.gid[x]], prob.gid[x]))
        members = [tuple(plan.members[prob.gid[x]]) for x in range(prob.Tk)]
        prob.load(sol, xs, members, self.cfg.lam)
        return float(sol[10][0])


# --- checks -------------------------------------------------------------------------------------------------------


def check_plan(state: PlanState, plan: DynPlan, scope: np.ndarray | None = None, relaxed: bool = False) -> list[str]:
    """Violations of the plan contract (empty list = valid):
    - every open or committed task is planned unless reported unplaced; finished/started tasks are not planned;
    - covers: free tasks are minimal covers by robots in scope; committed tasks keep their live frozen members, and
      extra members of a residual are a minimal cover of the residual requirement;
    - keys: committed tasks keep their keys unless demoted or released; every other task has a key above every
      committed key (key monotonicity, G1); keys are unique. ``relaxed``: committed tasks keep their relative
      order instead (every task is re-keyed);
    - heads: a robot locked to a committed task (travelling or waiting there) has it first in its route."""
    inst = state.inst
    ab, req = inst.ab, inst.req
    alive = np.asarray(state.alive, bool)
    scope = alive if scope is None else np.asarray(scope, bool) & alive
    errs = []
    live = state.released & ~state.done & ~state.started
    com = state.committed & live & np.array([any(alive[i] for i in m) for m in state.members], bool)
    moved = set(plan.demoted) | set(plan.released)
    kmax = max(np.max(state.keys[com], initial=-np.inf), state.key_floor)
    for j in range(inst.n_tasks):
        m = plan.members[j]
        if not live[j]:
            if m:
                errs.append(f"task {j} is finished, started or unreleased but planned")
            continue
        if not m:
            if j not in plan.unplaced:
                errs.append(f"task {j} is live but neither planned nor reported unplaced")
            continue
        if not _covers(ab, req[j], m):
            errs.append(f"task {j} not covered by {m}")
        fixed = tuple(i for i in state.members[j] if alive[i]) if com[j] and j not in plan.released else ()
        if not set(fixed) <= set(m):
            errs.append(f"committed task {j} lost frozen members {set(fixed) - set(m)}")
        extra = [i for i in m if i not in fixed]
        if any(not scope[i] for i in extra):
            errs.append(f"task {j} uses robots outside the scope")
        if fixed:
            base = ab[list(fixed)].sum(axis=0)
            for i in extra:
                if (base + ab[extra].sum(axis=0) - ab[i] >= req[j]).all():
                    errs.append(f"residual {j}: extra member {i} is redundant")
        elif not _minimal(ab, req[j], m):
            errs.append(f"task {j}: {m} is not a minimal cover")
        if relaxed:
            pass
        elif com[j] and j not in moved:
            if plan.keys[j] != state.keys[j]:
                errs.append(f"committed task {j} changed key {state.keys[j]} -> {plan.keys[j]}")
        elif not plan.keys[j] > kmax:
            errs.append(f"repaired task {j} has key {plan.keys[j]} <= committed max {kmax}")
    if relaxed:
        kept = [j for j in np.flatnonzero(com) if plan.members[j] and j not in moved]
        old = sorted(kept, key=lambda j: (state.keys[j], j))
        new = sorted(kept, key=lambda j: (plan.keys[j], j))
        if old != new:
            errs.append("committed tasks changed their relative order")
    planned = [j for j in range(inst.n_tasks) if plan.members[j]]
    if len({float(plan.keys[j]) for j in planned}) != len(planned):
        errs.append("duplicate keys")
    routes = plan.routes_for(inst.n_agents)
    for i in range(inst.n_agents):
        h = int(state.head[i])
        if alive[i] and h >= 0 and com[h] and h not in plan.released and (not routes[i] or routes[i][0] != h):
            errs.append(f"robot {i} is locked to task {h} but its route starts {routes[i][:1]}")
        if not alive[i] and routes[i]:
            errs.append(f"failed robot {i} has a route")
    return errs
