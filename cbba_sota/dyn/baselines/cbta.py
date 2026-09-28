"""B6: coalition auctions re-run on structural events (docs/trackD-spec.md Section 5.2) -- week-1 start.

Disclosed reimplementations: CBTA (Wang, Liu, Qiu, Zhou, RA-L 2022, doi 10.1109/LRA.2022.3220155) and CBGA (Hunt et
al., Cognitive Computation 2014) have no public code; these are our readings, run on our state-start timing model,
and wins over them are not external SOTA evidence (spec Section 10).

Good communication only (spec C1-dyn). Under instant, reliable messaging a CBBA-family auction that re-runs to
consensus has the allocation of its sequential-greedy fixed point (Choi, Brunet, How, T-RO 2009, for scores with
diminishing marginal gain; coalition timing couplings break that condition, which is why CBTA adds timetables).
Week 1 therefore implements the *fixed point* of each auction directly, one award per round, instead of simulating
message rounds; the message-level asynchronous version (bundle release, CBBA-PR partial reset) is needed only for
the C3 arm CBTA-dec and is week-2 work.

Every rule shares SPARC's belief, anchors and trigger rule (``RollingPolicy``): committed tasks keep their frozen
members and keys (the anchored prefix of ``planner.Problem``), residual requirements are repaired by SPARC's
insertion floor (G4 path), and only the open tasks are auctioned, from scratch at every structural event (the
incumbent is ignored: an auction re-run to consensus). Bids use the planner's state-start timing (robots free at
``ready`` at ``pos``, travel x kappa, nominal durations).

Rules (``AuctionRH(rule, objective)``):
- ``cbta``: CBTA-style timetable auction. Each robot's timetable grows at its end: its bid for a task is the time
  it can arrive there after its last timetabled task. A task's offer is its earliest start, formed by the earliest-
  arriving robots that cover it (then pruned to a minimal cover, latest arrival first; this is the earliest possible
  start of the task when appended). Each round the task with the best offer is awarded and appended to its winners'
  timetables. ``objective="start"`` (CBTA's own: minimise average start time) awards the earliest start;
  ``objective="makespan"`` (our makespan variant) awards the earliest finish (start + duration).
- ``cbga``: CBGA-style grouping auction, for the fidelity check only (CBTA's paper reports lower average start times
  than CBGA): the same append-only timetables, but robots bid their travel distance and a task's group is its
  cheapest-travel cover; each round the task with the cheapest group is awarded. Timing is not part of the bid.
- ``seq``: the spec's declared fallback, a coalition sequential auction with our bids: each round every open task
  bids its best insertion into the current plan (``rh_kernels.insertion``: any slot after the anchored prefix,
  earliest-arrival cover, the kernel's objective makespan + lambda x mean finish), and the cheapest is awarded
  (cheapest insertion; SPARC's floor uses regret-2 instead).

Budget: native (no search budget; the auction runs to completion). Keys are the award order (``cbta``, ``cbga``) or
the plan order (``seq``), above every committed key (G1).
"""
from __future__ import annotations

import time

import numpy as np

from cbba_sota.dyn import rh_kernels as K
from cbba_sota.dyn.controller import PlanController
from cbba_sota.dyn.executor import STRUCTURAL
from cbba_sota.dyn.planner import DynPlan, PlanState, Problem, RHPlanner, SearchConfig
from cbba_sota.dyn.sparc import RollingPolicy

__all__ = ["RULES", "AuctionRH", "append_offer", "make_controller"]

RULES = ("cbta", "cbga", "seq")


def append_offer(prob: Problem, free: np.ndarray, last: np.ndarray, x: int, by: str = "arrival"):
    """Offer for appending local task ``x`` at the end of the robots' timetables.

    ``free[i]``, ``last[i]``: time robot ``i`` is free after its last timetabled task and that task (-1: none; then
    it leaves from its belief position). Returns ``(members, start, cost)`` with ``members`` a minimal cover chosen
    greedily by earliest arrival (``by="arrival"``) or by shortest travel (``by="travel"``), ``start`` the task's
    start (latest member arrival, at least ``est``) and ``cost`` the members' summed travel; ``(None, inf, inf)`` if
    the robots in scope cannot cover it."""
    a = prob.arrays
    req, ab, tt, dout, est = a[K.I_REQ], a[K.I_AB], a[K.I_TT], a[K.I_DOUT], a[K.I_EST]
    capl, ncap = a[K.I_CAPL], a[K.I_NCAP]
    cand = capl[x, :ncap[x]]
    if cand.size == 0:
        return None, np.inf, np.inf
    leg = np.where(last[cand] >= 0, tt[np.maximum(last[cand], 0), x], dout[cand, x])
    arr = free[cand] + leg
    key = arr if by == "arrival" else leg
    need = req[x].astype(float).copy()
    chosen = []
    for k in np.lexsort((cand, key)):
        i = int(cand[k])
        if (np.minimum(ab[i], need) > 0).any():
            chosen.append(k)
            need = need - ab[i]
            if (need <= 0).all():
                break
    if (need > 0).any():
        return None, np.inf, np.inf
    total = ab[cand[chosen]].sum(axis=0)
    for k in sorted(chosen, key=lambda k: (key[k], int(cand[k])), reverse=True):
        i = int(cand[k])
        if (total - ab[i] >= req[x]).all():
            total = total - ab[i]
            chosen.remove(k)
    members = tuple(sorted(int(cand[k]) for k in chosen))
    start = max(float(est[x]), float(arr[chosen].max()))
    return members, start, float(leg[chosen].sum())


class AuctionRH(RollingPolicy):
    """B6 (see the module docstring)."""

    name = "auction-RH"

    def __init__(self, rule: str = "cbta", objective: str = "makespan", seed_base: int = 0,
                 search: SearchConfig | None = None, triggers: frozenset[str] = STRUCTURAL):
        if rule not in RULES:
            raise ValueError(f"rule must be one of {RULES}, got {rule!r}")
        if objective not in ("start", "makespan"):
            raise ValueError(f"objective must be 'start' or 'makespan', got {objective!r}")
        super().__init__(triggers, seed_base)
        self.rule, self.objective = rule, objective
        self.planner = RHPlanner(search)
        self.name = f"B6-{rule}" + (f"-{objective}" if rule == "cbta" else "")
        self.stats: list[dict] = []

    def solve(self, state: PlanState, scope: np.ndarray | None, seed: int, first: bool) -> DynPlan:
        c0 = time.process_time()
        planner = self.planner
        prob = Problem(state, scope)
        T = state.inst.n_tasks
        if prob.Tk == 0:
            return DynPlan(members=[()] * T, keys=np.full(T, np.nan), makespan=float("nan"), seed=int(seed),
                           cpu_s=time.process_time() - c0)
        K.seed(int(seed) & 0x7FFFFFFF)
        ctx = self._prefix(prob, seed)
        sol = ctx["sol"]
        placed = set(prob.order_of(sol))
        todo = [x for x in range(prob.na, prob.Tk) if x not in placed]
        floor_ms = float(sol[10][0])
        if self.rule == "seq":
            rounds = self._sequential(prob, ctx, todo)
        else:
            rounds = self._append(prob, sol, todo)
        plan = planner._extract(prob, sol, ctx["demoted"], ctx["released"], 0, seed, floor_ms,
                                time.process_time() - c0)
        plan.iterations = rounds
        self.stats.append({"t": float(state.now), "open": len(todo), "rounds": rounds, "unplaced": len(plan.unplaced),
                           "cpu_s": plan.cpu_s})
        return plan

    # -- the anchored prefix and residual repair (SPARC's floor, no open task placed) -----------------------------
    def _prefix(self, prob: Problem, seed: int) -> dict:
        planner = self.planner
        sol = prob.new_sol()
        ctx = {"sol": sol, "ws": prob.workspace(), "slot_ok": np.zeros(prob.Tk + 1, np.bool_),
               "posn": np.full(prob.Tk, -1, np.int64), "best_c": np.zeros(prob.A, np.int64),
               "regret_c": np.zeros(prob.A, np.int64)}
        members: list[tuple[int, ...]] = [()] * prob.Tk
        order = []
        for x in range(prob.na):
            if prob.mode[x] == K.M_ANCHOR:
                members[x] = tuple(prob.fmem[x, :prob.fcnt[x]])
                order.append(x)
        prob.load(sol, order, members, planner.cfg.lam)
        todo_res = [x for x in range(prob.na) if prob.mode[x] == K.M_RESIDUAL]
        ctx["demoted"], ctx["released"] = planner._floor(prob, sol, ctx["ws"], ctx["slot_ok"], ctx["posn"],
                                                         ctx["best_c"], ctx["regret_c"], todo_res, [], seed,
                                                         planner.cfg.floor, 0.0)
        return ctx

    # -- cbta / cbga: append-only timetables ----------------------------------------------------------------------
    def _append(self, prob: Problem, sol: tuple, todo: list[int]) -> int:
        a = prob.arrays
        dur, ready = a[K.I_DUR], a[K.I_READY]
        lam = self.planner.cfg.lam
        by = "travel" if self.rule == "cbga" else "arrival"
        rounds = 0
        todo = list(todo)
        while todo:
            finish, last = sol[5], sol[9]
            free = np.where(last >= 0, finish[np.maximum(last, 0)], ready)
            best, best_key = None, None
            for x in todo:
                m, s, cost = append_offer(prob, free, last, x, by)
                if m is None:
                    continue
                if self.rule == "cbga":
                    k = (cost, s, x)
                elif self.objective == "start":
                    k = (s, s + dur[x], x)
                else:
                    k = (s + dur[x], s, x)
                if best_key is None or k < best_key:
                    best, best_key = (x, m), k
            if best is None:
                break  # no robot in scope can cover the rest (reported unplaced)
            x, m = best
            members = np.array(m, np.int64)
            K.apply_insertion(x, int(sol[3][0]), members, len(m), sol)
            K.schedule(prob.arrays, sol, lam)
            todo.remove(x)
            rounds += 1
        return rounds

    # -- seq: cheapest insertion with the kernel's bids -----------------------------------------------------------
    def _sequential(self, prob: Problem, ctx: dict, todo: list[int]) -> int:
        cfg = self.planner.cfg
        sol, ws, slot_ok, posn = ctx["sol"], ctx["ws"], ctx["slot_ok"], ctx["posn"]
        cover = np.zeros(prob.A, np.int64)
        rounds = 0
        todo = list(todo)
        while todo:
            best = None
            for x in todo:
                K.mark_slots(x, prob.arrays, sol, slot_ok, posn, cfg.max_slots)
                p, obj, _, c, _ = K.insertion(x, prob.arrays, sol, ws, slot_ok, 0.0, cfg.lam, False, cover)
                if p < 0:
                    continue
                if best is None or (obj, x) < (best[1], best[0]):
                    best = (x, obj, p, c, cover[:c].copy())
            if best is None:
                break
            x, _, p, c, m = best
            K.apply_insertion(x, p, m, c, sol)
            K.schedule(prob.arrays, sol, cfg.lam)
            todo.remove(x)
            rounds += 1
        return rounds


def make_controller(env, realization, tier: str, seed: int, *, rule: str = "cbta", objective: str = "makespan",
                    seed_base: int = 0) -> PlanController:
    """``scripts/trackD_run.py`` factory (``ctrl:cbba_sota.dyn.baselines.cbta:make_controller?rule=...``); native
    tier (no budget). ``seed`` (the runner's) is unused: the auctions are deterministic."""
    policy = AuctionRH(rule, objective, seed_base=seed_base)
    return PlanController(policy, env.nominal_instance(), realization.kappa())
