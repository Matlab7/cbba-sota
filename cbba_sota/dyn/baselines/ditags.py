"""B5: D-ITAGS-style targeted repair (docs/trackD-spec.md Section 5.2). In-house reimplementation: D-ITAGS
(RA-L 2023) has no public code, so this is the spec's reading of it -- "insertion plus ALNS restricted to coalitions
touched by the event" -- on SPARC's planner and budget, not a port of the original.

At every structural event (same triggers as SPARC): warm start from the incumbent, insertion floor for new,
orphaned and residual tasks (as SPARC), then ``iters`` ALNS iterations whose destroy and swap moves may only touch
the *event region*:
- tasks the floor placed (not in the warm start): new releases, orphans of a failure, residual and re-covered
  committed tasks;
- for an idle robot (no planned task), the ``near_k`` open tasks nearest to it (its chance to take work);
- closed once under shared membership: every task in the route of a robot that serves a touched task.
Repair (insertion) may still place removed tasks anywhere. The first plan (no incumbent) searches everything, like
SPARC. The difference to SPARC is therefore only the restriction of the search neighbourhood to the event region.
"""
from __future__ import annotations

import numpy as np

from cbba_sota.dyn import rh_kernels as K
from cbba_sota.dyn.executor import STRUCTURAL
from cbba_sota.dyn.planner import DynPlan, PlanState, Problem, RHPlanner, SearchConfig
from cbba_sota.dyn.sparc import RollingPolicy

__all__ = ["DITAGS", "event_region"]


def event_region(prob: Problem, sol: tuple, warm: list[int], near_k: int = 5) -> np.ndarray:
    """[Tk] bool: tasks the targeted search may move (see the module docstring)."""
    Tk, A = prob.Tk, prob.A
    seq, mem, cnt, ni = sol[:4]
    order = [int(x) for x in seq[:ni[0]]]
    warm_set = set(warm)
    touched = np.zeros(Tk, bool)
    for x in order:
        if prob.mode[x] != K.M_ANCHOR and x not in warm_set:
            touched[x] = True
    busy = np.zeros(A, bool)
    for x in order:
        busy[mem[x, :cnt[x]]] = True
    free_tasks = np.array([x for x in order if prob.mode[x] == K.M_FREE], np.int64)
    dout = prob.arrays[K.I_DOUT]
    if free_tasks.size:
        for i in np.flatnonzero(prob.scope & ~busy):
            if prob.head[i] >= 0:
                continue
            near = free_tasks[np.argsort(dout[i, free_tasks], kind="stable")[:near_k]]
            touched[near] = True
    robots = set()
    for x in np.flatnonzero(touched):
        robots.update(int(i) for i in mem[x, :cnt[x]])
    for x in order:
        if any(int(i) in robots for i in mem[x, :cnt[x]]):
            touched[x] = True
    return touched


class DITAGS(RollingPolicy):
    name = "D-ITAGS-style"

    def __init__(self, iters: int = 300, iters0: int | None = None, seed_base: int = 0,
                 search: SearchConfig | None = None, near_k: int = 5, triggers: frozenset[str] = STRUCTURAL):
        super().__init__(triggers, seed_base)
        self.iters, self.iters0, self.near_k = int(iters), iters0, int(near_k)
        self.planner = RHPlanner(search)

    def solve(self, state: PlanState, scope: np.ndarray | None, seed: int, first: bool) -> DynPlan:
        if first:
            iters = self.iters0 if self.iters0 is not None else self.iters
            return self.planner.plan(state, None, scope, seed, iters)
        return self.planner.plan(state, self.incumbent, scope, seed, self.iters,
                                 removable=lambda prob, sol, warm: event_region(prob, sol, warm, self.near_k))
