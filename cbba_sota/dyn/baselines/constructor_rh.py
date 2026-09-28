"""B4: rolling constructor with restarts (docs/trackD-spec.md Section 5.2) -- the simple-rule re-planner and the
collapse control of test T1.

At every structural event (the same trigger rule as SPARC, 4.2) the plan of all open tasks is rebuilt from the
current state (anchored commitments kept, state start, same predictors): one deterministic regret-2 insertion, then
``restarts`` randomized constructions (seeded random-order best insertion, earliest-arrival covers with
multiplicative noise on every second one; the alternation of the pinned Phase-1 ``greedy.construct``), and the best
by (tasks placed, makespan, objective) is adopted. No warm start and no local search: the incumbent is ignored.

Budget: a restart count (deterministic). The spec's Light tier is the CPU-ms equivalent of 300 SPARC iterations
on one quiet core; ``restarts`` is to be calibrated once to that (``scripts/trackD_pilot.py --calibrate`` reports
CPU per construction and per ALNS iteration on dev states), so the default here is provisional.
"""
from __future__ import annotations

import numpy as np

from cbba_sota.dyn.executor import STRUCTURAL
from cbba_sota.dyn.planner import DynPlan, PlanState, RHPlanner, SearchConfig
from cbba_sota.dyn.sparc import RollingPolicy

__all__ = ["ConstructorRH"]


class ConstructorRH(RollingPolicy):
    name = "constructor-RH"

    def __init__(self, restarts: int = 8, restarts0: int | None = None, seed_base: int = 0,
                 search: SearchConfig | None = None, noise: float = 0.2, triggers: frozenset[str] = STRUCTURAL):
        super().__init__(triggers, seed_base)
        self.restarts, self.restarts0, self.noise = int(restarts), restarts0, float(noise)
        self.planner = RHPlanner(search)

    def solve(self, state: PlanState, scope: np.ndarray | None, seed: int, first: bool) -> DynPlan:
        r = self.restarts0 if first and self.restarts0 is not None else self.restarts
        return self.planner.plan(state, None, scope, seed, iters=0, warm=False, restarts=r,
                                 restart_noise=self.noise)
