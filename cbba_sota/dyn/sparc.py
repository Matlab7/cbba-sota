"""SPARC: Structural-event Planning with Anchored Replication of Coalitions (docs/trackD-spec.md Section 4).

This module holds the planning side of SPARC:
- the trigger rule (4.2): re-plan only on structural events (a release becomes known, an orphan from a failure
  detection or an abandon, a change of reachable membership, a live robot idle with an empty route while open
  tasks exist); never on noise events (finishes, starts, arrivals, stalls); no periodic trigger;
- the planner call (4.1): ``RHPlanner`` warm-started from the incumbent, insertion floor, ``iters`` ALNS
  iterations (Light 300, Heavy 3000), seed ``hash(replica digest, event index)`` so identical replicas compute
  identical plans (G3);
- the hook for the communication layer (agent C): ``SPARC.replan(state, scope, event_index)`` takes any replica's
  belief state and returns a plan for the robots in ``scope``; the protocol decides who calls it, with which
  belief and scope (4.5), and how the plan's route versions travel (4.4). Contract of that call:
  * robots outside ``scope`` (or not ``alive``) get no new work; if they must keep their believed routes (the
    station component, CEN-F), pass those routes' tasks as *committed* in the belief (``committed``, ``members``,
    ``keys``): the planner anchors them and may add extra members from the scope where a coalition is short;
  * a stranded leader re-planning everyone passes clamped snapshots of absent robots through ``ready``/``pos``
    (free at max(recorded ETA or finish, now), at the recorded target) and ``scope`` = every live robot;
  * the policy object keeps one incumbent (warm start); use one ``SPARC`` object per replica;
  * the returned ``DynPlan`` covers every live task (``routes_for`` gives per-robot route versions; ``released``
    lists commitments dropped by the G4 fallback, whose travelling members must leave on arrival).
Execution (4.3: follow the adopted route in key order, depart when the previous task finishes, coalitions start in
arrival order, lease = failure detector under good comms) lives in the world: ``controller.PlanController`` drives
the ground truth ``DynTaskEnvX``; ``executor.Executor`` is the faster surrogate (not K0-equivalent yet).

``RollingPolicy`` is the shared event loop of every rolling planner in the comparison (SPARC, its references R-a,
R-b, R-c, insertion-only, and the optimisation baselines of Section 5.2), so that triggers, beliefs, predictors,
seeds and plan adoption are identical and only ``solve`` differs.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np

from cbba_sota.dyn.executor import STRUCTURAL
from cbba_sota.dyn.planner import (
    DynPlan,
    PlanState,
    RHPlanner,
    SearchConfig,
    state_digest,
)

__all__ = [
    "NOISE_KINDS",
    "SPARC",
    "OpenLoop",
    "RollingPolicy",
    "SPARCConfig",
    "derive_seed",
    "every_event_rh",
    "insertion_only",
]

NOISE_KINDS = frozenset({"finish", "start", "arrive", "stall"})
EVERY = STRUCTURAL | frozenset({"finish"})  # R-b: every task finish as well (the pilot's every-event RH)


def derive_seed(digest: bytes, event_index: int, seed_base: int = 0) -> int:
    """Deterministic planner seed from a replica digest and an event index (spec 4.1)."""
    h = hashlib.blake2b(digest + int(event_index).to_bytes(8, "little") + int(seed_base).to_bytes(8, "little",
                                                                                                  signed=True),
                        digest_size=8)
    return int.from_bytes(h.digest(), "little") & 0x7FFFFFFF


class RollingPolicy:
    """Event loop shared by all rolling planners: trigger rule, belief, seed, incumbent, decision log."""

    name = "rolling"

    def __init__(self, triggers: frozenset[str] = STRUCTURAL, seed_base: int = 0):
        self.triggers = frozenset(triggers)
        self.seed_base = int(seed_base)
        self.incumbent: DynPlan | None = None
        self.n_calls = 0
        self.decisions: list[dict] = []

    def triggered(self, kinds: frozenset[str]) -> bool:
        return bool(kinds & self.triggers)

    def on_event(self, ex, t: float, kinds: frozenset[str]) -> DynPlan | None:
        if not self.triggered(kinds):
            self.decisions.append({"t": t, "kinds": sorted(kinds), "replan": False})
            return None
        plan = self.replan(ex.belief())
        self.decisions.append({"t": t, "kinds": sorted(kinds), "replan": True, "seed": plan.seed,
                               "iters": plan.iterations, "cpu_s": plan.cpu_s, "pred_ms": plan.makespan,
                               "n_open": plan.n_open})
        return plan

    def replan(self, state: PlanState, scope: np.ndarray | None = None, event_index: int | None = None) -> DynPlan:
        """Plan for ``scope`` from any replica's belief ``state`` (the comm-layer hook); updates the incumbent."""
        k = self.n_calls if event_index is None else int(event_index)
        seed = derive_seed(state_digest(state), k, self.seed_base)
        plan = self.solve(state, scope, seed, first=self.incumbent is None)
        self.incumbent = plan
        self.n_calls += 1
        return plan

    def solve(self, state: PlanState, scope: np.ndarray | None, seed: int, first: bool) -> DynPlan:
        raise NotImplementedError


@dataclass(frozen=True)
class SPARCConfig:
    iters: int = 300  # ALNS iterations per structural event (Light 300, Heavy 3000)
    iters0: int | None = None  # budget of the first plan (None: the same per-event budget)
    triggers: frozenset[str] = field(default=STRUCTURAL)
    seed_base: int = 0
    search: SearchConfig = field(default_factory=SearchConfig)
    warm: bool = True  # warm start from the incumbent (False: cold re-solve at every event)


class SPARC(RollingPolicy):
    """SPARC's planner under good communication (= central RH-ALNS with structural triggers, R-a)."""

    name = "SPARC"

    def __init__(self, config: SPARCConfig | None = None, name: str | None = None):
        cfg = config or SPARCConfig()
        super().__init__(cfg.triggers, cfg.seed_base)
        self.cfg = cfg
        self.planner = RHPlanner(cfg.search)
        if name:
            self.name = name

    def solve(self, state, scope, seed, first):
        iters = self.cfg.iters0 if first and self.cfg.iters0 is not None else self.cfg.iters
        return self.planner.plan(state, self.incumbent, scope, seed, iters, warm=self.cfg.warm)


def insertion_only(seed_base: int = 0, iters0: int | None = None) -> SPARC:
    """Insertion-only repair (T1b, ablation): the floor alone at every structural event."""
    return SPARC(SPARCConfig(iters=0, iters0=iters0, seed_base=seed_base), name="insertion-only")


def every_event_rh(iters: int = 300, seed_base: int = 0, iters0: int | None = None) -> SPARC:
    """R-b: the same planner and budget, re-planning at every task finish as well (design-rule comparison T2)."""
    return SPARC(SPARCConfig(iters=iters, iters0=iters0, triggers=EVERY, seed_base=seed_base), name="every-event-RH")


class OpenLoop(SPARC):
    """R-c: the first plan with the SPARC budget, then only insertion of new work (releases, orphans); no search
    after t = 0 and no re-planning on idle robots."""

    name = "open-loop"

    def __init__(self, iters0: int = 300, seed_base: int = 0):
        super().__init__(SPARCConfig(iters=0, iters0=iters0, triggers=frozenset({"release", "orphan", "membership"}),
                                     seed_base=seed_base))
