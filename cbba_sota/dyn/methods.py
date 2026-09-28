"""Rolling planners of the Track D comparison by name, for ``scripts/trackD_run.py`` and the K0 check.

One place builds every plan-following method (docs/trackD-spec.md Sections 4 and 5.2), so that the campaign runner,
the K0 fidelity check (``scripts/trackD_k0.py``) and the tests construct them identically:

  SPARC        SPARC (spec 4.1/4.2): warm ALNS repair on structural events; Light 300 / Heavy 3000 iterations
  SPARC-rk     the same with relaxed keys (committed order + head-first; ``planner.SearchConfig(keys="relaxed")``)
  ins-only     insertion floor only at every structural event (T1b)
  every-event  R-b: SPARC's planner and budget, also re-planning at every task finish (T2)
  open-loop    R-c: the t = 0 plan with SPARC's budget, then insertion of releases and orphans only
  B3           rolling state-start CP-SAT-LNS; deterministic time ``dtime`` per event (Light: 1 worker)
  B4           rolling constructor with restarts (collapse control T1); Light only
  B5           D-ITAGS-style targeted repair, SPARC's budget

Budgets are deterministic (spec 3.6): iteration counts, restart counts, CP-SAT deterministic time. ``TIERS`` holds the
defaults; any of them can be overridden by a parameter (``iters``, ``iters0``, ``restarts``, ``dtime``, ``workers``).
The B4 restart count (12) and the B3 deterministic time are provisional until the quiet-core calibration (gate K8).
Planner seeds are the spec's ``hash(belief digest, event index)`` with ``seed_base`` 0 (spec 4.1), not the runner's
method seed, so SPARC and its references R-a/R-b share seeds on identical beliefs (T4).
"""
from __future__ import annotations

__all__ = ["PLANNERS", "TIERS", "make_controller", "make_policy"]

PLANNERS = ("SPARC", "SPARC-rk", "ins-only", "every-event", "open-loop", "B3", "B4", "B5")

TIERS: dict[str, dict[str, dict]] = {
    "light": {"SPARC": {"iters": 300}, "SPARC-rk": {"iters": 300}, "ins-only": {}, "every-event": {"iters": 300},
              "open-loop": {"iters0": 300}, "B3": {"dtime": 0.02, "sub_dtime": 0.02, "workers": 1},
              "B4": {"restarts": 12}, "B5": {"iters": 300}},
    "heavy": {"SPARC": {"iters": 3000}, "SPARC-rk": {"iters": 3000}, "every-event": {"iters": 3000},
              "open-loop": {"iters0": 3000}, "B3": {"workers": 8}, "B5": {"iters": 3000}},
}


def make_policy(name: str, tier: str = "light", seed_base: int = 0, **params):
    """The ``RollingPolicy`` of planner ``name`` at budget ``tier`` (``params`` override the tier defaults)."""
    from cbba_sota.dyn.planner import SearchConfig
    from cbba_sota.dyn.sparc import (
        SPARC,
        OpenLoop,
        SPARCConfig,
        every_event_rh,
        insertion_only,
    )

    if name not in PLANNERS:
        raise ValueError(f"unknown planner {name!r}; known: {', '.join(PLANNERS)}")
    if tier not in TIERS:
        raise ValueError(f"planner {name} needs tier light or heavy, got {tier!r}")
    if name not in TIERS[tier]:
        raise ValueError(f"planner {name} has no {tier} tier")
    p = {**TIERS[tier][name], **params}
    if name == "B3" and "dtime" not in p:
        raise ValueError("B3 heavy needs an explicit dtime (deterministic time per event; calibration pending)")
    iters0 = p.get("iters0")
    if name == "SPARC":
        return SPARC(SPARCConfig(iters=int(p["iters"]), iters0=iters0, seed_base=seed_base), name="SPARC")
    if name == "SPARC-rk":
        return SPARC(SPARCConfig(iters=int(p["iters"]), iters0=iters0, seed_base=seed_base,
                                 search=SearchConfig(keys="relaxed")), name="SPARC-rk")
    if name == "ins-only":
        return insertion_only(seed_base=seed_base, iters0=iters0)
    if name == "every-event":
        return every_event_rh(iters=int(p["iters"]), seed_base=seed_base, iters0=iters0)
    if name == "open-loop":
        return OpenLoop(iters0=int(p["iters0"]), seed_base=seed_base)
    if name == "B4":
        from cbba_sota.dyn.baselines.constructor_rh import ConstructorRH

        return ConstructorRH(restarts=int(p["restarts"]), restarts0=p.get("restarts0"), seed_base=seed_base)
    if name == "B5":
        from cbba_sota.dyn.baselines.ditags import DITAGS

        return DITAGS(iters=int(p["iters"]), iters0=iters0, seed_base=seed_base)
    from cbba_sota.dyn.baselines.cpsat_rh import CPSATRH, CPSATConfig

    cfg = CPSATConfig(dtime=float(p["dtime"]), dtime0=p.get("dtime0"),
                      sub_dtime=float(p.get("sub_dtime", min(0.05, float(p["dtime"])))), workers=int(p["workers"]))
    return CPSATRH(cfg, seed_base=seed_base)


def make_controller(env, realization, tier: str, seed: int, method: str = "SPARC", seed_base: int = 0, **params):
    """``scripts/trackD_run.py`` factory: a ``PlanController`` running planner ``method`` in ``env`` (``run_plan``).
    ``seed`` (the runner's method seed) is recorded but not used: planner seeds come from the belief (spec 4.1)."""
    from cbba_sota.dyn.controller import PlanController

    policy = make_policy(method, tier, seed_base=seed_base, **params)
    return PlanController(policy, env.nominal_instance(), realization.kappa())
