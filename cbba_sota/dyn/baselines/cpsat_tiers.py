"""B3 budget tiers: rolling state-start CP-SAT-LNS (``cpsat_rh.CPSATRH``) at the Light and Heavy tiers.

``cpsat_rh`` bounds an event by the *summed CP-SAT deterministic time* of its sub-solves. That does not bound its CPU:
sub-solves that prove optimality report a tiny deterministic time, so the loop builds many small models in Python
(dev calibration 2026-09-28, MA-AT-25-5-50 dev 0 F1-R2: ``dtime=0.02`` gave a per-event CPU p95 of 7.8 s,
``dtime=2.0`` with 8 workers 70 s). The tiers here are therefore a fixed number of sub-solves per event, each with
its own ``max_deterministic_time`` (deterministic, spec 3.6), calibrated once to the tier's CPU:

- Light (1 core, the CPU of 300 SPARC iterations, about 15-30 ms per event on this host): one sub-solve of 0.02
  deterministic time. It cannot be made cheaper (building one sub-model costs 70-130 ms here), so B3-Light is
  *over* the Light budget by about 5x; a declared, generous choice.
- Heavy (8 cores x 1 s = 8 CPU seconds per event): sequential single-worker sub-solves of 0.05 deterministic time,
  as many as 8 CPU-s buys in the setting (``HEAVY_SUBSOLVES``: 0.08-0.25 CPU s each, measured). The 8 cores are
  spent in sequence rather than as 8 interleaved CP-SAT workers: at equal CPU the single-worker LNS found at least as many improvements in the
  calibration (8 interleaved workers used about 1.3 cores per sub-solve here), and a single worker keeps the process
  count within the host cap. Same determinism either way.

Calibration evidence and commands: ``docs/results/trackD-week1/baselines.md``. Provisional until the quiet-core
measurement (gate K8).
"""
from __future__ import annotations

from cbba_sota.dyn.baselines.cpsat_rh import CPSATRH, CPSATConfig
from cbba_sota.dyn.controller import PlanController

__all__ = ["B3_TIERS", "HEAVY_SUBSOLVES", "make_controller", "make_policy", "setting_of"]

B3_TIERS = {"light": {"subsolves": 1, "sub_dtime": 0.02, "workers": 1},
            "heavy": {"subsolves": "auto", "sub_dtime": 0.05, "workers": 1}}
# Heavy sub-solves per event by setting: 8 CPU-s / measured CPU per 0.05-dtime single-worker sub-solve (0.247, 0.171,
# 0.080, 0.082 s; one dev calibration episode per setting, loaded host, 2026-09-28); 40 for any other size.
HEAVY_SUBSOLVES = {"MA-AT-25-5-50": 32, "MA-AT-50-5-50": 47, "SA-AT-50-5-50": 100, "SA-BT-50-5-50": 98}


def setting_of(inst) -> str | None:
    """Primary-setting name of an ``Instance`` from its shape (agents, tasks, single-skill, binary requirements)."""
    sa = bool(((inst.ab > 0).sum(axis=1) == 1).all())
    bt = bool(inst.req.max() <= 1)
    fam = ("SA" if sa else "MA") + "-" + ("BT" if bt else "AT")
    name = f"{fam}-{inst.n_agents}-{inst.n_species}-{inst.n_tasks}"
    return name if name in HEAVY_SUBSOLVES else None


def make_policy(tier: str = "light", seed_base: int = 0, inst=None, **params) -> CPSATRH:
    """B3 at ``tier``; ``params`` (subsolves, sub_dtime, workers) override the tier. ``subsolves="auto"`` (Heavy)
    takes the per-setting count of ``HEAVY_SUBSOLVES`` for ``inst`` (40 if unknown)."""
    if tier not in B3_TIERS:
        raise ValueError(f"B3 tier must be light or heavy, got {tier!r}")
    p = {**B3_TIERS[tier], **params}
    if p["subsolves"] == "auto":
        p["subsolves"] = HEAVY_SUBSOLVES.get(setting_of(inst) if inst is not None else None, 40)
    cfg = CPSATConfig(dtime=float("inf"), sub_dtime=float(p["sub_dtime"]), workers=int(p["workers"]),
                      max_subsolves=int(p["subsolves"]))
    return CPSATRH(cfg, seed_base=seed_base)


def make_controller(env, realization, tier: str, seed: int, seed_base: int = 0, **params) -> PlanController:
    """``scripts/trackD_run.py`` factory: ``ctrl:cbba_sota.dyn.baselines.cpsat_tiers:make_controller`` with
    ``--tier light|heavy``. ``seed`` (the runner's) is unused: planner seeds come from the belief (spec 4.1)."""
    inst = env.nominal_instance()
    return PlanController(make_policy(tier, seed_base, inst=inst, **params), inst, realization.kappa())
