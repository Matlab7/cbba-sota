"""Track D optimisation baselines (agent B, spec 5.2) and the env controller, on ``DynTaskEnvX`` (ground truth).

Every rolling planner (SPARC, insertion-only, every-event RH, constructor with restarts, D-ITAGS-style targeted
repair, state-start CP-SAT-LNS) runs through ``PlanController`` in ``DynTaskEnvX.run_plan``; every plan is checked
against the plan contract (minimal covers, key order, frozen commitments, head locks), every episode must complete,
runs are deterministic, and SPARC's static run in the env equals its plan and the surrogate executor.
"""
import numpy as np
import pytest

from cbba_sota.bench.configs import get
from cbba_sota.dyn.baselines.constructor_rh import ConstructorRH
from cbba_sota.dyn.baselines.cpsat_rh import CPSATRH, CPSATConfig
from cbba_sota.dyn.baselines.ditags import DITAGS, event_region
from cbba_sota.dyn.controller import run_env
from cbba_sota.dyn.executor import Executor, Realization
from cbba_sota.dyn.perturb import realize
from cbba_sota.dyn.planner import PlanState, RHPlanner, SearchConfig, check_plan
from cbba_sota.dyn.sparc import SPARC, SPARCConfig, every_event_rh, insertion_only
from cbba_sota.hetero import Instance, Plan, replay

SETTING = "MA-AT-25-5-50"


@pytest.fixture(scope="module")
def inst() -> Instance:
    return Instance.from_pickle(get(SETTING).instance_path("dev", 0))


def _real(inst: Instance, cell: str, crn: int = 0):
    return realize(SETTING, get(SETTING).seed("dev", 0), crn, cell, inst.req, inst.ab, inst.dur)


def _checked(policy):
    errs: list[str] = []
    orig = policy.replan

    def replan(state, scope=None, event_index=None):
        plan = orig(state, scope, event_index)
        errs.extend(check_plan(state, plan, scope))
        return plan

    policy.replan = replan
    return policy, errs


FACTORIES = {
    "SPARC": lambda: SPARC(SPARCConfig(iters=100)),
    "insertion-only": lambda: insertion_only(),
    "every-event": lambda: every_event_rh(iters=50),
    "constructor": lambda: ConstructorRH(restarts=3),
    "ditags": lambda: DITAGS(iters=100),
    "cpsat": lambda: CPSATRH(CPSATConfig(dtime=0.01, sub_dtime=0.005)),
}


@pytest.mark.parametrize("name", list(FACTORIES))
@pytest.mark.parametrize("cell", ["F1-R2", "F3-pf0.2"])
def test_planner_in_env_valid_and_complete(inst, name, cell):
    real = _real(inst, cell)
    assert not real.excluded
    policy, errs = _checked(FACTORIES[name]())
    ep, _ = run_env(policy, inst, real)
    assert not errs, errs[:3]
    assert ep.success and ep.completion == 1.0, (name, cell, ep)
    assert ep.world.startswith("DynTaskEnvX")
    assert ep.replans == policy.n_calls >= 1
    if name != "every-event":  # trigger rule: never on noise events alone
        for d in policy.decisions:
            if d["replan"]:
                assert set(d["kinds"]) & {"release", "orphan", "idle", "membership"}


@pytest.mark.parametrize("name", ["SPARC", "constructor", "ditags"])
def test_deterministic(inst, name):
    real = _real(inst, "F2-R2N3")
    a, _ = run_env(FACTORIES[name](), inst, real)
    b, _ = run_env(FACTORIES[name](), inst, real)
    assert a.makespan == b.makespan and a.replans == b.replans


def test_static_env_equals_plan_and_executor(inst):
    real = _real(inst, "static")
    sp = SPARC(SPARCConfig(iters=300))
    ep, _ = run_env(sp, inst, real)
    plan = sp.incumbent
    assert ep.makespan == plan.makespan == replay(inst, Plan(plan.members, plan.keys, inst.n_agents))["makespan"]
    r = Executor(inst, Realization.from_perturb(real), detect_h=real.detect_after).run(SPARC(SPARCConfig(iters=300)))
    assert r["makespan"] == ep.makespan


def test_constructor_restarts_never_worse(inst):
    state = PlanState.static(inst)
    p0 = RHPlanner().plan(state, seed=5, iters=0, warm=False)
    p8 = RHPlanner().plan(state, seed=5, iters=0, warm=False, restarts=8)
    assert p8.makespan <= p0.makespan and check_plan(state, p8) == []


def test_ditags_region_is_event_local(inst):
    state = PlanState.static(inst)
    p0 = RHPlanner().plan(state, seed=1, iters=100)
    st = PlanState.static(inst)
    st.released[:] = True
    new = int(np.argmax(p0.keys))  # pretend the last task was just released: drop it from the incumbent
    inc = type(p0)(members=list(p0.members), keys=p0.keys.copy(), makespan=p0.makespan)
    inc.members[new] = ()
    seen = {}

    def region(prob, sol, warm):
        mask = event_region(prob, sol, warm)
        seen["mask"], seen["gid"] = mask, prob.gid
        return mask

    p = RHPlanner().plan(st, inc, seed=2, iters=50, removable=region)
    assert check_plan(st, p) == []
    touched = set(seen["gid"][seen["mask"]].tolist())
    assert new in touched and len(touched) < inst.n_tasks
    robots = set(p.members[new])
    assert all(j in touched for j in range(inst.n_tasks) if set(p.members[j]) & robots)


@pytest.mark.parametrize(("setting", "i", "crn", "make"), [
    ("SA-AT-50-5-50", 0, 1, lambda: SPARC(SPARCConfig(iters=100))),
    ("SA-AT-50-5-50", 3, 1, lambda: ConstructorRH(restarts=12)),
    ("SA-AT-50-5-50", 4, 1, lambda: SPARC(SPARCConfig(iters=300, search=SearchConfig(keys="relaxed")))),
    ("SA-AT-50-5-50", 7, 2, lambda: SPARC(SPARCConfig(iters=300, search=SearchConfig(keys="relaxed")))),
])
def test_released_commitment_regressions(setting, i, crn, make):
    """Regressions from the day-1 pilot (F3-pf0.2): a failure turns a committed task into a residual that cannot
    be covered at its rank and the G4 fallback releases it while members still travel there. Those robots leave on
    arrival and must not be re-locked to the task (neither from the env's member list nor from a snapshot taken
    before their abandon), and a new plan may send them back to it later."""
    st = get(setting)
    inst = Instance.from_pickle(st.instance_path("dev", i))
    real = realize(setting, st.seed("dev", i), crn, "F3-pf0.2", inst.req, inst.ab, inst.dur)
    policy = make()
    errs: list[str] = []
    orig = policy.replan
    relaxed = getattr(getattr(policy, "cfg", None), "search", SearchConfig()).keys == "relaxed"

    def replan(state, scope=None, event_index=None):
        plan = orig(state, scope, event_index)
        errs.extend(check_plan(state, plan, scope, relaxed=relaxed))
        return plan

    policy.replan = replan
    ep, _ = run_env(policy, inst, real)
    assert not errs, errs[:3]
    assert ep.success
