"""Track D planner (agent B, day 1): state-start kernel, SPARC planner and its execution loop.

- Static reduction: on a static problem the planner's prediction equals the core evaluator and the env replay.
- Noise-free static runs: SPARC in the event loop is within 1% of open-loop ALNS (spec Section 8, day 1 gate).
  World: ``cbba_sota.dyn.executor`` (surrogate, pre-K0); its open-loop makespan is checked against env replay.
- Plans are minimal covers and key-ordered at every re-plan, also under releases, noise and failures (residual
  repair of committed coalitions, G4 fallback), and every episode completes (G5, good communication).
- Determinism (G3): the same belief and event index give the same plan; budgets are iteration counts.
- Trigger rule (4.2): SPARC never re-plans on noise-only events.
"""
import numpy as np
import pytest

from cbba_sota.bench.configs import get
from cbba_sota.dyn.executor import Executor, make_realization
from cbba_sota.dyn.planner import (
    PlanState,
    RHPlanner,
    SearchConfig,
    check_plan,
    state_digest,
)
from cbba_sota.dyn.sparc import NOISE_KINDS, SPARC, OpenLoop, SPARCConfig, derive_seed
from cbba_sota.hetero import Instance, Plan, evaluate, replay

SETTINGS = ["MA-AT-25-5-50", "MA-AT-50-5-50", "SA-AT-50-5-50", "SA-BT-50-5-50"]
H = {"MA-AT-25-5-50": 25.0, "MA-AT-50-5-50": 15.0, "SA-AT-50-5-50": 20.0, "SA-BT-50-5-50": 12.0}


def _inst(setting: str, i: int) -> tuple[Instance, int]:
    st = get(setting)
    return Instance.from_pickle(st.instance_path("dev", i)), st.seed("dev", i)


class Checked(SPARC):
    """SPARC that validates every plan against the belief it was computed from."""

    def __init__(self, cfg: SPARCConfig):
        super().__init__(cfg)
        self.errors: list[str] = []
        self.plans = 0
        self.residuals = 0

    def replan(self, state, scope=None, event_index=None):
        plan = super().replan(state, scope, event_index)
        self.errors += check_plan(state, plan, scope, relaxed=self.cfg.search.keys == "relaxed")
        self.plans += 1
        self.residuals += len(plan.residual)
        return plan


def test_static_plan_is_exact_and_minimal():
    inst, _ = _inst("MA-AT-25-5-50", 0)
    state = PlanState.static(inst)
    for iters in (0, 200):
        p = RHPlanner().plan(state, seed=3, iters=iters)
        assert p.iterations == iters and not p.unplaced
        assert check_plan(state, p) == []
        plan = Plan(p.members, p.keys, inst.n_agents)
        assert plan.is_minimal_cover(inst)
        Plan.from_routes(plan.routes(), inst.n_tasks)  # acyclic
        assert evaluate(inst, plan).makespan == p.makespan
        assert replay(inst, plan)["makespan"] == p.makespan


def test_same_belief_same_plan():
    inst, _ = _inst("SA-BT-50-5-50", 1)
    state = PlanState.static(inst)
    seed = derive_seed(state_digest(state), 0)
    a = RHPlanner().plan(state, seed=seed, iters=150)
    b = RHPlanner().plan(PlanState.static(inst), seed=derive_seed(state_digest(PlanState.static(inst)), 0), iters=150)
    assert a.seed == b.seed and a.members == b.members and np.array_equal(a.keys, b.keys)
    c = RHPlanner().plan(state, seed=seed + 1, iters=150)
    assert c.seed != a.seed


@pytest.mark.parametrize(("idle", "keys"), [("env", "monotone"), ("env", "relaxed"), ("eager", "monotone"),
                                            ("eager", "relaxed")])
def test_noise_free_static_sparc_matches_open_loop(idle, keys):
    """Day-1 gate: every task released at 0, no noise; SPARC within 1% (geometric mean over 8 dev instances) of the
    open-loop ALNS plan (same first plan, 1000 iterations).

    ``idle="env"`` (the trigger semantics of ``DynTaskEnvX``: with H = 0 a robot out of work heads home, no idle
    trigger): SPARC never re-plans after t = 0 and must equal open loop exactly (the gate). ``idle="eager"`` is a
    planner stress in which every robot that runs out of work while open tasks exist triggers a re-plan. With exact
    predictions the relaxed key rule can only improve on the incumbent (asserted per instance). The spec's
    monotone rule can lose: on 20 dev instances (5 per primary setting, ``scripts/trackD_static_gate.py``) GM
    SPARC/open-loop was 1.0137 (worst +8.2%) with 2000 first-plan iterations and 1.0050 (worst +8.0%) with 1000,
    so for it only validity and completion are asserted here."""
    logs = []
    for setting in SETTINGS:
        for i in range(2):
            inst, key = _inst(setting, i)
            world = make_realization(inst, key, 0)
            ol = OpenLoop(iters0=1000)
            r_ol = Executor(inst, world).run(ol)
            plan0 = Plan(ol.incumbent.members, ol.incumbent.keys, inst.n_agents)
            assert r_ol["success"] and r_ol["makespan"] == ol.incumbent.makespan
            assert replay(inst, plan0)["makespan"] == r_ol["makespan"]  # executor == env replay (static)
            sp = Checked(SPARCConfig(iters=300, iters0=1000, search=SearchConfig(keys=keys)))
            r_sp = Executor(inst, world, idle=idle).run(sp)
            assert r_sp["success"] and not sp.errors, sp.errors[:3]
            ratio = r_sp["makespan"] / r_ol["makespan"]
            if idle == "env":
                assert ratio == 1.0 and r_sp["n_replans"] == 1
            elif keys == "relaxed":
                assert ratio <= 1 + 1e-9
            logs.append(np.log(ratio))
    if (idle, keys) != ("eager", "monotone"):
        assert np.exp(np.mean(logs)) <= 1.01


@pytest.mark.parametrize("keys", ["monotone", "relaxed"])
@pytest.mark.parametrize("family", ["R2+N3", "F3"])
def test_dynamic_plans_valid_and_complete(keys, family):
    for setting in ("MA-AT-25-5-50", "SA-AT-50-5-50"):
        inst, key = _inst(setting, 0)
        for crn in range(2):
            if family == "R2+N3":
                kw = {"release": "poisson", "dod": 0.5, "horizon": H[setting], "dur_sigma": 0.3, "p_delay": 0.05,
                      "min_delay": 1, "max_delay": 10}
            else:
                kw = {"dur_sigma": 0.3, "p_delay": 0.01, "min_delay": 1, "max_delay": 4, "fail_p": 0.2,
                      "fail_window": 21.0}
            world = make_realization(inst, key, crn, **kw)
            alive = ~np.isfinite(world.fail_t)
            if not (inst.ab[alive].sum(axis=0) >= inst.req).all():
                continue  # excluded realization (spec 3.3)
            pol = Checked(SPARCConfig(iters=100, search=SearchConfig(keys=keys)))
            r = Executor(inst, world).run(pol)
            assert not pol.errors, pol.errors[:3]
            assert r["success"] and r["completion"] == 1.0, r
            assert pol.plans == r["n_replans"]
            for d in pol.decisions:  # trigger rule: never on noise alone
                if set(d["kinds"]) <= NOISE_KINDS:
                    assert not d["replan"]
            if family == "F3" and r["n_failed"]:
                assert pol.residuals > 0 or r["aborts"] == 0


def test_residual_keeps_frozen_members_and_adds_minimal_extras():
    """A committed coalition that loses a member keeps its survivors (locked to it) and gets extra members only
    for the missing traits; the committed key is kept (monotone rule)."""
    inst, _ = _inst("MA-AT-25-5-50", 3)
    state = PlanState.static(inst)
    p0 = RHPlanner().plan(state, seed=0, iters=200)
    order = [int(j) for j in np.argsort(p0.keys) if p0.members[j]]
    j = next(j for j in order if len(p0.members[j]) >= 2)
    dead, survivors = p0.members[j][0], p0.members[j][1:]
    st = PlanState.static(inst)
    st.committed[j] = True
    st.members[j] = p0.members[j]
    st.keys[j] = 5.0
    st.alive[dead] = False
    for i in survivors:
        st.head[i] = j
        st.pos_task[i] = j
        st.pos[i] = inst.loc[j]
        st.ready[i] = inst.da[i, j]
    p = RHPlanner().plan(st, incumbent=p0, seed=1, iters=100)
    assert check_plan(st, p) == []
    assert set(survivors) <= set(p.members[j]) and dead not in p.members[j]
    covered = (inst.ab[list(p.members[j])].sum(axis=0) >= inst.req[j]).all()
    assert covered
    if (inst.ab[list(survivors)].sum(axis=0) >= inst.req[j]).all():
        assert p.members[j] == tuple(sorted(survivors))
    else:
        assert j in p.residual + p.demoted + p.released
    if j not in p.demoted + p.released:
        assert p.keys[j] == 5.0
        assert all(p.keys[k] > 5.0 for k in range(inst.n_tasks) if p.members[k] and k != j)
    routes = p.routes_for(inst.n_agents)
    for i in survivors:
        assert routes[i][0] == j
    assert not routes[dead]
