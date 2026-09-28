"""B6 coalition auctions (``cbba_sota.dyn.baselines.cbta``) and the B3 budget tiers (``cpsat_tiers``).

- Every auction plan satisfies the plan contract (``planner.check_plan``: minimal covers, frozen commitments, key
  monotonicity, head locks) at every re-plan, and episodes complete in ``DynTaskEnvX`` under release and failures.
- CBTA-style appends earliest-start offers: the offer's start is the earliest start any cover appended at the end
  of the timetables can reach (checked by brute force on small covers).
- Fidelity check of the spec (5.2, week-1 form): the CBTA-style auction has a lower average start time than the
  CBGA-style grouping auction on static dev instances.
- B3 tiers: the per-setting Heavy sub-solve counts, one sub-solve per event at Light.
"""
import numpy as np
import pytest

from cbba_sota.bench.configs import get
from cbba_sota.dyn import perturb
from cbba_sota.dyn import rh_kernels as K
from cbba_sota.dyn.baselines import cbta, cpsat_tiers
from cbba_sota.dyn.env import make_env
from cbba_sota.dyn.planner import PlanState, Problem, check_plan
from cbba_sota.hetero import Instance


def _checked(policy):
    errs: list[str] = []
    orig = policy.replan

    def replan(state, scope=None, event_index=None):
        plan = orig(state, scope, event_index)
        errs.extend(check_plan(state, plan, scope))
        return plan

    policy.replan = replan
    return errs


@pytest.mark.parametrize("rule,objective", [("cbta", "makespan"), ("cbta", "start"), ("seq", "start")])
@pytest.mark.parametrize("cell", ["F1-R2", "F3-pf0.2-R2"])
def test_auction_in_env_valid_and_complete(rule, objective, cell):
    setting, i = "MA-AT-25-5-50", 1
    rz = perturb.realize_instance(setting, "dev", i, 0, cell)
    env = make_env(get(setting).instance_path("dev", i), rz)
    ctl = cbta.make_controller(env, rz, "native", 0, rule=rule, objective=objective)
    errs = _checked(ctl.policy)
    ep = env.run_plan(ctl)
    assert ep.success, (rule, cell)
    assert not errs, errs[:3]
    assert ep.replans >= 2


def _starts(state, plan):
    prob = Problem(state)
    sol = prob.new_sol()
    xs = sorted((x for x in range(prob.Tk) if plan.members[prob.gid[x]]), key=lambda x: plan.keys[prob.gid[x]])
    prob.load(sol, xs, [tuple(plan.members[prob.gid[x]]) for x in range(prob.Tk)], 0.1)
    return sol[4][:prob.Tk].copy()


def test_cbta_beats_cbga_on_average_start():
    wins = 0
    for setting, i in [("MA-AT-25-5-50", 0), ("SA-AT-50-5-50", 1), ("SA-BT-50-5-50", 2)]:
        st = PlanState.static(Instance.from_pickle(get(setting).instance_path("dev", i)))
        a = cbta.AuctionRH("cbta", "start").replan(st)
        b = cbta.AuctionRH("cbga").replan(st)
        assert not check_plan(st, a) and not check_plan(st, b)
        wins += _starts(st, a).mean() < _starts(st, b).mean()
    assert wins == 3


def test_append_offer_is_earliest_start():
    inst = Instance.from_pickle(get("SA-BT-50-5-50").instance_path("dev", 0))
    st = PlanState.static(inst)
    prob = Problem(st)
    a = prob.arrays
    rng = np.random.default_rng(0)
    free = rng.uniform(0, 5, prob.A)
    last = np.where(rng.random(prob.A) < 0.5, rng.integers(0, prob.Tk, prob.A), -1).astype(np.int64)
    req, ab = a[K.I_REQ], a[K.I_AB]
    for x in range(0, prob.Tk, 7):
        m, s, _ = cbta.append_offer(prob, free, last, x)
        assert (ab[list(m)].sum(axis=0) >= req[x]).all()
        cand = a[K.I_CAPL][x, :a[K.I_NCAP][x]]
        leg = np.where(last[cand] >= 0, a[K.I_TT][np.maximum(last[cand], 0), x], a[K.I_DOUT][cand, x])
        arr = free[cand] + leg
        # single-skill robots, binary requirements: the earliest start is the latest of the per-trait earliest arrivals
        best = max(arr[ab[cand, k] > 0].min() for k in np.flatnonzero(req[x] > 0))
        assert s == best


def test_b3_tiers():
    for name, n in cpsat_tiers.HEAVY_SUBSOLVES.items():
        inst = Instance.from_pickle(get(name).instance_path("dev", 0))
        assert cpsat_tiers.setting_of(inst) == name
        assert cpsat_tiers.make_policy("heavy", inst=inst).cfg.max_subsolves == n
    setting, i = "SA-BT-50-5-50", 3
    rz = perturb.realize_instance(setting, "dev", i, 0, "F1-R2")
    env = make_env(get(setting).instance_path("dev", i), rz)
    ctl = cpsat_tiers.make_controller(env, rz, "light", 0)
    errs = _checked(ctl.policy)
    ep = env.run_plan(ctl)
    assert ep.success and not errs
    assert all(s["subsolves"] <= 1 for s in ctl.policy.stats)
    with pytest.raises(ValueError):
        cpsat_tiers.make_policy("native")
