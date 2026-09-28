"""B2 RL-MPC (``cbba_sota.dyn.baselines.rl_mpc``): nominal state clone, lockstep rollouts, rollout -> plan.

- Static reduction: a clone of the time-0 static state rolls out bit-identically to ``cbba_sota.solvers.rl.rollout``
  (greedy and sampled), with the fast observation / update methods and with the native ones.
- The fast methods equal the native ones on mid-episode clones taken at the structural events of release and
  failure episodes (bit-identical rollouts).
- The clone is causal and consistent with the belief: unreleased tasks are hidden, known-failed robots never decide
  and their rows are deleted, robots physically committed to a task keep it first in the plan.
- In ``DynTaskEnvX``: episodes complete, are deterministic, and the static open-loop plan is never later than the
  greedy RL rollout it came from (pruning never delays).
"""
import pickle

import numpy as np
import pytest
import torch

from cbba_sota.bench.configs import get
from cbba_sota.bench.heteromrta import load_env
from cbba_sota.dyn import perturb
from cbba_sota.dyn.baselines import rl_mpc
from cbba_sota.dyn.env import make_env
from cbba_sota.dyn.planner import PlanState
from cbba_sota.solvers import rl

torch.set_num_threads(1)


@pytest.fixture(scope="module")
def net():
    return rl_mpc._net("cpu")


def _sig(env):
    return ([list(map(float, a["arrival_time"])) for a in env.agent_dic.values()],
            [list(map(int, a["route"])) for a in env.agent_dic.values()],
            [(float(t["time_start"]), float(t["time_finish"])) for t in env.task_dic.values()],
            float(env.current_time))


def _static_clone(setting: str, i: int):
    path = get(setting).instance_path("dev", i)
    rz = perturb.realize_instance(setting, "dev", i, 0, "static")
    env = make_env(path, rz)
    base = rl_mpc.nominal_base(env)
    env.init_state()
    env._mode = "plan"
    return path, rl_mpc.build_clone(base, PlanState.static(env.nominal_instance()), env.state(), 1.0)


@pytest.mark.parametrize("setting,i", [("MA-AT-25-5-50", 0), ("SA-BT-50-5-50", 1)])
def test_static_clone_equals_rl_rollout(net, setting, i):
    path, clone = _static_clone(setting, i)
    for fast in (True, False):
        clone.fast = fast
        blob = pickle.dumps(clone)
        for seed, sample in ((11, False), (12, True))[:2 if fast else 1]:  # one rollout: == rl.rollout
            r = rl_mpc.lockstep(blob, net, [(0, seed, sample)], device="cpu")[0]
            e0 = load_env(path)
            r0 = rl.rollout(e0, net, sample=sample, seed=seed)
            assert _sig(r.env) == _sig(e0), (fast, seed)
            assert r.makespan == r0.makespan and r.complete == r0.success
        if fast:
            seeds = [21, 22, 23]  # a sampled batch shares one generator: == rl.lockstep
            res = rl_mpc.lockstep(blob, net, [(k, s, True) for k, s in enumerate(seeds)], device="cpu")
            ref = rl.lockstep(path.read_bytes(), net, seeds, sample=True)
            assert [r.makespan for r in res] == [r.makespan for r in ref]


def _clones(setting, i, crn, cell, decisions=1):
    """Clones built at the structural events of a small-budget B2 episode (and the episode)."""
    got = []
    orig = rl_mpc.build_clone

    def spy(*a, **k):
        c = orig(*a, **k)
        got.append((a[1], a[2], pickle.dumps(c)))
        return c

    rl_mpc.build_clone = spy
    try:
        rz = perturb.realize_instance(setting, "dev", i, crn, cell)
        env = make_env(get(setting).instance_path("dev", i), rz)
        ctl = rl_mpc.make_controller(env, rz, "heavy", 0, decisions=decisions, batch=1, device="cpu")
        ep = env.run_plan(ctl)
    finally:
        rl_mpc.build_clone = orig
    return got, ep, ctl


@pytest.fixture(scope="module")
def f3_clones():
    return _clones("SA-BT-50-5-50", 4, 1, "F3-pf0.2-R2")


def test_fast_methods_equal_native_mid_episode(net, f3_clones):
    got, ep, _ = f3_clones
    assert ep.success and len(got) >= 5
    for k in np.linspace(1, len(got) - 1, 5).round().astype(int):
        c = pickle.loads(got[k][2])
        sigs = []
        for fast in (True, False):
            c.fast = fast
            res = rl_mpc.lockstep(pickle.dumps(c), net, [(0, 5, False), (1, 6, True)], device="cpu")
            sigs.append([_sig(r.env) for r in res])
        assert sigs[0] == sigs[1], k


def test_clone_is_causal_and_consistent(net, f3_clones):
    got, _, _ = f3_clones
    seen_hidden = seen_gone = seen_head = False
    for state, view, blob in got:
        c = pickle.loads(blob)
        assert c.current_time == view.t
        np.testing.assert_array_equal(c.hidden, ~view.released)
        gone = np.array([m == "failed" for m in view.mode])
        np.testing.assert_array_equal(c.gone, gone)
        seen_hidden |= bool(c.hidden.any())
        for j in np.flatnonzero(c.hidden):  # hidden tasks: zero rows, never selectable
            assert c.task_dic[j]["feasible_assignment"] and c.task_dic[j]["finished"]
        if gone.any():
            seen_gone = True
            res = rl_mpc.lockstep(blob, net, [(0, 3, False)], device="cpu")[0]
            assert all(len(res.env.agent_dic[i]["route"]) == 1 for i in np.flatnonzero(gone))  # never moved
            assert not any(i in t["members"] for i in np.flatnonzero(gone) for t in res.env.task_dic.values())
        for i in range(len(view.mode)):
            h = int(state.head[i])
            if h >= 0:  # physically committed: travelling to / waiting at h, a member there, arrival = belief
                seen_head = True
                a = c.agent_dic[i]
                assert a["current_task"] == h and i in c.task_dic[h]["members"]
                assert a["arrival_time"][-1] == max(float(state.ready[i]), view.t) or view.mode[i] == "wait"
        res = rl_mpc.lockstep(blob, net, [(0, 3, False)], device="cpu")[0]
        if res.complete:
            phys = [list(c.task_dic[j]["members"]) for j in range(len(view.released))]
            plan = rl_mpc.rollout_plan(res.env, state, phys)
            routes = plan.routes_for(len(view.mode))
            live = state.released & ~state.done & ~state.started
            assert all(bool(plan.members[j]) for j in np.flatnonzero(live))
            assert not any(plan.members[j] for j in np.flatnonzero(~live))
            for i in range(len(view.mode)):
                if state.head[i] >= 0:
                    assert routes[i][0] == state.head[i]
                if gone[i]:
                    assert not routes[i]
    assert seen_hidden and seen_gone and seen_head


def test_static_open_loop_never_later_than_rl_rollout(net):
    setting, i = "MA-AT-25-5-50", 3
    _, ep, ctl = _clones(setting, i, 0, "static")
    assert ep.success and ctl.policy.stats[0]["rollouts"] == 1
    seed = ctl.policy.decisions[0]["seed"]
    r0 = rl.rollout(load_env(get(setting).instance_path("dev", i)), net, sample=False,
                    seed=rl_mpc.sample_seed(seed, 0))
    assert ctl.policy.stats[0]["best_ms"] == r0.makespan  # the model rollout is the static greedy rollout
    assert ep.makespan <= r0.makespan + 1e-9


@pytest.mark.parametrize("cell,runs", [("F1-R2", 1), ("F3-pf0.2", 2)])
def test_episode_deterministic_and_complete(cell, runs):
    out = []
    for _ in range(runs):
        rz = perturb.realize_instance("MA-AT-25-5-50", "dev", 2, 0, cell)
        env = make_env(get("MA-AT-25-5-50").instance_path("dev", 2), rz)
        ctl = rl_mpc.make_controller(env, rz, "heavy", 0, decisions=40, batch=2, device="cpu")
        ep = env.run_plan(ctl)
        assert ep.success
        assert all(s["rollouts"] >= 1 for s in ctl.policy.stats)
        out.append((ep.makespan, [(s["rollouts"], s["decisions"], s["best_k"]) for s in ctl.policy.stats]))
    assert all(o == out[0] for o in out)


def test_scope_and_view_required():
    rz = perturb.realize_instance("MA-AT-25-5-50", "dev", 0, 0, "static")
    env = make_env(get("MA-AT-25-5-50").instance_path("dev", 0), rz)
    ctl = rl_mpc.make_controller(env, rz, "heavy", 0, decisions=1, batch=1, device="cpu")
    st = PlanState.static(env.nominal_instance())
    with pytest.raises(RuntimeError):
        ctl.policy.replan(st)  # no StateView handed over
    with pytest.raises(ValueError):
        rl_mpc.make_controller(env, rz, "light", 0)  # Heavy-tier method
