"""Tests for the B1 online RL baseline (cbba_sota.dyn.baselines.rl_online)."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from cbba_sota.bench import configs
from cbba_sota.bench.heteromrta import load_env
from cbba_sota.dyn import perturb
from cbba_sota.dyn.baselines import rl_online as B1
from cbba_sota.solvers import rl

torch.set_num_threads(1)
SMALL = ("SA-BT-25-5-20", 0)
MID = ("MA-AT-25-5-50", 1)


@pytest.fixture(scope="module")
def net():
    return B1.load_policy("cpu")


def _path(setting, i):
    return configs.get(setting).instance_path("dev", i)


def _signature(env):
    return ([list(map(float, a["arrival_time"])) for a in env.agent_dic.values()],
            [list(map(int, a["route"])) for a in env.agent_dic.values()],
            [(float(t["time_start"]), float(t["time_finish"])) for t in env.task_dic.values()],
            float(env.current_time))


@pytest.mark.parametrize("setting,i", [SMALL, MID])
def test_static_reduction_bit_identical_to_official_loop(net, setting, i):
    """No release, no noise: run_online on the plain env and on the pilot world == rl.rollout (pinned loop)."""
    p = _path(setting, i)
    rz = perturb.realize_instance(setting, "dev", i, 0, "static")
    for sample, seed in ((False, 0), (True, 17)):
        e0 = load_env(p)
        r0 = rl.rollout(e0, net, sample=sample, seed=seed)
        e1 = load_env(p)
        r1 = B1.run_online(e1, net, sample=sample, seed=seed)
        e2 = B1.pilot_world(p, rz)
        r2 = B1.run_online(e2, net, sample=sample, seed=seed)
        assert _signature(e0) == _signature(e1) == _signature(e2)
        assert r0.makespan == r1.makespan == r2.makespan
        assert r0.success == r1.success == r2.success
        assert r1.n_decisions > 0 and r1.cpu_ms_p50 > 0


def _probs(net, t_obs, a_obs, m, idx):
    with torch.no_grad():
        p, _ = net(*(torch.tensor(x, dtype=torch.float32) for x in (t_obs, a_obs, m)), torch.tensor([[[idx]]]))
    return p


def test_zero_agent_row_equals_deleted_row_except_row0(net):
    """A zeroed agent row (the spec's failure notice) is an absent robot for the released network, except row 0,
    which the network's pooling cannot treat as padding (NaN); ``drop_rows`` is therefore what the driver uses."""
    env = load_env(_path(*MID))
    env.init_state()
    released, env.current_time = env.next_decision()
    aid = released[0][3]
    t_obs, a_obs, m = env.agent_observe(aid, False)
    gone = [7, aid + 1, 20]
    a_del, idx = B1.drop_rows(a_obs, gone, aid)
    assert idx == aid and a_del.shape[1] == a_obs.shape[1] - 3
    p_zero = _probs(net, t_obs, B1.zero_rows(a_obs, gone), m, aid)
    p_del = _probs(net, t_obs, a_del, m, idx)
    assert torch.max(torch.abs(p_zero - p_del)).item() < 1e-5
    assert int(torch.argmax(p_zero)) == int(torch.argmax(p_del))
    assert torch.isnan(_probs(net, t_obs, B1.zero_rows(a_obs, [0]), m, aid)).any()  # the documented pitfall
    a_del0, idx0 = B1.drop_rows(a_obs, [0, 1], aid)
    assert idx0 == aid - 2
    assert torch.isfinite(_probs(net, t_obs, a_del0, m, idx0)).all()
    with pytest.raises(ValueError):
        B1.drop_rows(a_obs, [aid], aid)


def test_oracle_observation_refused(net):
    rz = perturb.realize_instance(*SMALL[:1], "dev", SMALL[1], 0, "F1-R2")
    env = B1.pilot_world(_path(*SMALL), rz, obs="oracle")
    with pytest.raises(ValueError):
        B1.run_online(env, net, sample=False, seed=0)


def test_pilot_world_uses_crn_draws(net):
    setting, i = MID
    rz = perturb.realize_instance(setting, "dev", i, 1, "F2-R2N3")
    env, world = B1.make_world(setting, "dev", i, 1, "F2-R2N3", world="pilot")
    assert world.startswith("pilot:")
    np.testing.assert_array_equal(env.release, rz.release)
    np.testing.assert_array_equal(env.dur_real, rz.dur_real)
    assert all(np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1]) for a, b in zip(env.delays, rz.delays))
    assert env.sc.obs == "causal" and (rz.release > 0).any()
    r = B1.run_online(env, net, sample=False, seed=B1.policy_seed(rz.instance_key, 1))
    assert r.success
    # every finished task ran for its realized duration and started no earlier than its release
    for t in env.task_dic.values():
        assert t["finished"]
        assert t["time_finish"] - t["time_start"] == pytest.approx(rz.dur_real[t["ID"]], abs=1e-9)
        assert t["time_start"] >= rz.release[t["ID"]] - 1e-9
    with pytest.raises(PermissionError):
        B1.make_world(setting, "test", i, 0, "F1-R2")
    with pytest.raises(NotImplementedError):
        B1.make_world(setting, "dev", i, 0, "F3-pf0.2", world="pilot")


def test_failure_view_from_realization():
    class Rz:
        fail_onset = np.array([np.inf, 3.0, 10.0])
        detect_after = 0.5

    class E:
        realization = Rz()

    known, dead = B1.failure_view(E())
    assert known(3.4) == set() and known(3.5) == {1} and known(20.0) == {1, 2}
    assert dead(1, 3.0) and not dead(1, 2.9) and not dead(0, 100.0)

    class F:
        def known_failed(self, t):
            return [2] if t > 1 else []

        def is_failed(self, i, t):
            return i == 2 and t > 0.5

    known, dead = B1.failure_view(F())
    assert known(2.0) == {2} and dead(2, 0.6) and not dead(1, 5.0)


def test_failed_robot_is_masked_and_never_decides(net):
    """Env exposing failure notices: the notified robot's row is zero in every later observation and it never
    receives a decision; the policy input for everyone else is untouched."""
    fail_t, victim = 4.0, 2
    seen = []

    class FailEnv(type(load_env(_path(*SMALL)))):
        def known_failed(self, t):
            return [victim] if t >= fail_t else []

        def is_failed(self, i, t):
            return i == victim and t >= fail_t

    class Spy(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, tasks, agents, mask, index):
            decider = int(index.item())
            if agents.shape[1] < n_agents:  # victim removed: rows after it shift up by one
                decider += decider >= victim
            seen.append((float(env.current_time), decider, agents.shape[1]))
            return self.inner(tasks, agents, mask, index)

    env = load_env(_path(*SMALL))
    env.__class__ = FailEnv
    n_agents = len(env.agent_dic)
    r = B1.run_online(env, Spy(net), sample=False, seed=0, max_time=60.0)
    after = [s for s in seen if s[0] >= fail_t]
    before = [s for s in seen if s[0] < fail_t]
    assert after and before
    assert all(s[2] == n_agents - 1 for s in after) and all(s[1] != victim for s in after)
    assert all(s[2] == n_agents for s in before) and any(s[1] == victim for s in before)
    assert r.n_known_failed == 1
    # this fake env has no D6/D7 (a failed robot never comes home), so the driver must stop on the livelock
    assert r.stuck and not r.success


def test_policy_seed_is_crn_keyed():
    a = B1.policy_seed(500000, 0, "RL(s.1)")
    assert a == B1.policy_seed(500000, 0, "RL(s.1)")
    assert a != B1.policy_seed(500000, 1, "RL(s.1)") and a != B1.policy_seed(500001, 0, "RL(s.1)")
    assert a != B1.policy_seed(500000, 0, "RL(g.)")


def _first_failure_case():
    """(setting, instance, seed) of an F3-pf0.2 dev realization with at least one failure, not excluded."""
    for i in range(20):
        for sd in range(3):
            rz = perturb.realize_instance(SMALL[0], "dev", i, sd, "F3-pf0.2")
            if not rz.excluded and np.isfinite(rz.fail_onset).any():
                return SMALL[0], i, sd, rz
    pytest.skip("no failure realization found")


def test_dyntaskenvx_static_reduction_bit_identical(net):
    """B1 on DynTaskEnvX (the env's own run_policy loop, decision-order coalitions) == rl.rollout, static."""
    from cbba_sota.dyn.env import make_env

    p = _path(*MID)
    for sample, seed in ((False, 0), (True, 17)):
        e0 = load_env(p)
        r0 = rl.rollout(e0, net, sample=sample, seed=seed)
        e1 = make_env(p, None, coalition="decision")
        r1 = B1.run_b1(e1, net, sample=sample, seed=seed)
        assert _signature(e0) == _signature(e1)
        assert r0.makespan == r1.makespan and r1.world.startswith("DynTaskEnvX[decision,causal")
    with pytest.raises(TypeError):
        B1.run_online(make_env(p, None), net, sample=False, seed=0)


def test_dyntaskenvx_failures_are_notified(net):
    setting, i, sd, rz = _first_failure_case()
    env, label = B1.make_world(setting, "dev", i, sd, "F3-pf0.2")
    assert label is None
    r = B1.run_b1(env, net, sample=False, seed=B1.policy_seed(rz.instance_key, sd))
    detect = rz.fail_onset + rz.detect_after
    assert r.success and not r.stuck
    assert r.n_known_failed == int((detect <= r.makespan).sum()) >= 1
    assert r.extra["failures"] == int((rz.fail_onset <= r.makespan).sum())
    assert set(env.known_failed()) == set(np.flatnonzero(detect <= r.makespan).tolist())
