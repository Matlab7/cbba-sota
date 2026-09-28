"""DynTaskEnvX regression tests: the declared changes D1-D8 of docs/trackD-spec.md Section 3.2 and the static
reduction (RL, greedy and plan replay bit-identical to the static HeteroMRTA env). Dev split only."""
from __future__ import annotations

import contextlib
import io
import json
import os
import shutil
import subprocess
import sys
from dataclasses import replace
from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest

from cbba_sota.bench import configs
from cbba_sota.bench.configs import ROOT
from cbba_sota.bench.heteromrta import TaskEnv, load_env
from cbba_sota.dyn import perturb
from cbba_sota.dyn.env import (
    DynTaskEnvX,
    NearestPolicy,
    RLPolicy,
    make_env,
    predict_ready,
)
from cbba_sota.hetero import Instance, Plan, evaluate, random_plan
from cbba_sota.hetero.replay import env_from_instance
from cbba_sota.solvers import greedy, rl

PINNED = "4cf5e04"  # ALNS v2 commit pinned by the Track D spec (Section 8)
STATIC_CASES = [("MA-AT-25-5-50", 0), ("MA-AT-25-5-50", 1), ("MA-AT-50-5-50", 0), ("SA-AT-50-5-50", 0),
                ("SA-BT-50-5-50", 0)]


def dev_path(setting: str, i: int) -> Path:
    return configs.get(setting).instance_path("dev", i)


def signature(env) -> tuple:
    return ([list(map(float, a["arrival_time"])) for a in env.agent_dic.values()],
            [list(map(int, a["route"])) for a in env.agent_dic.values()],
            [(float(t["time_start"]), float(t["time_finish"])) for t in env.task_dic.values()],
            float(env.current_time))


def native_replay(source, routes_1based) -> TaskEnv:
    env = source if isinstance(source, TaskEnv) else load_env(source)
    env.init_state()
    for i, r in enumerate(routes_1based):
        env.pre_set_route(list(r), i)
    with contextlib.redirect_stdout(io.StringIO()):
        env.execute_by_route()
    return env


@pytest.fixture(scope="module")
def net():
    import torch

    torch.set_num_threads(1)
    return rl.load_policy()


# --- pinned ALNS plans (git archive of the pinned commit, run in a subprocess) ----------------------------------------


def _pinned_tree() -> Path:
    root = Path.home() / ".cache" / "cbba-sota" / f"pinned-{PINNED}"
    if not (root / "cbba_sota" / "solvers" / "alns.py").exists():
        tmp = root.parent / f".tmp-{PINNED}-{os.getpid()}"
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True)
        archive = subprocess.run(["git", "-C", str(ROOT), "archive", PINNED, "cbba_sota"], check=True,
                                 capture_output=True).stdout
        subprocess.run(["tar", "-x", "-C", str(tmp)], input=archive, check=True)
        (tmp / "third_party").symlink_to(ROOT / "third_party")
        try:
            tmp.rename(root)
        except OSError:  # another process extracted it first
            shutil.rmtree(tmp, ignore_errors=True)
    return root


_ALNS_CODE = """
import json, sys
import cbba_sota
from cbba_sota.hetero import Instance
from cbba_sota.solvers.alns import solve
out = {"package": cbba_sota.__file__, "plans": []}
for p in sys.argv[1:]:
    plan, st = solve(Instance.from_pickle(p), 120.0, seed=0, max_iters=150)
    out["plans"].append({"routes": plan.routes(), "makespan": st.makespan})
print(json.dumps(out))
"""


@pytest.fixture(scope="module")
def pinned_alns_plans():
    try:
        root = _pinned_tree()
    except (OSError, subprocess.CalledProcessError) as e:
        pytest.skip(f"cannot extract commit {PINNED}: {e}")
    paths = [str(dev_path(s, i)) for s, i in STATIC_CASES]
    env = {**os.environ, "OMP_NUM_THREADS": "1", "NUMBA_NUM_THREADS": "1", "PYTHONPATH": str(root)}
    res = subprocess.run([sys.executable, "-c", _ALNS_CODE, *paths], cwd=root, env=env, check=True,
                         capture_output=True, text=True, timeout=900)
    out = json.loads(res.stdout.strip().splitlines()[-1])
    assert out["package"].startswith(str(root)), out["package"]  # the pinned code ran, not the live package
    return dict(zip(paths, out["plans"]))


# --- D1: static reduction ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("setting,i", STATIC_CASES)
def test_static_reduction_rl(net, setting, i):
    """D1 (and D3, D4, D5, D7 no-ops): RL(g.) and RL(s.) rollouts are bit-identical to the static env, both through
    ``rl.rollout`` on the subclass and through ``run_policy`` (causal observations are the native ones without noise)."""
    p = dev_path(setting, i)
    for sample, seed in ((False, 0), (True, 11), (True, 12)):
        e0 = load_env(p)
        r0 = rl.rollout(e0, net, sample=sample, seed=seed)
        e1 = make_env(p)
        r1 = rl.rollout(e1, net, sample=sample, seed=seed)
        e2 = make_env(p)
        ep = e2.run_policy(RLPolicy(net, sample=sample, seed=seed), shuffle_seed=seed)
        assert signature(e1) == signature(e0) == signature(e2)
        assert r1.makespan == r0.makespan
        assert ep.success == r0.success and (ep.makespan == r0.makespan or not r0.success)
        assert "decision" in ep.world  # auto rule: decision order for policies


@pytest.mark.parametrize("setting,i", STATIC_CASES)
def test_static_reduction_greedy(setting, i):
    p = dev_path(setting, i)
    g0 = greedy.greedy_nearest(p)
    e0 = load_env(p)
    e0.init_state()
    greedy._run_nearest(e0, True, "wait")
    e1 = make_env(p)
    greedy._run_nearest(e1, True, "wait")
    e2 = make_env(p)
    e2.run_policy(NearestPolicy())
    assert signature(e1) == signature(e2)
    assert float(e2.current_time) == g0.makespan
    if g0.success:
        assert signature(e0) == signature(e1)
    else:
        # Past the time cap the native lease (200) drops a member waiting at an uncovered task, and the native
        # back-dating then departs it at that task's time_finish = 0 (an arrival before the cap). D4 back-dates only
        # agents that worked on their previous task, so only such post-cap legs may differ.
        s0, s1 = signature(e0), signature(e1)
        assert s0[1:] == s1[1:]
        for k, (a0, a1) in enumerate(zip(s0[0], s1[0])):
            if a0 != a1:
                assert a0[:-1] == a1[:-1] and e1.legs[k][-1][0] >= configs.MAX_TIME, k


def _plans(inst: Instance, rng: np.random.Generator) -> list[Plan]:
    return [greedy.construct(inst, restarts=4, seed=0), random_plan(inst, rng), random_plan(inst, rng, greedy=2.0)]


@pytest.mark.parametrize("setting,i", STATIC_CASES)
def test_static_reduction_plan_replay(setting, i, pinned_alns_plans):
    """D1 + D2: key-ordered minimal-cover plans (constructor, random, pinned ALNS v2) replay bit-identically in
    ``execute_by_route`` on the static env, on the subclass, and in ``run_plan`` under both coalition rules."""
    p = dev_path(setting, i)
    inst = Instance.from_pickle(p)
    plans = _plans(inst, np.random.default_rng(i))
    alns = pinned_alns_plans[str(p)]
    plans.append(Plan.from_routes(alns["routes"], inst.n_tasks))
    for k, plan in enumerate(plans):
        assert plan.is_minimal_cover(inst)
        ref = signature(native_replay(p, plan.to_env_routes()))
        assert signature(native_replay(make_env(p), plan.to_env_routes())) == ref
        for rule in ("arrival", "decision"):
            env = make_env(p, coalition=rule)
            ep = env.run_plan(routes=plan.routes())
            assert signature(env) == ref, (k, rule)
            assert ep.wasted_trips == 0 and ep.makespan == evaluate(inst, plan).makespan
            assert rule in ep.world
    assert evaluate(inst, plans[-1]).makespan == pytest.approx(alns["makespan"], abs=1e-9)


def test_zero_row_equals_deleted_row(net):
    """D1 masking: an unreleased task's all-zero row gives the policy output of deleting the row."""
    import torch

    p = dev_path("MA-AT-25-5-50", 2)
    rz = perturb.realize_instance("MA-AT-25-5-50", "dev", 2, 0, "F1-R1")
    env = make_env(p, rz)
    gaps, hidden = [], 0

    class Probe(RLPolicy):
        def act(self, env, agent_id):
            nonlocal hidden
            _, t_obs, a_obs, m = self._obs
            rel = np.concatenate([[True], env.released_mask()])
            idx = torch.tensor([[[agent_id]]])
            with torch.no_grad():
                full, _ = self.net(*(torch.tensor(x, dtype=torch.float32) for x in (t_obs, a_obs, m)), idx)
                comp, _ = self.net(*(torch.tensor(x, dtype=torch.float32) for x in (t_obs[:, rel], a_obs, m[:, rel])),
                                   idx)
            back = np.zeros(full.shape[1])
            back[rel] = comp[0].numpy()
            gaps.append(float(np.abs(full[0].numpy() - back).max()))
            hidden += int((~rel).any())
            assert int(np.argmax(full[0].numpy())) == int(np.argmax(back))
            return super().act(env, agent_id)

    env.run_policy(Probe(net, sample=False, seed=0), shuffle_seed=0)
    assert hidden > 10 and max(gaps) < 1e-5


# --- D2: arrival-order coalitions -------------------------------------------------------------------------------------


def _tiny(req, locs, durs, ab, depots) -> Instance:
    """Instance with one species per agent (traits and depot per agent)."""
    A = len(ab)
    return Instance(req=req, loc=locs, dur=durs, ab=ab, depot=depots, species=np.arange(A))


def test_arrival_order_first_cover_starts_and_late_member_leaves():
    # task 0 needs trait 0; agent 0 is 2.5 away, agent 1 (redundant, also trait 0) is 1.0 away
    inst = _tiny(req=[[1, 0]], locs=[[0.5, 0.0]], durs=[1.0], ab=[[1, 0], [1, 1]], depots=[[0.0, 0.0], [0.5, 0.2]])
    env = make_env(inst, coalition="arrival")
    ep = env.run_plan(routes=[[0], [0]])
    task = env.task_dic[0]
    assert task["time_start"] == pytest.approx(1.0) and task["coalition"] == [1]
    assert task["wasted"] == [0] and ep.wasted_trips == 1
    leg = env.legs[0]  # agent 0 leaves on arrival (2.5) and heads home
    assert leg[-1][3] < 0 and leg[-1][1] == pytest.approx(2.5)
    # decision order: agent 0 committed first and covers; agent 1 is refused and never travels
    env = make_env(inst, coalition="decision")
    ep = env.run_plan(routes=[[0], [0]])
    assert env.task_dic[0]["time_start"] == pytest.approx(2.5) and ep.wasted_trips == 0
    assert all(to < 0 for _, _, _, to in env.legs[1])


def test_arrival_order_present_redundant_member_stays():
    # needs (1, 1): agent 0 (1, 0) arrives at 1, agent 1 (1, 1) at 2 and alone would cover; agent 0 was there first
    inst = _tiny(req=[[1, 1]], locs=[[0.2, 0.0]], durs=[1.0], ab=[[1, 0], [1, 1]], depots=[[0.0, 0.0], [0.6, 0.0]])
    for rule in ("arrival", "decision"):
        env = make_env(inst, coalition=rule)
        env.run_plan(routes=[[0], [0]])
        assert env.task_dic[0]["time_start"] == pytest.approx(2.0)
        assert sorted(env.task_dic[0]["coalition"]) == [0, 1]


def test_rl_plans_replay_identically_under_arrival_order(net):
    """D2 regression with 15 RL rollouts: their coalitions pruned to minimal covers replay identically under arrival
    order and in the native (decision-order) env. (RL *policy* rollouts themselves are identical under arrival order
    only where RL coalitions are minimal, e.g. SA; see ``test_rl_policy_arrival_order_on_single_skill``.)"""
    n = 0
    for i in range(5):
        p = dev_path("MA-AT-25-5-50", i)
        inst = Instance.from_pickle(p)
        data = p.read_bytes()
        for sample, seed in ((False, 0), (True, 1), (True, 2)):
            r = rl.rollout(rl.loads_env(data), net, sample=sample, seed=seed)
            plan = rl.to_plan(r, inst.n_agents).prune_to_minimal(inst)
            ref = signature(native_replay(p, plan.to_env_routes()))
            env = make_env(p, coalition="arrival")
            env.run_plan(routes=plan.routes())
            assert signature(env) == ref
            n += 1
    assert n == 15


def test_rl_policy_arrival_order_on_single_skill(net):
    for setting in ("SA-BT-50-5-50", "SA-AT-50-5-50"):
        p = dev_path(setting, 1)
        e0 = load_env(p)
        rl.rollout(e0, net, sample=False, seed=0)
        env = make_env(p, coalition="arrival")
        env.run_policy(RLPolicy(net, sample=False, seed=0), shuffle_seed=0)
        assert signature(env) == signature(e0)


def test_arrival_equals_decision_under_noise():
    """Minimal-cover plans under travel delays and duration noise: both rules give the same episode."""
    p = dev_path("MA-AT-25-5-50", 3)
    inst = Instance.from_pickle(p)
    plan = greedy.construct(inst, restarts=4, seed=1)
    for cell in ("F0-N3", "S-N4"):
        rz = perturb.realize_instance("MA-AT-25-5-50", "dev", 3, 5, cell)
        sigs = []
        for rule in ("arrival", "decision"):
            env = make_env(p, rz, coalition=rule)
            ep = env.run_plan(routes=plan.routes())
            assert ep.success and ep.wasted_trips == 0
            sigs.append(signature(env))
        assert sigs[0] == sigs[1]


# --- D3: causal observations --------------------------------------------------------------------------------------


def _merge_stall(starts, ends, a, b):
    s = np.append(starts, a)
    e = np.append(ends, b)
    order = np.argsort(s)
    s, e = s[order], e[order]
    out_s, out_e = [s[0]], [e[0]]
    for x, y in zip(s[1:], e[1:]):
        if x <= out_e[-1]:
            out_e[-1] = max(out_e[-1], y)
        else:
            out_s.append(x)
            out_e.append(y)
    return np.array(out_s), np.array(out_e)


@pytest.mark.parametrize("obs", ["causal", "oracle"])
def test_causal_observations_do_not_leak_the_future(net, obs):
    """D3: change only what lies after tau (longer durations of tasks unfinished at tau, an extra stall after tau
    for every robot). Causal observations make every decision before tau identical; the native (oracle) rows leak
    realized future arrivals and finishes, so decisions before tau change."""
    setting, i = "MA-AT-25-5-50", 4
    p = dev_path(setting, i)
    rz = perturb.realize_instance(setting, "dev", i, 3, "S-N4")
    tau = 12.0
    log_a: list = []
    env = make_env(p, rz, obs=obs)
    env.run_policy(RLPolicy(net, sample=False, seed=0), shuffle_seed=0,
                   observer=lambda e, t, ev: log_a.append(t))
    finish_a = np.array([t["time_finish"] if t["feasible_assignment"] else np.inf for t in env.task_dic.values()])
    decisions_a = [(round(d, 9), i2, to) for i2, legs in enumerate(env.legs) for d, _, _, to in legs if d < tau]
    dur_b = np.where(finish_a > tau, rz.dur_real * 1.7, rz.dur_real)
    delays_b = [_merge_stall(s, e, tau + 0.05, tau + 3.0) for s, e in rz.delays]
    rz_b = replace(rz, dur_real=dur_b, delays=delays_b)
    env_b = make_env(p, rz_b, obs=obs)
    env_b.run_policy(RLPolicy(net, sample=False, seed=0), shuffle_seed=0)
    decisions_b = [(round(d, 9), i2, to) for i2, legs in enumerate(env_b.legs) for d, _, _, to in legs if d < tau]
    if obs == "causal":
        assert sorted(decisions_a) == sorted(decisions_b)
    else:
        assert sorted(decisions_a) != sorted(decisions_b)


# --- D4: departure back-dating cap ------------------------------------------------------------------------------------


def test_backdating_is_capped_at_the_information_epoch(net):
    """D4: under release no leg leaves before its task exists or before the latest release epoch known at the
    decision, and the cap binds (the native flashforward would have left before the release); trips home are not
    back-dated when H > 0 (D7). Without release the native flashforward is unchanged (static identity)."""
    setting, i = "SA-BT-50-5-50", 2
    p = dev_path(setting, i)
    binding = 0
    for cell in ("F1-R1", "F1-R2", "F1-R3"):
        rz = perturb.realize_instance(setting, "dev", i, 0, cell)
        env = make_env(p, rz)
        env.run_policy(RLPolicy(net, sample=False, seed=0), shuffle_seed=0)
        for a, legs in enumerate(env.legs):
            for k, (decided, dep, _, to) in enumerate(legs):
                assert dep <= decided + 1e-12
                if to < 0:
                    assert k == 0 or dep == decided  # home: at the decision (D7)
                    continue
                seen = rz.release[rz.release <= decided + 1e-12].max()
                assert dep >= rz.release[to] - 1e-12 and dep >= seen - 1e-12
                prev = legs[k - 1][3] if k else -1
                if prev >= 0 and a in env.task_dic[prev]["coalition"]:
                    f = env.task_dic[prev]["time_finish"]
                    binding += bool(f < seen - 1e-12 and f < decided - 1e-12)
    assert binding > 0
    env = make_env(p)
    env.run_policy(RLPolicy(net, sample=False, seed=0), shuffle_seed=0)
    assert any(dep < decided - 1e-12 for legs in env.legs for decided, dep, _, _ in legs)


# --- D5: causal lease ---------------------------------------------------------------------------------------------


def _native_loop(env: TaskEnv, routes_1based, rounds: int = 20):
    """``execute_by_route`` without its reset of ``max_waiting_time``, for a few rounds (the native loop spins
    forever once every agent is home while a task is uncovered)."""
    for i, r in enumerate(routes_1based):
        env.pre_set_route(list(r), i)
    for _ in range(rounds):
        if env.finished or env.current_time >= 200:
            break
        released, env.current_time = env.next_decision()
        for a in released[0] + released[1]:
            if not env.agent_dic[a]["pre_set_route"]:
                env.agent_step(a, 0, 0)
                env.agent_dic[a]["next_decision"] = np.nan
                continue
            env.agent_step(a, env.agent_dic[a]["pre_set_route"].pop(0), 0)
        env.finished = env.check_finished()


def test_lease_ignores_the_realized_arrival_spread():
    """D5: the native covered branch drops an early member at commit time from the realized arrival spread
    (non-causal); the causal lease keeps it, and the task starts when the partner arrives."""
    inst = _tiny(req=[[1, 1]], locs=[[0.2, 0.0]], durs=[1.0], ab=[[1, 0], [0, 1]], depots=[[0.0, 0.0], [2.2, 0.0]])
    native = env_from_instance(inst)
    native.init_state()
    native.max_waiting_time = 5.0
    _native_loop(native, [[1], [1]])
    assert 0 in native.task_dic[0]["abandoned_agent"]  # dropped at t = 0, before it even arrived
    env = make_env(inst, lease_L=5.0)
    ep = env.run_plan(routes=[[0], [0]])
    assert ep.success and ep.abandons == 0
    assert env.task_dic[0]["time_start"] == pytest.approx(10.0) and sorted(env.task_dic[0]["coalition"]) == [0, 1]


def test_lease_timeout_uses_own_waiting_time_only():
    # agent 0 waits alone at an uncovered task (nobody else is routed there) and leaves after L
    inst = _tiny(req=[[1, 1], [1, 0]], locs=[[0.2, 0.0], [0.4, 0.0]], durs=[1.0, 1.0], ab=[[1, 0], [0, 1]],
                 depots=[[0.0, 0.0], [5.0, 5.0]])
    env = make_env(inst, lease_L=5.0)
    seen = []

    class Rec:
        def on_events(self, env, t, evs):
            seen.extend(e for e in evs if e.kind == "abandon")
            return False

    env.run_plan(Rec(), routes=[[0, 1], []], max_time=30)
    assert len(seen) == 1 and seen[0].t == pytest.approx(1.0 + 5.0) and seen[0].agents == (0,)
    assert env.task_dic[1]["time_start"] == pytest.approx(6.0 + 1.0)  # continues with its route after leaving


# --- D6: failures -----------------------------------------------------------------------------------------------------


class Scripted:
    """Controller that records events and applies scripted routes at given times."""

    def __init__(self, script=None, wake=()):
        self.events, self.script, self.wake, self.rows = [], dict(script or {}), list(wake), {}

    def on_events(self, env, t, evs):
        if t == 0:
            for w in self.wake:
                env.request_wakeup(w)
        self.events += evs
        if any(e.kind == "wakeup" for e in evs):
            self.rows[t] = env.get_current_agent_status(env.agent_dic[1])
        for kind, routes in list(self.script.items()):
            if any(e.kind == kind for e in evs):
                for i, r in routes.items():
                    env.set_route(i, r)
                del self.script[kind]
        return bool(evs)


def _fail_realization(inst: Instance, onsets: dict[int, float], stalls: dict | None = None):
    rz = perturb.static_realization(inst.req, inst.dur, inst.n_agents)
    fail = np.full(inst.n_agents, np.inf)
    for i, t in onsets.items():
        fail[i] = t
    delays = list(rz.delays)
    for i, (a, b) in (stalls or {}).items():
        delays[i] = (np.array([a]), np.array([b]))
    return replace(rz, fail_onset=fail, delays=delays)


def test_failure_while_working_aborts_and_restarts_from_scratch():
    # agent 0 works on task 0 (duration 5) from t = 1; fails at 3; detected at 3.5; agent 1 is sent and restarts it
    inst = _tiny(req=[[1, 0]], locs=[[0.2, 0.0]], durs=[5.0], ab=[[1, 0], [1, 0]], depots=[[0.0, 0.0], [0.2, 0.4]])
    env = make_env(inst, _fail_realization(inst, {0: 3.0}))
    ctl = Scripted({"failure": {1: [0]}})
    ep = env.run_plan(ctl, routes=[[0], []])
    fails = [e for e in ctl.events if e.kind == "failure"]
    assert len(fails) == 1 and fails[0].t == pytest.approx(3.5) and fails[0].get("aborted")
    task = env.task_dic[0]
    assert task["restarts"] == 1 and task["coalition"] == [1]
    assert task["time_start"] == pytest.approx(3.5 + 2.0)  # agent 1 departs at detection, 2.0 away
    assert task["time_finish"] - task["time_start"] == pytest.approx(5.0)  # full duration again
    assert ep.success and ep.failures == 1 and ep.detected == 1
    assert ep.makespan == pytest.approx(env.agent_dic[1]["arrival_time"][-1])  # the failed robot is not awaited
    assert env.known_failed() == [0] and env.is_failed(0)


def test_failed_robot_looks_stalled_until_detected():
    """D6 + D3: between onset and detection a halted robot is observed exactly like a stalled one."""
    inst = _tiny(req=[[1, 1]], locs=[[0.6, 0.0]], durs=[1.0], ab=[[1, 0], [0, 1]], depots=[[0.0, 0.0], [0.6, 0.2]])
    rows = []
    for rz in (_fail_realization(inst, {0: 1.0}), _fail_realization(inst, {}, stalls={0: (1.0, 50.0)})):
        env = make_env(inst, rz)
        ctl = Scripted(wake=(1.2, 1.4))
        env.run_plan(ctl, routes=[[0], [0]], max_time=3.0)
        rows.append(ctl.rows)
    assert rows[0].keys() == rows[1].keys() == {1.2, 1.4}
    for t in (1.2, 1.4):
        np.testing.assert_array_equal(rows[0][t], rows[1][t])
    assert rows[0][1.4][0, 5] > rows[0][1.2][0, 5] - 0.2 + 1e-9  # its ETA slips with time


def test_failure_mid_leg_partner_abandons_and_is_released():
    inst = _tiny(req=[[1, 1]], locs=[[0.6, 0.0]], durs=[1.0], ab=[[1, 0], [0, 1]], depots=[[0.0, 0.0], [0.6, 0.2]])
    env = make_env(inst, _fail_realization(inst, {0: 1.0}))  # agent 0 (3.0 away) halts at 1.0; agent 1 arrives at 1.0
    ctl = Scripted()
    env.run_plan(ctl, routes=[[0], [0]], max_time=10.0)
    kinds = [(e.kind, round(e.t, 9)) for e in ctl.events if e.structural]
    assert ("failure", 1.5) in kinds and ("abandon", 1.5) in kinds
    assert env.agent_dic[0]["arrival_time"][-1] == np.inf
    assert env.legs[1][-1][0] == pytest.approx(1.5)  # agent 1 is released at detection (and heads home)
    assert not env.task_dic[0]["finished"]  # nobody left who can cover it: the episode cannot succeed


def test_failures_run_in_the_native_loop(net):
    """Onsets and detections are processed in next_decision, so a native-loop driver gets D6 too."""
    setting, i = "SA-BT-50-5-50", 0
    rz = perturb.realize_instance(setting, "dev", i, 0, "F3-pf0.2")
    assert np.isfinite(rz.fail_onset).sum() >= 3 and not rz.excluded
    env = make_env(dev_path(setting, i), rz)
    r = rl.rollout(env, net, sample=False, seed=0)
    assert r.success and env.stats["failures"] == np.isfinite(rz.fail_onset).sum()
    for t in env.task_dic.values():  # every coalition that finished was alive through its work
        for m in t["coalition"]:
            assert not (t["time_start"] < rz.fail_onset[m] < t["time_finish"])


# --- D7 / D8: termination and waiting in place -------------------------------------------------------------------------


def test_window_end_termination_and_wait_in_place(net):
    setting, i = "MA-AT-25-5-50", 5
    p = dev_path(setting, i)
    rz = perturb.realize_instance(setting, "dev", i, 0, "F1-R2")
    assert rz.H == pytest.approx(0.5 * 50.158)
    runs = {"rl": make_env(p, rz), "greedy": make_env(p, rz)}
    eps = {"rl": runs["rl"].run_policy(RLPolicy(net, sample=False, seed=0), shuffle_seed=0),
           "greedy": runs["greedy"].run_policy(NearestPolicy())}
    for name, env in runs.items():
        ep = eps[name]
        last_finish = max(t["time_finish"] for t in env.task_dic.values())
        for legs in env.legs:
            for _, dep, _, to in legs[:-1]:
                assert to >= 0 or dep < 1e-12, name  # no depot trip except the final one (D8)
            _, dep, _, to = legs[-1]
            assert to < 0 and dep >= env.H - 1e-9 and dep >= last_finish - 1e-9  # home only after H and all done
        assert ep.success and ep.makespan == pytest.approx(max(a["arrival_time"][-1] for a in env.agent_dic.values()))


def test_idle_agents_wait_until_the_window_ends():
    inst = _tiny(req=[[1, 0]], locs=[[0.2, 0.0]], durs=[1.0], ab=[[1, 0], [1, 0]], depots=[[0.0, 0.0], [0.0, 0.2]])
    rz = replace(perturb.static_realization(inst.req, inst.dur, 2), H=7.0)
    env = make_env(inst, rz)
    ctl = Scripted()
    ep = env.run_plan(ctl, routes=[[0], []])
    idle = [e for e in ctl.events if e.kind == "idle"]
    assert idle and idle[0].t == 0.0 and idle[0].agents == (1,)
    assert [e.t for e in ctl.events if e.kind == "window_end"] == [7.0]
    assert env.legs[1] == []  # agent 1 never leaves its depot
    assert env.legs[0][-1][3] < 0 and env.legs[0][-1][1] == pytest.approx(7.0)  # agent 0 waits at the task until H
    assert ep.success and ep.makespan == pytest.approx(7.0 + 1.0)


# --- controller API and state view -------------------------------------------------------------------------------------


def test_release_events_and_route_validation():
    setting, i = "MA-AT-25-5-50", 0
    rz = perturb.realize_instance(setting, "dev", i, 0, "F1-R1")
    env = make_env(dev_path(setting, i), rz)
    got = []

    class Ctl:
        def on_events(self, env, t, evs):
            for e in evs:
                if e.kind == "release":
                    got.append((t, e.tasks))
            if t == 0:
                with pytest.raises(ValueError):
                    env.set_route(0, [30])  # not released yet
                v = env.state()
                assert v.released.sum() == 21 and all(m == "home" for m in v.mode)
                ready, at, _ = predict_ready(v)
                assert (ready == 0).all() and (at == -1).all()
            return False

    env.run_plan(Ctl(), routes=None, max_time=25.0)
    assert [t for t, _ in got] == [0.0, 10.0, 20.0]
    assert got[0][1] == tuple(range(21)) and got[1][1] == tuple(range(21, 41)) and got[2][1] == tuple(range(41, 50))


def test_world_label_and_setup_checks():
    p = dev_path("SA-BT-50-5-50", 0)
    env = make_env(p)
    assert isinstance(env, DynTaskEnvX) and env.world.startswith("DynTaskEnvX[decision,causal")
    with pytest.raises(ValueError):
        make_env(p, coalition="first")
    rz = perturb.realize_instance("SA-BT-50-5-50", "dev", 0, 0, "F3-pf0.2")
    with pytest.raises(ValueError):
        make_env(p, rz, obs="oracle")
    with pytest.raises(PermissionError):
        perturb.realize_instance("SA-BT-50-5-50", "test", 0, 0, "static")


# --- invariants on every family (a test-only list-scheduling controller, RL(g.) and the greedy) ---------------------


class ListScheduler:
    """Test-only controller: at every structural event, list-schedule every uncovered released task (committed
    members kept, extra members by earliest predicted arrival) from ``predict_ready``. Not a Track D method."""

    def __init__(self, kappa: float = 1.0):
        self.kappa = kappa

    def on_events(self, env, t, evs):
        if not any(e.structural for e in evs):
            return False
        v, inst = env.state(), env.nominal_instance()
        ready, at, _ = predict_ready(v, self.kappa)
        A, T = len(v.mode), len(v.released)
        avail = ready.copy()
        pos = [inst.depot[i] if at[i] < 0 else inst.loc[at[i]] for i in range(A)]
        held = {i: v.target[i] for i in range(A) if v.mode[i] in ("wait", "travel") and not np.isfinite(ready[i])}
        todo = [j for j in range(T) if v.released[j] and not v.finished[j] and not v.started[j]
                and (v.residual[j] > 0).any()]
        routes: dict[int, list[int]] = {i: [] for i in range(A)}
        while todo:
            best = None
            for j in todo:
                arr = [v.eta[m] if v.eta[m] <= t else t + max(v.tau[m] - (t - v.dep[m] - v.stall[m]), 0) * self.kappa
                       for m in v.committed[j]]
                need = v.residual[j].copy()
                cand = sorted((avail[i] + np.linalg.norm(pos[i] - inst.loc[j]) / inst.speed, i) for i in range(A)
                              if v.alive[i] and np.isfinite(avail[i]) and i not in v.committed[j])
                extra = []
                for a_i, i in cand:
                    if (need <= 0).all():
                        break
                    if (np.minimum(inst.ab[i], need) > 0).any():
                        extra.append(i)
                        arr.append(a_i)
                        need = need - inst.ab[i]
                if (need <= 0).all():
                    s = max(arr) if arr else t
                    if best is None or s < best[0]:
                        best = (s, j, extra)
            if best is None:
                break
            s, j, extra = best
            for i in extra + [i for i, h in held.items() if h == j]:
                routes[i] += [j] if i in extra else []
                avail[i], pos[i] = s + inst.dur[j], inst.loc[j]
            held = {i: h for i, h in held.items() if h != j}
            todo.remove(j)
        changed = False
        for i in range(A):
            if v.alive[i]:
                changed |= env.set_route(i, routes[i])
        return changed


def check_invariants(env: DynTaskEnvX, ep) -> None:
    for j, t in env.task_dic.items():
        if not t["finished"]:
            continue
        ab = sum((env.agent_dic[m]["abilities"] for m in t["coalition"]), np.zeros(env.traits_dim))
        assert (ab >= t["requirements"]).all(), j
        assert t["time_start"] >= env.release[j] - 1e-9
        assert all(env.get_arrival_time(m, j) <= t["time_start"] + 1e-9 for m in t["coalition"])
        assert t["time_finish"] - t["time_start"] == pytest.approx(env.dur_real[j], abs=1e-9)
        assert all(not (t["time_start"] < env.fail_onset[m] < t["time_finish"]) for m in t["coalition"])
    for i, legs in enumerate(env.legs):
        last = {to: k for k, (_, _, _, to) in enumerate(legs)}
        for k, ((_, _, _, to0), (_, dep1, _, _)) in enumerate(pairwise(legs)):
            if to0 >= 0 and last[to0] == k and i in env.task_dic[to0]["coalition"]:
                assert dep1 >= env.task_dic[to0]["time_finish"] - 1e-9  # worked (last visit), then left
        for _, dep, _, to in legs[1:]:
            assert to >= 0 or env.H <= 0 or dep >= env.H - 1e-9  # D7/D8: home only after H
    if ep.success:
        assert ep.makespan >= max(t["time_finish"] for t in env.task_dic.values()) - 1e-9
        for i, a in env.agent_dic.items():
            if not env.dead[i]:
                assert a["route"][-1] < 0 and a["arrival_time"][-1] <= ep.makespan + 1e-9


@pytest.mark.parametrize("cell", ["F1-R2", "F2-R2N3", "F3-pf0.2", "F3-pf0.2-R2"])
def test_invariants_on_every_family(net, cell):
    setting, i = "MA-AT-25-5-50", 6
    rz = perturb.realize_instance(setting, "dev", i, 1, cell)
    p = dev_path(setting, i)
    env = make_env(p, rz)
    ep = env.run_plan(ListScheduler(rz.kappa()))
    check_invariants(env, ep)
    assert ep.success and ep.replans >= 1 and "arrival" in ep.world
    for pol in (RLPolicy(net, sample=False, seed=1), NearestPolicy()):
        env = make_env(p, rz)
        ep = env.run_policy(pol, shuffle_seed=1 if isinstance(pol, RLPolicy) else None)
        check_invariants(env, ep)
