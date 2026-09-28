"""Gate K0 regressions (docs/trackD-spec.md Section 8, day 2): the executor matches the env, arrival order matches
decision order for minimal-cover plans, and the env/controller fixes found while establishing K0.

The campaign-scale check is ``scripts/trackD_k0.py`` (rows in ``runs/trackD/k0``); these tests pin the episodes
that exposed each fix, so a regression shows up as a named failure. Dev split only.
"""
from __future__ import annotations

import sys
from dataclasses import replace

import numpy as np
import pytest

from cbba_sota.bench.configs import ROOT, get
from cbba_sota.dyn import perturb
from cbba_sota.dyn.controller import PlanController
from cbba_sota.dyn.env import code_hash, make_env
from cbba_sota.dyn.executor import Executor, Realization
from cbba_sota.dyn.methods import PLANNERS, make_policy
from cbba_sota.hetero import Instance

sys.path.insert(0, str(ROOT / "scripts"))
import trackD_k0 as K

# (setting, dev instance, CRN seed, cell, planner): the episodes that exposed each K0 fix
EPISODES = [
    ("MA-AT-25-5-50", 0, 0, "F1-R2", "SPARC"),  # release epoch while every robot is busy (env check_finished)
    ("MA-AT-25-5-50", 0, 0, "F1-R1", "SPARC"),  # belief conventions: started / committed / freed at a finish
    ("MA-AT-25-5-50", 0, 0, "F3-pf0.2", "SPARC"),  # frozen partner fails before departing (lease = detector)
    ("MA-AT-25-5-50", 1, 0, "F3-pf0.2", "SPARC"),  # a partner that abandons keeps its later commitments
    ("SA-BT-50-5-50", 1, 0, "F3-pf0.2", "SPARC"),  # commitment frozen at departure, not at the next call
    ("SA-BT-50-5-50", 0, 0, "F3-pf0.2", "SPARC"),  # a halted traveller looks stalled until detected
    ("MA-AT-50-5-50", 0, 0, "F0-N3", "SPARC"),  # a zero-length trip home is not stalled
    ("SA-AT-50-5-50", 0, 0, "F3-pf0.2-R2", "SPARC"),  # undetected halted robot leaves; zero-length re-join
    ("MA-AT-50-5-50", 2, 1, "F3-pf0.1", "SPARC"),  # a route task that started without the robot is skipped
    ("MA-AT-50-5-50", 1, 0, "F3-pf0.1", "every-event"),  # no head lock on a task that already started
    ("MA-AT-50-5-50", 9, 1, "F3-pf0.2", "ins-only"),  # a wasted trip leaves the frozen coalition (then abort)
    ("MA-AT-25-5-50", 18, 1, "F3-pf0.2", "B4"),  # a commitment ends at the start; aborted task is open again
    ("SA-AT-50-5-50", 1, 0, "F1-R3", "B4"),
    ("MA-AT-50-5-50", 1, 1, "F2-R2N3", "every-event"),
    ("SA-BT-50-5-50", 2, 1, "F3-pf0.1", "B5"),
    ("MA-AT-25-5-50", 2, 1, "static", "ins-only"),
]


def _worlds(setting, i, seed, cell, method, coalition="arrival"):
    rz = perturb.realize_instance(setting, "dev", i, seed, cell)
    assert not rz.excluded
    path = get(setting).instance_path("dev", i)
    inst = Instance.from_pickle(path)
    pe, px = make_policy(method, "light"), make_policy(method, "light")
    nonmin = K._track_minimality(pe, inst)
    env = make_env(path, rz, coalition=coalition)
    ep = env.run_plan(PlanController(pe, inst, rz.kappa()))
    ex = Executor(inst, Realization.from_perturb(rz), detect_h=rz.detect_after)
    res = ex.run(px)
    return K.trajectory_env(env, ep, pe), K.trajectory_executor(ex, res, px), nonmin[0], (inst, rz, path)


@pytest.mark.parametrize(("setting", "i", "seed", "cell", "method"), EPISODES)
def test_executor_matches_env(setting, i, seed, cell, method):
    """K0: same re-plans and seeds (bit-identical beliefs), legs, task starts/finishes, makespan and counters."""
    te, tx, nonmin, (inst, rz, path) = _worlds(setting, i, seed, cell, method)
    cmp = K.compare(te, tx, 1e-6)
    assert cmp["match"], cmp["first"]
    assert cmp["exact"] and not cmp["counters_differ"], cmp
    assert te["success"]
    if nonmin == 0:  # D2: decision order == arrival order for minimal-cover plans
        pol = make_policy(method, "light")
        env = make_env(path, rz, coalition="decision")
        ep = env.run_plan(PlanController(pol, inst, rz.kappa()))
        cd = K.compare(te, K.trajectory_env(env, ep, pol), 1e-6)
        assert cd["exact"], cd["first"]


def test_residual_extras_make_decision_order_differ_without_crashing():
    """Residual repair keeps every frozen member, so a not-yet-departed member can become redundant once extras join
    (MA-AT-50-5-50 dev 1, F3-pf0.1: robot 41 at task 7). Arrival order: it makes a wasted trip; decision order: its
    join is refused (event ``refused``), the controller drops it and the episode completes."""
    setting, i, cell = "MA-AT-50-5-50", 1, "F3-pf0.1"
    te, tx, nonmin, (inst, rz, path) = _worlds(setting, i, 0, cell, "SPARC")
    assert nonmin > 0 and K.compare(te, tx, 1e-6)["exact"]
    pol = make_policy("SPARC", "light")
    env = make_env(path, rz, coalition="decision")
    seen = []
    ctl = PlanController(pol, inst, rz.kappa())
    orig = ctl.on_events

    def on(env, t, evs):
        seen.extend(e.kind for e in evs)
        return orig(env, t, evs)

    ctl.on_events = on
    ep = env.run_plan(ctl)
    assert ep.success and "refused" in seen
    assert not K.compare(te, K.trajectory_env(env, ep, pol), 1e-6)["match"]


# --- env fixes on tiny instances ----------------------------------------------------------------------------------------


def _tiny(req, locs, durs, ab, depots) -> Instance:
    return Instance(req=req, loc=locs, dur=durs, ab=ab, depot=depots, species=np.arange(len(ab)))


class Recorder:
    def __init__(self, act=None):
        self.calls, self.act = [], act or {}

    def on_events(self, env, t, evs):
        self.calls.append((t, [(e.kind, e.tasks, e.agents, e.info) for e in evs]))
        for when, fn in list(self.act.items()):
            if t >= when:
                fn(env)
                del self.act[when]
        return False


def test_release_is_an_epoch_while_every_robot_is_busy():
    """Before the fix, ``check_finished`` jumped the clock past a pending release epoch when no agent was due, so
    the release was first reported at the next agent epoch (6.0 here) instead of at 2.0."""
    inst = _tiny(req=[[1], [1]], locs=[[0.2, 0.0], [0.4, 0.0]], durs=[5.0, 1.0], ab=[[1]], depots=[[0.0, 0.0]])
    rz = replace(perturb.static_realization(inst.req, inst.dur, 1), release=np.array([0.0, 2.0]), H=2.0)
    rec = Recorder({2.0: lambda env: env.set_route(0, [1])})
    env = make_env(inst, rz)
    ep = env.run_plan(rec, routes=[[0]])
    rel = [t for t, evs in rec.calls if any(k == "release" and tasks == (1,) for k, tasks, _, _ in evs)]
    assert rel == [2.0]
    assert ep.success and env.legs[0][1][1] == pytest.approx(6.0)  # leaves for task 1 when task 0 finishes


def test_zero_length_trip_is_not_stalled():
    """A robot at its depot sent home does not move, so a stall on its calendar cannot delay it."""
    inst = _tiny(req=[[1, 0]], locs=[[0.2, 0.0]], durs=[1.0], ab=[[1, 0], [0, 1]], depots=[[0.0, 0.0], [0.5, 0.5]])
    rz = perturb.static_realization(inst.req, inst.dur, 2)
    rz = replace(rz, delays=[rz.delays[0], (np.array([0.0]), np.array([0.4]))])
    env = make_env(inst, rz)
    ep = env.run_plan(routes=[[0], []])
    assert env.agent_dic[1]["arrival_time"][-1] == 0.0
    assert ep.makespan == pytest.approx(3.0)  # robot 0: task 0 on [1, 2], home at 3


def test_leave_on_arrival_uses_the_real_arrival():
    """A released traveller leaves when it actually arrives (3.0, after a stall), not at the ETA observed when it
    was released (2.0), as a wasted trip; ``on=False`` takes the release back."""
    inst = _tiny(req=[[1], [1]], locs=[[0.4, 0.0], [0.4, 0.4]], durs=[1.0, 1.0], ab=[[1], [1]],
                 depots=[[0.0, 0.0], [0.4, 0.8]])
    rz = perturb.static_realization(inst.req, inst.dur, 2)
    rz = replace(rz, delays=[(np.array([1.5]), np.array([2.5])), rz.delays[1]])

    def release(env):
        assert env.leave_on_arrival(0)
        env.set_route(0, [1])
        env.set_route(1, [0])

    rec = Recorder({1.0: release})
    env = make_env(inst, rz)
    ep = env.run_plan(_Waker(rec, 1.0), routes=[[0], []])
    wasted = [(t, e) for t, evs in rec.calls for e in evs if e[0] == "wasted"]
    assert [t for t, _ in wasted] == [pytest.approx(3.0)] and dict(wasted[0][1][3])["reason"] == "released"
    assert ep.wasted_trips == 1 and ep.success
    assert env.legs[0][1][3] == 1 and env.legs[0][1][1] == pytest.approx(3.0)  # then serves task 1
    assert env.task_dic[0]["coalition"] == [1]

    def release_and_undo(env):
        assert env.leave_on_arrival(0) and env.leave_on_arrival(0, False)

    rec = Recorder({1.0: release_and_undo})
    env = make_env(inst, rz)
    ep = env.run_plan(_Waker(rec, 1.0), routes=[[0], [1]])
    assert ep.wasted_trips == 0 and env.task_dic[0]["time_start"] == pytest.approx(3.0)


class _Waker:
    """Asks to be woken at ``t`` (the first call happens at 0)."""

    def __init__(self, inner, t):
        self.inner, self.t = inner, t

    def on_events(self, env, t, evs):
        if t == 0:
            env.request_wakeup(self.t)
        return self.inner.on_events(env, t, evs)


def test_refused_join_is_reported_and_the_robot_moves_on():
    """Decision order: a route step onto a task already covered by the departed members is refused; the env reports
    ``refused`` and the robot takes its next route task at once."""
    inst = _tiny(req=[[1], [1]], locs=[[0.2, 0.0], [0.6, 0.0]], durs=[1.0, 1.0], ab=[[1], [1]],
                 depots=[[0.0, 0.0], [0.0, 0.2]])
    rec = Recorder()
    env = make_env(inst, coalition="decision")
    ep = env.run_plan(rec, routes=[[0], [0, 1]])
    assert any(e[0] == "refused" and e[2] == (1,) for _, evs in rec.calls for e in evs)
    assert env.legs[1][0][3] == 1 and env.legs[1][0][1] == 0.0 and ep.success


def test_undetected_halted_robot_is_believed_to_leave():
    """Lease = failure detector: at a partner's detection the present members leave. A present member that has
    halted but is not detected yet is believed alive, so it "leaves" too (it stays halted in place); keeping it in
    the cover would reveal its failure before detection (B1 observes the task's cover)."""
    inst = _tiny(req=[[3]], locs=[[0.2, 0.0]], durs=[1.0], ab=[[1], [1], [1]],
                 depots=[[0.2, 0.2], [0.2, 0.3], [0.2, 0.8]])
    rz = perturb.static_realization(inst.req, inst.dur, 3)
    fail = np.array([np.inf, 2.2, 2.0])  # robot 2 halts on its way (detected 2.5), robot 1 halts waiting (2.7)
    rz = replace(rz, fail_onset=fail)
    views = {}

    class Probe(Recorder):
        def on_events(self, env, t, evs):
            views[round(t, 6)] = env.state()
            return super().on_events(env, t, evs)

    rec = Probe()
    env = make_env(inst, rz)
    env.run_plan(rec, routes=[[0], [0], [0]])
    at = [(t, e) for t, evs in rec.calls for e in evs if e[0] == "failure"]
    assert [round(t, 6) for t, _ in at] == [2.5, 2.7]
    assert sorted(dict(at[0][1][3])["abandoned"]) == [0, 1]
    assert views[2.5].mode[1] == "idle" and 1 not in env.task_dic[0]["members"]


# --- integration through scripts/trackD_run.py --------------------------------------------------------------------------


def _case(method, cell, world="env", tier="light"):
    import trackD_run

    opts = {"coalition": "auto", "obs": "causal", "lease_L": 200.0, "abandon_on_failure": True}
    if world != "env":
        opts["world"] = world
    case = {"setting": "SA-BT-50-5-50", "split": "dev", "instance": 3, "seed": 1, "cell": cell,
            "family": perturb.cell(cell).family, "method": method, "tier": tier, "options": opts}
    case["case_id"] = trackD_run.case_id(case)
    return trackD_run.run_case(case, {"git": "", "git_dirty_dyn": None})


@pytest.mark.parametrize("method", ["SPARC", "B4"])
def test_runner_planners_env_and_executor(method):
    e, x = _case(method, "F3-pf0.2-R2"), _case(method, "F3-pf0.2-R2", "executor")
    assert e["status"] == x["status"] == "completed", (e.get("error"), x.get("error"))
    assert e["world"].startswith("DynTaskEnvX[arrival,") and e["world"].endswith(code_hash())
    assert x["world"].startswith("Executor[surrogate]@")
    assert e["makespan"] == x["makespan"] and e["replans"] == x["replans"]
    for k in ("wasted_trips", "abandons", "restarts", "failures"):
        assert e[k] == x[k], k
    assert set(PLANNERS) >= {method}


def test_runner_rl_static_equals_plain_rollout():
    """Check (iv) through the runner: RL(g.) in DynTaskEnvX on the static cell == ``rl.rollout`` on the plain env
    with the runner's method seed."""
    from cbba_sota.bench.heteromrta import load_env
    from cbba_sota.solvers import rl

    row = _case("RL(g.)", "static", tier="native")
    assert row["status"] == "completed" and row["world"].startswith("DynTaskEnvX[decision,")
    ref = rl.rollout(load_env(get("SA-BT-50-5-50").instance_path("dev", 3)), rl.load_policy("cpu"), sample=False,
                     seed=row["method_seed"])
    assert row["makespan"] == ref.makespan


def test_planner_needs_a_budget_tier():
    row = _case("SPARC", "F1-R2", tier="native")
    assert row["status"] == "error" and "tier" in row["error"]
