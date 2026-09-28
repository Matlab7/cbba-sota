"""Tests for the C3 world (``cbba_sota.dyn.c3world``): the env extension (rally, per-replica D7, tick-polled idle
robots), rule 4.6(c), and end-to-end protocol episodes on the ground-truth env with the real planner (dev only)."""
from __future__ import annotations

import math

import numpy as np
import pytest

from cbba_sota.bench import configs
from cbba_sota.dyn import perturb
from cbba_sota.dyn.c3world import C3Config, C3Controller, C3Env, make_c3_env, run_c3
from cbba_sota.dyn.comm import CRNKey
from cbba_sota.dyn.env import make_env
from cbba_sota.dyn.replica import (
    ABORT,
    FINISH,
    RELEASE,
    START,
    WORK,
    KnowledgeBase,
    Layout,
    effective_kind,
    overdue_aborts,
)

SET = "MA-AT-25-5-50"


def _real(setting, i, cell, seed=0):
    return perturb.realize_instance(setting, "dev", i, seed, cell)


def _path(setting, i):
    return configs.get(setting).instance_path("dev", i)


# ------------------------------------------------------------------------------------------------ env extension
@pytest.mark.parametrize("cell", ["static", "F1-R2", "F3-pf0.2"])
def test_c3env_without_controller_hooks_equals_dyntaskenvx(cell):
    """Unused, the extension changes nothing: SPARC's good-comms controller gives bit-identical episodes."""
    from cbba_sota.dyn.controller import PlanController
    from cbba_sota.dyn.methods import make_policy

    real = _real(SET, 1, cell)
    sigs = []
    for maker in (make_env, make_c3_env):
        env = maker(_path(SET, 1), real)
        ctl = PlanController(make_policy("SPARC", "light", iters=30), env.nominal_instance(), real.kappa())
        ep = env.run_plan(ctl)
        sigs.append((env.signature(), ep.makespan, ep.wasted_trips))
    assert sigs[0] == sigs[1]


def test_rally_leg_is_stall_aware_and_the_next_leg_starts_where_it_stopped():
    real = _real(SET, 0, "F1-R2")
    env = make_c3_env(_path(SET, 0), real)
    env._mode = "plan"
    i = 0
    a = env.agent_dic[i]
    p0 = np.array(a["location"], float)
    target = np.array([0.5, 0.5])
    assert env.idle_here(i) and env.rally_start(i, target)
    assert env.away[i] and env.rallying[i]
    env.current_time = 1.0
    p1 = env.position(i, 1.0)
    moved = (1.0 - env._stalled(i, 0.0, 1.0)) * a["velocity"]
    assert np.linalg.norm(p1 - p0) == pytest.approx(min(moved, np.linalg.norm(target - p0)))
    d0 = a["travel_dist"]
    env.rally_stop(i)
    assert not env.rallying[i] and np.allclose(a["location"], p1)
    assert a["travel_dist"] - d0 == pytest.approx(np.linalg.norm(p1 - p0))
    j = 3
    env.release[:] = 0.0
    env.agent_step(i, j + 1, 0)  # departs from the rally stop, not from its depot
    assert a["leg"][1] == pytest.approx(np.linalg.norm(np.asarray(env.task_dic[j]["location"]) - p1) / a["velocity"])
    assert not env.away[i]


def test_c3_home_rule_is_the_mission_complete_signal_also_at_h0():
    """C3 mode: a robot may go home only when t >= H and every released task is finished, for H = 0 too (the native
    H = 0 rule sends a robot with an empty route home at once); the per-replica variant asks the callback."""
    env = make_c3_env(_path(SET, 0), _real(SET, 0, "F3-pf0.2"))  # H = 0
    env._ctx = 2
    assert env.may_go_home() is True  # not in C3 mode: DynTaskEnvX's native H = 0 rule
    env.c3 = True
    assert env.may_go_home() is False  # tasks unfinished
    for task in env.task_dic.values():
        task["finished"] = True
    assert env.may_go_home() is True
    real = _real(SET, 0, "F1-R2")
    env2 = make_c3_env(_path(SET, 0), real)
    env2.c3, env2._ctx = True, 2
    calls = []
    env2.home_ok = lambda i: calls.append(i) or False
    env2.current_time = real.H + 1.0
    assert env2.may_go_home() is False and calls == [2]
    env2.current_time = 0.0
    assert env2.may_go_home() is False and calls == [2]  # t < H: the callback is not even asked


# ------------------------------------------------------------------------------------------------ rule 4.6(c)
def test_overdue_work_of_a_silent_crew_counts_as_aborted():
    L = Layout(3, 2, 1)
    kb = KnowledgeBase(L)
    dur = np.array([2.0, 2.0])
    kb.task_event(0, RELEASE, 0.0, observers=[3])
    kb.task_event(1, RELEASE, 0.0, observers=[3])
    kb.heartbeat(0, 1.0, WORK, target=0, t_ref=3.0, force=True)
    kb.heartbeat(1, 1.0, WORK, target=1, t_ref=3.0, force=True)
    kb.task_event(0, START, 1.0, members=(0,), observers=[0])
    kb.task_event(1, START, 1.0, members=(1,), observers=[1])
    kb.sync()
    assert overdue_aborts(kb.belief(2, 12.9), dur, 10.0) == frozenset()  # start 1 + dur 2 + h_f 10 = 13
    assert overdue_aborts(kb.belief(2, 13.0), dur, 10.0) == {0, 1}
    kb.heartbeat(1, 12.5, WORK, target=1, t_ref=3.0, force=True)  # robot 1 is heard again: not silent
    kb.sync()
    assert overdue_aborts(kb.belief(2, 13.0), dur, 10.0) == {0}
    k = effective_kind(kb.belief(2, 13.0), dur, 10.0)
    assert k[0] == ABORT and k[1] == START
    kb.task_event(0, FINISH, 3.0, observers=[2])
    assert effective_kind(kb.belief(2, 13.0), dur, 10.0)[0] == FINISH


def test_lease_names_a_declared_failed_head_member_absent():
    """Rule 4.6(b) keeps a declared-failed robot on its committed head; the lease must still publish its absent
    record, otherwise planners keep sending robots to wait for it (livelock found in the dev pilot)."""
    from cbba_sota.dyn.replica import TRAVEL, WAIT, LeaseConfig, lease_check

    L = Layout(3, 2, 1)
    kb = KnowledgeBase(L)
    kb.heartbeat(0, 20.0, WAIT, target=1, force=True)
    kb.heartbeat(2, 1.0, TRAVEL, target=1, t_ref=3.0, force=True)  # silent since t = 1
    kb.sync()
    d = lease_check(kb.belief(0, 22.0), 0, 1, 20.0, cfg=LeaseConfig("patient", 1.0, 30.0), failed=frozenset({2}))
    assert d.abandon and d.reason == "no_partner" and d.absent == (2,)


# ------------------------------------------------------------------------------------------------ episodes
@pytest.fixture(scope="module")
def full_and_t2():
    out = {}
    for arm, rho in (("FULL", 1.0), ("SPARC", 1.0), ("CEN-F", 0.5)):
        out[arm] = run_c3(SET, "dev", 0, 0, "F1-R2", C3Config(arm=arm, rho=rho))
    return out


def test_episodes_complete_and_report_their_world(full_and_t2):
    for arm, r in full_and_t2.items():
        assert r["success"] and r["completion"] == 1.0, arm
        assert r["world"].startswith("C3Env<DynTaskEnvX[arrival,causal") and r["c3_hash"]
    f, s, c = full_and_t2["FULL"], full_and_t2["SPARC"], full_and_t2["CEN-F"]
    assert f["bytes"] == 0 and f["plan_calls_off_station"] == 0 and f["station_frac"] == 1.0
    assert s["bytes"] > 0 and s["plan_calls_off_station"] > 0  # stranded leaders plan
    assert c["plan_calls_off_station"] == 0 and c["rally_starts"] > 0  # CEN-F: only the station plans; rally
    assert c["station_frac"] < s["station_frac"] < 1.0  # rho 0.5 < rho 1


def test_c3_episode_is_deterministic_given_the_crn_key(full_and_t2):
    again = run_c3(SET, "dev", 0, 0, "F1-R2", C3Config(arm="SPARC", rho=1.0))
    ref = full_and_t2["SPARC"]
    for k in ("makespan", "wasted_trips", "abandons", "plan_calls", "versions", "bytes", "frames"):
        assert again[k] == ref[k], k


def test_full_matches_the_good_comms_controller_closely():
    """FULL (C3 world, ideal comms) vs SPARC's good-comms controller in DynTaskEnvX on the same realizations: the
    same planner and triggers, planned at the next comm tick instead of the event instant (dev pilot over 24 pairs:
    GM 0.997); per episode the chaotic planner makes them differ, so only a loose bound is asserted."""
    from cbba_sota.dyn.controller import run_env
    from cbba_sota.dyn.methods import make_policy
    from cbba_sota.hetero.instance import Instance

    for i, cell in ((3, "F1-R2"), (3, "F3-pf0.2")):
        r = run_c3(SET, "dev", i, 0, cell, C3Config(arm="FULL"))
        real = _real(SET, i, cell)
        ep, _ = run_env(make_policy("SPARC", "light"), Instance.from_pickle(_path(SET, i)), real)
        assert r["success"] and ep.success
        assert 0.85 < r["makespan"] / ep.makespan < 1.15


def test_lone_worker_failure_does_not_block_completion_under_t2():
    """Regression: SA-BT (single-trait tasks) with failures. A robot failing while working alone leaves no witness;
    rule 4.6(c) reopens its task (this episode used to run to the time cap)."""
    r = run_c3("SA-BT-50-5-50", "dev", 0, 0, "F3-pf0.2", C3Config(arm="SPARC", rho=1.0))
    assert r["n_failures"] > 0 and r["restarts"] > 0
    assert r["success"], r


def test_full_follows_a_version_that_releases_its_commitment():
    """Regression: the planner's G4 fallback releases a residual's frozen members; a robot that kept waiting at the
    released task under the detector lease cross-waited forever (FULL, SA-AT-50 dev 18 seed 0 F3, 30% completion).
    Following the adopted version (as controller.PlanController does) removes the deadlock."""
    r = run_c3("SA-AT-50-5-50", "dev", 18, 0, "F3-pf0.2", C3Config(arm="FULL"))
    assert r["success"] and r["releases"] > 0


def test_ideal_comms_replicas_are_global_whenever_the_planner_reads_them():
    real = _real(SET, 2, "F1-R2")
    env = make_c3_env(_path(SET, 2), real, coalition="arrival", abandon_on_failure=True)
    s = configs.get(SET)
    ctl = C3Controller(env, env.nominal_instance(), real, C3Config(arm="FULL"), CRNKey(s.index, s.seed("dev", 2), 0))
    base = ctl.runner.planner
    seen = []

    def planner(scope, belief):
        g = ctl.kb.global_view()
        seen.append(all(np.array_equal(ctl.kb.v[n], g) for n in np.flatnonzero(np.append(~env.dead, True))))
        return base(scope, belief)

    ctl.runner.planner = planner
    ep = env.run_plan(ctl, skip_finished=False)
    assert ep.success and len(seen) > 5 and all(seen)


def test_vectorized_kinds_equal_the_replica_rule():
    real = _real("SA-BT-50-5-50", 0, "F3-pf0.2")
    s = configs.get("SA-BT-50-5-50")
    env = make_c3_env(_path("SA-BT-50-5-50", 0), real, coalition="arrival", abandon_on_failure=False)
    ctl = C3Controller(env, env.nominal_instance(), real, C3Config(arm="CEN-F", rho=0.5),
                       CRNKey(s.index, s.seed("dev", 0), 0))
    checks = []
    orig = ctl._tick

    def tick(env_, k, t):
        orig(env_, k, t)
        if k % 50 == 0:
            vec = ctl.kinds(t)
            for n in range(ctl.L.N):
                checks.append(np.array_equal(vec[n], effective_kind(ctl.kb.belief(n, t), ctl.dur, 10.0)))

    ctl._tick = tick
    env.run_plan(ctl, max_time=120.0, skip_finished=False)
    assert len(checks) > 100 and all(checks)


def test_test_split_is_refused():
    with pytest.raises(PermissionError):
        run_c3(SET, "test", 0, 0, "F1-R2", C3Config(arm="FULL"))


def test_c3env_class_and_full_rho():
    env = make_c3_env(_path(SET, 0), _real(SET, 0, "F1-R2"))
    assert isinstance(env, C3Env) and "C3Env<" in env.world
    assert C3Config(arm="FULL").ideal and math.isinf(C3Config(arm="FULL", rho=0.5).ideal * math.inf)


# ------------------------------------------------------------------------------------------------ analysis rules
def _analysis():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts" / "trackD_c3_analyze.py"
    spec = importlib.util.spec_from_file_location("trackD_c3_analyze", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _rows(gm: dict) -> list[dict]:
    """Synthetic rows: gm[(arm, cell, rho)] = makespan, identical over 3 instances x 2 seeds."""
    out = []
    for (a, cell, rho), v in gm.items():
        for inst in range(3):
            for seed in range(2):
                out.append({"arm": a, "cell": cell, "rho": rho, "setting": "S", "inst": inst, "seed": seed,
                            "makespan_or_cap": v * (1 + 0.01 * inst), "case_id": f"{a}{cell}{rho}{inst}{seed}"})
    return out


def test_regret_ds_and_variant_rules():
    an = _analysis()
    base = {"CEN-F": {2.0: 100, 0.5: 200}, "HYB": {2.0: 104, 0.5: 150}, "REP-clamp": {2.0: 110, 0.5: 140},
            "INF-r": {2.0: 120, 0.5: 160}, "SPARC": {2.0: 101, 0.5: 141}, "SPARC-V3": {2.0: 103, 0.5: 141}}
    rows = _rows({(a, "F1-R2", r): v for a, d in base.items() for r, v in d.items()})
    cells = an.cells_of(rows)
    reg, mx = an.regret_table(rows, list(base), cells)
    assert reg["CEN-F"][("F1-R2", 2.0)] == pytest.approx(1.0)
    assert reg["REP-clamp"][("F1-R2", 0.5)] == pytest.approx(1.0)
    assert mx["SPARC"] == pytest.approx(101 / 100)  # max over cells: 1.01 at rho 2, 141/140 at rho 0.5
    rho_star, by = an.choose_ds(rows, cells)
    assert rho_star == 1.0 or rho_star == 2.0  # CEN-F at 2, REP-clamp at 0.5: regret 1 everywhere
    assert by[rho_star] == pytest.approx(1.0)
    ds = an.ds_rows(rows, rho_star)
    assert {r["rho"] for r in ds if r["makespan_or_cap"] in (100, 140)} == {2.0, 0.5}
    # tie rule: V2 within 1 point of the best is preferred
    assert an.choose_variant({"HYB": 1.05, "SPARC": 1.059, "SPARC-V3": 1.05}) == "V2"
    assert an.choose_variant({"HYB": 1.05, "SPARC": 1.07, "SPARC-V3": 1.04}) == "V1"
    assert an.choose_variant({"HYB": 1.08, "SPARC": 1.07, "SPARC-V3": 1.04}) == "V3"
