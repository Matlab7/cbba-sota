"""C1-dyn pilot helpers (cbba_sota/dyn/c1dyn.py): the B4 calibration probe, budget fit, labels and gate rules."""
from __future__ import annotations

import numpy as np
import pytest

from cbba_sota.dyn import c1dyn
from cbba_sota.dyn.stats import GMRatio, gm_ratio, pair_rows


def _g(ratio: float, lo: float | None = None, hi: float | None = None) -> GMRatio:
    lo = ratio - 0.01 if lo is None else lo
    hi = ratio + 0.01 if hi is None else hi
    return GMRatio(ratio, lo, hi, ratio, ratio, 0.01, 0.02, 20, 40, 0.05, 10, 5, 5, 20, 10)


def test_probe_leaves_the_sparc_episode_identical():
    """ProbeSPARC times B4 on every belief it plans, but adopts only SPARC's plans: same episode as plain SPARC."""
    from cbba_sota.bench import configs
    from cbba_sota.dyn import perturb
    from cbba_sota.dyn.controller import PlanController
    from cbba_sota.dyn.env import make_env
    from cbba_sota.dyn.methods import make_controller

    setting, i, cell = "MA-AT-50-5-50", 0, "F1-R1"
    s = configs.get(setting)
    rz = perturb.realize_instance(setting, "dev", i, 0, cell)
    e1 = make_env(s.instance_path("dev", i), rz)
    ep1 = e1.run_plan(make_controller(e1, rz, "light", 0, method="SPARC"))
    e2 = make_env(s.instance_path("dev", i), rz)
    pol = c1dyn.ProbeSPARC(iters=300, probe_restarts=(0, 2))
    ep2 = e2.run_plan(PlanController(pol, e2.nominal_instance(), rz.kappa()))
    assert e1.signature() == e2.signature()
    assert ep1.makespan == ep2.makespan and ep1.replans == ep2.replans
    assert len(pol.probes) == ep2.replans
    for p in pol.probes:
        assert set(p["b4_s"]) == {0, 2} and p["sparc_s"] > 0 and p["problem_s"] > 0
    assert pol.probes[0]["first"] and not any(p["first"] for p in pol.probes[1:])


def test_restarts_equal_cpu_fit():
    rng = np.random.default_rng(0)
    n = 200
    sparc = np.full(n, 0.010)
    b4 = {r: 0.001 + 0.0005 * r + rng.normal(0, 1e-6, n) for r in (0, 8, 24)}
    fit = c1dyn.restarts_equal_cpu(sparc, b4)
    assert fit["r_star"] == pytest.approx(18.0, abs=0.05)
    fair = c1dyn.restarts_equal_cpu(sparc, b4, overhead_s=np.full(n, 0.00025))  # half of each restart not charged
    assert fair["r_star"] == pytest.approx(36.0, abs=0.1)
    with pytest.raises(ValueError):
        c1dyn.restarts_equal_cpu(sparc, {0: b4[0]})


def test_labels_and_b4_levels():
    budget = {"settings": {"MA-AT-50-5-50": {"restarts": 44, "restarts_impl": 12}}}
    b4 = c1dyn.b4_levels(budget)
    assert b4["MA-AT-50-5-50"] == {"B4-eq": 44, "B4-2eq": 88, "B4-4eq": 176, "B4-impl": 12}
    row = {"setting": "MA-AT-50-5-50", "tier": "light", "options": {"coalition": "auto"}}
    assert c1dyn.labels({**row, "method": "SPARC"}, b4) == ["SPARC-L"]
    assert c1dyn.labels({**row, "method": "SPARC", "tier": "heavy"}, b4) == ["SPARC-H"]
    assert c1dyn.labels({**row, "method": "B4?restarts=44"}, b4) == ["B4-r44", "B4-eq"]
    assert c1dyn.labels({**row, "method": "B4?restarts=12"}, b4) == ["B4-r12", "B4-impl"]
    assert c1dyn.labels({**row, "method": "B4?restarts=7"}, b4) == ["B4-r7"]
    assert c1dyn.labels({**row, "method": "B4?restarts=44&restarts0=440"}, b4) == ["B4-r44+t0", "B4-eq+t0"]
    assert c1dyn.labels({**row, "method": "SPARC?iters0=3000"}, b4) == ["SPARC-L+t0"]
    assert c1dyn.labels({**row, "method": "RL(g.)", "tier": "native"}, b4) == ["RL(g.)"]
    arr = {**row, "method": "greedy", "tier": "native", "options": {"coalition": "arrival"}}
    assert c1dyn.labels(arr, b4) == ["greedy-arr"]
    assert c1dyn.labels({**row, "method": "B5"}, b4) == ["B5"]
    assert c1dyn.labels({**row, "method": "B5", "tier": "heavy"}, b4) == ["B5-H"]


def test_gate_rules():
    per = {"a": _g(0.95), "b": _g(0.96), "c": _g(0.98), "d": _g(1.01)}
    assert c1dyn.k1_decision(_g(0.965, 0.95, 0.98), per).passed
    assert not c1dyn.k1_decision(_g(0.9723, 0.9679, 0.9767), per).passed  # pooled above 0.97
    assert not c1dyn.k1_decision(_g(0.96, 0.92, 1.001), per).passed  # CI upper bound not < 1
    per2 = {"a": _g(0.95), "b": _g(1.02), "c": _g(1.01), "d": _g(0.9)}
    assert not c1dyn.k1_decision(_g(0.96, 0.95, 0.97), per2).passed  # < 1 on only 2 of 4 settings
    assert c1dyn.k2_decision(_g(0.99), _g(0.975)).passed  # both conditions: no dynamic separation
    assert not c1dyn.k2_decision(_g(0.99), _g(0.96)).passed
    assert not c1dyn.k2_decision(_g(0.85), _g(0.99)).passed


def test_family_balanced_weights_families_equally():
    rows = []
    fam_of = {"F1-a": "F1", "F1-b": "F1", "F1-c": "F1", "F2-a": "F2"}
    for inst in range(4):
        for cell, (va, vb) in {"F1-a": (1.0, 2.0), "F1-b": (1.0, 2.0), "F1-c": (1.0, 2.0),
                               "F2-a": (2.0, 1.0)}.items():
            for m, v in (("A", va), ("B", vb)):
                rows.append({"method": m, "setting": "S", "instance": inst, "cell": cell, "seed": 0,
                             "makespan": v * (1 + inst / 10), "success": True})
    d = pair_rows(rows, "A", "B", pair_keys=c1dyn.PAIR_KEYS, cluster_keys=c1dyn.CLUSTER_KEYS)
    pooled = gm_ratio(d["log_r"], d["clusters"], reps=200)
    assert pooled.ratio == pytest.approx(np.exp((3 * np.log(0.5) + np.log(2.0)) / 4))
    v, cl = c1dyn.family_balanced(d, fam_of)
    bal = gm_ratio(v, cl, reps=200)
    assert bal.ratio == pytest.approx(1.0)  # F1 (ratio 0.5) and F2 (ratio 2) weigh the same
    assert len(set(cl)) == 4


def test_episode_summary():
    rows = [{"success": True, "makespan": 10.0, "replans": 3, "cpu_ms_event": [1.0, 3.0], "cpu_ms_p50": 2.0,
             "cpu_ms_p95": 2.9, "cpu_method_s": 0.01},
            {"success": False, "makespan": None, "replans": 5, "cpu_ms_event": [5.0], "cpu_ms_p50": 5.0,
             "cpu_ms_p95": 5.0, "cpu_method_s": 0.02}]
    m = c1dyn.episode_summary(rows)
    assert m["n"] == 2 and m["success"] == 1 and m["mean_makespan"] == 10.0
    assert m["replans"] == 4.0 and m["cpu_p50"] == 3.0 and m["cpu_ep_p95_max"] == 5.0


@pytest.mark.parametrize("name,params", [("SPARC", {}), ("B4", {"restarts": 3})])
def test_hybrid_of_a_planner_with_itself_is_that_planner(name, params):
    from cbba_sota.bench import configs
    from cbba_sota.dyn import perturb
    from cbba_sota.dyn.env import make_env
    from cbba_sota.dyn.methods import make_controller

    setting, i, cell = "SA-BT-50-5-50", 2, "F3-pf0.2-R2"
    s = configs.get(setting)
    rz = perturb.realize_instance(setting, "dev", i, 1, cell)
    e1 = make_env(s.instance_path("dev", i), rz)
    ep1 = e1.run_plan(make_controller(e1, rz, "light", 0, method=name, **params))
    e2 = make_env(s.instance_path("dev", i), rz)
    ep2 = e2.run_plan(c1dyn.hybrid(e2, rz, "light", 0, first=name, then=name, **params))
    assert e1.signature() == e2.signature() and ep1.makespan == ep2.makespan and ep1.replans == ep2.replans


def test_hybrid_labels():
    b4 = {"S": {"B4-eq": 44, "B4-2eq": 88, "B4-4eq": 176, "B4-impl": 12}}
    row = {"setting": "S", "tier": "light", "options": {"coalition": "auto"}}
    m = "ctrl:cbba_sota.dyn.c1dyn:hybrid?first=SPARC&then=B4&restarts=44"
    assert c1dyn.labels({**row, "method": m}, b4) == ["SPARC-L>B4-eq"]
    m = "ctrl:cbba_sota.dyn.c1dyn:hybrid?first=B4&then=SPARC&restarts=44"
    assert c1dyn.labels({**row, "method": m}, b4) == ["B4-eq>SPARC-L"]
