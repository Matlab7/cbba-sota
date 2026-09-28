"""Adversarial checks of the exact core, CP-SAT, the campaign scripts and the stored campaign rows.

Passing tests pin invariants that survived degenerate inputs (ties, zero durations, identical points, collinear
points) and regression tests of fixed verifier defects: one success definition at the 200 cap, CP-SAT keys on
zero-weight arcs, env-replayed pilot references, RL scored by its own rollout, instance fingerprints and code
versions in the rows and resume keys of every campaign script, CPU pinning with host conditions per row, and the
dev dry run spanning the prereg's settings with a tuning-contamination label.
"""
import argparse
import contextlib
import importlib.util
import io
import json
import multiprocessing as mp
import os
import re
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from itertools import pairwise

import numpy as np
import pytest

from cbba_sota.bench import runtime
from cbba_sota.bench.configs import HETEROMRTA_DIR, MAX_TIME, ROOT, RUNS_DIR
from cbba_sota.bench.heteromrta import dumps_env
from cbba_sota.hetero import (
    Instance,
    Plan,
    evaluate,
    random_plan,
    replay,
    replay_routes,
)
from cbba_sota.hetero.replay import env_from_instance, succeeded
from cbba_sota.solvers import cpsat, greedy, rl

REDTEAM = ROOT / "pilots" / "heteromrta" / "redteam"
RAL0 = HETEROMRTA_DIR / "RALTestSet" / "env_0.pkl"
SCRIPTS = ROOT / "scripts"


def _script(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def degenerate(rng: np.random.Generator, per: int, n_species: int, n_tasks: int, mode: int, k: int = 5) -> Instance:
    """Instances built to provoke ties: 0 = grid points (spacing 0.25) and integer durations including 0;
    1 = all tasks and depots on two points; 2 = everything on one point, half the durations 0;
    3 = random points, all durations 0."""
    while True:
        traits = rng.integers(0, 2, (n_species, k)).astype(float)
        if (traits.sum(1) > 0).all():
            break
    species = np.repeat(np.arange(n_species), per)
    req = np.zeros((n_tasks, k))
    for j in range(n_tasks):
        while not req[j].any():
            req[j] = np.minimum(rng.integers(0, 3, k), traits.sum(0) * per)
    if mode == 0:
        pts = rng.integers(0, 5, (n_tasks + n_species, 2)) * 0.25
        loc, dep, dur = pts[:n_tasks], pts[n_tasks:], rng.integers(0, 3, n_tasks).astype(float)
    elif mode == 1:
        two = np.array([[0.25, 0.5], [0.75, 0.5]])
        loc, dep = two[rng.integers(0, 2, n_tasks)], two[rng.integers(0, 2, n_species)]
        dur = rng.integers(0, 2, n_tasks) * 0.5
    elif mode == 2:
        loc, dep = np.full((n_tasks, 2), 0.5), np.full((n_species, 2), 0.5)
        dur = np.where(rng.random(n_tasks) < 0.5, 0.0, rng.random(n_tasks))
    else:
        loc, dep, dur = rng.random((n_tasks, 2)), rng.random((n_species, 2)), np.zeros(n_tasks)
    return Instance(req=req, loc=loc, dur=dur, ab=traits[species], depot=dep[species], species=species)


@pytest.mark.parametrize("mode", range(4))
def test_degenerate_minimal_covers_equal_env(mode):
    rng = np.random.default_rng(100 + mode)
    for per, n_species, n_tasks in ((1, 3, 6), (2, 4, 12), (3, 5, 20)):
        inst = degenerate(rng, per, n_species, n_tasks, mode)
        for greedy_level in (0.0, 2.0, 50.0):
            plan = random_plan(inst, rng, greedy=greedy_level)
            back = Plan.from_routes(plan.to_env_routes(), inst.n_tasks, one_based=True)
            assert back.members == plan.members and back.routes() == plan.routes()
            extra = [tuple(set(m) | set(rng.choice(inst.n_agents, 2, replace=False).tolist())) for m in plan.members]
            pruned = Plan(extra, plan.keys, plan.n_agents).prune_to_minimal(inst)
            for p in (plan, pruned):
                assert p.is_minimal_cover(inst)
                e, r = evaluate(inst, p), replay(inst, p)
                assert (e.makespan, e.success, 0) == (r["makespan"], r["success"], r["skipped"])
                assert abs(e.awt - r["awt"]) < 1e-9


def test_prune_on_collinear_zero_duration_points():
    """Triangle equalities hold exactly in the reals, so pruning gains nothing and float rounding may lose an ulp."""
    rng = np.random.default_rng(7)
    species = np.repeat(np.arange(3), 3)
    for it in range(150):
        x = rng.random(15)
        pts = np.stack([x, 0.3 * x + 0.1], 1)
        req = np.eye(3)[rng.integers(0, 3, 12)] + rng.integers(0, 2, (12, 3))
        inst = Instance(req=req, loc=pts[:12], dur=np.zeros(12), ab=np.eye(3)[species], depot=pts[12:][species],
                        species=species)
        base = random_plan(inst, rng, greedy=5.0 * (it % 2))
        extra = [tuple(set(m) | set(rng.choice(9, 2, replace=False).tolist())) for m in base.members]
        plan = Plan(extra, base.keys, 9)
        pruned = plan.prune_to_minimal(inst)
        assert pruned.is_minimal_cover(inst)
        assert evaluate(inst, pruned).makespan <= evaluate(inst, plan).makespan + 1e-9
        if it % 15 == 0:
            assert replay(inst, pruned)["makespan"] == evaluate(inst, pruned).makespan


def _one_task(dur: float) -> tuple[Instance, Plan]:
    """One agent, one task at distance sqrt(2) (7.07 time units each way) from the depot (5 traits, as the RL
    policy expects)."""
    inst = Instance(req=[[1, 0, 0, 0, 0]], loc=[[1.0, 1.0]], dur=[dur], ab=[[1, 0, 0, 0, 0]], depot=[[0.0, 0.0]],
                    species=[0])
    return inst, Plan([(0,)], [0.0], 1)


def test_success_agrees_with_env_below_cap():
    inst, plan = _one_task(180.0)
    e, r = evaluate(inst, plan), replay(inst, plan)
    assert e.makespan == r["makespan"] < 200 and e.success and r["success"]


def test_success_agrees_with_env_just_past_cap():
    inst, plan = _one_task(190.0)  # task ends at 197.07 < 200, return at 204.14
    e, r = evaluate(inst, plan), replay(inst, plan)
    assert e.makespan == r["makespan"] > 200
    assert e.success == r["success"] is False and r["env_finished"]


@pytest.mark.parametrize("dur, ok", [(185.0, True), (190.0, False), (199.0, False)])
def test_one_success_definition_near_cap(dur, ok):
    """Returns at 199.14 (success), 204.14 (task ends before the cap) and 213.14 (task ends after it): the env
    finishes all three runs, and every scorer agrees on success = all tasks finished and makespan < 200."""
    inst, plan = _one_task(dur)
    want = dur + 2 * np.sqrt(2) / 0.2
    e, r = evaluate(inst, plan), replay(inst, plan)
    env = env_from_instance(inst)
    ms, rl_ok = rl.replay(dumps_env(env), [[0]])
    g, gr = greedy.greedy_nearest(env), greedy.greedy_repo(env)
    for makespan, success in ((e.makespan, e.success), (r["makespan"], r["success"]), (ms, rl_ok),
                              (g.makespan, g.success), (gr.makespan, gr.success)):
        assert makespan == pytest.approx(want, abs=1e-9) and success is ok
    assert r["env_finished"] and g.env_finished and gr.env_finished
    assert succeeded(True, 199.99) and not succeeded(True, MAX_TIME) and not succeeded(False, 10.0)
    assert not succeeded(True, None)


def test_rl_result_uses_one_success_definition():
    """``rl._result`` of an env that finished all tasks after the cap: success False, env_finished True."""
    inst, _ = _one_task(190.0)
    env = env_from_instance(inst)
    env.init_state()
    env.pre_set_route([1], 0)
    with contextlib.redirect_stdout(io.StringIO()):
        env.execute_by_route()
    res = rl._result(env, MAX_TIME)
    assert res.makespan > MAX_TIME and res.completion == 1.0 and res.env_finished and not res.success


# CP-SAT on a degenerate instance: every node on one point and task 2 takes no time, so dur_2 + tt_2k = 0; without the
# one-unit floor on arc weights the CP starts tie.
_TIE = {"req": [[1, 2, 0, 2, 1], [0, 0, 0, 0, 1], [1, 0, 0, 2, 1], [1, 2, 0, 1, 0]], "loc": [[0.5, 0.5]] * 4,
        "dur": [0.5636784723769704, 0.6304285021208718, 0.0, 0.4744735614328901],
        "ab": [[1, 1, 0, 1, 0], [0, 1, 0, 1, 0], [0, 1, 0, 0, 1]], "depot": [[0.5, 0.5]] * 3, "species": [0, 1, 2]}


def test_cpsat_plan_on_zero_weight_arcs_is_env_feasible():
    inst = Instance(**_TIE)
    res = cpsat.solve_full(inst, 5.0, workers=1, symmetry=False)
    r = replay(inst, res.plan)
    assert res.plan.is_minimal_cover(inst) and r["success"] and r["skipped"] == 0
    assert r["makespan"] == evaluate(inst, res.plan).makespan


def test_cpsat_plan_not_worse_than_objective_on_zero_weight_arcs():
    """Task-to-task weights are floored at one unit, so CP starts (the plan keys) strictly increase along every
    circuit and the plan keeps the circuit order (before the floor: 1.669 vs objective 1.195 here)."""
    inst = Instance(**_TIE)
    assert cpsat.Weights.of(inst).w.min() >= 1
    res = cpsat.solve_full(inst, 5.0, workers=1, symmetry=False)
    assert res.status == "OPTIMAL"
    assert evaluate(inst, res.plan).makespan <= res.objective + 1e-6
    lns = cpsat.solve_lns(inst, 1.0, greedy.construct(inst, restarts=2), workers=1)
    for plan in (res.plan, lns.plan):
        assert all(plan.keys[a] < plan.keys[b] for r in plan.routes() for a, b in pairwise(r))


@pytest.mark.parametrize("mode", [0, 2])
def test_cpsat_keys_follow_circuits_on_degenerate_instances(mode):
    rng = np.random.default_rng(40 + mode)
    for _ in range(4):
        inst = degenerate(rng, 1, 3, 5, mode)
        res = cpsat.solve_full(inst, 5.0, workers=1, symmetry=False)
        if res.status == "OPTIMAL":
            assert evaluate(inst, res.plan).makespan <= res.objective + 1e-6
        assert all(res.plan.keys[a] < res.plan.keys[b] for r in res.plan.routes() for a, b in pairwise(r))


def _ctasd(i: int) -> tuple[float, list[list[int]]]:
    """timeCost / 100 and per-agent 1-based routes of the shipped CTAS-D solution (vehicle type = species + 1)."""
    txt = (HETEROMRTA_DIR / "RALTestSet" / f"env_{i}" / "results.yaml").read_text()
    vehicles = re.findall(r"  vv(\d+):\n    id: \d+\n    type: (\d+)\n(?:.*\n)*?    node: \[([^\]]*)\]", txt)
    species = Instance.from_pickle(HETEROMRTA_DIR / "RALTestSet" / f"env_{i}.pkl").species.tolist()
    free = {s: [a for a, sa in enumerate(species) if sa == s] for s in set(species)}
    routes: list[list[int]] = [[] for _ in species]
    for _, typ, nodes in sorted(vehicles, key=lambda v: int(v[0])):
        routes[free[int(typ) - 1].pop(0)] = [int(n) for n in nodes.split(",") if int(n) != 0]
    return float(re.search(r"timeCost: ([\d.]+)", txt).group(1)) / 100, routes


def test_ctasd_reference_equals_env_replay():
    """alns_report uses timeCost / 100 of the shipped CTAS-D runs; the solutions replay to exactly that."""
    for i in range(0, 50, 5):
        cost, routes = _ctasd(i)
        r = replay_routes(HETEROMRTA_DIR / "RALTestSet" / f"env_{i}.pkl", routes)
        assert r["success"] and r["skipped"] == 0 and abs(r["makespan"] - cost) < 1e-5


def test_pilot_lns_references_are_env_replays():
    """alns_report replays the stored pilot routes; the pilot's own values (evaluator of non-minimal plans) differ
    on 29/30 plans, so they must not be used."""
    path = REDTEAM / "scale50_lns_13s.json"
    if not path.exists():
        pytest.skip("pilot files missing")
    sys.path.insert(0, str(SCRIPTS))
    from run_alns import instances_from_json

    refs = _script("alns_report")._scale50_refs()
    insts = instances_from_json(REDTEAM / "instances_scale50.json")
    differs = 0
    for key, (value, routes) in json.loads(path.read_text()).items():
        name = "ScaleSet50/" + key.split("/")[-1].removesuffix(".pkl")
        assert refs[name]["pilot LNS 13s"] == replay_routes(insts[name], routes)["makespan"], name
        differs += abs(refs[name]["pilot LNS 13s"] - value) > 1e-9
    assert differs > 0


def test_pilot_cpsat_references_are_env_replays():
    rows = json.loads((REDTEAM / "cpsat_30s_8w.json").read_text())
    refs = _script("alns_report")._ral_refs()
    for name, value, *_, routes in rows[:10]:
        i = int(name.split("env_")[1].removesuffix(".pkl"))
        rep = replay_routes(HETEROMRTA_DIR / "RALTestSet" / f"env_{i}.pkl", routes)
        assert refs[f"RALTestSet/test/{i}"]["CP-SAT 30s x8"] == rep["makespan"] == pytest.approx(value, abs=1e-9)


def test_rl_references_are_rollout_scores(tmp_path, monkeypatch):
    """RL enters comparisons with its rollout's own makespan and success, never ``replay_makespan``."""
    report = _script("alns_report")
    monkeypatch.setattr(report, "RUNS_DIR", tmp_path)
    (tmp_path / "rl" / "X").mkdir(parents=True)
    rl_row = {"setting": "X", "method": "RL(s.64)", "workers": 16, "success": True, "replay_success": True}
    rows = [rl_row | {"instance": 0, "makespan": 13.0, "replay_makespan": 11.0},  # superseded by the rerun below
            rl_row | {"instance": 0, "makespan": 10.0, "replay_makespan": 11.0},
            rl_row | {"instance": 1, "makespan": 201.0, "replay_makespan": 12.0},
            rl_row | {"instance": 1, "method": "construct", "makespan": 5.0}]  # not an RL row
    (tmp_path / "rl" / "X" / "dev.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    refs = report._rl_ref_refs()
    assert refs == {"X/dev/0": {"RL(s.64) 16cpu": 10.0}, "X/dev/1": {"RL(s.64) 16cpu": MAX_TIME}}
    assert report._pilot_rl({"g": float("inf"), "s10": 30.0}) == {"RL(g.)": MAX_TIME, "RL(s.10)": 30.0}
    cd = _script("compare_dev")
    assert cd._score({"makespan": 204.0, "success": True}) == cd.FAIL
    assert cd._score({"makespan": 199.0, "success": True}) == 199.0
    spec = importlib.util.spec_from_file_location("analyze", RUNS_DIR / "baselines" / "analyze.py")
    analyze = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(analyze)
    assert analyze.score({"makespan": 204.0, "success": True}) == MAX_TIME
    assert analyze.score({"makespan": 199.0, "success": True}) == 199.0


def test_stored_rows_match_current_instances():
    """Every stored row of every setting and split (first 2 instances) belongs to the current instance: a stored
    fingerprint must match it, and rows without one must re-score exactly (scripts/check_rows.py; a negative control
    on the superseded kb = 3 MA-AT-9-3-20 rows is below)."""
    cr = _script("check_rows")
    _, groups = cr.scan(cr.jsonl_files([RUNS_DIR]))
    stale, checked = [], 0
    for key, rows in groups.items():
        if key[0] != "bench" or key[3] >= 2:
            continue
        for f, n, status, reason, _ in cr.check_group(key, rows, rescore=False):
            checked += status == "ok"
            if status == "stale":
                stale.append((f, n + 1, reason))
    if not checked:
        pytest.skip("no stored rows")
    assert not stale, f"{len(stale)} rows disagree with the current instances, e.g. {stale[:4]}"


def test_check_rows_flags_superseded_rows():
    """Negative control: the baseline rows computed on the kb = 3 MA-AT-9-3-20 instances before their regeneration."""
    cr = _script("check_rows")
    path = RUNS_DIR / "baselines" / "_superseded" / "MA-AT-9-3-20" / "dev.jsonl"
    if not path.exists():
        pytest.skip("no superseded rows")
    _, groups = cr.scan([path])
    results = [r for key, rows in groups.items() if key[3] < 3 for r in cr.check_group(key, rows)]
    assert results and all(status == "stale" for _, _, status, _, _ in results)


# --- provenance, resume keys and CPU pinning ------------------------------------------------------------------


def test_fingerprint_is_the_instance_data(tmp_path):
    inst = Instance.from_pickle(RAL0)
    fp = runtime.fingerprint(inst)
    inst.save(tmp_path / "x.npz")
    assert runtime.fingerprint(Instance.load(tmp_path / "x.npz")) == fp
    assert runtime.fingerprint(Instance.from_env(env_from_instance(inst))) == fp
    assert runtime.instance_fingerprint("RALTestSet", "test", 0) == fp and len(fp) == 40
    dur = inst.dur.copy()
    dur[3] += 1e-12
    other = Instance(req=inst.req, loc=inst.loc, dur=dur, ab=inst.ab, depot=inst.depot, species=inst.species)
    assert runtime.fingerprint(other) != fp


def test_code_version():
    v = runtime.code_version()
    assert v == "unknown" or re.fullmatch(r"[0-9a-f]{40}|([0-9a-f]{40}|nocommit)-dirty-[0-9a-f]{12}", v)


def test_timed_campaigns_refuse_a_dirty_tree(monkeypatch):
    dirty, clean = "0" * 40 + "-dirty-" + "1" * 12, "0" * 40
    monkeypatch.setattr(runtime, "code_version", lambda: dirty)
    with pytest.raises(SystemExit, match="dirty"):
        runtime.require_clean()
    assert runtime.require_clean(allow_dirty=True) == dirty
    monkeypatch.setattr(runtime, "code_version", lambda: clean)
    assert runtime.require_clean() == clean


def _rows_with_fingerprints(fp: str) -> list[dict]:
    base = {"setting": "RALTestSet", "split": "test", "instance": 0, "budget_s": 1.0, "budget": "B1"}
    return [base | {"method": "current", "fingerprint": fp}, base | {"method": "stale", "fingerprint": "0" * 40},
            base | {"method": "none"}]


def test_resume_keys_require_the_current_fingerprint(tmp_path):
    rows = _rows_with_fingerprints(runtime.instance_fingerprint("RALTestSet", "test", 0))
    path = tmp_path / "rows.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows) + '{"torn')
    assert _script("run_baselines").done_keys(path) == {(0, "current", 1.0)}
    assert _script("run_rl").done_keys(path) == {(0, "current")}
    assert _script("compare_dev")._done(tmp_path) == {("RALTestSet", 0, "B1", "current")}


_GIT = re.compile(r"[0-9a-f]{40}(-dirty-[0-9a-f]{12})?|nocommit-dirty-[0-9a-f]{12}|unknown")


def _stale_row(**fields) -> str:
    return json.dumps(fields | {"fingerprint": "0" * 40}) + "\n"


def _run_script(*args: str) -> str:
    env = os.environ | dict.fromkeys(("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                                      "NUMBA_NUM_THREADS"), "1")
    return subprocess.run([sys.executable, str(SCRIPTS / args[0]), *args[1:]], check=True, capture_output=True,
                          text=True, env=env).stdout


def _new_rows(path, n_old: int = 1) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) > n_old and all(r["fingerprint"] == "0" * 40 for r in rows[:n_old])
    return rows[n_old:]


def test_baseline_and_rl_campaigns_write_provenance_and_recompute_stale_rows(tmp_path):
    """run_baselines and run_rl store the instance fingerprint and code version in every row; a rerun skips current
    rows but recomputes a row whose fingerprint is not the current instance's."""
    fp = runtime.instance_fingerprint("RALTestSet", "test", 0)
    key = {"setting": "RALTestSet", "split": "test", "instance": 0}
    base = tmp_path / "baselines" / "RALTestSet" / "test.jsonl"
    base.parent.mkdir(parents=True)
    base.write_text(_stale_row(**key, method="construct", budget_s=0.2))
    args = ("run_baselines.py", "--settings", "RALTestSet", "--split", "test", "--methods", "construct", "--budgets",
            "0.2", "--instances", "1", "--workers", "1", "--cap", "1", "--out", str(tmp_path / "baselines"),
            "--allow-dirty")
    assert _run_script(*args).startswith("1 jobs") and _run_script(*args).startswith("0 jobs")
    (row,) = _new_rows(base)
    assert row["fingerprint"] == fp and _GIT.fullmatch(row["git"]) and row["success"]
    assert {"affinity", "load1", "load1_end"} <= row.keys() and len(runtime.parse_cpus(row["affinity"])) == 1

    rl_path = tmp_path / "rl" / "RALTestSet" / "test.jsonl"
    rl_path.parent.mkdir(parents=True)
    rl_path.write_text(_stale_row(**key, method="RL(g.)"))
    args = ("run_rl.py", "--settings", "RALTestSet", "--split", "test", "--methods", "g", "--instances", "1",
            "--procs", "1", "--workers", "1", "--out", str(tmp_path / "rl"))
    assert _run_script(*args).startswith("1 jobs") and _run_script(*args).startswith("0 jobs")
    (row,) = _new_rows(rl_path)
    assert row["fingerprint"] == fp and _GIT.fullmatch(row["git"])
    assert row["success"] == succeeded(True, row["makespan"]) and "replay_makespan" in row


def test_alns_campaign_writes_provenance_and_recomputes_stale_rows(tmp_path, monkeypatch, capsys):
    sys.path.insert(0, str(SCRIPTS))
    import run_alns

    from cbba_sota.bench import configs

    monkeypatch.setattr(configs, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(configs, "ROOT", tmp_path)
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
        monkeypatch.setenv(var, "1")
    out = tmp_path / "runs" / "alns" / "t" / "SA-BT-9-3-20-dev.jsonl"
    out.parent.mkdir(parents=True)
    out.write_text(_stale_row(setting="SA-BT-9-3-20", split="dev", index=0, name="SA-BT-9-3-20/dev/0", makespan=1.0,
                              eval_makespan=1.0, success=True, skipped=0, it_per_s=1.0, init_makespan=1.0))
    argv = ["run_alns.py", "--settings", "SA-BT-9-3-20", "--split", "dev", "--n", "2", "--time", "0.2", "--procs", "2",
            "--tag", "t", "--allow-dirty"]
    for want in ("2 jobs", "0 jobs"):
        monkeypatch.setattr(sys, "argv", argv)
        run_alns.main()
        assert capsys.readouterr().out.startswith(want)
    rows = _new_rows(out)
    assert sorted(r["index"] for r in rows) == [0, 1]
    for r in rows:
        assert r["fingerprint"] == runtime.instance_fingerprint("SA-BT-9-3-20", "dev", r["index"])
        assert _GIT.fullmatch(r["git"]) and {"host", "affinity", "load1", "load1_end"} <= r.keys()
        assert len(runtime.parse_cpus(r["affinity"])) == 1 and r["routes_base"] == 1  # pinned, 1-based env routes
        assert r["makespan"] == r["eval_makespan"] and r["success"]


def test_compare_dev_covers_the_prereg_settings_and_labels_dev(tmp_path, capsys):
    """The dry run spans the 8 H1 settings of docs/prereg-phase1.md and labels dev ratios as tuning-contaminated."""
    cd = _script("compare_dev")
    assert cd.SETTINGS == ("SA-BT-25-5-50", "SA-BT-50-5-50", "SA-AT-50-5-50", "MA-AT-25-5-50", "MA-AT-50-5-50",
                           "MA-AT-50-5-200", "MA-AT-150-10-500", "MA-AT-150-5-500")
    # every competitor runs on the same cores as its ALNS variant; the RL + ALNS hybrid is an ablation only
    assert all(cd.CORES[c] == cd.CORES[ours] and c != cd.HYBRID for ours, pool in cd.COMPETITORS.items() for c in pool)
    cd.report(argparse.Namespace(out=tmp_path, split="dev", csv=None))
    out = capsys.readouterr().out
    assert "tuning-contaminated" in out and "0 of 8" in out


def test_reports_drop_stale_rows(tmp_path):
    """Reports drop rows whose fingerprint is not the current one; rows without one (never verified) stay."""
    fp = runtime.instance_fingerprint("RALTestSet", "test", 0)
    assert [runtime.is_stale(r) for r in _rows_with_fingerprints(fp)] == [False, True, False]
    base = {"setting": "RALTestSet", "split": "test", "fingerprint": fp}
    rows = [base | {"index": 0, "makespan": 1.0}, base | {"index": 0, "makespan": 2.0},  # a rerun: the last counts
            base | {"index": 1, "makespan": 3.0}, base | {"index": 2, "makespan": 4.0}]
    del rows[-1]["fingerprint"]
    path = tmp_path / "rows.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    kept = _script("alns_report")._rows(path)
    assert sorted((r["index"], r["makespan"]) for r in kept) == [(0, 2.0), (2, 4.0)]


def test_pinned_job_rows_carry_provenance():
    """A run_baselines job pins its process to its CPUs and records affinity, load, throttling and fingerprint
    (in a spawned worker, so the test process keeps its affinity)."""
    sys.path.insert(0, str(SCRIPTS))
    import run_baselines as rb

    cpu = min(runtime.choose_cpus(1, sample_s=0.1))
    job = rb.Job("RALTestSet", "test", 0, "construct", 0.1, 1, (cpu,))
    with ProcessPoolExecutor(1, mp_context=mp.get_context("spawn")) as ex:
        row = ex.submit(rb._run, job).result()
    assert row["affinity"] == str(cpu) and row["fingerprint"] == runtime.instance_fingerprint("RALTestSet", "test", 0)
    assert {"load1", "load1_end"} <= row.keys() and row["success"] and row["makespan"] == row["eval_makespan"]
    if runtime.cgroup_cpu_stat():
        assert row["nr_throttled"] >= 0 and row["throttled_usec"] >= 0


def test_pin_covers_threads_and_children():
    cpus = runtime.choose_cpus(2, sample_s=0.1)
    assert len({runtime._core(c) for c in cpus}) == 2
    with ProcessPoolExecutor(1, mp_context=mp.get_context("spawn"), initializer=runtime.pin,
                             initargs=(cpus,)) as ex:
        assert ex.submit(runtime.affinity).result() == runtime.format_cpus(cpus)
    assert runtime.parse_cpus(runtime.format_cpus([0, 1, 2, 5, 7, 8])) == [0, 1, 2, 5, 7, 8]
    assert runtime.format_cpus([3, 1, 2, 9]) == "1-3,9" and runtime.blocks(list(range(10)), 4) == [[0, 1, 2, 3],
                                                                                                    [4, 5, 6, 7]]
