"""Classical baselines: env greedies, plan constructors and CP-SAT (every plan checked against the env replay)."""
import contextlib
import io

import numpy as np
import pytest

from cbba_sota.bench.configs import HETEROMRTA_DIR, Setting
from cbba_sota.bench.heteromrta import generate_env, save_env
from cbba_sota.hetero import Instance, Plan, evaluate, make_env, random_plan, replay
from cbba_sota.solvers import cpsat, greedy

RAL = sorted(HETEROMRTA_DIR.glob("RALTestSet/env_*.pkl"), key=lambda p: int(p.stem.split("_")[1]))
FAMILIES = [("SA-BT", 9, 3, 20, 3), ("SA-AT", 15, 5, 20, 5), ("MA-AT", 25, 5, 50, 5), ("MA-AT", 50, 5, 50, 5)]
CP_WORKERS = 4


@pytest.fixture(scope="module")
def instances(tmp_path_factory) -> list[Instance]:
    root = tmp_path_factory.mktemp("baselines")
    out = []
    for k, (family, A, S, T, K) in enumerate(FAMILIES):
        path = root / f"{family}-{A}-{S}-{T}.pkl"
        save_env(generate_env(Setting(0, family, 0, A, S, T, K, {}), seed=7_000_000 + k), path)
        out.append(Instance.from_pickle(path))
    return out


def _line_instance() -> Instance:
    """One agent at the origin, three single-trait tasks at x = 0.5, 0.1, 0.3."""
    return Instance(req=[[1], [1], [1]], loc=[[0.5, 0], [0.1, 0], [0.3, 0]], dur=[1, 1, 1], ab=[[1]],
                    depot=[[0, 0]], species=[0])


# --- env greedies ----------------------------------------------------------------------------------------------


def test_greedy_repo_reproduces_pilot_mean():
    res = [greedy.greedy_repo(p) for p in RAL]
    assert all(r.success for r in res)
    assert np.mean([r.makespan for r in res]) == pytest.approx(32.42, abs=0.005)  # pilots/README.md


def test_loop_with_last_index_rule_is_the_repo_greedy(monkeypatch):
    """Our env loop with the repo's effective rule (last selectable index) replays execute_greedy_action exactly,
    and ``selectable`` is the complement of agent_observe's task mask."""

    def last_index(env, agent_id, max_waiting):
        mask = env.agent_observe(agent_id, max_waiting)[2][0, 1:]
        sel = greedy.selectable(env, agent_id, max_waiting)
        assert np.array_equal(sel, ~mask)
        return int(np.flatnonzero(sel)[-1]) + 1 if sel.any() else 0

    monkeypatch.setattr(greedy, "_nearest_action", last_index)
    for path in RAL[:10]:
        ref = make_env(path)
        with contextlib.redirect_stdout(io.StringIO()):
            ref.execute_greedy_action(plot_figure=False)
        env = make_env(path)
        greedy._run_nearest(env, True, "depot")
        assert env.current_time == ref.current_time
        assert [a["route"] for a in env.agent_dic.values()] == [a["route"] for a in ref.agent_dic.values()]


def test_greedy_nearest_takes_closest_task():
    res = greedy.greedy_nearest(_line_instance())
    assert res.success and res.routes == [[1, 2, 0]]
    assert res.makespan == pytest.approx(3 + 2 * 0.5 / 0.2)
    with pytest.raises(ValueError):
        greedy.greedy_nearest(_line_instance(), idle="nowhere")
    with pytest.raises(ValueError):
        greedy.construct(_line_instance())


def test_greedy_nearest_idle_modes_run():
    for path in RAL[:5]:
        for idle in ("wait", "depot"):
            res = greedy.greedy_nearest(path, idle=idle)
            assert res.makespan > 0 and len(res.routes) == 15


# --- plan constructors -----------------------------------------------------------------------------------------


def test_cover_is_minimal_earliest_arrival():
    rng = np.random.default_rng(0)
    for _ in range(200):
        A, K = 8, 4
        ab = (rng.random((A, K)) < 0.5).astype(float)
        req = rng.integers(0, 3, (1, K)).astype(float)
        arr = rng.random(A)
        out = np.empty(A, np.int64)
        c = greedy._cover(0, arr, req, ab, out)
        if (ab.sum(axis=0) < req[0]).any():
            assert c == -1
            continue
        m = out[:c]
        total = ab[m].sum(axis=0)
        assert (total >= req[0]).all()
        assert all(((total - ab[i]) < req[0]).any() for i in m)  # minimal


def test_constructors_are_exact_minimal_covers(instances):
    for inst in instances:
        rng = np.random.default_rng(1)
        plans = [greedy.dispatch(inst), greedy.dispatch(inst, noise=0.3, seed=3), greedy.insertion(inst, rng),
                 greedy.insertion(inst, rng, regret=4), greedy.construct(inst, time_limit=0.05)]
        for plan in plans:
            assert plan.is_minimal_cover(inst)
            sched, rep = evaluate(inst, plan), replay(inst, plan)
            assert rep["success"] and sched.success and rep["skipped"] == 0
            assert rep["makespan"] == sched.makespan
        again = greedy.construct(inst, restarts=4)
        assert again.members == greedy.construct(inst, restarts=4).members
        best = min(evaluate(inst, p).makespan for p in plans[:4])
        assert evaluate(inst, plans[-1]).makespan <= evaluate(inst, plans[0]).makespan
        assert best < evaluate(inst, random_plan(inst, np.random.default_rng(0))).makespan


def test_dispatch_cache_matches_recomputation(instances):
    """The cached-cover dispatch equals a from-scratch earliest-start list scheduler."""
    for inst in instances[:3]:
        plan = greedy.dispatch(inst)
        free, pos = np.zeros(inst.n_agents), np.full(inst.n_agents, -1)
        left = set(range(inst.n_tasks))
        buf = np.empty(inst.n_agents, np.int64)
        for j_plan in plan.order():
            best = None
            for j in sorted(left):
                arr = free + np.where(pos < 0, inst.da[:, j], inst.tt[np.maximum(pos, 0), j])
                c = greedy._cover(j, arr, inst.req, inst.ab, buf)
                s = arr[buf[:c]].max()
                if best is None or s < best[0]:
                    best = (s, j, tuple(sorted(buf[:c])))
            assert best[1] == j_plan and best[2] == plan.members[j_plan]
            left.remove(j_plan)
            free[list(best[2])], pos[list(best[2])] = best[0] + inst.dur[j_plan], j_plan


# --- CP-SAT ----------------------------------------------------------------------------------------------------


def test_weights_round_up(instances):
    inst = instances[0]
    W = cpsat.Weights.of(inst)
    assert (W.w >= (inst.dur[:, None] + inst.tt) * cpsat.SCALE - 1e-6).all()
    assert (W.w <= (inst.dur[:, None] + inst.tt) * cpsat.SCALE + 1).all()
    plan = greedy.dispatch(inst)
    start, ms = W.schedule(plan)
    sched = evaluate(inst, plan)
    assert (start >= sched.start * cpsat.SCALE - 1e-6).all() and ms >= sched.makespan * cpsat.SCALE - 1e-6


def test_species_cap_holds_for_minimal_covers(instances):
    for inst in instances:
        cap = cpsat.species_cap(inst)
        for seed in range(5):
            plan = random_plan(inst, np.random.default_rng(seed))
            for j, m in enumerate(plan.members):
                counts = np.bincount(inst.species[list(m)], minlength=inst.n_species)
                assert (counts <= cap[:, j]).all()


def test_canonical_orders_identical_agents(instances):
    for inst in instances:
        plan = random_plan(inst, np.random.default_rng(2))
        canon = cpsat.canonical(inst, plan)
        assert evaluate(inst, canon).makespan == evaluate(inst, plan).makespan
        routes = canon.routes()
        for s in range(inst.n_species):
            firsts = [min(routes[a], default=inst.n_tasks) for a in np.flatnonzero(inst.species == s)]
            assert firsts == sorted(firsts)


def test_full_model_hint_and_replay(instances):
    inst = instances[0]
    hint = greedy.dispatch(inst)
    res = cpsat.solve_full(inst, 3.0, workers=CP_WORKERS, hint=hint)
    assert res.plan.is_minimal_cover(inst)
    assert res.makespan <= evaluate(inst, hint).makespan + 1e-9
    assert replay(inst, res.plan)["makespan"] == res.makespan
    assert res.bound <= res.makespan + 1e-6 and res.makespan <= res.objective + 1e-9
    assert res.at(1e9)[0] == res.makespan and res.at(-1.0) is None
    for _, ms, plan in res.trajectory:
        assert evaluate(inst, plan).makespan == ms


def test_full_model_without_hint_and_knn(instances):
    inst = instances[1]
    for kw in ({}, {"hint": greedy.dispatch(inst), "knn": 5, "symmetry": False}):
        res = cpsat.solve_full(inst, 3.0, workers=CP_WORKERS, **kw)
        assert res.plan is not None and res.plan.is_minimal_cover(inst)
        assert replay(inst, res.plan)["makespan"] == res.makespan


def test_tiny_instance_is_solved_to_optimality():
    """Three tasks on a line, one agent: optimal tours reach x = 0.5 and come back (travel 1.0, time 5)."""
    inst = _line_instance()
    res = cpsat.solve_full(inst, 5.0, workers=CP_WORKERS)
    assert res.status == "OPTIMAL"
    assert sorted(res.plan.routes()[0]) == [0, 1, 2]
    assert res.makespan == pytest.approx(3 + 2 * 0.5 / 0.2)
    assert res.bound == pytest.approx(res.objective)


def test_lns_improves_and_replays(instances):
    inst = instances[2]
    init = greedy.dispatch(inst)
    res = cpsat.solve_lns(inst, 4.0, init, workers=CP_WORKERS, sub_time=0.5)
    assert res.iterations > 0
    assert res.makespan <= evaluate(inst, init).makespan
    assert res.plan.is_minimal_cover(inst) and replay(inst, res.plan)["makespan"] == res.makespan
    for kind in ("critical", "spatial", "temporal", "random"):
        free = cpsat._window(inst, res.plan, 10, kind, np.random.default_rng(0))
        assert len(set(free)) == 10


def test_lns_subproblem_keeps_fixed_coalitions(instances):
    inst = instances[3]
    plan = greedy.dispatch(inst)
    W = cpsat.Weights.of(inst)
    free = cpsat._window(inst, plan, 8, "spatial", np.random.default_rng(1))
    model, fixed = cpsat._subproblem(inst, W, plan, free, 2, W.schedule(plan)[1])
    model.hint(plan, W)
    solver = cpsat._solver(2.0, CP_WORKERS, 0)
    assert solver.StatusName(solver.Solve(model.m)) in ("OPTIMAL", "FEASIBLE")
    members, _ = model.extract(solver.Value, fixed)
    for j in range(inst.n_tasks):
        if j not in free:
            assert tuple(sorted(members[j])) == plan.members[j]
    new = Plan(members, np.arange(inst.n_tasks), inst.n_agents)
    assert new.covers(inst).all()


# --- campaign runner -------------------------------------------------------------------------------------------


def _runner():
    import importlib.util

    spec = importlib.util.spec_from_file_location("run_baselines", HETEROMRTA_DIR.parents[1] / "scripts" /
                                                  "run_baselines.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_runner_rows_are_env_scores(tmp_path):
    rb = _runner()
    rows = [rb._run(rb.Job("RALTestSet", "test", 0, m, b, CP_WORKERS))
            for m, b in (("greedy_repo", None), ("construct", 0.1), ("cpsat_hint", 2.0))]
    assert rows[0]["makespan"] == greedy.greedy_repo(RAL[0]).makespan and rows[0]["workers"] == 1
    for row in rows[1:]:
        assert row["success"] and row["makespan"] == row["eval_makespan"]
        plan = Plan.from_routes(row["routes"], 20)
        assert replay(RAL[0], plan)["makespan"] == row["makespan"]
    assert rows[2]["makespan"] <= rows[2]["hint_makespan"] and rows[2]["workers"] == CP_WORKERS
    for row in rows:
        rb._write(tmp_path, row)
    assert rb.done_keys(rb.out_path(tmp_path, "RALTestSet", "test")) == {(0, "greedy_repo", None),
                                                                         (0, "construct", 0.1),
                                                                         (0, "cpsat_hint", 2.0)}
