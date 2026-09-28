"""Integration of the components: RL rollouts as ALNS warm starts, the anytime RL sampler and the matched-budget
comparison script's bookkeeping."""
import importlib.util
import random
import time

import numpy as np
import pytest

from cbba_sota.bench import configs
from cbba_sota.bench.configs import ROOT, Setting
from cbba_sota.bench.heteromrta import generate_env, save_env
from cbba_sota.hetero import Instance, evaluate, replay
from cbba_sota.solvers import rl
from cbba_sota.solvers.alns import solve


@pytest.fixture(scope="module")
def net():
    import torch

    torch.set_num_threads(1)
    return rl.load_policy()


@pytest.fixture(scope="module")
def case(tmp_path_factory):
    """MA-AT instance (non-minimal RL coalitions are common there) and its env bytes."""
    path = tmp_path_factory.mktemp("integration") / "ma.pkl"
    save_env(generate_env(Setting(0, "MA-AT", 0, 15, 5, 30, 5, {}), seed=8_000_000), path)
    return Instance.from_pickle(path), path.read_bytes()


def test_rollout_plan_matches_env_and_warm_starts_alns(case, net):
    inst, data = case
    results = rl.lockstep(data, net, [rl.sample_seed(1, k) for k in range(6)])
    assert all(r.success for r in results)
    pruned_better = 0
    for r in results:
        plan = rl.to_plan(r, inst.n_agents)
        assert evaluate(inst, plan).makespan == pytest.approx(r.makespan, abs=1e-9)
        pruned = plan.prune_to_minimal(inst)
        ms = evaluate(inst, pruned).makespan
        assert pruned.is_minimal_cover(inst) and ms <= r.makespan + 1e-9
        assert replay(inst, pruned)["makespan"] == ms
        pruned_better += ms < r.makespan - 1e-9
    best = rl.best_of(results)
    init = rl.to_plan(best, inst.n_agents).prune_to_minimal(inst)
    plan, st = solve(inst, 30.0, seed=0, init_plan=rl.to_plan(best, inst.n_agents), max_iters=300)
    assert st.init_makespan <= evaluate(inst, init).makespan + 1e-9
    assert st.makespan <= st.init_makespan
    rep = replay(inst, plan)
    assert rep["success"] and rep["skipped"] == 0 and rep["makespan"] == st.makespan


def test_to_plan_leaves_unfinished_tasks_empty(case, net):
    inst, data = case
    r = rl.rollout(rl.loads_env(data), net, sample=True, seed=3)
    r.starts[0], r.coalitions[0] = float("nan"), [0]  # as if task 0 never started
    plan = rl.to_plan(r, inst.n_agents)
    assert plan.members[0] == () and plan.order()[-1] == 0
    full, _ = solve(inst, 30.0, seed=0, init_plan=plan, max_iters=50)
    assert full.is_minimal_cover(inst)


def test_sample_until(case, net):
    _, data = case

    def seed_of(n: int) -> int:
        return rl.sample_seed(5, n)

    first = rl.sample_until(data, net, seed_of, time.monotonic() - 1.0, first=3)  # past the deadline
    assert len(first.makespans) == 3
    assert first.makespans == [r.makespan for r in rl.lockstep(data, net, [seed_of(n) for n in range(3)])]
    t0 = time.monotonic()
    timed = rl.sample_until(data, net, seed_of, t0 + 4 * first.wall_s, first=3)
    assert len(timed.makespans) > 3 and timed.makespans[:3] == first.makespans
    assert time.monotonic() - t0 < 8 * first.wall_s  # later batches fill the budget instead of overrunning it


def _compare_dev():
    spec = importlib.util.spec_from_file_location("compare_dev", ROOT / "scripts" / "compare_dev.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_compare_dev_units_and_ratio():
    cd = _compare_dev()
    units = cd._units(list(cd.SETTINGS), 3, set(), random.Random(0), 1.0)
    jobs = [j for u in units for j in u]
    keys = [(j["setting"], j["instance"], j["budget"], j["method"]) for j in jobs]
    assert len(keys) == len(set(keys))
    per_setting = 3 * (2 * (len(cd.CORES) - 1) + 1)  # two budgets for every method but the greedy
    assert len(jobs) == len(cd.SETTINGS) * per_setting
    for u in units:
        assert sum(j["cores"] for j in u) <= cd.LANE
        assert len({(j["setting"], j["budget"]) for j in u}) == 1
        assert len(u) == 1 or all(j["cores"] == 1 for j in u)
    b1 = [j["budget_s"] for j in jobs if j["setting"] == "MA-AT-50-5-200" and j["budget"] == "B1"]
    assert set(b1) == {configs.get("MA-AT-50-5-200").paper["RL(s.10)"].time_s}
    budgets = [max(j["budget_s"] or 0 for j in u) for u in units]
    assert budgets == sorted(budgets, reverse=True)
    done = {keys[0]}
    assert len([j for u in cd._units(list(cd.SETTINGS), 3, done, random.Random(0), 1.0) for j in u]) == len(jobs) - 1

    ours, other = np.array([9.0, 9.0, 18.0]), np.array([10.0, 10.0, 20.0])
    res = cd._ratio(ours, other)
    assert res["ratio"] == pytest.approx(0.9) and res["wins"] == 3 and res["lo"] == res["hi"] == pytest.approx(0.9)
    assert cd._score({"makespan": 250.0, "success": False}) == cd.FAIL
