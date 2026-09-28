"""Exact core: the numba forward pass equals the HeteroMRTA env replay on key-ordered minimal covers."""
import pickle
import sys
from dataclasses import replace

import numpy as np
import pytest

from cbba_sota.bench.configs import HETEROMRTA_DIR, Setting
from cbba_sota.bench.heteromrta import TaskEnv, generate_env, load_env, save_env
from cbba_sota.hetero import (
    Instance,
    Plan,
    evaluate,
    evaluate_arrays,
    evaluate_reference,
    forward_pass,
    load_instance,
    random_plan,
    replay,
    replay_batch,
    replay_routes,
)

WORKERS = 15  # CPU cap of 16 processes: 15 replay workers + pytest
RAL = sorted(HETEROMRTA_DIR.glob("RALTestSet/env_*.pkl"), key=lambda p: int(p.stem.split("_")[1]))
# (family, agents, species, tasks, skills, instances, plans per instance)
FUZZ = [("SA-BT", 9, 3, 20, 3, 2, 20), ("SA-BT", 25, 5, 20, 5, 2, 20), ("SA-BT", 50, 5, 50, 5, 2, 20),
        ("SA-AT", 9, 3, 20, 3, 2, 20), ("SA-AT", 15, 5, 20, 5, 2, 20), ("SA-AT", 50, 5, 50, 5, 2, 20),
        ("MA-AT", 9, 3, 20, 3, 2, 20), ("MA-AT", 25, 5, 50, 5, 2, 20), ("MA-AT", 50, 5, 50, 5, 2, 20),
        ("MA-AT", 150, 5, 500, 5, 1, 4)]
GREEDY = (0.0, 0.5, 2.0)  # uniform, noisy and near-greedy coalitions


@pytest.fixture(scope="module")
def instances(tmp_path_factory) -> dict[tuple, list[Instance]]:
    """Generated with the benchmark generator, saved as env pickles, loaded back (source = pickle)."""
    root = tmp_path_factory.mktemp("hetero")
    out = {}
    for k, spec in enumerate(FUZZ):
        family, A, S, T, K, n, _ = spec
        out[spec] = []
        for s in range(n):
            path = root / f"{family}-{A}-{S}-{T}-{s}.pkl"
            save_env(generate_env(Setting(0, family, 0, A, S, T, K, {}), seed=9_000_000 + 100 * k + s), path)
            out[spec].append(Instance.from_pickle(path))
    return out


def _nonminimal(inst: Instance, plan: Plan, rng: np.random.Generator) -> Plan:
    """Add 1-3 random extra members to a third of the tasks (same keys)."""
    members = list(plan.members)
    for j in rng.choice(inst.n_tasks, size=max(1, inst.n_tasks // 3), replace=False):
        extra = rng.choice(inst.n_agents, size=rng.integers(1, 4), replace=False)
        members[j] = tuple(set(members[j]) | set(extra.tolist()))
    return Plan(members, plan.keys, plan.n_agents)


def _same(a: np.ndarray, b: np.ndarray) -> bool:
    return bool(np.allclose(a, b, rtol=0, atol=1e-9, equal_nan=True))


def test_raltestset_loads_all_50():
    assert len(RAL) == 50
    for path in RAL:
        inst = Instance.from_pickle(path)
        env = load_env(path)
        assert (inst.n_tasks, inst.n_agents, inst.n_traits, inst.n_species) == (20, 15, 5, 5)
        assert inst.speed == 0.2 and inst.source == str(path.resolve()) and inst.name == f"RALTestSet/{path.stem}"
        assert np.array_equal(inst.ab, np.eye(5)[inst.species]) and set(np.unique(inst.req)) <= {0.0, 1.0}  # SA-BT
        tasks, agents = env.task_dic, env.agent_dic
        dist = TaskEnv.calculate_eulidean_distance
        assert all(inst.tt[a, b] == dist(tasks[a], tasks[b]) / 0.2 for a in range(20) for b in range(20))
        assert all(inst.da[i, b] == dist(agents[i], tasks[b]) / 0.2 for i in range(15) for b in range(20))
    assert not hasattr(sys.modules["__main__"], "TaskEnv")  # class mapping lives in the unpickler only


def test_npz_roundtrip_and_pickling(tmp_path):
    inst = Instance.from_pickle(RAL[3])
    inst.save(tmp_path / "i.npz")
    back = load_instance(tmp_path / "i.npz")
    for f in ("req", "loc", "dur", "ab", "depot", "species", "tt", "da"):
        assert np.array_equal(getattr(inst, f), getattr(back, f))
    assert (back.speed, back.name, back.source) == (inst.speed, inst.name, inst.source)
    anon = replace(inst, source=None)
    anon.save(tmp_path / "anon.npz")
    assert Instance.load(tmp_path / "anon.npz").source is None
    assert "tt" not in pickle.loads(pickle.dumps(inst)).__dict__  # derived matrices are not shipped


def test_replay_sources_agree(tmp_path):
    """Pickle path, Instance (source pickle), npz, env object and arrays-only Instance replay identically."""
    inst = Instance.from_pickle(RAL[7])
    inst.save(tmp_path / "i.npz")
    plan = random_plan(inst, np.random.default_rng(6), 0.5)
    sources = [RAL[7], inst, load_instance(tmp_path / "i.npz"), load_env(RAL[7]), replace(inst, source=None)]
    results = [replay(s, plan) for s in sources]
    assert all(r == results[0] for r in results) and results[0]["makespan"] == evaluate(inst, plan).makespan
    with pytest.raises(ValueError, match="differs from its source"):
        replay(replace(inst, dur=inst.dur + 1), plan)


def test_kernel_equals_reference(instances):
    """numba == pure Python on minimal, non-minimal and partial plans (tasks without members)."""
    rng = np.random.default_rng(0)
    for key, insts in instances.items():
        inst = insts[0]
        for g in (2.0,) if inst.n_tasks > 100 else GREEDY:
            plan = random_plan(inst, rng, g)
            partial = Plan([m if rng.random() < 0.8 else () for m in plan.members], plan.keys, plan.n_agents)
            for p in (plan, _nonminimal(inst, plan, rng), partial):
                a, b = evaluate(inst, p), evaluate_reference(inst, p)
                assert a.makespan == b.makespan and a.success == b.success, key
                assert all(_same(getattr(a, f), getattr(b, f)) for f in ("start", "finish", "ret", "wait")), key
            assert evaluate(inst, plan).success == (evaluate(inst, plan).makespan < 200)
            assert not evaluate(inst, partial).success


def test_evaluate_equals_replay_on_minimal_covers(instances):
    rng = np.random.default_rng(1)
    jobs, evals, labels = [], [], []
    for key, insts in instances.items():
        for inst in insts:
            for k in range(key[-1]):
                plan = random_plan(inst, rng, GREEDY[k % 3] if inst.n_tasks <= 100 else 2.0)
                assert plan.is_minimal_cover(inst)
                jobs.append((inst, plan))
                evals.append(evaluate(inst, plan))
                labels.append(f"{key[0]} {key[1]}x{key[3]}")
    for path in RAL[:5]:
        inst = Instance.from_pickle(path)
        for k in range(10):
            plan = random_plan(inst, rng, GREEDY[k % 3])
            jobs.append((inst, plan))
            evals.append(evaluate(inst, plan))
            labels.append("RALTestSet")
    reps = replay_batch(jobs, WORKERS)
    ms_diff, awt_diff, exact, per = [], [], 0, {}
    for e, r, label in zip(evals, reps, labels):
        assert e.success == r["success"], label
        if e.success:
            ms_diff.append(abs(e.makespan - r["makespan"]))
            awt_diff.append(abs(e.awt - r["awt"]))
            exact += e.makespan == r["makespan"]
            per[label] = per.get(label, 0) + 1
            assert r["skipped"] == 0
    print(f"\nminimal covers: {len(jobs)} plans, {len(ms_diff)} successful and compared, {exact} bit-identical "
          f"makespans, max |dmakespan| {max(ms_diff):.3g}, max |dawt| {max(awt_diff):.3g}; per setting {per}")
    assert len(ms_diff) >= 300 and max(ms_diff) < 1e-9 and max(awt_diff) < 1e-9
    assert {"MA-AT 50x50", "SA-BT 50x50", "SA-AT 50x50", "MA-AT 150x500"} <= set(per)


def test_nonminimal_cover_rule():
    """Env rule: members joining an already covered task are rejected (decision order, here agent id order at t=0).

    Agent 0 is near and agent 1 far; both cover the task alone. The evaluator waits for both, the env starts
    with agent 0 and sends agent 1 home. Pruning drops the latest arrival and restores equality."""
    inst = Instance(req=[[1]], loc=[[0.5, 0.5]], dur=[1.0], ab=[[1], [1]], depot=[[0.5, 0.4], [0.0, 0.0]],
                    species=[0, 1])
    plan = Plan([(0, 1)], [0.0], 2)
    r, e = replay(inst, plan), evaluate(inst, plan)
    assert r["skipped"] == 1 and r["makespan"] < e.makespan - 1.0
    pruned = plan.prune_to_minimal(inst)
    assert pruned.members == [(0,)] and replay(inst, pruned)["makespan"] == evaluate(inst, pruned).makespan


def test_nonminimal_differs_and_prune_fixes(instances):
    rng = np.random.default_rng(2)
    jobs = []
    for insts in instances.values():
        if insts[0].n_tasks <= 50:
            for _ in range(4):
                plan = _nonminimal(insts[0], random_plan(insts[0], rng, 2.0), rng)
                assert not plan.is_minimal_cover(insts[0])
                pruned = plan.prune_to_minimal(insts[0])
                assert pruned.is_minimal_cover(insts[0])
                jobs += [(insts[0], plan), (insts[0], pruned)]
    reps = replay_batch(jobs, WORKERS)
    differ = 0
    for (inst, plan), r in zip(jobs, reps):
        e = evaluate(inst, plan)
        if plan.is_minimal_cover(inst):
            assert r["success"] == e.success and abs(r["makespan"] - e.makespan) < 1e-9 and r["skipped"] == 0
        elif r["success"] and e.success:
            assert r["makespan"] <= e.makespan + 1e-9  # the env runs a sub-plan (rejected members removed)
            differ += abs(r["makespan"] - e.makespan) > 1e-6
    print(f"\nnon-minimal covers: {differ}/{len(jobs) // 2} replays differ from the evaluator; pruned: all equal")
    assert differ > 0


def test_prune_never_increases_makespan(instances):
    rng = np.random.default_rng(3)
    for insts in instances.values():
        for inst in insts[:1 if insts[0].n_tasks > 100 else 2]:
            for _ in range(2 if inst.n_tasks > 100 else 10):
                plan = _nonminimal(inst, random_plan(inst, rng, rng.choice(GREEDY)), rng)
                pruned = plan.prune_to_minimal(inst)
                assert pruned.is_minimal_cover(inst) and pruned.prune_to_minimal(inst).members == pruned.members
                assert evaluate(inst, pruned).makespan <= evaluate(inst, plan).makespan + 1e-9


def test_routes_roundtrip_and_cycles(instances):
    rng = np.random.default_rng(4)
    inst = instances[FUZZ[8]][0]
    plan = random_plan(inst, rng, 0.5)
    back = Plan.from_routes(plan.to_env_routes(), inst.n_tasks, one_based=True)
    assert back.members == plan.members and back.routes() == plan.routes()
    assert evaluate(inst, back).makespan == evaluate(inst, plan).makespan
    arrays = Plan.from_arrays(*plan.to_arrays(), inst.n_agents)
    assert arrays.routes() == plan.routes()
    with pytest.raises(ValueError, match="cycle"):
        Plan.from_routes([[0, 1], [1, 0]], 2)
    # the same cycle deadlocks the env: each agent waits at a task that needs the other one first
    pair = Instance(req=[[1, 1], [1, 1]], loc=[[0.2, 0.2], [0.8, 0.8]], dur=[1.0, 1.0], ab=[[1, 0], [0, 1]],
                    depot=[[0.0, 0.0], [1.0, 1.0]], species=[0, 1])
    assert not replay_routes(pair, [[1, 2], [2, 1]])["success"]
    assert replay_routes(pair, [[1, 2], [1, 2]])["success"]
    with pytest.raises(ValueError, match="repeats"):
        Plan.from_routes([[0, 1, 0]], 2)


def test_arrays_entry_point_and_failures(instances):
    rng = np.random.default_rng(5)
    inst = instances[FUZZ[5]][0]
    plan = random_plan(inst, rng, 2.0)
    order, mem, cnt = plan.to_arrays(width=12)
    T, A = inst.n_tasks, inst.n_agents
    start, finish, ret, wait = np.empty(T), np.empty(T), np.empty(A), np.empty(A)
    ms = forward_pass(order, mem, cnt, inst.dur, inst.tt, inst.da, start, finish, ret, wait)
    e = evaluate(inst, plan)
    assert ms == e.makespan and np.array_equal(start, e.start) and np.array_equal(wait, e.wait)
    assert not evaluate_arrays(inst, order[:-1], mem, cnt).success  # a task left out of the order
    with pytest.raises(ValueError, match="out of range"):
        evaluate_arrays(inst, order, mem, cnt + 12)
    with pytest.raises(ValueError, match="without members"):
        replay(inst, Plan([()] + plan.members[1:], plan.keys, A))
    slow = replace(inst, dur=inst.dur * 20, source=None)  # past the 200 time cap: both report failure
    plan = random_plan(slow, rng, 2.0)
    e, r = evaluate(slow, plan), replay(slow, plan)
    assert e.makespan > 200 and not e.success and not r["success"]
