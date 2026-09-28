import pickle
import random
from dataclasses import replace

import numpy as np
import pytest
import torch

from cbba_sota.bench import SETTINGS, configs, get
from cbba_sota.bench.heteromrta import (
    TaskEnv,
    dumps_env,
    generate_env,
    load_env,
    loads_env,
)

SHIPPED = configs.HETEROMRTA_DIR / "RALTestSet"


def _arrays(env: TaskEnv) -> dict[str, np.ndarray]:
    tasks = env.task_dic.values()
    return {"req": np.array([t["requirements"] for t in tasks]), "loc": np.array([t["location"] for t in tasks]),
            "dur": np.array([t["time"] for t in tasks]), "ab": np.array([a["abilities"] for a in env.agent_dic.values()]),
            "depot": np.array([d["location"] for d in env.depot_dic.values()])}


def test_settings_table():
    assert len(SETTINGS) == 17 and len({s.name for s in SETTINGS}) == 17
    assert [s.index for s in SETTINGS] == list(range(17))
    for s in SETTINGS[:16]:
        assert {"RL(g.)", "RL(s.10)"} <= set(s.paper) and s.n_agents % s.n_species == 0
        assert not s.single_skill or s.n_species == s.n_skills
    s = get("MA-AT-150-10-500")
    assert (s.per_species, s.n_skills, s.single_skill, s.binary) == (15, 5, False, False)
    assert s.seed("test", 3) == 100_000 + 1000 * 14 + 3 and s.seed("dev", 3) == 500_000 + 1000 * 14 + 3
    seeds = [(t.seed(split, i)) for t in SETTINGS[:16] for split in ("test", "dev") for i in range(t.n_instances(split))]
    assert len(seeds) == len(set(seeds))
    assert get("RALTestSet").splits() == ("test",) and get("RALTestSet").n_instances("test") == 50
    with pytest.raises(IndexError):
        s.seed("dev", 20)


def test_single_skill_generator_reproduces_shipped_testset():
    ral = replace(get("RALTestSet"), source=None)  # SA-BT, 5 species x 3 agents, 20 tasks; shipped seeds are 0..49
    for i in (0, 7, 49):
        ours, shipped = _arrays(generate_env(ral, i)), _arrays(load_env(SHIPPED / f"env_{i}.pkl"))
        for k in ours:
            assert np.array_equal(ours[k], shipped[k]), k


@pytest.mark.parametrize("name", ["SA-BT-9-3-20", "SA-AT-15-5-20", "MA-AT-9-3-20", "MA-AT-25-5-20"])
def test_generated_instances(name):
    s = get(name)
    env = generate_env(s, s.seed("dev", 0))
    a = _arrays(env)
    assert type(env) is TaskEnv and env.traits_dim == 5 and env.rng is None
    assert a["req"].shape == (s.n_tasks, 5) and a["ab"].shape == (s.n_agents, 5)
    assert (a["req"].sum(1) > 0).all() and a["req"].max() == (1 if s.binary else 2)
    assert not a["req"][:, s.n_skills:].any() and not a["ab"][:, s.n_skills:].any()
    assert ((a["ab"].sum(1) == 1) if s.single_skill else (a["ab"].sum(1) >= 1)).all()
    assert (a["ab"].sum(0) >= a["req"].max(0)).all()  # the generator guarantees coverage by all agents
    assert ((a["dur"] >= 0) & (a["dur"] < 5)).all() and a["depot"].shape == (s.n_species, 2)
    again = loads_env(dumps_env(generate_env(s, s.seed("dev", 0))))
    assert all(np.array_equal(v, _arrays(again)[k]) for k, v in a.items())
    assert b"__main__" not in dumps_env(env)


def test_loads_shipped_main_pickle():
    raw = (SHIPPED / "env_0.pkl").read_bytes()
    assert b"__main__" in raw
    with pytest.raises(AttributeError):
        pickle.loads(raw)
    assert type(loads_env(raw)) is TaskEnv


@pytest.fixture(scope="module")
def net():
    torch.set_num_threads(1)
    from cbba_sota.solvers import rl

    return rl.load_policy()


@pytest.mark.parametrize("sample", [False, True])
def test_rollout_matches_official_worker(net, sample):
    from worker import Worker

    from cbba_sota.solvers import rl

    worker = Worker(0, net, net, 0, torch.device("cpu"))
    s = get("MA-AT-9-3-20")
    for data in ((SHIPPED / "env_3.pkl").read_bytes(), dumps_env(generate_env(s, s.seed("dev", 1)))):
        for seed in (0, 11):
            env = loads_env(data)
            env.init_state()
            worker.env = env
            random.seed(seed)
            torch.manual_seed(seed)
            _, _, ref = worker.run_episode(False, sample, False)
            ours = rl.rollout(loads_env(data), net, sample=sample, seed=seed)
            assert ours.makespan == ref["makespan"][0] and ours.awt == ref["waiting_time"][0]
            assert ours.completion == ref["success_rate"][0] and ours.success == (ours.completion == 1)
            assert ours.routes == [[t for t in a["route"] if t >= 0] for a in env.agent_dic.values()]


def test_lockstep_and_blocks(net):
    from cbba_sota.solvers import rl

    data = (SHIPPED / "env_5.pkl").read_bytes()
    seeds = [rl.sample_seed(5, k) for k in range(4)]
    seq = [rl.rollout(loads_env(data), net, sample=False, seed=x) for x in seeds]
    lock = rl.lockstep(data, net, seeds, sample=False)
    assert [r.routes for r in lock] == [r.routes for r in seq]
    parts = [rl.sample_many(data, net, seeds[:3], sample=True), rl.sample_many(data, net, seeds[3:], sample=True)]
    whole = rl.merge(parts)
    assert whole.makespans == rl.sample_many(data, net, seeds, sample=True).makespans
    assert whole.best.makespan == min(whole.makespans) and whole.wall_s == max(p.wall_s for p in parts)
    assert rl.replay(data, whole.best.routes) == (whole.best.makespan, True)


def test_best_of_prefers_success():
    from cbba_sota.solvers.rl import Result, best_of

    fail, ok = Result(200.1, False, 0.9, 0.0, []), Result(180.0, True, 1.0, 0.0, [])
    assert best_of([fail, ok]) is ok and best_of([fail]) is fail
