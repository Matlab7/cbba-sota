"""Checks for the Track D prototype (run with the project .venv from the repo root).

1. Static regression: DynTaskEnv with no release and no noise == original TaskEnv, bit for bit, for RL(g.),
   RL sampled rollouts, the paper greedy and plan replay (ALNS plan).
2. Masking: zeroed rows for unreleased tasks give the same policy output as deleting those rows (compact obs).
3. CRN: realized durations/arrivals in episodes equal the pre-drawn realization for every method.
4. Delay calendar: continuous-time arrival == brute-force tick simulation of LoRR semantics.
"""
from __future__ import annotations

import contextlib
import copy
import io
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dyn_env import TICK, Scenario, make_dyn_env  # noqa: E402

from cbba_sota.bench.heteromrta import load_env  # noqa: E402
from cbba_sota.hetero import Instance  # noqa: E402
from cbba_sota.solvers import greedy, rl  # noqa: E402
from cbba_sota.solvers.alns import solve  # noqa: E402

torch.set_num_threads(1)
DATA = Path(__file__).resolve().parents[2] / "data" / "hetero"


def episode_signature(env):
    return ([list(map(float, a["arrival_time"])) for a in env.agent_dic.values()],
            [list(map(int, a["route"])) for a in env.agent_dic.values()],
            [(float(t["time_start"]), float(t["time_finish"])) for t in env.task_dic.values()],
            float(env.current_time))


def check_static(net, paths):
    n = 0
    for p in paths:
        for sample, seed in ((False, 0), (True, 11), (True, 12)):
            e0 = load_env(p)
            r0 = rl.rollout(e0, net, sample=sample, seed=seed)
            e1 = make_dyn_env(p, Scenario(), inst_key=0)
            r1 = rl.rollout(e1, net, sample=sample, seed=seed)
            assert episode_signature(e0) == episode_signature(e1), p
            assert r0.makespan == r1.makespan
            n += 1
        g0 = greedy.greedy_nearest(load_env(p))
        e1 = make_dyn_env(p, Scenario(), inst_key=0)
        greedy._run_nearest(e1, True, "wait")
        assert g0.makespan == float(e1.current_time), (g0.makespan, e1.current_time)
        inst = Instance.from_pickle(p)
        plan, _ = solve(inst, 1.0, seed=0, max_iters=200)
        routes = plan.to_env_routes()
        outs = []
        for env in (load_env(p), make_dyn_env(p, Scenario(), inst_key=0)):
            env.init_state()
            for i, r in enumerate(routes):
                env.pre_set_route(list(r), i)
            with contextlib.redirect_stdout(io.StringIO()):
                env.execute_by_route()
            outs.append(episode_signature(env))
        assert outs[0] == outs[1]
    print(f"[1] static regression OK: {n} RL rollouts + greedy + ALNS replay identical on {len(paths)} instances")


def compact(task_obs, agent_obs, mask, released):
    keep = np.concatenate([[True], released])
    return task_obs[:, keep], agent_obs, mask[:, keep]


def check_masking(net, path):
    sc = Scenario("batch", release="batch")
    env = make_dyn_env(path, sc, inst_key=1)
    env.init_state()
    gaps, n = [], 0
    rng = np.random.default_rng(0)
    # drive a greedy-decoded episode and compare full vs compact at every decision
    while not env.finished and env.current_time < 200:
        released, env.current_time = env.next_decision()
        for aid in released[0] + released[1]:
            t_obs, a_obs, m = env.agent_observe(aid, False)
            if m[0, 1:].all():
                if not all(t["feasible_assignment"] for t in env.task_dic.values()):
                    env.agent_dic[aid]["no_choice"] = True
                    continue
                if env.agent_dic[aid]["current_task"] < 0:
                    continue
            rel = env.released_mask()
            idx = torch.tensor([[[aid]]])
            with torch.no_grad():
                p_full, _ = net(*(torch.tensor(x, dtype=torch.float32) for x in (t_obs, a_obs, m)), idx)
                tc, ac, mc = compact(t_obs, a_obs, m, rel)
                p_cmp, _ = net(*(torch.tensor(x, dtype=torch.float32) for x in (tc, ac, mc)), idx)
            p_full = p_full[0].numpy()
            p_back = np.zeros_like(p_full)
            p_back[np.concatenate([[True], rel])] = p_cmp[0].numpy()
            gaps.append(np.abs(p_full - p_back).max())
            n += (~rel).sum() > 0
            assert np.argmax(p_full) == np.argmax(p_back)
            env.agent_step(aid, int(np.argmax(p_full)), 0)
        env.finished = env.check_finished()
    print(f"[2] zero-row == deleted-row policy output: {len(gaps)} decisions ({n} with hidden tasks), "
          f"max |dp| = {max(gaps):.2e}, argmax identical; episode makespan {env.current_time:.2f}")


def check_crn(net, path):
    sc = Scenario("noise", dur_sigma=0.5, p_delay=0.05, min_delay=1, max_delay=10, release="poisson")
    envs = {}
    e = make_dyn_env(path, sc, inst_key=7, noise_seed=3, release_seed=5)
    rl.rollout(e, net, sample=False, seed=0)
    envs["RL(g.)"] = e
    e = make_dyn_env(path, sc, inst_key=7, noise_seed=3, release_seed=5)
    greedy._run_nearest(e, True, "wait")
    envs["greedy"] = e
    ref = make_dyn_env(path, sc, inst_key=7, noise_seed=3, release_seed=5)
    for name, env in envs.items():
        assert np.array_equal(env.dur_real, ref.dur_real) and np.array_equal(env.release, ref.release)
        for t in env.task_dic.values():
            if t["feasible_assignment"]:
                assert abs((t["time_finish"] - t["time_start"]) - ref.dur_real[t["ID"]]) < 1e-9
        assert all(np.array_equal(x[0], y[0]) for x, y in zip(env.delays, ref.delays))
        print(f"[3] CRN {name}: realized durations and delay calendar = shared draw; makespan {env.current_time:.2f}")


def brute_arrival(starts, ends, dep, tau, dt=TICK / 1000):
    """Tick-level integration: move at unit rate except inside delay intervals."""
    t, moved = dep, 0.0
    while moved < tau - 1e-12:
        inside = np.any((starts <= t + 1e-12) & (t + 1e-12 < ends))
        step = min(dt, tau - moved)
        if not inside:
            moved += step
        t += step
    return t


def check_calendar(path):
    sc = Scenario(p_delay=0.05, min_delay=1, max_delay=20)
    env = make_dyn_env(path, sc, inst_key=9, noise_seed=1)
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(200):
        i = int(rng.integers(len(env.agent_dic)))
        dep, tau = float(rng.uniform(0, 50)), float(rng.uniform(0.1, 6))
        a = env._arrive(i, dep, tau)
        b = brute_arrival(*env.delays[i], dep, tau)
        worst = max(worst, abs(a - b))
    rates = [len(s) / 400.0 for s, _ in env.delays]
    frac = np.mean([np.sum(e - s) / 400.0 for s, e in env.delays])
    exp_frac = 10.5 / ((1 - 0.05) / 0.05 + 10.5)  # renewal: E[d] / (E[failures before event] + E[d])
    print(f"[4] calendar: max |analytic - brute| = {worst:.4f} (brute step {TICK / 1000}); delayed fraction "
          f"{frac:.3f} vs renewal expectation {exp_frac:.3f}; events/unit time {np.mean(rates):.2f}")


if __name__ == "__main__":
    t0 = time.time()
    net = rl.load_policy()
    paths = [DATA / "MA-AT-25-5-50" / "dev" / f"env_{i}.pkl" for i in range(3)] + \
            [DATA / "SA-BT-25-5-20" / "dev" / f"env_{i}.pkl" for i in range(2)]
    check_static(net, paths)
    check_masking(net, paths[0])
    check_crn(net, paths[1])
    check_calendar(paths[0])
    print(f"done in {time.time() - t0:.1f} s")
