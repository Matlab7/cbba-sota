"""Released HeteroMRTA policy (RA-L 2025) as a baseline: RL(g.) and RL(s.N).

``_episode`` mirrors ``Worker.run_episode(training=False, max_waiting=False)`` of marmotlab/HeteroMRTA @ db51e29
(released agents are shuffled, blocked agents are skipped, one policy call per decision) but takes its randomness
from per-rollout generators, so every sample is an independent, reproducible rollout (bit-identical to the
official loop under the same seeds, see tests). ``rollout`` decodes one episode; ``lockstep`` runs many episodes of
one instance and batches their policy calls (one forward pass per round, useful on a GPU).

Scale (Table IV, up to 150 agents x 500 tasks) needs no overrides: observations are padded only in training
(``Worker.obs_padding``; ``PADDING_SIZE`` is never read), the attention network is size-agnostic, ``DECISION_DIM`` is
unused, and the 300-decision guard of ``run_episode`` counts training steps only. The simulated-time cap stays at
``MAX_TIME = 200`` as in test.py and ``execute_by_route``.
"""
from __future__ import annotations

import contextlib
import io
import random
import time
from collections.abc import Callable, Generator, Sequence
from dataclasses import dataclass, field

import numpy as np
import torch

from cbba_sota.bench.configs import HETEROMRTA_DIR, MAX_TIME, TRAIT_DIM
from cbba_sota.bench.heteromrta import TaskEnv, loads_env
from cbba_sota.hetero.plan import Plan
from cbba_sota.hetero.replay import succeeded

CHECKPOINT = HETEROMRTA_DIR / "model" / "save" / "checkpoint.pth"


@dataclass
class Result:
    makespan: float  # env current_time at the end
    success: bool  # every task finished and makespan < MAX_TIME (``succeeded``)
    completion: float  # fraction of tasks finished (the repo's per-episode "success_rate")
    awt: float  # mean over agents of the env's sum_waiting_time
    routes: list[list[int]]  # per agent, 0-based task ids in visiting order (depot visits dropped)
    depot_revisits: int = 0  # intermediate depot visits dropped from ``routes``
    coalitions: list[list[int]] = field(default_factory=list)  # per task, the env's final members
    starts: list[float] = field(default_factory=list)  # per task, env time_start (nan if unfinished)
    env_finished: bool = False  # the env's raw ``finished`` flag


@dataclass
class Sampled:
    """Best of several rollouts plus per-sample bookkeeping."""

    best: Result
    makespans: list[float]
    successes: list[bool]
    wall_s: float
    cpu_s: float
    sample_s: list[float] = field(default_factory=list)


def sample_seed(instance_seed: int, k: int) -> int:
    """Seed of sample ``k``; RL(s.N) samples are a prefix of RL(s.M) samples for M > N."""
    return int(np.random.SeedSequence([instance_seed, k]).generate_state(1)[0])


def load_policy(device: str | torch.device = "cpu") -> torch.nn.Module:
    from attention import AttentionNet

    net = AttentionNet(6 + TRAIT_DIM, 5 + 2 * TRAIT_DIM, 128)
    # Official checkpoint (trusted); it pickles numpy scalars, which weights_only=True rejects under numpy 2.
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    net.load_state_dict(checkpoint["best_model"])
    return net.to(device).eval()


Decision = tuple[int, np.ndarray, np.ndarray, np.ndarray]  # agent id, task obs, agent obs, mask


def _episode(env: TaskEnv, rng: random.Random, max_time: float) -> Generator[Decision, int, None]:
    """Yield one observation per policy decision and receive the action (0 = depot, j = task j - 1)."""
    tasks = env.task_dic.values()
    step = 0
    while not env.finished and env.current_time < max_time:
        released, env.current_time = env.next_decision()
        rng.shuffle(released[0])
        for agent_id in released[0] + released[1]:
            task_obs, agent_obs, mask = env.agent_observe(agent_id, False)
            if mask[0, 1:].all():
                if not all(t["feasible_assignment"] for t in tasks):
                    env.agent_dic[agent_id]["no_choice"] = True
                    continue
                if env.agent_dic[agent_id]["current_task"] < 0:
                    continue
            action = yield agent_id, task_obs, agent_obs, mask
            env.agent_step(agent_id, action, step)
        env.finished = env.check_finished()
        step += 1


def _result(env: TaskEnv, max_time: float) -> Result:
    _, finished = env.get_episode_reward(max_time)  # also fills the waiting times
    routes = [[t for t in a["route"] if t >= 0] for a in env.agent_dic.values()]
    revisits = sum(sum(t < 0 for t in a["route"][1:-1]) for a in env.agent_dic.values())
    awt = float(np.mean([a["sum_waiting_time"] for a in env.agent_dic.values()]))
    tasks = [env.task_dic[j] for j in range(len(env.task_dic))]
    coalitions = [[int(i) for i in t["members"]] for t in tasks]
    starts = [float(t["time_start"]) if done else float("nan") for t, done in zip(tasks, finished)]
    makespan = float(env.current_time)
    return Result(makespan, succeeded(np.all(finished), makespan), float(np.mean(finished)), awt, routes,
                  int(revisits), coalitions, starts, bool(env.finished))


def _choose(probs: torch.Tensor, sample: bool, gen: torch.Generator) -> torch.Tensor:
    if not sample:
        return torch.argmax(probs, dim=1)
    # Same draw as torch.distributions.Categorical(probs).sample() with the default generator.
    return torch.multinomial(probs / probs.sum(-1, keepdim=True), 1, True, generator=gen).squeeze(1)


def _as_tensors(obs: Sequence[Decision], device) -> tuple[torch.Tensor, ...]:
    def cat(i):
        return torch.as_tensor(np.concatenate([o[i] for o in obs]), dtype=torch.float32, device=device)

    index = torch.as_tensor([o[0] for o in obs], dtype=torch.long, device=device).view(-1, 1, 1)
    return cat(1), cat(2), cat(3), index


@torch.no_grad()
def rollout(env: TaskEnv, net: torch.nn.Module, *, sample: bool, seed: int, device="cpu",
            max_time: float = MAX_TIME) -> Result:
    """Decode one episode on ``env`` (reset here) with the policy; greedy if not ``sample``."""
    env.init_state()
    gen = torch.Generator(device).manual_seed(seed)
    episode = _episode(env, random.Random(seed), max_time)
    try:
        decision = next(episode)
        while True:
            probs, _ = net(*_as_tensors([decision], device))
            decision = episode.send(_choose(probs, sample, gen).item())
    except StopIteration:
        pass
    return _result(env, max_time)


@torch.no_grad()
def lockstep(env_bytes: bytes, net: torch.nn.Module, seeds: Sequence[int], *, sample: bool = True, device="cpu",
             max_time: float = MAX_TIME) -> list[Result]:
    """Run ``len(seeds)`` independent episodes of one instance, batching their policy calls."""
    envs = [loads_env(env_bytes) for _ in seeds]
    episodes, pending = [], {}
    for k, (env, seed) in enumerate(zip(envs, seeds)):
        env.init_state()
        episodes.append(_episode(env, random.Random(seed), max_time))
        with contextlib.suppress(StopIteration):
            pending[k] = next(episodes[k])
    gen = torch.Generator(device).manual_seed(int(seeds[0]))
    while pending:
        ks = list(pending)
        probs, _ = net(*_as_tensors([pending[k] for k in ks], device))
        for k, action in zip(ks, _choose(probs, sample, gen).tolist()):
            try:
                pending[k] = episodes[k].send(action)
            except StopIteration:
                del pending[k]
    return [_result(env, max_time) for env in envs]


def best_of(results: Sequence[Result]) -> Result:
    """Lowest makespan among successful rollouts (test.py keeps the lowest makespan; failures end at the cap)."""
    return min(results, key=lambda r: (not r.success, r.makespan))


def sample_many(env_bytes: bytes, net: torch.nn.Module, seeds: Sequence[int], *, sample: bool,
                device="cpu") -> Sampled:
    """Sequential rollouts in this process; wall/CPU time cover unpickling and decoding."""
    wall0, cpu0 = time.perf_counter(), time.process_time()
    results, sample_s = [], []
    for seed in seeds:
        t0 = time.perf_counter()
        results.append(rollout(loads_env(env_bytes), net, sample=sample, seed=seed, device=device))
        sample_s.append(time.perf_counter() - t0)
    return Sampled(best_of(results), [r.makespan for r in results], [r.success for r in results],
                   time.perf_counter() - wall0, time.process_time() - cpu0, sample_s)


def sample_lockstep(env_bytes: bytes, net: torch.nn.Module, seeds: Sequence[int], *, sample: bool,
                    device="cpu") -> Sampled:
    """``lockstep`` with timing; wall/CPU time cover unpickling and decoding."""
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)
    wall0, cpu0 = time.perf_counter(), time.process_time()
    results = lockstep(env_bytes, net, seeds, sample=sample, device=device)
    return Sampled(best_of(results), [r.makespan for r in results], [r.success for r in results],
                   time.perf_counter() - wall0, time.process_time() - cpu0)


def sample_until(env_bytes: bytes, net: torch.nn.Module, seed_of: Callable[[int], int], deadline: float, *,
                 first: int = 4, max_batch: int = 64, safety: float = 0.95, device="cpu") -> Sampled:
    """Anytime RL(s.N) in this process: ``lockstep`` batches of fresh samples (seeds ``seed_of(0)``, ``seed_of(1)``,
    ...) while the next batch is predicted to end before ``deadline`` (``time.monotonic()``). The first batch
    (``first`` samples) always runs. Later batches fill ``safety`` of the remaining time at the slowest per-sample
    time seen so far; lockstep gets cheaper per sample as batches grow, so this rarely overruns."""
    wall0, cpu0 = time.perf_counter(), time.process_time()
    results: list[Result] = []
    per_sample, size = 0.0, first
    while size > 0:
        t = time.perf_counter()
        results += lockstep(env_bytes, net, [seed_of(len(results) + k) for k in range(size)], device=device)
        per_sample = max(per_sample, (time.perf_counter() - t) / size)
        size = min(max_batch, int(safety * (deadline - time.monotonic()) / per_sample))
    return Sampled(best_of(results), [r.makespan for r in results], [r.success for r in results],
                   time.perf_counter() - wall0, time.process_time() - cpu0)


def merge(parts: Sequence[Sampled]) -> Sampled:
    """Combine sample blocks that ran concurrently in separate processes: wall = slowest block, CPU = sum."""
    return Sampled(best_of([p.best for p in parts]), [m for p in parts for m in p.makespans],
                   [s for p in parts for s in p.successes], max(p.wall_s for p in parts),
                   sum(p.cpu_s for p in parts), [t for p in parts for t in p.sample_s])


def to_plan(result: Result, n_agents: int) -> Plan:
    """The rollout's final coalitions as a plan keyed by env start times (unfinished tasks get no members), e.g. to
    warm start ALNS. Keys follow every agent's visiting order, so the plan is deadlock-free, and its forward pass is
    no later than the rollout (agents leave at their previous finish and skip idling and detours). Coalitions may be
    non-minimal: prune with ``Plan.prune_to_minimal`` (never later) before comparing with the env replay."""
    finished = [not np.isnan(s) for s in result.starts]
    members = [m if done else () for m, done in zip(result.coalitions, finished)]
    keys = [s if done else np.inf for s, done in zip(result.starts, finished)]
    return Plan(members, keys, n_agents)


def replay(env_bytes: bytes, routes: Sequence[Sequence[int]], max_time: float = MAX_TIME) -> tuple[float, bool]:
    """Makespan and success (``succeeded``) of 0-based per-agent routes under ``pre_set_route`` +
    ``execute_by_route``."""
    env = loads_env(env_bytes)
    env.init_state()
    for agent_id, route in enumerate(routes):
        env.pre_set_route([t + 1 for t in route], agent_id)
    with contextlib.redirect_stdout(io.StringIO()):
        env.execute_by_route()
    _, finished = env.get_episode_reward(max_time)
    return float(env.current_time), succeeded(np.all(finished), env.current_time)
