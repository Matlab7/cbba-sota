"""B1: the published HeteroMRTA policy run online in the dynamic env (docs/trackD-spec.md Sections 3.2, 5.1, 5.2).

The released checkpoint (marmotlab/HeteroMRTA @ db51e29, unmodified) decides for every robot at every decision
epoch of the env, exactly as ``Worker.run_episode(training=False, max_waiting=False)``: released robots are shuffled,
blocked robots are re-polled, one policy call per decision. ``RL(g.)`` decodes greedily (argmax), ``RL(s.1)`` samples
one action per decision. Best-of-N over realized episodes is not an online policy and is not offered here (RL-MPC,
B2, is the legal way to spend more compute on the policy).

Causal observation rules (what the policy is allowed to see):
- Unreleased tasks are all-zero rows and masked (D1, done by the env). The network treats an all-zero row as padding
  (``attention.get_attn_pad_mask``), so a zero row equals a deleted row; checked in the tests.
- Arrival and duration features are causal (D3, the env's ``obs="causal"`` mode): nominal durations until observed,
  ETA = departure + nominal travel + stall seen so far. ``run_online`` refuses an oracle-observation env unless
  ``allow_oracle=True`` (the diagnosis-only reference).
- Failure notices: once a robot's failure is *detected* (heartbeat timeout, ``Realization.detect_after`` after the
  onset under good communication) it is removed from the team the policy sees and is never asked for a decision.
  Removal deletes its agent row and remaps the decider's index (``drop_rows``). For every robot but robot 0 this is
  numerically the spec's "zero row" (max |dp| ~ 2.5e-7, tests); a zero row 0 makes the released network return NaN,
  because ``AttentionNet.encoding_agents`` pools with the padding mask of the *first* row, so zeroing is not used.
  A failed robot that is not yet detected stays visible (the policy cannot know). What happens physically to a
  failed robot and to its coalition (D5/D6) is the env's business, not the policy's.

Entry point ``run_b1``: on ``DynTaskEnvX`` it runs ``B1Policy`` inside the env's own ``run_policy`` loop; on any other
``TaskEnv`` (the pilot env, a plain static env) it runs ``run_online``, a transcription of the native loop, which
refuses ``DynTaskEnvX``. Failure information (env contract, implemented by ``DynTaskEnvX``) is read, in order of
preference, from ``env.known_failed(t) -> iterable of robot ids`` and ``env.is_failed(i, t) -> bool``, or else from
``env.realization.fail_onset`` / ``detect_after`` (``cbba_sota.dyn.perturb.Realization``); an env with none of these
has no failures.

Worlds (``make_world``), every result carries the ``world`` string that produced it:
- ``"x"`` (ground truth): ``cbba_sota.dyn.env.DynTaskEnvX`` (D1-D8, failures). ``run_b1`` drives it through the env's
  own ``run_policy`` loop with ``B1Policy`` (the same decision rule, timing and ``drop_rows``), because failure
  detections free agents that only that loop re-polls. The coalition rule is an explicit option: ``"decision"``
  (the native rule; the env's default for policies, bit-identical static reduction) or ``"arrival"`` (D2, the rule
  every plan-following method is scored under). The released policy often forms non-minimal coalitions on MA-AT
  settings, and the two rules then give different episodes, so B1 is reported under both.
- ``"pilot"``: the day-1 prototype ``pilots/trackD/dyn_env.DynTaskEnv`` (release, noise and causal observations; no
  failures, no D5/D7; decision-order coalitions) driven by ``run_online`` from a ``cbba_sota.dyn.perturb``
  realization, so its draws are the CRN draws of the real benchmark.

Pinned code: the policy loader and the sampling rule are copied from ``cbba_sota/solvers/rl.py`` at commit 4cf5e04
(identical to db51e29's ``Worker`` semantics); the loop is a transcription of ``cbba_sota.solvers.rl._episode`` at the
same commit, so under the static reduction it is bit-identical to ``rl.rollout`` (tests).
"""
from __future__ import annotations

import hashlib
import importlib.util
import random
import sys
import time
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from cbba_sota.bench.configs import HETEROMRTA_DIR, MAX_TIME, ROOT, TRAIT_DIM
from cbba_sota.bench.heteromrta import TaskEnv, load_env

PINNED_COMMIT = "4cf5e04"
CHECKPOINT = HETEROMRTA_DIR / "model" / "save" / "checkpoint.pth"
PILOT_ENV = ROOT / "pilots" / "trackD" / "dyn_env.py"

METHODS = {"RL(g.)": False, "RL(s.1)": True}  # method name -> sample


# ---------------------------------------------------------------------------------------------------------------
# policy (copied from cbba_sota/solvers/rl.py @ 4cf5e04)
# ---------------------------------------------------------------------------------------------------------------
def load_policy(device: str | torch.device = "cpu") -> torch.nn.Module:
    """The released checkpoint, unmodified (``AttentionNet(6 + 5, 5 + 2 * 5, 128)``)."""
    # third_party/HeteroMRTA is on sys.path via cbba_sota.bench.heteromrta (imported above)
    from attention import AttentionNet

    net = AttentionNet(6 + TRAIT_DIM, 5 + 2 * TRAIT_DIM, 128)
    # Official checkpoint (trusted); it pickles numpy scalars, which weights_only=True rejects under numpy 2.
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    net.load_state_dict(checkpoint["best_model"])
    return net.to(device).eval()


def _choose(probs: torch.Tensor, sample: bool, gen: torch.Generator) -> torch.Tensor:
    if not sample:
        return torch.argmax(probs, dim=1)
    # Same draw as torch.distributions.Categorical(probs).sample() with the default generator.
    return torch.multinomial(probs / probs.sum(-1, keepdim=True), 1, True, generator=gen).squeeze(1)


def policy_seed(instance_key: int, crn_seed: int, method: str = "RL(g.)") -> int:
    """Seed of the policy's own randomness (robot shuffle, sampling), keyed like the CRN draws by (instance, seed).
    Same derivation as ``scripts/trackD_run.py`` so B1 rows from either runner are the same episodes."""
    tag = int(hashlib.sha256(method.encode()).hexdigest()[:8], 16)
    return int(np.random.SeedSequence([int(instance_key), int(crn_seed), tag]).generate_state(1)[0])


# ---------------------------------------------------------------------------------------------------------------
# failure notices
# ---------------------------------------------------------------------------------------------------------------
def failure_view(env: TaskEnv) -> tuple[Callable[[float], set[int]], Callable[[int, float], bool]]:
    """(known_failed(t), is_failed(i, t)) for ``env`` per the env contract in the module docstring."""
    if hasattr(env, "known_failed"):
        known = lambda t: {int(i) for i in env.known_failed(t)}
    else:
        rz = getattr(env, "realization", None)
        onset = None if rz is None else np.asarray(getattr(rz, "fail_onset", []), float)
        if onset is None or onset.size == 0 or not np.isfinite(onset).any():
            known = lambda t: set()
        else:
            det = onset + float(getattr(rz, "detect_after", 0.0))
            known = lambda t: set(np.flatnonzero(det <= t + 1e-12).tolist())
    if hasattr(env, "is_failed"):
        dead = lambda i, t: bool(env.is_failed(i, t))
    else:
        rz = getattr(env, "realization", None)
        onset = None if rz is None else np.asarray(getattr(rz, "fail_onset", []), float)
        if onset is None or onset.size == 0:
            dead = lambda i, t: False
        else:
            dead = lambda i, t: bool(onset[i] <= t + 1e-12)
    return known, dead


def zero_rows(agent_obs: np.ndarray, rows: Iterable[int]) -> np.ndarray:
    """Agent observation (1 x A x F) with the given robots' rows zeroed (padding for the network; equal to deleting
    them except for row 0, see the module docstring). Kept for the equivalence tests; the driver uses ``drop_rows``."""
    rows = list(rows)
    if not rows:
        return agent_obs
    out = np.array(agent_obs, copy=True)
    out[0, rows] = 0.0
    return out


def drop_rows(agent_obs: np.ndarray, rows: Iterable[int], agent_id: int) -> tuple[np.ndarray, int]:
    """Agent observation without the given robots' rows, and ``agent_id``'s index in it (``agent_id`` is kept)."""
    rows = set(rows)
    if not rows:
        return agent_obs, agent_id
    if agent_id in rows:
        raise ValueError("the deciding robot cannot be removed")
    keep = np.array([k not in rows for k in range(agent_obs.shape[1])])
    return agent_obs[:, keep], int(keep[:agent_id].sum())


# ---------------------------------------------------------------------------------------------------------------
# the online episode
# ---------------------------------------------------------------------------------------------------------------
@dataclass
class OnlineResult:
    method: str
    world: str
    makespan: float  # env current_time at the end (physical: every task finished and every live robot home)
    success: bool  # every task finished and makespan < MAX_TIME
    completion: float  # fraction of tasks finished
    env_finished: bool
    n_decisions: int
    cpu_ms_p50: float  # CPU per policy decision (observation + forward pass), this process
    cpu_ms_p95: float
    cpu_s: float  # CPU of the whole episode (env included)
    wall_s: float
    travel: float  # total travel distance of all robots
    depot_revisits: int
    abandons: int  # robots that left an uncovered task (env ``abandoned_agent`` lists)
    n_known_failed: int  # robots whose failure was notified to the policy during the episode
    stuck: bool = False  # the loop stopped on a livelock (see run_online); counted as a failure
    release_polls: int = 0
    policy_seed: int = 0  # the policy's own seed (shuffle and sampling), not the CRN seed
    routes: list[list[int]] = field(default_factory=list)
    extra: dict = field(default_factory=dict)  # world-specific counters (DynTaskEnvX: wasted trips, restarts, ...)

    def row(self) -> dict:
        d = asdict(self)
        d.pop("routes")
        d.update(d.pop("extra"))
        return d


@torch.no_grad()
def run_online(env: TaskEnv, net: torch.nn.Module, *, sample: bool, seed: int, method: str | None = None,
               world: str = "unknown", device: str = "cpu", max_time: float = MAX_TIME,
               allow_oracle: bool = False, keep_routes: bool = False) -> OnlineResult:
    """Run the policy online on ``env`` (reset here) and score the episode. ``seed`` drives the robot shuffle and
    the sampling generator (use ``policy_seed``)."""
    if hasattr(env, "run_policy"):
        # DynTaskEnvX re-polls agents freed by failures and D7/D8 epochs only in its own loop; the native loop can
        # spin there (seen on F1-R1 with N12), so the env's loop is mandatory: use run_b1.
        raise TypeError("run_online drives native-loop worlds only; use run_b1 for DynTaskEnvX")
    obs_mode = getattr(getattr(env, "sc", None), "obs", None) or getattr(env, "obs_mode", None)
    if obs_mode == "oracle" and not allow_oracle:
        raise ValueError("oracle observations leak realized arrivals; B1 must run with causal observations")
    method = method or ("RL(s.1)" if sample else "RL(g.)")
    known_failed, is_failed = failure_view(env)
    cpu0, wall0 = time.process_time(), time.perf_counter()
    env.init_state()
    gen = torch.Generator(device).manual_seed(seed)
    rng = random.Random(seed)
    tasks = env.task_dic.values()
    step, dec_cpu, notified = 0, [], set()
    # A round that takes no step and does not advance the clock leaves the env state unchanged apart from
    # no_choice flags (at most one per robot), so more than A + 2 such rounds in a row is a livelock, e.g. an env
    # without D6/D7 waiting for a failed robot to come home. The episode then ends unsuccessfully (``stuck``).
    stall, stall_limit, stuck, last_t = 0, len(env.agent_dic) + 2, False, None
    while not env.finished and env.current_time < max_time:
        released, env.current_time = env.next_decision()
        rng.shuffle(released[0])
        now = float(env.current_time)
        gone = known_failed(now)
        notified |= gone
        n_steps = 0
        for agent_id in released[0] + released[1]:
            if agent_id in gone or is_failed(agent_id, now):
                # A failed robot takes no decision (its physics is the env's). Park it the native way, as the
                # loop does for a robot with no choice, so an env that still releases it cannot stall the clock.
                env.agent_dic[agent_id]["no_choice"] = True
                continue
            c0 = time.process_time()
            task_obs, agent_obs, mask = env.agent_observe(agent_id, False)
            if mask[0, 1:].all():
                if not all(t["feasible_assignment"] for t in tasks):
                    env.agent_dic[agent_id]["no_choice"] = True
                    continue
                if env.agent_dic[agent_id]["current_task"] < 0:
                    continue
            agent_obs, index = drop_rows(agent_obs, gone, agent_id)
            x = [torch.as_tensor(v, dtype=torch.float32, device=device) for v in (task_obs, agent_obs, mask)]
            probs, _ = net(*x, torch.as_tensor([[[index]]], dtype=torch.long, device=device))
            action = _choose(probs, sample, gen).item()
            dec_cpu.append(time.process_time() - c0)
            env.agent_step(agent_id, action, step)
            n_steps += 1
        env.finished = env.check_finished()
        step += 1
        stall = stall + 1 if (n_steps == 0 and now == last_t) else 0
        last_t = now
        if stall > stall_limit:
            stuck = True
            break
    return _score(env, method, world, seed, max_time, dec_cpu, time.process_time() - cpu0,
                  time.perf_counter() - wall0, len(notified), keep_routes, stuck)


def _score(env, method, world, seed, max_time, dec_cpu, cpu_s, wall_s, n_known_failed, keep_routes,
           stuck) -> OnlineResult:
    finished = [bool(t["finished"]) for t in env.task_dic.values()]
    makespan = float(env.current_time)
    if not np.isfinite(makespan):
        makespan = max_time
    success = bool(all(finished) and makespan < max_time and not stuck)
    agents = env.agent_dic.values()
    dc = np.asarray(dec_cpu) * 1e3 if dec_cpu else np.zeros(1)
    return OnlineResult(
        method=method, world=world, makespan=makespan, success=success, completion=float(np.mean(finished)),
        env_finished=bool(env.finished), n_decisions=len(dec_cpu), cpu_ms_p50=float(np.percentile(dc, 50)),
        cpu_ms_p95=float(np.percentile(dc, 95)), cpu_s=float(cpu_s), wall_s=float(wall_s),
        travel=float(sum(a["travel_dist"] for a in agents)),
        depot_revisits=int(sum(sum(t < 0 for t in a["route"][1:-1]) for a in agents)),
        abandons=int(sum(len(t.get("abandoned_agent", [])) for t in env.task_dic.values())),
        n_known_failed=int(n_known_failed), stuck=bool(stuck), release_polls=int(getattr(env, "n_release_polls", 0)), policy_seed=int(seed),
        routes=[[int(t) for t in a["route"] if t >= 0] for a in agents] if keep_routes else [])


# ---------------------------------------------------------------------------------------------------------------
# DynTaskEnvX: the policy in the env's own loop
# ---------------------------------------------------------------------------------------------------------------
class B1Policy:
    """``cbba_sota.dyn.env.Policy`` for ``DynTaskEnvX.run_policy``: the released network with the causal rules of
    this module (known-failed robots deleted from the agent observation) and per-decision CPU timing."""

    def __init__(self, net: torch.nn.Module, *, sample: bool, seed: int, device: str = "cpu"):
        self.net, self.sample, self.device = net, sample, device
        self.gen = torch.Generator(device).manual_seed(int(seed))
        self.dec_cpu: list[float] = []
        self.notified: set[int] = set()
        self._pending = None

    def selectable(self, env, agent_id: int) -> bool:
        c0 = time.process_time()
        task_obs, agent_obs, mask = env.agent_observe(agent_id, False)
        self._pending = (agent_id, c0, task_obs, agent_obs, mask)
        return not bool(mask[0, 1:].all())

    @torch.no_grad()
    def act(self, env, agent_id: int) -> int:
        aid, c0, task_obs, agent_obs, mask = self._pending
        assert aid == agent_id, "act must follow selectable for the same agent"
        gone = {int(i) for i in env.known_failed(env.current_time)}
        self.notified |= gone
        agent_obs, index = drop_rows(agent_obs, gone, agent_id)
        x = [torch.as_tensor(v, dtype=torch.float32, device=self.device) for v in (task_obs, agent_obs, mask)]
        probs, _ = self.net(*x, torch.as_tensor([[[index]]], dtype=torch.long, device=self.device))
        action = int(_choose(probs, self.sample, self.gen).item())
        self.dec_cpu.append(time.process_time() - c0)
        return action


def run_b1(env: TaskEnv, net: torch.nn.Module, *, sample: bool, seed: int, method: str | None = None,
           world: str | None = None, device: str = "cpu", max_time: float = MAX_TIME,
           keep_routes: bool = False) -> OnlineResult:
    """B1 on any world: ``DynTaskEnvX.run_policy`` when the env has it (ground truth), else ``run_online``."""
    method = method or ("RL(s.1)" if sample else "RL(g.)")
    if not hasattr(env, "run_policy"):
        return run_online(env, net, sample=sample, seed=seed, method=method, world=world or "unknown",
                          device=device, max_time=max_time, keep_routes=keep_routes)
    if getattr(env, "obs_mode", "causal") != "causal":
        raise ValueError("oracle observations leak realized arrivals; B1 must run with causal observations")
    pol = B1Policy(net, sample=sample, seed=seed, device=device)
    cpu0, wall0 = time.process_time(), time.perf_counter()
    ep = env.run_policy(pol, shuffle_seed=seed, max_time=max_time)
    cpu_s, wall_s = time.process_time() - cpu0, time.perf_counter() - wall0
    dc = np.asarray(pol.dec_cpu) * 1e3 if pol.dec_cpu else np.zeros(1)
    agents = env.agent_dic.values()
    res = OnlineResult(
        method=method, world=world or ep.world, makespan=float(ep.makespan) if ep.makespan is not None else max_time,
        success=bool(ep.success), completion=float(ep.completion), env_finished=bool(ep.env_finished),
        n_decisions=len(pol.dec_cpu), cpu_ms_p50=float(np.percentile(dc, 50)), cpu_ms_p95=float(np.percentile(dc, 95)),
        cpu_s=float(cpu_s), wall_s=float(wall_s), travel=float(ep.travel),
        depot_revisits=int(sum(sum(t < 0 for t in a["route"][1:-1]) for a in agents)), abandons=int(ep.abandons),
        n_known_failed=int(ep.detected), stuck=ep.makespan is None, release_polls=int(getattr(env, "n_release_polls", 0)),
        policy_seed=int(seed), routes=[[int(t) for t in a["route"] if t >= 0] for a in agents] if keep_routes else [])
    res.extra = {"wasted_trips": int(ep.wasted_trips), "restarts": int(ep.restarts), "failures": int(ep.failures),
                 "wait": float(ep.wait), "epochs": int(ep.epochs), "structural_events": int(ep.structural_events),
                 "notified": len(pol.notified)}
    return res


# ---------------------------------------------------------------------------------------------------------------
# worlds
# ---------------------------------------------------------------------------------------------------------------
_PILOT = None


def _pilot_module():
    """Import ``pilots/trackD/dyn_env.py`` without letting its frozen ``_snap`` copy shadow the live package."""
    global _PILOT
    if _PILOT is None:
        saved = list(sys.path)
        try:
            spec = importlib.util.spec_from_file_location("trackD_pilot_dyn_env", PILOT_ENV)
            mod = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = mod  # dataclasses resolve annotations through sys.modules
            spec.loader.exec_module(mod)
        finally:
            sys.path[:] = saved
        _PILOT = mod
    return _PILOT


def pilot_world_name() -> str:
    return "pilot:pilots/trackD/dyn_env.py(DynTaskEnv; D1,D3,D4; no D5-D8, no failures)"


def pilot_world(path: str | Path, realization, *, obs: str = "causal") -> TaskEnv:
    """The pilot ``DynTaskEnv`` for one instance pickle, driven by a ``cbba_sota.dyn.perturb.Realization`` (its
    release times, realized durations and delay calendars replace the pilot's own draws)."""
    if np.isfinite(np.asarray(realization.fail_onset, float)).any():
        raise NotImplementedError("the pilot env has no failures (D6); F3 needs DynTaskEnvX")
    P = _pilot_module()
    env = load_env(path)
    env.__class__ = P.DynTaskEnv
    T, A = len(env.task_dic), len(env.agent_dic)
    if len(realization.release) != T or len(realization.delays) != A:
        raise ValueError("realization does not match the instance")
    env.sc = P.Scenario(name=realization.cell.name, obs=obs)
    env.dur_nom = np.array([float(env.task_dic[j]["time"]) for j in range(T)])
    if not np.allclose(env.dur_nom, realization.dur_nom):
        raise ValueError("nominal durations differ between the pickle and the realization")
    env.dur_real = np.asarray(realization.dur_real, float).copy()
    for j in range(T):
        env.task_dic[j]["time"] = float(env.dur_real[j])
    env.release = np.asarray(realization.release, float).copy()
    env.delays = [(np.asarray(s, float), np.asarray(e, float)) for s, e in realization.delays]
    env.realization = realization
    env.meta = {**realization.summary(), "obs": obs}
    env.init_state()
    return env


def make_world(setting: str, split: str, i: int, crn_seed: int, cell: str, *, world: str = "x",
               obs: str = "causal", coalition: str = "decision", **options) -> tuple[TaskEnv, str | None]:
    """(env, world label) for one dev/validation episode. ``world="x"``: ``DynTaskEnvX`` with the given coalition
    rule (label None: the env names itself after the run); ``world="pilot"``: the day-1 prototype."""
    from cbba_sota.bench import configs
    from cbba_sota.dyn import perturb

    if split == "test":
        raise PermissionError("the test split is frozen until the Track D prereg")
    rz = perturb.realize_instance(setting, split, i, crn_seed, cell)
    path = configs.get(setting).instance_path(split, i)
    if world == "pilot":
        return pilot_world(path, rz, obs=obs), pilot_world_name()
    if world == "x":
        from cbba_sota.dyn.env import make_env

        return make_env(path, rz, obs=obs, coalition=coalition, **options), None
    raise ValueError(f"unknown world {world!r}")
