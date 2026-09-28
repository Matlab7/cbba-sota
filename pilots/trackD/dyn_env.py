"""Track D prototype: a dynamic HeteroMRTA env (online task release + execution noise) with minimal changes.

``DynTaskEnv`` subclasses ``env.task_env.TaskEnv`` (marmotlab/HeteroMRTA @ db51e29, numpy-2 patch) and overrides
seven methods; everything else (coalition semantics, masks, waiting, rewards) is the original code:

- ``get_unfinished_tasks``      unreleased tasks are not selectable (so they are masked for RL and greedy).
- ``get_current_task_status``   unreleased task rows are all-zero. The released network treats an all-zero row as
                                padding (``attention.get_attn_pad_mask``), so a zero row is exactly an absent task
                                (checked numerically in ``check_equivalence.py``); causal mode shows nominal durations.
- ``get_current_agent_status``  ``obs="oracle"``: native rows (realized future arrivals/starts leak to the policy);
                                ``obs="causal"``: the same features from nominal predictions until an event happens.
- ``agent_step``                realized arrival through a per-agent LoRR-2026 delay calendar; departure backdating
                                (the env's "flashforward" of blocked agents) is capped at the latest release epoch.
- ``task_update``               unchanged code path; ``task['time']`` holds the REALIZED duration (nominal kept in
                                ``dur_nom``), so task finish times are realized.
- ``next_decision``             release epochs are decision epochs (blocked/idle agents are re-polled); with no
                                release pending the original method runs unchanged.
- ``init_state``                also resets the dynamic bookkeeping.

Noise is common-random-number (CRN) by construction: durations are keyed by (noise seed, instance, task), the delay
calendar by (noise seed, instance, agent) in absolute time, releases by (release seed, instance, task). Any method
executed on the same (instance, seeds) faces the same realization. With all releases at 0 and no noise the env is
bit-identical to the static one (``check_equivalence.py``).

Delay model = LoRR 2026 Start-Kit ``DelayGenerator`` (bernoulli event model, uniform integer durations): every tick,
each agent that is not delayed starts a delay with probability ``p_delay``, lasting U{min_delay..max_delay} ticks;
a delayed agent does not move. One tick = the env's own ``dt`` = 0.1 time units. Delays only matter while travelling
(waiting/working agents lose nothing), exactly as a stalled LoRR agent that has nothing to execute.
"""
from __future__ import annotations

import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SNAP = Path(__file__).resolve().parent / "_snap"  # frozen copy of cbba_sota (the package is edited concurrently)
sys.path.insert(0, str(SNAP if (SNAP / "cbba_sota").exists() else ROOT))
from cbba_sota.bench.heteromrta import TaskEnv, load_env  # noqa: E402  (adds third_party/HeteroMRTA to sys.path)

TICK = 0.1  # env.dt


@dataclass(frozen=True)
class Scenario:
    name: str = "static"
    # --- release process ---
    release: str = "none"  # none | batch | poisson
    batch_first: int = 20  # batch: tasks with id <= batch_first visible at 0 (HeteroMRTA reactive_planning demo)
    batch_size: int = 20  # ... then batch_size more every batch_period (demo: 20 every 10)
    batch_period: float = 10.0
    dod: float = 0.5  # poisson: degree of dynamism = fraction of tasks NOT known at time 0
    horizon: float = 20.0  # poisson: arrivals of the dynamic tasks are a Poisson process conditioned on [0, horizon]
    # --- execution noise ---
    dur_sigma: float = 0.0  # lognormal sigma of mean-preserving multiplicative duration noise
    p_delay: float = 0.0  # LoRR pDelay per tick per available agent
    min_delay: int = 1  # LoRR minDelay (ticks)
    max_delay: int = 4  # LoRR maxDelay (ticks)
    # --- what the policy observes ---
    obs: str = "oracle"  # oracle | causal


def _rng(*keys: int) -> np.random.Generator:
    return np.random.default_rng([int(k) for k in keys])


def release_times(sc: Scenario, n_tasks: int, inst_key: int, release_seed: int) -> np.ndarray:
    if sc.release == "none":
        return np.zeros(n_tasks)
    if sc.release == "batch":  # plot_animation(reactive_planning=True): hidden iff id > clip(t//10*20+20, 20, 100)
        ids = np.arange(n_tasks)
        return np.where(ids <= sc.batch_first, 0.0,
                        np.ceil((ids - sc.batch_first) / sc.batch_size) * sc.batch_period)
    if sc.release == "poisson":
        rng = _rng(release_seed, inst_key, 3)
        n_dyn = int(round(sc.dod * n_tasks))
        rel = np.zeros(n_tasks)
        dyn = rng.choice(n_tasks, n_dyn, replace=False)
        rel[dyn] = np.sort(rng.uniform(0.0, sc.horizon, n_dyn))  # order statistics of a conditioned Poisson process
        return rel
    raise ValueError(sc.release)


def realized_durations(sc: Scenario, dur_nom: np.ndarray, inst_key: int, noise_seed: int) -> np.ndarray:
    if sc.dur_sigma <= 0:
        return dur_nom.copy()
    z = _rng(noise_seed, inst_key, 2).standard_normal(len(dur_nom))
    return dur_nom * np.exp(sc.dur_sigma * z - 0.5 * sc.dur_sigma ** 2)


def delay_calendar(sc: Scenario, n_agents: int, inst_key: int, noise_seed: int, t_max: float = 400.0):
    """Per agent, (starts, ends) arrays of delay intervals in env time, LoRR bernoulli/uniform semantics."""
    out = []
    n_ticks = int(round(t_max / TICK))
    for i in range(n_agents):
        if sc.p_delay <= 0:
            out.append((np.zeros(0), np.zeros(0)))
            continue
        rng = _rng(noise_seed, inst_key, 1, i)
        starts, ends, k = [], [], 0
        while True:
            k += int(rng.geometric(sc.p_delay)) - 1  # first success among the ticks where the agent is available
            if k >= n_ticks:
                break
            d = int(rng.integers(sc.min_delay, sc.max_delay + 1))
            starts.append(k * TICK)
            ends.append((k + d) * TICK)
            k += d
        out.append((np.array(starts), np.array(ends)))
    return out


class DynTaskEnv(TaskEnv):
    """See module docstring. Build with ``make_dyn_env``."""

    # ---- setup -------------------------------------------------------------------------------------------------
    def setup_dynamic(self, sc: Scenario, inst_key: int, noise_seed: int = 0, release_seed: int = 0) -> None:
        T, A = len(self.task_dic), len(self.agent_dic)
        self.sc = sc
        self.dur_nom = np.array([float(self.task_dic[j]["time"]) for j in range(T)])
        self.dur_real = realized_durations(sc, self.dur_nom, inst_key, noise_seed)
        for j in range(T):
            self.task_dic[j]["time"] = float(self.dur_real[j])  # the engine runs on realized durations
        self.release = release_times(sc, T, inst_key, release_seed)
        self.delays = delay_calendar(sc, A, inst_key, noise_seed)
        self.meta = {"inst_key": inst_key, "noise_seed": noise_seed, "release_seed": release_seed, **asdict(sc)}
        self.init_state()

    def init_state(self):
        super().init_state()
        for a in self.agent_dic.values():
            a["leg"] = (0.0, 0.0)  # (departure, nominal travel) of the current leg
        self.n_release_polls = 0

    # ---- release ----------------------------------------------------------------------------------------------
    def released_mask(self) -> np.ndarray:
        return self.release <= self.current_time + 1e-12

    def _info_time(self) -> float:
        seen = self.release[self.release <= self.current_time + 1e-12]
        return float(seen.max()) if seen.size else 0.0

    def get_unfinished_tasks(self):
        rel = self.released_mask()
        return [task["feasible_assignment"] is False and np.any(task["status"] > 0) and bool(rel[j])
                for j, task in self.task_dic.items()]

    def next_decision(self):
        pending = self.release[self.release > self.current_time + 1e-12]
        if pending.size == 0:
            return super().next_decision()  # original semantics once everything is released
        r = float(pending.min())
        dt = np.array(self.get_matrix(self.agent_dic, "next_decision"), dtype=float)
        nc = np.array(self.get_matrix(self.agent_dic, "no_choice"), dtype=bool)
        dt = np.where(nc, np.inf, dt)
        finite = dt[np.isfinite(dt)]
        t = min(float(finite.min()) if finite.size else np.inf, r)
        finished = np.flatnonzero(dt == t).tolist()
        blocked = [i for i in np.flatnonzero(np.isinf(dt)).tolist()
                   if t >= self.agent_dic[i]["arrival_time"][-1]]
        if t == r:
            self.n_release_polls += len(blocked)
        return (finished, blocked), t

    # ---- execution noise --------------------------------------------------------------------------------------
    def _arrive(self, agent_id: int, dep: float, tau: float) -> float:
        starts, ends = self.delays[agent_id]
        if starts.size == 0:
            return dep + tau  # same float op as the original env
        t, rem = dep, tau
        k = int(np.searchsorted(ends, t, side="right"))  # first interval ending after t
        while k < starts.size:
            s, e = starts[k], ends[k]
            if s <= t:  # stalled at t
                t = e
            elif t + rem <= s:
                break
            else:
                rem -= s - t
                t = e
            k += 1
        return t + rem

    def _stalled(self, agent_id: int, a: float, b: float) -> float:
        """Delay time inside [a, b] (the agent was travelling throughout)."""
        starts, ends = self.delays[agent_id]
        if starts.size == 0 or b <= a:
            return 0.0
        return float(np.clip(np.minimum(ends, b) - np.maximum(starts, a), 0.0, None).sum())

    def agent_step(self, agent_id, task_id, decision_step):
        task_id = task_id - 1
        agent = self.agent_dic[agent_id]
        if task_id != -1:
            task = self.task_dic[task_id]
            if task["feasible_assignment"]:
                return -1, False, []
            assert self.release[task_id] <= self.current_time + 1e-12, "policy chose an unreleased task"
        else:
            task = self.depot_dic[agent["species"]]
        agent["route"].append(task["ID"])
        previous_task = agent["current_task"]
        agent["current_task"] = task_id
        dist = self.calculate_eulidean_distance(agent, task)
        travel_time = dist / agent["velocity"]
        agent["travel_time"] = travel_time
        agent["travel_dist"] += dist
        prev = self.task_dic[previous_task] if previous_task >= 0 else None
        if prev is not None and prev["feasible_assignment"] and prev["time_finish"] < self.current_time:
            # original: depart at the previous finish ("flashforward" of blocked agents). Causal cap: not before
            # the latest release epoch (no-op when every task is released at 0).
            dep = max(prev["time_finish"], self._info_time())
        else:
            dep = self.current_time
        agent["arrival_time"] += [self._arrive(agent_id, dep, travel_time)]
        agent["leg"] = (dep, travel_time)
        agent["location"] = task["location"]
        agent["decision_step"] = decision_step
        agent["no_choice"] = False
        if agent_id not in task["members"]:
            task["members"].append(agent_id)
        f_t = self.task_update()
        self.agent_update()
        return 0, True, f_t

    # ---- observations -----------------------------------------------------------------------------------------
    def get_current_task_status(self, agent):
        st = super().get_current_task_status(agent)
        if self.sc.obs == "causal":
            st[1:, 2 * self.traits_dim] = self.dur_nom
        st[1:][~self.released_mask()] = 0.0
        return st

    def _eta_obs(self, a) -> float:
        """Observed arrival if it happened, else nominal ETA plus the stall time seen so far."""
        arr = a["arrival_time"][-1]
        now = self.current_time
        if arr <= now:
            return arr
        dep, tau = a["leg"]
        return max(dep + tau + self._stalled(a["ID"], dep, now), now)

    def get_current_agent_status(self, agent):
        if self.sc.obs == "oracle":
            return super().get_current_agent_status(agent)
        now = self.current_time
        rows = []
        for a in self.agent_dic.values():
            travel_time = remaining = waiting = 0.0
            if a["current_task"] >= 0:
                j = a["current_task"]
                task = self.task_dic[j]
                eta = self._eta_obs(a)
                travel_time = max(eta - now, 0.0)
                if not task["feasible_assignment"]:
                    start = 0.0  # native time_start of an uncovered task
                elif task["time_start"] <= now:
                    start = task["time_start"]  # observed
                else:
                    start = max(self._eta_obs(self.agent_dic[m]) for m in task["members"])  # predicted
                if now <= start:
                    waiting = max(now - eta, 0.0)
                    remaining = max(start + self.dur_nom[j] - now, 0.0)
            rows.append(np.hstack([a["abilities"], travel_time, remaining, waiting,
                                   agent["location"] - a["location"], a["assigned"]]))
        return np.vstack(rows)


def make_dyn_env(path_or_env, sc: Scenario, inst_key: int, noise_seed: int = 0, release_seed: int = 0) -> DynTaskEnv:
    env = load_env(path_or_env) if not isinstance(path_or_env, TaskEnv) else path_or_env
    env.__class__ = DynTaskEnv
    env.setup_dynamic(sc, inst_key, noise_seed, release_seed)
    return env
