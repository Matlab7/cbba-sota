"""B2: RL-MPC, the released HeteroMRTA policy used as a rollout planner (docs/trackD-spec.md Section 5.2).

At every structural event (the same trigger rule as SPARC, spec 4.2) the causal state is cloned into a *nominal
model*, the released policy is rolled out many times in that model (RL(s.N), GPU lockstep), the best rollout is
converted into key-ordered routes, and the routes are executed under Section 4.3 by the shared plan-following
controller (``controller.PlanController`` in ``DynTaskEnvX.run_plan``). RL(s.N) over *realized* episodes is not a
legal online policy and is not used anywhere.

Nominal model (``NominalClone``, a ``TaskEnv`` subclass: the policy runs in the env it was trained in, native
decision-order coalitions and native observations; ``third_party`` is not edited). Built only from what a planner may
know (``DynTaskEnvX.state()``, the causal ``StateView``, plus the controller's belief ``PlanState``):
- tasks: released tasks only. Unreleased tasks are all-zero rows and done (``hidden``: masked, never block
  termination), as D1 shows them to B1. Finished tasks are finished rows; started tasks are in progress with their
  observed start and a nominal finish ``max(start + nominal duration, now)``; the other released tasks are open, with
  the robots that physically departed to them as members (a covered one starts at its members' predicted arrival).
- durations are nominal; there are no further releases, stalls, duration noise or failures.
- robots: known-failed robots are removed from the team (their agent rows are deleted from every observation, as
  B1 does; never polled). Every other robot starts where the belief puts it: travelling robots arrive at their
  predicted arrival (``PlanState.ready``: nominal remainder x kappa), waiting and working robots stay at their task,
  idle robots and robots at home are free now, robots heading home are free when they arrive; a robot released
  from its task (it will leave on arrival) is free at its arrival there. A failed robot whose failure is not
  detected yet is believed alive (causal).
- predictors (spec 4.1, shared by all methods): travel times are nominal x kappa (the clone's robots move at
  ``speed / kappa``), residual durations are nominal.
- commitments (spec 4.1, the shared plan-following rule): a committed task keeps its frozen coalition. A frozen
  member that has not departed yet goes there first (in key order) as soon as it is free (a *forced* step, not a
  policy decision); while the frozen members cover a committed task, it is masked for everyone else (reserved).
  A committed task whose frozen members do not cover it (residual, after a failure or abandon) stays open to the
  policy, which may add members.

Rollouts (``lockstep``): the transcription of ``Worker.run_episode(training=False, max_waiting=False)`` /
``cbba_sota.solvers.rl._episode`` (released robots shuffled by ``random.Random(seed)``, blocked robots re-polled,
one policy call per decision), plus the forced steps above; the forward passes of all rollouts in a batch are
batched (one forward per round, on ``device``). Candidate 0 is the greedy decode (``greedy=True``, a declared
strong-baseline choice); the others sample (RL(s.N)). The best rollout is the one that finishes every released task
with the lowest nominal makespan (the rollout's end time, every robot home).

Budget (deterministic, spec 3.6): ``decisions`` policy decisions per event, spent in lockstep batches of ``batch``
rollouts (the first batch always runs; a batch that is started runs to completion; at most ``n_max`` rollouts).
Heavy tier: 8 cores x 1 s per event = 8 CPU seconds at the measured CPU per decision (the GPU forward passes come
on top: "8 cores *and* a GPU" is the generous reading), per setting in ``HEAVY_DECISIONS`` (4450-5990 decisions,
about 15-40 rollouts of a full 50-task episode, many more late in an episode, at most ``n_max``); the calibration is
provisional until the quiet-core measurement (gate K8).

Plan conversion (``rollout_plan``): each released, unstarted task gets the coalition that started it in the best
rollout, pruned to a minimal cover (latest arrival first, as ``Plan.prune_to_minimal``; robots that are physically
committed to it, travelling or waiting there, are never pruned), and its key is its start rank in the rollout. Start
times of consecutive visits of one robot are increasing, so the keys are one consistent order (G1's precondition)
and every robot's route begins with the task it is physically committed to. If no rollout finishes every released
task (the policy livelocks in the model), the plan is SPARC's insertion floor warm-started from the previous plan
(``RHPlanner`` with 0 iterations; counted in ``stats['fallback']``).

Pinned code: the rollout loop and sampling rule follow ``cbba_sota/solvers/rl.py`` at commit **4cf5e04**
(``git diff 4cf5e04 -- cbba_sota/solvers/rl.py`` is empty at the time of writing; ``_as_tensors`` is imported from
it read-only, the policy loader and ``_choose`` from ``cbba_sota.dyn.baselines.rl_online``, a pinned copy).
"""
from __future__ import annotations

import copy
import os
import pickle
import random
import time
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch

from cbba_sota.bench.configs import MAX_TIME
from cbba_sota.bench.heteromrta import TaskEnv, slim
from cbba_sota.dyn.controller import PlanController
from cbba_sota.dyn.executor import STRUCTURAL
from cbba_sota.dyn.planner import DynPlan, PlanState, RHPlanner
from cbba_sota.dyn.sparc import RollingPolicy

__all__ = [
    "HEAVY_DECISIONS",
    "HEAVY_DECISIONS_DEFAULT",
    "PINNED_COMMIT",
    "RLMPC",
    "NominalClone",
    "RLMPCConfig",
    "RLMPCController",
    "build_clone",
    "lockstep",
    "make_controller",
    "nominal_base",
    "rollout_plan",
]

PINNED_COMMIT = "4cf5e04"
# Heavy tier budget in policy decisions per event: 8 CPU-s / CPU per decision (lockstep batch 16, forward passes on
# an H100). Calibrated 2026-09-28 on the loaded shared host (``scripts/trackD_baselines.py calibrate-b2``: 1.375,
# 1.796, 1.336, 1.418 ms per decision on 6 mid-episode clones of one dev episode per setting; pooled 1.464 ms);
# provisional until gate K8. Other settings: the pooled value.
HEAVY_DECISIONS = {"MA-AT-25-5-50": 5820, "MA-AT-50-5-50": 4450, "SA-AT-50-5-50": 5990, "SA-BT-50-5-50": 5640}
HEAVY_DECISIONS_DEFAULT = 5460
_TASK_KEYS = ("rng", "per_species_range", "species_range", "tasks_range", "max_task_size", "duration_scale",
              "plot_figure", "traits_dim", "decision_dim", "task_dic", "agent_dic", "depot_dic", "species_dict",
              "species_distance_matrix", "species_neighbor_matrix", "tasks_num", "agents_num", "species_num",
              "coalition_matrix", "current_time", "dt", "max_waiting_time", "depot_waiting_time", "finished",
              "reactive_planning")


# ---------------------------------------------------------------------------------------------------------------
# nominal model
# ---------------------------------------------------------------------------------------------------------------
class NominalClone(TaskEnv):
    """``TaskEnv`` started from a causal snapshot (see the module docstring). Native rules everywhere except:
    hidden task rows are zero; reserved committed tasks are masked; robots with a pending start (``pending``: robot ->
    time it becomes free) are polled at that time; known-failed robots (``gone``) never decide.

    Speed (``fast = True``): the observation builders, ``task_update``, ``agent_update`` and ``get_arrival_time`` are
    re-implemented with the same float operations as the native code (distances precomputed per location with the
    native ``np.linalg.norm(a - b)``, loop-invariant ``get_matrix`` calls hoisted, the arrival at the last route entry
    read directly); ``fast = False`` runs the native methods. Rollouts are bit-identical either way (tests)."""

    fast = True
    hidden: np.ndarray
    gone: np.ndarray
    pending: dict
    forced: dict
    reserve: dict
    t0: float

    # ---- precomputation ----------------------------------------------------------------------------------------
    def precompute(self) -> None:
        """Distances between every location an agent can be at (tasks, then depots) and every task / depot."""
        tasks = [self.task_dic[j] for j in range(len(self.task_dic))]
        depots = [self.depot_dic[s] for s in range(len(self.depot_dic))]
        locs = [t["location"] for t in tasks] + [d["location"] for d in depots]
        self._tloc = np.array([t["location"] for t in tasks], dtype=float)
        self._D = np.array([[np.linalg.norm(p - t["location"]) for t in tasks] for p in locs])
        self._E = np.array([[np.linalg.norm(p - d["location"]) for d in depots] for p in locs])
        self._req = [np.asarray(t["requirements"]) for t in tasks]

    def _row(self, agent) -> int | None:
        """Row of ``agent``'s location in the distance tables (None: not a task or its depot; native fallback)."""
        c = agent["current_task"]
        loc = self.task_dic[c]["location"] if c >= 0 else self.depot_dic[agent["species"]]["location"]
        if agent["location"] is loc or np.array_equal(agent["location"], loc):
            return c if c >= 0 else len(self.task_dic) + agent["species"]
        return None

    # ---- observations --------------------------------------------------------------------------------------------
    def get_current_task_status(self, agent):
        row = self._row(agent) if self.fast else None
        if row is None:
            st = super().get_current_task_status(agent)
        else:
            tasks = self.task_dic.values()
            K = self.traits_dim
            T = len(self.task_dic)
            st = np.empty((T + 1, 2 * K + 5))
            v = agent["velocity"]
            st[0, :K] = 0.0
            st[0, K:2 * K] = -1.0
            st[0, 2 * K] = 0.0
            st[0, 2 * K + 1] = self._E[row, agent["species"]] / v
            st[0, 2 * K + 2:2 * K + 4] = agent["location"] - agent["depot"]
            st[0, 2 * K + 4] = 1.0
            st[1:, :K] = [t["status"] for t in tasks]
            st[1:, K:2 * K] = [t["requirements"] for t in tasks]
            st[1:, 2 * K] = [t["time"] for t in tasks]
            st[1:, 2 * K + 1] = self._D[row] / v
            st[1:, 2 * K + 2:2 * K + 4] = agent["location"] - self._tloc
            st[1:, 2 * K + 4] = [t["feasible_assignment"] for t in tasks]
        if self.hidden.any():
            st[1:][self.hidden] = 0.0
        return st

    def get_current_agent_status(self, agent):
        if not self.fast:
            return super().get_current_agent_status(agent)
        now = self.current_time
        agents = list(self.agent_dic.values())
        A, K = len(agents), self.traits_dim
        trav, rem, wait = np.zeros(A), np.zeros(A), np.zeros(A)
        for n, a in enumerate(agents):
            c = a["current_task"]
            if c >= 0:
                task = self.task_dic[c]
                arrival = self.get_arrival_time(a["ID"], c)
                trav[n] = arrival - now
                if now <= task["time_start"]:
                    wait[n] = now - arrival
                    rem[n] = task["time_start"] + task["time"] - now
        st = np.empty((A, K + 6))
        st[:, :K] = [a["abilities"] for a in agents]
        st[:, K] = np.clip(trav, a_min=0, a_max=None)
        st[:, K + 1] = np.clip(rem, a_min=0, a_max=None)
        st[:, K + 2] = np.clip(wait, a_min=0, a_max=None)
        st[:, K + 3:K + 5] = [agent["location"] - a["location"] for a in agents]
        st[:, K + 5] = [a["assigned"] for a in agents]
        return st

    def get_unfinished_tasks(self):
        out = super().get_unfinished_tasks()
        for j, n in self.reserve.items():
            if n > 0:
                out[j] = False
        return out

    def get_arrival_time(self, agent_id, task_id):
        a = self.agent_dic[agent_id]
        if self.fast and a["route"][-1] == task_id:
            return float(a["arrival_time"][-1])
        return super().get_arrival_time(agent_id, task_id)

    # ---- state updates -------------------------------------------------------------------------------------------
    def task_update(self):
        if not self.fast:
            return super().task_update()
        f_task = []
        now, L = self.current_time, self.max_waiting_time
        for task in self.task_dic.values():  # native code (TaskEnv.task_update)
            if not task["feasible_assignment"]:
                abilities = self.get_abilities(task["members"])
                arrival = np.array([self.get_arrival_time(member, task["ID"]) for member in task["members"]])
                task["status"] = task["requirements"] - abilities
                if (task["status"] <= 0).all():
                    if np.max(arrival) - np.min(arrival) <= L:
                        task["time_start"] = float(np.max(arrival))
                        task["time_finish"] = float(np.max(arrival) + task["time"])
                        task["feasible_assignment"] = True
                        f_task.append(task["ID"])
                    else:
                        task["feasible_assignment"] = False
                        infeasible_members = arrival <= np.max(arrival) - L
                        for member in np.array(task["members"])[infeasible_members]:
                            task["members"].remove(member)
                            task["abandoned_agent"].append(member)
                else:
                    task["feasible_assignment"] = False
                    for member in np.array(task["members"]):
                        if now - self.get_arrival_time(member, task["ID"]) >= L:
                            task["members"].remove(member)
                            task["abandoned_agent"].append(member)
            elif now >= task["time_finish"]:
                task["finished"] = True
        if all(t["feasible_assignment"] for t in self.task_dic.values()):  # hoisted (pure, loop-invariant)
            for depot in self.depot_dic.values():
                for member in depot["members"]:
                    if now >= self.get_arrival_time(member, depot["ID"]):
                        self.agent_dic[member]["returned"] = True
        return f_task

    def agent_update(self):
        if not self.fast:
            super().agent_update()
        else:
            all_feasible = all(t["feasible_assignment"] for t in self.task_dic.values())  # hoisted
            L = self.max_waiting_time
            for agent in self.agent_dic.values():  # native code (TaskEnv.agent_update)
                if agent["current_task"] < 0:
                    if all_feasible:
                        agent["next_decision"] = np.nan
                    elif not np.isnan(agent["next_decision"]):
                        agent["next_decision"] = np.inf
                else:
                    current_task = self.task_dic[agent["current_task"]]
                    if current_task["feasible_assignment"]:
                        if agent["ID"] in current_task["members"]:
                            agent["next_decision"] = float(current_task["time_finish"])
                            if self.current_time >= float(current_task["time_start"]):
                                agent["assigned"] = True
                        else:
                            agent["next_decision"] = self.get_arrival_time(agent["ID"], current_task["ID"]) + L
                            agent["assigned"] = False
                    else:
                        agent["next_decision"] = self.get_arrival_time(agent["ID"], current_task["ID"]) + L
                        agent["assigned"] = False
        for i, t in self.pending.items():
            self.agent_dic[i]["next_decision"] = t
        for i in np.flatnonzero(self.gone):
            self.agent_dic[int(i)]["next_decision"] = np.nan

    # ---- commitments ---------------------------------------------------------------------------------------------
    def next_forced(self, i: int) -> int | None:
        """Next committed task robot ``i`` must go to (0-based), skipping tasks already covered or finished."""
        q = self.forced.get(i)
        while q:
            j = q.pop(0)
            self.reserve[j] -= 1
            task = self.task_dic[j]
            if not task["finished"] and not task["feasible_assignment"]:
                return j
        return None

    def visible_done(self) -> bool:
        return all(t["finished"] for t in self.task_dic.values())


def nominal_base(env) -> bytes:
    """Pickled static ``TaskEnv`` of ``env``'s instance with nominal durations (``env``: a ``DynTaskEnvX`` or any
    ``TaskEnv``), the starting point of every clone."""
    base = copy.deepcopy(env)
    for k in list(vars(base)):
        if k not in _TASK_KEYS:
            delattr(base, k)
    base.__class__ = TaskEnv
    dur = getattr(env, "dur_nom", None)
    for j, task in base.task_dic.items():
        if dur is not None:
            task["time"] = float(dur[j])
        for k in ("coalition", "wasted", "not_before", "aborted", "restarts"):
            task.pop(k, None)
    for a in base.agent_dic.values():
        a.pop("leg", None)
    slim(base)
    base.init_state()
    return pickle.dumps(base, protocol=pickle.HIGHEST_PROTOCOL)


def build_clone(base: bytes, state: PlanState, view, kappa: float = 1.0) -> NominalClone:
    """The nominal model at ``view.t`` (``view``: ``DynTaskEnvX.state()``; ``state``: the controller's belief)."""
    env = pickle.loads(base)
    env.__class__ = NominalClone
    env.init_state()
    now = float(view.t)
    T, A = len(env.task_dic), len(env.agent_dic)
    if state.now != now:
        raise ValueError(f"belief at {state.now} but view at {now}")
    env.current_time = now
    env.t0 = now
    env.hidden = ~np.asarray(view.released, bool)
    env.gone = np.array([m == "failed" for m in view.mode], bool)
    env.pending, env.forced, env.reserve = {}, {}, {}
    env.precompute()
    dur = np.asarray(view.dur_nom, float)
    ab = np.array([env.agent_dic[i]["abilities"] for i in range(A)], float)
    for a in env.agent_dic.values():
        a["velocity"] = float(a["velocity"]) / float(kappa)
    # robots
    phys: list[list[int]] = [[] for _ in range(T)]  # robots physically committed (travelling / waiting / working)
    for i in range(A):
        a = env.agent_dic[i]
        depot_id = -a["species"] - 1
        m, c = view.mode[i], int(view.target[i])
        if m in ("failed", "home", "to_home"):
            arr = float(state.ready[i]) if m == "to_home" else now
            a.update(route=[depot_id], arrival_time=[max(arr, now)], current_task=-1, location=a["depot"])
            if m == "failed":
                a.update(next_decision=np.nan, no_choice=True)
            else:
                env.pending[i] = max(arr, now)
            continue
        loc = env.task_dic[c]["location"]
        if m == "travel":
            arr = max(float(state.ready[i]), now)
        else:
            arr = min(float(view.eta[i]), now)
        a.update(route=[depot_id, c], arrival_time=[0.0, arr], current_task=c, location=loc)
        if view.finished[c]:  # idle at a finished task: a member of it if it worked there (native layout)
            member = False
        elif view.started[c]:
            member = m == "work"
        else:
            member = m in ("travel", "wait") and i in view.committed[c]
        if member:
            phys[c].append(i)
        else:  # idle, released from its task, or heading to a task that started without it: free on arrival
            env.pending[i] = max(arr, now)
    # tasks
    for j in range(T):
        task = env.task_dic[j]
        req = np.asarray(task["requirements"])
        if env.hidden[j]:
            task.update(feasible_assignment=True, finished=True, status=req * 0, members=[])
        elif view.finished[j]:
            coal = list(view.committed[j])
            task.update(feasible_assignment=True, finished=True, members=[i for i in phys[j]],
                        status=req - (ab[coal].sum(axis=0) if coal else 0), time_start=float(view.start[j])
                        if np.isfinite(view.start[j]) else now, time_finish=now)
        elif view.started[j]:
            mem = list(phys[j])
            task.update(feasible_assignment=True, finished=False, members=mem,
                        status=req - (ab[mem].sum(axis=0) if mem else 0), time_start=float(view.start[j]),
                        time_finish=max(float(view.start[j]) + float(dur[j]), now))
        else:
            task.update(members=list(phys[j]))
    for j in range(T):  # idle robots at a finished task they worked on are its members (native layout)
        if view.finished[j]:
            for i in range(A):
                if view.mode[i] == "idle" and int(view.target[i]) == j and i in view.committed[j]:
                    env.task_dic[j]["members"].append(i)
    env.task_update()  # covers and predicted starts of the open tasks with physical members
    # commitments: frozen members that have not departed go first, in key order; covered committed tasks reserved
    live = state.released & ~state.done & ~state.started
    com = [j for j in np.flatnonzero(state.committed & live)]
    com.sort(key=lambda j: (float(state.keys[j]), int(j)))
    for j in com:
        frozen = [int(i) for i in state.members[j] if state.alive[i] and not env.gone[i]]
        extra = [i for i in frozen if i not in phys[j]]
        for i in extra:
            env.forced.setdefault(i, []).append(int(j))
        if extra:  # reserved (masked for the others) while the frozen members still to come complete the cover
            covered = bool((ab[frozen].sum(axis=0) >= np.asarray(env.task_dic[j]["requirements"])).all())
            env.reserve[int(j)] = len(extra) if covered else 0
    env.agent_update()
    env.finished = False
    return env


# ---------------------------------------------------------------------------------------------------------------
# rollouts
# ---------------------------------------------------------------------------------------------------------------
def _drop_gone(env: NominalClone, agent_id: int, agent_obs: np.ndarray) -> tuple[int, np.ndarray]:
    if not env.gone.any():
        return agent_id, agent_obs
    keep = ~env.gone
    return int(keep[:agent_id].sum()), agent_obs[:, keep]


def _episode(env: NominalClone, rng: random.Random, max_time: float):
    """Yield (index, task obs, agent obs, mask) per policy decision; receive the action (0 = depot, j = task j-1).
    ``Worker.run_episode`` / ``rl._episode`` plus forced commitment steps and deleted rows of failed robots."""
    tasks = env.task_dic.values()
    step = 0
    while not env.finished and env.current_time < max_time:
        released, t = env.next_decision()
        env.current_time = max(t, env.current_time)
        rng.shuffle(released[0])
        for agent_id in released[0] + released[1]:
            if env.gone[agent_id]:
                continue
            if env.pending.pop(agent_id, None) is not None:
                env.agent_update()  # its start is taken: back to the native next_decision (idempotent update)
            j = env.next_forced(agent_id)
            if j is not None:
                env.agent_step(agent_id, j + 1, step)
                continue
            task_obs, agent_obs, mask = env.agent_observe(agent_id, False)
            if mask[0, 1:].all():
                if not all(t["feasible_assignment"] for t in tasks):
                    env.agent_dic[agent_id]["no_choice"] = True
                    continue
                if env.agent_dic[agent_id]["current_task"] < 0:
                    continue
            idx, agent_obs = _drop_gone(env, agent_id, agent_obs)
            action = yield idx, task_obs, agent_obs, mask
            env.agent_step(agent_id, action, step)
        env.finished = env.check_finished()
        step += 1


@dataclass
class Rollout:
    k: int
    seed: int
    sample: bool
    makespan: float
    complete: bool  # every visible task finished before the cap
    decisions: int
    env: NominalClone | None = None


@torch.no_grad()
def lockstep(clone: bytes, net: torch.nn.Module, specs: Sequence[tuple[int, int, bool]], *, device="cpu",
             max_time: float = MAX_TIME, keep: bool = True) -> list[Rollout]:
    """Run rollouts ``specs`` = [(k, seed, sample)] of one pickled clone, batching their forward passes."""
    from cbba_sota.dyn.baselines.rl_online import _choose
    from cbba_sota.solvers.rl import _as_tensors

    envs, eps, pending, count = [], [], {}, []
    for n, (_, seed, _) in enumerate(specs):
        env = pickle.loads(clone)
        envs.append(env)
        eps.append(_episode(env, random.Random(seed), max_time))
        count.append(0)
        try:
            pending[n] = next(eps[n])
        except StopIteration:
            pass
    sampled = [n for n, s in enumerate(specs) if s[2]]
    gen = torch.Generator(device).manual_seed(int(specs[sampled[0]][1]) if sampled else 0)
    while pending:
        ns = list(pending)
        probs, _ = net(*_as_tensors([pending[n] for n in ns], device))
        acts = torch.argmax(probs, dim=1)
        rows = [r for r, n in enumerate(ns) if specs[n][2]]
        if rows:
            acts[rows] = _choose(probs[rows], True, gen)
        for n, a in zip(ns, acts.tolist()):
            count[n] += 1
            try:
                pending[n] = eps[n].send(int(a))
            except StopIteration:
                del pending[n]
    out = []
    for n, (k, seed, sample) in enumerate(specs):
        env = envs[n]
        ms = float(env.current_time)
        out.append(Rollout(k, int(seed), bool(sample), ms, bool(env.visible_done() and ms < max_time), count[n],
                           env if keep else None))
    return out


def sample_seed(seed: int, k: int) -> int:
    """Seed of rollout ``k`` of an event whose planner seed is ``seed`` (``rl.sample_seed`` rule)."""
    return int(np.random.SeedSequence([int(seed), int(k)]).generate_state(1)[0])


# ---------------------------------------------------------------------------------------------------------------
# rollout -> plan
# ---------------------------------------------------------------------------------------------------------------
def rollout_plan(env: NominalClone, state: PlanState, phys: Sequence[Sequence[int]] | None = None) -> DynPlan:
    """Key-ordered plan of the released, unstarted tasks from a finished rollout (see the module docstring)."""
    inst = state.inst
    ab, req = inst.ab, inst.req
    T = inst.n_tasks
    live = state.released & ~state.done & ~state.started
    members: list[tuple[int, ...]] = [()] * T
    start = np.full(T, np.nan)
    unplaced = []
    for j in np.flatnonzero(live):
        task = env.task_dic[int(j)]
        if not task["feasible_assignment"] or not np.isfinite(task["time_start"]):
            unplaced.append(int(j))
            continue
        m = [int(i) for i in task["members"]]
        keep = set(phys[j]) if phys is not None else set()
        arr = {i: env.get_arrival_time(i, int(j)) for i in m}
        total = ab[m].sum(axis=0)
        for i in sorted(m, key=lambda i: (arr[i], i), reverse=True):
            if i not in keep and (total - ab[i] >= req[j]).all():
                total = total - ab[i]
                m.remove(i)
        members[int(j)] = tuple(sorted(m))
        start[j] = float(task["time_start"])
    planned = [j for j in range(T) if members[j]]
    keys = np.full(T, np.nan)
    base = float(state.key_floor) + 1.0 if np.isfinite(state.key_floor) else 0.0
    for rank, j in enumerate(sorted(planned, key=lambda j: (start[j], float(env.task_dic[j]["time_finish"]), j))):
        keys[j] = base + rank
    return DynPlan(members=members, keys=keys, makespan=float(env.current_time), unplaced=unplaced,
                   n_open=int(live.sum()), n_anchored=int((state.committed & live).sum()))


# ---------------------------------------------------------------------------------------------------------------
# the rolling policy
# ---------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class RLMPCConfig:
    decisions: int = HEAVY_DECISIONS_DEFAULT  # policy decisions per event (deterministic budget)
    decisions0: int | None = None  # first plan (None: the same)
    batch: int = 16  # rollouts per lockstep batch
    n_max: int = 256  # rollouts per event at most
    greedy: bool = True  # candidate 0 is the greedy decode
    device: str = "cpu"
    max_time: float = MAX_TIME


_NETS: dict[str, torch.nn.Module] = {}


def _net(device: str) -> torch.nn.Module:
    if device not in _NETS:
        from cbba_sota.dyn.baselines.rl_online import load_policy

        _NETS[device] = load_policy(device)
    return _NETS[device]


class RLMPC(RollingPolicy):
    """B2 (see the module docstring). ``view`` (the env's causal ``StateView`` at the planning time) is set by
    ``RLMPCController`` before each call."""

    name = "RL-MPC"

    def __init__(self, base: bytes, config: RLMPCConfig | None = None, seed_base: int = 0, kappa: float = 1.0,
                 net: torch.nn.Module | None = None, triggers: frozenset[str] = STRUCTURAL):
        super().__init__(triggers, seed_base)
        self.base, self.cfg, self.kappa = base, config or RLMPCConfig(), float(kappa)
        self.net = net if net is not None else _net(self.cfg.device)
        self.floor = RHPlanner()
        self.view = None
        self.stats: list[dict] = []

    def solve(self, state: PlanState, scope, seed: int, first: bool) -> DynPlan:
        c0 = time.process_time()
        if scope is not None and not np.asarray(scope, bool)[np.asarray(state.alive, bool)].all():
            raise NotImplementedError("RL-MPC plans for every live robot (good communication only)")
        view = self.view
        if view is None or float(view.t) != float(state.now):
            raise RuntimeError("RL-MPC needs the env's causal StateView at the planning time (RLMPCController)")
        cfg = self.cfg
        clone = build_clone(self.base, state, view, self.kappa)
        phys = [list(clone.task_dic[j]["members"]) if not (clone.hidden[j] or clone.task_dic[j]["finished"]
                                                          or view.started[j]) else [] for j in range(len(view.released))]
        blob = pickle.dumps(clone, protocol=pickle.HIGHEST_PROTOCOL)
        budget = cfg.decisions0 if first and cfg.decisions0 is not None else cfg.decisions
        best: Rollout | None = None
        used, n, n_complete = 0, 0, 0
        while n < cfg.n_max and (n == 0 or used < budget):
            size = min(cfg.batch, cfg.n_max - n)
            specs = [(n + b, sample_seed(seed, n + b), not (cfg.greedy and n + b == 0)) for b in range(size)]
            for r in lockstep(blob, self.net, specs, device=cfg.device, max_time=cfg.max_time):
                used += r.decisions
                n_complete += r.complete
                if best is None or (not r.complete, r.makespan, r.k) < (not best.complete, best.makespan, best.k):
                    best = r
            n += size
        assert best is not None
        fallback = not best.complete
        if fallback:
            plan = self.floor.plan(state, self.incumbent, None, seed, iters=0)
        else:
            plan = rollout_plan(best.env, state, phys)
            if plan.unplaced:  # cannot happen for a complete rollout; kept as a guard
                fallback = True
                plan = self.floor.plan(state, self.incumbent, None, seed, iters=0)
        plan.seed, plan.iterations = int(seed), int(n)
        plan.cpu_s = time.process_time() - c0
        self.stats.append({"t": float(state.now), "rollouts": n, "decisions": used, "complete": n_complete,
                           "best_k": best.k, "best_ms": best.makespan, "fallback": fallback})
        return plan


class RLMPCController(PlanController):
    """``PlanController`` that also hands the env's causal snapshot to the policy (the clone needs where every robot
    is and what it is doing; ``PlanState`` alone does not say whether a robot is waiting, working or idle)."""

    def belief(self, env, sv):
        self.policy.view = sv
        return super().belief(env, sv)


def make_controller(env, realization, tier: str, seed: int, *, decisions: int | None = None,
                    decisions0: int | None = None, batch: int = 16, n_max: int = 256, greedy: bool = True,
                    device: str = "auto", seed_base: int = 0, scale: float = 1.0) -> RLMPCController:
    """``scripts/trackD_run.py`` factory (``ctrl:cbba_sota.dyn.baselines.rl_mpc:make_controller``). ``tier``: heavy
    (``HEAVY_DECISIONS`` of the setting) unless ``decisions`` is given; ``device``: "auto" = one of cuda:0 / cuda:1 by process id
    (identical GPUs), else "cpu" when CUDA is absent. ``seed`` (the runner's) is recorded but unused: planner seeds
    come from the belief digest (spec 4.1)."""
    if decisions is None:
        if tier != "heavy":
            raise ValueError("RL-MPC is a Heavy-tier method; pass decisions=... for another budget")
        from cbba_sota.dyn.baselines.cpsat_tiers import setting_of

        decisions = HEAVY_DECISIONS.get(setting_of(env.nominal_instance()), HEAVY_DECISIONS_DEFAULT)
    decisions = round(float(scale) * decisions)  # ``scale``: budget sensitivity (x the tier)
    if device == "auto":
        device = f"cuda:{os.getpid() % 2}" if torch.cuda.is_available() else "cpu"
    cfg = RLMPCConfig(decisions=int(decisions), decisions0=decisions0, batch=int(batch), n_max=int(n_max),
                      greedy=bool(greedy), device=device)
    policy = RLMPC(nominal_base(env), cfg, seed_base=seed_base, kappa=realization.kappa())
    return RLMPCController(policy, env.nominal_instance(), realization.kappa())
