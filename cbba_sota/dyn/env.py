"""DynTaskEnvX: the Track D ground truth (docs/trackD-spec.md Section 3), a subclass of HeteroMRTA's ``TaskEnv``.

``third_party/HeteroMRTA`` (marmotlab @ db51e29) is not edited: the class is swapped onto a loaded env
(``make_env``) and a handful of methods are overridden. Everything else (coalition bookkeeping, masks, travel
geometry, rewards) is the original code. Started from ``pilots/trackD/dyn_env.py`` (prototype of D1, D3, D4).

Declared semantic changes (Section 3.2; each has a regression test in ``tests/test_dyn_env.py``):

- D1 release masking: a task with ``release > t`` is not selectable and its observation row is all zero (the released
  network treats an all-zero row as padding, ``attention.get_attn_pad_mask``). Release epochs re-poll idle agents.
- D2 arrival-order coalitions (``coalition="arrival"``): among a task's committed members (in commit order), the
  task starts at the first arrival at which the members present cover it; members arriving later (late) or at the
  same instant but not needed (redundant) leave on arrival (a *wasted trip*) and are free again at that time.
  Members present before the start stay and work, as in the native env. ``coalition="decision"`` keeps the native
  decision-order rule (start at the last committed member's arrival; joining a covered task is refused). The two
  coincide for minimal covers, so plan replay is identical under both (tests), but NOT for the released RL policy
  on MA-AT settings: its coalitions are often non-minimal (a later member makes an earlier one redundant) and under
  arrival order the redundant member leaves early, which changes the whole rollout (e.g. RL(g.) on MA-AT-25-5-50
  dev 0: 44.54 native vs 39.66). The default ``coalition="auto"`` therefore resolves to "arrival" for plan following
  (``run_plan``) and to "decision" for policies and native loops (the RL's own env, bit-identical static
  reduction); ``world`` records the rule in force.
- D3 causal observations (``obs="causal"``, default): an agent's travel feature is its observed arrival if it has
  arrived, else ETA = departure + nominal travel + stall observed so far; durations are nominal until the task
  finishes; predicted starts use those ETAs. ``obs="oracle"`` keeps the native rows (realized future times leak).
- D4 departure back-dating (the native "flashforward" of a blocked agent to its previous finish, policy mode only) is
  capped at the latest information epoch (release or failure detection), applies only to an agent that worked on
  the previous task, and never applies to the trip home when H > 0 (D7 allows it only from the epoch at which every
  task is finished). Plan-following agents depart at the decision time. No-op when static.
- D5 causal lease: the native covered-branch rule that drops early arrivals from the *realized* arrival spread is
  removed. A waiting agent leaves an uncovered task only (i) after waiting ``lease_L`` (native 200, i.e. never
  before the time cap), (ii) at the failure detection of a partner when ``abandon_on_failure`` (the good-comms
  "lease = failure detector" of Section 4.3), or (iii) when a controller calls ``abandon``.
- D6 fail-stop failures: at onset the robot halts (a travelling robot never arrives: its calendar gets an infinite
  stall, so it looks stalled until detection), a task it was working on is aborted, and a task still waiting for
  it cannot start. At detection (onset + h) the robot leaves every coalition (``known_failed``), an aborted task is
  reset and restarts from scratch (full duration) when re-covered, and a structural ``failure`` event is emitted.
  Policies delete known-failed rows from the agent observation (``RLPolicy``; a zero row is the same for the network
  except row 0, where the released network returns NaN, so rows are deleted, not zeroed). Onsets and detections are
  processed inside ``next_decision``, so every driver of the native loop gets them.
- D7 termination: the release window end H is known, the task count is not. With H > 0 an idle agent returns to its
  depot only when t >= H and every released task is finished; the makespan is the time at which every task is
  finished and every live robot is home. With H = 0 the native end-of-episode rule runs unchanged.
- D8 with H > 0, an idle agent waits in place while it may not go home (it is re-polled at every epoch).

Two execution APIs (both drive the env's own decision loop: ``next_decision`` / ``agent_step`` /
``check_finished``):

- ``run_plan(controller, routes)``: plan following (the loop of ``TaskEnv.execute_by_route``). Each agent follows
  its route (0-based task ids) and departs when its previous task finishes. At every epoch the controller receives
  the events of that instant (``Event``; structural kinds: release, failure, abandon, idle, membership; noise kinds:
  finish, wasted, window_end, wakeup) and may rewrite routes (``set_route``), abandon waiting agents (``abandon``),
  ask to be woken (``request_wakeup``) or inject events (``post_event``, e.g. comm-layer membership changes).
  ``state()`` is the causal snapshot a planner may read; ``predict_ready`` the Section 4.1 predictor on it.
- ``run_policy(policy)``: a per-agent policy (the loop of ``Worker.run_episode`` / ``cbba_sota.solvers.rl``), e.g.
  ``RLPolicy`` (released checkpoint) or ``NearestPolicy`` (paper greedy, distance bug fixed).

With ``perturb.static_realization`` (all releases at 0, no noise, no failures) both are bit-identical to the static
env (RL rollouts, the paper greedy, ``execute_by_route`` plan replay). The native loops (``rl.rollout``,
``greedy._run_nearest``, ``execute_by_route``) also run on this class for the static and release-only cases; failures
need ``run_plan`` or ``run_policy`` (they process the onset and detection events).

Every result carries ``world``, which names this class, its options and a hash of the ``cbba_sota/dyn`` env code.
"""
from __future__ import annotations

import copy
import hashlib
import heapq
import random
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from os import PathLike
from pathlib import Path
from typing import Protocol

import numpy as np

from cbba_sota.bench.configs import MAX_TIME
from cbba_sota.bench.heteromrta import TaskEnv, load_env
from cbba_sota.dyn.perturb import Realization, static_realization
from cbba_sota.hetero.instance import Instance

EPS = 1e-12
STRUCTURAL = frozenset({"release", "failure", "abandon", "idle", "membership"})
_HERE = Path(__file__).resolve().parent


def code_hash() -> str:
    """Short hash of the env and perturbation code (rows record which world produced them)."""
    h = hashlib.sha1()
    for name in ("env.py", "perturb.py"):
        h.update((_HERE / name).read_bytes())
    return h.hexdigest()[:10]


@dataclass(frozen=True)
class Event:
    """Something that happened at time ``t``. ``structural`` kinds are the Section 4.2 re-plan triggers."""

    kind: str  # release | failure | abandon | idle | membership | finish | wasted | window_end | wakeup
    t: float
    tasks: tuple[int, ...] = ()
    agents: tuple[int, ...] = ()
    info: tuple = ()  # kind-specific extras as (key, value) pairs

    @property
    def structural(self) -> bool:
        return self.kind in STRUCTURAL

    def get(self, key, default=None):
        return dict(self.info).get(key, default)


class Controller(Protocol):
    """Plan-following controller (a planner plugged into ``run_plan``)."""

    def on_events(self, env: DynTaskEnvX, t: float, events: list[Event]) -> bool | None:
        """Called once per epoch that has events; may call ``env.set_route`` etc. Return True if it re-planned."""


class Policy(Protocol):
    """Per-agent policy for ``run_policy``."""

    def selectable(self, env: DynTaskEnvX, agent_id: int) -> bool:
        """Whether the agent has a task it may choose now (False: it postpones, or goes home when allowed)."""

    def act(self, env: DynTaskEnvX, agent_id: int) -> int:
        """1-based task id (0 = depot); called right after ``selectable`` for the same agent."""


@dataclass
class StateView:
    """Causal snapshot at time ``t`` (what an ideal-comms planner may know). A failed robot is believed alive (and
    stalled) until its failure is detected; realized durations of unfinished tasks are not shown."""

    t: float
    H: float
    released: np.ndarray  # [T] bool
    finished: np.ndarray  # [T] bool
    started: np.ndarray  # [T] bool, started and not finished (as observed)
    start: np.ndarray  # [T] observed start (nan if not started)
    committed: list[tuple[int, ...]]  # per task: committed members not known to have failed
    present: list[tuple[int, ...]]  # per task: committed members that have arrived
    residual: np.ndarray  # [T, K] requirement minus committed traits, clipped at 0
    alive: np.ndarray  # [A] bool, not known to have failed
    mode: list[str]  # per agent: home | to_home | travel | wait | work | idle | failed
    target: np.ndarray  # [A] current task (-1: depot)
    pos: np.ndarray  # [A, 2] location of the current target (the native agent['location'])
    dep: np.ndarray  # [A] departure of the current leg
    tau: np.ndarray  # [A] nominal travel time of the current leg
    stall: np.ndarray  # [A] stall time observed on the current leg so far
    eta: np.ndarray  # [A] observed arrival if arrived, else dep + tau + stall so far (>= t)
    routes: list[list[int]]  # future routes (plan mode)
    dur_nom: np.ndarray  # [T]


@dataclass
class Episode:
    """Outcome of one episode on ``DynTaskEnvX``."""

    world: str
    makespan: float | None  # physical end (D7); None if the episode never ended
    success: bool  # every task finished and makespan < MAX_TIME
    completion: float  # fraction of tasks finished
    env_finished: bool
    travel: float  # summed travel distance (untravelled part of a failed robot's leg removed)
    wait: float  # summed waiting at tasks (start - arrival; leave - arrival for abandons)
    wasted_trips: int
    abandons: int
    restarts: int  # aborted tasks restarted
    failures: int  # robots that failed (onset reached)
    detected: int
    decisions: int  # policy decisions (run_policy)
    controller_calls: int
    replans: int  # controller calls that returned True
    route_versions: int  # set_route calls that changed a route
    structural_events: int
    epochs: int
    cpu_event_s: list[float] = field(default_factory=list)  # CPU per controller call with a structural event
    # (run_plan) or per policy decision (run_policy), process time of this process
    cpu_total_s: float = 0.0  # controller or policy CPU over the episode

    @property
    def makespan_or_cap(self) -> float:
        """Makespan, or MAX_TIME for a failed episode (the spec's failure imputation, sensitivity only)."""
        return float(self.makespan) if self.success else MAX_TIME


class DynTaskEnvX(TaskEnv):
    """See the module docstring. Build with ``make_env``."""

    # ---- setup ------------------------------------------------------------------------------------------------
    def setup(self, real: Realization | None = None, *, coalition: str = "auto", obs: str = "causal",
              lease_L: float = 200.0, abandon_on_failure: bool = True) -> None:
        T, A = len(self.task_dic), len(self.agent_dic)
        if coalition not in ("auto", "arrival", "decision"):
            raise ValueError(f"coalition must be 'auto', 'arrival' or 'decision', got {coalition!r}")
        if obs not in ("causal", "oracle"):
            raise ValueError(f"obs must be 'causal' or 'oracle', got {obs!r}")
        dur = getattr(self, "dur_nom", None)  # re-setup: task times already hold realized durations
        if dur is None:
            dur = np.array([float(self.task_dic[j]["time"]) for j in range(T)])
        if real is None:
            req = np.array([self.task_dic[j]["requirements"] for j in range(T)], dtype=float)
            real = static_realization(req, dur, A)
        if real.n_tasks != T or real.n_agents != A:
            raise ValueError(f"realization is for {real.n_tasks} tasks x {real.n_agents} agents, env has {T} x {A}")
        if not np.array_equal(real.dur_nom, dur):
            raise ValueError("realization's nominal durations differ from the env's task times")
        if obs == "oracle" and np.isfinite(real.fail_onset).any():
            raise ValueError("oracle observations are undefined under failures (a halted robot's arrival is inf)")
        self.real = real
        self.coalition_rule, self.obs_mode = coalition, obs
        self.lease_L, self.abandon_on_failure = float(lease_L), bool(abandon_on_failure)
        self.dur_nom = real.dur_nom.copy()
        self.dur_real = real.dur_real.copy()
        for j in range(T):
            self.task_dic[j]["time"] = float(self.dur_real[j])  # the engine runs on realized durations
        self.release = np.asarray(real.release, dtype=float).copy()
        self.H = float(real.H)
        self._base_delays = [(np.asarray(s, float).copy(), np.asarray(e, float).copy()) for s, e in real.delays]
        self.fail_onset = np.asarray(real.fail_onset, dtype=float).copy()
        self.detect_after = float(real.detect_after)
        self.init_state()

    @property
    def coalition(self) -> str:
        """Coalition rule in force: the option, with "auto" = arrival in plan following, decision otherwise."""
        rule = getattr(self, "coalition_rule", "decision")
        if rule == "auto":
            return "arrival" if getattr(self, "_mode", None) == "plan" else "decision"
        return rule

    @property
    def world(self) -> str:
        return (f"DynTaskEnvX[{self.coalition},{self.obs_mode},L={self.lease_L:g},"
                f"{'det' if self.abandon_on_failure else 'nodet'}]@{code_hash()}")

    def init_state(self):
        super().init_state()
        self.max_waiting_time = getattr(self, "lease_L", 200.0)
        T, A = len(self.task_dic), len(self.agent_dic)
        for a in self.agent_dic.values():
            a["leg"] = (0.0, 0.0, a["depot"], a["depot"])  # (departure, nominal travel, from, to)
        for t in self.task_dic.values():
            t.update(coalition=[], wasted=[], not_before=0.0, aborted=False, restarts=0)
        self.delays = [(s.copy(), e.copy()) for s, e in getattr(self, "_base_delays", [])] or \
            [(np.zeros(0), np.zeros(0)) for _ in range(A)]
        self.dead = np.zeros(A, bool)  # physically failed (onset reached)
        self.failed_known = np.zeros(A, bool)  # failure detected
        self._dead_pos: dict[int, np.ndarray] = {}
        self._left = [None] * A  # time an agent left its current task without working (abandon / wasted trip)
        self._idle = np.zeros(A, bool)
        self._route: list[list[int]] = [[] for _ in range(A)]
        self.legs: list[list[tuple[float, float, float, int]]] = [[] for _ in range(A)]  # (decided, dep, tau, to)
        self._info_epoch = 0.0
        self._mode = None
        self._active: list[tuple[float, int, str, int]] = []  # (time, order, kind, agent): failure onsets/detections
        onset = getattr(self, "fail_onset", np.full(A, np.inf))
        for i in range(A):
            if np.isfinite(onset[i]):
                self._active.append((float(onset[i]), 0, "onset", i))
                self._active.append((float(onset[i]) + self.detect_after, 1, "detect", i))
        self._active.sort()
        self._act_ptr = 0
        self._wakeups: list[float] = []
        self._reported_release = np.zeros(T, bool)
        self._reported_finish = np.zeros(T, bool)
        self._window_reported = False
        self._pending_events: list[Event] = []
        self._freed_now: list[int] = []
        self.n_release_polls = 0
        self.stats = {"wasted": 0, "abandons": 0, "restarts": 0, "failures": 0, "detected": 0, "abandon_wait": 0.0,
                      "route_versions": 0, "decisions": 0, "controller_calls": 0, "replans": 0, "structural": 0,
                      "epochs": 0, "cpu_total": 0.0}
        self.cpu_event: list[float] = []
        self.route_log: list[tuple[float, int, tuple[int, ...]]] | None = None

    # ---- time / release helpers --------------------------------------------------------------------------------
    def released_mask(self) -> np.ndarray:
        return self.release <= self.current_time + EPS

    def _next_passive(self) -> float:
        """Earliest future release epoch or window end (re-poll epochs; nothing to process)."""
        now = self.current_time
        pending = self.release[self.release > now + EPS]
        t = float(pending.min()) if pending.size else np.inf
        if self.H > now + EPS:
            t = min(t, self.H)
        return t

    def _next_active(self) -> float:
        """Earliest unprocessed failure onset/detection or controller wake-up."""
        t = self._active[self._act_ptr][0] if self._act_ptr < len(self._active) else np.inf
        if self._wakeups:
            t = min(t, self._wakeups[0])
        return t

    def may_go_home(self) -> bool:
        """D7: with H > 0, only once t >= H and every released task is finished; with H = 0 always (native)."""
        if self.H <= 0:
            return True
        if self.current_time < self.H - EPS:
            return False
        rel = self.released_mask()
        return all(self.task_dic[j]["finished"] for j in np.flatnonzero(rel))

    # ---- D1: release masking ------------------------------------------------------------------------------------
    def get_unfinished_tasks(self):
        rel = self.released_mask()
        return [task["feasible_assignment"] is False and np.any(task["status"] > 0) and bool(rel[j])
                for j, task in self.task_dic.items()]

    # ---- the decision loop ------------------------------------------------------------------------------------
    def next_decision(self):
        """Native ``next_decision`` plus the dynamic epochs. Failure onsets, detections and controller wake-ups that
        are due before (or at) the next agent epoch are processed here, so any driver of the native loop gets D6:
        a silent onset only changes the physics; a detection or wake-up is an epoch of its own (re-polling idle
        agents and the agents that abandoned), whose events wait in ``drain_events``. When every live agent is
        heading home but some task is not covered (a failure aborted it), the next active event is the next epoch;
        if there is none, nobody will ever act again and the epoch is inf (the episode fails)."""
        while True:
            (fin, blk), t, end = self._agent_next()
            ta = self._next_active()
            if end and not self._all_covered():
                if ta == np.inf:
                    return ([], []), np.inf
            elif ta == np.inf or ta > t:
                return (fin, blk), t
            self.current_time = max(self.current_time, ta)
            n0 = len(self._pending_events)
            self._process_active(ta)
            if len(self._pending_events) > n0:
                (fin, blk), t, end = self._agent_next()
                if t == ta and not end:
                    return (fin, blk), t
                return ([], self._blocked_at(ta)), ta

    def _all_covered(self) -> bool:
        """Every task has a coalition with a finite finish (nothing aborted or waiting for cover)."""
        return all(t["feasible_assignment"] and np.isfinite(t["time_finish"]) for t in self.task_dic.values())

    def _agent_next(self):
        """Next agent epoch without processing anything: ``((finished, blocked), t, end)`` where ``end`` flags the
        native end-of-episode branch (every live agent heading home). Native rules; dead agents are excluded;
        release epochs and the window end re-poll idle agents as in the pilot."""
        passive = self._next_passive()
        agents = self.agent_dic
        dt = np.array(self.get_matrix(agents, "next_decision"), dtype=float)
        if passive == np.inf and not self.dead.any():
            (fin, blk), t = super().next_decision()  # native semantics
            return (fin, blk), t, bool(np.all(np.isnan(dt)))
        alive = ~self.dead
        dt[self.dead] = np.nan
        nc = np.array(self.get_matrix(agents, "no_choice"), dtype=bool) & alive
        last_arr = np.array([a["arrival_time"][-1] for a in agents.values()], dtype=float)
        if passive == np.inf:  # native rules without dead agents
            if np.all(np.isnan(dt)):
                # a robot that failed after reaching its depot was home alive, so it still counts
                ends = [max(a["arrival_time"]) if a["arrival_time"] else 0 for i, a in agents.items()
                        if alive[i] or (a["current_task"] < 0 and a["arrival_time"][-1] <= self.fail_onset[i])]
                return ([], []), max(max(ends, default=0.0), self.current_time), True
            d = np.where(nc, np.inf, dt)
            t = np.nanmin(d)
            if np.isinf(t):  # native: idle agents decide at their last arrival (never in the past here)
                d = np.where(nc, np.inf, np.where(alive, np.maximum(last_arr, self.current_time), np.nan))
                t = float(np.nanmin(d))
        else:  # a release epoch or the window end is pending (pilot rule: min of finite times and the epoch)
            d = np.where(nc, np.inf, dt)
            finite = d[np.isfinite(d)]
            t = min(float(finite.min()) if finite.size else np.inf, passive)
        finished = np.flatnonzero(d == t).tolist()
        blocked = [i for i in np.flatnonzero(np.isinf(d)).tolist() if t >= last_arr[i]]
        if t == passive:
            self.n_release_polls += len(blocked)
        return (finished, blocked), t, False

    def _blocked_at(self, t: float) -> list[int]:
        """Agents re-polled at an event epoch: undecided (inf or postponed) and not travelling."""
        out = []
        for i, a in self.agent_dic.items():
            if self.dead[i]:
                continue
            d = np.inf if a["no_choice"] else a["next_decision"]
            if np.isinf(d) and t >= a["arrival_time"][-1]:
                out.append(i)
        return out

    def check_finished(self):
        self.task_update()
        decision_agents, current_time, end = self._agent_next()
        ta = self._next_active()
        if ta <= current_time or (end and not self._all_covered()):
            return False  # an onset, detection or wake-up comes first; the next next_decision processes it
        if len(decision_agents[0]) + len(decision_agents[1]) == 0:
            self.current_time = current_time
            returned = all(a["returned"] or self.dead[i] for i, a in self.agent_dic.items())
            return bool(returned and np.all(self.get_matrix(self.task_dic, "finished")))
        return False

    # ---- failure view (the contract of cbba_sota.dyn.baselines.rl_online) ------------------------------------
    @property
    def realization(self) -> Realization:
        return self.real

    def known_failed(self, t: float | None = None) -> list[int]:
        """Robots whose failure has been detected (processed) by now."""
        return np.flatnonzero(self.failed_known).tolist()

    def is_failed(self, i: int, t: float | None = None) -> bool:
        """Robot ``i`` has physically failed (onset processed)."""
        return bool(self.dead[i])

    # ---- D2 / D5: coalition formation ---------------------------------------------------------------------------
    def _eff_arrival(self, m: int, j: int) -> float:
        return np.inf if self.dead[m] else self.get_arrival_time(m, j)

    def task_update(self):
        f_task = []
        now = self.current_time
        arrival_rule = self.coalition == "arrival"
        for task in self.task_dic.values():
            j = task["ID"]
            tentative = (arrival_rule and task["feasible_assignment"] and not task["finished"]
                         and task["time_start"] > now)
            if not task["feasible_assignment"] or tentative:
                members = task["members"]
                abilities = self.get_abilities(members)
                task["status"] = task["requirements"] - abilities
                if (task["status"] <= 0).all():
                    if arrival_rule:
                        order = sorted(range(len(members)), key=lambda k: (self._eff_arrival(members[k], j), k))
                        need = np.array(task["requirements"], dtype=float)
                        coal, start = [], np.inf
                        for k in order:
                            m = members[k]
                            coal.append(m)
                            need = need - self.agent_dic[m]["abilities"]
                            if (need <= 0).all():
                                start = self._eff_arrival(m, j)
                                break
                        start = float(start)
                    else:
                        arrival = np.array([self._eff_arrival(m, j) for m in members])
                        start, coal = float(np.max(arrival)), list(members)
                    start = max(start, task["not_before"])  # returns ``start`` itself when not binding
                    if not task["feasible_assignment"]:
                        f_task.append(j)
                    task["time_start"] = start
                    task["time_finish"] = start + task["time"]
                    task["feasible_assignment"] = True
                    task["coalition"] = coal
                else:
                    if tentative:
                        task["time_start"], task["time_finish"] = 0, 0
                    task["feasible_assignment"] = False
                    task["coalition"] = []
                    for member in list(members):  # D5 (i): own waiting time only
                        arr = self._eff_arrival(member, j)
                        if arr <= now and now - arr >= self.max_waiting_time:
                            self._abandon(member, j, now, "lease")
            else:
                if now >= task["time_finish"]:
                    task["finished"] = True
        for depot in self.depot_dic.values():
            for member in depot["members"]:
                if now >= self.get_arrival_time(member, depot["ID"]) and \
                        np.all(self.get_matrix(self.task_dic, "feasible_assignment")):
                    self.agent_dic[member]["returned"] = True
        return f_task

    def agent_update(self):
        super().agent_update()
        arrival_rule = self.coalition == "arrival"
        for i, a in self.agent_dic.items():
            if self.dead[i]:
                a["next_decision"] = np.nan
                continue
            if self._left[i] is not None:
                a["next_decision"], a["assigned"] = self._left[i], False
                continue
            c = a["current_task"]
            if c >= 0:
                task = self.task_dic[c]
                if task["feasible_assignment"] and i in task["members"]:
                    if arrival_rule and i not in task["coalition"]:
                        a["next_decision"], a["assigned"] = self.get_arrival_time(i, c), False  # leaves on arrival
                    elif not np.isfinite(task["time_finish"]):
                        a["next_decision"] = self.get_arrival_time(i, c) + self.max_waiting_time  # held, waiting
            elif self._mode == "plan" and self._route[i]:
                a["next_decision"] = max(a["arrival_time"][-1], self.current_time)  # re-routed from its depot

    def _abandon(self, i: int, j: int, t: float, reason: str) -> None:
        task = self.task_dic[j]
        task["members"].remove(i)
        if i in task["coalition"]:
            task["coalition"].remove(i)
        task["abandoned_agent"].append(i)
        self.stats["abandons"] += 1
        self.stats["abandon_wait"] += max(t - self.get_arrival_time(i, j), 0.0)
        self._left[i] = t
        a = self.agent_dic[i]
        a["next_decision"], a["no_choice"] = t, False
        self._freed_now.append(i)
        self._pending_events.append(Event("abandon", t, (j,), (i,), (("reason", reason),)))

    def abandon(self, i: int) -> bool:
        """Controller action: agent ``i`` leaves the task it is waiting at (arrived, not started). Returns False if it
        is not waiting at an unstarted task."""
        a = self.agent_dic[i]
        c = a["current_task"]
        if self.dead[i]:  # a halted robot does nothing; answer as if it left, so nothing leaks before detection
            return True
        if c < 0 or self._left[i] is not None:
            return False
        task = self.task_dic[c]
        now = self.current_time
        if i not in task["members"] or self.get_arrival_time(i, c) > now:
            return False
        if task["feasible_assignment"] and task["time_start"] <= now and i in task["coalition"]:
            return False  # working
        self._abandon(i, c, now, "controller")
        self.task_update()
        self.agent_update()
        return True

    # ---- D4: departures ----------------------------------------------------------------------------------------
    def _departure(self, i: int, previous_task: int, target: int) -> float:
        now = self.current_time
        if target < 0 and self.H > 0:
            return now  # D7: the trip home is decided when it becomes allowed; never back-dated before that
        if self._mode != "plan" and previous_task >= 0:
            prev = self.task_dic[previous_task]
            if prev["feasible_assignment"] and prev["time_finish"] < now and i in prev["members"] \
                    and self._left[i] is None:
                # native flashforward of a blocked agent, capped at the latest information epoch (D4)
                return max(prev["time_finish"], self._info_time())
        return now

    def _info_time(self) -> float:
        """Latest release epoch reached (time-based, as the pilot) or failure detection processed."""
        seen = self.release[self.release <= self.current_time + EPS]
        return max(float(seen.max()) if seen.size else 0.0, self._info_epoch)

    def _arrive(self, agent_id: int, dep: float, tau: float) -> float:
        """Arrival through the agent's stall calendar (continuous travel at nominal speed outside stalls)."""
        starts, ends = self.delays[agent_id]
        if starts.size == 0:
            return dep + tau  # same float op as the native env
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
        """Stall time inside [a, b]."""
        starts, ends = self.delays[agent_id]
        if starts.size == 0 or b <= a:
            return 0.0
        return float(np.clip(np.minimum(ends, b) - np.maximum(starts, a), 0.0, None).sum())

    def agent_step(self, agent_id, task_id, decision_step):
        task_id = task_id - 1
        agent = self.agent_dic[agent_id]
        if self.dead[agent_id]:
            raise RuntimeError(f"agent {agent_id} has failed")
        if task_id != -1:
            task = self.task_dic[task_id]
            if self.coalition == "decision" and task["feasible_assignment"]:
                return -1, False, []
            if self.release[task_id] > self.current_time + EPS:
                raise ValueError(f"task {task_id} is not released at t={self.current_time}")
        else:
            if self.H > 0 and not self.may_go_home():  # D7/D8: wait in place, re-polled every epoch
                agent["no_choice"] = True
                return 0, False, []
            task = self.depot_dic[agent["species"]]
        agent["route"].append(task["ID"])
        previous_task = agent["current_task"]
        agent["current_task"] = task_id
        dist = self.calculate_eulidean_distance(agent, task)
        travel_time = dist / agent["velocity"]
        agent["travel_time"] = travel_time
        agent["travel_dist"] += dist
        dep = self._departure(agent_id, previous_task, task_id)
        agent["arrival_time"] += [self._arrive(agent_id, dep, travel_time)]
        agent["leg"] = (dep, travel_time, agent["location"], task["location"])
        self.legs[agent_id].append((self.current_time, dep, travel_time, int(task["ID"])))
        agent["location"] = task["location"]
        agent["decision_step"] = decision_step
        agent["no_choice"] = False
        if task_id >= 0 and previous_task < 0:
            agent["returned"] = False  # leaves its depot (possible after a re-plan)
        self._left[agent_id] = None
        self._idle[agent_id] = False
        if agent_id not in task["members"]:
            task["members"].append(agent_id)
        f_t = self.task_update()
        self.agent_update()
        return 0, True, f_t

    # ---- D6: failures ------------------------------------------------------------------------------------------
    def position(self, i: int, t: float | None = None) -> np.ndarray:
        """Physical position of agent ``i`` at ``t`` (default now), on its straight leg, stall-aware."""
        t = self.current_time if t is None else t
        if i in self._dead_pos and t >= self.fail_onset[i]:
            return self._dead_pos[i]
        dep, tau, p0, p1 = self.agent_dic[i]["leg"]
        if tau <= 0 or t <= dep:
            return np.asarray(p0 if t <= dep else p1, dtype=float)
        moved = (t - dep) - self._stalled(i, dep, t)
        frac = min(max(moved / tau, 0.0), 1.0)
        return np.asarray(p0, float) + (np.asarray(p1, float) - np.asarray(p0, float)) * frac

    def _fail_onset(self, i: int, t: float) -> None:
        a = self.agent_dic[i]
        self._dead_pos[i] = self.position(i, t)
        self.dead[i] = True
        self.stats["failures"] += 1
        c = a["current_task"]
        if a["arrival_time"][-1] > t:  # travelling: never arrives; looks stalled until detected
            left = float(np.linalg.norm(np.asarray(a["leg"][3], float) - self._dead_pos[i]))
            a["travel_dist"] -= left
            a["arrival_time"][-1] = np.inf
            s, e = self.delays[i]
            keep = s < t
            s, e = s[keep], np.minimum(e[keep], t)
            self.delays[i] = (np.append(s, t), np.append(e, np.inf))
        if c >= 0:
            task = self.task_dic[c]
            if task["feasible_assignment"] and not task["finished"]:
                if task["time_start"] <= t and i in task["coalition"]:
                    task["time_finish"] = np.inf  # in progress: aborted (known at detection)
                    task["aborted"] = True
                elif task["time_start"] > t:
                    task["feasible_assignment"] = False  # recompute: it cannot start with a dead member
        a["next_decision"], a["no_choice"] = np.nan, False
        self.task_update()
        self.agent_update()

    def _fail_detect(self, i: int, t: float) -> list[Event]:
        self.failed_known[i] = True
        self._route[i] = []
        self.stats["detected"] += 1
        self._info_epoch = max(self._info_epoch, t)
        a = self.agent_dic[i]
        c = a["current_task"]
        info: dict = {"aborted": False, "abandoned": ()}
        tasks: tuple[int, ...] = ()
        if c >= 0 and not self.task_dic[c]["finished"] and i in self.task_dic[c]["members"]:
            task = self.task_dic[c]
            tasks = (c,)
            task["members"].remove(i)
            if i in task["coalition"]:
                task["coalition"].remove(i)
            if task["aborted"]:  # it was working: the task restarts from scratch once re-covered
                task["aborted"] = False
                task["restarts"] += 1
                self.stats["restarts"] += 1
                info["aborted"] = True
                reset = True
            else:  # not started yet (a start already reached without it, as a late member, stands)
                reset = not (task["feasible_assignment"] and task["time_start"] <= t)
            if reset:
                task["feasible_assignment"] = False
                task["time_start"], task["time_finish"] = 0, 0
                task["not_before"] = t
            self.task_update()
            if not task["feasible_assignment"]:
                if self.abandon_on_failure:
                    gone = []
                    for m in list(task["members"]):
                        if not self.dead[m] and self.get_arrival_time(m, c) <= t:
                            self._abandon(m, c, t, "partner-failed")
                            gone.append(m)
                    info["abandoned"] = tuple(gone)
                if self._mode != "plan":  # policy mode: agents at or heading to their depot are re-polled
                    for m, b in self.agent_dic.items():
                        if not self.dead[m] and b["current_task"] < 0 and np.isnan(b["next_decision"]):
                            b["next_decision"] = np.inf
                            b["returned"] = False
        self.agent_update()
        return [Event("failure", t, tasks, (i,), tuple(info.items()))]

    # ---- events ------------------------------------------------------------------------------------------------
    def request_wakeup(self, t: float) -> None:
        """Controller hook: be called back at time ``t`` (clamped to now) even if nothing happens."""
        heapq.heappush(self._wakeups, max(float(t), self.current_time))

    def _process_active(self, t: float) -> None:
        """Failure onsets/detections and wake-ups due at ``t`` (events go to ``_pending_events``)."""
        while self._act_ptr < len(self._active) and self._active[self._act_ptr][0] <= t:
            _, _, kind, i = self._active[self._act_ptr]
            self._act_ptr += 1
            if kind == "onset":
                if not self.dead[i]:
                    self._fail_onset(i, t)
            elif self.dead[i]:
                self._pending_events += self._fail_detect(i, t)
        woke = False
        while self._wakeups and self._wakeups[0] <= t:
            heapq.heappop(self._wakeups)
            woke = True
        if woke:
            self._pending_events.append(Event("wakeup", t))

    def _report(self, t: float) -> list[Event]:
        """Release and window-end notices at epoch ``t`` (time-based knowledge; nothing to process)."""
        evs: list[Event] = []
        rel = np.flatnonzero((self.release <= t + EPS) & ~self._reported_release)
        if rel.size:
            self._reported_release[rel] = True
            evs.append(Event("release", t, tuple(int(j) for j in rel)))
        if self.H > 0 and t >= self.H - EPS and not self._window_reported:
            self._window_reported = True
            evs.append(Event("window_end", t))
        return evs

    def post_event(self, event: Event) -> None:
        """Hook for an external layer (e.g. the C3 comm layer's ``membership`` changes, computed at its own
        wake-ups): the event is delivered with the next epoch's events."""
        self._pending_events.append(event)

    def drain_events(self) -> list[Event]:
        """Events generated since the last drain (failure detections, abandons, wake-ups)."""
        evs, self._pending_events = self._pending_events, []
        return evs

    def _settle(self, agents: Sequence[int], t: float) -> list[Event]:
        """Released agents that arrived late or redundant at a task leave it (D2 wasted trip); finishes at t."""
        evs = []
        for i in agents:
            if self.dead[i] or self._left[i] is not None:
                continue
            c = self.agent_dic[i]["current_task"]
            if c < 0:
                continue
            task = self.task_dic[c]
            if task["feasible_assignment"] and i in task["members"] and i not in task["coalition"] \
                    and self.get_arrival_time(i, c) <= t:
                task["members"].remove(i)
                task["wasted"].append(i)
                self.stats["wasted"] += 1
                self._left[i] = t
                evs.append(Event("wasted", t, (c,), (i,)))
        done = [j for j, task in self.task_dic.items()
                if task["feasible_assignment"] and task["time_finish"] <= t and not self._reported_finish[j]]
        if done:
            self._reported_finish[done] = True
            evs.append(Event("finish", t, tuple(done)))
        return evs

    # ---- routes (plan mode) ------------------------------------------------------------------------------------
    def route(self, i: int) -> list[int]:
        """Future route of agent ``i`` (0-based task ids after its current commitment)."""
        return list(self._route[i])

    def set_route(self, i: int, tasks: Sequence[int]) -> bool:
        """Replace agent ``i``'s future route; returns True if it changed. Tasks must be released and distinct."""
        new = [int(j) for j in tasks]
        if len(set(new)) != len(new):
            raise ValueError(f"route of agent {i} repeats a task: {new}")
        if any(self.release[j] > self.current_time + EPS for j in new):
            raise ValueError(f"route of agent {i} contains an unreleased task at t={self.current_time}")
        if new == self._route[i]:
            return False
        self._route[i] = new
        self.stats["route_versions"] += 1
        if self.route_log is not None:
            self.route_log.append((self.current_time, i, tuple(new)))
        return True

    # ---- execution APIs ----------------------------------------------------------------------------------------
    def run_plan(self, controller: Controller | None = None, routes: Sequence[Sequence[int]] | None = None, *,
                 max_time: float = MAX_TIME, skip_finished: bool = True) -> Episode:
        """Plan following in the env loop (``execute_by_route`` generalized). ``routes``: initial 0-based routes.
        ``skip_finished``: agents skip route tasks that are already finished (good comms; set False when a
        controller models per-robot beliefs and wants stale visits to become wasted trips)."""
        self.init_state()
        self._mode = "plan"
        if routes is not None:
            if len(routes) != len(self.agent_dic):
                raise ValueError(f"{len(routes)} routes for {len(self.agent_dic)} agents")
            for i, r in enumerate(routes):
                self._route[i] = [int(j) for j in r]
        while not self.finished and self.current_time < max_time:
            (fin, blk), t = self.next_decision()
            self.current_time = t
            if not np.isfinite(t):
                break
            self.stats["epochs"] += 1
            self.task_update()
            evs = self.drain_events() + self._report(t)
            released = _merge(fin, self._freed_now, blk)
            self._freed_now = []
            evs += self._settle(released, t)
            idle = []
            for i in released:
                if not self.dead[i]:
                    self._skip_finished(i, skip_finished)
                    if not self._route[i] and not self.may_go_home() and not self._idle[i]:
                        idle.append(i)
            if idle:
                n_open = sum(1 for j, task in self.task_dic.items() if self.release[j] <= t + EPS
                             and not task["feasible_assignment"] and not task["finished"])
                evs.append(Event("idle", t, (), tuple(idle), (("open_tasks", n_open),)))
            if controller is not None and evs:
                self._call(controller, t, evs)
                evs2 = self.drain_events()  # abandons by the controller (it knows them); posted events wait
                self._pending_events = [e for e in evs2 if e.kind != "abandon"]
                released = _merge(released, self._freed_now)
                self._freed_now = []
                self.stats["structural"] += sum(e.kind == "abandon" for e in evs2)
                self.agent_update()
                for i, a in self.agent_dic.items():  # idle or home agents that just got work
                    if i not in released and not self.dead[i] and self._route[i] and \
                            a["arrival_time"][-1] <= t and (a["no_choice"] or a["current_task"] < 0
                                                           or self._left[i] is not None or self._idle[i]):
                        released.append(i)
            else:
                self.stats["structural"] += sum(e.structural for e in evs)
            for i in released:
                if self.dead[i]:
                    continue
                self._skip_finished(i, skip_finished)
                a = self.agent_dic[i]
                if self._route[i]:
                    self.agent_step(i, self._route[i].pop(0) + 1, 0)
                elif self.may_go_home():
                    if a["current_task"] < 0 and a["arrival_time"][-1] <= t and np.isnan(a["next_decision"]):
                        continue  # already home
                    self.agent_step(i, 0, 0)
                    a["next_decision"] = np.nan
                else:  # D8: wait in place, re-polled every epoch
                    a["no_choice"] = True
                    self._idle[i] = True
            self.finished = self.check_finished()
        return self._episode()

    def _skip_finished(self, i: int, skip: bool) -> None:
        if skip:
            r = self._route[i]
            while r and self.task_dic[r[0]]["finished"]:
                r.pop(0)

    def _call(self, controller: Controller, t: float, evs: list[Event]) -> None:
        n_struct = sum(e.structural for e in evs)
        self.stats["structural"] += n_struct
        c0 = time.process_time()
        replanned = controller.on_events(self, t, evs)
        dc = time.process_time() - c0
        self.stats["controller_calls"] += 1
        self.stats["replans"] += bool(replanned)
        self.stats["cpu_total"] += dc
        if n_struct:
            self.cpu_event.append(dc)

    def run_policy(self, policy: Policy, *, shuffle_seed: int | None = None, max_time: float = MAX_TIME,
                   observer=None) -> Episode:
        """Per-agent policy in the loop of ``Worker.run_episode`` (released finished agents are shuffled with
        ``random.Random(shuffle_seed)`` when given, as in ``cbba_sota.solvers.rl``). ``observer(env, t, events)``
        sees every epoch's events (logging only)."""
        self.init_state()
        self._mode = "policy"
        rng = random.Random(shuffle_seed) if shuffle_seed is not None else None
        step = 0
        while not self.finished and self.current_time < max_time:
            (fin, blk), t = self.next_decision()
            self.current_time = t
            if not np.isfinite(t):
                break
            self.stats["epochs"] += 1
            self.task_update()
            evs = self.drain_events() + self._report(t)
            freed, self._freed_now = self._freed_now, []
            evs += self._settle(_merge(fin, freed, blk), t)
            self.stats["structural"] += sum(e.structural for e in evs)
            if observer is not None:
                observer(self, t, evs)
            if rng is not None:
                rng.shuffle(fin)
            for i in _merge(fin, freed, blk):
                if self.dead[i]:
                    continue
                c0 = time.process_time()
                ok = policy.selectable(self, i)
                a = self.agent_dic[i]
                if not ok:
                    if not all(task["feasible_assignment"] for task in self.task_dic.values()):
                        a["no_choice"] = True
                        self.stats["cpu_total"] += time.process_time() - c0
                        continue
                    if a["current_task"] < 0:
                        self.stats["cpu_total"] += time.process_time() - c0
                        continue
                action = policy.act(self, i)
                dc = time.process_time() - c0
                self.stats["cpu_total"] += dc
                self.cpu_event.append(dc)  # per policy decision (observation + forward pass)
                self.stats["decisions"] += 1
                self.agent_step(i, action, step)
            self.finished = self.check_finished()
            step += 1
        return self._episode()

    # ---- results -----------------------------------------------------------------------------------------------
    def _episode(self) -> Episode:
        tasks = [self.task_dic[j] for j in range(len(self.task_dic))]
        fin = [bool(t["finished"]) for t in tasks]
        ms = float(self.current_time)
        ended = bool(self.finished) and np.isfinite(ms)
        wait = self.stats["abandon_wait"]
        for t in tasks:
            if t["finished"]:
                wait += sum(t["time_start"] - self.get_arrival_time(m, t["ID"]) for m in t["coalition"])
        from cbba_sota.hetero.replay import succeeded

        return Episode(
            world=self.world, makespan=ms if ended else None, success=succeeded(all(fin), ms if ended else None),
            completion=float(np.mean(fin)), env_finished=bool(self.finished),
            travel=float(sum(a["travel_dist"] for a in self.agent_dic.values())), wait=float(wait),
            wasted_trips=self.stats["wasted"], abandons=self.stats["abandons"], restarts=self.stats["restarts"],
            failures=self.stats["failures"], detected=self.stats["detected"], decisions=self.stats["decisions"],
            controller_calls=self.stats["controller_calls"], replans=self.stats["replans"],
            route_versions=self.stats["route_versions"], structural_events=self.stats["structural"],
            epochs=self.stats["epochs"], cpu_event_s=list(self.cpu_event), cpu_total_s=self.stats["cpu_total"])

    def signature(self) -> tuple:
        """Everything the episode did (per-agent arrivals and routes, task starts/finishes, end time)."""
        return ([list(map(float, a["arrival_time"])) for a in self.agent_dic.values()],
                [list(map(int, a["route"])) for a in self.agent_dic.values()],
                [(float(t["time_start"]), float(t["time_finish"])) for t in self.task_dic.values()],
                float(self.current_time))

    # ---- D3: observations --------------------------------------------------------------------------------------
    def get_current_task_status(self, agent):
        st = super().get_current_task_status(agent)
        if self.obs_mode == "causal":
            st[1:, 2 * self.traits_dim] = self.dur_nom
        st[1:][~self.released_mask()] = 0.0
        return st

    def _eta_obs(self, a) -> float:
        """Observed arrival if it happened, else nominal ETA plus the stall time seen so far."""
        arr = a["arrival_time"][-1]
        now = self.current_time
        if arr <= now:
            return arr
        dep, tau = a["leg"][0], a["leg"][1]
        return max(dep + tau + self._stalled(a["ID"], dep, now), now)

    def get_current_agent_status(self, agent):
        if self.obs_mode == "oracle":
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
                    start = max(self._eta_obs(self.agent_dic[m]) for m in task["coalition"] or task["members"])
                if now <= start:
                    waiting = max(now - eta, 0.0)
                    remaining = max(start + self.dur_nom[j] - now, 0.0)
            rows.append(np.hstack([a["abilities"], travel_time, remaining, waiting,
                                   agent["location"] - a["location"], a["assigned"]]))
        return np.vstack(rows)

    # ---- planner views -----------------------------------------------------------------------------------------
    def nominal_instance(self) -> Instance:
        """The instance with nominal durations (``Instance.from_env`` would read the realized ones)."""
        T, A = len(self.task_dic), len(self.agent_dic)
        tasks = [self.task_dic[j] for j in range(T)]
        agents = [self.agent_dic[i] for i in range(A)]
        return Instance(req=[t["requirements"] for t in tasks], loc=[t["location"] for t in tasks],
                        dur=self.dur_nom, ab=[a["abilities"] for a in agents], depot=[a["depot"] for a in agents],
                        species=[a["species"] for a in agents], speed=float(agents[0]["velocity"]))

    def state(self) -> StateView:
        now = self.current_time
        T, A = len(self.task_dic), len(self.agent_dic)
        K = self.traits_dim
        released = self.released_mask()
        finished = np.zeros(T, bool)
        started = np.zeros(T, bool)
        start = np.full(T, np.nan)
        committed, present = [], []
        residual = np.zeros((T, K))
        for j in range(T):
            task = self.task_dic[j]
            done = bool(task["feasible_assignment"] and task["time_finish"] <= now)
            finished[j] = done
            if task["feasible_assignment"] and task["time_start"] <= now:
                start[j] = task["time_start"]
                started[j] = not done
            mem = tuple(m for m in (task["coalition"] if done else task["members"]) if not self.failed_known[m])
            committed.append(mem)
            present.append(tuple(m for m in mem if self.get_arrival_time(m, j) <= now))
            ab = sum((self.agent_dic[m]["abilities"] for m in mem), np.zeros(K))
            residual[j] = np.clip(task["requirements"] - ab, 0, None)
        mode, target = [], np.full(A, -1)
        pos = np.zeros((A, 2))
        dep, tau, stall, eta = np.zeros(A), np.zeros(A), np.zeros(A), np.zeros(A)
        for i in range(A):
            a = self.agent_dic[i]
            d0, tau0 = a["leg"][0], a["leg"][1]
            dep[i], tau[i] = d0, tau0
            eta[i] = self._eta_obs(a)
            stall[i] = self._stalled(i, d0, min(now, eta[i]))
            pos[i] = a["location"]
            c = a["current_task"]
            target[i] = c
            arrived = a["arrival_time"][-1] <= now
            if self.failed_known[i]:
                mode.append("failed")
            elif c < 0:
                mode.append("home" if arrived else "to_home")
            elif not arrived:
                mode.append("travel")
            elif finished[c] or self._left[i] is not None or i not in self.task_dic[c]["members"]:
                mode.append("idle")
            elif started[c] and i in self.task_dic[c]["coalition"]:
                mode.append("work")
            else:
                mode.append("wait")
        return StateView(now, self.H, released, finished, started, start, committed, present, residual,
                         ~self.failed_known, mode, target, pos, dep, tau, stall, eta,
                         [list(r) for r in self._route], self.dur_nom.copy())


def _merge(*lists: Sequence[int]) -> list[int]:
    out: list[int] = []
    seen: set[int] = set()
    for lst in lists:
        for i in lst:
            if i not in seen:
                seen.add(i)
                out.append(i)
    return out


# ---- construction -------------------------------------------------------------------------------------------------


def make_env(source: TaskEnv | Instance | str | PathLike, real: Realization | None = None, **options
             ) -> DynTaskEnvX:
    """``DynTaskEnvX`` on a copy of ``source`` (env object, env pickle path or ``Instance``) with realization
    ``real`` (default: the static realization). ``options``: coalition, obs, lease_L, abandon_on_failure."""
    if isinstance(source, TaskEnv):
        env = copy.deepcopy(source)
    elif isinstance(source, Instance):
        from cbba_sota.hetero.replay import make_env as _make

        env = _make(source)
    else:
        env = load_env(source)
    env.__class__ = DynTaskEnvX
    env.setup(real, **options)
    return env


# ---- policies ---------------------------------------------------------------------------------------------------


class RLPolicy:
    """The released HeteroMRTA checkpoint (unmodified): greedy decode or sampling, one forward pass per decision.
    With ``shuffle_seed == seed`` in ``run_policy`` it reproduces ``cbba_sota.solvers.rl.rollout`` bit for bit on a
    static env."""

    def __init__(self, net, *, sample: bool = False, seed: int = 0, device: str = "cpu"):
        import torch

        self.net, self.sample, self.device = net, sample, device
        self.gen = torch.Generator(device).manual_seed(int(seed))
        self._obs = None

    def selectable(self, env: DynTaskEnvX, agent_id: int) -> bool:
        task_obs, agent_obs, mask = env.agent_observe(agent_id, False)
        self._agent = agent_id
        self._obs = drop_failed_rows(env, agent_id, task_obs, agent_obs, mask)
        return not bool(mask[0, 1:].all())

    def act(self, env: DynTaskEnvX, agent_id: int) -> int:
        import torch

        from cbba_sota.solvers.rl import _as_tensors, _choose

        assert self._obs is not None and self._agent == agent_id
        with torch.no_grad():
            probs, _ = self.net(*_as_tensors([self._obs], self.device))
            return int(_choose(probs, self.sample, self.gen).item())


def drop_failed_rows(env: DynTaskEnvX, agent_id: int, task_obs, agent_obs, mask) -> tuple:
    """Policy input without the rows of known-failed robots: (index of ``agent_id`` among the kept agent rows,
    task obs, agent obs, mask), the tuple layout of ``cbba_sota.solvers.rl._as_tensors``."""
    gone = env.known_failed()
    if not gone:
        return agent_id, task_obs, agent_obs, mask
    keep = np.ones(agent_obs.shape[1], bool)
    keep[gone] = False
    return int(keep[:agent_id].sum()), task_obs, agent_obs[:, keep], mask


class NearestPolicy:
    """The paper greedy (nearest contributable task, env masks incl. the max-open-task rule), argmin bug fixed,
    as ``cbba_sota.solvers.greedy.greedy_nearest(idle="wait")``."""

    def __init__(self, max_waiting: bool = True):
        self.max_waiting = max_waiting
        self._action = 0

    def selectable(self, env: DynTaskEnvX, agent_id: int) -> bool:
        from cbba_sota.solvers.greedy import _nearest_action

        self._action = _nearest_action(env, agent_id, self.max_waiting)
        return self._action != 0

    def act(self, env: DynTaskEnvX, agent_id: int) -> int:
        return self._action


# ---- a simple predictor for planners ----------------------------------------------------------------------------


def predict_ready(view: StateView, kappa: float = 1.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Causal prediction of when each agent is free again after its current commitment, and where.

    Returns ``ready`` [A] (inf for failed agents and for agents held at a task that is not covered yet),
    ``at`` [A] (task id where it becomes free, -1 = its depot) and ``task_finish`` [T] (predicted finish of started
    or covered committed tasks, nan otherwise). Travel still to go is the nominal remainder times ``kappa``;
    residual durations are nominal. This is the Section 4.1 predictor; planners may use their own."""
    t = view.t
    A, T = len(view.mode), len(view.released)
    arrive = np.full(A, np.nan)
    for i in range(A):
        moved = (t - view.dep[i]) - view.stall[i]
        rem = max(view.tau[i] - max(moved, 0.0), 0.0)
        arrive[i] = view.eta[i] if view.eta[i] <= t else t + rem * kappa
    finish = np.full(T, np.nan)
    for j in range(T):
        if view.finished[j] or not view.committed[j]:
            continue
        if view.started[j]:
            finish[j] = max(view.start[j] + view.dur_nom[j], t)
        elif (view.residual[j] <= 0).all():
            finish[j] = max(arrive[m] for m in view.committed[j]) + view.dur_nom[j]
    ready, at = np.full(A, np.inf), np.full(A, -1)
    for i in range(A):
        m = view.mode[i]
        if m == "failed":
            continue
        if m in ("home", "idle"):
            ready[i], at[i] = t, (view.target[i] if m == "idle" else -1)
        elif m == "to_home":
            ready[i] = arrive[i]
        else:
            j = view.target[i]
            at[i] = j
            if not np.isnan(finish[j]):
                ready[i] = finish[j]
    return ready, at, finish
