"""C3 world: the protocol arms of Section 5.3 with SPARC's planner inside the T2 comm layer, on the ground-truth env.

Ground truth is ``DynTaskEnvX`` (``cbba_sota.dyn.env``), driven through its plan-following loop ``run_plan``. The
comm layer (``comm.py``), the replicas (``replica.py``) and the protocol arms (``protocols.py``) run at the comm tick
(0.1 = env dt): the controller asks the env for a wake-up at every tick and, at each tick, (1) writes the robots'
local observations into their replicas, (2) delivers the frames of the previous tick, (3) lets every triggering
leader plan with the real planner (``protocols.rh_planner`` = agent B's ``RHPlanner``, the SPARC kernel, 300
iterations), (4) makes every robot adopt the newest route version its own replica holds, (5) applies the patient
lease (4.3) with the absent records of 4.6(a), (6) applies the rally rule, and (7) sends the frames of this tick.
Physics between ticks stays in the env: departures happen at the exact finish epoch (a robot follows its adopted
route, skipping only the tasks its OWN replica knows are started or finished, so stale knowledge becomes a wasted
trip, D2), coalitions start in arrival order (D2), stalls and durations follow the realization.

``C3Env`` (a subclass of ``DynTaskEnvX``; ``third_party`` and ``env.py`` untouched) adds the two C3 execution rules
the env lacks, both declared conventions of this module:

- **Rally (4.3).** ``rally_start(i, xy)``: an idle robot (arrived, empty route, not waiting at or working on a task)
  moves toward ``xy`` on a straight leg at nominal speed through its LoRR stall calendar; ``rally_stop(i)`` ends the
  leg where the robot is (it can stop at any tick). Any departure stops a rally first, so the next leg starts at the
  robot's physical position. A robot that rallies away from its depot is not home (``away``) until it goes back.
- **Termination (D7) with a global mission-complete signal.** A robot goes home when t >= H and every released task
  is finished (the env's D7 rule), for H = 0 cells too: while the mission is incomplete a robot that runs out of work
  waits in place (D8) or rallies, instead of the native trip home at H = 0, so the rally rule also works in F3. This
  1-bit global signal is a declared simplification. The spec's text ("its replica shows every known task finished")
  is available as ``C3Config.home_rule="replica"`` (``home_ok`` callback), but it has no termination liveness under
  partitions: in the dev design pilot the crew of the last task, knowing everything was finished, went home out of
  range and the robots waiting at the station never learned it (episodes ran to the time cap with every task done).
  Under ideal comms the global rule costs FULL a small end-of-episode tail on F3 against the good-comms controller
  (robots wait in the field and travel home after the last finish); it applies to every arm alike.
- Idle robots are polled at comm ticks. The native fallback "idle agents decide at their last arrival" (which can
  repeat the same epoch forever, or go back in time, once robots may wait after every task is finished) is replaced
  by "no agent epoch": the next epoch is the next tick, a release, a failure event or a real agent decision.

With ideal comms (arm FULL) the transport joins every replica at every epoch, the lease is the failure detector
(4.3, the env's ``abandon_on_failure``) and failures are known at the detector timeout; FULL is SPARC with good
comms inside this same world (the reference of K5 and E3a). T2 arms use the patient lease with parameters (g, L),
rule 4.6 (a silent robot is clamped, not failed; declared failed after h_f of silence plus an absent record), and
``abandon_on_failure=False`` (co-workers of a failed robot learn the abort on site and publish ``absent``; robots
waiting elsewhere rely on their lease).

Observation model (who learns what, when): the station learns a release at its release time; coalition members
observe a start and a finish (they are on site); the surviving members present at an aborted task observe the
abort at the failure detection and publish ``absent(failed robot, task)`` (on-site evidence, 4.6(a)); a robot at a
task site observes that task's current status (e.g. a wasted trip); robots write state records (heartbeats) on
change and every period (1.0 under T2, one tick under ideal comms). Everything else travels by gossip.

Declared simplifications (reported with the results):
- D4 back-dating is off (plan mode); legs to tasks are atomic (no mid-leg redirection), as in the env.
- Adoption. A robot adopts the newest version its replica holds for it and drops its old suffix (4.4). Under T2
  (``C3Config.adopt="keep"``, the spec-literal rule: a commitment is frozen once a member departed, 4.1) it keeps its
  current commitment even if the version omits it; the patient lease resolves a released commitment. With ideal comms
  (FULL) it follows the version, as ``controller.PlanController`` does: a waiting robot abandons a commitment the
  version dropped (the planner's G4 fallback releases frozen members in rare cases), a travelling one is released and
  leaves on arrival (``env.leave_on_arrival``, a D2 wasted trip). Keeping it under the detector lease produced
  permanent cross-waits (FULL, SA-AT-50 dev 18, seed 0, F3: 30% completion). On the dev design seed 2 (4 settings,
  192 T2 episodes per rule) "keep" and "follow" had GM makespan 82.4 vs 83.9 and success 92.2% vs 91.7%, with about
  half the wasted trips for "keep", which is therefore the T2 rule.
- Heartbeat records carry the robot's own causal ETA (nominal remaining travel times kappa, stalls so far).
"""
from __future__ import annotations

import hashlib
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from cbba_sota.dyn.env import EPS, DynTaskEnvX, code_hash
from cbba_sota.dyn.replica import (
    ABANDON,
    ABORT,
    ABSENT,
    FINISH,
    HOMING,
    IDLE,
    RALLY,
    RELEASE,
    START,
    STATION,
    TICK,
    TRAVEL,
    WAIT,
    WORK,
    LeaseConfig,
    lease_check,
    publish_abandon,
)

__all__ = ["ARM_NAMES", "C3Config", "C3Controller", "C3Env", "c3_hash", "make_c3_env", "resolve_arm", "run_c3"]

_HERE = Path(__file__).resolve().parent
_FILES = ("env.py", "perturb.py", "c3world.py", "comm.py", "replica.py", "protocols.py", "planner.py",
          "rh_kernels.py")


def _hash_files() -> str:
    h = hashlib.sha1()
    for name in _FILES:
        h.update((_HERE / name).read_bytes())
    return h.hexdigest()[:10]


_C3_HASH = _hash_files()  # at import (rows record the code the worker runs)


def c3_hash() -> str:
    """Hash of the C3 world code (env, perturbations, comm, replicas, protocols, planner) imported by this process."""
    return _C3_HASH


# ================================================================================================ env extension
class C3Env(DynTaskEnvX):
    """``DynTaskEnvX`` with the rally leg, the per-replica D7 rule and tick-polled idle robots (module docstring)."""

    c3 = False  # set by C3Controller: the C3 home rule and tick-polled idle robots (unset: exactly DynTaskEnvX)
    home_ok = None  # callback(agent_id) -> bool for the per-replica D7 variant (``C3Config.home_rule="replica"``)

    def init_state(self):
        super().init_state()
        A = len(self.agent_dic)
        self.away = np.zeros(A, bool)  # rallied away from its depot while its current_task is the depot
        self.rallying = np.zeros(A, bool)
        self.rally_dist = 0.0
        self.rally_legs = 0
        self._ctx = -1

    # ---- per-replica D7 -----------------------------------------------------------------------------------------
    def may_go_home(self) -> bool:
        if not self.c3 or self._ctx < 0:
            return super().may_go_home()
        if self.current_time < self.H - EPS:
            return False
        if self.home_ok is not None:
            return bool(self.home_ok(self._ctx))
        rel = self.released_mask()  # the mission-complete signal (also for H = 0, see the module docstring)
        return all(self.task_dic[j]["finished"] for j in np.flatnonzero(rel))

    def _skip_finished(self, i: int, skip: bool) -> None:
        self._ctx = i  # run_plan asks may_go_home() for agent i right after this call
        super()._skip_finished(i, skip)

    # ---- rally --------------------------------------------------------------------------------------------------
    def idle_here(self, i: int) -> bool:
        """Arrived, empty route, not waiting at or working on a task (at a finished task, after leaving one, or at
        its depot)."""
        a = self.agent_dic[i]
        t = self.current_time
        if self.dead[i] or self._route[i] or a["arrival_time"][-1] > t + EPS:
            return False
        c = a["current_task"]
        if c < 0:
            return True
        task = self.task_dic[c]
        return bool(task["finished"] or self._left[i] is not None or i not in task["members"])

    def rally_start(self, i: int, xy) -> bool:
        """Idle robot ``i`` starts (or restarts) a straight leg toward ``xy``; returns False if already there."""
        a = self.agent_dic[i]
        t = self.current_time
        if self.rallying[i]:
            self.rally_stop(i)
        p = np.asarray(self.position(i, t), float)
        xy = np.asarray(xy, float)
        tau = float(np.linalg.norm(xy - p)) / a["velocity"]
        if tau <= 1e-12:
            return False
        a["leg"] = (t, tau, p, xy)
        self.rallying[i] = True
        if a["current_task"] < 0:
            self.away[i] = True
            a["returned"] = False
        self.rally_legs += 1
        return True

    def rally_stop(self, i: int) -> None:
        """End a rally leg at the robot's current physical position (its location for the next departure)."""
        if not self.rallying[i]:
            return
        a = self.agent_dic[i]
        t = self.current_time
        p = np.asarray(self.position(i, t), float)
        d = float(np.linalg.norm(p - np.asarray(a["leg"][2], float)))
        a["travel_dist"] += d
        self.rally_dist += d
        a["location"] = p
        a["leg"] = (t, 0.0, p, p)
        self.rallying[i] = False

    def agent_step(self, agent_id, task_id, decision_step):
        self._ctx = agent_id
        if self.rallying[agent_id]:
            self.rally_stop(agent_id)
        out = super().agent_step(agent_id, task_id, decision_step)
        if out[1]:  # it departed (to a task or home)
            self.away[agent_id] = False
        return out

    def task_update(self):
        f = super().task_update()
        for i in np.flatnonzero(getattr(self, "away", ())):
            self.agent_dic[int(i)]["returned"] = False
        return f

    def agent_update(self):
        super().agent_update()
        for i in np.flatnonzero(getattr(self, "away", ())):
            a = self.agent_dic[int(i)]
            if not self.dead[i] and a["current_task"] < 0 and np.isnan(a["next_decision"]):
                a["next_decision"] = np.inf  # not home: undecided (polled at ticks), never "already home"

    # ---- idle robots are polled at comm ticks -------------------------------------------------------------------
    def _agent_next(self):
        (fin, blk), t, end = super()._agent_next()
        if not self.c3 or end or t == self._next_passive():
            return (fin, blk), t, end
        dt = np.array(self.get_matrix(self.agent_dic, "next_decision"), dtype=float)
        nc = np.array(self.get_matrix(self.agent_dic, "no_choice"), dtype=bool)
        d = np.where(nc | self.dead, np.nan, dt)
        if not np.isfinite(d).any():  # only the native "idle agents decide at their last arrival" fallback is left
            return ([], []), np.inf, False
        return (fin, blk), t, end

    @property
    def world(self) -> str:
        return f"C3Env<{super().world}>@{c3_hash()}"


def make_c3_env(source, real, **options) -> C3Env:
    """``C3Env`` on a copy of ``source`` (env pickle path, env or Instance) with realization ``real``."""
    from cbba_sota.dyn.env import make_env

    env = make_env(source, real, **options)
    env.__class__ = C3Env
    env.init_state()
    return env


# ================================================================================================ configuration
ARM_NAMES = ("FULL", "CEN-F", "HYB", "REP-clamp", "INF-r", "SPARC", "SPARC-V3", "REP-TO", "REP-KO")


@dataclass(frozen=True)
class C3Config:
    """One C3 episode's protocol configuration (the realization is separate)."""

    arm: str = "SPARC"  # ARM_NAMES; SPARC = V2 (structural triggers); HYB = SPARC-V1
    rho: float = 1.0  # range R = rho * r_c(robots + 1); ignored by FULL
    lease_g: float = 1.0  # patient lease grace g (T2 arms)
    lease_L: float = 30.0  # patient lease max wait L (T2 arms)
    iters: int = 300  # SPARC Light
    loss: str = "ge"
    p_loss: float = 0.2
    rho_ge: float = 0.8
    budget: float = 1500.0  # bytes per node per tick
    beacon: float | None = None  # beacon period (None: no beacon)
    frozen_model: str = "anchor"  # station outsiders: "anchor" (spec 4.5) | "head"
    keys: str = "monotone"  # planner key rule: "monotone" (spec 4.1) | "relaxed"
    theta: float = 0.7  # V3 gate
    member_window: float = 1.5
    membership_debounce: float = 0.5
    h_f: float = 10.0
    silence: float = 2.0  # lease: a planned member unheard for longer is "silent" (absent record)
    rally: bool = True
    adopt: str = "keep"  # T2: a version that drops the current commitment: "keep" it (spec 4.1/4.4) | "follow" (FULL: follow)
    home_rule: str = "global"  # "global" (mission-complete signal) | "replica" (spec D7 text; no liveness)
    station: tuple[float, float] = (0.5, 0.5)
    max_time: float = 200.0

    @property
    def ideal(self) -> bool:
        return self.arm == "FULL"

    def label(self) -> str:
        if self.ideal:
            return "FULL"
        return f"{self.arm}@rho={self.rho:g},g={self.lease_g:g},L={self.lease_L:g}"


def resolve_arm(cfg: C3Config):
    """The ``protocols.ArmSpec`` of ``cfg.arm`` (with the rally knockout applied)."""
    from dataclasses import replace

    from cbba_sota.dyn.protocols import ARMS, sparc

    name = cfg.arm
    if name in ("SPARC", "SPARC-V2"):
        spec = sparc("V2")
    elif name in ("HYB", "SPARC-V1"):
        spec = ARMS["HYB"]
    elif name == "SPARC-V3":
        spec = sparc("V3", cfg.theta)
    elif name in ARMS:
        spec = ARMS[name]
    else:
        raise ValueError(f"unknown C3 arm {name!r}; known: {', '.join(ARM_NAMES)}")
    return spec if cfg.rally else replace(spec, rally=False)


# ================================================================================================ controller
@dataclass
class _Stats:
    plan_calls: int = 0
    plan_calls_off_station: int = 0
    plan_cpu: list = field(default_factory=list)
    abandons_lease: int = 0
    releases: int = 0
    absents: int = 0
    rally_starts: int = 0
    adoptions: int = 0
    ticks: int = 0
    station_frac: list = field(default_factory=list)


class C3Controller:
    """``DynTaskEnvX.run_plan`` controller running one protocol arm on per-robot replicas (module docstring)."""

    def __init__(self, env: C3Env, inst, real, cfg: C3Config, key):
        from cbba_sota.dyn.planner import SearchConfig
        from cbba_sota.dyn.protocols import ProtocolConfig, make_stack, rh_planner

        self.env, self.inst, self.real, self.cfg = env, inst, real, cfg
        A, T = inst.n_agents, inst.n_tasks
        self.A, self.T = A, T
        self.kappa = float(real.kappa())
        self.st = _Stats()
        base = rh_planner(inst, iters=cfg.iters, kappa=self.kappa, config=SearchConfig(keys=cfg.keys),
                          frozen_model=cfg.frozen_model, h_f=cfg.h_f)
        st = self.st

        def planner(scope, belief):
            c0 = time.process_time()
            out = base(scope, belief)
            st.plan_cpu.append(time.process_time() - c0)
            st.plan_calls += 1
            st.plan_calls_off_station += int(not scope.is_station)
            return out

        self.arm = resolve_arm(cfg)
        if cfg.ideal:
            pcfg = ProtocolConfig.good_comms()
            self.lease = LeaseConfig("detector", detector_timeout=real.detect_after)
        else:
            pcfg = ProtocolConfig(member_window=cfg.member_window, failure_rule="suspicion",
                                  detector_timeout=real.detect_after, h_f=cfg.h_f,
                                  membership_debounce=cfg.membership_debounce)
            self.lease = LeaseConfig("patient", cfg.lease_g, cfg.lease_L, detector_timeout=real.detect_after,
                                     silence=cfg.silence, overdue_h=cfg.h_f)
        self.dur = np.asarray(inst.dur, float)
        self.kb, self.tr, self.runner = make_stack(A, T, inst.n_traits, self.arm, planner, rho=cfg.rho,
                                                   loss=cfg.loss, p=cfg.p_loss, key=key, budget=cfg.budget,
                                                   beacon_period=cfg.beacon, cfg=pcfg, dur=self.dur)
        self.L = self.kb.layout
        self.station_xy = np.asarray(cfg.station, float)
        self.kind_tab = np.full((T, 8), -1, np.int64)  # kind of each task event seq (I write every task event)
        self.rec_kind = np.full(T, -1, np.int64)
        self.rec_restarts = np.zeros(T, np.int64)
        self.adopted_key = np.full(A, -1, np.int64)
        self.hub = np.ones(A)
        self.k = -1
        self.wake_at = -np.inf
        self.done = False
        env.c3 = True
        env.home_ok = self.home_ok if cfg.home_rule == "replica" else None

    # ---- knowledge helpers --------------------------------------------------------------------------------------
    def kinds(self, t: float | None = None) -> np.ndarray:
        """[N, T] newest known event kind of every task in every replica (-1 unknown); under T2 with rule 4.6(c)
        (``replica.overdue_aborts``) applied at time ``t``: a START overdue by h_f whose crew is silent for h_f
        counts as ABORT."""
        seq = self.kb.v[:, self.L.TASK]
        tab = self.kind_tab[np.arange(self.T)[None, :], np.maximum(seq, 0)]
        out = np.where(seq >= 0, tab, -1)
        if self.cfg.ideal or t is None:
            return out
        h_f, ev = self.cfg.h_f, self.kb.facts.task_events
        heard = None
        for n, j in np.argwhere(out == START):
            e = ev[j][seq[n, j]]
            if not e.members or t < e.time + self.dur[j] + h_f - EPS:
                continue
            if heard is None:
                heard = self.kb.heard_matrix()
            if all(t - heard[n, m] >= h_f - EPS for m in e.members):
                out[n, j] = ABORT
        return out

    def _event(self, j: int, kind: int, t: float, members=(), observers=()) -> None:
        seq = self.kb.task_event(j, kind, t, members=members, observers=observers)
        if seq >= self.kind_tab.shape[1]:
            grown = np.full((self.T, 2 * self.kind_tab.shape[1]), -1, np.int64)
            grown[:, :self.kind_tab.shape[1]] = self.kind_tab
            self.kind_tab = grown
        self.kind_tab[j, seq] = kind
        self.rec_kind[j] = kind

    def home_ok(self, i: int) -> bool:
        """D7 per replica (``home_rule="replica"``): robot i's replica shows every task it knows finished (t >= H is
        checked by the env)."""
        seq = self.kb.v[i, self.L.TASK]
        known = np.flatnonzero(seq >= 0)
        return bool((self.kind_tab[known, seq[known]] == FINISH).all())

    # ---- env -> replicas ----------------------------------------------------------------------------------------
    def _observe(self, env: C3Env, t: float, events) -> None:
        """Physical task events and on-site observations up to now (called at every epoch)."""
        st_node = self.L.station
        rel = env.release
        for j in range(self.T):
            if rel[j] <= t + EPS and self.rec_kind[j] < 0:
                self._event(j, RELEASE, float(rel[j]), observers=[st_node])
            task = env.task_dic[j]
            if task["restarts"] > self.rec_restarts[j]:  # aborted and reset at a failure detection
                self.rec_restarts[j] = task["restarts"]
                if self.rec_kind[j] != START:
                    self._event(j, START, t)
                start_ev = self.kb.facts.task_events[j][-1]
                crew = [int(m) for m in start_ev.members]
                obs = [m for m in crew if not env.dead[m]]  # the co-workers saw it stop (on site)
                if self.cfg.ideal:  # good comms: the detection is global (the env's failure detector)
                    self._event(j, ABORT, t, members=crew, observers=np.flatnonzero(np.append(~env.dead, True)))
                else:  # T2: only the surviving crew; a lone worker's abort is inferred by rule 4.6(c)
                    self._event(j, ABORT, t, members=crew, observers=obs)
                if obs:  # on-site evidence (4.6 a): the halted co-worker is absent
                    for m in crew:
                        if env.dead[m]:
                            self.kb.append_log(min(obs), ABSENT, m, j, t)
                            self.st.absents += 1
            started = task["feasible_assignment"] and task["time_start"] <= t + EPS and np.isfinite(task["time_start"])
            if started and self.rec_kind[j] in (RELEASE, ABORT):
                coal = tuple(int(m) for m in task["coalition"])
                self._event(j, START, float(task["time_start"]), members=coal, observers=coal)
            if task["finished"] and self.rec_kind[j] != FINISH:
                coal = tuple(int(m) for m in task["coalition"])
                if self.rec_kind[j] != START:
                    self._event(j, START, float(task["time_start"]), members=coal, observers=coal)
                self._event(j, FINISH, float(task["time_finish"]), observers=coal)
        for e in events:
            if e.kind == "wasted":  # the robot sees the task's status on arrival
                for i in e.agents:
                    for j in e.tasks:
                        self.kb.observe_task([int(i)], int(j))
            elif e.kind == "abandon" and e.get("reason") == "partner-failed":  # good comms: the env's detector
                for i in e.agents:
                    for j in e.tasks:
                        self.kb.append_log(int(i), ABANDON, int(i), int(j), t)

    def _mode(self, env: C3Env, i: int, t: float):
        """(mode, target, t_ref) of a live robot from the env's physical state (its own causal view)."""
        a = env.agent_dic[i]
        if env.rallying[i]:
            return RALLY, -1, t
        c = a["current_task"]
        arr = a["arrival_time"][-1]
        dep, tau = a["leg"][0], a["leg"][1]
        if arr > t + EPS:
            moved = (t - dep) - env._stalled(i, dep, t)
            eta = t + max(tau - max(moved, 0.0), 0.0) * self.kappa
            if c < 0:
                return HOMING, -1, eta
            return (TRAVEL, -1, eta) if env._leave[i] else (TRAVEL, c, eta)  # released: not committed to c
        if c < 0:
            return IDLE, -1, t
        task = env.task_dic[c]
        if task["finished"] or env._left[i] is not None or i not in task["members"]:
            return IDLE, -1, t
        if task["feasible_assignment"] and task["time_start"] <= t + EPS and i in task["coalition"]:
            return WORK, c, float(task["time_start"]) + float(env.dur_nom[c])
        return WAIT, c, float(arr)

    def _heartbeats(self, env: C3Env, t: float, pos: np.ndarray) -> None:
        from cbba_sota.dyn.protocols import hub_fraction

        kb = self.kb
        for i in range(self.A):
            if env.dead[i]:
                continue
            mode, target, t_ref = self._mode(env, i, t)
            if mode == HOMING:
                p = self.inst.depot[i]
            elif mode == TRAVEL and target < 0:  # released traveller: free at its destination at its ETA
                p = self.inst.loc[env.agent_dic[i]["current_task"]]
            else:
                p = pos[i]
            if self.arm.theta is not None:
                h = hub_fraction(kb.belief(i, t), self.cfg.member_window)
                if h is not None:
                    self.hub[i] = h
            kb.heartbeat(i, t, mode, target=target, t_ref=t_ref, pos=p, route_key=int(self.adopted_key[i]),
                         route=tuple(env.route(i)), hub_frac=float(self.hub[i]))
        kb.heartbeat(self.L.station, t, STATION, pos=self.station_xy)

    def _positions(self, env: C3Env, t: float) -> np.ndarray:
        pos = np.empty((self.A + 1, 2))
        for i in range(self.A):
            pos[i] = env.position(i, t)
        pos[self.A] = self.station_xy
        return pos

    # ---- replicas -> env ----------------------------------------------------------------------------------------
    def _adopt(self, env: C3Env, b: int, ver, kinds: np.ndarray, t: float) -> None:
        mode, target, _ = self._mode(env, b, t)
        c = env.agent_dic[b]["current_task"]
        follow = self.cfg.ideal or self.cfg.adopt == "follow"
        if env._leave[b]:  # released traveller: a member again if the new version plans it there
            head = c if c in ver.tasks and env.leave_on_arrival(b, False) else -1
        else:
            head = target if mode in (TRAVEL, WAIT, WORK) else -1
            if follow and mode in (TRAVEL, WAIT) and head >= 0 and head not in ver.tasks:
                # the adopted plan releases the robot from its current commitment (the planner's G4 fallback, or a
                # leader that planned it elsewhere): a waiting robot leaves now, a travelling one on arrival (D2)
                if mode == WAIT and env.abandon(b):
                    self.kb.append_log(b, ABANDON, b, head, t)
                    self.st.releases += 1
                elif mode == TRAVEL and env.leave_on_arrival(b):
                    self.st.releases += 1
                head = -1
        known = kinds[b]
        r = [int(x) for x in ver.tasks if x != head and known[x] not in (START, FINISH)
             and env.release[x] <= t + EPS]
        env.set_route(b, r)
        self.adopted_key[b] = self.L.route_key(ver.epoch, ver.author)
        self.st.adoptions += 1

    def _prune(self, env: C3Env, kinds: np.ndarray) -> None:
        """Every robot drops the route tasks its own replica knows are started or finished."""
        for i in range(self.A):
            if env.dead[i]:
                continue
            r = env._route[i]
            if r and any(kinds[i, x] in (START, FINISH) for x in r):
                env.set_route(i, [x for x in r if kinds[i, x] not in (START, FINISH)])

    def _present(self, env: C3Env, j: int, t: float) -> list[int]:
        return [int(m) for m in env.task_dic[j]["members"] if not env.dead[m]
                and env.get_arrival_time(m, j) <= t + EPS and env._left[m] is None]

    def _leases(self, env: C3Env, t: float) -> None:
        for i in range(self.A):
            if env.dead[i]:
                continue
            mode, j, wait_since = self._mode(env, i, t)
            if mode != WAIT:
                continue
            bel = self.kb.belief(i, t)
            d = lease_check(bel, i, j, wait_since, present=self._present(env, j, t), cfg=self.lease,
                            failed=self.runner._failed(bel))
            if d.abandon and env.abandon(i):
                publish_abandon(self.kb, i, j, t, d)
                self.st.abandons_lease += 1
                self.st.absents += len(d.absent)

    def _idle(self, env: C3Env, t: float) -> None:
        """Rally (4.3): an idle robot outside the station component with an empty route moves toward the station;
        it stops once it hears the station again (or arrives)."""
        from cbba_sota.dyn.protocols import outside_station_component

        if self.cfg.ideal or not self.arm.rally:
            return
        w = self.cfg.member_window
        for i in range(self.A):
            if env.dead[i]:
                if env.rallying[i]:
                    env.rally_stop(i)
                continue
            if env.rallying[i]:
                bel = self.kb.belief(i, t)
                if env._route[i] or not outside_station_component(bel, w) or \
                        np.linalg.norm(env.position(i, t) - self.station_xy) < 1e-9:
                    env.rally_stop(i)
                continue
            env._ctx = i
            if not env.idle_here(i) or env.may_go_home():  # it goes home at this epoch
                continue
            if outside_station_component(self.kb.belief(i, t), w) and env.rally_start(i, self.station_xy):
                self.st.rally_starts += 1

    # ---- the per-tick protocol ----------------------------------------------------------------------------------
    def _tick(self, env: C3Env, k: int, t: float) -> None:
        rn = self.runner
        alive = np.append(~env.dead, True)
        pos = self._positions(env, t)
        self._heartbeats(env, t, pos)
        rn.deliver(k, t, pos, alive)
        if k == 0:
            rn.briefing(k, t, pos, alive)
        rn.plan(k, t, pos, alive)
        kinds = self.kinds(t)
        for b, ver in rn.adoptions(alive).items():
            self._adopt(env, int(b), ver, kinds, t)
        self._prune(env, kinds)
        self._leases(env, t)
        self._idle(env, t)
        self._heartbeats(env, t, pos)
        rn.send(k, t, pos, alive)
        self.st.ticks += 1

    def _finished(self, env: C3Env) -> bool:
        if not all(task["finished"] for task in env.task_dic.values()):
            return False
        for i, a in env.agent_dic.items():
            if env.dead[i]:
                continue
            if a["current_task"] >= 0 or env.away[i] or env.rallying[i]:
                return False
        return True

    def on_events(self, env: C3Env, t: float, events) -> bool:
        self._observe(env, t, events)
        k = round(t / TICK)
        n0 = self.st.plan_calls
        if abs(k * TICK - t) < 1e-9 and k > self.k:
            if self.k < 0 and k > 0:
                raise RuntimeError(f"the first controller epoch is t={t}, not 0")
            self.k = k
            self._tick(env, k, t)
        elif self.tr.ideal:
            self.tr.sync(np.append(~env.dead, True))
        if self.k >= 0:
            self._prune(env, self.kinds(t))
        if not self.done and self._finished(env):
            self.done = True
        if not self.done and self.wake_at <= t + EPS:
            self.wake_at = (math.floor(t / TICK + 1e-6) + 1) * TICK
            env.request_wakeup(self.wake_at)
        return self.st.plan_calls > n0


# ================================================================================================ one episode
def run_c3(setting: str, split: str, i: int, crn_seed: int, cell: str, cfg: C3Config) -> dict:
    """One C3 episode on dev instance ``i`` of ``setting`` (ground truth ``C3Env``); returns a result row."""
    from cbba_sota.bench import configs
    from cbba_sota.dyn import perturb
    from cbba_sota.dyn.comm import CRNKey, comm_range

    if split == "test":
        raise PermissionError("Track D C3 runs use dev (and validation) only; the test split is frozen")
    w0, c0 = time.time(), time.process_time()
    s = configs.get(setting)
    real = perturb.realize_instance(setting, split, i, crn_seed, cell)
    env = make_c3_env(s.instance_path(split, i), real, coalition="arrival", obs="causal", lease_L=200.0,
                      abandon_on_failure=cfg.ideal)
    inst = env.nominal_instance()
    key = CRNKey(s.index, int(s.seed(split, i)), int(crn_seed))
    ctl = C3Controller(env, inst, real, cfg, key)
    ep = env.run_plan(ctl, max_time=cfg.max_time, skip_finished=False)
    summ = ctl.runner.summary()
    st = ctl.st
    ms = ep.makespan
    span = float(ms) if ms is not None else float(env.current_time)
    A = inst.n_agents
    cpu = np.array(st.plan_cpu) if st.plan_cpu else np.zeros(1)
    return {
        "setting": setting, "split": split, "inst": int(i), "instance_key": int(s.seed(split, i)),
        "seed": int(crn_seed), "cell": cell, "family": real.cell.family, "arm": cfg.arm,
        "rho": math.inf if cfg.ideal else float(cfg.rho), "lease_g": cfg.lease_g, "lease_L": cfg.lease_L,
        "config": {k: v for k, v in asdict(cfg).items()},
        "makespan": None if ms is None else float(ms), "success": bool(ep.success),
        "completion": float(ep.completion), "makespan_or_cap": float(ep.makespan_or_cap),
        "excluded": bool(real.excluded), "H": float(real.H), "n_failures": int(np.isfinite(real.fail_onset).sum()),
        "travel": float(ep.travel), "rally_dist": float(env.rally_dist), "wait": float(ep.wait),
        "wasted_trips": int(ep.wasted_trips), "abandons": int(ep.abandons), "restarts": int(ep.restarts),
        "abandons_lease": st.abandons_lease, "releases": st.releases, "absents": st.absents,
        "rally_starts": st.rally_starts, "adoptions": st.adoptions,
        "plan_calls": st.plan_calls, "plan_calls_off_station": st.plan_calls_off_station,
        "plan_cpu_ms_p50": float(np.median(cpu) * 1e3), "plan_cpu_ms_p95": float(np.quantile(cpu, 0.95) * 1e3),
        "plan_cpu_s": float(cpu.sum()), "versions": summ["versions"], "versions_off_station":
            summ["versions_off_station"], "station_frac": summ["station_frac"],
        "frames": summ["frames"], "bytes": summ["bytes"], "delta_bytes": summ["delta_bytes"],
        "beacon_bytes": summ["beacon_bytes"], "truncated": summ["truncated"],
        "bytes_per_robot_time": float(summ["bytes"]) / max(A * span, 1e-9),
        "delta_bytes_per_robot_time": float(summ["delta_bytes"]) / max(A * span, 1e-9),
        "frames_per_robot_time": float(summ["frames"]) / max(A * span, 1e-9),
        "comm_R": comm_range(cfg.rho, A) if not cfg.ideal else math.inf, "ticks": st.ticks,
        "world": env.world, "c3_hash": c3_hash(), "env_hash": code_hash(),
        "wall_s": time.time() - w0, "cpu_s": time.process_time() - c0,
    }
