"""Plan-following event executor for DynHeteroMRTA-X: a fast *surrogate* world for planner pilots.

Ground truth is the env (``DynTaskEnvX``, ``cbba_sota/dyn/env.py``, driven by ``controller.PlanController``). This
executor is the hand-written world that spec Section 8 allows for speed only after gate K0 (executor vs env
replay, ``scripts/trackD_k0.py``); every number it produces is labelled with ``world`` =
``Executor[surrogate]@<hash of this file>``, and K0 holds only for the env / controller / executor code it was run
on. Where the env and this file differ, the env wins and this file is aligned. Aligned on day 2 (K0): the belief
conventions of ``PlanController.belief`` (started = in progress, committed = departed and not started, a finish at
this epoch frees its members now, failed robots at their depot), commitment at the first departure with the
plan's coalition, ending at the task's start (a wasted trip also leaves it), head locks only on committed tasks, a
halted traveller looks stalled until detected, a released commitment makes a traveller leave on arrival
(``released``, the env's ``leave_on_arrival``), a zero-length leg is not stalled, the D7 end time, and the lease =
failure detector rule for frozen partners (survivors leave unless the departed members still cover).

Semantics (spec 3.2, 4.3; good communication = instant, lossless, global knowledge):
- Robots follow their adopted route in key order and depart when their previous task finishes (at t = 0 from the
  depot, or as soon as work appears). A task starts when the members present cover its requirement (D2, arrival
  order); later arrivals leave (a wasted trip). Travel follows the LoRR delay calendar (stalls while moving),
  durations are the realized ones.
- A task is committed once any member departs to it; its members and key are then frozen (spec 4.1).
- Releases (D1): a task is known from its release time on; release epochs are events.
- Failures (D6): a robot halts at its failure time; a task it works on cannot finish and one it is expected at
  cannot start. At detection (h later, heartbeat timeout, default 5 ticks) the robot leaves every coalition, an
  aborted task restarts from scratch when re-covered, and if the remaining present members do not cover the task
  they abandon it and continue with their routes (good comms: the lease is the failure detector, spec 4.3).
- Termination (D7/D8): with H = 0 (static release) a robot with an empty route heads home at once (native env
  behaviour); with H > 0 it waits in place and heads home once t >= H and every known task is finished. Makespan
  = the time every task is finished and every live robot is home. A robot heading home can take new work only once
  it is home (as in ``DynTaskEnvX``); a robot that arrives at a task it is no longer planned for leaves (wasted).
- The ``idle`` trigger fires when a freed robot has an empty route, may not go home (D7) and open tasks exist (with
  H = 0 robots go home, so it never fires; as ``DynTaskEnvX``).

The policy is called once per event batch (all events at one time) with the set of event kinds, and may return a
new plan to adopt (``DynPlan``); the executor builds the policy's belief (``belief``) with the declared predictors.
CRN: ``make_realization`` draws releases, durations, delay calendars and failures from keys (instance, CRN seed,
stream, entity) independent of any decision with the formulas of ``pilots/trackD/dyn_env.py``;
``Realization.from_perturb`` takes agent A's ``cbba_sota.dyn.perturb`` draws instead (the env's CRN streams), which
is what any comparison with the env must use.
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from cbba_sota.dyn.planner import AT_DEPOT, AT_POINT, DynPlan, PlanState, travel_from
from cbba_sota.hetero.instance import Instance

__all__ = ["WORLD", "Executor", "Policy", "Realization", "code_hash", "kappa_of", "make_realization"]

TICK = 0.1  # env dt
MAX_TIME = 200.0
WORLD = "Executor[surrogate]"


def _hash_file() -> str:
    import hashlib
    from pathlib import Path

    return hashlib.sha1(Path(__file__).read_bytes()).hexdigest()[:10]


_CODE_HASH = _hash_file()


def code_hash() -> str:
    """Short hash of this file as imported (the executor world's code; rows record it)."""
    return _CODE_HASH

IDLE, MOVE, WAIT, WORK, HOME, DEAD = range(6)
STRUCTURAL = frozenset({"release", "orphan", "idle", "membership"})


# --- realization (CRN) ------------------------------------------------------------------------------------------


def _rng(*keys: int) -> np.random.Generator:
    return np.random.default_rng([int(k) for k in keys])


def kappa_of(p_delay: float, min_delay: int, max_delay: int) -> float:
    """Travel predictor factor 1 / (1 - expected stall fraction) of the LoRR delay model (renewal argument)."""
    if p_delay <= 0:
        return 1.0
    ed = 0.5 * (min_delay + max_delay)
    frac = ed / (1.0 / p_delay - 1.0 + ed)
    return 1.0 / (1.0 - frac)


@dataclass(frozen=True)
class Realization:
    release: np.ndarray  # [T] release times
    dur_real: np.ndarray  # [T] realized durations
    delays: tuple  # per robot (starts, ends) of stall intervals
    fail_t: np.ndarray  # [A] failure onset (inf: never)
    horizon: float  # release window end H (0: static)
    kappa: float = 1.0
    meta: dict = field(default_factory=dict)

    @classmethod
    def from_perturb(cls, real) -> Realization:
        """The same draws as agent A's ``cbba_sota.dyn.perturb.Realization`` (the CRN streams of the env)."""
        return cls(np.asarray(real.release, float), np.asarray(real.dur_real, float), tuple(real.delays),
                   np.asarray(real.fail_onset, float), float(real.H), float(real.kappa()),
                   {"from": "perturb", **real.summary()})


def make_realization(inst: Instance, inst_key: int, crn_seed: int = 0, *, release: str = "none", dod: float = 0.5,
                     horizon: float = 0.0, batch_first: int = 20, batch_size: int = 20, batch_period: float = 10.0,
                     dur_sigma: float = 0.0, p_delay: float = 0.0, min_delay: int = 1, max_delay: int = 4,
                     fail_p: float = 0.0, fail_window: float = 0.0, t_max: float = 400.0) -> Realization:
    """One CRN realization. Streams (key = (crn_seed, inst_key, stream[, entity])): 1 delays (per robot),
    2 durations, 3 releases, 4 failures (per robot). ``release``: none | batch (R1) | poisson (R2/R3, degree of
    dynamism ``dod``, arrivals uniform order statistics on [0, horizon])."""
    T, A = inst.n_tasks, inst.n_agents
    if release == "none":
        rel = np.zeros(T)
    elif release == "batch":
        ids = np.arange(T)
        rel = np.where(ids <= batch_first, 0.0, np.ceil((ids - batch_first) / batch_size) * batch_period)
    elif release == "poisson":
        rng = _rng(crn_seed, inst_key, 3)
        n_dyn = round(dod * T)
        rel = np.zeros(T)
        dyn = rng.choice(T, n_dyn, replace=False)
        rel[dyn] = np.sort(rng.uniform(0.0, horizon, n_dyn))
    else:
        raise ValueError(release)
    dur = np.asarray(inst.dur, float)
    if dur_sigma > 0:
        z = _rng(crn_seed, inst_key, 2).standard_normal(T)
        dur = dur * np.exp(dur_sigma * z - 0.5 * dur_sigma ** 2)
    else:
        dur = dur.copy()
    delays = []
    n_ticks = round(t_max / TICK)
    for i in range(A):
        if p_delay <= 0:
            delays.append((np.zeros(0), np.zeros(0)))
            continue
        rng = _rng(crn_seed, inst_key, 1, i)
        starts, ends, k = [], [], 0
        while True:
            k += int(rng.geometric(p_delay)) - 1
            if k >= n_ticks:
                break
            d = int(rng.integers(min_delay, max_delay + 1))
            starts.append(k * TICK)
            ends.append((k + d) * TICK)
            k += d
        delays.append((np.array(starts), np.array(ends)))
    fail = np.full(A, np.inf)
    if fail_p > 0:
        for i in range(A):
            rng = _rng(crn_seed, inst_key, 4, i)
            u, onset = rng.random(), rng.uniform(0.0, fail_window)
            if u < fail_p:
                fail[i] = onset
    H = float(rel.max()) if release != "none" else 0.0
    if release == "poisson":
        H = float(horizon)
    meta = {"inst_key": inst_key, "crn_seed": crn_seed, "release": release, "dod": dod, "horizon": horizon,
            "dur_sigma": dur_sigma, "p_delay": p_delay, "min_delay": min_delay, "max_delay": max_delay,
            "fail_p": fail_p, "fail_window": fail_window}
    return Realization(rel, dur, tuple(delays), fail, H, kappa_of(p_delay, min_delay, max_delay), meta)


# --- policy protocol ----------------------------------------------------------------------------------------------


class Policy(Protocol):
    name: str

    def on_event(self, ex: Executor, t: float, kinds: frozenset[str]) -> DynPlan | None: ...


# --- executor ------------------------------------------------------------------------------------------------------


class Executor:
    """One episode of plan following on one realization (see the module docstring)."""

    def __init__(self, inst: Instance, world: Realization, detect_h: float = 5 * TICK, max_time: float = MAX_TIME,
                 predictor: str = "kappa", idle: str = "env"):
        """``idle``: "env" = the idle trigger of ``DynTaskEnvX`` (a freed robot with an empty route that may not go
        home); "eager" = any freed robot with an empty route while open tasks exist (a planner stress option)."""
        if idle not in ("env", "eager"):
            raise ValueError(idle)
        self.idle_rule = idle
        self.inst, self.world = inst, world
        self.detect_h, self.max_time, self.predictor = float(detect_h), float(max_time), predictor
        T, A = inst.n_tasks, inst.n_agents
        self.T, self.A = T, A
        self.loc, self.depot, self.speed = inst.loc, inst.depot, float(inst.speed)
        self.t = 0.0
        # tasks
        self.known = np.zeros(T, bool)
        self.done = np.zeros(T, bool)
        self.started = np.zeros(T, bool)
        self.committed = np.zeros(T, bool)
        self.start_t = np.full(T, np.nan)
        self.finish_t = np.full(T, np.nan)
        self.frozen: list[tuple[int, ...]] = [()] * T
        self.planned: list[tuple[int, ...]] = [()] * T
        self.key = np.full(T, np.nan)
        self.present: list[dict[int, float]] = [{} for _ in range(T)]
        self.working: list[tuple[int, ...]] = [()] * T
        self.fin_ver = np.zeros(T, np.int64)
        self.key_floor = -1.0
        # robots
        self.mode = np.full(A, HOME, np.int64)
        self.tgt = np.full(A, -1, np.int64)  # task of MOVE/WAIT/WORK (-1: heading home)
        self.leg: list[tuple | None] = [None] * A  # (dep, tau, origin xy, dest xy, realized arrival)
        self.at = np.full(A, AT_DEPOT, np.int64)  # where an IDLE/HOME robot is (task id, AT_DEPOT, AT_POINT)
        self.xy = np.array(inst.depot, float)
        self.route: list[list[int]] = [[] for _ in range(A)]
        self.ver = np.zeros(A, np.int64)
        self.phys_dead = np.zeros(A, bool)
        self.known_dead = np.zeros(A, bool)
        self.delays = [(np.asarray(s, float), np.asarray(e, float)) for s, e in world.delays]  # stall calendars
        self.legs: list[list[tuple[float, int, float]]] = [[] for _ in range(A)]  # (departure, task | -1, arrival)
        self.released = np.zeros(A, bool)  # travelling to a task whose commitment was released: leaves on arrival
        self.home_t = np.zeros(A)
        # bookkeeping
        self.heap: list = []
        self.seq = 0
        self.n_events = 0
        self.log: list[dict] = []
        self.wasted = 0
        self.aborts = 0
        self.abandons = 0
        self.aborted = np.zeros(T, bool)
        self.redirects = 0
        self.travel = 0.0

    # ---- physics -------------------------------------------------------------------------------------------------
    def arrive(self, i: int, dep: float, tau: float) -> float:
        starts, ends = self.delays[i]
        if starts.size == 0 or tau <= 0:  # no stalls, or no movement: not stalled (as the env)
            return dep + tau
        t, rem = dep, tau
        k = int(np.searchsorted(ends, t, side="right"))
        while k < starts.size:
            s, e = starts[k], ends[k]
            if s <= t:
                t = e
            elif t + rem <= s:
                break
            else:
                rem -= s - t
                t = e
            k += 1
        return t + rem

    def stalled(self, i: int, a: float, b: float) -> float:
        starts, ends = self.delays[i]
        if starts.size == 0 or b <= a:
            return 0.0
        return float(np.clip(np.minimum(ends, b) - np.maximum(starts, a), 0.0, None).sum())

    def _point(self, i: int, t: float) -> np.ndarray:
        """Position of a moving robot at ``t``."""
        dep, tau, o, d, arr = self.leg[i]
        if t >= arr:
            return np.array(d, float)
        moved = (t - dep) - self.stalled(i, dep, t)
        frac = min(max(moved / tau, 0.0), 1.0) if tau > 0 else 1.0
        return o + frac * (d - o)

    def _origin(self, i: int, t: float) -> tuple[int, np.ndarray]:
        """(at code, xy) of robot ``i`` for a departure at ``t``."""
        if self.mode[i] == MOVE:  # stopped on its way (only when waiting in place is forced)
            return AT_POINT, self._point(i, t)
        if self.mode[i] in (WAIT, WORK):
            j = int(self.tgt[i])
            return j, self.loc[j]
        return int(self.at[i]), self.xy[i]

    def _travel(self, i: int, at: int, xy: np.ndarray, j: int) -> float:
        """Nominal travel time to task ``j`` (``j = -1``: depot), bit-exact with ``Instance.tt``/``da``."""
        if j < 0:
            if at == AT_DEPOT:
                return 0.0
            if at >= 0:
                return float(self.inst.da[i, at])
            return float(np.linalg.norm(xy - self.depot[i]) / self.speed)
        if at == AT_DEPOT:
            return float(self.inst.da[i, j])
        if at >= 0:
            return float(self.inst.tt[at, j])
        return float(travel_from(xy, self.loc[j:j + 1], self.speed)[0])

    def _push(self, t: float, prio: int, kind: str, a: int, b: int = 0) -> None:
        heapq.heappush(self.heap, (t, prio, self.seq, kind, a, b))
        self.seq += 1

    def _depart(self, i: int, t: float, j: int) -> None:
        """Robot ``i`` leaves for task ``j`` (-1: its depot) at ``t``."""
        at, xy = self._origin(i, t)
        if self.mode[i] == MOVE:
            self.redirects += 1
        if self.mode[i] == WAIT:
            self.present[int(self.tgt[i])].pop(i, None)
        tau = self._travel(i, at, xy, j)
        dest = self.depot[i] if j < 0 else self.loc[j]
        self.ver[i] += 1
        if j < 0 and tau == 0.0:
            self.legs[i].append((t, -1, t))
            self.mode[i], self.tgt[i], self.at[i] = HOME, -1, AT_DEPOT
            self.xy[i] = self.depot[i]
            self.home_t[i] = t
            return
        arr = self.arrive(i, t, tau)
        self.legs[i].append((t, j, arr))
        self.travel += float(np.linalg.norm(np.asarray(xy, float) - dest))  # distance, as the env counts it
        self.mode[i], self.tgt[i] = MOVE, j
        self.leg[i] = (t, tau, np.array(xy, float), np.array(dest, float), arr)
        self.released[i] = False
        if j >= 0:
            if not self.frozen[j]:  # first departure: the task is committed with the plan's coalition (spec 4.1)
                self.committed[j] = True
                self.frozen[j] = tuple(sorted(set(self.planned[j]) | {i}))
            elif i not in self.frozen[j]:
                self.frozen[j] = tuple(sorted(set(self.frozen[j]) | {i}))
            self._push(arr, 2, "ARR", i, int(self.ver[i]))
        else:
            self._push(arr, 3, "HOME", i, int(self.ver[i]))

    def _go_idle(self, i: int, t: float) -> None:
        """Robot ``i`` has nothing to do: native (H = 0) heads home, else waits in place (D7/D8)."""
        if self.may_go_home(t):
            if self.mode[i] == MOVE and self.tgt[i] < 0:
                return  # already heading home
            if self.mode[i] == HOME:
                return
            self._depart(i, t, -1)
        else:
            if self.mode[i] == MOVE:  # stop where it is
                at, xy = AT_POINT, self._point(i, t)
                self.ver[i] += 1
            elif self.mode[i] in (WAIT, WORK):
                at, xy = int(self.tgt[i]), self.loc[int(self.tgt[i])]
            else:
                return
            self.mode[i], self.tgt[i], self.at[i] = IDLE, -1, at
            self.xy[i] = xy

    def _all_known_done(self) -> bool:
        return bool(self.done[self.known].all())

    def may_go_home(self, t: float) -> bool:
        """D7: with H > 0 only once t >= H and every released task is finished; with H = 0 always."""
        H = self.world.horizon
        return H <= 0 or (t >= H - 1e-12 and self._all_known_done())

    def _try_start(self, j: int, t: float) -> None:
        pres = [i for i in self.present[j] if not self.phys_dead[i]]
        if pres and (self.inst.ab[pres].sum(axis=0) >= self.inst.req[j]).all():
            self.started[j] = True
            self.start_t[j] = t
            self.finish_t[j] = t + self.world.dur_real[j]
            self.working[j] = tuple(sorted(pres))
            self.frozen[j] = ()  # a started task is no longer a commitment (after an abort it is open again)
            for i in pres:
                self.mode[i] = WORK
            self.fin_ver[j] += 1
            self._push(float(self.finish_t[j]), 1, "FIN", j, int(self.fin_ver[j]))

    # ---- knowledge ---------------------------------------------------------------------------------------------
    def open_tasks(self) -> np.ndarray:
        return np.flatnonzero(self.known & ~self.committed & ~self.started & ~self.done)

    def belief(self) -> PlanState:
        """Good-communication belief at ``self.t``: global state with the declared predictors (spec 4.1)."""
        t, inst = self.t, self.inst
        A = self.A
        kap = self.world.kappa if self.predictor == "kappa" else 1.0
        ready = np.zeros(A)
        pos = np.array(self.xy, float)
        pos_task = np.array(self.at, np.int64)
        head = np.full(A, -1, np.int64)
        # PlanState conventions (as ``controller.PlanController.belief``): started = in progress, committed = a
        # member departed and not started / finished (head locks too); members only for committed tasks
        started = self.started & ~self.done
        committed = self.committed & ~self.started & ~self.done
        for i in range(A):
            m = int(self.mode[i])
            if self.known_dead[i]:  # takes no work; canonical entries (as ``PlanController.belief``)
                pos[i], pos_task[i] = self.depot[i], AT_DEPOT
                continue
            if m == MOVE and self.tgt[i] >= 0:
                j = int(self.tgt[i])
                dep, tau, _, _, _ = self.leg[i]
                st = self.stalled(i, dep, t)
                if kap == 1.0:
                    ready[i] = max(dep + st + tau, t)
                else:
                    moved = (t - dep) - st
                    ready[i] = t + max(tau - moved, 0.0) * kap
                pos[i], pos_task[i] = self.loc[j], j  # not locked if it leaves on arrival (j later in its route)
                head[i] = j if committed[j] and i in self.frozen[j] and not self.released[i] else -1
            elif m == MOVE:  # heading home: free once there
                dep, tau, _, _, _ = self.leg[i]
                st = self.stalled(i, dep, t)
                if kap == 1.0:
                    ready[i] = max(dep + st + tau, t)
                else:
                    ready[i] = t + max(tau - ((t - dep) - st), 0.0) * kap
                pos[i], pos_task[i] = self.depot[i], AT_DEPOT
            elif m == WAIT:
                j = int(self.tgt[i])
                ready[i], pos[i], pos_task[i] = self.present[j].get(i, t), self.loc[j], j
                head[i] = j if committed[j] and i in self.frozen[j] else -1
            elif m == WORK:
                j = int(self.tgt[i])
                # a task that finished at this epoch frees its members now (they are dispatched after the call)
                ready[i] = t if self.done[j] else max(self.start_t[j] + inst.dur[j], t)
                pos[i], pos_task[i] = self.loc[j], j
            elif m == HOME:
                ready[i], pos[i], pos_task[i] = t, self.depot[i], AT_DEPOT
            else:  # IDLE
                ready[i] = t
        members = [tuple(self.frozen[j]) if committed[j] else () for j in range(self.T)]
        return PlanState(inst=inst, now=t, released=self.known.copy(), done=self.done.copy(),
                         started=started, committed=committed, members=members,
                         keys=np.nan_to_num(self.key.copy(), nan=0.0), alive=~self.known_dead, ready=ready, pos=pos,
                         pos_task=pos_task, head=head, kappa=kap, key_floor=self.key_floor)

    # ---- plan adoption -------------------------------------------------------------------------------------------
    def adopt(self, plan: DynPlan, t: float) -> list[int]:
        """Adopt ``plan``; returns robots that must (re)depart now."""
        live = ~self.done & ~self.started
        released = set(plan.released)
        for j in np.flatnonzero(live):
            j = int(j)
            m = tuple(plan.members[j])
            self.planned[j] = m
            self.key[j] = plan.keys[j] if m else np.nan
            if self.committed[j]:
                if j in plan.released:
                    self.committed[j] = False
                    self.frozen[j] = ()
                elif m:
                    self.frozen[j] = m
        if np.isfinite(plan.keys).any():
            self.key_floor = max(self.key_floor, float(np.nanmax(plan.keys)))
        routes = plan.routes_for(self.A)
        go = []
        for i in range(self.A):
            if self.phys_dead[i] and self.known_dead[i]:
                self.route[i] = []
                continue
            r = [j for j in routes[i] if live[j]]
            m = int(self.mode[i])
            if m in (MOVE, WAIT) and self.tgt[i] >= 0:
                j = int(self.tgt[i])
                locked = not self.released[i] and j not in released  # as ``controller.PlanController.adopt``
                if r and r[0] == j:
                    r = r[1:]
                    self.released[i] = False  # planned there (again): a member
                elif j in r and locked:
                    raise AssertionError(f"robot {i} is committed to task {j} but the plan puts it later: {r}")
                else:  # released commitment (or no longer a member): leave (on arrival if travelling)
                    self.route[i] = r
                    if m == MOVE:  # leaves when it arrives (the env's ``leave_on_arrival``: a wasted trip)
                        self.released[i] = True
                    if m == WAIT:  # leaves the waiting place now (the env's ``abandon``)
                        self.abandons += 1
                        if self.phys_dead[i]:  # halted, not detected: believed to leave, stays in place (env)
                            self.present[j].pop(i, None)
                            self.mode[i], self.tgt[i], self.at[i] = IDLE, -1, j
                            self.xy[i] = self.loc[j]
                        else:
                            go.append(i)
                    continue
            elif m == WORK:
                j = int(self.tgt[i])
                r = [x for x in r if x != j]
            self.route[i] = r
            if not self.phys_dead[i] and r and m in (IDLE, HOME):
                go.append(i)
        return go

    def _dispatch(self, i: int, t: float) -> None:
        """Robot ``i`` is free (or redirected) at ``t``: leave for its next task, or idle / head home."""
        if self.phys_dead[i]:
            return
        r = self.route[i]
        while r and (self.done[r[0]] or self.started[r[0]]):
            r.pop(0)
        if r:
            j = r.pop(0)
            if self.mode[i] == WAIT and self.tgt[i] == j:
                return
            self._depart(i, t, j)
        else:
            if self.mode[i] == WAIT:
                self.present[int(self.tgt[i])].pop(i, None)
            self._go_idle(i, t)

    # ---- main loop -----------------------------------------------------------------------------------------------
    def run(self, policy: Policy) -> dict:
        w = self.world
        for r in np.unique(w.release):
            self._push(float(r), 5, "REL", 0)
        for i in np.flatnonzero(np.isfinite(w.fail_t)):
            self._push(float(w.fail_t[i]), 0, "FAIL", int(i))
        if w.horizon > 0:
            self._push(float(w.horizon), 7, "HORIZON", 0)
        n_replans = 0
        cpu = []
        status = "ok"
        end_t = 0.0
        while self.heap:
            t = self.heap[0][0]
            if t > self.max_time:
                status = "timeout"
                break
            self.t = t
            kinds: set[str] = set()
            freed: list[int] = []
            while self.heap and self.heap[0][0] == t:
                _, _, _, kind, a, b = heapq.heappop(self.heap)
                if kind == "FAIL":
                    self._fail(a, t)
                    self._push(t + self.detect_h, 6, "DETECT", a)
                elif kind == "FIN":
                    if b != self.fin_ver[a] or self.done[a]:
                        continue
                    self.done[a] = True
                    kinds.add("finish")
                    for i in self.working[a]:
                        if not self.phys_dead[i]:
                            freed.append(i)
                elif kind == "ARR":
                    if b != self.ver[a] or self.phys_dead[a]:
                        continue
                    j = int(self.tgt[a])
                    kinds.add("arrive")
                    # not (or no longer) expected here now: a released commitment, possibly re-planned for later
                    stale = bool(self.released[a])  # its commitment was released: it leaves (env ``_settle``)
                    self.released[a] = False
                    if self.done[j] or self.started[j] or stale:
                        self.wasted += 1  # a wasted trip: it leaves the coalition (``PlanController`` on ``wasted``)
                        self.frozen[j] = tuple(x for x in self.frozen[j] if x != a)
                        self.mode[a], self.tgt[a], self.at[a] = IDLE, -1, j
                        self.xy[a] = self.loc[j]
                        freed.append(a)
                        continue
                    self.mode[a], self.tgt[a] = WAIT, j
                    self.present[j][a] = t
                    self._try_start(j, t)
                    if self.started[j]:
                        kinds.add("start")
                elif kind == "HOME":
                    if b != self.ver[a] or self.phys_dead[a]:
                        continue
                    self.mode[a], self.tgt[a], self.at[a] = HOME, -1, AT_DEPOT
                    self.xy[a] = self.depot[a]
                    self.home_t[a] = t
                    if self.route[a]:
                        freed.append(a)
                elif kind == "REL":
                    self.known |= w.release <= t + 1e-12
                    kinds.add("release")
                elif kind == "DETECT":
                    freed += self._detect(a, t)
                    kinds.add("orphan")
                elif kind == "HORIZON":
                    kinds.add("horizon")
            open_now = self.open_tasks().size > 0
            empty = [i for i in freed if not any(not (self.done[j] or self.started[j]) for j in self.route[i])]
            if empty and open_now and (self.idle_rule == "eager" or not self.may_go_home(t)):
                kinds.add("idle")
            fk = frozenset(kinds)
            plan = policy.on_event(self, t, fk)
            go = []
            if plan is not None:
                n_replans += 1
                cpu.append(plan.cpu_s)
                go = self.adopt(plan, t)
            self.log.append({"t": t, "kinds": sorted(fk), "replan": plan is not None})
            for i in sorted(set(freed) | set(go)):
                self._dispatch(i, t)
            # D7: idle robots head home once everything known is done
            if ("horizon" in fk or "finish" in fk) and w.horizon > 0 and t >= w.horizon and self._all_known_done():
                for i in np.flatnonzero(self.mode == IDLE):
                    if not self.route[i] and not self.phys_dead[i]:
                        self._go_idle(int(i), t)
            if self.done.all() and all(self.mode[i] == HOME for i in range(self.A) if not self.phys_dead[i]):
                end_t = t  # D7: the first time every task is finished and every live robot is home (possibly
                break      # a failure onset of the last robot still on its way home, as in the env)
        alive = ~self.phys_dead
        ok = status == "ok" and bool(self.done.all()) and bool((self.mode[alive] == HOME).all())
        ms = float(max(end_t, self.home_t[alive].max(initial=0.0))) if ok else self.max_time
        if ok and ms >= self.max_time:
            ok = False
        return {"makespan": ms if ok else self.max_time, "raw_makespan": ms, "success": ok,
                "completion": float(self.done.mean()), "status": status if ok or status != "ok" else "stuck",
                "n_replans": n_replans, "cpu_s_sum": float(np.sum(cpu)) if cpu else 0.0,
                "cpu_s_p50": float(np.median(cpu)) if cpu else 0.0,
                "cpu_s_p95": float(np.percentile(cpu, 95)) if cpu else 0.0, "wasted": self.wasted,
                "aborts": self.aborts, "abandons": self.abandons, "redirects": self.redirects,
                "travel": self.travel,
                "n_failed": int(self.phys_dead.sum()), "world": f"{WORLD}@{code_hash()}"}

    def _fail(self, i: int, t: float) -> None:
        """Onset: the robot halts; a task it works on cannot finish (known at detection), one it is expected at
        cannot start (it is excluded from coverage)."""
        if self.phys_dead[i]:
            return
        self.phys_dead[i] = True
        if self.mode[i] == MOVE and self.leg[i][4] > t:  # halted mid-leg: looks stalled until detected (env D6)
            self.travel -= float(np.linalg.norm(self.leg[i][3] - self._point(i, t)))  # the untravelled rest
            s, e = self.delays[i]
            keep = s < t
            self.delays[i] = (np.append(s[keep], t), np.append(np.minimum(e[keep], t), np.inf))
            dep, j, _ = self.legs[i][-1]
            self.legs[i][-1] = (dep, j, np.inf)  # never arrives
        if self.mode[i] == WORK:
            j = int(self.tgt[i])
            if not self.done[j]:
                self.fin_ver[j] += 1  # void the finish
                self.aborted[j] = True
        self.ver[i] += 1  # pending arrival / home events are void; belief keeps the last reported state

    def _detect(self, i: int, t: float) -> list[int]:
        """Detection: the robot leaves every coalition; aborted tasks reset; uncovered tasks are abandoned by the
        present survivors (lease = failure detector). Returns the robots freed now."""
        self.known_dead[i] = True
        self.mode[i] = DEAD
        self.route[i] = []
        freed = []
        for j in range(self.T):
            touched = i in self.frozen[j] or i in self.present[j] or i in self.working[j]
            if i in self.planned[j]:
                self.planned[j] = tuple(k for k in self.planned[j] if k != i)
            if not touched:
                continue
            self.frozen[j] = tuple(k for k in self.frozen[j] if k != i)
            self.present[j].pop(i, None)
            if self.aborted[j]:  # D6: restart from scratch when re-covered
                self.aborted[j] = False
                self.aborts += 1
                self.started[j] = False
                self.start_t[j] = np.nan
                for k in self.working[j]:
                    if k != i and not self.phys_dead[k]:
                        self.mode[k] = WAIT
                self.working[j] = ()
            if self.done[j] or self.started[j]:
                continue
            self._try_start(j, t)
            # the members that departed to j (present, or travelling there; failures not yet detected included, as
            # in the env) still cover it: it starts when they arrive, nobody leaves (env ``_fail_detect``)
            dep = set(self.present[j]) | {int(k) for k in np.flatnonzero((self.mode == MOVE) & (self.tgt == j)
                                                                          & ~self.released)}
            dep = [k for k in dep if not self.known_dead[k]]
            covered = bool(dep) and bool((self.inst.ab[dep].sum(axis=0) >= self.inst.req[j]).all())
            if not self.started[j] and not covered:
                for k in list(self.present[j]):  # an undetected halted robot is believed to leave too (env)
                    self.present[j].pop(k)
                    self.frozen[j] = tuple(x for x in self.frozen[j] if x != k)
                    self.mode[k], self.tgt[k], self.at[k] = IDLE, -1, j
                    self.xy[k] = self.loc[j]
                    self.abandons += 1
                    if not self.phys_dead[k]:
                        freed.append(k)
            if not self.frozen[j]:
                self.committed[j] = False  # nobody left who departed: the task is open again
        return freed
