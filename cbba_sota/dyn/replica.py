"""Track D replicated knowledge layer: CRDT replicas, heartbeats, leases, failure suspicion (spec 4.3, 4.4, 4.6, G3).

Nodes are the robots ``0 .. A-1`` and the station, node ``A`` (``N = A + 1`` nodes). Every node holds a replica of
the shared knowledge. A replica is ONE int64 *version vector* per node, and the join is the elementwise maximum, so
the replica lattice is a product of total orders: merge is commutative, associative and idempotent (strong eventual
consistency, Shapiro et al. 2011; guarantee G3). The vector only holds version numbers; the content they name lives
in an append-only ``FactStore`` shared by the simulation (content addressing: a version number names exactly one
immutable content, so shipping the number in a simulated frame is equivalent to shipping the content, whose bytes
are charged by ``comm.ByteModel``).

Vector segments (``Layout``), each merged by max:

- ``TASK[j]``  sequence number of the newest known physical event of task j (-1 = unknown). Events per task are
  ``RELEASE`` (seq 0), then ``START`` / ``ABORT`` / ``START`` ... / ``FINISH`` (terminal). This refines the spec's
  "released: grow-only set; start and finish: earliest time wins": every observer writes the same physical fact,
  and an abort (a robot failure, D6) is a newer fact rather than an overwrite.
- ``REC[b]``   counter of robot b's newest state record (mode, target, ETA or finish, position, adopted route
  version, remaining route). Only b writes it: last-writer-wins on the owner's counter. The station (b = A) writes
  heartbeat records too. ``heartbeat`` writes a record on every state change and at least every period.
- ``ROUTE[b]`` key ``epoch * N + author`` of the newest route version for robot b: last-writer-wins on
  (Lamport epoch, author id), ACBBA's timestamp rule. An author's epoch is 1 + the largest epoch in its replica.
- ``LOG[a]``   length of author a's append-only event log (abandons, ``absent`` records of rule 4.6(a), failure
  declarations). Logs are grow-only per author and ship in order, so a prefix length is a complete digest.

The whole vector doubles as the version-vector digest used by delta gossip (``comm.GossipTransport``).

Interfaces for the world (Agent A) and the planner (Agent B) -- see also ``protocols.py``:

- the world writes local observations: ``KnowledgeBase.task_event`` (a physical task event and the nodes that
  observe it on site), ``observe_task`` (nodes that learn a task's current status on site, e.g. a late arrival),
  ``heartbeat`` (robot state records), ``append_log`` (abandons, absents), and the station learns releases with
  ``task_event(j, RELEASE, t, observers=[station])``;
- planners and robots read ``KnowledgeBase.belief(node, now)``, an immutable ``Belief`` snapshot of one replica;
- ``lease_check`` and the failure predicates below are pure functions of a ``Belief``.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

TICK = 0.1  # env dt; one comm tick
EPS = 1e-9

# robot modes carried in state records (the world maps its own states onto these)
TRAVEL, WAIT, WORK, HOMING, IDLE, RALLY, FAILED, STATION = 0, 1, 2, 3, 4, 5, 6, 7
MODE_NAMES = ("TRAVEL", "WAIT", "WORK", "HOMING", "IDLE", "RALLY", "FAILED", "STATION")
# task event kinds
RELEASE, START, FINISH, ABORT = 0, 1, 2, 3
# log event kinds
ABANDON, ABSENT, DECLARE_FAILED = 0, 1, 2


class TaskEvent(NamedTuple):
    kind: int  # RELEASE | START | FINISH | ABORT
    time: float
    members: tuple[int, ...] = ()  # coalition present at START; members at ABORT


class RobotRecord(NamedTuple):
    time: float  # author clock when written (heartbeat time)
    mode: int
    target: int  # task id, or -1
    t_ref: float  # TRAVEL: ETA; WAIT: wait start; WORK: predicted finish; otherwise the record time
    pos: tuple[float, float]
    route_key: int  # key of the adopted route version (-1: none)
    route: tuple[int, ...] = ()  # remaining route after the current target
    hub_frac: float = 1.0  # station-component fraction last observed by this robot (HALO / V3 gate)


class RouteVersion(NamedTuple):
    robot: int
    epoch: int
    author: int
    tasks: tuple[int, ...]  # full route in key order, committed head first
    keys: tuple[float, ...] | None  # optional global key of each task (G1); None if the planner does not use keys
    time: float


class LogEvent(NamedTuple):
    kind: int  # ABANDON | ABSENT | DECLARE_FAILED
    robot: int  # the robot abandoning / declared absent / declared failed
    task: int  # -1 for DECLARE_FAILED
    time: float
    author: int


@dataclass(frozen=True)
class Layout:
    """Segment layout of the replica vector for A robots, T tasks and K traits."""

    n_robots: int
    n_tasks: int
    n_traits: int = 5

    @property
    def A(self) -> int:
        return self.n_robots

    @property
    def T(self) -> int:
        return self.n_tasks

    @property
    def N(self) -> int:
        return self.n_robots + 1

    @property
    def station(self) -> int:
        return self.n_robots

    @property
    def TASK(self) -> slice:
        return slice(0, self.T)

    @property
    def REC(self) -> slice:
        return slice(self.T, self.T + self.N)

    @property
    def ROUTE(self) -> slice:
        return slice(self.T + self.N, self.T + self.N + self.A)

    @property
    def LOG(self) -> slice:
        return slice(self.T + self.N + self.A, self.T + 2 * self.N + self.A)

    @property
    def D(self) -> int:
        return self.T + 2 * self.N + self.A

    def bottom(self) -> np.ndarray:
        """Least element: nothing known, empty logs."""
        v = np.full(self.D, -1, np.int64)
        v[self.LOG] = 0
        return v

    def segment_of(self) -> np.ndarray:
        """Segment id per vector entry: 0 TASK, 1 REC, 2 ROUTE, 3 LOG."""
        s = np.empty(self.D, np.int8)
        for sid, sl in enumerate((self.TASK, self.REC, self.ROUTE, self.LOG)):
            s[sl] = sid
        return s

    def route_key(self, epoch: int, author: int) -> int:
        return int(epoch) * self.N + int(author)

    def split_key(self, key: int) -> tuple[int, int]:
        return int(key) // self.N, int(key) % self.N


def join(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """The CRDT merge: elementwise maximum of two version vectors (or stacks of them)."""
    return np.maximum(a, b)


class FactStore:
    """Append-only content named by replica versions. Shared by all nodes of one episode (simulation device)."""

    def __init__(self, layout: Layout):
        self.layout = layout
        self.task_events: list[list[TaskEvent]] = [[] for _ in range(layout.T)]
        self.records: list[list[RobotRecord]] = [[] for _ in range(layout.N)]
        self.routes: dict[tuple[int, int], RouteVersion] = {}  # (robot, key) -> version
        self.logs: list[list[LogEvent]] = [[] for _ in range(layout.N)]
        self.rec_time = np.full((layout.N, 64), -np.inf)  # rec_time[b, counter] (vectorized heard-time lookups)

    def add_record(self, b: int, rec: RobotRecord) -> int:
        recs = self.records[b]
        recs.append(rec)
        ctr = len(recs) - 1
        if ctr >= self.rec_time.shape[1]:
            grown = np.full((self.layout.N, 2 * self.rec_time.shape[1]), -np.inf)
            grown[:, :self.rec_time.shape[1]] = self.rec_time
            self.rec_time = grown
        self.rec_time[b, ctr] = rec.time
        return ctr

    def add_task_event(self, j: int, ev: TaskEvent) -> int:
        """Append a physical event of task j (idempotent for a repeat of the newest event); return its seq."""
        evs = self.task_events[j]
        if evs and evs[-1] == ev:
            return len(evs) - 1
        last = evs[-1].kind if evs else None
        ok = {None: (RELEASE,), RELEASE: (START,), START: (FINISH, ABORT), ABORT: (START,), FINISH: ()}[last]
        if ev.kind not in ok:
            raise ValueError(f"task {j}: illegal transition {last} -> {ev.kind}")
        evs.append(ev)
        return len(evs) - 1


@dataclass(frozen=True)
class HeartbeatConfig:
    period: float = 1.0  # spec 4.4 / 4.7; with ideal comms use TICK so the detector sees every tick

    def due(self, last_time: float, now: float) -> bool:
        return now - last_time >= self.period - EPS


class KnowledgeBase:
    """Replicas of all N nodes (``v``: [N, D] int64) plus the shared ``FactStore``.

    Only the owning node writes a record, a log entry or an authored route version; physical task events are
    written by the world for the nodes that observe them. Transports (``comm.py``) move versions between rows.
    """

    def __init__(self, layout: Layout, heartbeat: HeartbeatConfig | None = None):
        self.layout = layout
        self.facts = FactStore(layout)
        self.v = np.tile(layout.bottom(), (layout.N, 1))
        self.clock = np.zeros(layout.N, np.int64)  # Lamport clock per node (route epochs)
        self.hb = heartbeat or HeartbeatConfig()
        self._last_sig: dict[int, tuple] = {}
        self.n_versions = np.zeros(layout.N, np.int64)  # route versions authored per node

    # ---------------------------------------------------------------- writes (local observations)
    def task_event(self, j: int, kind: int, time: float, members=(), observers=()) -> int:
        seq = self.facts.add_task_event(j, TaskEvent(int(kind), float(time), tuple(int(m) for m in members)))
        obs = np.asarray(list(observers), np.int64)
        if obs.size:
            self.v[obs, j] = np.maximum(self.v[obs, j], seq)
        return seq

    def observe_task(self, nodes, j: int) -> None:
        """``nodes`` learn the current physical status of task j (on-site observation)."""
        seq = len(self.facts.task_events[j]) - 1
        if seq >= 0:
            obs = np.asarray(list(nodes), np.int64)
            self.v[obs, j] = np.maximum(self.v[obs, j], seq)

    def write_record(self, b: int, rec: RobotRecord) -> int:
        ctr = self.facts.add_record(b, rec)
        self.v[b, self.layout.REC.start + b] = ctr
        self._last_sig[b] = (rec.mode, rec.target, rec.route_key, rec.route)
        return ctr

    def heartbeat(self, b: int, now: float, mode: int, target: int = -1, t_ref: float | None = None,
                  pos=(0.0, 0.0), route_key: int = -1, route=(), hub_frac: float = 1.0, force: bool = False) -> bool:
        """Write b's state record if its state changed or the heartbeat is due. Returns True if written."""
        sig = (int(mode), int(target), int(route_key), tuple(int(x) for x in route))
        recs = self.facts.records[b]
        if not force and recs and self._last_sig.get(b) == sig and not self.hb.due(recs[-1].time, now):
            return False
        self.write_record(b, RobotRecord(float(now), sig[0], sig[1], float(now if t_ref is None else t_ref),
                                         (float(pos[0]), float(pos[1])), sig[2], sig[3], float(hub_frac)))
        return True

    def append_log(self, author: int, kind: int, robot: int, task: int, time: float) -> int:
        log = self.facts.logs[author]
        log.append(LogEvent(int(kind), int(robot), int(task), float(time), int(author)))
        self.v[author, self.layout.LOG.start + author] = len(log)
        return len(log)

    def author_routes(self, author: int, now: float, routes: dict, keys: dict | None = None) -> int:
        """Author new route versions for the robots in ``routes`` (robot -> task sequence). Returns the epoch."""
        L = self.layout
        seen = int(self.v[author, L.ROUTE].max())
        epoch = max(int(self.clock[author]), seen // L.N if seen >= 0 else -1) + 1
        self.clock[author] = epoch
        key = L.route_key(epoch, author)
        for b, tasks in routes.items():
            kk = None if keys is None or b not in keys else tuple(float(x) for x in keys[b])
            self.facts.routes[(int(b), key)] = RouteVersion(int(b), epoch, author, tuple(int(x) for x in tasks), kk,
                                                             float(now))
            self.v[author, L.ROUTE.start + int(b)] = key
        self.n_versions[author] += len(routes)
        return epoch

    # ---------------------------------------------------------------- merges (used by transports and tests)
    def merge_into(self, dst, src_vec: np.ndarray) -> None:
        self.v[dst] = np.maximum(self.v[dst], src_vec)

    def sync(self, nodes=None) -> None:
        """Global join over ``nodes`` (default all): the ideal-comm transport and the t = 0 briefing."""
        idx = np.arange(self.layout.N) if nodes is None else np.asarray(list(nodes), np.int64)
        if idx.size:
            self.v[idx] = self.v[idx].max(axis=0)

    def heard_matrix(self) -> np.ndarray:
        """[N, N] author time of the newest record of node b known to node n (-inf if none)."""
        ctr = self.v[:, self.layout.REC]
        rt = self.facts.rec_time
        cols = np.arange(self.layout.N)[None, :]
        return np.where(ctr >= 0, rt[cols, np.maximum(ctr, 0)], -np.inf)

    def global_view(self) -> np.ndarray:
        """Join of all replicas = global knowledge (what an oracle would know)."""
        return self.v.max(axis=0)

    # ---------------------------------------------------------------- reads
    def belief(self, node: int, now: float) -> Belief:
        return Belief(self, int(node), float(now), self.v[node].copy())

    def belief_of_vector(self, vec: np.ndarray, node: int, now: float) -> Belief:
        return Belief(self, int(node), float(now), np.asarray(vec, np.int64).copy())


class Belief:
    """Immutable snapshot of one replica (``vec``) with typed accessors. Everything a planner may use."""

    def __init__(self, kb: KnowledgeBase, node: int, now: float, vec: np.ndarray):
        self.kb, self.node, self.now, self.vec = kb, node, now, vec
        self.layout = L = kb.layout
        self.task_seq = vec[L.TASK]
        self.rec_ctr = vec[L.REC]
        self.route_keys = vec[L.ROUTE]
        self.log_len = vec[L.LOG]
        self._logs: list[LogEvent] | None = None
        self._kind: np.ndarray | None = None

    # ---- tasks
    def task_state(self, j: int) -> TaskEvent | None:
        s = int(self.task_seq[j])
        return self.kb.facts.task_events[j][s] if s >= 0 else None

    def task_kind(self) -> np.ndarray:
        """Newest known event kind per task (-1 unknown). Cached: a Belief is an immutable snapshot."""
        if self._kind is None:
            out = np.full(self.layout.T, -1, np.int64)
            evs = self.kb.facts.task_events
            for j in np.flatnonzero(self.task_seq >= 0):
                out[j] = evs[j][int(self.task_seq[j])].kind
            out.setflags(write=False)
            self._kind = out
        return self._kind

    @property
    def released(self) -> np.ndarray:
        return self.task_seq >= 0

    def open_tasks(self) -> np.ndarray:
        """Released and neither in progress nor finished (an aborted task is open again)."""
        k = self.task_kind()
        return (k == RELEASE) | (k == ABORT)

    def finished(self) -> np.ndarray:
        return self.task_kind() == FINISH

    def started(self) -> np.ndarray:
        return self.task_kind() == START

    # ---- robots
    def record(self, b: int) -> RobotRecord | None:
        c = int(self.rec_ctr[b])
        return self.kb.facts.records[b][c] if c >= 0 else None

    def heard_time(self) -> np.ndarray:
        """Author time of the newest known record of every node (-inf if never heard)."""
        rt = self.kb.facts.rec_time
        return np.where(self.rec_ctr >= 0, rt[np.arange(self.layout.N), np.maximum(self.rec_ctr, 0)], -np.inf)

    def route(self, b: int) -> RouteVersion | None:
        k = int(self.route_keys[b])
        return self.kb.facts.routes[(b, k)] if k >= 0 else None

    def remaining_route(self, b: int) -> tuple[int, ...]:
        """What b is believed to execute: head (if travelling/waiting) + remaining adopted route, or the newest
        version if b has not adopted it yet (it will on contact); tasks known started/finished are dropped."""
        rec, ver = self.record(b), self.route(b)
        kind = self.task_kind()
        live = (kind == RELEASE) | (kind == ABORT)
        head = ()
        if rec is not None and rec.mode in (TRAVEL, WAIT) and rec.target >= 0:
            head = (rec.target,)
        if ver is not None and (rec is None or ver.epoch * self.layout.N + ver.author > rec.route_key):
            seq = tuple(x for x in ver.tasks if x not in head)
        else:
            seq = rec.route if rec is not None else ()
        return tuple(x for x in head + tuple(seq) if live[x] or x in head)

    # ---- logs
    def log_events(self) -> list[LogEvent]:
        if self._logs is None:
            out = []
            for a in np.flatnonzero(self.log_len > 0):
                out.extend(self.kb.facts.logs[a][:int(self.log_len[a])])
            self._logs = out
        return self._logs

    def abandons(self) -> set[tuple[int, int]]:
        return {(e.robot, e.task) for e in self.log_events() if e.kind == ABANDON}

    def absents(self) -> set[tuple[int, int]]:
        """(member, task) pairs declared absent on site (rule 4.6(a))."""
        return {(e.robot, e.task) for e in self.log_events() if e.kind == ABSENT}

    def n_orphan_events(self) -> int:
        return int(self.log_len.sum())

    # ---- digest and seed (G3: identical replicas -> identical seeds -> identical plans)
    def digest(self) -> bytes:
        return hashlib.blake2b(self.vec.tobytes(), digest_size=16).digest()

    def seed(self, event_index: int) -> int:
        h = hashlib.blake2b(self.digest() + int(event_index).to_bytes(8, "little", signed=True), digest_size=8)
        return int.from_bytes(h.digest(), "little") & 0x7FFFFFFF


# -------------------------------------------------------------------- failure detection and suspicion (4.3, 4.6)
def suspected(belief: Belief, b: int, timeout: float) -> bool:
    """Good-comms failure detector: b is silent for more than ``timeout`` (h ticks). Needs per-tick heartbeats."""
    rec = belief.record(b)
    return rec is not None and rec.mode != STATION and belief.now - rec.time > timeout + EPS


def declared_failed(belief: Belief, b: int, h_f: float = 10.0) -> bool:
    """Rule 4.6(b): silence >= h_f AND at least one ``absent`` record about b. A silent robot without on-site
    evidence is only unreachable (clamped), never failed."""
    rec = belief.record(b)
    if rec is None or belief.now - rec.time < h_f - EPS:
        return False
    return any(e.kind == ABSENT and e.robot == b for e in belief.log_events())


def failed_set(belief: Belief, rule: str, timeout: float = 5 * TICK, h_f: float = 10.0) -> frozenset[int]:
    """Robots the belief treats as failed: ``rule`` = "detector" (good comms) or "suspicion" (C3, rule 4.6)."""
    A = belief.layout.A
    if rule == "detector":
        return frozenset(b for b in range(A) if suspected(belief, b, timeout))
    if rule == "suspicion":
        return frozenset(b for b in range(A) if declared_failed(belief, b, h_f))
    if rule == "none":
        return frozenset()
    raise ValueError(rule)


# -------------------------------------------------------------------- leases (4.3) and on-site evidence (4.6a)
@dataclass(frozen=True)
class LeaseConfig:
    """``patient`` (C3): abandon after waiting >= g with no partner known to be coming, or after >= L.
    ``detector`` (good comms): wait while partners heartbeat; abandon only when no partner that is not suspected
    failed is still coming (the lease is the failure detector, spec 4.3; this is what G5 needs)."""

    mode: str = "patient"
    grace: float = 1.0  # g, grid {0.3, 1, 3}
    max_wait: float = 30.0  # L, grid {10, 30, 60}
    detector_timeout: float = 5 * TICK  # h = 5 ticks
    silence: float = 2.0  # a planned member unheard for longer than this is "silent" (4.6a)

    @staticmethod
    def grid() -> list[LeaseConfig]:
        return [LeaseConfig("patient", g, L) for g in (0.3, 1.0, 3.0) for L in (10.0, 30.0, 60.0)]


class LeaseDecision(NamedTuple):
    abandon: bool
    reason: str  # "" | "no_partner" | "max_wait" | "suspected"
    absent: tuple[int, ...]  # silent planned members to publish as absent(member, task, now)


def planned_members(belief: Belief, task: int, exclude: int = -1) -> list[int]:
    """Robots the belief says will serve ``task``: heading there / waiting there by record, or with the task in
    their believed remaining route (adopted route or newest version)."""
    out = []
    for m in range(belief.layout.A):
        if m == exclude:
            continue
        rec = belief.record(m)
        heading = rec is not None and rec.mode in (TRAVEL, WAIT) and rec.target == task
        if heading or task in belief.remaining_route(m):
            out.append(m)
    return out


def lease_check(belief: Belief, robot: int, task: int, wait_since: float, present=(), cfg: LeaseConfig | None = None,
                failed=frozenset()) -> LeaseDecision:
    """Decide whether ``robot``, waiting on site at ``task`` since ``wait_since``, abandons now.

    ``present``: robots physically on site (observed locally). ``failed``: robots the belief treats as failed.
    """
    cfg = cfg or LeaseConfig()
    now = belief.now
    waited = now - wait_since
    present = {int(x) for x in present} | {robot}
    absent_known = {m for (m, j) in belief.absents() if j == task}
    planned = [m for m in planned_members(belief, task, exclude=robot)
               if m not in present and m not in absent_known and m not in failed]
    heard = belief.heard_time()
    if cfg.mode == "detector":
        coming = [m for m in planned if not suspected(belief, m, cfg.detector_timeout)]
        if waited > EPS and not coming:
            return LeaseDecision(True, "suspected" if planned else "no_partner", ())
        return LeaseDecision(False, "", ())
    if cfg.mode != "patient":
        raise ValueError(cfg.mode)
    reason = ""
    if waited >= cfg.max_wait - EPS:
        reason = "max_wait"
    elif waited >= cfg.grace - EPS and not planned:
        reason = "no_partner"
    if not reason:
        return LeaseDecision(False, "", ())
    silent = tuple(m for m in planned if now - heard[m] > cfg.silence + EPS)
    return LeaseDecision(True, reason, silent)


def publish_abandon(kb: KnowledgeBase, robot: int, task: int, now: float, decision: LeaseDecision) -> None:
    """Write the abandon record and the rule-4.6(a) ``absent`` records into the abandoning robot's replica."""
    kb.append_log(robot, ABANDON, robot, task, now)
    for m in decision.absent:
        kb.append_log(robot, ABSENT, m, task, now)
