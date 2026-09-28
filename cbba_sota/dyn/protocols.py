"""Track D protocol arms for C3 (spec 4.5, 5.3): leader election, planning scope, triggers, authoring, adoption.

Interface to the planner (Agent B). A planner is a callable ``planner(scope, belief) -> routes`` where
- ``scope: PlanScope`` says who plans (``node``), which robots it may rewrite (``writable``), which outsiders keep
  their believed routes (``frozen``: their traits count toward those coalitions), which outsiders are modelled by
  clamped snapshots (``clamped``: free at max(recorded ETA or finish, now), located at the recorded target, no dead
  reckoning), which keep only their committed head (``head_only``, INF-r), which are failed, and the deterministic
  seed ``hash(replica digest, event index)`` (G3: identical replicas compute identical plans);
- ``belief: replica.Belief`` is the planner's own replica (the only knowledge it may use);
- ``routes`` maps each rewritten robot to its FULL new route in key order, committed head first (a ``PlanOutput``
  may also carry per-task keys). Robots missing from the mapping keep their routes.
Planners must be pure functions of (scope, belief) (the T4 shadow arms re-call them on the same replica).
``rh_planner(inst, iters, kappa)`` wraps agent B's ``cbba_sota.dyn.planner.RHPlanner`` (the SPARC kernel) as such a
callback: ``plan_state`` maps (scope, belief) to B's ``PlanState`` (committed tasks, frozen members, one consistent
key order rebuilt from possibly conflicting route versions, heads, clamped ready times) and the plan's routes and
keys come back as a ``PlanOutput``. ``planning_view(scope, belief, inst)`` is a planner-agnostic array view.

Interface to the world (Agent A). Per tick k (time t_k), after physics and after the world has written local
observations into ``kb`` (``KnowledgeBase.task_event`` / ``observe_task`` / ``heartbeat`` / ``append_log``):
    runner.deliver(k, t, pos, alive)          # frames of tick k-1 arrive (comm.Transport)
    runner.briefing(k, t, pos, alive)         # k = 0 only: full-comm briefing, same start plan for every arm
    runner.plan(k, t, pos, alive)             # leaders that trigger call the planner and author versions
    for b, ver in runner.adoptions(alive).items(): world.adopt(b, ver)  # robot keeps its committed head,
                                              # drops the old suffix (CBBA bundle release), then heartbeats
    ... world: leases (replica.lease_check / publish_abandon), rally (outside_station_component), departures
    runner.send(k, t, pos, alive)             # frames of tick k leave
``pos`` is [N, 2] (robots then the station), ``alive`` is [N] bool (the station is always alive).

Arms (``ARMS``; ``arm(name)``; ``deployment_switch(rho, rho_star)``):
- FULL       ideal comms, the station plans every live robot (the good-comms SPARC = central RH, structural triggers)
- CEN-F      the station plans its component; outsiders follow their last route (frozen), then rally
- HYB        CEN-F + stranded components re-plan everyone (clamped) on every knowledge change and every 1.0 (V1)
- REP-clamp  every component leader re-plans all robots with clamped snapshots; newest version wins
- INF-r      leaders plan only their component; outsiders keep only their physically committed head task
- SPARC      anchored replication: station component = CEN-F; other components re-plan everyone (clamped),
             structural triggers (V2); ``sparc("V1")`` = HYB, ``sparc("V3", theta)`` = HALO's regime gate
- DS         CEN-F if rho >= rho*, else REP-clamp (``deployment_switch``)
- PER-ROBOT  every robot plans only itself on its own belief (hook for RL-dec / CBTA-dec)
- REP-TO / REP-KO  REP-clamp with the teammate-state / knowledge oracle (diagnosis only)
Default triggers are structural (4.2) for every arm except where the spec says V1 (HYB). They are ``ArmSpec``
fields and can be overridden; the defaults are this module's reading of Section 5.3 and must be frozen with it.
Leaders: the station if a node has heard it within ``member_window`` (1.5 = heartbeat period 1.0 + hops), else the
lowest robot id heard; a membership change triggers a re-plan at most every ``membership_debounce`` (0.5, the
pilots' rate limit). ``ProtocolRunner(shadow=...)`` logs, at each plan call, whether other arms would author
different versions from the same replica and seed (T4 decision-difference log).

Station outsiders (resolved 2026-09-28 on the real stack): spec 4.5 says robots outside the station component "keep
their believed routes, and their traits count toward those coalitions" (``frozen_model="anchor"``, the default). The
day-1 toy world suggested anchoring was badly hurt by G1 key monotonicity; the C3 design pilot in the real world
(``cbba_sota.dyn.c3world``, dev seed 2, docs/results/trackD-week1/c3.md) did not reproduce that (anchor vs head within
5-16%, not consistent in sign), so the spec-literal anchoring and monotone keys are used. ``frozen_model="head"``
(only the committed head is kept) remains available as a sensitivity.

C3 liveness conventions (rule 4.6(c), ``replica.overdue_absents`` / ``replica.effective_kind``): overdue work of a
silent crew counts as aborted, and an overdue arrival or finish promised by a silent robot counts as an absent record
and as failure evidence; absent records are superseded by newer knowledge of the member; the lease names declared-
failed planned members absent too. Each closes a livelock or deadlock found on dev episodes (c3.md lists them).
"""
from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace

import numpy as np

from cbba_sota.dyn.comm import Transport, components
from cbba_sota.dyn.replica import (
    ABORT,
    FINISH,
    HOMING,
    IDLE,
    RALLY,
    RELEASE,
    START,
    TICK,
    TRAVEL,
    WAIT,
    WORK,
    Belief,
    KnowledgeBase,
    RouteVersion,
    effective_kind,
    failed_set,
    overdue_absents,
)

EPS = 1e-9


# ------------------------------------------------------------------------------------------------ arms
@dataclass(frozen=True)
class ArmSpec:
    name: str
    ideal: bool = False  # FULL: good comms
    station_scope: str = "component"  # station writes: "component" | "all"
    station_outsiders: str = "frozen"  # robots outside the station component: "frozen" | "clamp" | "stale" | "head_only"
    stranded: str = "none"  # leaders of components without the station: "none" | "all" | "component"
    stranded_outsiders: str = "clamp"
    station_trigger: str = "structural"  # "structural" (V2) | "v1"
    stranded_trigger: str = "structural"  # "structural" | "v1" | "v3"
    theta: float | None = None  # V3 regime gate
    per_robot: bool = False
    oracle: str | None = None  # transport oracle hook: "agent" | "know" | "release"
    membership: str = "heartbeat"  # "heartbeat" (causal) | "oracle" (exact unit-disk components, T1-like)
    rally: bool = True  # shared rally rule (knockout in T5)
    relay: bool = True  # gossip relaying (knockout in T5)


ARMS: dict[str, ArmSpec] = {
    "FULL": ArmSpec("FULL", ideal=True, station_scope="all", station_outsiders="clamp"),
    "CEN-F": ArmSpec("CEN-F"),
    "HYB": ArmSpec("HYB", stranded="all", stranded_outsiders="clamp", stranded_trigger="v1"),
    "REP-clamp": ArmSpec("REP-clamp", station_scope="all", station_outsiders="clamp", stranded="all",
                         stranded_outsiders="clamp"),
    "INF-r": ArmSpec("INF-r", station_outsiders="head_only", stranded="component", stranded_outsiders="head_only"),
    "SPARC": ArmSpec("SPARC", stranded="all", stranded_outsiders="clamp", stranded_trigger="structural"),
    "PER-ROBOT": ArmSpec("PER-ROBOT", per_robot=True),
    "REP-TO": ArmSpec("REP-TO", station_scope="all", station_outsiders="clamp", stranded="all",
                      stranded_outsiders="clamp", oracle="agent"),
    "REP-KO": ArmSpec("REP-KO", station_scope="all", station_outsiders="clamp", stranded="all",
                      stranded_outsiders="clamp", oracle="know"),
}


def arm(name: str, **overrides) -> ArmSpec:
    return replace(ARMS[name], **overrides) if overrides else ARMS[name]


def sparc(variant: str = "V2", theta: float = 0.7) -> ArmSpec:
    """SPARC stranded-component variants (4.5): V1 = HYB triggers, V2 = structural (default), V3 = theta gate."""
    trig = {"V1": "v1", "V2": "structural", "V3": "v3"}[variant]
    return replace(ARMS["SPARC"], name=f"SPARC-{variant}", stranded_trigger=trig,
                   theta=theta if variant == "V3" else None)


def deployment_switch(rho: float, rho_star: float) -> ArmSpec:
    """DS: CEN-F if rho >= rho*, else REP-clamp (range and site size are known at deployment)."""
    base = ARMS["CEN-F"] if rho >= rho_star else ARMS["REP-clamp"]
    return replace(base, name=f"DS@{rho_star:g}")


def factorial_arms() -> list[ArmSpec]:
    """T5: {clamp vs stale snapshot} x {station anchoring} x {stranded re-plan} x {V1 vs V2} = 16 arms."""
    out = []
    for clamp in (True, False):
        model = "clamp" if clamp else "stale"
        for anchor in (True, False):
            for strand in (True, False):
                for trig in ("v1", "structural"):
                    out.append(ArmSpec(
                        f"F[{'C' if clamp else 's'}{'A' if anchor else 'a'}{'S' if strand else 's'}"
                        f"{'1' if trig == 'v1' else '2'}]",
                        station_scope="component" if anchor else "all",
                        station_outsiders="frozen" if anchor else model,
                        stranded="all" if strand else "none", stranded_outsiders=model, stranded_trigger=trig))
    return out


# ------------------------------------------------------------------------------------------------ planner interface
@dataclass(frozen=True)
class PlanScope:
    node: int  # planning node (A = station)
    now: float
    tick: int  # event index used in the seed
    seed: int
    is_station: bool
    component: frozenset[int]  # robots the planner reaches (their records are fresh)
    writable: frozenset[int]  # robots whose routes this call may rewrite
    frozen: frozenset[int]  # outsiders that keep their believed routes (traits count toward those coalitions)
    clamped: frozenset[int]  # outsiders in ``writable`` modelled by clamped snapshots
    stale: frozenset[int]  # outsiders in ``writable`` modelled by raw stale snapshots (T5 ablation only)
    head_only: frozenset[int]  # outsiders that keep only their committed head, unavailable for new work (INF-r)
    failed: frozenset[int]
    failure_rule: str  # "detector" (good comms: failed robots vanish) | "suspicion" (4.6b: future route only)
    reason: str  # trigger that fired


@dataclass
class PlanOutput:
    routes: dict[int, tuple[int, ...]]
    keys: dict[int, tuple[float, ...]] | None = None  # per robot, the global key of each task of its route


Planner = Callable[[PlanScope, Belief], "Mapping[int, Sequence[int]] | PlanOutput"]


@dataclass
class PlanningView:
    """Arrays a state-start planner needs, derived from (scope, belief) by the rules of Section 4.5 / 4.6."""

    now: float
    ready: np.ndarray  # [A] time the robot is free to start its fixed prefix / new work (inf: never heard or failed)
    pos: np.ndarray  # [A, 2] where it is free
    fixed: list[tuple[int, ...]]  # [A] tasks it must still do first, in order (committed head; frozen route)
    avail: np.ndarray  # [A] may receive new tasks (= writable and not failed)
    writable: np.ndarray  # [A]
    failed: np.ndarray  # [A]
    known: np.ndarray  # [T] released and known
    open: np.ndarray  # [T] known, not in progress, not finished
    committed: np.ndarray  # [T] open and in some robot's fixed prefix (members departed)
    resid: np.ndarray  # [T, K] requirement not covered by fixed members (clipped at 0)


def planning_view(scope: PlanScope, belief: Belief, inst) -> PlanningView:
    """Teammate model of 4.5 plus the suspicion rule of 4.6. ``inst`` needs ``req, loc, ab, depot``."""
    L = belief.layout
    A, T = L.A, L.T
    now = belief.now
    kind = belief.task_kind()
    known = kind >= 0
    open_ = (kind == RELEASE) | (kind == ABORT)
    absents = belief.absents()
    ready = np.full(A, np.inf)
    pos = np.asarray(inst.depot, float).copy()
    fixed: list[tuple[int, ...]] = [() for _ in range(A)]
    failed = np.zeros(A, bool)
    for b in range(A):
        rec = belief.record(b)
        if rec is None:
            continue
        head = rec.target if (rec.mode in (TRAVEL, WAIT) and rec.target >= 0 and open_[rec.target]
                              and (b, rec.target) not in absents) else -1
        if b in scope.failed:
            failed[b] = True
            if scope.failure_rule == "suspicion" and head >= 0:
                fixed[b] = (head,)  # 4.6(b): declaring failure affects the future route, not the committed head
            continue
        if rec.mode in (TRAVEL, WAIT, WORK) and rec.target >= 0:
            pos[b] = inst.loc[rec.target]
            t = rec.t_ref if rec.mode != WAIT else now
        else:
            pos[b] = rec.pos
            t = rec.time
        ready[b] = t if b in scope.stale else max(t, now)
        if b in scope.frozen:
            fixed[b] = ((head,) if head >= 0 else ()) + tuple(
                x for x in belief.remaining_route(b) if x != head and open_[x] and (b, x) not in absents)
        elif head >= 0:
            fixed[b] = (head,)
    writable = np.zeros(A, bool)
    writable[list(scope.writable)] = True
    avail = writable & ~failed
    committed = np.zeros(T, bool)
    req = np.asarray(inst.req, float)
    resid = np.where(known[:, None], req, 0.0)
    for b in range(A):
        for j in fixed[b]:
            committed[j] = True
            resid[j] = resid[j] - np.asarray(inst.ab[b], float)
    return PlanningView(now, ready, pos, fixed, avail, writable, failed, known, open_, committed & open_,
                        np.maximum(resid, 0.0))


# ------------------------------------------------------------------------------------------------ agent B's planner
def _route_keys(belief: Belief, b: int) -> dict[int, float]:
    """Key of each task in b's newest known route version (empty if the version carries no keys)."""
    ver = belief.route(b)
    if ver is None or ver.keys is None:
        return {}
    return {int(j): float(k) for j, k in zip(ver.tasks, ver.keys, strict=True)}


def plan_state(scope: PlanScope, belief: Belief, inst, kappa: float = 1.0, frozen_model: str = "anchor",
               h_f: float = 10.0):
    """(scope, belief) -> (``planner.PlanState``, scope mask, incumbent ``DynPlan``) for agent B's ``RHPlanner``.

    Mapping (spec 4.1, 4.5, 4.6):
    - committed task: open and some robot's record says it travels to / waits at it (not marked absent there); its
      frozen members are every robot whose believed route contains it (the coalition is frozen at the first
      departure); keys come from the route versions.
    - frozen outsiders (CEN-F, SPARC station component): every open task of their believed route is anchored with
      the frozen robots as fixed members (their traits count; a missing trait makes it a residual that robots in
      scope fill at its rank);
    - robots: ready / position from their records (clamped to now unless ``stale``), head = committed task they
      travel to or wait at; failed robots are dead ("detector") or keep only their head ("suspicion", 4.6b).
    - incumbent: the believed coalitions of open tasks (warm start).
    ``frozen_model``: "anchor" (spec 4.5 literally: a frozen outsider's whole believed route is anchored; B's kernel
    then puts those residual anchors ahead of every repaired task, G1) or "head" (only its committed head is kept;
    its later tasks are open and it takes no new work). See the module notes on the choice.
    Under the suspicion rule (C3) rule 4.6(c) applies: overdue work of a silent crew counts as aborted
    (``replica.effective_kind``, nominal durations ``inst.dur``) and an overdue arrival of a silent robot counts as an
    absent record (``replica.overdue_absents``)."""
    from cbba_sota.dyn.planner import AT_DEPOT, AT_POINT, DynPlan, PlanState

    dur = np.asarray(inst.dur, float)

    L = belief.layout
    A, T = L.A, L.T
    now = belief.now
    suspicion = scope.failure_rule == "suspicion"
    kind = effective_kind(belief, dur, h_f) if suspicion else belief.task_kind()
    open_ = (kind == RELEASE) | (kind == ABORT)
    absents = belief.absents() | (overdue_absents(belief, h_f) if suspicion else frozenset())
    recs = [belief.record(b) for b in range(A)]
    routes = [belief.remaining_route(b) for b in range(A)]
    keys_of = [_route_keys(belief, b) for b in range(A)]
    alive = np.array([recs[b] is not None and (b not in scope.failed or suspicion) for b in range(A)])
    ready = np.zeros(A)
    pos = np.asarray(inst.depot, float).copy()
    pos_task = np.full(A, AT_POINT, np.int64)
    head = np.full(A, -1, np.int64)
    departed = [set() for _ in range(T)]
    for b in range(A):
        r = recs[b]
        if r is None:
            continue
        if r.mode in (TRAVEL, WAIT, WORK) and r.target >= 0:
            j = r.target
            pos[b], pos_task[b] = inst.loc[j], j
            t = r.t_ref
            if r.mode in (TRAVEL, WAIT) and open_[j] and (b, j) not in absents:
                head[b] = j
                departed[j].add(b)
        else:
            pos[b] = r.pos
            if np.allclose(r.pos, inst.depot[b]):
                pos_task[b] = AT_DEPOT
            # legs are atomic: a robot heading home, or travelling released from its task, is free at its ETA
            t = r.t_ref if r.mode in (HOMING, TRAVEL) else now
        ready[b] = t if (b in scope.stale or r.mode == WAIT) else max(t, now)
    committed = np.array([open_[j] and bool(departed[j]) for j in range(T)])
    mem_sets: list[set[int]] = [set() for _ in range(T)]
    old_max = max((max(k.values()) for k in keys_of if k), default=-1.0)

    def key_of(j, mem):
        ks = [keys_of[m][j] for m in mem if j in keys_of[m]]
        return max(ks) if ks else old_max + 1.0 + j * 1e-6

    for j in np.flatnonzero(committed):
        mem = {m for m in range(A) if alive[m] and (j in routes[m] or m in departed[j]) and (m, j) not in absents}
        if suspicion:  # a declared-failed robot keeps only its committed head
            mem = {m for m in mem if m not in scope.failed or head[m] == j}
        mem_sets[j] = mem
    for b in (scope.frozen if frozen_model == "anchor" else ()):
        if alive[b]:
            for j in routes[b]:
                if open_[j] and (b, j) not in absents:
                    committed[j] = True
                    mem_sets[j].add(b)
    # One consistent key order for the committed tasks. Replicas can hold route versions of different leaders whose
    # key spaces disagree, so each member's committed chain (its believed route, head first) is merged into a DAG,
    # freshest route versions first; a chain edge that would close a cycle drops that robot from the later task.
    stamp = [int(belief.route_keys[m]) for m in range(A)]
    succ: dict[int, set[int]] = {int(j): set() for j in np.flatnonzero(committed)}

    def reaches(a, z):
        stack, seen = [a], {a}
        while stack:
            u = stack.pop()
            if u == z:
                return True
            for w in succ[u] - seen:
                seen.add(w)
                stack.append(w)
        return False

    for m in sorted(range(A), key=lambda m: (-stamp[m], m)):
        chain = [j for j in routes[m] if committed[j] and m in mem_sets[j]]
        if head[m] >= 0 and chain and chain[0] != head[m]:
            chain = [head[m]] + [j for j in chain if j != head[m]]
        prev = -1
        for j in chain:
            if prev >= 0:
                if reaches(j, prev):
                    mem_sets[j].discard(m)
                    continue
                succ[prev].add(j)
            prev = j
    prio = {j: key_of(j, mem_sets[j]) for j in succ}
    indeg = {j: 0 for j in succ}
    for ws in succ.values():
        for w in ws:
            indeg[w] += 1
    ready_q = sorted((prio[j], j) for j in succ if indeg[j] == 0)
    rank: dict[int, int] = {}
    while ready_q:
        _, u = ready_q.pop(0)
        rank[u] = len(rank)
        for w in succ[u]:
            indeg[w] -= 1
            if indeg[w] == 0:
                ready_q.append((prio[w], w))
                ready_q.sort()
    members: list[tuple[int, ...]] = [()] * T
    keys = np.zeros(T)
    for j, r in rank.items():
        members[j] = tuple(sorted(mem_sets[j]))
        keys[j] = float(r)
    committed &= np.array([bool(members[j]) for j in range(T)])
    key_floor = float(len(rank) - 1)
    scope_mask = np.zeros(A, bool)
    scope_mask[[b for b in scope.writable if alive[b] and b not in scope.failed]] = True
    inc_mem: list[tuple[int, ...]] = [()] * T
    inc_key = np.full(T, np.nan)
    for j in np.flatnonzero(open_ & ~committed):
        mem = tuple(sorted(m for m in range(A) if scope_mask[m] and j in routes[m]))
        if mem:
            inc_mem[j] = mem
            inc_key[j] = key_of(j, mem)
    state = PlanState(inst=inst, now=now, released=kind >= 0, done=kind == FINISH, started=kind == START,
                      committed=committed, members=members, keys=keys, alive=alive, ready=ready, pos=pos,
                      pos_task=pos_task, head=head, kappa=kappa, key_floor=key_floor)
    return state, scope_mask, DynPlan(members=inc_mem, keys=inc_key, makespan=float("nan"))


def rh_planner(inst, iters: int = 300, kappa: float = 1.0, config=None, frozen_model: str = "anchor",
               h_f: float = 10.0) -> Planner:
    """Planner callback running agent B's ``RHPlanner`` (the SPARC kernel) on a replica, with the scope's seed.
    Returns full key-ordered routes (with keys) for the writable robots."""
    from cbba_sota.dyn.planner import RHPlanner

    rh = RHPlanner(config)

    def plan(scope: PlanScope, belief: Belief) -> PlanOutput:
        state, mask, incumbent = plan_state(scope, belief, inst, kappa, frozen_model, h_f)
        dp = rh.plan(state, incumbent=incumbent, scope=mask, seed=scope.seed, iters=iters)
        A = belief.layout.A
        by_robot = dp.routes_for(A)
        routes, keys = {}, {}
        for b in scope.writable:
            r = tuple(int(j) for j in by_robot[b])
            routes[b] = r
            keys[b] = tuple(float(dp.keys[j]) for j in r)
        plan.last = dp
        return PlanOutput(routes, keys)

    plan.last = None
    return plan


# ------------------------------------------------------------------------------------------------ membership
def reach_matrix(kb: KnowledgeBase, now: float, window: float) -> np.ndarray:
    """[N, N] reach[n, b]: n heard b (a record authored no earlier than now - window); n always reaches itself."""
    reach = kb.heard_matrix() >= now - window - EPS
    np.fill_diagonal(reach, True)
    return reach


def leaders(reach: np.ndarray, station: int) -> np.ndarray:
    """Leader seen by each node: the station if reachable, else the lowest reachable robot id."""
    robots = reach[:, :station]
    low = np.where(robots.any(1), robots.argmax(1), np.arange(len(reach)))
    return np.where(reach[:, station], station, low)


def outside_station_component(belief: Belief, window: float) -> bool:
    """Rally test (4.3): the robot has not heard the station recently."""
    st = belief.layout.station
    return bool(belief.heard_time()[st] < belief.now - window - EPS) and belief.node != st


def hub_fraction(belief: Belief, window: float) -> float | None:
    """Station-component fraction observed by a robot that currently reaches the station (V3 gate input)."""
    heard = belief.heard_time() >= belief.now - window - EPS
    st = belief.layout.station
    if not heard[st]:
        return None
    return float(heard[:st].mean())


# ------------------------------------------------------------------------------------------------ runner
@dataclass
class ProtocolConfig:
    member_window: float = 1.5  # heartbeat period 1.0 + a few hops
    failure_rule: str = "suspicion"  # "detector" with good comms
    detector_timeout: float = 5 * TICK
    h_f: float = 10.0
    v1_period: float = 1.0
    membership_debounce: float = 0.5  # a membership-only trigger waits >= this since the last plan (pilot practice)

    @classmethod
    def good_comms(cls) -> ProtocolConfig:
        return cls(member_window=math.inf, failure_rule="detector")


@dataclass
class _TrigState:
    released: np.ndarray
    n_log: int
    n_abort: int
    failed: frozenset
    members: frozenset
    idle_empty: frozenset
    sig: bytes
    last_plan: float


@dataclass
class PlanCall:
    tick: int
    node: int
    reason: str
    members: frozenset
    authored: dict  # robot -> tasks
    epoch: int
    shadow: dict = field(default_factory=dict)  # T4: shadow arm name -> its authored versions differ from ours


@dataclass
class ProtocolRunner:
    kb: KnowledgeBase
    transport: Transport
    arm: ArmSpec
    planner: Planner
    cfg: ProtocolConfig = field(default_factory=ProtocolConfig)
    comm_R: float = math.inf  # only for membership="oracle"
    shadow: tuple = ()  # T4: arms evaluated counterfactually (same replica, same seed) at each of our plan calls;
    # the planner must be a pure function of (scope, belief) for this to be meaningful
    dur: np.ndarray | None = None  # nominal task durations: enables rule 4.6(c) in the trigger state (C3)

    def __post_init__(self):
        L = self.kb.layout
        self.L = L
        self.adopted = np.full(L.A, -1, np.int64)
        self._trig: dict[int, _TrigState] = {}
        self._idle: dict[int, frozenset] = {}
        self.calls: list[PlanCall] = []
        self.n_plans = 0
        self.n_plans_off_station = 0
        self.station_frac: list[float] = []

    # ---- transport passthrough
    def deliver(self, k: int, t: float, pos: np.ndarray, alive: np.ndarray) -> None:
        self.transport.deliver(k, pos, alive)

    def send(self, k: int, t: float, pos: np.ndarray, alive: np.ndarray) -> None:
        self.transport.send(k, pos, alive)

    # ---- membership view of every node
    def _reach(self, t: float, pos: np.ndarray, alive: np.ndarray) -> np.ndarray:
        """Who each node can reach. Causal (heartbeats) by default: a failed robot drops out only when it has been
        silent for ``member_window``; with ideal comms everyone reaches everyone and failures are left to the
        detector (``failed_set``), so no failure information leaks through membership."""
        N = self.L.N
        if self.transport.ideal:
            return np.ones((N, N), bool)
        if self.arm.membership == "oracle":  # diagnosis only: exact unit-disk components (dead radios excluded)
            lab = components(pos, self.comm_R, alive)
            return lab[:, None] == lab[None, :]
        return reach_matrix(self.kb, t, self.cfg.member_window)

    def _failed(self, belief: Belief) -> frozenset[int]:
        return failed_set(belief, self.cfg.failure_rule, self.cfg.detector_timeout, self.cfg.h_f)

    # ---- trigger evaluation (4.2 structural; V1; V3)
    def _state(self, belief: Belief, members: frozenset, failed: frozenset, t: float) -> _TrigState:
        kind = effective_kind(belief, self.dur, self.cfg.h_f) if self.cfg.failure_rule == "suspicion" \
            else belief.task_kind()
        open_any = bool(((kind == RELEASE) | (kind == ABORT)).any())
        idle = frozenset()
        if open_any:
            idle = frozenset(b for b in members if (r := belief.record(b)) is not None and r.mode in (IDLE, RALLY)
                             and not belief.remaining_route(b))
        L = self.L
        sig = np.concatenate([belief.vec[L.TASK], belief.vec[L.LOG], belief.vec[L.ROUTE]]).tobytes()
        return _TrigState(belief.released.copy(), belief.n_orphan_events(), int((kind == ABORT).sum()), failed,
                          members, idle, sig, -math.inf)

    def _structural(self, prev: _TrigState | None, cur: _TrigState, t: float, idle_prev: frozenset) -> str:
        """Structural events since the node's last plan (4.2). Noise (finishes, stalls, arrivals) never triggers.
        A robot becoming idle with an empty route is an edge relative to the previous tick, not to the last plan."""
        if prev is None:
            return "init"
        if (cur.released & ~prev.released).any():
            return "release"
        if cur.n_log > prev.n_log or cur.n_abort > prev.n_abort or (cur.failed - prev.failed):
            return "orphan"
        if cur.members != prev.members and t - prev.last_plan >= self.cfg.membership_debounce - EPS:
            return "membership"
        if cur.idle_empty - idle_prev:
            return "idle"
        return ""

    def _trigger(self, mode: str, prev: _TrigState | None, cur: _TrigState, t: float, idle_prev: frozenset) -> str:
        why = self._structural(prev, cur, t, idle_prev)
        if mode == "structural" or why:
            return why
        if prev is None:
            return "init"
        if mode in ("v1", "v3"):
            if cur.sig != prev.sig:
                return "knowledge"
            if t - prev.last_plan >= self.cfg.v1_period - EPS:
                return "period"
            return ""
        raise ValueError(mode)

    def _v3_gate(self, belief: Belief, members: frozenset, cur: _TrigState) -> bool:
        """HALO gate: plan only if the station-component fraction the members last observed is below theta, or a
        member is idle with an empty route."""
        if cur.idle_empty:
            return True
        fr = [r.hub_frac for b in members if (r := belief.record(b)) is not None]
        return (min(fr) if fr else 1.0) < (self.arm.theta if self.arm.theta is not None else 1.0)

    # ---- scope
    def scope_for(self, node: int, belief: Belief, members: frozenset, failed: frozenset, tick: int,
                  reason: str, arm_spec: ArmSpec | None = None) -> PlanScope | None:
        L, a = self.L, arm_spec or self.arm
        is_station = node == L.station
        live = frozenset(range(L.A)) - failed
        comp = frozenset(members) & live
        outs = live - comp
        if a.per_robot:
            kind, model = "self", "frozen"
        elif is_station:
            kind, model = a.station_scope, a.station_outsiders
        else:
            kind, model = a.stranded, a.stranded_outsiders
        if kind == "none":
            return None
        frozen = clamped = stale = head_only = frozenset()
        if kind == "self":
            writable = frozenset({node}) & live
            frozen = live - writable
        elif kind == "component":
            writable = comp
            if model == "head_only":
                head_only = outs
            else:
                frozen = outs
        elif kind == "all":
            writable = live
            if model == "clamp":
                clamped = outs
            elif model == "stale":
                stale = outs
            else:
                raise ValueError(f"scope 'all' needs a clamp/stale outsider model, got {model}")
        else:
            raise ValueError(kind)
        return PlanScope(node, belief.now, tick, belief.seed(tick), is_station, comp, writable, frozen, clamped,
                         stale, head_only, failed, self.cfg.failure_rule, reason)

    # ---- t = 0 briefing
    def briefing(self, k: int, t: float, pos: np.ndarray, alive: np.ndarray) -> PlanCall:
        """Full-comm briefing (as in the pilots): every method starts from the same knowledge and the same station
        plan of the tasks released so far. Replicas are joined, the station plans every live robot, the plan is
        shared, digests are exchanged, and every node's trigger state is initialised (no spurious "init" plans).
        Call once after the world has written the t = 0 releases and heartbeats, before ``plan``."""
        L = self.L
        alive = np.asarray(alive, bool)
        idx = np.flatnonzero(alive)
        self.kb.sync(idx)
        belief = self.kb.belief(L.station, t)
        failed = self._failed(belief)
        live = frozenset(range(L.A)) - failed
        scope = PlanScope(L.station, t, k, belief.seed(k), True, live, live, frozenset(), frozenset(), frozenset(),
                          frozenset(), failed, self.cfg.failure_rule, "briefing")
        call = self._author(L.station, t, k, scope, belief, self.planner(scope, belief))
        self.n_plans += 1
        self.kb.sync(idx)
        self.transport.warm_start(k)
        for n in idx:
            b = self.kb.belief(int(n), t)
            st = self._state(b, live, failed, t)
            st.last_plan = t
            self._trig[int(n)] = st
            self._idle[int(n)] = st.idle_empty
        return call

    # ---- planning phase
    def plan(self, k: int, t: float, pos: np.ndarray, alive: np.ndarray) -> list[PlanCall]:
        L, a = self.L, self.arm
        alive = np.asarray(alive, bool)
        self.transport.sync(alive)
        reach = self._reach(t, pos, alive)
        lead = leaders(reach, L.station)
        self.station_frac.append(self._station_fraction(pos, alive))
        nodes = range(L.A) if a.per_robot else [n for n in range(L.N) if alive[n] and lead[n] == n]
        out = []
        for n in nodes:
            if not alive[n]:
                continue
            is_station = n == L.station
            if not a.per_robot and not is_station and a.stranded == "none":
                continue
            belief = self.kb.belief(n, t)
            failed = self._failed(belief)
            members = frozenset(int(b) for b in np.flatnonzero(reach[n, :L.A])) - failed
            cur = self._state(belief, members, failed, t)
            prev = self._trig.get(n)
            idle_prev = self._idle.get(n, frozenset())
            self._idle[n] = cur.idle_empty
            mode = a.station_trigger if (is_station or a.per_robot) else a.stranded_trigger
            reason = self._trigger(mode, prev, cur, t, idle_prev)
            if reason and mode == "v3" and not is_station and not self._v3_gate(belief, members, cur):
                reason = ""
            if not reason:  # keep the reference state of the last plan: a deferred event still fires later
                continue
            scope = self.scope_for(n, belief, members, failed, k, reason)
            if scope is None:
                continue
            res = self.planner(scope, belief)
            shadow = self._shadow(n, belief, members, failed, k, reason, res) if self.shadow else {}
            call = self._author(n, t, k, scope, belief, res)
            call.shadow = shadow
            out.append(call)
            self.n_plans += 1
            self.n_plans_off_station += int(not is_station)
            post = self.kb.belief(n, t)
            cur = self._state(post, members, failed, t)
            cur.last_plan = t
            self._trig[n] = cur
        self.transport.sync(alive)
        return out

    @staticmethod
    def _changes(n: int, scope: PlanScope, belief: Belief, res) -> tuple[dict, dict | None]:
        routes, keys = (res.routes, res.keys) if isinstance(res, PlanOutput) else (dict(res), None)
        changed = {}
        for b, r in routes.items():
            b = int(b)
            if b not in scope.writable:
                raise ValueError(f"planner at node {n} wrote robot {b} outside its scope")
            r = tuple(int(x) for x in r)
            if r != belief.remaining_route(b):
                changed[b] = r
        return changed, keys

    def _shadow(self, n, belief, members, failed, k, reason, res) -> dict:
        """T4 decision difference: would each shadow arm author different versions from the same replica?"""
        L = self.L
        scope = self.scope_for(n, belief, members, failed, k, reason)
        ours, _ = self._changes(n, scope, belief, res)
        out = {}
        for s in self.shadow:
            sc = self.scope_for(n, belief, members, failed, k, reason, arm_spec=s)
            if sc is None or (n != L.station and s.stranded == "none"):
                theirs = {}
            else:
                theirs, _ = self._changes(n, sc, belief, self.planner(sc, belief))
            out[s.name] = ours != theirs
        return out

    def _author(self, n: int, t: float, k: int, scope: PlanScope, belief: Belief, res) -> PlanCall:
        changed, keys = self._changes(n, scope, belief, res)
        epoch = -1
        if changed:
            epoch = self.kb.author_routes(n, t, changed, None if keys is None else {b: keys[b] for b in changed
                                                                                    if b in keys})
        call = PlanCall(k, n, scope.reason, scope.component, changed, epoch)
        self.calls.append(call)
        return call

    def _station_fraction(self, pos: np.ndarray, alive: np.ndarray) -> float:
        """Metric (3.7): share of live robots in the station's exact unit-disk component at this tick."""
        L = self.L
        n_live = max(1, int(alive[:L.A].sum()))
        if self.transport.ideal or not math.isfinite(self.comm_R):
            return float(alive[:L.A].sum()) / n_live
        lab = components(pos, self.comm_R, alive)
        return float(((lab[:L.A] == lab[L.station]) & alive[:L.A]).sum()) / n_live

    # ---- adoption phase
    def adoptions(self, alive: np.ndarray) -> dict[int, RouteVersion]:
        """Robots whose own replica holds a newer route version for themselves than the one adopted."""
        L = self.L
        out = {}
        for b in range(L.A):
            if not alive[b]:
                continue
            key = int(self.kb.v[b, L.ROUTE.start + b])
            if key > self.adopted[b]:
                self.adopted[b] = key
                out[b] = self.kb.facts.routes[(b, key)]
        return out

    # ---- metrics
    def summary(self) -> dict:
        return {"plans": self.n_plans, "plans_off_station": self.n_plans_off_station,
                "versions": int(self.kb.n_versions.sum()),
                "versions_off_station": int(self.kb.n_versions[:self.L.A].sum()),
                "station_frac": float(np.mean(self.station_frac)) if self.station_frac else float("nan"),
                **self.transport.stats.as_dict()}


def make_stack(n_robots: int, n_tasks: int, n_traits: int, arm_spec: ArmSpec, planner: Planner, *,
               rho: float = math.inf, loss: str = "ge", p: float = 0.2, key=None, budget: float = 1500.0,
               beacon_period: float | None = None, cfg: ProtocolConfig | None = None, shadow: tuple = (),
               dur=None):
    """Build (kb, transport, runner) for one episode: ideal comms for ``arm_spec.ideal``, else tier T2 at range
    rho * r_c(n) with the named loss model. Heartbeat period: one tick with ideal comms (failure detector), 1.0
    otherwise."""
    from cbba_sota.dyn.comm import DEFAULT_KEY, comm_range, make_transport
    from cbba_sota.dyn.replica import HeartbeatConfig, Layout

    L = Layout(n_robots, n_tasks, n_traits)
    ideal = arm_spec.ideal
    kb = KnowledgeBase(L, HeartbeatConfig(TICK if ideal else 1.0))
    R = comm_range(rho, n_robots)
    tr = make_transport(kb, ideal, R=R, loss=loss, p=p, key=key or DEFAULT_KEY, budget=budget,
                        oracle=arm_spec.oracle, beacon_period=beacon_period, relay=arm_spec.relay)
    cfg = cfg or (ProtocolConfig.good_comms() if ideal else ProtocolConfig())
    return kb, tr, ProtocolRunner(kb, tr, arm_spec, planner, cfg, comm_R=R, shadow=tuple(shadow),
                                  dur=None if dur is None else np.asarray(dur, float))
