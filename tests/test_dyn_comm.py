"""Tests for the Track D comm tier T2, the replicated knowledge layer and the protocol arms (spec 3.5, 4.4-4.6, 5.3).

CRDT merge laws, CRN determinism of the channel, ideal-comm reduction to global knowledge, multi-hop latency, byte
budget, leader election, leases and failure suspicion, arm scopes, an end-to-end toy world, and the cbja cross-check.
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from cbba_sota.dyn.comm import (
    Bernoulli,
    CRNKey,
    GilbertElliott,
    GossipTransport,
    IdealTransport,
    NoLoss,
    Rayleigh,
    comm_range,
    components,
    connectivity_radius,
)
from cbba_sota.dyn.protocols import (
    ARMS,
    ProtocolConfig,
    ProtocolRunner,
    arm,
    deployment_switch,
    factorial_arms,
    leaders,
    make_stack,
    planning_view,
    reach_matrix,
    sparc,
)
from cbba_sota.dyn.replica import (
    ABANDON,
    ABSENT,
    FINISH,
    IDLE,
    RALLY,
    RELEASE,
    START,
    STATION,
    TICK,
    TRAVEL,
    WAIT,
    WORK,
    HeartbeatConfig,
    KnowledgeBase,
    Layout,
    LeaseConfig,
    declared_failed,
    join,
    lease_check,
    publish_abandon,
    suspected,
)

ROOT = Path(__file__).resolve().parents[1]
CBJA_PY = Path("/home/jovyan/dev/cbja/.venv/bin/python")


# ------------------------------------------------------------------------------------------------ CRDT laws (G3)
def _random_vectors(rng, L, n):
    v = np.tile(L.bottom(), (n, 1))
    v += rng.integers(0, 6, v.shape)
    return v


def test_join_is_commutative_associative_idempotent():
    rng = np.random.default_rng(0)
    L = Layout(7, 11, 3)
    for _ in range(200):
        a, b, c = _random_vectors(rng, L, 3)
        assert np.array_equal(join(a, b), join(b, a))
        assert np.array_equal(join(join(a, b), c), join(a, join(b, c)))
        assert np.array_equal(join(a, a), a)
        assert np.array_equal(join(a, L.bottom()), a)  # bottom is the identity


def test_same_deltas_any_order_give_identical_replicas_and_seeds():
    rng = np.random.default_rng(1)
    L = Layout(5, 8, 2)
    kb = KnowledgeBase(L)
    deltas = _random_vectors(rng, L, 12)
    out = []
    for perm in (np.arange(12), rng.permutation(12), rng.permutation(12)[::-1]):
        r = L.bottom()
        for i in perm:
            r = join(r, deltas[i])
            r = join(r, deltas[i])  # duplicates are harmless
        out.append(r)
    assert all(np.array_equal(out[0], o) for o in out)
    b1, b2 = kb.belief_of_vector(out[0], 0, 1.0), kb.belief_of_vector(out[1], 3, 1.0)
    assert b1.seed(7) == b2.seed(7) and b1.seed(7) != b1.seed(8)


def test_route_lww_follows_lamport_epoch_then_author():
    L = Layout(3, 4, 1)
    kb = KnowledgeBase(L)
    e2 = kb.author_routes(2, 0.0, {0: (1, 2)})
    e1 = kb.author_routes(1, 0.0, {0: (3,)})
    assert e1 == e2 == 1  # concurrent authors, same epoch: the higher author id wins the tie
    kb.sync()
    assert kb.belief(0, 0.0).route(0).tasks == (1, 2)
    e = kb.author_routes(1, 0.1, {0: (0,)})  # an author that has seen epoch 0 writes above it
    assert e == 2
    kb.sync()
    assert kb.belief(2, 0.1).route(0).tasks == (0,)


def test_task_events_are_monotone_facts():
    L = Layout(2, 2, 1)
    kb = KnowledgeBase(L)
    assert kb.task_event(0, RELEASE, 0.0, observers=[L.station]) == 0
    assert kb.task_event(0, START, 1.0, members=(0,), observers=[0]) == 1
    assert kb.task_event(0, START, 1.0, members=(0,), observers=[1]) == 1  # a second observer, same fact
    with pytest.raises(ValueError):
        kb.task_event(0, RELEASE, 2.0)
    kb.observe_task([1], 0)
    b = kb.belief(1, 1.0)
    assert b.started()[0] and not b.open_tasks()[0] and not b.released[1]
    kb.sync()
    assert kb.belief(0, 1.0).released[0] and kb.belief(0, 1.0).task_state(0).members == (0,)


# ------------------------------------------------------------------------------------------------ CRN determinism
@pytest.mark.parametrize("model", [Bernoulli(0.3), GilbertElliott(0.2, 0.8)])
def test_loss_calendar_depends_only_on_key_and_tick(model):
    key = CRNKey("MA-AT-25-5-50", 3, 1)
    a = type(model)(**{k: getattr(model, k) for k in ("p", "rho") if hasattr(model, k)}).bind(key, 6)
    b = type(model)(**{k: getattr(model, k) for k in ("p", "rho") if hasattr(model, k)}).bind(key, 6)
    d = np.zeros((6, 6))
    full = [a.lost(k, d) for k in range(120)]
    for k in (119, 50, 7, 80):  # any query order, skipped ticks
        assert np.array_equal(b.lost(k, d), full[k])
    c = type(model)(**{k: getattr(model, k) for k in ("p", "rho") if hasattr(model, k)}).bind(CRNKey("x", 3, 2), 6)
    assert not all(np.array_equal(c.lost(k, d), full[k]) for k in range(120))


def test_ge_stationary_loss_and_correlation():
    ge = GilbertElliott(0.2, 0.8).bind(CRNKey("s", 0, 0), 30)
    states = np.array([ge.lost(k, None) for k in range(2000)])
    assert abs(states.mean() - 0.2) < 0.01
    x = states.reshape(2000, -1).astype(float)
    x0, x1 = x[:-1] - x.mean(), x[1:] - x.mean()
    assert abs((x0 * x1).mean() / x.var() - 0.8) < 0.03  # lag-one correlation rho


def test_rayleigh_calibration_matches_mean_drop():
    R = comm_range(1.0, 25)
    ra = Rayleigh.calibrated(0.2, R)
    d = np.sqrt(np.random.default_rng(0).random(200000)) * R  # uniform over the disk
    assert abs((1 - ra.p_success(d)).mean() - 0.2) < 0.005
    assert ra.p_success(np.array([0.1 * R]))[0] > 0.99 > ra.p_success(np.array([R]))[0]


def test_channel_realization_is_independent_of_traffic():
    """Two methods on the same CRN key see the same delivered hops, whatever they send."""
    L = Layout(5, 4, 1)
    rng = np.random.default_rng(3)
    pos_seq = [rng.random((L.N, 2)) * 0.4 for _ in range(40)]
    ups = []
    for traffic in (0, 1):
        kb = KnowledgeBase(L)
        tr = GossipTransport(kb, 0.3, GilbertElliott(0.3, 0.8), key=CRNKey("s", 1, 1), budget=1e9)
        rec = []
        for k, pos in enumerate(pos_seq):
            tr.deliver(k, pos, np.ones(L.N, bool))
            rec.append(tr.last_up.copy())
            if traffic and k % 3 == 0:
                kb.append_log(k % L.A, ABANDON, k % L.A, 0, k * TICK)
            tr.send(k, pos, np.ones(L.N, bool))
        ups.append(rec)
    assert all(np.array_equal(a, b) for a, b in zip(*ups))


# ------------------------------------------------------------------------------------------------ reductions
def test_ideal_transport_equals_global_knowledge_every_tick():
    L = Layout(6, 10, 2)
    kb = KnowledgeBase(L)
    tr = IdealTransport(kb)
    rng = np.random.default_rng(4)
    alive = np.ones(L.N, bool)
    for k in range(30):
        tr.deliver(k, None, alive)
        n = int(rng.integers(L.N))
        kb.heartbeat(min(n, L.A - 1), k * TICK, IDLE, force=True)
        kb.append_log(n, ABANDON, 0, int(rng.integers(L.T)), k * TICK)
        if k < L.T:
            kb.task_event(k, RELEASE, k * TICK, observers=[L.station])
        tr.sync(alive)
        g = kb.global_view()
        assert all(np.array_equal(kb.v[i], g) for i in range(L.N))
        tr.send(k, None, alive)


def test_t2_unlimited_reduces_to_global_knowledge_with_one_tick_per_hop():
    """R = inf, no loss, no budget: once digests are exchanged, every write reaches every node one tick later."""
    L = Layout(8, 6, 1)
    kb = KnowledgeBase(L)
    tr = GossipTransport(kb, math.inf, NoLoss(), budget=math.inf)
    alive = np.ones(L.N, bool)
    pos = np.random.default_rng(0).random((L.N, 2))
    snap = kb.global_view()
    for k in range(40):
        tr.deliver(k, pos, alive)
        if k >= 2:
            assert all(np.array_equal(kb.v[i], snap) for i in range(L.N))  # the writes of tick k-1 arrived
        if k < 30:
            kb.append_log(k % L.N, ABANDON, 0, 0, k * TICK)
            kb.heartbeat(k % L.A, k * TICK, IDLE, force=True)
        snap = kb.global_view()
        tr.send(k, pos, alive)


def test_multi_hop_latency_is_one_tick_per_hop():
    L = Layout(6, 1, 1)
    kb = KnowledgeBase(L)
    R = 0.11
    pos = np.array([[0.1 * i, 0.0] for i in range(L.A)] + [[5.0, 5.0]])
    tr = GossipTransport(kb, R, NoLoss(), budget=math.inf)
    alive = np.ones(L.N, bool)
    k0, arrival = 5, {}
    for k in range(20):
        tr.deliver(k, pos, alive)
        for i in range(L.A):
            if i not in arrival and kb.v[i, L.LOG.start] >= 1:
                arrival[i] = k
        if k == k0:
            kb.append_log(0, ABANDON, 0, 0, k * TICK)
            arrival[0] = k
        tr.send(k, pos, alive)
    assert [arrival[i] - k0 for i in range(L.A)] == list(range(L.A))
    assert kb.v[L.station, L.LOG.start] == 0  # the far station never hears it


def test_byte_budget_is_respected_and_backlog_drains():
    L = Layout(4, 40, 3)
    kb = KnowledgeBase(L)
    budget = 400.0
    tr = GossipTransport(kb, math.inf, NoLoss(), budget=budget)
    alive = np.ones(L.N, bool)
    pos = np.zeros((L.N, 2))
    for j in range(L.T):
        kb.task_event(j, RELEASE, 0.0, observers=[L.station])
    prev = np.zeros(L.N)
    for k in range(60):
        tr.deliver(k, pos, alive)
        tr.send(k, pos, alive)
        assert np.all(tr.stats.bytes - prev <= budget)
        prev = tr.stats.bytes.astype(float).copy()
    assert tr.stats.truncated > 0
    g = kb.global_view()
    assert all(np.array_equal(kb.v[i], g) for i in range(L.N))


def test_beacon_reaches_out_of_range_nodes():
    L = Layout(3, 2, 1)
    kb = KnowledgeBase(L)
    from cbba_sota.dyn.comm import Beacon
    tr = GossipTransport(kb, 0.01, NoLoss(), budget=math.inf, beacon=Beacon(period=1.0, loss=0.0))
    pos = np.array([[0.0, 0.0], [0.5, 0.0], [1.0, 0.0], [0.5, 0.5]])
    alive = np.ones(L.N, bool)
    kb.task_event(0, RELEASE, 0.0, observers=[L.station])
    for k in range(12):
        tr.deliver(k, pos, alive)
        tr.send(k, pos, alive)
    assert (kb.v[:, 0] == 0).all() and tr.stats.beacon_bytes > 0


def test_oracle_hooks():
    L = Layout(3, 2, 1)
    kb = KnowledgeBase(L)
    tr = GossipTransport(kb, 0.0, NoLoss(), oracle="agent")
    kb.heartbeat(1, 0.0, WAIT, target=0)
    kb.task_event(0, RELEASE, 0.0, observers=[L.station])
    tr.deliver(0, np.zeros((L.N, 2)), np.ones(L.N, bool))
    assert (kb.v[:, L.REC.start + 1] == 0).all() and kb.v[0, 0] == -1  # records global, tasks not


# ------------------------------------------------------------------------------------------------ membership
def test_connectivity_radius_and_components():
    assert abs(connectivity_radius(51) - math.sqrt(math.log(51) / (math.pi * 51))) < 1e-15
    assert comm_range(0.5, 50) == pytest.approx(0.5 * connectivity_radius(51))
    pos = np.array([[0, 0], [0.05, 0], [0.5, 0.5], [0.55, 0.5], [0.52, 0.52]], float)
    lab = components(pos, 0.1)
    assert lab[0] == lab[1] != lab[2] == lab[3] == lab[4]
    assert len(set(components(pos, 0.1, alive=np.array([1, 0, 1, 1, 1], bool)))) == 3


def test_leader_is_station_or_lowest_reachable_id():
    L = Layout(5, 1, 1)
    kb = KnowledgeBase(L)
    tr = GossipTransport(kb, 0.1, NoLoss())
    pos = np.array([[0.9, 0.9], [0.5, 0.52], [0.95, 0.9], [0.52, 0.5], [0.9, 0.95], [0.5, 0.5]])
    alive = np.ones(L.N, bool)
    for k in range(12):
        t = k * TICK
        tr.deliver(k, pos, alive)
        for b in range(L.A):
            kb.heartbeat(b, t, IDLE, pos=pos[b])
        kb.heartbeat(L.station, t, STATION, pos=pos[L.station])
        tr.send(k, pos, alive)
    lead = leaders(reach_matrix(kb, 11 * TICK, 1.5), L.station)
    assert lead.tolist() == [0, L.station, 0, L.station, 0, L.station]


# ------------------------------------------------------------------------------------------------ leases, 4.6
def _kb_with(records):
    L = Layout(4, 3, 1)
    kb = KnowledgeBase(L)
    for b, (t, mode, target, route) in records.items():
        kb.heartbeat(b, t, mode, target=target, route=route, force=True)
    kb.sync()
    return kb


def test_patient_lease_rules_and_absent_records():
    kb = _kb_with({0: (5.0, WAIT, 1, ()), 1: (5.0, IDLE, -1, ()), 2: (1.0, TRAVEL, 1, ())})
    cfg = LeaseConfig("patient", grace=1.0, max_wait=10.0, silence=2.0)
    b = kb.belief(0, 6.0)
    assert not lease_check(b, 0, 1, 5.0, cfg=cfg).abandon  # robot 2 is (believed) coming
    d = lease_check(kb.belief(0, 15.0), 0, 1, 5.0, cfg=cfg)
    assert d.abandon and d.reason == "max_wait" and d.absent == (2,)  # 2 is silent since t = 1
    publish_abandon(kb, 0, 1, 15.0, d)
    b = kb.belief(0, 15.0)
    assert (0, 1) in b.abandons() and (2, 1) in b.absents()
    kb2 = _kb_with({0: (5.0, WAIT, 1, ()), 1: (5.0, IDLE, -1, ())})
    assert not lease_check(kb2.belief(0, 5.5), 0, 1, 5.0, cfg=cfg).abandon  # grace not over
    d = lease_check(kb2.belief(0, 6.0), 0, 1, 5.0, cfg=cfg)
    assert d.abandon and d.reason == "no_partner" and d.absent == ()


def test_detector_lease_waits_while_partners_heartbeat():
    cfg = LeaseConfig("detector", detector_timeout=0.5)
    kb = _kb_with({0: (3.0, WAIT, 1, ()), 1: (3.0, TRAVEL, 1, ())})
    assert not lease_check(kb.belief(0, 3.4), 0, 1, 2.0, cfg=cfg).abandon
    d = lease_check(kb.belief(0, 3.6), 0, 1, 2.0, cfg=cfg)
    assert d.abandon and d.reason == "suspected"


def test_failure_needs_silence_and_on_site_evidence():
    kb = _kb_with({0: (5.0, WAIT, 1, ()), 1: (5.0, IDLE, -1, ()), 2: (1.0, TRAVEL, 1, ())})
    assert suspected(kb.belief(0, 1.6), 2, 0.5)
    assert not declared_failed(kb.belief(0, 20.0), 2, 10.0)  # silent but no absent record: only unreachable
    kb.append_log(0, ABSENT, 2, 1, 12.0)
    kb.sync()
    assert not declared_failed(kb.belief(0, 10.5), 2, 10.0)  # evidence, not yet silent for h_f
    assert declared_failed(kb.belief(0, 11.0), 2, 10.0)


# ------------------------------------------------------------------------------------------------ arm scopes
def _scope(arm_spec, node, members, n_robots=6):
    kb, _tr, runner = make_stack(n_robots, 4, 1, arm_spec, planner=lambda s, b: {}, rho=1.0)
    for b in range(n_robots):
        kb.heartbeat(b, 0.0, IDLE, force=True)
    return runner.scope_for(node, kb.belief(node, 0.0), frozenset(members), frozenset(), 0, "test")


def test_arm_scopes_under_a_partition():
    st, comp_st, comp_x = 6, {0, 1, 2}, {3, 4}  # robot 5 is alone
    s = _scope(ARMS["CEN-F"], st, comp_st)
    assert s.writable == comp_st and s.frozen == {3, 4, 5} and not s.clamped
    assert _scope(ARMS["CEN-F"], 3, comp_x) is None  # stranded components do not plan
    s = _scope(ARMS["SPARC"], st, comp_st)
    assert s.writable == comp_st and s.frozen == {3, 4, 5}  # station anchoring = CEN-F
    s = _scope(ARMS["SPARC"], 3, comp_x)
    assert s.writable == set(range(6)) and s.clamped == {0, 1, 2, 5}
    s = _scope(ARMS["REP-clamp"], st, comp_st)
    assert s.writable == set(range(6)) and s.clamped == {3, 4, 5}
    s = _scope(ARMS["INF-r"], 3, comp_x)
    assert s.writable == comp_x and s.head_only == {0, 1, 2, 5} and not s.frozen
    assert ARMS["HYB"].stranded_trigger == "v1" and sparc("V1").stranded_trigger == "v1"
    assert sparc("V3", 0.5).theta == 0.5
    assert deployment_switch(1.0, 0.75).station_scope == "component"
    assert deployment_switch(0.5, 0.75).station_scope == "all"
    assert len({a.name for a in factorial_arms()}) == 16
    assert arm("CEN-F", rally=False).rally is False


def test_planning_view_clamp_frozen_and_absent():
    inst = SimpleNamespace(req=np.ones((4, 1)), loc=np.array([[0.1 * j, 0.0] for j in range(4)]),
                           ab=np.ones((3, 1)), depot=np.zeros((3, 2)), dur=np.ones(4))
    kb, _tr, runner = make_stack(3, 4, 1, ARMS["SPARC"], planner=lambda s, b: {}, rho=1.0)
    for j in range(4):
        kb.task_event(j, RELEASE, 0.0, observers=[3])
    kb.heartbeat(0, 1.0, TRAVEL, target=0, t_ref=2.0, route=(1,), force=True)  # stale: ETA 2.0 in the past
    kb.heartbeat(1, 5.0, WAIT, target=2, t_ref=4.5, force=True)
    kb.heartbeat(2, 5.0, WORK, target=3, t_ref=6.0, force=True)
    kb.task_event(3, START, 5.0, members=(2,), observers=[2])
    kb.append_log(1, ABSENT, 0, 0, 5.0)
    kb.sync()
    b = kb.belief(1, 5.0)
    sc = runner.scope_for(1, b, frozenset({1, 2}), frozenset(), 50, "t")  # stranded leader 1: 0 is clamped
    pv = planning_view(sc, b, inst)
    assert pv.ready[0] == 5.0 and pv.fixed[0] == ()  # clamped to now; head 0 dropped by the absent record
    assert pv.fixed[1] == (2,) and pv.ready[1] == 5.0 and pv.ready[2] == 6.0
    assert pv.open.tolist() == [True, True, True, False] and pv.committed.tolist() == [False, False, True, False]
    sc2 = runner.scope_for(3, b, frozenset({1, 2}), frozenset(), 50, "t")  # station: 0 frozen on its route
    assert planning_view(sc2, b, inst).fixed[0] == (1,)


# ------------------------------------------------------------------------------------------------ toy world
class ToyWorld:
    """Single-trait robots (1 each), tasks needing req robots, straight legs at speed v, tick-quantized events.
    Test harness only: exercises the per-tick protocol of protocols.py end to end."""

    def __init__(self, inst, release, depot_xy, runner, kb, lease, v=0.2, t_max=80.0, fail=None):
        self.inst, self.rel, self.runner, self.kb, self.lease, self.v, self.t_max = inst, release, runner, kb, lease, \
            v, t_max
        A = len(depot_xy)
        self.A, self.T = A, len(release)
        self.xy = np.array(depot_xy, float)
        self.p0, self.p1 = self.xy.copy(), self.xy.copy()
        self.dep = np.zeros(A)
        self.arr = np.zeros(A)
        self.mode = np.full(A, IDLE)
        self.target = np.full(A, -1)
        self.route = [[] for _ in range(A)]
        self.key = np.full(A, -1)
        self.wait_since = np.zeros(A)
        self.present = [set() for _ in range(self.T)]
        self.state = np.zeros(self.T, int)  # 0 open, 1 started, 2 done
        self.fin = np.full(self.T, np.inf)
        self.members = [() for _ in range(self.T)]
        self.alive = np.ones(A + 1, bool)
        self.fail = fail or {}
        self.station = np.array([0.5, 0.5])
        self.wasted = 0
        self.check = None  # optional callback(kb) after every planning phase
        self.briefing = True  # t = 0 full-comm briefing
        self.n_checks = 0

    def pos(self, t):
        out = self.xy.copy()
        for b in range(self.A):
            if self.mode[b] in (TRAVEL, RALLY) and self.arr[b] > self.dep[b]:
                f = min(max((t - self.dep[b]) / (self.arr[b] - self.dep[b]), 0.0), 1.0)
                out[b] = self.p0[b] + (self.p1[b] - self.p0[b]) * f
        return np.vstack([out, self.station])

    def depart(self, b, t, xy, target, mode):
        here = self.pos(t)[b]
        self.p0[b], self.p1[b] = here, np.asarray(xy, float)
        self.dep[b], self.arr[b] = t, t + float(np.linalg.norm(self.p1[b] - here)) / self.v
        self.mode[b], self.target[b] = mode, target

    def run(self):
        kb, L, rn = self.kb, self.kb.layout, self.runner
        for k in range(int(self.t_max / TICK) + 1):
            t = k * TICK
            for b, tf in self.fail.items():
                if self.alive[b] and t >= tf:
                    self.alive[b] = False
                    self.mode[b] = IDLE
                    for j in range(self.T):
                        self.present[j].discard(b)
            # physics
            for j in np.flatnonzero((self.state == 1) & (self.fin <= t + 1e-9)):
                self.state[j] = 2
                kb.task_event(j, FINISH, self.fin[j], observers=self.members[j])
                for m in self.members[j]:
                    self.mode[m], self.target[m] = IDLE, -1
            for b in range(self.A):
                if not self.alive[b] or self.mode[b] not in (TRAVEL, RALLY) or self.arr[b] > t + 1e-9:
                    continue
                self.xy[b] = self.p1[b]
                if self.mode[b] == RALLY:
                    self.mode[b] = IDLE
                    continue
                j = self.target[b]
                if self.state[j] != 0:
                    self.wasted += 1
                    kb.observe_task([b], j)
                    self.mode[b], self.target[b] = IDLE, -1
                    continue
                self.mode[b], self.wait_since[b] = WAIT, t
                self.present[j].add(b)
            for j in range(self.T):
                if self.state[j] == 0 and self.present[j] and (
                        self.inst.ab[sorted(self.present[j])].sum(0) >= self.inst.req[j] - 1e-9).all():
                    self.state[j] = 1
                    self.fin[j] = t + self.inst.dur[j]
                    self.members[j] = tuple(sorted(self.present[j]))
                    kb.task_event(j, START, t, members=self.members[j], observers=self.members[j])
                    for m in self.members[j]:
                        self.mode[m] = WORK
                    self.present[j] = set()
            for j in np.flatnonzero(self.rel <= t + 1e-9):
                if kb.v[L.station, j] < 0:
                    kb.task_event(j, RELEASE, float(self.rel[j]), observers=[L.station])
            if (self.state == 2).all():
                return True, t
            self.heartbeats(t)
            pos = self.pos(t)
            rn.deliver(k, t, pos, self.alive)
            if k == 0 and self.briefing:
                rn.briefing(k, t, pos, self.alive)
            rn.plan(k, t, pos, self.alive)
            if self.check is not None:
                self.check(kb)
                self.n_checks += 1
            for b, ver in rn.adoptions(self.alive).items():
                head = self.target[b] if self.mode[b] in (TRAVEL, WAIT, WORK) else -1
                self.route[b] = [x for x in ver.tasks if x != head]
                self.key[b] = L.route_key(ver.epoch, ver.author)
            self.leases(t)
            self.decide(t)
            self.heartbeats(t)
            rn.send(k, t, self.pos(t), self.alive)
        return False, self.t_max

    def heartbeats(self, t):
        kb = self.kb
        for b in range(self.A):
            if not self.alive[b]:
                continue
            m, j = int(self.mode[b]), int(self.target[b])
            t_ref = self.arr[b] if m == TRAVEL else (self.wait_since[b] if m == WAIT else
                                                      (self.fin[j] if m == WORK else t))
            kb.heartbeat(b, t, m, target=j, t_ref=float(t_ref), pos=self.pos(t)[b], route_key=int(self.key[b]),
                         route=tuple(self.route[b]))
        kb.heartbeat(kb.layout.station, t, STATION, pos=self.station)

    def leases(self, t):
        for b in np.flatnonzero(self.mode == WAIT):
            if not self.alive[b]:
                continue
            j = int(self.target[b])
            bel = self.kb.belief(b, t)
            d = lease_check(bel, b, j, self.wait_since[b], present=self.present[j], cfg=self.lease)
            if d.abandon:
                publish_abandon(self.kb, b, j, t, d)
                self.present[j].discard(b)
                self.mode[b], self.target[b] = IDLE, -1

    def decide(self, t):
        for b in range(self.A):
            if not self.alive[b] or self.mode[b] not in (IDLE, RALLY):
                continue
            bel = self.kb.belief(b, t)
            live = bel.open_tasks()
            self.route[b] = [x for x in self.route[b] if live[x] or self.state[x] == 0 and bel.task_seq[x] < 0]
            if self.route[b]:
                j = self.route[b].pop(0)
                self.depart(b, t, self.inst.loc[j], j, TRAVEL)
            elif self.mode[b] == IDLE and self.runner.arm.rally and not np.allclose(self.xy[b], self.station):
                from cbba_sota.dyn.protocols import outside_station_component
                if outside_station_component(bel, self.runner.cfg.member_window):
                    self.depart(b, t, self.station, -1, RALLY)


def greedy_planner(inst, v=0.2):
    """List scheduler over ``planning_view``: fixed prefixes first, then open tasks in id order to the earliest
    available robots (one global placement order, so routes are key-consistent)."""

    def plan(scope, belief):
        pv = planning_view(scope, belief, inst)
        A = len(pv.ready)
        routes = {b: list(pv.fixed[b]) for b in scope.writable}
        free, xy = pv.ready.copy(), pv.pos.copy()
        for b in range(A):
            for j in pv.fixed[b]:
                free[b] = max(free[b] + np.linalg.norm(xy[b] - inst.loc[j]) / v, pv.now) + inst.dur[j]
                xy[b] = inst.loc[j]
        for j in np.flatnonzero(pv.open & (pv.resid[:, 0] > 0)):
            need = int(np.ceil(pv.resid[j, 0]))
            cand = [b for b in range(A) if pv.avail[b] and np.isfinite(free[b]) and j not in routes.get(b, [])]
            eta = {b: free[b] + np.linalg.norm(xy[b] - inst.loc[j]) / v for b in cand}
            pick = sorted(cand, key=lambda b: (eta[b], b))[:need]
            if len(pick) < need:
                continue
            start = max(eta[b] for b in pick)
            for b in pick:
                routes[b].append(int(j))
                free[b], xy[b] = start + inst.dur[j], inst.loc[j]
        return {b: tuple(r) for b, r in routes.items()}

    return plan


def _toy(n_robots=6, n_tasks=12, seed=0):
    rng = np.random.default_rng(seed)
    loc = rng.random((n_tasks, 2))
    req = rng.integers(1, 3, (n_tasks, 1)).astype(float)
    inst = SimpleNamespace(req=req, loc=loc, ab=np.ones((n_robots, 1)), dur=rng.uniform(0.5, 2.0, n_tasks),
                           depot=np.tile([[0.5, 0.5]], (n_robots, 1)))
    release = np.where(np.arange(n_tasks) < n_tasks // 2, 0.0, rng.uniform(0, 6, n_tasks))
    depot = np.array([[0.5, 0.5]] * n_robots) + rng.normal(0, 0.05, (n_robots, 2))
    return inst, release, depot


def _run(arm_spec, rho, seed=0, loss="ge", fail=None, check=None, shadow=()):
    inst, release, depot = _toy(seed=seed)
    kb, _tr, rn = make_stack(len(depot), len(release), 1, arm_spec, greedy_planner(inst), rho=rho, loss=loss,
                             key=CRNKey("toy", seed, 0), shadow=shadow)
    lease = LeaseConfig("detector") if arm_spec.ideal else LeaseConfig("patient", 1.0, 10.0)
    w = ToyWorld(inst, release, depot, rn, kb, lease, fail=fail)
    w.check = check
    ok, ms = w.run()
    return ok, ms, rn, kb, w


def test_toy_full_completes_and_replicas_stay_global():
    def check(kb):  # ideal comms: every replica IS global knowledge whenever a planner reads it
        g = kb.global_view()
        assert all(np.array_equal(kb.v[i], g) for i in range(kb.layout.N))

    ok, ms, rn, _kb, w = _run(ARMS["FULL"], math.inf, check=check)
    assert ok and ms < 60 and w.n_checks > 100
    assert rn.n_plans >= 2 and rn.n_plans_off_station == 0


def test_toy_full_detects_a_failure_and_recovers():
    ok, _ms, rn, _kb, _w = _run(ARMS["FULL"], math.inf, fail={0: 2.0})
    assert ok
    assert any(c.reason == "orphan" for c in rn.calls)  # detector fired -> structural re-plan


@pytest.mark.parametrize("name", ["SPARC", "CEN-F", "HYB", "REP-clamp", "INF-r"])
def test_toy_arms_run_under_t2(name):
    ok, _ms, rn, _kb, _w = _run(ARMS[name], 2.0)
    assert ok, name
    s = rn.summary()
    assert s["frames"] > 0 and s["bytes"] > 0 and 0 < s["station_frac"] <= 1


def test_toy_stack_is_deterministic_given_the_crn_key():
    runs = [_run(ARMS["SPARC"], 1.0) for _ in range(2)]
    (_ok1, ms1, rn1, kb1, _), (_ok2, ms2, rn2, kb2, _) = runs
    assert ms1 == ms2 and np.array_equal(kb1.v, kb2.v)
    assert [(c.tick, c.node, c.reason, c.authored) for c in rn1.calls] == \
           [(c.tick, c.node, c.reason, c.authored) for c in rn2.calls]


def test_toy_sparc_equals_full_station_decisions_with_ideal_comms():
    """Good comms: SPARC is central RH with structural triggers (identity checked as an implementation test)."""
    a = _run(ARMS["FULL"], math.inf)
    b = _run(arm("SPARC", ideal=True), math.inf)  # station anchoring: with ideal comms everyone is a member
    assert a[1] == b[1]
    assert [(c.tick, c.authored) for c in a[2].calls] == [(c.tick, c.authored) for c in b[2].calls]


def test_shadow_arms_log_decision_differences():
    _ok, _ms, rn, _kb, _w = _run(ARMS["SPARC"], 1.0, shadow=(ARMS["SPARC"], ARMS["REP-clamp"], ARMS["CEN-F"]))
    calls = [c for c in rn.calls if c.shadow]
    assert calls and not any(c.shadow["SPARC"] for c in calls)  # an arm never differs from itself
    off = [c for c in calls if c.node != 6]
    assert all(c.shadow["CEN-F"] == bool(c.authored) for c in off)  # CEN-F never plans in stranded components


def test_idle_trigger_fires_again_when_a_robot_goes_idle_again():
    kb, _tr, rn = make_stack(2, 2, 1, ARMS["FULL"], planner=lambda s, b: {})
    pos, alive = np.zeros((3, 2)), np.ones(3, bool)
    kb.task_event(0, RELEASE, 0.0, observers=[2])
    reasons = []
    for k, mode in enumerate([TRAVEL, TRAVEL, IDLE, TRAVEL, IDLE, IDLE]):
        t = k * TICK
        kb.heartbeat(0, t, mode, target=1 if mode == TRAVEL else -1, force=True)
        kb.heartbeat(1, t, WORK, target=1, force=True)
        kb.heartbeat(2, t, STATION, force=True)
        rn.deliver(k, t, pos, alive)
        if k == 0:
            rn.briefing(k, t, pos, alive)
        reasons.append([c.reason for c in rn.plan(k, t, pos, alive)])
    assert reasons == [[], [], ["idle"], [], ["idle"], []]


def _dev_instance():
    from cbba_sota.hetero import Instance
    return Instance.from_pickle(ROOT / "data" / "hetero" / "MA-AT-25-5-50" / "dev" / "env_0.pkl")


@pytest.mark.parametrize(("arm_name", "rho", "frozen_model"),
                         [("FULL", math.inf, "anchor"), ("SPARC", 1.0, "anchor"), ("CEN-F", 0.5, "anchor"),
                          ("CEN-F", 1.0, "head")])
def test_agent_b_planner_runs_through_the_runner(arm_name, rho, frozen_model):
    """Integration: agent B's RHPlanner (SPARC kernel) on replicas, real dev instance (MA-AT-25-5-50 dev 0), toy
    world physics (no noise). Checks the adapter produces valid plans and the episode completes."""
    planner_mod = pytest.importorskip("cbba_sota.dyn.planner")
    from cbba_sota.dyn.protocols import plan_state, rh_planner
    inst = _dev_instance()
    T, A = inst.n_tasks, inst.n_agents
    rng = np.random.default_rng(0)
    release = np.where(rng.random(T) < 0.5, 0.0, rng.uniform(0, 10, T))
    base = rh_planner(inst, iters=20, frozen_model=frozen_model)
    problems = []

    def planner(scope, belief):
        out = base(scope, belief)
        state, mask, _ = plan_state(scope, belief, inst, frozen_model=frozen_model)
        problems.extend(planner_mod.check_plan(state, base.last, mask))
        return out

    arm_spec = ARMS[arm_name]
    kb, _tr, rn = make_stack(A, T, inst.n_traits, arm_spec, planner, rho=rho, key=CRNKey(0, 500000, 0))
    lease = LeaseConfig("detector") if arm_spec.ideal else LeaseConfig("patient", 1.0, 30.0)
    w = ToyWorld(inst, release, inst.depot, rn, kb, lease, t_max=150.0)
    ok, ms = w.run()
    assert not problems, problems[:5]
    assert ok and 10 < ms < 150
    assert rn.n_plans >= 2
    assert all(v.keys is not None for v in kb.facts.routes.values())


def test_runner_rejects_writes_outside_scope():
    kb, _tr, rn = make_stack(3, 2, 1, ARMS["CEN-F"], planner=lambda s, b: {2: (0,)}, rho=0.1,
                            cfg=ProtocolConfig(member_window=1.5))
    for b in range(3):
        kb.heartbeat(b, 0.0, IDLE, pos=(0.1 * b, 0.0), force=True)
    kb.heartbeat(3, 0.0, STATION, pos=(0.5, 0.5), force=True)
    pos = np.array([[0.0, 0.0], [0.1, 0.0], [0.2, 0.0], [0.5, 0.5]])
    with pytest.raises(ValueError):
        rn.plan(0, 0.0, pos, np.ones(4, bool))
    assert isinstance(rn, ProtocolRunner)


def test_ideal_heartbeat_is_every_tick():
    kb, _tr, rn = make_stack(2, 1, 1, ARMS["FULL"], planner=lambda s, b: {})
    assert kb.hb.period == TICK and rn.cfg.failure_rule == "detector"
    assert HeartbeatConfig(1.0).due(0.0, 1.0) and not HeartbeatConfig(1.0).due(0.0, 0.9)
    assert RALLY != IDLE


# ------------------------------------------------------------------------------------------------ cbja oracle
@pytest.mark.skipif(not CBJA_PY.exists(), reason="cbja venv (simpy) not available")
def test_channel_matches_cbja_network_on_five_cases():
    out = subprocess.run([str(CBJA_PY), str(ROOT / "scripts" / "trackD_comm_crosscheck.py"), "--json"],
                         capture_output=True, text=True, timeout=300, env={"OMP_NUM_THREADS": "1", "PATH": ""},
                         check=False)
    assert out.returncode == 0, out.stdout + out.stderr
    res = json.loads(out.stdout.strip().splitlines()[-1])
    assert res["ok"] and len(res["cases"]) == 6  # case 2 has two variants
    assert sys.version_info >= (3, 10)
