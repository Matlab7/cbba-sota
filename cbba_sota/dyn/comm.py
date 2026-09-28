"""Track D communication tier T2 (spec 3.3, 3.5): channel calendars with common random numbers, delta gossip.

Geometry. Nodes are robots ``0 .. A-1`` and the station (node ``A``). Range ``R = rho * r_c(n)`` with the
random-geometric-graph connectivity radius ``r_c(n) = sqrt(ln n / (pi n))``, ``n = robots + 1``. A frame sent at
tick k travels during ``[t_k, t_k+1]``; the link is up iff the analytic distance between the two nodes is <= R at
both ends of the hop. Distance between two straight legs is convex in time, so this equals "in range during the
whole hop" whenever neither node turns inside the tick (exact against cbja/network.py, which drops a packet whose
link breaks in flight; see ``scripts/trackD_comm_crosscheck.py``). Latency is one tick per hop, no instant flooding.

Loss (CV_MRTA protocol, arXiv 2609.13711, reimplemented; the code has no licence), applied to links in range:
- ``Bernoulli(p)``: each directed copy is lost with probability p.
- ``GilbertElliott(p, rho)``: a two-state chain per directed link, ``pGG = 1-(1-rho)p``, ``pBB = rho+(1-rho)p``,
  lost in the bad state; initialised from the stationary law (bad w.p. p), delivery evaluated from the current
  state, then the state transitions. CV_MRTA transitions per message; here every node broadcasts one frame per
  tick, so the chain advances per tick (which also keeps it independent of method decisions, i.e. CRN).
- ``Rayleigh``: ``P_rx = Ptx - L0 - 10 eta log10(d/d0) + 10 log10 h``, ``h ~ Exp(1)``, delivered iff
  ``P_rx >= S``. CV_MRTA: L0 40 dB, eta 3.0, d0 1 m, Ptx 30 dBm, sensitivity ladder calibrated to its own arena
  ({-59.40 ... -32.58} dBm at 1 m/cell). Our arena is the unit square at a declared ``m_per_unit`` (1000 m), so
  ``Rayleigh.calibrated`` picks S such that the mean drop over links uniformly spread in the range disk equals p
  (the same mean as Bernoulli and GE). A fixed S can be given instead.

CRN. Every draw is keyed by (setting index, instance key, CRN seed, stream 5, substream, tick) -- the key convention
of ``cbba_sota.dyn.perturb.crn_rng``, which reserves stream 5 (``STREAM_CHANNEL``) for this layer -- and never by
method decisions: one uniform per directed link per tick, whether or not anything useful is sent. Two methods on
the same key face the same channel realization (the same (i, j, k) hops succeed).

Transports move replica versions between the rows of ``replica.KnowledgeBase.v`` (the version vectors):
- ``IdealTransport``: good comms (instant, lossless): every call joins all live replicas.
- ``GossipTransport``: tier T2. Each live node broadcasts one frame per tick: a header, its digest (its whole
  version vector, charged ``ByteModel.digest``) and the deltas that some neighbour lacks according to that
  neighbour's last digest (optimistically joined with what this node has sent since), in priority order
  (records, task events, logs, route versions; rotated per tick to avoid starvation) until the per-node byte
  budget is spent. Neighbours are the nodes heard in the previous tick, so a new contact costs one extra tick of
  handshake. Store-carry-forward is implicit: every node keeps and re-offers everything it holds.
- ``Beacon``: optional low-rate global channel (spec 3.3 "beacon factor"): every ``period`` each node sends one
  frame of ``budget`` bytes to every other node, each copy lost w.p. ``loss``. Symmetric payload: the same
  selection rule for every method (own record, own-authored route versions, task events, logs, others' records,
  others' routes; entries changed since the node's last beacon).
- ``oracle`` hooks (diagnosis arms): "agent" (teammate-state oracle), "know" (knowledge oracle: records, task
  events and logs are global) and "release" (release knowledge global).

Per-tick protocol for a world driving a transport (all transports implement it):
    physics to t_k -> world writes local observations -> ``deliver(k, pos_k, alive_k)`` (frames of tick k-1)
    -> planning / adoption / leases (they write into replicas) -> ``send(k, pos_k, alive_k)`` (frames of tick k).
This module needs only numpy (it is also imported under the cbja venv by the cross-check script).
"""
from __future__ import annotations

import math
import zlib
from dataclasses import dataclass, field

import numpy as np

from cbba_sota.dyn.replica import START, TICK, KnowledgeBase, Layout

# CRN: stream 5 is reserved for the channel in cbba_sota.dyn.perturb (STREAM_CHANNEL); these are its substreams
STREAM_CHANNEL = 5
S_LOSS, S_GE_INIT, S_GE_STEP, S_BEACON = 1, 2, 3, 4


def connectivity_radius(n_nodes: int) -> float:
    """r_c(n) = sqrt(ln n / (pi n)) (Penrose 2003; Gupta and Kumar 1998)."""
    return math.sqrt(math.log(n_nodes) / (math.pi * n_nodes))


def comm_range(rho: float, n_robots: int) -> float:
    """R = rho * r_c(robots + 1); rho = inf gives an unlimited range."""
    return math.inf if not math.isfinite(rho) else rho * connectivity_radius(n_robots + 1)


@dataclass(frozen=True)
class CRNKey:
    """(setting index, instance key, CRN seed) as in ``perturb.crn_rng``; a setting name is hashed (tests only)."""

    setting: int | str
    instance: int
    seed: int

    def ints(self) -> list[int]:
        s = self.setting if isinstance(self.setting, int) else zlib.crc32(str(self.setting).encode())
        return [int(s), int(self.instance), int(self.seed)]


DEFAULT_KEY = CRNKey("default", 0, 0)


def crn_rng(key: CRNKey, substream: int, *extra: int) -> np.random.Generator:
    return np.random.default_rng([*key.ints(), STREAM_CHANNEL, int(substream), *[int(x) for x in extra]])


def pairwise_dist(pos: np.ndarray) -> np.ndarray:
    d = pos[:, None, :] - pos[None, :, :]
    return np.sqrt((d ** 2).sum(-1))


def in_range(pos_a: np.ndarray, pos_b: np.ndarray, R: float) -> np.ndarray:
    """[N, N] link-up matrix for a hop from tick a to tick b (distance <= R at both ends)."""
    n = len(pos_a)
    if not math.isfinite(R):
        return ~np.eye(n, dtype=bool)
    tol = 1e-12
    up = (pairwise_dist(pos_a) <= R + tol) & (pairwise_dist(pos_b) <= R + tol)
    np.fill_diagonal(up, False)
    return up


def components(pos: np.ndarray, R: float, alive: np.ndarray | None = None) -> np.ndarray:
    """Connected-component label per node of the unit-disk graph at one instant (dead nodes are singletons)."""
    n = len(pos)
    alive = np.ones(n, bool) if alive is None else np.asarray(alive, bool)
    adj = np.ones((n, n), bool) if not math.isfinite(R) else pairwise_dist(pos) <= R + 1e-12
    adj &= alive[:, None] & alive[None, :]
    lab = np.full(n, -1, np.int64)
    c = 0
    for s in range(n):
        if lab[s] >= 0:
            continue
        lab[s] = c
        stack = [s]
        while stack:
            u = stack.pop()
            for w in np.flatnonzero(adj[u] & (lab < 0)):
                lab[w] = c
                stack.append(int(w))
        c += 1
    return lab


# ------------------------------------------------------------------------------------------------ loss models
class LossModel:
    """``lost(k, dist)`` -> [N, N] bool (sender, receiver) for frames sent at tick k; a pure function of (key, k)."""

    name = "none"

    def bind(self, key: CRNKey, n: int) -> LossModel:
        self.key, self.n = key, n
        return self

    def lost(self, k: int, dist: np.ndarray) -> np.ndarray:
        return np.zeros((self.n, self.n), bool)

    def _u(self, k: int) -> np.ndarray:
        return crn_rng(self.key, S_LOSS, k).random((self.n, self.n))


class NoLoss(LossModel):
    pass


@dataclass
class Bernoulli(LossModel):
    p: float = 0.2
    name: str = "bernoulli"

    def lost(self, k, dist):
        return self._u(k) < self.p


@dataclass
class GilbertElliott(LossModel):
    p: float = 0.2
    rho: float = 0.8
    name: str = "ge"
    _k: int = field(default=-1, repr=False)
    _bad: np.ndarray | None = field(default=None, repr=False)

    @property
    def pGG(self) -> float:
        return 1.0 - (1.0 - self.rho) * self.p

    @property
    def pBB(self) -> float:
        return self.rho + (1.0 - self.rho) * self.p

    def _reset(self):
        self._bad = crn_rng(self.key, S_GE_INIT).random((self.n, self.n)) < self.p
        self._k = 0

    def state(self, k: int) -> np.ndarray:
        """Bad-state matrix at tick k (advances the per-link calendar deterministically)."""
        if self._bad is None or k < self._k:
            self._reset()
        while self._k < k:
            u = crn_rng(self.key, S_GE_STEP, self._k).random((self.n, self.n))
            self._bad = np.where(self._bad, u < self.pBB, u >= self.pGG)
            self._k += 1
        return self._bad

    def lost(self, k, dist):
        return self.state(k).copy()


@dataclass
class Rayleigh(LossModel):
    sens_dbm: float = -80.0
    ptx_dbm: float = 30.0
    l0_db: float = 40.0
    eta: float = 3.0
    d0_m: float = 1.0
    m_per_unit: float = 1000.0
    name: str = "rayleigh"

    def h_min(self, dist: np.ndarray) -> np.ndarray:
        d_m = np.maximum(np.asarray(dist, float) * self.m_per_unit, self.d0_m)
        pl = self.l0_db + 10.0 * self.eta * np.log10(d_m / self.d0_m)
        return 10.0 ** ((self.sens_dbm - self.ptx_dbm + pl) / 10.0)

    def p_success(self, dist) -> np.ndarray:
        return np.exp(-self.h_min(dist))

    def lost(self, k, dist):
        u = self._u(k)
        return u < 1.0 - np.exp(-self.h_min(dist))  # h = -ln(1-u) ~ Exp(1); lost iff h < h_min

    @classmethod
    def calibrated(cls, p: float, R: float, **kw) -> Rayleigh:
        """Sensitivity S such that the mean drop over distances with density 2d/R^2 on [0, R] equals p."""
        if not math.isfinite(R):
            raise ValueError("calibration needs a finite range")
        d = (np.arange(4000) + 0.5) / 4000 * R
        w = 2 * d / R ** 2 * (R / 4000)
        lo, hi = -200.0, 50.0
        for _ in range(100):
            mid = 0.5 * (lo + hi)
            drop = 1.0 - float((cls(sens_dbm=mid, **kw).p_success(d) * w).sum())
            lo, hi = (lo, mid) if drop > p else (mid, hi)
        return cls(sens_dbm=0.5 * (lo + hi), **kw)


def make_loss(name: str, p: float = 0.2, rho: float = 0.8, R: float = math.inf, **kw) -> LossModel:
    if name in ("none", "ideal"):
        return NoLoss()
    if name == "bernoulli":
        return Bernoulli(p)
    if name == "ge":
        return GilbertElliott(p, rho)
    if name == "rayleigh":
        return Rayleigh.calibrated(p, R, **kw) if "sens_dbm" not in kw else Rayleigh(**kw)
    raise ValueError(name)


# ------------------------------------------------------------------------------------------------ byte model
@dataclass(frozen=True)
class ByteModel:
    """Declared wire sizes (bytes). Convention, frozen with the spec; the same for every method."""

    header: int = 8  # src 2, tick 4, flags 2
    task: int = 7  # id 2, seq 1, time 4
    release_payload: int = 6  # loc 2x2 (int16), duration 2 ... plus K requirement bytes
    record: int = 20  # robot 2, counter 2, mode 1, target 2, t_ref 4, pos 2x2, route key 3, hub 1, time 1
    route: int = 6  # robot 2, key 3, length 1 ... plus 2 per task (+2 per key)
    log: int = 3  # author 2, count 1 ... plus event bytes
    log_event: int = 8  # kind 1, robot 2, task 2, time 3

    def digest(self, L: Layout) -> int:
        return L.T * 1 + L.N * 2 + L.A * 3 + L.N * 1

    def members(self, L: Layout) -> int:
        return (L.A + 7) // 8


class _Costs:
    """Byte cost of shipping entry e of a replica vector, given the neighbour's believed version."""

    def __init__(self, kb: KnowledgeBase, bm: ByteModel):
        self.kb, self.bm, self.L = kb, bm, kb.layout
        L = self.L
        self.seg = L.segment_of()
        self.off = np.array([L.TASK.start, L.REC.start, L.ROUTE.start, L.LOG.start])

    def cost(self, e: int, have: int, bel_min: int) -> int:
        L, bm, f = self.L, self.bm, self.kb.facts
        s = int(self.seg[e])
        x = e - int(self.off[s])
        if s == 0:
            ev = f.task_events[x][have]
            c = bm.task + (bm.release_payload + L.n_traits if bel_min < 0 else 0)
            if ev.kind in (START,):
                c += bm.members(L)
            return c
        if s == 1:
            return bm.record + 2 * len(f.records[x][have].route)
        if s == 2:
            v = f.routes[(x, have)]
            return bm.route + 2 * len(v.tasks) + (2 * len(v.keys) if v.keys else 0)
        return bm.log + bm.log_event * (have - max(bel_min, 0))


DEFAULT_BYTES = ByteModel()


# ------------------------------------------------------------------------------------------------ transports
@dataclass
class CommStats:
    frames: np.ndarray  # frames sent per node
    bytes: np.ndarray  # bytes sent per node
    frames_rx: np.ndarray  # frames received per node
    entries: np.ndarray  # delta entries sent per segment (TASK, REC, ROUTE, LOG)
    delta_bytes: np.ndarray = None  # bytes of deltas only (bytes minus headers and digests), per node
    truncated: int = 0  # frames that hit the byte budget
    beacon_frames: int = 0
    beacon_bytes: int = 0

    @classmethod
    def zeros(cls, n: int) -> CommStats:
        return cls(np.zeros(n, np.int64), np.zeros(n, np.int64), np.zeros(n, np.int64), np.zeros(4, np.int64),
                   np.zeros(n, np.int64))

    def as_dict(self) -> dict:
        return {"frames": int(self.frames.sum()), "bytes": int(self.bytes.sum()),
                "delta_bytes": int(self.delta_bytes.sum()),
                "frames_rx": int(self.frames_rx.sum()), "entries": self.entries.tolist(),
                "truncated": self.truncated, "beacon_frames": self.beacon_frames, "beacon_bytes": self.beacon_bytes}


class Transport:
    """Base: holds the knowledge base, oracle hook and stats. Subclasses implement ``deliver`` / ``send``."""

    ideal = False

    def __init__(self, kb: KnowledgeBase, oracle: str | None = None, beacon: Beacon | None = None):
        self.kb, self.L = kb, kb.layout
        self.oracle = oracle
        self.beacon = beacon
        self.stats = CommStats.zeros(self.L.N)
        if beacon is not None:
            beacon.attach(kb, self.stats)

    def deliver(self, k: int, pos: np.ndarray, alive: np.ndarray) -> None:
        raise NotImplementedError

    def send(self, k: int, pos: np.ndarray, alive: np.ndarray) -> None:
        raise NotImplementedError

    def sync(self, alive: np.ndarray | None = None) -> None:
        """Only the ideal transport joins instantly; others ignore it."""

    def warm_start(self, k: int) -> None:
        """Called after the t = 0 full-comm briefing (no-op for the ideal transport)."""

    def _apply_oracle(self, alive: np.ndarray) -> None:
        if self.oracle is None:
            return
        L, v = self.L, self.kb.v
        idx = np.flatnonzero(alive)
        if self.oracle in ("agent", "know"):
            v[np.ix_(idx, np.arange(L.REC.start, L.REC.stop))] = v[:, L.REC].max(0)
        if self.oracle == "know":
            v[np.ix_(idx, np.arange(L.TASK.start, L.TASK.stop))] = v[:, L.TASK].max(0)
            v[np.ix_(idx, np.arange(L.LOG.start, L.LOG.stop))] = v[:, L.LOG].max(0)
        if self.oracle == "release":
            rel = np.where(v[L.station, L.TASK] >= 0, 0, -1)
            v[idx, L.TASK] = np.maximum(v[idx, L.TASK], rel)
        if self.oracle not in ("agent", "know", "release"):
            raise ValueError(self.oracle)


class IdealTransport(Transport):
    """Good comms: instant and lossless. Every call joins all live replicas (no bytes are charged)."""

    ideal = True

    def deliver(self, k, pos, alive):
        self.sync(alive)

    def send(self, k, pos, alive):
        self.sync(alive)

    def sync(self, alive=None):
        alive = np.ones(self.L.N, bool) if alive is None else np.asarray(alive, bool)
        self.kb.sync(np.flatnonzero(alive))


class GossipTransport(Transport):
    """Tier T2: unit-disk range, per-hop latency of one tick, CRN loss, per-node byte budget, digest-driven deltas."""

    def __init__(self, kb: KnowledgeBase, R: float, loss: LossModel | None = None, key: CRNKey = DEFAULT_KEY,
                 budget: float = 1500.0, byte_model: ByteModel = DEFAULT_BYTES, relay: bool = True,
                 oracle: str | None = None, beacon: Beacon | None = None):
        super().__init__(kb, oracle, beacon)
        L = self.L
        self.R = float(R)
        self.key = key
        self.loss = (loss or NoLoss()).bind(key, L.N)
        self.budget = float(budget)
        self.bm = byte_model
        self.relay = relay
        self.costs = _Costs(kb, byte_model)
        self.bottom = L.bottom()
        self.bel = np.tile(self.bottom, (L.N, L.N, 1))  # bel[i, j]: i's belief of j's replica
        self.heard = np.full((L.N, L.N), -(10 ** 9), np.int64)  # heard[i, j]: last tick i received j's frame
        self._fly = None  # frames in flight (sent at tick k, delivered at k + 1)
        self.digest_bytes = byte_model.digest(L)
        seg = L.segment_of()
        self._segs = [np.flatnonzero(seg == s) for s in (1, 0, 3, 2)]  # priority: REC, TASK, LOG, ROUTE
        self.last_up = np.zeros((L.N, L.N), bool)  # delivered hops of the last deliver() (for tests / metrics)
        self.hops_up = 0

    def warm_start(self, k: int) -> None:
        """After a full-comm briefing at tick k: every node knows every digest and has heard everyone."""
        self.bel[:] = self.kb.v[None, :, :]
        self.heard[:] = k
        self._fly = None

    # ---- delivery of the frames sent at tick k - 1
    def deliver(self, k, pos, alive):
        alive = np.asarray(alive, bool)
        fly, self._fly = self._fly, None
        self.last_up = np.zeros((self.L.N, self.L.N), bool)
        if fly is not None and fly["k"] == k - 1:
            kk, pos0, incl, snap, sent = fly["k"], fly["pos"], fly["incl"], fly["snap"], fly["sent"]
            up = in_range(pos0, pos, self.R) & sent[:, None] & alive[None, :]
            up &= ~self.loss.lost(kk, pairwise_dist(pos0))
            self.last_up = up
            self.hops_up += int(up.sum())
            vals = np.where(incl, snap, self.bottom[None, :])
            v = self.kb.v
            for i in np.flatnonzero(up.any(1)):
                recv = np.flatnonzero(up[i])
                v[recv] = np.maximum(v[recv], vals[i])
                own = np.where(incl[recv], snap[recv], self.bottom[None, :])  # receivers' own frames of tick k-1
                self.bel[recv, i] = np.maximum(snap[i][None, :], own)
                self.heard[recv, i] = kk
            self.stats.frames_rx += up.sum(0)
        if self.beacon is not None:
            self.beacon.deliver(k, alive)
        self._apply_oracle(alive)

    # ---- composing and sending the frames of tick k
    def send(self, k, pos, alive):
        alive = np.asarray(alive, bool)
        L, v = self.L, self.kb.v
        nbr = (self.heard >= k - 1) & alive[:, None]
        np.fill_diagonal(nbr, False)
        gt = (v[:, None, :] > self.bel) & nbr[:, :, None]
        need = gt.any(1)
        if not self.relay:  # gossip knockout: only own entries (task observations are always offered)
            own = np.zeros_like(need)
            own[:, L.TASK] = True
            for i in range(L.N):
                own[i, L.REC.start + i] = True
                own[i, L.LOG.start + i] = True
                ks = v[i, L.ROUTE]
                own[i, L.ROUTE] = (ks >= 0) & (ks % L.N == i)
            need &= own
        incl = np.zeros_like(need)
        avail = self.budget - self.bm.header - self.digest_bytes
        order = np.concatenate([np.roll(s, -(k % max(len(s), 1))) for s in self._segs])
        for i in np.flatnonzero(alive):
            used = 0
            cand = order[need[i, order]]
            if cand.size == 0:
                self.stats.frames[i] += 1
                self.stats.bytes[i] += self.bm.header + self.digest_bytes
                continue
            belmin = np.where(nbr[i][:, None], self.bel[i][:, cand], np.iinfo(np.int64).max).min(0)
            for e, bmin in zip(cand.tolist(), belmin.tolist()):
                c = self.costs.cost(e, int(v[i, e]), int(bmin))
                if used + c > avail:
                    self.stats.truncated += 1
                    break
                used += c
                incl[i, e] = True
            self.stats.frames[i] += 1
            self.stats.bytes[i] += self.bm.header + self.digest_bytes + used
            self.stats.delta_bytes[i] += used
        seg = L.segment_of()
        for s, sid in enumerate((0, 1, 2, 3)):
            self.stats.entries[s] += int(incl[:, seg == sid].sum())
        # optimistic: what I just sent is assumed received by my current neighbours
        framed = np.where(incl, v, self.bottom[None, :])
        for i in np.flatnonzero(incl.any(1)):
            js = np.flatnonzero(nbr[i])
            self.bel[i, js] = np.maximum(self.bel[i, js], framed[i][None, :])
        self._fly = {"k": k, "pos": np.array(pos, float), "incl": incl, "snap": v.copy(), "sent": alive.copy()}
        if self.beacon is not None:
            self.beacon.send(k, alive)


@dataclass
class Beacon:
    """Low-rate global channel (LoRa-class): period in time units, per-copy loss, per-frame byte budget."""

    period: float = 1.0
    loss: float = 0.2
    budget: int = 222  # LoRa maximum payload at SF7 (declared convention)
    key: CRNKey = DEFAULT_KEY
    byte_model: ByteModel = DEFAULT_BYTES

    def attach(self, kb: KnowledgeBase, stats: CommStats) -> None:
        self.kb, self.L, self.stats = kb, kb.layout, stats
        self.costs = _Costs(kb, self.byte_model)
        self.sent_v = np.tile(kb.layout.bottom(), (kb.layout.N, 1))  # values last beaconed per node
        self._fly = None
        self.every = max(1, round(self.period / TICK))

    def _order(self, i: int) -> np.ndarray:
        L, v = self.L, self.kb.v
        own_rec = np.array([L.REC.start + i])
        rk = v[i, L.ROUTE]
        own_rt = L.ROUTE.start + np.flatnonzero((rk >= 0) & (rk % L.N == i))
        oth_rec = np.array([e for e in range(L.REC.start, L.REC.stop) if e != L.REC.start + i])
        oth_rt = np.setdiff1d(np.arange(L.ROUTE.start, L.ROUTE.stop), own_rt)
        return np.concatenate([own_rec, own_rt, np.arange(L.TASK.start, L.TASK.stop),
                               np.arange(L.LOG.start, L.LOG.stop), oth_rec, oth_rt]).astype(np.int64)

    def send(self, k: int, alive: np.ndarray) -> None:
        if k % self.every:
            return
        v = self.kb.v
        incl = np.zeros_like(v, dtype=bool)
        for i in np.flatnonzero(alive):
            used = self.byte_model.header
            for e in self._order(i).tolist():
                if v[i, e] <= self.sent_v[i, e]:
                    continue
                c = self.costs.cost(e, int(v[i, e]), int(self.sent_v[i, e]))
                if used + c > self.budget:
                    break
                used += c
                incl[i, e] = True
            self.sent_v[i] = np.where(incl[i], v[i], self.sent_v[i])
            self.stats.beacon_frames += 1
            self.stats.beacon_bytes += used
        self._fly = {"k": k, "incl": incl, "snap": v.copy(), "sent": np.asarray(alive, bool).copy()}

    def deliver(self, k: int, alive: np.ndarray) -> None:
        fly, self._fly = self._fly, None
        if fly is None or fly["k"] != k - 1:
            return
        N = self.L.N
        ok = crn_rng(self.key, S_BEACON, fly["k"]).random((N, N)) >= self.loss
        ok &= fly["sent"][:, None] & np.asarray(alive, bool)[None, :]
        np.fill_diagonal(ok, False)
        vals = np.where(fly["incl"], fly["snap"], self.L.bottom()[None, :])
        v = self.kb.v
        for i in np.flatnonzero(ok.any(1)):
            recv = np.flatnonzero(ok[i])
            v[recv] = np.maximum(v[recv], vals[i])


def make_transport(kb: KnowledgeBase, ideal: bool, R: float = math.inf, loss: str = "ge", p: float = 0.2,
                   rho_ge: float = 0.8, key: CRNKey = DEFAULT_KEY, budget: float = 1500.0,
                   oracle: str | None = None, beacon_period: float | None = None, beacon_loss: float = 0.2,
                   relay: bool = True) -> Transport:
    """Factory used by protocol runs: ideal comms, or T2 with the named loss model and optional beacon."""
    beacon = None if beacon_period is None else Beacon(beacon_period, beacon_loss, key=key)
    if ideal:
        return IdealTransport(kb, oracle, beacon)
    lm = make_loss(loss, p=p, rho=rho_ge, R=R)
    return GossipTransport(kb, R, lm, key=key, budget=budget, relay=relay, oracle=oracle, beacon=beacon)


__all__ = [
    "Beacon",
    "Bernoulli",
    "ByteModel",
    "CRNKey",
    "CommStats",
    "GilbertElliott",
    "GossipTransport",
    "IdealTransport",
    "LossModel",
    "NoLoss",
    "Rayleigh",
    "Transport",
    "comm_range",
    "components",
    "connectivity_radius",
    "crn_rng",
    "in_range",
    "make_loss",
    "make_transport",
    "pairwise_dist",
]
