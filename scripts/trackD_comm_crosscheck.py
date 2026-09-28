"""Cross-check the Track D T2 channel (cbba_sota.dyn.comm) against cbja/network.py on 5 small cases (spec 3.5).

cbja/network.py is an event-driven radio (analytic link boundaries, per-packet loss, in-flight drop when a link
breaks) and needs simpy, so this script runs under the cbja venv:

    /home/jovyan/dev/cbja/.venv/bin/python scripts/trackD_comm_crosscheck.py [--json]

Mapping: 1 time unit = 1 s, TICK = 0.1 s = cbja baseline latency 100 ms, bandwidth 1e9 kbps (no serialization
delay), no ACKs, no retries. cbja is used as a small-case ORACLE only.

Level 1, channel (cases 1-3, 5): a probe packet is sent at every tick t_k; in cbja the hop "succeeds" iff the probe
arrives exactly at t_k + 0.1 (link up at send and through the flight, not lost). In T2 the hop succeeds iff
``in_range(pos(t_k), pos(t_k+1), R)`` and the CRN loss draw passes; the GossipTransport's delivered-hop matrix is
checked against the same prediction. Geometric cases must match tick for tick; loss cases are compared
statistically (different random streams by design).
Level 2, information latency (case 4): a fact written at one end of a 3-node chain reaches the other end. cbja
relays at the application level; T2 relays by digest gossip. On established links both take one tick per hop. On a
NEW contact T2 adds exactly one handshake tick (digests first, then deltas), which the script reports.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CBJA = Path("/home/jovyan/dev/cbja")
sys.path[:0] = [str(ROOT), str(CBJA)]

from cbba_sota.dyn.comm import (
    Bernoulli,
    CRNKey,
    GilbertElliott,
    GossipTransport,
    NoLoss,
    in_range,
    pairwise_dist,
)
from cbba_sota.dyn.replica import ABANDON, KnowledgeBase, Layout

TICK = 0.1


def lin(p0, v, t0=0.0, t_end=math.inf):
    """Straight motion from p0 at t0 with velocity v until t_end, then at rest."""
    p0, v = np.asarray(p0, float), np.asarray(v, float)

    def pos(t):
        tt = min(max(t, t0), t_end)
        return tuple(p0 + v * (tt - t0))

    def vel(t):
        return tuple(v) if t0 <= t < t_end else (0.0, 0.0)

    return pos, vel, (None if not np.any(v) else (t_end if math.isfinite(t_end) else 1e6))


# ------------------------------------------------------------------------------------------------ cbja side
def cbja_probe(traj: dict, R: float, t_end: float, pair=("a", "b"), loss=0.0, burst=None, seed=7):
    """Per-tick probe from pair[0] to pair[1]; returns (sent ticks, delivered-on-time bool array)."""
    from cbja.kernel import Kernel
    from cbja.network import Network

    k = Kernel(seed=seed, max_events=10 ** 7)
    zones = []
    if burst is not None:
        zones = [{"id": "z", "polygon_m": [[-10, -10], [10, -10], [10, 10], [-10, 10]], "applies_to": ["peer"],
                  "endpoint_rule": "either_inside", "burst": burst}]
    cfg = {"baseline_range_m": R, "baseline_latency_ms": 100.0, "baseline_bandwidth_kbps": 1e9,
           "baseline_loss_probability": loss, "ttl_s": 0.35, "retries": 0, "zones": zones}
    net = Network(k, cfg, position_fn=lambda n, at=None: traj[n][0](k.now if at is None else at),
                  velocity_fn=lambda n, at=None: traj[n][1](k.now if at is None else at),
                  motion_end_fn=lambda n: traj[n][2])
    got = {}
    for n in traj:
        net.register(n, (lambda msg, n=n: got.setdefault(msg["payload"]["tick"], msg["received_at"])
                         if n == pair[1] and "tick" in msg["payload"] else None))
    n_ticks = round(t_end / TICK)
    for kk in range(n_ticks):
        k.schedule(kk * TICK, (lambda kk=kk: net.send(pair[0], pair[1], {"tick": kk}, require_ack=False,
                                                        ttl_s=0.35)), priority=40)
    k.run(until=t_end + 1.0)
    ok = np.array([kk in got and abs(got[kk] - (kk * TICK + TICK)) < 1e-6 for kk in range(n_ticks)])
    return ok


def cbja_relay(traj: dict, R: float, t0: float, chain=("a", "b", "c"), t_end=5.0):
    """Application-level relay a -> b -> c of one message sent at t0; returns arrival time at c (or inf)."""
    from cbja.kernel import Kernel
    from cbja.network import Network

    k = Kernel(seed=3, max_events=10 ** 7)
    cfg = {"baseline_range_m": R, "baseline_latency_ms": 100.0, "baseline_bandwidth_kbps": 1e9,
           "baseline_loss_probability": 0.0, "ttl_s": 30.0, "retries": 0}
    net = Network(k, cfg, position_fn=lambda n, at=None: traj[n][0](k.now if at is None else at),
                  velocity_fn=lambda n, at=None: traj[n][1](k.now if at is None else at),
                  motion_end_fn=lambda n: traj[n][2])
    out = {}

    def recv(n):
        def f(msg):
            if msg["payload"].get("type") != "fact":
                return
            if n == chain[-1]:
                out.setdefault("t", msg["received_at"])
            else:
                nxt = chain[chain.index(n) + 1]
                net.send(n, nxt, msg["payload"], require_ack=False, ttl_s=30.0)
        return f

    for n in traj:
        net.register(n, recv(n))
    k.schedule(t0, lambda: net.send(chain[0], chain[1], {"type": "fact"}, require_ack=False, ttl_s=30.0),
               priority=40)
    k.run(until=t_end)
    return out.get("t", math.inf)


# ------------------------------------------------------------------------------------------------ T2 side
def positions(traj: dict, names, t):
    return np.array([traj[n][0](t) for n in names], float)


def t2_probe(traj: dict, R: float, t_end: float, loss, pair=(0, 1)):
    """Hop success per tick predicted by the channel, and the transport's delivered-hop matrix, for pair."""
    names = list(traj)
    n_ticks = round(t_end / TICK)
    key = CRNKey("crosscheck", 0, 1)
    lm = loss.bind(key, len(names))
    pred = np.array([in_range(positions(traj, names, kk * TICK), positions(traj, names, (kk + 1) * TICK), R)[pair]
                     and not lm.lost(kk, pairwise_dist(positions(traj, names, kk * TICK)))[pair]
                     for kk in range(n_ticks)])
    # the transport itself (last node is a dummy station far away)
    L = Layout(len(names), 1, 1)
    kb = KnowledgeBase(L)
    loss2 = type(loss)(**{k_: getattr(loss, k_) for k_ in ("p", "rho") if hasattr(loss, k_)})
    tr = GossipTransport(kb, R, loss2, key=key, budget=1e9)
    alive = np.ones(L.N, bool)
    seen = np.zeros(n_ticks, bool)
    far = np.array([[1e3, 1e3]])
    for kk in range(n_ticks + 1):
        pos = np.vstack([positions(traj, names, kk * TICK), far])
        tr.deliver(kk, pos, alive)
        if kk >= 1:
            seen[kk - 1] = tr.last_up[pair]
        tr.send(kk, pos, alive)
    return pred, seen


def t2_relay(traj: dict, R: float, t0: float, warm: bool):
    """Tick at which a fact written by node 0 at t0 reaches node 2 through gossip (3-node chain + far station)."""
    names = list(traj)
    L = Layout(len(names), 1, 1)
    kb = KnowledgeBase(L)
    tr = GossipTransport(kb, R, NoLoss(), key=CRNKey("crosscheck", 0, 2), budget=1e9)
    alive = np.ones(L.N, bool)
    far = np.array([[1e3, 1e3]])
    k0 = round(t0 / TICK)
    if not warm:  # nobody has heard anybody before t0: first contact at t0
        tr.heard[:] = -(10 ** 9)
    for kk in range(200):
        t = kk * TICK
        pos = np.vstack([positions(traj, names, t), far])
        if not warm and kk < k0:
            pos[:len(names)] = 1e3 * np.arange(1, len(names) + 1)[:, None]  # out of range until t0
        tr.deliver(kk, pos, alive)
        if kk == k0:
            kb.append_log(0, ABANDON, 0, 0, t)
        if kk > k0 and kb.v[2, L.LOG.start + 0] >= 1:
            return kk * TICK
        tr.send(kk, pos, alive)
    return math.inf


def runs(x: np.ndarray) -> float:
    """Mean length of runs of True (loss bursts)."""
    if not x.any():
        return 0.0
    d = np.diff(np.concatenate([[0], x.astype(int), [0]]))
    starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    return float(np.mean(ends - starts))


# ------------------------------------------------------------------------------------------------ cases
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    R = 0.2
    res = []

    # 1. static pair in range
    tr1 = {"a": lin((0.3, 0.5), (0, 0)), "b": lin((0.45, 0.5), (0, 0))}
    c = cbja_probe(tr1, R, 3.0)
    p, s = t2_probe(tr1, R, 3.0, NoLoss())
    res.append({"case": "1 static in range", "cbja_ok": int(c.sum()), "t2_ok": int(p.sum()), "ticks": len(c),
                "match": bool((c == p).all() and (p == s).all())})

    # 2. approach: b moves toward a and enters range at t = 1.0 (on a tick) and at t = 1.05 (mid-tick)
    for t_in in (1.0, 1.05):
        v = 0.2
        x0 = 0.3 + R + v * t_in
        tr2 = {"a": lin((0.3, 0.5), (0, 0)), "b": lin((x0, 0.5), (-v, 0), 0.0, t_in + 1.0)}
        c = cbja_probe(tr2, R, 3.0)
        p, s = t2_probe(tr2, R, 3.0, NoLoss())
        first_c = int(np.argmax(c)) if c.any() else -1
        first_p = int(np.argmax(p)) if p.any() else -1
        res.append({"case": f"2 approach, enters range at t={t_in}", "cbja_first_ok_tick": first_c,
                    "t2_first_ok_tick": first_p, "match": bool((c == p).all() and (p == s).all())})

    # 3. separation: b leaves range at t = 2.05 (mid-tick): cbja drops the in-flight probe of t = 2.0
    v = 0.2
    x0 = 0.3 + R - v * 2.05
    tr3 = {"a": lin((0.3, 0.5), (0, 0)), "b": lin((x0, 0.5), (v, 0))}
    c = cbja_probe(tr3, R, 3.0)
    p, s = t2_probe(tr3, R, 3.0, NoLoss())
    res.append({"case": "3 pass-by, enters at t=0.05, leaves at t=2.05", "cbja_last_ok_tick": int(np.flatnonzero(c).max()),
                "t2_last_ok_tick": int(np.flatnonzero(p).max()), "match": bool((c == p).all() and (p == s).all())})

    # 4. relay chain a - b - c (a and c out of range), fact written at t0 = 1.0
    tr4 = {"a": lin((0.2, 0.5), (0, 0)), "b": lin((0.35, 0.5), (0, 0)), "c": lin((0.5, 0.5), (0, 0))}
    t_c = cbja_relay(tr4, R, 1.0)
    t_w = t2_relay(tr4, R, 1.0, warm=True)
    t_n = t2_relay(tr4, R, 1.0, warm=False)
    res.append({"case": "4 two-hop relay", "cbja_arrival": round(t_c, 6), "t2_arrival_established": round(t_w, 6),
                "t2_arrival_new_contact": round(t_n, 6),
                "match": bool(abs(t_c - t_w) < 1e-6),  # cbja adds ~1e-9 s serialization per hop
                "note": f"new contact: T2 - cbja = {t_n - t_c:.3f} (one digest-handshake tick at first contact)"})

    # 5. loss statistics on a static pair: Bernoulli p = 0.3 and Gilbert-Elliott (p = 0.2, rho = 0.8)
    T5 = 1000.0
    c = cbja_probe(tr1, R, T5, loss=0.3)
    p, s = t2_probe(tr1, R, T5, Bernoulli(0.3))
    n = len(c)
    se = math.sqrt(0.3 * 0.7 / n)
    ok_b = abs((1 - c.mean()) - 0.3) < 4 * se and abs((1 - p.mean()) - 0.3) < 4 * se and (p == s).all()
    ge = GilbertElliott(0.2, 0.8)
    burst = {"type": "continuous_time_two_state", "initial_state": "stationary_sample",
             "good_mean_duration_s": TICK / (1 - ge.pGG), "bad_mean_duration_s": TICK / (1 - ge.pBB),
             "loss_probability_good": 0.0, "loss_probability_bad": 0.999}  # < 1: a bad link stays "up" in cbja
    # (loss 1 would mark it down, queue the packet and drop in-flight ones, a different semantics)
    cg = cbja_probe(tr1, R, T5, burst=burst)
    pg, sg = t2_probe(tr1, R, T5, GilbertElliott(0.2, 0.8))
    lc, lp = 1 - cg.mean(), 1 - pg.mean()
    rc, rp = runs(~cg), runs(~pg)
    ok_g = abs(lc - 0.2) < 0.05 and abs(lp - 0.2) < 0.05 and abs(rc - rp) / rp < 0.35 and (pg == sg).all()
    res.append({"case": "5 loss statistics", "bernoulli_cbja_loss": round(1 - c.mean(), 4),
                "bernoulli_t2_loss": round(1 - p.mean(), 4), "ge_cbja_loss": round(lc, 4), "ge_t2_loss": round(lp, 4),
                "ge_cbja_mean_burst_ticks": round(rc, 3), "ge_t2_mean_burst_ticks": round(rp, 3),
                "ge_theory_mean_burst_ticks": round(1 / (1 - ge.pBB), 3), "n_probes": n,
                "match": bool(ok_b and ok_g)})

    ok = all(r["match"] for r in res)
    if args.json:
        print(json.dumps({"ok": ok, "cases": res}))
    else:
        for r in res:
            print(json.dumps(r))
        print("ALL MATCH" if ok else "MISMATCH")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
