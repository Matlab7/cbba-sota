"""Track D comm-first pilot P1: does it matter WHERE the coalition planner runs when links are range-limited?

A plan-level discrete-time simulator (tick = the HeteroMRTA env's dt = 0.1) of HeteroMRTA instances with
- online task release at a dispatch station at the arena centre (Poisson, degree of dynamism dod, horizon H as in
  pilots/trackD/dyn_env.py; tasks with release 0 are briefed to everyone before the mission),
- mean-preserving lognormal task-duration noise (sigma), common random numbers across arms,
- a unit-disk radio graph over robots + station with range R, instant multi-hop flooding inside a connected
  component every tick, optional i.i.d. per-link per-tick Bernoulli loss (CRN: per-tick uniforms keyed by seed),
- arrival-order coalitions: a task starts when the robots present cover its requirement (in arrival order); later
  arrivals find it started (wasted trip); waiting robots hold a lease (predicted start + margin) and then abandon.

Every node (robot, station) keeps a replicated belief that merges as a state-based CRDT (task knowledge = grow-only
set; task start/finish = min; robot physical state = last-writer-wins on the robot's own event counter; robot route
= last-writer-wins on the planner's version counter). All arms use the SAME planner (from-state list scheduling with
committed members, frozen queues and 8 noisy restarts) and the same leases; they differ only in who plans for whom:

  O   (proposed family) every connected component plans for its own members on its merged belief; robots outside
      the component keep their last known route (frozen), and in-component robots keep every task they share with an
      out-of-component robot (cross-component coalition commitments are frozen). The station is a relay node.
  C   central: only the station's component plans (plannable = robots currently reachable from the station);
      unreachable robots follow their last route, then a local fallback (lf): idle | greedy | return.
  Rn  naive replicated solver: every component plans for ALL robots on its belief (ignores partitions) and its
      members follow their own routes from that plan.
  Oh  like O, but only the current commitment (head) of an out-of-component robot is kept; the rest of its route
      is not frozen and it gets no new tasks (pessimistic about robots out of reach, no stale-route freezing).
  O at R = inf is the global-information rolling re-plan reference (G).

Makespan = max over robots of (last activity end + travel from there to the depot), the HeteroMRTA evaluator's
return convention. Failure = not all tasks finished by t = 200.
"""
from __future__ import annotations

import zlib
from pathlib import Path

import numpy as np
from numba import njit

DT = 0.1
MAX_T = 200.0
STATION = np.array([0.5, 0.5])
MARGIN = 3.0  # lease margin (time units) after the predicted start
RESTARTS = 8
HORIZON = {"MA-AT-25-5-50": 25.0, "MA-AT-50-5-50": 15.0, "SA-AT-50-5-50": 20.0, "SA-BT-50-5-50": 12.0}


# ------------------------------------------------------------------------------------------------ planner kernel
@njit(cache=True)
def _dispatch_state(now, req, ab, dur, tt, da, open_, plannable, avail0, trav0, ret0, head0, heta0, queue, qlen,
                    noise, seed, order, mem, cnt, start):
    """List scheduling from the current state. Returns (estimated makespan incl. 1000 per unscheduled open task,
    number scheduled). ``order[:n]`` are the scheduled tasks in key order, ``mem/cnt`` their coalitions."""
    np.random.seed(seed)
    T, K = req.shape
    A = ab.shape[0]
    avail = avail0.copy()
    pos = np.full(A, -1, np.int64)
    head = head0.copy()
    qptr = np.zeros(A, np.int64)
    cl = np.full((T, A), -1, np.int64)
    ncl = np.zeros(T, np.int64)
    for r in range(A):
        if head[r] >= 0:
            j = head[r]
            cl[j, ncl[j]] = r
            ncl[j] += 1
        for q in range(qlen[r]):
            j = queue[r, q]
            cl[j, ncl[j]] = r
            ncl[j] += 1
    scheduled = np.zeros(T, np.bool_)
    n_open = 0
    for j in range(T):
        if open_[j]:
            n_open += 1
    cnt[:] = 0
    start[:] = np.nan
    need = np.empty(K)
    arr = np.empty(A)
    cand = np.empty(A, np.int64)
    best_mem = np.empty(A, np.int64)
    n_sched = 0
    while n_sched < n_open:
        best_j, best_score, best_s, best_c = -1, np.inf, 0.0, 0
        for j in range(T):
            if not open_[j] or scheduled[j]:
                continue
            ok = True
            s = now
            for k in range(K):
                need[k] = req[j, k]
            c = 0
            for p in range(ncl[j]):
                r = cl[j, p]
                if head[r] >= 0:
                    nx = head[r]
                elif qptr[r] < qlen[r]:
                    nx = queue[r, qptr[r]]
                else:
                    nx = -1
                if nx != j:
                    ok = False
                    break
                if head[r] == j and heta0[r] >= 0:
                    a = heta0[r]
                else:
                    a = avail[r] + (trav0[r, j] if pos[r] < 0 else tt[pos[r], j])
                s = max(s, a)
                for k in range(K):
                    need[k] -= ab[r, k]
                cand[c] = r
                c += 1
            if not ok:
                continue
            n_forced = c
            short = False
            for k in range(K):
                if need[k] > 0:
                    short = True
            if short:
                # earliest-arrival cover from free plannable robots
                nf = 0
                for r in range(A):
                    if plannable[r] and head[r] < 0 and qptr[r] >= qlen[r]:
                        arr[r] = avail[r] + (trav0[r, j] if pos[r] < 0 else tt[pos[r], j])
                    else:
                        arr[r] = np.inf
                idx = np.argsort(arr, kind="mergesort")
                for q in range(A):
                    r = idx[q]
                    if not np.isfinite(arr[r]):
                        break
                    gain = False
                    for k in range(K):
                        if need[k] > 0 and ab[r, k] > 0:
                            gain = True
                    if gain:
                        cand[c] = r
                        c += 1
                        nf += 1
                        for k in range(K):
                            need[k] -= ab[r, k]
                        done = True
                        for k in range(K):
                            if need[k] > 0:
                                done = False
                        if done:
                            break
                cover_ok = True
                for k in range(K):
                    if need[k] > 0:
                        cover_ok = False
                if not cover_ok:
                    continue
                # prune redundant free members, latest first
                for p in range(c - 1, n_forced - 1, -1):
                    r = cand[p]
                    red = True
                    for k in range(K):
                        if need[k] + ab[r, k] > 0:
                            red = False
                            break
                    if red:
                        for k in range(K):
                            need[k] += ab[r, k]
                        cand[p] = -1
                q = n_forced
                for p in range(n_forced, c):
                    if cand[p] >= 0:
                        cand[q] = cand[p]
                        q += 1
                c = q
                for p in range(n_forced, c):
                    r = cand[p]
                    s = max(s, avail[r] + (trav0[r, j] if pos[r] < 0 else tt[pos[r], j]))
            score = s * (1.0 + noise * np.random.random())
            if score < best_score:
                best_j, best_score, best_s, best_c = j, score, s, c
                best_mem[:c] = cand[:c]
        if best_j < 0:
            break
        j = best_j
        order[n_sched] = j
        n_sched += 1
        scheduled[j] = True
        start[j] = best_s
        f = best_s + dur[j]
        cnt[j] = best_c
        for p in range(best_c):
            r = best_mem[p]
            mem[j, p] = r
            if head[r] == j:
                head[r] = -1
            elif head[r] < 0 and qptr[r] < qlen[r] and queue[r, qptr[r]] == j:
                qptr[r] += 1
            avail[r] = f
            pos[r] = j
    ms = 0.0
    for r in range(A):
        ret = avail[r] + (da[r, pos[r]] if pos[r] >= 0 else ret0[r])
        ms = max(ms, ret)
    return ms + 1000.0 * (n_open - n_sched), n_sched


def plan(now, inst, open_, plannable, avail, trav0, ret0, head, heta, queue, qlen, seed=0, restarts=RESTARTS):
    T, A = inst.n_tasks, inst.n_agents
    W = A
    best = None
    for k in range(restarts):
        order, cnt, start = np.empty(T, np.int64), np.zeros(T, np.int64), np.empty(T)
        mem = np.full((T, W), -1, np.int64)
        ms, n = _dispatch_state(float(now), inst.req, inst.ab, inst.dur, inst.tt, inst.da, open_, plannable, avail,
                                trav0, ret0, head, heta, queue, qlen, 0.0 if k == 0 else 0.3,
                                int(seed * 1000 + k), order, mem, cnt, start)
        if best is None or ms < best[0] - 1e-9:
            best = (ms, n, order[:n].copy(), mem, cnt, start)
    return best


# ------------------------------------------------------------------------------------------------ scenario
def _rng(*keys) -> np.random.Generator:
    return np.random.default_rng([int(k) for k in keys])


def release_times(n_tasks, inst_key, seed, dod, horizon):
    if dod <= 0:
        return np.zeros(n_tasks)
    rng = _rng(seed, inst_key, 3)
    n_dyn = int(round(dod * n_tasks))
    rel = np.zeros(n_tasks)
    dyn = rng.choice(n_tasks, n_dyn, replace=False)
    rel[dyn] = np.sort(rng.uniform(0.0, horizon, n_dyn))
    return rel


def realized_durations(dur, inst_key, seed, sigma):
    if sigma <= 0:
        return dur.copy()
    z = _rng(seed, inst_key, 2).standard_normal(len(dur))
    return dur * np.exp(sigma * z - 0.5 * sigma ** 2)


# ------------------------------------------------------------------------------------------------ simulator
IDLE, TRAVEL, WAIT, WORK, MOVE = 0, 1, 2, 3, 4


class Sim:
    def __init__(self, inst, rel, dreal, R, arm, lf="idle", p_loss=0.0, seed=0, margin=MARGIN):
        self.inst = inst
        T, A = inst.n_tasks, inst.n_agents
        self.T, self.A, self.N = T, A, A + 1
        self.loc, self.req, self.ab = inst.loc, inst.req, inst.ab
        self.dnom, self.dreal, self.speed = inst.dur, dreal, inst.speed
        self.depot, self.rel = inst.depot, rel
        self.R, self.arm, self.lf, self.p_loss, self.seed, self.margin = R, arm, lf, p_loss, seed, margin
        self.clock = 0
        # ground truth
        self.rs = np.zeros(A, np.int64)
        self.rtask = np.full(A, -1)
        self.p0 = inst.depot.copy()
        self.p1 = inst.depot.copy()
        self.tdep = np.zeros(A)
        self.tarr = np.zeros(A)
        self.idle_since = np.zeros(A)
        self.last_end = np.zeros(A)
        self.last_loc = inst.depot.copy()
        self.used = np.zeros(A, bool)
        self.tst = np.where(rel <= 0, 1, 0)
        self.tstart = np.full(T, np.nan)
        self.tfin = np.full(T, np.nan)
        self.present = [dict() for _ in range(T)]
        self.members = [[] for _ in range(T)]
        # beliefs [node, ...]; node A = station
        N = self.N
        self.Bk = np.zeros((N, T), bool)
        self.Bk[:, rel <= 0] = True
        self.Bts = np.full((N, T), np.nan)
        self.Btf = np.full((N, T), np.nan)
        self.Bs = np.zeros((N, A), np.int64)
        self.Bt = np.full((N, A), -1, np.int64)
        self.Be = np.zeros((N, A))
        self.Bp = np.repeat(inst.depot[None], N, axis=0)
        self.Bst = np.zeros((N, A), np.int64)
        self.Br = np.full((N, A, T), -1, np.int64)
        self.Brn = np.zeros((N, A), np.int64)
        self.Brst = np.zeros((N, A), np.int64)
        self.Bpred = np.full((N, T), np.nan)
        self.Bpst = np.zeros((N, T), np.int64)
        self.cache = {}
        self.stats = dict(replans=0, wasted=0, abandons=0, lf_moves=0, lf_greedy=0, extras=0, msgs=0,
                          comp_ticks=0, st_conn=0.0)

    # --- helpers -----------------------------------------------------------------------------------------------
    def tick_id(self):
        self.clock += 1
        return self.clock

    def set_state(self, a, s, task, e, p):
        """Robot a's own belief row records its new physical state (authoritative, LWW on the event counter)."""
        self.Bs[a, a], self.Bt[a, a], self.Be[a, a] = s, task, e
        self.Bp[a, a] = p
        self.Bst[a, a] = self.tick_id()

    def positions(self, t):
        P = self.p1.copy()
        mv = (self.rs == TRAVEL) | (self.rs == MOVE)
        if mv.any():
            span = np.maximum(self.tarr[mv] - self.tdep[mv], 1e-12)
            frac = np.clip((t - self.tdep[mv]) / span, 0.0, 1.0)[:, None]
            P[mv] = self.p0[mv] + (self.p1[mv] - self.p0[mv]) * frac
        return P

    def components(self, t, k):
        A = self.A
        if not np.isfinite(self.R):
            return np.zeros(A + 1, np.int64)
        from scipy.sparse.csgraph import connected_components
        P = np.vstack([self.positions(t), STATION])
        D = np.linalg.norm(P[:, None] - P[None], axis=-1)
        adj = D <= self.R
        if self.p_loss > 0:
            u = _rng(self.seed, 77, k).random((A + 1, A + 1))
            u = np.triu(u, 1)
            u = u + u.T
            adj &= u >= self.p_loss
        return connected_components(adj, directed=False)[1]

    def merge(self, idx):
        if len(idx) <= 1:
            return
        A, T = self.A, self.T
        ar = np.arange(A)
        k = self.Bk[idx].any(0)
        ts = np.fmin.reduce(self.Bts[idx], axis=0)
        tf = np.fmin.reduce(self.Btf[idx], axis=0)
        w = idx[self.Bst[idx].argmax(0)]
        s, tk, e, p, st = self.Bs[w, ar], self.Bt[w, ar], self.Be[w, ar], self.Bp[w, ar], self.Bst[w, ar]
        wr = idx[self.Brst[idx].argmax(0)]
        r, rn, rst = self.Br[wr, ar], self.Brn[wr, ar], self.Brst[wr, ar]
        wp = idx[self.Bpst[idx].argmax(0)]
        at = np.arange(T)
        pr, pst = self.Bpred[wp, at], self.Bpst[wp, at]
        self.Bk[idx], self.Bts[idx], self.Btf[idx] = k, ts, tf
        self.Bs[idx], self.Bt[idx], self.Be[idx], self.Bp[idx], self.Bst[idx] = s, tk, e, p, st
        self.Br[idx], self.Brn[idx], self.Brst[idx] = r, rn, rst
        self.Bpred[idx], self.Bpst[idx] = pr, pst

    # --- planning ----------------------------------------------------------------------------------------------
    def plan_component(self, row, members, plannable, now, write_rows):
        """Plan on belief row ``row`` for the robots in ``plannable`` (bool [A]); write routes of plannable robots
        into ``write_rows``."""
        inst, A, T = self.inst, self.A, self.T
        known, ts, tf = self.Bk[row], self.Bts[row], self.Btf[row]
        open_ = known & np.isnan(ts) & np.isnan(tf)
        s_, tk, e_, p_ = self.Bs[row], self.Bt[row], self.Be[row], self.Bp[row]
        head = np.full(A, -1, np.int64)
        heta = np.full(A, -1.0)
        avail = np.full(A, float(now))
        anchor = p_.copy()
        for r in range(A):
            j = tk[r]
            if s_[r] in (TRAVEL, WAIT) and j >= 0:
                anchor[r] = self.loc[j]
                if open_[j]:
                    head[r], heta[r] = j, max(e_[r], 0.0)
                else:
                    avail[r] = max(e_[r], now)
            elif s_[r] == WORK and j >= 0:
                anchor[r] = self.loc[j]
                if np.isnan(tf[j]):
                    st = ts[j] if not np.isnan(ts[j]) else e_[r]
                    avail[r] = max(st + self.dnom[j], now)
                else:
                    avail[r] = max(tf[j], now)
        trav0 = np.linalg.norm(anchor[:, None] - self.loc[None], axis=-1) / self.speed
        ret0 = np.linalg.norm(anchor - self.depot, axis=-1) / self.speed
        queue = np.full((A, T), -1, np.int64)
        qlen = np.zeros(A, np.int64)
        routes = [self.Br[row, r, :self.Brn[row, r]] for r in range(A)]
        frozen = ~plannable
        if self.arm in ("O", "C"):
            X = np.zeros(T, bool)
            for r in np.flatnonzero(frozen):
                for j in routes[r]:
                    if open_[j] and j != head[r]:
                        queue[r, qlen[r]] = j
                        qlen[r] += 1
                        X[j] = True
                if head[r] >= 0:
                    X[head[r]] = True
            for r in np.flatnonzero(plannable):
                for j in routes[r]:
                    if X[j] and open_[j] and j != head[r]:
                        queue[r, qlen[r]] = j
                        qlen[r] += 1
            pl = plannable
        elif self.arm == "Oh":
            pl = plannable  # out-of-component robots: only their current commitment (head) is kept
        else:
            pl = np.ones(A, bool)  # Rn: plans everyone it knows of
        key = (self.arm, members.tobytes(), plannable.tobytes(),
               zlib.crc32(b"".join([open_.astype(np.int8).tobytes(), self.Bst[row].tobytes(),
                                 (self.Brst[row] * frozen).tobytes(),
                                 np.round(np.nan_to_num(ts, nan=-1), 6).tobytes()])))
        if self.cache.get(members.tobytes()) == key:
            return
        self.cache[members.tobytes()] = key
        ms, n, order, mem, cnt, start = plan(now, inst, open_, pl, avail, trav0, ret0, head, heta, queue, qlen,
                                             seed=self.seed)
        self.stats["replans"] += 1
        newr = [[] for _ in range(A)]
        for j in order:
            for q in range(cnt[j]):
                newr[mem[j, q]].append(j)
        ver = self.tick_id()
        for r in np.flatnonzero(plannable):
            rr = newr[r]
            self.Br[write_rows, r, :] = -1
            self.Br[write_rows, r, :len(rr)] = rr
            self.Brn[write_rows, r] = len(rr)
            self.Brst[write_rows, r] = ver
        sch = order
        if len(sch):
            for w in write_rows:
                self.Bpred[w, sch] = start[sch]
                self.Bpst[w, sch] = ver

    # --- ground truth events -----------------------------------------------------------------------------------
    def go_idle(self, a, t, p, activity=True):
        self.rs[a], self.rtask[a] = IDLE, -1
        self.p0[a] = p
        self.p1[a] = p
        self.idle_since[a] = t
        if activity:
            self.last_end[a] = t
            self.last_loc[a] = p
        self.set_state(a, IDLE, -1, t, p)

    def depart(self, a, j, t_dep, target=None):
        p = self.p1[a].copy()
        dest = self.loc[j] if j >= 0 else target
        self.p0[a] = p
        self.p1[a] = dest
        self.tdep[a] = t_dep
        self.tarr[a] = t_dep + float(np.linalg.norm(dest - p)) / self.speed
        if j >= 0:
            self.rs[a], self.rtask[a] = TRAVEL, j
            self.used[a] = True
            self.set_state(a, TRAVEL, j, self.tarr[a], dest)
        else:
            self.rs[a], self.rtask[a] = MOVE, -1
            self.set_state(a, MOVE, -1, self.tarr[a], dest)

    def step_physics(self, t):
        # arrivals
        for a in np.flatnonzero((self.rs == TRAVEL) & (self.tarr <= t + 1e-12)):
            j = self.rtask[a]
            if self.tst[j] == 1:
                self.present[j][a] = self.tarr[a]
                self.rs[a] = WAIT
                self.set_state(a, WAIT, j, self.tarr[a], self.loc[j])
            else:
                self.stats["wasted"] += 1
                self._observe_task(a, j)
                self.go_idle(a, self.tarr[a], self.loc[j])
        for a in np.flatnonzero((self.rs == MOVE) & (self.tarr <= t + 1e-12)):
            self.go_idle(a, self.tarr[a], self.p1[a], activity=False)
        # starts (arrival order)
        for j in np.flatnonzero(self.tst == 1):
            pr = self.present[j]
            if not pr:
                continue
            need = self.req[j].copy()
            got = []
            s = None
            for a, ta in sorted(pr.items(), key=lambda x: (x[1], x[0])):
                got.append(a)
                need -= self.ab[a]
                if np.all(need <= 0):
                    s = ta
                    break
            if s is None:
                continue
            self.tst[j] = 2
            self.tstart[j] = s
            self.tfin[j] = s + self.dreal[j]
            self.members[j] = got
            for a in got:
                self.rs[a], self.rtask[a] = WORK, j
                self.Bts[a, j] = s
                self.set_state(a, WORK, j, s, self.loc[j])
            for a in list(pr):
                if a not in got:  # arrived after the start (same tick): wasted
                    self.stats["extras"] += 1
                    self.Bts[a, j] = s
                    self.go_idle(a, max(pr[a], s), self.loc[j])
            self.present[j] = {}
        # finishes
        for j in np.flatnonzero((self.tst == 2) & (self.tfin <= t + 1e-12)):
            self.tst[j] = 3
            for a in self.members[j]:
                self.Btf[a, j] = self.tfin[j]
                self.go_idle(a, self.tfin[j], self.loc[j])
        # leases
        for a in np.flatnonzero(self.rs == WAIT):
            j = self.rtask[a]
            pred = self.Bpred[a, j]
            base = max(self.present[j].get(a, t), pred if np.isfinite(pred) else -np.inf)
            if t > base + self.margin:
                self.stats["abandons"] += 1
                del self.present[j][a]
                self.drop_from_route(a, j)
                self.go_idle(a, t, self.loc[j])

    def _observe_task(self, a, j):
        if not np.isnan(self.tstart[j]):
            self.Bts[a, j] = self.tstart[j]
        if not np.isnan(self.tfin[j]) and self.tst[j] == 3:
            self.Btf[a, j] = self.tfin[j]

    def drop_from_route(self, a, j):
        r = [x for x in self.Br[a, a, :self.Brn[a, a]] if x != j]
        self.Br[a, a, :] = -1
        self.Br[a, a, :len(r)] = r
        self.Brn[a, a] = len(r)
        self.Brst[a, a] = self.tick_id()

    # --- decisions ---------------------------------------------------------------------------------------------
    def next_route_task(self, a):
        for j in self.Br[a, a, :self.Brn[a, a]]:
            if self.Bk[a, j] and np.isnan(self.Bts[a, j]) and np.isnan(self.Btf[a, j]):
                return int(j)
        return -1

    def decide(self, t, lab):
        st_comp = lab[self.A]
        movers = np.flatnonzero(self.rs == MOVE)
        if movers.size:
            P = self.positions(t)
            for a in movers:
                if self.next_route_task(a) >= 0:  # got a route while heading for the station: stop here
                    self.go_idle(a, t, P[a].copy(), activity=False)
        for a in np.flatnonzero(self.rs == IDLE):
            j = self.next_route_task(a)
            if j >= 0:
                # a robot that became idle during the last tick leaves at that moment (<= DT of look-ahead, the
                # same for every arm); otherwise it leaves now
                dep = self.idle_since[a] if self.idle_since[a] > t - DT else t
                self.depart(a, j, dep)
                continue
            if lab[a] == st_comp or self.arm == "G":
                continue
            if self.lf == "return":
                d = float(np.linalg.norm(self.p1[a] - STATION))
                if d > 1e-9:
                    self.stats["lf_moves"] += 1
                    self.depart(a, -1, t, target=STATION.copy())
            elif self.lf == "greedy":
                j = self.greedy_pick(a)
                if j >= 0:
                    self.stats["lf_greedy"] += 1
                    self.Br[a, a, :] = -1
                    self.Br[a, a, 0] = j
                    self.Brn[a, a] = 1
                    self.Brst[a, a] = self.tick_id()
                    self.depart(a, j, t)

    def greedy_pick(self, a):
        """Nearest known open task that ``a`` contributes to and that its belief does not show as covered."""
        open_ = self.Bk[a] & np.isnan(self.Bts[a]) & np.isnan(self.Btf[a])
        best, bd = -1, np.inf
        for j in np.flatnonzero(open_):
            need = self.req[j].copy()
            for r in range(self.A):
                if r == a:
                    continue
                if (self.Bs[a, r] in (TRAVEL, WAIT) and self.Bt[a, r] == j) or j in self.Br[a, r, :self.Brn[a, r]]:
                    need = need - self.ab[r]
            if not np.any((need > 0) & (self.ab[a] > 0)):
                continue
            d = float(np.linalg.norm(self.loc[j] - self.p1[a]))
            if d < bd:
                best, bd = j, d
        return best

    def stop_movers(self, t, lab):
        st_comp = lab[self.A]
        idx = np.flatnonzero((self.rs == MOVE) & (lab[: self.A] == st_comp))
        if idx.size:
            P = self.positions(t)
            for a in idx:
                self.go_idle(a, t, P[a].copy(), activity=False)

    # --- main loop ---------------------------------------------------------------------------------------------
    def run(self):
        A, T = self.A, self.T
        all_rows = np.arange(self.N)
        # pre-mission briefing: everyone connected, global plan over the tasks known at 0
        self.plan_component(A, np.ones(A + 1, bool), np.ones(A, bool), 0.0, all_rows)
        self.cache.clear()
        k = 0
        t = 0.0
        st_conn = []
        while True:
            t = k * DT
            if t > MAX_T:
                break
            newly = np.flatnonzero((self.tst == 0) & (self.rel <= t + 1e-12))
            if newly.size:
                self.tst[newly] = 1
                self.Bk[A, newly] = True
            self.step_physics(t)
            if np.all(self.tst == 3):
                break
            lab = self.components(t, k)
            st_conn.append(float(np.mean(lab[:A] == lab[A])))
            self.stop_movers(t, lab)
            comps = {}
            for n, c in enumerate(lab):
                comps.setdefault(int(c), []).append(n)
            for c, nodes in comps.items():
                idx = np.array(nodes)
                self.merge(idx)
                members = np.zeros(A + 1, bool)
                members[idx] = True
                robots = members[:A]
                has_st = members[A]
                row = idx[0]
                if self.arm in ("O", "Oh"):
                    if robots.any():
                        self.plan_component(row, members, robots.copy(), t, idx)
                elif self.arm == "C":
                    if has_st:
                        self.plan_component(row, members, robots.copy(), t, idx)
                elif self.arm == "Rn":
                    if robots.any():
                        self.plan_component(row, members, robots.copy(), t, idx)
            self.decide(t, lab)
            k += 1
        done = bool(np.all(self.tst == 3))
        ret = np.where(self.used, self.last_end + np.linalg.norm(self.last_loc - self.depot, axis=1) / self.speed,
                       0.0)
        ms = float(ret.max()) if done else MAX_T
        self.stats["st_conn"] = float(np.mean(st_conn)) if st_conn else 1.0
        return dict(makespan=ms, success=done, t_end=t, **self.stats)


def run_episode(inst, inst_key, R, arm, lf, dod, horizon, sigma, seed, p_loss=0.0, margin=MARGIN):
    rel = release_times(inst.n_tasks, inst_key, seed, dod, horizon)
    dreal = realized_durations(inst.dur, inst_key, seed, sigma)
    sim = Sim(inst, rel, dreal, R, arm, lf, p_loss, seed, margin)
    return sim.run()
