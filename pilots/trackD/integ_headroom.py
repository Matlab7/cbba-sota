"""Track D integrated-stressor headroom pilot (proposer, 2026-09-28).

Question: under online release + execution noise + range-limited lossy links, how much makespan does a comm-limited
CENTRAL re-planner lose against FULL-information re-planning with the same planner, and how much of that gap does
COMPONENT-LOCAL repair (every connected component re-plans its own robots, everyone else frozen) recover?
If the central loss is small at realistic ranges, no decentralized mechanism can beat central by a clear margin.

Simplified but exact-in-time world (plan semantics of cbba_sota.hetero, HeteroMRTA travel/duration model):
- physics in continuous time (arrivals, starts, finishes), arrival-order coalitions (a task starts when the robots
  present cover its requirement; late or redundant arrivals skip it), LoRR-2026 travel-delay calendars and
  mean-preserving lognormal durations from dyn_env (common random numbers across methods);
- knowledge on a 0.1 tick: unit-disk graph over robots + a station at (0.5, 0.5) with i.i.d. per-link per-tick
  Bernoulli loss; instant multi-hop flooding inside a component (state-based merge: max version per record);
- tasks with release > 0 become known to the station at release (dispatch centre) and spread by gossip;
- t = 0 is a full-comm briefing: every method starts from the same plan of the tasks released at 0.

Planner (same for every method): state-aware list scheduling (earliest cover start) with frozen partial
coalitions; best of 8 (1 deterministic + 7 noisy). Scope agents keep only their committed task; frozen agents keep
their believed remaining route (precedence respected), and their traits are subtracted from each task's need.

Methods
  FULL   global instant comm, station re-plans all robots (reference for "re-planning with perfect information")
  CEN-F  station re-plans only robots in its component; others frozen (central + local fallback)
  CEN-R  station re-plans ALL robots on its (stale) beliefs; routes reach robots by gossip (central + DTN relay)
  LOC-F  every component re-plans its own robots; others frozen (component-scoped repair = proposed core)
  REP    every component re-plans ALL robots on its beliefs; newest version wins (replicated solver + gossip)
  OPEN   the t=0 plan; the station inserts released tasks only (no re-plan on execution events), CEN-F scope
Lease: a waiting robot abandons its task if no partner it knows of is still coming (after 1.0) or after 10.0.

Usage: .venv/bin/python pilots/trackD/integ_headroom.py --n 10 --procs 48
"""
from __future__ import annotations

import argparse
import heapq
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np
from numba import njit
from scipy.sparse.csgraph import connected_components

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
TICK = 0.1
STATION = np.array([0.5, 0.5])
HORIZON = {"MA-AT-25-5-50": 25.0, "MA-AT-50-5-50": 15.0, "SA-AT-50-5-50": 20.0, "SA-BT-50-5-50": 12.0}
MAX_T = 200.0
TRAVEL, WAIT, WORK, HOMING, IDLE, RALLY = 0, 1, 2, 3, 4, 5


# ------------------------------------------------------------------------------------------------ planner kernel
@njit(cache=True)
def _cover_need(need, arr, ab, buf):
    K = need.shape[0]
    nd = need.copy()
    n = 0
    for i in np.argsort(arr, kind="mergesort"):
        if not np.isfinite(arr[i]):
            break
        gain = False
        for k in range(K):
            if nd[k] > 0 and ab[i, k] > 0:
                gain = True
        if gain:
            buf[n] = i
            n += 1
            done = True
            for k in range(K):
                nd[k] -= ab[i, k]
                if nd[k] > 0:
                    done = False
            if done:
                break
    for k in range(K):
        if nd[k] > 0:
            return -1
    m = n
    for p in range(n - 1, -1, -1):  # drop a member if the slack absorbs it (latest arrival first)
        i = buf[p]
        ok = True
        for k in range(K):
            if ab[i, k] > 0 and nd[k] + ab[i, k] > 0:
                ok = False
                break
        if ok:
            for k in range(K):
                nd[k] += ab[i, k]
            buf[p] = -1
            m -= 1
    q = 0
    for p in range(n):
        if buf[p] >= 0:
            buf[q] = buf[p]
            q += 1
    return m


@njit(cache=True)
def plan_dispatch(req, ab, loc, dur, dep_xy, speed, active, free0, xy0, pre, npre, scope, tnow, noise, seed,
                  order, mem, cnt, out):
    np.random.seed(seed)
    T, K = req.shape
    A = ab.shape[0]
    free = free0.copy()
    px = xy0[:, 0].copy()
    py = xy0[:, 1].copy()
    ptr = np.zeros(A, np.int64)
    fixcnt = np.zeros(T, np.int64)
    fixmem = np.full((T, A), -1, np.int64)
    for i in range(A):
        for q in range(npre[i]):
            j = pre[i, q]
            fixmem[j, fixcnt[j]] = i
            fixcnt[j] += 1
    placed = ~active
    arr = np.empty(A)
    need = np.empty(K)
    buf = np.empty(A, np.int64)
    bbuf = np.empty(A, np.int64)
    cnt[:] = 0
    n = 0
    sumfin = 0.0
    while True:
        best, bscore, bstart, bc = -1, np.inf, 0.0, 0
        und, und_s = -1, np.inf  # fallback: an eligible task whose need no scope agent can cover now
        for j in range(T):
            if placed[j]:
                continue
            ok = True
            s = tnow
            for q in range(fixcnt[j]):
                m = fixmem[j, q]
                if ptr[m] >= npre[m] or pre[m, ptr[m]] != j:
                    ok = False
                    break
                a = free[m] + np.sqrt((px[m] - loc[j, 0]) ** 2 + (py[m] - loc[j, 1]) ** 2) / speed
                s = max(s, a)
            if not ok:
                continue
            anyneed = False
            for k in range(K):
                need[k] = req[j, k]
                for q in range(fixcnt[j]):
                    need[k] -= ab[fixmem[j, q], k]
                if need[k] > 0:
                    anyneed = True
            c = 0
            if anyneed:
                for i in range(A):
                    if scope[i] and ptr[i] >= npre[i]:
                        arr[i] = free[i] + np.sqrt((px[i] - loc[j, 0]) ** 2 + (py[i] - loc[j, 1]) ** 2) / speed
                    else:
                        arr[i] = np.inf
                c = _cover_need(need, arr, ab, buf)
                if c < 0:
                    if fixcnt[j] > 0 and s < und_s:
                        und, und_s = j, s
                    continue
                else:
                    for p in range(c):
                        s = max(s, arr[buf[p]])
            score = s * (1.0 + noise * np.random.random())
            if score < bscore:
                best, bscore, bstart, bc = j, score, s, c
                bbuf[:c] = buf[:c]
        if best < 0:
            if und < 0:
                break
            best, bstart, bc = und, und_s, 0  # under-covered: fixed members only (a lease resolves it)
        f = bstart + dur[best]
        w = 0
        for q in range(fixcnt[best]):
            m = fixmem[best, q]
            ptr[m] += 1
            mem[best, w] = m
            w += 1
        for p in range(bc):
            mem[best, w] = bbuf[p]
            w += 1
        cnt[best] = w
        for q in range(w):
            m = mem[best, q]
            free[m] = f
            px[m] = loc[best, 0]
            py[m] = loc[best, 1]
        order[n] = best
        n += 1
        placed[best] = True
        sumfin += f
    ms = 0.0
    for i in range(A):
        r = free[i] + np.sqrt((px[i] - dep_xy[i, 0]) ** 2 + (py[i] - dep_xy[i, 1]) ** 2) / speed
        ms = max(ms, r)
    out[0] = ms
    out[1] = sumfin
    return n


# ------------------------------------------------------------------------------------------------ world
class World:
    def __init__(self, inst, rel, dur_real, delays, R, p_loss, method, comm_seed, lease_grace=1.0, lease_max=10.0,
                 replan_period=1.0, k_plans=8, hyst=0.005):
        self.inst, self.rel, self.dur_real, self.delays = inst, rel, dur_real, delays
        self.R, self.p_loss, self.method = R, p_loss, method
        self.T, self.A = inst.n_tasks, inst.n_agents
        self.loc, self.req, self.ab, self.dur = inst.loc, inst.req, inst.ab, inst.dur
        self.v = inst.speed
        self.lease_grace, self.lease_max, self.period, self.k_plans = lease_grace, lease_max, replan_period, k_plans
        self.hyst = hyst
        self.comm_seed = comm_seed
        A, T = self.A, self.T
        N = A + 1  # node A = station
        self.N = N
        # truth
        self.done = np.zeros(T, bool)
        self.started = np.zeros(T, bool)
        self.t_start = np.full(T, np.nan)
        self.t_fin = np.full(T, np.nan)
        self.present = [set() for _ in range(T)]
        self.task_members = [set() for _ in range(T)]
        # agents
        self.mode = np.full(A, IDLE)
        self.target = np.full(A, -1)
        self.dep = np.zeros(A)
        self.tau = np.zeros(A)
        self.arr = np.zeros(A)
        self.p0 = inst.depot.copy()
        self.p1 = inst.depot.copy()
        self.xy = inst.depot.copy()
        self.wait_since = np.zeros(A)
        self.route = [[] for _ in range(A)]
        self.served = [set() for _ in range(A)]
        self.epoch = np.zeros(A, np.int64)
        self.adopted = np.full(A, -1)
        self.last_free = np.zeros(A)
        self.last_xy = inst.depot.copy()
        self.did = np.zeros(A, bool)
        # knowledge (per node)
        self.rel_k = np.zeros((N, T), bool)
        self.rel_k[:, rel <= 0] = True
        self.done_k = np.zeros((N, T), bool)
        self.start_k = np.zeros((N, T), bool)
        self.ver_k = np.full((N, A), -1)
        self.rec_k = np.full((N, A), -1)  # tick index of the freshest known record of each agent
        self.versions = [[] for _ in range(A)]  # (tick, author, route tuple)
        self.records = [dict() for _ in range(A)]  # tick -> record
        self.events = []
        self.seq = 0
        self.now = 0.0
        self.k = 0
        self.last_plan = {}
        self.stats = dict(plans=0, versions=0, versions_off_station=0, abandons=0, wasted=0,
                          plans_off_station=0, plan_s=0.0)
        self.lab = np.zeros(N, np.int64)
        self.rng_link = np.random.default_rng([comm_seed, 77])

    # ---------------- helpers
    def _push(self, t, kind, i):
        self.seq += 1
        heapq.heappush(self.events, (t, self.seq, kind, i, self.epoch[i] if kind != "fin" else 0))

    def _arrive(self, i, dep, tau):
        starts, ends = self.delays[i]
        if starts.size == 0:
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

    def _stalled(self, i, a, b):
        starts, ends = self.delays[i]
        if starts.size == 0 or b <= a:
            return 0.0
        return float(np.clip(np.minimum(ends, b) - np.maximum(starts, a), 0.0, None).sum())

    def _pos(self, i, t):
        if self.mode[i] in (TRAVEL, HOMING, RALLY):
            moved = (t - self.dep[i]) - self._stalled(i, self.dep[i], t)
            f = min(max(moved / max(self.tau[i], 1e-12), 0.0), 1.0)
            return self.p0[i] + (self.p1[i] - self.p0[i]) * f
        return self.xy[i]

    def _eta(self, i, t):
        return max(self.dep[i] + self.tau[i] + self._stalled(i, self.dep[i], t), t)

    def _depart(self, i, t, dest_xy, target, mode):
        p = self._pos(i, t) if self.mode[i] in (TRAVEL, HOMING, RALLY) else self.xy[i]
        tau = float(np.linalg.norm(dest_xy - p)) / self.v
        self.epoch[i] += 1
        self.mode[i], self.target[i] = mode, target
        self.dep[i], self.tau[i], self.p0[i], self.p1[i] = t, tau, p, dest_xy
        self.arr[i] = self._arrive(i, t, tau)
        self._push(self.arr[i], "arr", i)

    def _decide(self, i, t):
        r = self.route[i]
        while r and (self.done_k[i, r[0]] or r[0] in self.served[i]):
            r.pop(0)
        if r:
            j = r.pop(0)
            self._depart(i, t, self.loc[j], j, TRAVEL)
            return
        here = self._pos(i, t) if self.mode[i] in (TRAVEL, HOMING, RALLY) else self.xy[i]
        if self.done_k[i].all():  # knows everything is finished: go home
            dep_xy = self.inst.depot[i]
            if np.linalg.norm(here - dep_xy) < 1e-12:
                self.xy[i] = dep_xy
                self.mode[i], self.target[i] = IDLE, -1
                self.epoch[i] += 1
            else:
                self._depart(i, t, dep_xy, -1, HOMING)
            return
        # idle policy (method-agnostic): wait in place if the station's component reaches you, else rally to it
        self.xy[i] = np.array(here)
        self.mode[i], self.target[i] = IDLE, -1
        self.epoch[i] += 1

    def idle_policy(self, t):
        for i in range(self.A):
            if self.mode[i] == IDLE and not self.done_k[i].all():
                if self.lab[i] != self.lab[self.A] and np.linalg.norm(self.xy[i] - STATION) > 1e-9:
                    self._depart(i, t, STATION, -1, RALLY)
            elif self.mode[i] == RALLY and self.lab[i] == self.lab[self.A]:
                self.xy[i] = self._pos(i, t)
                self.mode[i], self.target[i] = IDLE, -1
                self.epoch[i] += 1
            elif self.mode[i] == IDLE and self.done_k[i].all() and np.linalg.norm(self.xy[i] - self.inst.depot[i]) > 1e-12:
                self._decide(i, t)

    def _start_if_covered(self, j, t):
        pres = self.present[j]
        if not pres:
            return
        tot = self.ab[list(pres)].sum(0)
        if np.all(tot >= self.req[j] - 1e-9):
            self.started[j] = True
            self.t_start[j] = t
            self.t_fin[j] = t + self.dur_real[j]
            self.task_members[j] = set(pres)
            for m in pres:
                self.mode[m] = WORK
                self.start_k[m, j] = True
                self.epoch[m] += 1
            self._push(self.t_fin[j], "fin", j)

    # ---------------- physics
    def physics_until(self, t_end):
        while self.events and self.events[0][0] <= t_end + 1e-12:
            t, _, kind, i, ep = heapq.heappop(self.events)
            self.now = t
            if kind == "fin":
                j = i
                self.done[j] = True
                for m in self.task_members[j]:
                    self.done_k[m, j] = True
                    self.served[m].add(j)
                    self.xy[m] = self.loc[j]
                    self.last_free[m], self.last_xy[m], self.did[m] = t, self.loc[j], True
                    self.mode[m] = IDLE
                    self._decide(m, t)
                self.present[j] = set()
                continue
            if ep != self.epoch[i]:
                continue
            if kind == "arr":
                if self.mode[i] == HOMING:
                    self.xy[i] = self.inst.depot[i]
                    self.mode[i], self.target[i] = IDLE, -1
                    continue
                if self.mode[i] == RALLY:
                    self.xy[i] = STATION.copy()
                    self.mode[i], self.target[i] = IDLE, -1
                    continue
                j = self.target[i]
                self.xy[i] = self.loc[j]
                if self.started[j] or self.done[j]:
                    self.stats["wasted"] += 1
                    if self.done[j]:
                        self.done_k[i, j] = True
                    else:
                        self.start_k[i, j] = True
                    self.served[i].add(j)  # never come back to it
                    self.last_free[i], self.last_xy[i], self.did[i] = t, self.loc[j], True
                    self.mode[i] = IDLE
                    self._decide(i, t)
                    continue
                self.mode[i] = WAIT
                self.wait_since[i] = t
                self.present[j].add(i)
                self._start_if_covered(j, t)

    # ---------------- knowledge
    def record(self, i):
        m = int(self.mode[i])
        j = int(self.target[i])
        if m == TRAVEL:
            tref = self._eta(i, self.now)
        elif m == WAIT:
            tref = self.now
        elif m == WORK:
            tref = max(self.t_start[j] + self.dur[j], self.now)
        else:
            tref = self.now
        xy = self._pos(i, self.now) if m in (TRAVEL, HOMING, RALLY) else self.xy[i]
        return (m, j, float(tref), np.array(xy), tuple(self.route[i]))

    def comm(self):
        A = self.A
        P = np.vstack([np.array([self._pos(i, self.now) for i in range(A)]), STATION])
        if not np.isfinite(self.R):
            lab = np.zeros(self.N, np.int64)
        else:
            D = np.linalg.norm(P[:, None] - P[None], axis=-1)
            adj = D <= self.R
            if self.p_loss > 0:
                up = self.rng_link.random(adj.shape) >= self.p_loss
                up = np.triu(up, 1)
                up = up | up.T
                adj = adj & (up | np.eye(self.N, dtype=bool))
            _, lab = connected_components(adj, directed=False)
        self.lab = lab
        return lab

    def merge(self, lab):
        for c in np.unique(lab):
            idx = np.flatnonzero(lab == c)
            if idx.size < 2:
                continue
            self.rel_k[idx] = self.rel_k[idx].any(0)
            self.done_k[idx] = self.done_k[idx].any(0)
            self.start_k[idx] = self.start_k[idx].any(0)
            self.ver_k[idx] = self.ver_k[idx].max(0)
            self.rec_k[idx] = self.rec_k[idx].max(0)

    # ---------------- planning
    def believed(self, n, i, fresh):
        """(mode, target, tref, xy, remaining) of agent i as known at node n (fresh: its true current record)."""
        if fresh:
            return self.records[i][self.k]
        kk = self.rec_k[n, i]
        return self.records[i][kk]

    def plan(self, n, scope_mask, comp_mask, rng_seed):
        """Planner at node n; scope agents re-planned; agents outside scope frozen on beliefs."""
        A, T = self.A, self.T
        t0 = time.perf_counter()
        known = self.rel_k[n] & ~self.done_k[n] & ~self.start_k[n]
        free0 = np.empty(A)
        xy0 = np.empty((A, 2))
        W = T
        pre = np.full((A, W + 1), -1, np.int64)
        npre = np.zeros(A, np.int64)
        for i in range(A):
            m, j, tref, xy, rem = self.believed(n, i, fresh=bool(comp_mask[i]))
            if m in (TRAVEL, WAIT, WORK) and j >= 0:
                free0[i], xy0[i] = tref, self.loc[j]  # ETA (travel), now (wait) or predicted finish (work)
                lst = [j] if (m != WORK and known[j]) else []
            else:
                free0[i], xy0[i] = tref, xy
                lst = []
            if not scope_mask[i] or self.method == "OPEN":
                lst = lst + [x for x in rem if known[x] and x not in lst]
            # drop duplicates keeping order
            seen, clean = set(), []
            for x in lst:
                if x not in seen:
                    seen.add(x)
                    clean.append(x)
            npre[i] = len(clean)
            pre[i, :len(clean)] = clean
        active = known.copy()
        order = np.empty(T, np.int64)
        mem = np.full((T, A), -1, np.int64)
        cnt = np.zeros(T, np.int64)
        out = np.empty(2)
        best = None
        for q in range(self.k_plans):
            noise = 0.0 if q == 0 else 0.3
            nplaced = plan_dispatch(self.req, self.ab, self.loc, self.dur, self.inst.depot, self.v, active, free0, xy0,
                                    pre, npre, scope_mask, self.now, noise, int(rng_seed * 1000 + q), order, mem, cnt, out)
            key = (-nplaced, out[0], out[1])
            if best is None or key < best[0]:
                best = (key, order[:nplaced].copy(), mem.copy(), cnt.copy())
        # incumbent: every scope agent keeps its believed route (no free assignment)
        pre_i, npre_i = pre.copy(), npre.copy()
        for i in np.flatnonzero(scope_mask):
            m, j, tref, xy, rem = self.believed(n, i, fresh=bool(comp_mask[i]))
            lst = list(pre[i, :npre[i]]) + [x for x in rem if known[x]]
            seen, clean = set(), []
            for x in lst:
                if x not in seen:
                    seen.add(x)
                    clean.append(int(x))
            npre_i[i] = len(clean)
            pre_i[i, :] = -1
            pre_i[i, :len(clean)] = clean
        n_inc = plan_dispatch(self.req, self.ab, self.loc, self.dur, self.inst.depot, self.v, active, free0, xy0,
                              pre_i, npre_i, np.zeros(A, np.bool_), self.now, 0.0, 0, order, mem, cnt, out)
        inc_key = (-n_inc, out[0], out[1])
        if (best[0][0] > inc_key[0]) or (best[0][0] == inc_key[0] and best[0][1] >= inc_key[1] * (1 - self.hyst)):
            self.stats["plan_s"] += time.perf_counter() - t0
            return {}
        _, order_b, mem_b, cnt_b = best
        # new routes for scope agents: committed prefix + tasks in placement order
        routes = {int(i): [] for i in np.flatnonzero(scope_mask)}
        for j in order_b:
            for q in range(cnt_b[j]):
                i = int(mem_b[j, q])
                if i in routes:
                    routes[i].append(int(j))
        self.stats["plan_s"] += time.perf_counter() - t0
        return routes

    def author(self, n, routes, comp_mask):
        """Write versions; node n and its component learn them at once."""
        idx = np.flatnonzero(self.lab == self.lab[n])
        changed = 0
        for i, r in routes.items():
            cur = self.versions[i][self.ver_k[n, i]][2] if self.ver_k[n, i] >= 0 else None
            # compare with what i currently executes as believed (committed + remaining)
            m, j, _, _, rem = self.believed(n, i, fresh=bool(comp_mask[i]))
            cur_full = ([j] if m in (TRAVEL, WAIT) and j >= 0 else []) + list(rem)
            if list(r) == cur_full:
                continue
            self.versions[i].append((self.k, n, tuple(r)))
            v = len(self.versions[i]) - 1
            self.ver_k[idx, i] = v
            changed += 1
            if not bool(comp_mask[i]) or self.lab[n] != self.lab[self.A]:
                self.stats["versions_off_station"] += 1
        self.stats["versions"] += changed
        return changed

    def adopt(self, t):
        for i in range(self.A):
            v = self.ver_k[i, i]
            if v > self.adopted[i]:
                self.adopted[i] = v
                r = list(self.versions[i][v][2])
                if r:
                    self.rel_k[i, r] = True
                cur = self.target[i] if self.mode[i] in (TRAVEL, WAIT, WORK) else -1
                if cur >= 0 and r and r[0] == cur:
                    r = r[1:]
                self.route[i] = [x for x in r if x not in self.served[i] and not self.done_k[i, x]]
                if self.mode[i] in (IDLE, HOMING, RALLY) and self.route[i]:
                    self._decide(i, t)
                elif self.mode[i] == HOMING and not self.route[i]:
                    pass

    def leases(self, t):
        for i in np.flatnonzero(self.mode == WAIT):
            j = int(self.target[i])
            waited = t - self.wait_since[i]
            if waited < self.lease_grace:
                continue
            coming = False
            if waited < self.lease_max:
                for m in range(self.A):
                    if m == i:
                        continue
                    kk = self.rec_k[i, m] if m != i else self.k
                    if kk < 0:
                        continue
                    mm, jj, _, _, rem = self.records[m][kk]
                    if (mm in (TRAVEL, WAIT) and jj == j) or j in rem:
                        coming = True
                        break
            if not coming:
                self.present[j].discard(i)
                self.stats["abandons"] += 1
                # self-authored version: remaining route without j
                self.versions[i].append((self.k, i, tuple(self.route[i])))
                v = len(self.versions[i]) - 1
                self.ver_k[i, i] = v
                self.adopted[i] = v
                self.mode[i] = IDLE
                self._decide(i, t)

    # ---------------- main loop
    def run(self):
        A = self.A
        # t = 0 briefing: full comm, one plan by the station
        self.k = 0
        for i in range(A):
            self.records[i][0] = self.record(i)
        self.rec_k[:, :] = 0
        allm = np.ones(A, bool)
        self.lab = np.zeros(self.N, np.int64)
        routes = self.plan(A, allm, allm, 0)
        self.author(A, routes, allm)
        self.ver_k[:, :] = self.ver_k[A]
        self.adopt(0.0)
        self.last_plan = {("st",): 0.0}
        sig_prev = {}
        method = self.method
        full = method == "FULL"
        while True:
            self.k += 1
            t = self.k * TICK
            if t > MAX_T:
                return False, MAX_T
            self.physics_until(t)
            self.now = t
            # station learns releases
            self.rel_k[A, self.rel <= t + 1e-12] = True
            for i in range(A):
                self.records[i][self.k] = self.record(i)
                self.rec_k[i, i] = self.k
            lab = np.zeros(self.N, np.int64) if full else self.comm()
            self.lab = lab
            self.merge(lab)
            # planners
            planners = []
            st_comp = lab[A]
            if method in ("FULL", "CEN-F", "CEN-R", "OPEN"):
                planners = [A]
            else:  # LOC-F, REP: leader of every component (station if present, else lowest id)
                for c in np.unique(lab):
                    idx = np.flatnonzero(lab == c)
                    planners.append(A if c == st_comp else int(idx.min()))
            for n in planners:
                comp_mask = (lab[:A] == lab[n])
                if method in ("CEN-R", "REP"):
                    scope = np.ones(A, bool)
                else:
                    scope = comp_mask.copy()
                if not scope.any():
                    continue
                members = tuple(np.flatnonzero(comp_mask))
                know = (int(self.rel_k[n].sum()), int(self.done_k[n].sum()), int(self.start_k[n].sum()),
                        int(self.ver_k[n].sum()))
                last = self.last_plan.get(n, -np.inf)
                prev = sig_prev.get(n)
                if method == "OPEN":  # insert unassigned known tasks only; never re-plan for execution noise
                    assigned = np.zeros(self.T, bool)
                    for i in range(A):
                        mm, jj, _, _, rem = self.believed(n, i, fresh=bool(comp_mask[i]))
                        if jj >= 0:
                            assigned[jj] = True
                        assigned[list(rem)] = True
                    unassigned = (self.rel_k[n] & ~self.done_k[n] & ~self.start_k[n] & ~assigned).any()
                    trig = prev is None or know[0] != prev[0][0] or (unassigned and (members != prev[1] or
                                                                                   t - last >= self.period - 1e-9))
                else:
                    trig = (prev is None or know != prev[0] or (members != prev[1] and t - last >= 0.5 - 1e-9)
                            or (t - last) >= self.period - 1e-9)
                if not trig:
                    continue
                self.last_plan[n] = t
                routes = self.plan(n, scope, comp_mask, self.k * 131 + n)
                self.stats["plans"] += 1
                if n != A:
                    self.stats["plans_off_station"] += 1
                self.author(n, routes, comp_mask)
                sig_prev[n] = ((int(self.rel_k[n].sum()), int(self.done_k[n].sum()), int(self.start_k[n].sum()),
                                int(self.ver_k[n].sum())), members)
            self.adopt(t)
            self.leases(t)
            self.idle_policy(t)
            if self.done.all():
                return True, self.hindsight_makespan()

    def hindsight_makespan(self):
        """All tasks finished and every robot back home, each leaving right after its last task (HeteroMRTA
        makespan without the termination-detection latency, which would add comm effects unrelated to planning)."""
        v = self.v
        ret = [self.last_free[i] + np.linalg.norm(self.last_xy[i] - self.inst.depot[i]) / v
               for i in range(self.A) if self.did[i]]
        return float(max(np.nanmax(self.t_fin), max(ret) if ret else 0.0))


# ------------------------------------------------------------------------------------------------ campaign
def job(args):
    setting, i, path, scen, R, p_loss, method, seed = args
    snap = HERE / "_snap"
    sys.path.insert(0, str(snap))
    from cbba_sota.hetero import Instance
    from dyn_env import Scenario, delay_calendar, realized_durations, release_times

    inst = Instance.from_pickle(path)
    sc = Scenario(scen["name"], release=scen["release"], dod=scen.get("dod", 0.5), horizon=HORIZON[setting],
                  dur_sigma=scen["sigma"], p_delay=scen["p"], min_delay=scen["dmin"], max_delay=scen["dmax"])
    rel = release_times(sc, inst.n_tasks, i, seed)
    dur_real = realized_durations(sc, np.asarray(inst.dur), i, seed)
    delays = delay_calendar(sc, inst.n_agents, i, seed)
    t0 = time.perf_counter()
    w = World(inst, rel, dur_real, delays, R, p_loss, method, comm_seed=seed)
    ok, ms = w.run()
    return dict(setting=setting, inst=i, scen=scen["name"], R=R, p_loss=p_loss, method=method, seed=seed,
                success=ok, makespan=ms, wall=time.perf_counter() - t0, **w.stats)


SCENS = {
    "RN12": dict(name="RN12", release="poisson", dod=0.5, sigma=0.3, p=0.01, dmin=1, dmax=4),
    "RN3": dict(name="RN3", release="poisson", dod=0.5, sigma=0.3, p=0.05, dmin=1, dmax=10),
    "N12": dict(name="N12", release="none", sigma=0.3, p=0.01, dmin=1, dmax=4),
    "R": dict(name="R", release="poisson", dod=0.5, sigma=0.0, p=0.0, dmin=1, dmax=4),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--procs", type=int, default=48)
    ap.add_argument("--settings", nargs="*", default=["MA-AT-25-5-50", "MA-AT-50-5-50", "SA-AT-50-5-50",
                                                        "SA-BT-50-5-50"])
    ap.add_argument("--scens", nargs="*", default=["RN12", "RN3"])
    ap.add_argument("--radii", nargs="*", type=float, default=[0.3, 0.2, 0.15, 0.1, 0.05])
    ap.add_argument("--loss", type=float, default=0.2)
    ap.add_argument("--methods", nargs="*", default=["CEN-F", "CEN-R", "LOC-F", "REP", "OPEN"])
    ap.add_argument("--out", default=str(HERE / "out" / "integ_headroom.jsonl"))
    args = ap.parse_args()
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ[k] = "1"
    data = HERE / "_snap" / "data" / "hetero"
    jobs = []
    for s in args.settings:
        for i in range(args.n):
            path = str(data / s / "dev" / f"env_{i}.pkl")
            for sc in args.scens:
                for seed in range(args.seeds):
                    jobs.append((s, i, path, SCENS[sc], np.inf, 0.0, "FULL", seed))
                    for R in args.radii:
                        for m in args.methods:
                            jobs.append((s, i, path, SCENS[sc], R, args.loss, m, seed))
    print(len(jobs), "jobs", flush=True)
    t0 = time.perf_counter()
    rows = []
    with open(args.out, "w") as f, mp.get_context("spawn").Pool(args.procs) as pool:
        for k, r in enumerate(pool.imap_unordered(job, jobs, chunksize=1)):
            rows.append(r)
            f.write(json.dumps(r, default=float) + "\n")
            f.flush()
            if (k + 1) % 50 == 0:
                print(f"{k + 1}/{len(jobs)} {time.perf_counter() - t0:.0f}s", flush=True)
    print("done", time.perf_counter() - t0)


if __name__ == "__main__":
    main()
