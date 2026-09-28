"""Comm-first proposer pilot (2026-09-28): does delivery-aware replicated planning beat the replicated solver?

Reuses the integrated-stressor world of pilots/trackD/integ_headroom.py unchanged (online release at a station,
lognormal durations, LoRR-2026 delays, unit-disk range R + Bernoulli per-link loss, instant flooding inside a
component, versioned routes, leases, rally-to-station idle policy shared by every method) and adds two arms:

  DAR    delivery-aware replicated planning. Every component leader plans for ALL robots (as REP), but a robot
         outside the planner's component can only switch to the new version at tau_i, its predicted information
         arrival time: the earliest time a version authored now reaches i by epidemic flooding over the robots'
         BELIEVED trajectories (temporal BFS on a 0.5 time grid, range R, horizon 40; idle robots are predicted to
         rally to the station, the shared idle policy). Before tau_i the robot is assumed to keep executing its
         believed route: every believed task it would depart for before tau_i is a frozen prefix, and new
         assignments may only start travel at max(prefix end, tau_i).
         tau_i = now for everyone  -> REP;   tau_i = inf for every out-of-component robot -> LOC-F.
  DAR-r  ablation: the ready time tau_i only (no frozen prefix beyond the committed head, as in REP).
Collapse arms added later (all share the DAR code path; out-of-component robots keep only their committed head
unless stated):
  FRZ<k> freeze the first k believed tasks of every out-of-component robot, ready = now. FRZ1 = "REP+": REP with
         a stale robot's free time clamped to now (the original REP lets the planner believe stale robots can
         arrive in the past); FRZ1 is the strongest replicated-solver control.
  CST<c> DAR with tau_i = now + c (constant delay, frozen prefix);  CSR<c> DAR-r with tau_i = now + c.
  INF-r  tau_i = inf: never give new work to robots outside the component (component-scoped planning, no freezing,
         redundant service tolerated; = P1's "Oh" arm).
  DTH<t> DAR-r with tau_i set to inf when the predicted delay exceeds t (thresholded scope).
  <arm>+o  oracle release broadcast (every node learns releases at once; robot-robot state stays range-limited).
  <arm>@A<iters>  coalition ALNS instead of list scheduling, COLD-started at every re-plan (no warm start, no
         coalition commitment): kept as a documented negative. It churns routes and loses to the dispatcher even
         with global comm; a proper rolling ALNS (warm start + commitment) is in planner_collapse.py.
Summaries: out/*_summary.txt (summarize.py, summarize3.py).

Usage: .venv/bin/python pilots/trackD_comm2/dar_pilot.py --n 10 --seeds 2 --procs 24
"""
from __future__ import annotations

import argparse
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
TD = HERE.parent / "trackD"
sys.path.insert(0, str(TD))

GRID_DT = 0.5
GRID_H = 40.0


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
    for p in range(n - 1, -1, -1):
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
def plan_dispatch_ready(req, ab, loc, dur, dep_xy, speed, active, free0, xy0, pre, npre, scope, ready, tnow, noise,
                        seed, order, mem, cnt, out):
    """integ_headroom.plan_dispatch with a per-agent ready time for FREE assignments (after the prefix)."""
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
        und, und_s = -1, np.inf
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
                    if scope[i] and ptr[i] >= npre[i] and np.isfinite(ready[i]):
                        arr[i] = max(free[i], ready[i]) + np.sqrt((px[i] - loc[j, 0]) ** 2 +
                                                                  (py[i] - loc[j, 1]) ** 2) / speed
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
            best, bstart, bc = und, und_s, 0
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


_A = None


def _alns():
    """Snapshot ALNS with the state-start kernel copy of the robustness pilot (first leg = dout)."""
    global _A
    if _A is None:
        sys.path.insert(0, str(TD / "_snap"))
        sys.path.insert(0, str(HERE.parent / "trackD_robust"))
        import robust_rh_kernels as RK

        from cbba_sota.solvers import _alns_pool as P
        from cbba_sota.solvers import alns as A

        orig = A._inst_arrays

        def inst_arrays(inst):
            dout = getattr(inst, "_dout", None)
            return orig(inst) + (A._own(inst.da if dout is None else dout),)

        A._inst_arrays = inst_arrays
        A.K = RK
        P.K = RK
        _A = A
    return _A


def make_world_cls():
    import integ_headroom as ih

    TRAVEL, WAIT, WORK, HOMING, IDLE, RALLY = ih.TRAVEL, ih.WAIT, ih.WORK, ih.HOMING, ih.IDLE, ih.RALLY
    STATION, TICK = ih.STATION, ih.TICK

    class DarWorld(ih.World):
        variant = "DAR"

        def _rec(self, n, i, fresh):
            kk = self.k if fresh else int(self.rec_k[n, i])
            return kk * TICK, self.records[i][kk]

        def predicted_tau(self, n, comp_mask):
            """Earliest predicted arrival time of a version authored now at node n, per agent (inf if > horizon)."""
            A, v = self.A, self.v
            grid = self.now + np.arange(0.0, GRID_H + 1e-9, GRID_DT)
            P = np.empty((grid.size, A + 1, 2))
            P[:, A] = STATION
            for i in range(A):
                t_rec, (m, j, tref, xy, rem) = self._rec(n, i, bool(comp_mask[i]))
                ts, xs = [t_rec], [np.asarray(xy, float)]
                if m == TRAVEL and j >= 0:
                    ts += [max(tref, t_rec), max(tref, t_rec) + self.dur[j]]
                    xs += [self.loc[j], self.loc[j]]
                elif m == WAIT and j >= 0:
                    ts += [max(t_rec, self.now) + self.dur[j]]
                    xs += [self.loc[j]]
                elif m == WORK and j >= 0:
                    ts += [max(tref, t_rec)]
                    xs += [self.loc[j]]
                elif m == HOMING:
                    dp = self.inst.depot[i]
                    ts += [t_rec + np.linalg.norm(dp - xs[-1]) / v]
                    xs += [dp]
                t, pos = ts[-1], xs[-1]
                if m != HOMING:
                    for x in rem:
                        if x == j or self.done_k[n, x]:
                            continue
                        t = t + np.linalg.norm(self.loc[x] - pos) / v
                        ts.append(t)
                        xs.append(self.loc[x])
                        t = t + self.dur[x]
                        ts.append(t)
                        xs.append(self.loc[x])
                        pos = self.loc[x]
                    # shared idle policy: rally to the station once the route is exhausted
                    t = t + np.linalg.norm(STATION - pos) / v
                    ts.append(t)
                    xs.append(STATION)
                ts = np.maximum.accumulate(np.asarray(ts)) + np.arange(len(ts)) * 1e-9
                xs = np.asarray(xs)
                P[:, i, 0] = np.interp(grid, ts, xs[:, 0])
                P[:, i, 1] = np.interp(grid, ts, xs[:, 1])
            inf_ = np.zeros(A + 1, bool)
            inf_[:A] = comp_mask
            inf_[A] = self.lab[A] == self.lab[n]
            tau = np.full(A + 1, np.inf)
            tau[inf_] = self.now
            for g, t in enumerate(grid):
                if inf_.all():
                    break
                D = np.linalg.norm(P[g][:, None] - P[g][None], axis=-1)
                _, lab = connected_components(D <= self.R, directed=False)
                hit = np.isin(lab, np.unique(lab[inf_]))
                new = hit & ~inf_
                tau[new] = t
                inf_ |= hit
            return tau[:A]

        use_alns = False
        alns_iters = 300

        def plan_alns(self, n, scope_mask, comp_mask, rng_seed):
            """Coalition ALNS (Phase-1 kernels, state-start copy) instead of list scheduling. Committed heads stay;
            a head task still missing traits is re-planned with its residual requirement (its committed members are
            busy there); robots outside the planning scope are unavailable (heads count as fixed members)."""
            from cbba_sota.hetero import Instance

            A_ = _alns()
            A, T, v = self.A, self.T, self.v
            t0 = time.perf_counter()
            meth = self.method
            if meth.startswith("INF"):
                avail_scope = comp_mask.copy()
            else:
                avail_scope = scope_mask.copy()
            known = self.rel_k[n] & ~self.done_k[n] & ~self.start_k[n]
            recs = [self.believed(n, i, fresh=bool(comp_mask[i])) for i in range(A)]
            fixed = [[] for _ in range(T)]
            eta = np.zeros(A)
            for i, (m, j, tref, xy, rem) in enumerate(recs):
                if m in (TRAVEL, WAIT) and j >= 0 and known[j]:
                    fixed[j].append(i)
                    eta[i] = max(tref, self.now) if m == TRAVEL else self.now
            resid = self.req.astype(float).copy()
            full_head = np.zeros(T, bool)
            fin = np.full(T, np.nan)
            for j in range(T):
                if fixed[j]:
                    resid[j] = np.maximum(self.req[j] - self.ab[fixed[j]].sum(0), 0.0)
                    if not (resid[j] > 0).any():
                        full_head[j] = True
                        fin[j] = max(eta[f] for f in fixed[j]) + self.dur[j]
            BIG = 1e6
            ready = np.full(A, BIG)
            pos = np.zeros((A, 2))
            for i, (m, j, tref, xy, rem) in enumerate(recs):
                if not avail_scope[i]:
                    continue
                if m in (TRAVEL, WAIT) and j >= 0 and known[j]:
                    if full_head[j]:
                        ready[i], pos[i] = fin[j], self.loc[j]
                elif m == WORK and j >= 0:
                    ready[i], pos[i] = max(tref, self.now), self.loc[j]
                else:
                    ready[i], pos[i] = max(tref, self.now), np.asarray(xy, float)
            av = ready < BIG / 2
            cap = self.ab[av].sum(0)
            F = np.array([j for j in range(T) if known[j] and not full_head[j] and (resid[j] > 0).any()
                          and (cap >= resid[j] - 1e-9).all()], dtype=np.int64)
            routes = {int(i): ([int(recs[i][1])] if recs[i][0] in (TRAVEL, WAIT) and recs[i][1] >= 0
                               and known[recs[i][1]] else []) for i in np.flatnonzero(avail_scope)}
            if F.size:
                sub = Instance(req=resid[F], loc=self.loc[F], dur=self.dur[F], ab=self.ab, depot=self.inst.depot,
                               species=self.inst.species, speed=v)
                dout = np.where(av[:, None], ready[:, None] + np.linalg.norm(pos[:, None] - self.loc[F][None], axis=-1) / v,
                                BIG)
                object.__setattr__(sub, "_dout", dout)
                cfg = A_.ALNSConfig.v1(verify=False)
                plan, _ = A_.solve(sub, 60.0, seed=int(rng_seed) % 100000, config=cfg, max_iters=self.alns_iters)
                for f in plan.order():
                    for i in plan.members[f]:
                        if int(i) in routes and av[i]:
                            routes[int(i)].append(int(F[f]))
            self.stats["plan_s"] += time.perf_counter() - t0
            # every robot in the planning scope gets a version (a robot held at a partially covered head keeps
            # only that head; it is re-planned when the head finishes)
            return routes

        def plan(self, n, scope_mask, comp_mask, rng_seed):
            meth = self.method
            if self.use_alns:
                return self.plan_alns(n, scope_mask, comp_mask, rng_seed)
            if not meth.startswith(("DAR", "CST", "FRZ", "CSR", "INF", "DTH")):
                return super().plan(n, scope_mask, comp_mask, rng_seed)
            import re
            num = re.match(r"[A-Z]+-?r?(\d*\.?\d*)", meth).group(1)
            freeze_k = int(num) if meth.startswith("FRZ") else 0
            A, T = self.A, self.T
            t0 = time.perf_counter()
            known = self.rel_k[n] & ~self.done_k[n] & ~self.start_k[n]
            out_mask = ~comp_mask
            if meth.startswith(("CST", "CSR")):
                tau = np.full(A, self.now + float(num))
            elif meth.startswith("INF"):
                tau = np.full(A, np.inf)
            elif meth.startswith("FRZ") or not out_mask.any():
                tau = np.full(A, self.now)
            else:
                tau = self.predicted_tau(n, comp_mask)
                if meth.startswith("DTH"):  # thresholded scope: out-of-reach robots beyond theta are unavailable
                    tau = np.where(tau - self.now > float(num), np.inf, tau)
            self.stats["tau_sum"] = self.stats.get("tau_sum", 0.0) + float(
                np.sum(np.minimum(tau[out_mask], self.now + GRID_H) - self.now))
            self.stats["tau_n"] = self.stats.get("tau_n", 0) + int(out_mask.sum())
            self.stats["tau_inf"] = self.stats.get("tau_inf", 0) + int(np.isinf(tau[out_mask]).sum())
            free0 = np.empty(A)
            xy0 = np.empty((A, 2))
            pre = np.full((A, T + 1), -1, np.int64)
            npre = np.zeros(A, np.int64)
            ready = np.full(A, float(self.now))
            rems = {}
            for i in range(A):
                m, j, tref, xy, rem = self.believed(n, i, fresh=bool(comp_mask[i]))
                rems[i] = rem
                if m in (TRAVEL, WAIT, WORK) and j >= 0:
                    free0[i], xy0[i] = tref, self.loc[j]
                    lst = [j] if (m != WORK and known[j]) else []
                else:
                    free0[i], xy0[i] = tref, xy
                    lst = []
                if out_mask[i]:
                    ready[i] = tau[i]
                    if meth.startswith("FRZ"):
                        lst = (lst + [int(x) for x in rem if known[x] and x not in lst])[:freeze_k]
                    elif meth in ("DAR",) or meth.startswith("CST"):
                        t, pos = free0[i], xy0[i]
                        for x in rem:
                            if not known[x] or x in lst:
                                continue
                            if t < tau[i]:
                                lst.append(int(x))
                                t = max(t, self.now) + np.linalg.norm(self.loc[x] - pos) / self.v + self.dur[x]
                                pos = self.loc[x]
                            else:
                                break
                seen, clean = set(), []
                for x in lst:
                    if x not in seen:
                        seen.add(x)
                        clean.append(x)
                npre[i] = len(clean)
                pre[i, :len(clean)] = clean
            scope = np.ones(A, np.bool_)
            active = known.copy()
            order = np.empty(T, np.int64)
            mem = np.full((T, A), -1, np.int64)
            cnt = np.zeros(T, np.int64)
            out = np.empty(2)
            best = None
            for q in range(self.k_plans):
                noise = 0.0 if q == 0 else 0.3
                nplaced = plan_dispatch_ready(self.req, self.ab, self.loc, self.dur, self.inst.depot, self.v, active,
                                              free0, xy0, pre, npre, scope, ready, self.now, noise,
                                              int(rng_seed * 1000 + q), order, mem, cnt, out)
                key = (-nplaced, out[0], out[1])
                if best is None or key < best[0]:
                    best = (key, order[:nplaced].copy(), mem.copy(), cnt.copy())
            # incumbent: every agent keeps its believed route
            pre_i, npre_i = pre.copy(), npre.copy()
            for i in range(A):
                lst = list(pre[i, :npre[i]]) + [x for x in rems[i] if known[x]]
                seen, clean = set(), []
                for x in lst:
                    if x not in seen:
                        seen.add(x)
                        clean.append(int(x))
                npre_i[i] = len(clean)
                pre_i[i, :] = -1
                pre_i[i, :len(clean)] = clean
            n_inc = plan_dispatch_ready(self.req, self.ab, self.loc, self.dur, self.inst.depot, self.v, active, free0,
                                        xy0, pre_i, npre_i, np.zeros(A, np.bool_), ready, self.now, 0.0, 0, order,
                                        mem, cnt, out)
            inc_key = (-n_inc, out[0], out[1])
            if (best[0][0] > inc_key[0]) or (best[0][0] == inc_key[0] and best[0][1] >= inc_key[1] * (1 - self.hyst)):
                self.stats["plan_s"] += time.perf_counter() - t0
                return {}
            _, order_b, mem_b, cnt_b = best
            routes = {int(i): [] for i in range(A)}
            for j in order_b:
                for q in range(cnt_b[j]):
                    routes[int(mem_b[j, q])].append(int(j))
            self.stats["plan_s"] += time.perf_counter() - t0
            return routes

    # run(): identical to integ_headroom.World.run except that DAR arms, like REP, give every planner the full
    # scope (so a station without robots in its component still authors versions); plan() then splits in/out.
    import inspect
    import textwrap

    src = textwrap.dedent(inspect.getsource(ih.World.run))
    old = 'if method in ("CEN-R", "REP"):'
    assert src.count(old) == 1
    src = src.replace(old, 'if method in ("CEN-R", "REP") or method.startswith(("DAR", "CST", "FRZ", "CSR", "INF", "DTH")):')
    old2 = "self.rel_k[A, self.rel <= t + 1e-12] = True"
    assert src.count(old2) == 1
    # "+o" arms: oracle release broadcast (a long-range dispatch downlink); robot-robot state stays range-limited
    ind = next(ln[: len(ln) - len(ln.lstrip())] for ln in src.splitlines() if old2 in ln)
    src = src.replace(old2, old2 + f"\n{ind}if method.endswith('+o'):\n{ind}    self.rel_k[:, self.rel <= t + 1e-12] = True")
    ns = dict(vars(ih))
    exec(compile(src, "dar_run", "exec"), ns)
    DarWorld.run = ns["run"]
    return DarWorld


def job(args):
    setting, i, path, scen, R, p_loss, method, seed = args
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ[k] = "1"
    snap = TD / "_snap"
    sys.path.insert(0, str(snap))
    from cbba_sota.hetero import Instance
    from dyn_env import Scenario, delay_calendar, realized_durations, release_times

    import integ_headroom as ih

    inst = Instance.from_pickle(path)
    sc = Scenario(scen["name"], release=scen["release"], dod=scen.get("dod", 0.5), horizon=ih.HORIZON[setting],
                  dur_sigma=scen["sigma"], p_delay=scen["p"], min_delay=scen["dmin"], max_delay=scen["dmax"])
    rel = release_times(sc, inst.n_tasks, i, seed)
    dur_real = realized_durations(sc, np.asarray(inst.dur), i, seed)
    delays = delay_calendar(sc, inst.n_agents, i, seed)
    W = make_world_cls()
    # DAR arms run as component-leader planners with scope handled inside plan(); run() treats unknown methods
    # like LOC-F for the planner set and passes scope = component (overridden in plan()).
    t0 = time.perf_counter()
    base, _, planner = method.partition("@")
    w = W(inst, rel, dur_real, delays, R, p_loss, {"CEN": "CEN-F"}.get(base, base), comm_seed=seed)
    if planner:
        w.use_alns = True
        w.alns_iters = int(planner[1:] or 300)
    ok, ms = w.run()
    return dict(setting=setting, inst=i, scen=scen["name"], R=R, p_loss=p_loss, method=method, seed=seed,
                success=ok, makespan=ms, wall=time.perf_counter() - t0, **w.stats)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--procs", type=int, default=24)
    ap.add_argument("--settings", nargs="*", default=["MA-AT-25-5-50", "MA-AT-50-5-50", "SA-BT-50-5-50"])
    ap.add_argument("--scens", nargs="*", default=["RN12"])
    ap.add_argument("--radii", nargs="*", type=float, default=[0.2, 0.15, 0.1])
    ap.add_argument("--loss", type=float, default=0.2)
    ap.add_argument("--methods", nargs="*", default=["REP", "DAR", "DAR-r", "LOC-F", "CEN-F"])
    ap.add_argument("--out", default=str(HERE / "out" / "dar.jsonl"))
    ap.add_argument("--no_full", action="store_true")
    ap.add_argument("--full_methods", nargs="*", default=[], help="global-comm arms (R = inf), e.g. FULL@A300")
    args = ap.parse_args()
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ[k] = "1"
    import integ_headroom as ih

    data = TD / "_snap" / "data" / "hetero"
    SCEN_ALL = dict(ih.SCENS, S0=dict(name="S0", release="none", sigma=0.0, p=0.0, dmin=1, dmax=4))
    jobs = []
    for s in args.settings:
        for i in range(args.n):
            path = str(data / s / "dev" / f"env_{i}.pkl")
            for sc in args.scens:
                for seed in range(args.seeds):
                    if not args.no_full:
                        jobs.append((s, i, path, SCEN_ALL[sc], np.inf, 0.0, "FULL", seed))
                    for m in args.full_methods:
                        jobs.append((s, i, path, SCEN_ALL[sc], np.inf, 0.0, m, seed))
                    for R in args.radii:
                        for m in args.methods:
                            jobs.append((s, i, path, SCEN_ALL[sc], R, args.loss, m, seed))
    # longest first
    jobs.sort(key=lambda j: (not j[6].startswith(("REP", "DAR", "CST", "FRZ", "CSR", "INF", "DTH")),
                             -(0 if not np.isfinite(j[4]) else 1 / j[4])))
    print(len(jobs), "jobs", flush=True)
    t0 = time.perf_counter()
    with open(args.out, "w") as f, mp.get_context("spawn").Pool(args.procs) as pool:
        for k, r in enumerate(pool.imap_unordered(job, jobs, chunksize=1)):
            f.write(json.dumps(r, default=float) + "\n")
            f.flush()
            if (k + 1) % 100 == 0:
                print(f"{k + 1}/{len(jobs)} {time.perf_counter() - t0:.0f}s", flush=True)
    print("done", time.perf_counter() - t0, flush=True)


if __name__ == "__main__":
    main()
