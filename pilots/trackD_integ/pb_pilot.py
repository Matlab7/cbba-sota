"""Track D integrated proposer, pilot P4 (2026-09-28): predictive beliefs (PB) on top of the replicated solver (REP).

Pilot P3 (decomp_pilot.py) showed that a global KNOWLEDGE oracle closes 57-75% of REP's makespan gap to FULL at
R <= r_c while a release oracle closes ~0%: REP loses mainly because a component plans on frozen snapshots of
out-of-contact robots (last heard record) and on task statuses it has not heard about (so it sends robots to tasks
that already started: 7-44 wasted trips per episode). Communication cannot deliver that knowledge inside a
partition, but the replicated plan PREDICTS it: every robot follows its adopted route in key order, so a node can
dead-reckon each silent robot along its last known route with the nominal model.

PB = REP + at every plan, for each robot b outside the planner's component:
  - roll b forward from its last record (time t_b) along its believed route: a travelling b arrives at its ETA,
    a waiting b starts at t_b, a working b finishes at its predicted finish, then leg by leg with nominal travel
    time x KAPPA and nominal durations, starting each task on b's own arrival (optimistic; partners unknown);
  - tasks whose predicted start is earlier than now - M_START are treated as started (not re-planned, no new
    members), tasks whose predicted finish is earlier than now - M_FIN as finished;
  - b's planning state becomes its PREDICTED current state (working until its predicted finish, or travelling to
    its predicted next task, or idle at the end of its route) instead of its stale snapshot;
  - evidence wins: a task some in-component robot is travelling to or waiting at is never predicted started, and
    gossip truth (start/finish/record) replaces predictions as soon as it arrives; nothing is locked for the future
    (future tasks of b stay re-plannable exactly as in REP -- the CAS lesson of pilot P2).
Variants: PB (both), PB-S (task statuses only, REP's snapshot of b), PB-A (b's predicted state only),
CLAMP (REP's stale snapshot of b with its free time clamped to >= now: "still doing what it was last seen doing").
M_START / M_FIN are the prediction margins (the only place where "slack" enters); two settings are piloted.

Usage: .venv/bin/python pilots/trackD_integ/pb_pilot.py --procs 24
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

HERE = Path(__file__).resolve().parent
TD = HERE.parent / "trackD"
sys.path.insert(0, str(TD))
KAPPA = 1.0


def make_cls():
    import integ_headroom as IH

    class PBWorld(IH.World):
        def __init__(self, *a, pb="PB", m_start=0.5, m_fin=0.5, **kw):
            super().__init__(*a, **kw)
            self.pb, self.m_start, self.m_fin = pb, m_start, m_fin
            # staleness gate (SG<tau>): a component that does not contain the station keeps the station's plan
            # until its freshest member last saw the station more than tau ago (or a member is idle with no route)
            self.sg_tau = float(pb[2:]) if pb.startswith("SG") else None
            if self.sg_tau is not None:
                self.pb = "CLAMP"
            # HYB: asymmetric scope by information quality -- the station re-plans only robots in its component
            # (central + frozen fallback, as CEN-F), every other component re-plans ALL robots optimistically with
            # the clamped teammate model (as CLAMP)
            # HYB3<theta>: HYB, but a non-station component re-plans only if the station-component fraction its
            # members last observed (when they were in it) is below theta (network-regime detector), or a member is
            # idle with an empty route; otherwise it keeps the station's plan (= CEN-F behaviour)
            self.hub_theta = float(pb[4:]) if pb.startswith("HYB3") else None
            if self.hub_theta is not None:
                pb = "HYB"
            self.hub_frac_seen = np.ones(self.A + 1)
            self.hyb = pb in ("HYB", "HYB2")
            self.local_write = pb == "HYB2"  # HYB2: non-station components plan globally but write only own routes
            if self.hyb:
                self.pb = "CLAMP"
            self.last_st = np.zeros(self.A + 1)
            self.stats.update(sg_skipped=0, sg_planned=0)
            self.stats.update(pred_started=0, pred_done=0, pred_calls=0, pred_wrong_started=0, pred_wrong_done=0,
                              acc_n=0, acc_target_ok=0, acc_stale_target_ok=0, acc_free_err=0.0, acc_stale_free_err=0.0)

        def _roll(self, n, b):
            """Predicted (mode, target, free_time, xy, started_set, done_set) of silent robot b at node n."""
            m, j, tref, xy, rem = self.records[b][self.rec_k[n, b]]
            t_b = self.rec_k[n, b] * IH.TICK
            now = self.now
            started, done = [], []
            if m in (IH.TRAVEL, IH.WAIT, IH.WORK) and j >= 0:
                if m == IH.TRAVEL:
                    st = tref
                elif m == IH.WAIT:
                    st = t_b
                else:
                    st = tref - self.dur[j]
                fin = tref if m == IH.WORK else st + self.dur[j]
                cur = (j, st, fin)
                chain = [x for x in rem if x != j]
            else:
                cur = None
                chain = list(rem)
            pos = self.loc[cur[0]] if cur else np.asarray(xy, float)
            t = cur[2] if cur else max(tref, t_b)
            state = None
            while True:
                if cur is not None:
                    x, st, fin = cur
                    if st < now - self.m_start:
                        started.append(x)
                    if fin < now - self.m_fin:
                        done.append(x)
                    if st > now:  # still travelling to x
                        state = (IH.TRAVEL, x, st, self.loc[x])
                        break
                    if fin > now:  # working on x
                        state = (IH.WORK, x, fin, self.loc[x])
                        break
                    pos, t = self.loc[x], fin
                if not chain:
                    state = (IH.IDLE, -1, max(t, now), pos)
                    break
                x = chain.pop(0)
                arr = t + KAPPA * float(np.linalg.norm(self.loc[x] - pos)) / self.v
                cur = (x, arr, arr + self.dur[x])
            return state, started, done

        def merge(self, lab):
            super().merge(lab)
            inhub = lab == lab[self.A]
            self.last_st[inhub] = self.now
            self.hub_frac_seen[inhub] = inhub[:self.A].mean()

        def plan(self, n, scope_mask, comp_mask, rng_seed):
            A, T = self.A, self.T
            if self.hyb and n == A:
                scope_mask = comp_mask.copy()
            if self.hub_theta is not None and n != A:
                members = np.flatnonzero(comp_mask)
                idle_empty = any(self.mode[i] == IH.IDLE and not self.route[i] for i in members)
                if self.hub_frac_seen[members].min() >= self.hub_theta and not idle_empty:
                    self.stats["sg_skipped"] += 1
                    return {}
                self.stats["sg_planned"] += 1
            if self.sg_tau is not None and n != A:
                members = np.flatnonzero(comp_mask)
                stale = self.now - self.last_st[members].max()
                idle_empty = any(self.mode[i] == IH.IDLE and not self.route[i] for i in members)
                if stale < self.sg_tau and not idle_empty:
                    self.stats["sg_skipped"] += 1
                    return {}
                self.stats["sg_planned"] += 1
            t0 = time.perf_counter()
            known = self.rel_k[n] & ~self.done_k[n] & ~self.start_k[n]
            # evidence: tasks an in-component robot is travelling to / waiting at are not started
            evid = np.zeros(T, bool)
            for i in np.flatnonzero(comp_mask):
                m, j, _, _, _ = self.records[i][self.k]
                if m in (IH.TRAVEL, IH.WAIT) and j >= 0:
                    evid[j] = True
            pred = {}
            if self.pb in ("PB", "PB-S", "PB-A"):
                for b in np.flatnonzero(~comp_mask):
                    if self.rec_k[n, b] < 0:
                        continue
                    state, st, dn = self._roll(n, b)
                    pred[int(b)] = state
                    self.stats["pred_calls"] += 1
                    # accuracy vs truth (logged only): predicted target and free time vs the stale snapshot
                    tm, tj = int(self.mode[b]), int(self.target[b])
                    sm, sj, stref, _, _ = self.records[b][self.rec_k[n, b]]
                    true_free = (self.arr[b] if tm == IH.TRAVEL else (self.t_fin[tj] if tm == IH.WORK else self.now))
                    true_free = max(float(true_free), self.now) if np.isfinite(true_free) else self.now
                    self.stats["acc_n"] += 1
                    self.stats["acc_target_ok"] += int(state[1] == tj)
                    self.stats["acc_stale_target_ok"] += int(sj == tj)
                    self.stats["acc_free_err"] += abs(max(state[2], self.now) - true_free)
                    self.stats["acc_stale_free_err"] += abs(max(stref, self.now) - true_free)
                    if self.pb in ("PB", "PB-S"):
                        for x in st:
                            if known[x] and not evid[x]:
                                known[x] = False
                                self.stats["pred_started"] += 1
                                if not self.started[x]:
                                    self.stats["pred_wrong_started"] += 1
                        for x in dn:
                            if not self.done[x] and self.rel_k[n, x] and not self.done_k[n, x]:
                                self.stats["pred_wrong_done"] += 1
            free0 = np.empty(A)
            xy0 = np.empty((A, 2))
            pre = np.full((A, T + 1), -1, np.int64)
            npre = np.zeros(A, np.int64)
            for i in range(A):
                m, j, tref, xy, rem = self.believed(n, i, fresh=bool(comp_mask[i]))
                if self.pb == "CLAMP" and not comp_mask[i]:  # stale snapshot, but never available before now
                    tref = max(tref, self.now)
                if self.pb in ("PB", "PB-A") and int(i) in pred:
                    m, j, tref, xy = pred[int(i)]
                    tref = max(tref, self.now) if m != IH.IDLE else tref
                if m in (IH.TRAVEL, IH.WAIT, IH.WORK) and j >= 0:
                    free0[i], xy0[i] = tref, self.loc[j]
                    lst = [j] if (m != IH.WORK and known[j]) else []
                else:
                    free0[i], xy0[i] = tref, xy
                    lst = []
                if not scope_mask[i]:
                    lst = lst + [x for x in rem if known[x] and x not in lst]
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
                nplaced = IH.plan_dispatch(self.req, self.ab, self.loc, self.dur, self.inst.depot, self.v, active,
                                           free0, xy0, pre, npre, scope_mask, self.now, noise,
                                           int(rng_seed * 1000 + q), order, mem, cnt, out)
                key = (-nplaced, out[0], out[1])
                if best is None or key < best[0]:
                    best = (key, order[:nplaced].copy(), mem.copy(), cnt.copy())
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
            n_inc = IH.plan_dispatch(self.req, self.ab, self.loc, self.dur, self.inst.depot, self.v, active, free0,
                                     xy0, pre_i, npre_i, np.zeros(A, np.bool_), self.now, 0.0, 0, order, mem, cnt,
                                     out)
            inc_key = (-n_inc, out[0], out[1])
            if (best[0][0] > inc_key[0]) or (best[0][0] == inc_key[0] and best[0][1] >= inc_key[1] * (1 - self.hyst)):
                self.stats["plan_s"] += time.perf_counter() - t0
                return {}
            _, order_b, mem_b, cnt_b = best
            write = scope_mask & comp_mask if (getattr(self, "local_write", False) and n != A) else scope_mask
            routes = {int(i): [] for i in np.flatnonzero(write)}
            for j in order_b:
                for q in range(cnt_b[j]):
                    i = int(mem_b[j, q])
                    if i in routes:
                        routes[i].append(int(j))
            self.stats["plan_s"] += time.perf_counter() - t0
            return routes

    return IH, PBWorld


def job(args):
    setting, i, path, scen, R, p_loss, method, seed = args
    sys.path.insert(0, str(TD / "_snap"))
    from cbba_sota.hetero import Instance
    from dyn_env import Scenario, delay_calendar, realized_durations, release_times

    IH, PBW = make_cls()
    inst = Instance.from_pickle(path)
    sc = Scenario(scen["name"], release=scen["release"], dod=scen.get("dod", 0.5), horizon=IH.HORIZON[setting],
                  dur_sigma=scen["sigma"], p_delay=scen["p"], min_delay=scen["dmin"], max_delay=scen["dmax"])
    rel = release_times(sc, inst.n_tasks, i, seed)
    dur_real = realized_durations(sc, np.asarray(inst.dur), i, seed)
    delays = delay_calendar(sc, inst.n_agents, i, seed)
    t0 = time.perf_counter()
    lease = {}
    if "~G" in method:  # <variant>~G<grace>:<max>
        method_core, lp = method.split("~G")
        g, mx = lp.split(":")
        lease = dict(lease_grace=float(g), lease_max=float(mx))
    else:
        method_core = method
    base, _, marg = method_core.partition("@")
    ms_, mf_ = (0.5, 0.5) if not marg else (float(marg), float(marg))
    w = PBW(inst, rel, dur_real, delays, R, p_loss, "REP", comm_seed=seed, pb=base, m_start=ms_, m_fin=mf_, **lease)
    ok, ms = w.run()
    return dict(setting=setting, inst=i, scen=scen["name"], R=R, p_loss=p_loss, method=method, seed=seed,
                success=ok, makespan=ms, wall=time.perf_counter() - t0, **w.stats)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--procs", type=int, default=24)
    ap.add_argument("--settings", nargs="*", default=["MA-AT-25-5-50", "MA-AT-50-5-50", "SA-AT-50-5-50",
                                                        "SA-BT-50-5-50"])
    ap.add_argument("--scens", nargs="*", default=["RN12", "RN3"])
    ap.add_argument("--radii", nargs="*", type=float, default=[0.3, 0.2, 0.15, 0.1])
    ap.add_argument("--loss", type=float, default=0.2)
    ap.add_argument("--methods", nargs="*", default=["PB", "PB-S", "PB-A", "PB@1.5"])
    ap.add_argument("--out", default=str(HERE / "out" / "pb_pilot.jsonl"))
    args = ap.parse_args()
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ[k] = "1"
    import integ_headroom as IH

    data = TD / "_snap" / "data" / "hetero"
    jobs = []
    for s in args.settings:
        for i in range(args.n):
            path = str(data / s / "dev" / f"env_{i}.pkl")
            for sc in args.scens:
                for seed in range(args.seeds):
                    for R in args.radii:
                        for m in args.methods:
                            jobs.append((s, i, path, IH.SCENS[sc], R, args.loss, m, seed))
    jobs.sort(key=lambda j: (j[4], j[0] != "MA-AT-25-5-50"))
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
