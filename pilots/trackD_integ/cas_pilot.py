"""Track D integrated proposer, pilot P2 (2026-09-28): contact-aware scope (CAS) vs the replicated solver (REP).

Diagnosis from integ_headroom.jsonl (pilots/trackD): at R <= r_c the replicated solver + gossip (REP) is the best
baseline, but it writes 10-100x more route versions than global-comm re-planning and makes 10-75 wasted trips per
episode (robots arriving at a task another component already started). REP plans every robot as if a new route
reached it at once; an out-of-contact robot actually keeps executing its last adopted route until some node
carrying a newer version meets it. Frozen-scope variants (LOC-F here, O in trackD_comm) assume it NEVER re-plans and
lose to REP.

CAS plans like REP (every component leader plans ALL robots on its merged belief; newest version wins) but models an
out-of-component robot b as following its believed remaining route up to a predicted re-contact time t_c(b), and as
free (re-plannable) after that. REP is t_c = now; frozen scope is t_c = inf.
  CAS-L     t_c(b) = now + max(TAU_MIN, now - t_heard(b))    (elapsed-silence predictor)
  CAS-INF   t_c(b) = inf, but b stays in scope (it may get NEW tasks appended after its believed route)
  CAS-1     keep b's head + next 1 believed task (a fixed-depth rule; the simple-rule collapse check)
Same planner, leases, idle rally rule, channel and CRN as integ_headroom.py; only the out-of-component prefix differs.

Usage: .venv/bin/python pilots/trackD_integ/cas_pilot.py --procs 24
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

TAU_MIN = 1.0


def make_world_cls():
    import integ_headroom as IH

    class CASWorld(IH.World):
        def __init__(self, *a, cas="REP", **kw):
            super().__init__(*a, **kw)
            self.cas = cas
            self.stats["prefix_tasks"] = 0
            self.stats["prefix_calls"] = 0

        def _prefix(self, n, i, rem, free_i, xy_i, known):
            """Believed remaining tasks of out-of-component agent i that it will still execute before re-contact."""
            rem = [x for x in rem if known[x]]
            if self.cas == "REP" or not rem:
                return []
            if self.cas == "CAS-INF":
                return rem
            if self.cas == "CAS-1":
                return rem[:1]
            # CAS-L: elapsed-silence predictor of the re-contact time
            t_heard = self.rec_k[n, i] * IH.TICK
            t_c = self.now + max(TAU_MIN, self.now - t_heard)
            out, t, p = [], free_i, np.asarray(xy_i, float)
            for x in rem:
                t += float(np.linalg.norm(self.loc[x] - p)) / self.v
                if t >= t_c:
                    break
                out.append(x)
                t += float(self.dur[x])
                p = self.loc[x]
            return out

        def plan(self, n, scope_mask, comp_mask, rng_seed):
            A, T = self.A, self.T
            t0 = time.perf_counter()
            known = self.rel_k[n] & ~self.done_k[n] & ~self.start_k[n]
            free0 = np.empty(A)
            xy0 = np.empty((A, 2))
            pre = np.full((A, T + 1), -1, np.int64)
            npre = np.zeros(A, np.int64)
            for i in range(A):
                m, j, tref, xy, rem = self.believed(n, i, fresh=bool(comp_mask[i]))
                if m in (IH.TRAVEL, IH.WAIT, IH.WORK) and j >= 0:
                    free0[i], xy0[i] = tref, self.loc[j]
                    lst = [j] if (m != IH.WORK and known[j]) else []
                else:
                    free0[i], xy0[i] = tref, xy
                    lst = []
                if not scope_mask[i]:
                    lst = lst + [x for x in rem if known[x] and x not in lst]
                elif not comp_mask[i]:  # CAS: out-of-component agent keeps its believed route until re-contact
                    pf = self._prefix(n, i, [x for x in rem if x not in lst], free0[i], xy0[i], known)
                    self.stats["prefix_tasks"] += len(pf)
                    self.stats["prefix_calls"] += 1
                    lst = lst + pf
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
            routes = {int(i): [] for i in np.flatnonzero(scope_mask)}
            for j in order_b:
                for q in range(cnt_b[j]):
                    i = int(mem_b[j, q])
                    if i in routes:
                        routes[i].append(int(j))
            self.stats["plan_s"] += time.perf_counter() - t0
            return routes

    return IH, CASWorld


def job(args):
    setting, i, path, scen, R, p_loss, method, seed = args
    sys.path.insert(0, str(TD / "_snap"))
    from cbba_sota.hetero import Instance
    from dyn_env import Scenario, delay_calendar, realized_durations, release_times

    IH, CASWorld = make_world_cls()
    inst = Instance.from_pickle(path)
    sc = Scenario(scen["name"], release=scen["release"], dod=scen.get("dod", 0.5), horizon=IH.HORIZON[setting],
                  dur_sigma=scen["sigma"], p_delay=scen["p"], min_delay=scen["dmin"], max_delay=scen["dmax"])
    rel = release_times(sc, inst.n_tasks, i, seed)
    dur_real = realized_durations(sc, np.asarray(inst.dur), i, seed)
    delays = delay_calendar(sc, inst.n_agents, i, seed)
    t0 = time.perf_counter()
    lease, core = {}, method
    if "~G" in method:  # <method>~G<grace>:<max>  equal-infrastructure lease tuning
        core, lp = method.split("~G")
        g, mx = lp.split(":")
        lease = dict(lease_grace=float(g), lease_max=float(mx))
    if core.startswith("CAS") or core == "REP":
        w = CASWorld(inst, rel, dur_real, delays, R, p_loss, "REP", comm_seed=seed, cas=core, **lease)
    else:
        w = IH.World(inst, rel, dur_real, delays, R, p_loss, core, comm_seed=seed, **lease)
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
    ap.add_argument("--methods", nargs="*", default=["CEN-F", "REP", "CAS-L", "CAS-INF", "CAS-1"])
    ap.add_argument("--out", default=str(HERE / "out" / "cas_pilot.jsonl"))
    args = ap.parse_args()
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ[k] = "1"
    import integ_headroom as IH

    data = TD / "_snap" / "data" / "hetero"
    done = set()
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    if outp.exists():
        for line in outp.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done.add((r["setting"], r["inst"], r["scen"], r["R"], r["method"], r["seed"]))
    jobs = []
    for s in args.settings:
        for i in range(args.n):
            path = str(data / s / "dev" / f"env_{i}.pkl")
            for sc in args.scens:
                for seed in range(args.seeds):
                    if (s, i, sc, float("inf"), "FULL", seed) not in done:
                        jobs.append((s, i, path, IH.SCENS[sc], np.inf, 0.0, "FULL", seed))
                    for R in args.radii:
                        for m in args.methods:
                            if (s, i, sc, R, m, seed) not in done:
                                jobs.append((s, i, path, IH.SCENS[sc], R, args.loss, m, seed))
    # longest first: small radii and 25-agent runs re-plan most
    jobs.sort(key=lambda j: (j[4] if np.isfinite(j[4]) else 9.0, j[0] != "MA-AT-25-5-50"))
    print(len(jobs), "jobs", flush=True)
    t0 = time.perf_counter()
    with open(outp, "a") as f, mp.get_context("spawn").Pool(args.procs) as pool:
        for k, r in enumerate(pool.imap_unordered(job, jobs, chunksize=1)):
            f.write(json.dumps(r, default=float) + "\n")
            f.flush()
            if (k + 1) % 100 == 0:
                print(f"{k + 1}/{len(jobs)} {time.perf_counter() - t0:.0f}s", flush=True)
    print("done", time.perf_counter() - t0, flush=True)


if __name__ == "__main__":
    main()
