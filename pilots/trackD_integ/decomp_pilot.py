"""Track D integrated proposer, pilot P3 (2026-09-28): where does the replicated solver (REP) lose against FULL?

REP (every component leader re-plans all robots on its merged belief, routes spread by store-carry-forward) is the
strongest comm-limited baseline in integ_headroom.jsonl and in pilot P2 (cas_pilot.py). Its remaining gap to FULL
(instant global comm, same planner) at R <= r_c bounds what ANY planning-side mechanism (consensus, repair scope,
slack) can add. This pilot splits that gap with knowledge oracles that change only what nodes KNOW, never what reaches
a robot (route versions still travel only by contact):

  REP       as in integ_headroom.py
  REP-RO    release oracle: every node learns a released task at its release time (dispatch downlink to all)
  REP-KO    knowledge oracle: every node knows every release, start, finish and every robot's current record
            (global state estimation), but new routes still reach a robot only by contact
  REP-SO    task-status oracle: every node knows every task start and finish (nothing else)
  REP-AO    teammate-state oracle: every node knows every robot's current record (mode, target, ETA, position,
            remaining route) but task starts/finishes only by gossip
  FULL      global knowledge AND global route delivery (reference)

REP-KO / FULL = the cost of actuation (robots out of reach keep their old routes): no belief/consensus mechanism can
recover it, only motion (rendezvous, data mules) or a longer range. REP / REP-KO = the cost of stale knowledge that a
better information layer could at most recover. REP-RO isolates task-release latency.

Usage: .venv/bin/python pilots/trackD_integ/decomp_pilot.py --procs 24
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


def make_cls():
    import integ_headroom as IH

    class OracleWorld(IH.World):
        def __init__(self, *a, oracle="none", **kw):
            super().__init__(*a, **kw)
            self.oracle = oracle

        def merge(self, lab):
            super().merge(lab)
            if self.oracle.startswith("beacon"):
                # low-rate long-range state beacon: every P time units each robot's current record reaches every
                # node with probability 1 - q (i.i.d., CRN-keyed by (comm seed, tick)); nothing else is carried
                _, P, q = self.oracle.split(":")
                every = max(1, int(round(float(P) / IH.TICK)))
                if self.k % every == 0:
                    ok = np.random.default_rng([self.comm_seed, 99, self.k]).random(self.A) >= float(q)
                    for i in np.flatnonzero(ok):
                        self.rec_k[:, i] = np.maximum(self.rec_k[:, i], self.k)
            if self.oracle == "release":
                self.rel_k[:] = self.rel_k.any(0)
            elif self.oracle == "status":
                self.done_k[:] = self.done_k.any(0)
                self.start_k[:] = self.start_k.any(0)
            elif self.oracle == "agent":
                self.rec_k[:] = self.rec_k.max(0)
            elif self.oracle == "know":
                self.rel_k[:] = self.rel_k.any(0)
                self.done_k[:] = self.done_k.any(0)
                self.start_k[:] = self.start_k.any(0)
                self.rec_k[:] = self.rec_k.max(0)

        def run(self):
            # the station learns releases inside run(); make the oracle see them in the same tick
            return super().run()

        def physics_until(self, t_end):
            super().physics_until(t_end)
            if self.oracle in ("release", "know"):
                self.rel_k[:, self.rel <= t_end + 1e-12] = True

    return IH, OracleWorld


def job(args):
    setting, i, path, scen, R, p_loss, method, seed = args
    sys.path.insert(0, str(TD / "_snap"))
    from cbba_sota.hetero import Instance
    from dyn_env import Scenario, delay_calendar, realized_durations, release_times

    IH, OW = make_cls()
    inst = Instance.from_pickle(path)
    sc = Scenario(scen["name"], release=scen["release"], dod=scen.get("dod", 0.5), horizon=IH.HORIZON[setting],
                  dur_sigma=scen["sigma"], p_delay=scen["p"], min_delay=scen["dmin"], max_delay=scen["dmax"])
    rel = release_times(sc, inst.n_tasks, i, seed)
    dur_real = realized_durations(sc, np.asarray(inst.dur), i, seed)
    delays = delay_calendar(sc, inst.n_agents, i, seed)
    t0 = time.perf_counter()
    oracle = {"REP": "none", "REP-RO": "release", "REP-KO": "know", "REP-SO": "status", "REP-AO": "agent"}.get(method)
    if method.startswith("REP-B"):  # REP-B<P>:<q>  e.g. REP-B1:0.2
        P, q = method[5:].split(":")
        oracle = f"beacon:{P}:{q}"
    lease = {}
    if "~G" in method:  # <BASE>~G<grace>:<max>, BASE in {FULL, CEN-F, REP}: equal-infrastructure lease tuning
        base_m, lp = method.split("~G")
        g, mx = lp.split(":")
        w = IH.World(inst, rel, dur_real, delays, R, p_loss, base_m, comm_seed=seed, lease_grace=float(g),
                     lease_max=float(mx))
        ok, ms = w.run()
        return dict(setting=setting, inst=i, scen=scen["name"], R=R, p_loss=p_loss, method=method, seed=seed,
                    success=ok, makespan=ms, wall=time.perf_counter() - t0, **w.stats)
    if method.startswith("REP-G"):  # REP-G<grace>:<max>  lease variants (partner-wait slack)
        g, mx = method[5:].split(":")
        lease = dict(lease_grace=float(g), lease_max=float(mx))
        oracle = "none"
    if oracle is None:
        w = IH.World(inst, rel, dur_real, delays, R, p_loss, method, comm_seed=seed)
    elif lease:
        w = OW(inst, rel, dur_real, delays, R, p_loss, "REP", comm_seed=seed, oracle=oracle, **lease)
    else:
        w = OW(inst, rel, dur_real, delays, R, p_loss, "REP", comm_seed=seed, oracle=oracle)
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
    ap.add_argument("--scens", nargs="*", default=["RN12"])
    ap.add_argument("--radii", nargs="*", type=float, default=[0.2, 0.15, 0.1])
    ap.add_argument("--loss", type=float, default=0.2)
    ap.add_argument("--methods", nargs="*", default=["REP-RO", "REP-KO"])
    ap.add_argument("--out", default=str(HERE / "out" / "decomp_pilot.jsonl"))
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
                    for m in args.methods:
                        if m.startswith("FULL"):
                            jobs.append((s, i, path, IH.SCENS[sc], np.inf, 0.0, m, seed))
                    for R in args.radii:
                        for m in args.methods:
                            if not m.startswith("FULL"):
                                jobs.append((s, i, path, IH.SCENS[sc], R, args.loss, m, seed))
    jobs.sort(key=lambda j: (j[4] if np.isfinite(j[4]) else 9.0, j[0] != "MA-AT-25-5-50"))
    print(len(jobs), "jobs", flush=True)
    t0 = time.perf_counter()
    outp = Path(args.out)
    with open(outp, "w") as f, mp.get_context("spawn").Pool(args.procs) as pool:
        for k, r in enumerate(pool.imap_unordered(job, jobs, chunksize=1)):
            f.write(json.dumps(r, default=float) + "\n")
            f.flush()
            if (k + 1) % 100 == 0:
                print(f"{k + 1}/{len(jobs)} {time.perf_counter() - t0:.0f}s", flush=True)
    print("done", time.perf_counter() - t0, flush=True)


if __name__ == "__main__":
    main()
