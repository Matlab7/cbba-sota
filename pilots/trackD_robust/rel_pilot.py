"""Track D / robustness angle, pilot P2: do STRUCTURAL events (online task release) create headroom that execution
noise does not, and how much per-event compute does the repair need?

Background (out/c2, summarize_c2.py): under LoRR delays and lognormal durations, open-loop execution of a key-ordered
plan is within ~2% of rolling re-planning at 3000 ALNS iterations per event, and re-planning at 30 iterations per
event is 3-5% WORSE than not re-planning (nervousness). Here tasks arrive online (release models of the trackD pilot:
R1 = the authors' batch demo, R2/R3 = Poisson with degree of dynamism 0.5/0.8, horizon = 0.5 x paper RL(g.)
makespan; release seed = noise seed, so the RL(g.) rows of pilots/trackD/out/rows.jsonl face the same arrivals).

Executor = rh_pilot.Episode (arrival of the last coalition member starts a task, LoRR delay calendars, realized
durations, empty queue -> head home, re-dispatch from the current point when new work arrives) plus release events.
The planner only sees released tasks.

Methods (predictor 'ce' = calibrated means, identical to 'nom' without delays):
  RH-ce@N   re-plan all released uncommitted tasks at every task finish AND every release epoch (N ALNS iterations)
  ES-ce@N   event-sparse: re-plan only at release epochs (structural events); between them the plan runs open loop
            (N = 0: insertion of the new tasks into the warm plan only, no search)
"""
from __future__ import annotations

import heapq
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rh_pilot as R  # noqa: E402  (sets up the _snap path)
from dyn_env import Scenario, delay_calendar, realized_durations, release_times  # noqa: E402

HORIZON = {"MA-AT-25-5-50": 25.0, "MA-AT-50-5-50": 15.0, "SA-AT-50-5-50": 20.0, "SA-BT-50-5-50": 12.0}


def scen(name, horizon):
    return {
        "R1-batch": Scenario("R1-batch", release="batch"),
        "R2-pois.5": Scenario("R2-pois.5", release="poisson", dod=0.5, horizon=horizon),
        "R3-pois.8": Scenario("R3-pois.8", release="poisson", dod=0.8, horizon=horizon),
        "R2+N3/causal": Scenario("R2+N3/causal", release="poisson", dod=0.5, horizon=horizon, obs="causal",
                                 dur_sigma=0.3, **R.MOD),
    }[name]


class RelEpisode(R.Episode):
    def __init__(self, inst, dur_real, delays, kappa, rel):
        super().__init__(inst, dur_real, delays, kappa)
        self.rel = np.asarray(rel, float)

    def released(self, t):
        return self.rel <= t + 1e-9

    def run(self, planner=None, on_finish=True):
        for i in range(self.A):
            self.dispatch(i, 0.0)
        for r in np.unique(self.rel[self.rel > 0]):
            self.push(float(r), -1, "REL", 0)
        while self.heap:
            t = self.heap[0][0]
            if t > R.MAX_TIME:
                break
            freed, fin_batch, rel_batch = [], False, False
            while self.heap and self.heap[0][0] == t:
                _, _, _, kind, a, b = heapq.heappop(self.heap)
                if kind == "FIN":
                    self.done[a] = True
                    self.finish[a] = t
                    freed += list(self.mem[a])
                    fin_batch = True
                elif kind == "ARR":
                    self.arrived[b][a] = t
                    if len(self.arrived[b]) == len(self.mem[b]):
                        self.start[b] = t
                        self.push(t + self.dur_real[b], 0, "FIN", b)
                elif kind == "HOME":
                    if self.ver[a] == b:
                        self.tgt[a] = -2
                        self.home_t[a] = t
                elif kind == "REL":
                    rel_batch = True
            open_tasks = (self.released(t) & ~self.committed).any()
            if planner is not None and open_tasks and (rel_batch or (fin_batch and on_finish)):
                idle_before = [i for i in range(self.A) if self.tgt[i] < 0]
                t0 = time.perf_counter()
                res = planner(self, t)
                self.solver_s += time.perf_counter() - t0
                self.n_replans += 1
                if res is not None:
                    self.set_plan(*res)
                for i in idle_before:
                    if self.q[i]:
                        self.dispatch(i, t)
            for i in sorted(set(freed)):
                self.dispatch(i, t)
        ok = bool(self.done.all()) and bool((self.tgt == -2).all())
        ms = float(self.home_t.max()) if ok else R.MAX_TIME
        return {"makespan": ms, "success": ok and ms < R.MAX_TIME, "completion": float(self.done.mean()),
                "n_replans": self.n_replans, "solver_s": self.solver_s, "changed": self.changed}


def make_planner(pred, iters, seed):
    A_ = R.alns_mod()
    from cbba_sota.hetero import Instance, Plan

    cfg = A_.ALNSConfig.v1(verify=False)
    calls = [0]

    def planner(ep, t):
        F = np.flatnonzero(ep.released(t) & ~ep.committed)
        if F.size == 0:
            return None
        kap = 1.0 if pred == "nom" else ep.kappa
        durp = ep.dur_nom
        ready, pos = R.predicted_state(ep, t, pred)
        inst = ep.inst
        sub = Instance(req=inst.req[F], loc=inst.loc[F], dur=durp[F], ab=inst.ab, depot=inst.depot,
                       species=inst.species, speed=inst.speed / kap)
        dout = np.empty((ep.A, F.size))
        for i in range(ep.A):
            for f, j in enumerate(F):
                dout[i, f] = ready[i] + ep.travel(i, pos[i], j) * kap
        object.__setattr__(sub, "_dout", dout)
        warm = Plan([ep.plan_mem[j] for j in F], ep.key[F], ep.A) if t > 0 else None
        plan, st = A_.solve(sub, 60.0, seed=seed + calls[0], init_plan=warm, config=cfg, max_iters=iters)
        calls[0] += 1
        order = [int(F[f]) for f in plan.order() if plan.members[f]]
        if warm is not None and any(tuple(plan.members[f]) != tuple(warm.members[f]) for f in range(F.size)
                                    if warm.members[f]):
            ep.changed += 1
        return [int(j) for j in F], [plan.members[f] for f in range(F.size)], order

    return planner


def run_job(job):
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ[k] = "1"
    from cbba_sota.hetero import Instance

    sc = scen(job["scenario"], HORIZON[job["setting"]])
    inst = Instance.from_pickle(job["path"])
    ns = job["noise_seed"]
    dur_real = realized_durations(sc, np.asarray(inst.dur, float), job["inst_key"], ns)
    delays = delay_calendar(sc, inst.n_agents, job["inst_key"], ns)
    rel = release_times(sc, inst.n_tasks, job["inst_key"], ns)
    kind, rest = job["method"].split("-", 1)
    pred, iters = rest.split("@")
    iters = int(iters)
    t0 = time.perf_counter()
    ep = RelEpisode(inst, dur_real, delays, R.kappa_of(sc), rel)
    tasks, members, order = make_planner(pred, job["iters0"], seed=ns)(ep, 0.0)
    ep.set_plan(tasks, members, order)
    res = ep.run(make_planner(pred, iters, seed=1000 * ns + 7), on_finish=(kind == "RH"))
    return {**{k: job[k] for k in ("setting", "inst", "scenario", "noise_seed", "method")}, **res,
            "wall_s": time.perf_counter() - t0, "iters0": job["iters0"], "n_rel_epochs": int(np.unique(rel[rel > 0]).size)}


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--settings", nargs="+", default=list(HORIZON))
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--procs", type=int, default=32)
    ap.add_argument("--iters0", type=int, default=20000)
    ap.add_argument("--methods", nargs="+", default=["RH-ce@3000", "RH-ce@30", "ES-ce@0", "ES-ce@30", "ES-ce@300",
                                                     "ES-ce@3000"])
    ap.add_argument("--scenarios", nargs="+", default=["R1-batch", "R2-pois.5", "R3-pois.8", "R2+N3/causal"])
    ap.add_argument("--out", default=str(HERE / "out" / "rel" / "rel.jsonl"))
    args = ap.parse_args()
    from cbba_sota.bench.configs import get

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            r = json.loads(line)
            done.add((r["setting"], r["inst"], r["scenario"], r["noise_seed"], r["method"]))
    jobs = []
    for s in args.settings:
        st = get(s)
        for i in range(args.n):
            for scn in args.scenarios:
                for ns in range(args.seeds):
                    for m in args.methods:
                        if (s, i, scn, ns, m) not in done:
                            jobs.append({"setting": s, "inst": i, "path": str(st.instance_path("dev", i)),
                                         "inst_key": st.seed("dev", i), "scenario": scn, "noise_seed": ns,
                                         "method": m, "iters0": args.iters0})
    jobs.sort(key=lambda j: j["method"].startswith("RH") and j["method"].endswith("3000"), reverse=True)
    print(f"{len(jobs)} jobs", flush=True)
    t0 = time.time()
    ctx = mp.get_context("spawn")
    with ctx.Pool(args.procs) as pool, out.open("a") as f:
        for k, r in enumerate(pool.imap_unordered(run_job, jobs)):
            f.write(json.dumps(r) + "\n")
            f.flush()
            if k % 100 == 0:
                print(f"{k}/{len(jobs)} {time.time() - t0:.0f}s", flush=True)
    print("done", time.time() - t0, flush=True)


if __name__ == "__main__":
    main()
