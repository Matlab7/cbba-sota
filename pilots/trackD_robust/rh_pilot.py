"""Track D / robustness angle, pilot P1: how much headroom does a competent REACTIVE re-planner leave?

Open-loop plans lose 11-16% under duration noise (trackD pilot). The deciding question for a robustness-first
design is how much of that a rolling-horizon ALNS re-planner recovers WITHOUT anticipation, and how far it stays
from re-planning with perfect foresight. Whatever gap remains is the most any proactive/robust mechanism
(SAA, buffers, robust key order, leases) could add on top of the strongest reactive baseline.

Execution semantics (= TaskEnv.execute_by_route for key-ordered minimal covers, checked against the env rows of
pilots/trackD/out/rows.jsonl): an agent departs to its next task when its current task finishes (at t=0 from the
depot), travel follows the LoRR-2026 stall calendar of dyn_env (same CRN keys), a task starts when its last member
arrives and lasts its REALIZED duration, an agent with an empty queue heads home, the makespan is the last return.

Re-planning (RH): at every task-finish event, before the freed agents depart. A task is COMMITTED once any member
has departed to it; its coalition is then frozen and its members keep it in their queues. Every uncommitted task
is re-planned by the ALNS from the predicted state: agent i is ready at the predicted finish of its committed chain
(forward pass over committed tasks with the predictor) at that position; the kernel copy robust_rh_kernels.py takes the
first leg as dout[i, t] = ready_i + travel(pos_i, t). Warm start = previous plan restricted to the free tasks.
The combined order (committed tasks by old key, then the new plan) is a global key order, so execution is
deadlock-free.

Predictors (information the planner uses; execution is always realized):
  nom   nominal durations, nominal travel, causal ETA (stall so far, none ahead) -- what the published RL sees
  ce    calibrated means: travel x kappa, kappa = 1/(1 - stall fraction of the delay model); mean durations
  orc   perfect foresight of durations (+ exact in-flight arrivals); free-task travel still x kappa
Methods: OL-nom (cached trackD plan, validation), OL-<p> (t=0 plan with predictor p, open loop), RH-<p>.
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
TD = HERE.parent / "trackD"
SNAP = TD / "_snap"
sys.path.insert(0, str(SNAP))
sys.path.insert(0, str(TD))
sys.path.insert(0, str(HERE))  # first: never pick up a same-named module of another pilot

from dyn_env import Scenario, delay_calendar, realized_durations  # noqa: E402

MAX_TIME = 200.0
LORR = dict(p_delay=0.01, min_delay=1, max_delay=4)
MOD = dict(p_delay=0.05, min_delay=1, max_delay=10)
SEV = dict(p_delay=0.05, min_delay=10, max_delay=50)
SCEN = {
    "N2-dur.3/causal": Scenario("N2-dur.3/causal", obs="causal", dur_sigma=0.3),
    "N3-mod/causal": Scenario("N3-mod/causal", obs="causal", dur_sigma=0.3, **MOD),
    "N4-sev/causal": Scenario("N4-sev/causal", obs="causal", dur_sigma=0.6, **SEV),
}


def kappa_of(sc: Scenario) -> float:
    if sc.p_delay <= 0:
        return 1.0
    ed = 0.5 * (sc.min_delay + sc.max_delay)
    frac = ed / (1.0 / sc.p_delay - 1.0 + ed)  # renewal: gap of geometric(p)-1 free ticks, then ed stalled ticks
    return 1.0 / (1.0 - frac)


_A = None


def alns_mod():
    """Snapshot ALNS with the state-start kernel copy patched in (single worker, v1 config, no pool)."""
    global _A
    if _A is None:
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


class Episode:
    def __init__(self, inst, dur_real, delays, kappa):
        self.inst = inst
        self.T, self.A = inst.n_tasks, inst.n_agents
        self.loc, self.depot, self.v = inst.loc, inst.depot, inst.speed
        self.tt, self.da = inst.tt, inst.da
        self.dur_nom = np.asarray(inst.dur, float)
        self.dur_real = np.asarray(dur_real, float)
        self.delays = delays
        self.kappa = kappa
        T, A = self.T, self.A
        self.committed = np.zeros(T, bool)
        self.done = np.zeros(T, bool)
        self.mem = [()] * T
        self.arrived = [dict() for _ in range(T)]
        self.start = np.full(T, np.nan)
        self.finish = np.full(T, np.nan)
        self.plan_mem = [()] * T
        self.key = np.zeros(T)
        self.q = [[] for _ in range(A)]
        self.tgt = np.full(A, -2)  # task id, -1 heading home, -2 at home
        self.origin = [("depot", i) for i in range(A)]  # where the current leg started
        self.leg = [None] * A  # (dep, tau, origin_xy, target_xy, arr)
        self.ver = np.zeros(A, np.int64)
        self.home_t = np.zeros(A)
        self.heap = []
        self.seq = 0
        self.n_replans = 0
        self.solver_s = 0.0
        self.changed = 0  # re-plans whose new plan differs from the warm start on some free task

    # ---- physics ------------------------------------------------------------------------------------------
    def arrive(self, i, dep, tau):
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

    def stalled(self, i, a, b):
        starts, ends = self.delays[i]
        if starts.size == 0 or b <= a:
            return 0.0
        return float(np.clip(np.minimum(ends, b) - np.maximum(starts, a), 0.0, None).sum())

    def pos_now(self, i, t):
        """(kind, value) of agent i's position at t when it is not bound to a task: depot, task or point."""
        if self.tgt[i] == -2:
            return ("depot", i)
        if self.tgt[i] == -1:
            dep, tau, o, d, arr = self.leg[i]
            if t >= arr:
                return ("depot", i)
            moved = (t - dep) - self.stalled(i, dep, t)
            frac = min(max(moved / tau, 0.0), 1.0) if tau > 0 else 1.0
            return ("point", o + frac * (d - o))
        j = self.tgt[i]
        return ("task", j)  # finished task (freed agent)

    def xy(self, p):
        kind, v = p
        return self.depot[v] if kind == "depot" else self.loc[v] if kind == "task" else v

    def travel(self, i, p, j):
        """Nominal travel time from position p to task j (bit-exact for depot/task origins)."""
        kind, v = p
        if kind == "depot":
            return self.da[i, j]
        if kind == "task":
            return self.tt[v, j]
        return float(np.linalg.norm(v - self.loc[j])) / self.v

    def push(self, t, prio, kind, a, b=0):
        heapq.heappush(self.heap, (t, prio, self.seq, kind, a, b))
        self.seq += 1

    def dispatch(self, i, t):
        p = self.pos_now(i, t) if self.tgt[i] < 0 else ("task", int(self.tgt[i]))
        self.ver[i] += 1
        if self.q[i]:
            j = self.q[i].pop(0)
            if not self.committed[j]:
                self.committed[j] = True
                self.mem[j] = tuple(self.plan_mem[j])
            assert i in self.mem[j], (i, j, self.mem[j])
            tau = self.travel(i, p, j)
            arr = self.arrive(i, t, tau)
            self.tgt[i] = j
            self.leg[i] = (t, tau, self.xy(p), self.loc[j], arr)
            self.push(arr, 1, "ARR", i, j)
        else:
            kind, v = p
            if kind == "depot" and self.tgt[i] == -2:
                return  # idle at home
            tau = self.da[i, v] if kind == "task" else float(np.linalg.norm(self.xy(p) - self.depot[i])) / self.v
            arr = self.arrive(i, t, tau)
            self.tgt[i] = -1
            self.leg[i] = (t, tau, self.xy(p), self.depot[i], arr)
            self.push(arr, 2, "HOME", i, int(self.ver[i]))

    # ---- plan bookkeeping ---------------------------------------------------------------------------------
    def set_plan(self, tasks, members, order):
        """Install a plan for ``tasks`` (members per task, order = task ids in key order) and rebuild queues."""
        base = float(self.key.max()) + 1.0 if self.committed.any() else 0.0
        for r, j in enumerate(order):
            self.key[j] = base + r
        for j, m in zip(tasks, members):
            self.plan_mem[j] = tuple(m)
        for i in range(self.A):
            chain = [j for j in self.q[i] if self.committed[j]]
            self.q[i] = chain + [j for j in order if i in self.plan_mem[j]]

    # ---- run ----------------------------------------------------------------------------------------------
    def run(self, planner=None):
        """``planner(ep, t)`` -> (tasks, members, order) for the uncommitted tasks, or None (open loop)."""
        for i in range(self.A):
            self.dispatch(i, 0.0)
        while self.heap:
            t = self.heap[0][0]
            if t > MAX_TIME:
                break
            freed = []
            fin_batch = False
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
            if fin_batch and planner is not None and not self.committed.all():
                idle_before = [i for i in range(self.A) if self.tgt[i] < 0]
                t0 = time.perf_counter()
                res = planner(self, t)
                self.solver_s += time.perf_counter() - t0
                self.n_replans += 1
                if res is not None:
                    self.set_plan(*res)
                for i in idle_before:  # idle / homebound agents with new work leave now
                    if self.q[i]:
                        self.dispatch(i, t)
            for i in sorted(set(freed)):
                self.dispatch(i, t)
        ok = bool(self.done.all()) and bool((self.tgt == -2).all())
        ms = float(self.home_t.max()) if ok else MAX_TIME
        return {"makespan": ms, "success": ok and ms < MAX_TIME, "completion": float(self.done.mean()),
                "n_replans": self.n_replans, "solver_s": self.solver_s, "changed": self.changed}


# ---- predictors and the rolling planner -----------------------------------------------------------------------
def predicted_state(ep: Episode, t: float, pred: str):
    """Forward pass over committed unfinished tasks with the predictor; returns ready [A], positions [A]."""
    kap = 1.0 if pred == "nom" else ep.kappa
    durp = ep.dur_real if pred == "orc" else ep.dur_nom
    A = ep.A
    free_t = np.full(A, t)
    pos = [ep.pos_now(i, t) if ep.tgt[i] < 0 or ep.done[ep.tgt[i]] else None for i in range(A)]
    C = [j for j in np.argsort(ep.key, kind="stable") if ep.committed[j] and not ep.done[j]]
    for j in C:
        if np.isfinite(ep.start[j]):
            s = ep.start[j]
            f = s + ep.dur_real[j] if pred == "orc" else max(s + durp[j], t)
        else:
            s = -np.inf
            for m in ep.mem[j]:
                if m in ep.arrived[j]:
                    a = ep.arrived[j][m]
                elif ep.tgt[m] == j:
                    dep, tau, o, d, arr = ep.leg[m]
                    if pred == "orc":
                        a = arr
                    elif pred == "nom":
                        a = max(dep + tau + ep.stalled(m, dep, t), t)
                    else:
                        moved = (t - dep) - ep.stalled(m, dep, t)
                        a = t + max(tau - moved, 0.0) * kap
                else:
                    a = free_t[m] + ep.travel(m, pos[m], j) * kap
                s = max(s, a)
            f = s + durp[j]
        for m in ep.mem[j]:
            free_t[m] = f
            pos[m] = ("task", int(j))
    return free_t, pos


def make_planner(pred: str, iters: int, seed: int, cfg=None):
    A_ = alns_mod()
    from cbba_sota.hetero import Instance, Plan

    cfg = cfg or A_.ALNSConfig.v1(verify=False)
    calls = [0]

    def planner(ep: Episode, t: float):
        F = np.flatnonzero(~ep.committed)
        if F.size == 0:
            return None
        kap = 1.0 if pred == "nom" else ep.kappa
        durp = ep.dur_real if pred == "orc" else ep.dur_nom
        ready, pos = predicted_state(ep, t, pred)
        inst = ep.inst
        sub = Instance(req=inst.req[F], loc=inst.loc[F], dur=durp[F], ab=inst.ab, depot=inst.depot,
                       species=inst.species, speed=inst.speed / kap)
        dout = np.empty((ep.A, F.size))
        for i in range(ep.A):
            for f, j in enumerate(F):
                dout[i, f] = ready[i] + ep.travel(i, pos[i], j) * kap
        object.__setattr__(sub, "_dout", dout)
        warm = None
        if t > 0:
            warm = Plan([ep.plan_mem[j] for j in F], ep.key[F], ep.A)
        plan, st = A_.solve(sub, 60.0, seed=seed + calls[0], init_plan=warm, config=cfg, max_iters=iters)
        calls[0] += 1
        order = [int(F[f]) for f in plan.order() if plan.members[f]]
        if warm is not None and any(tuple(plan.members[f]) != tuple(warm.members[f]) for f in range(F.size)):
            ep.changed += 1
        return [int(j) for j in F], [plan.members[f] for f in range(F.size)], order

    return planner


# ---- jobs ------------------------------------------------------------------------------------------------------
def build(job):
    from cbba_sota.hetero import Instance

    sc = SCEN[job["scenario"]]
    inst = Instance.from_pickle(job["path"])
    dur_real = realized_durations(sc, np.asarray(inst.dur, float), job["inst_key"], job["noise_seed"])
    delays = delay_calendar(sc, inst.n_agents, job["inst_key"], job["noise_seed"])
    return inst, dur_real, delays, kappa_of(sc)


def run_job(job):
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ[k] = "1"
    inst, dur_real, delays, kap = build(job)
    m = job["method"]
    key = {k: job[k] for k in ("setting", "inst", "scenario", "noise_seed", "method")}
    t0 = time.perf_counter()
    ep = Episode(inst, dur_real, delays, kap)
    if m == "OL-cached":
        routes = job["routes"]
        tasks = list(range(inst.n_tasks))
        members = [[] for _ in tasks]
        order_pos = {}
        from cbba_sota.hetero import Plan

        plan = Plan.from_routes(routes, inst.n_tasks, one_based=True)
        order = [int(j) for j in plan.order() if plan.members[j]]
        ep.set_plan(tasks, plan.members, order)
        res = ep.run(None)
        del members, order_pos
    else:
        kind, pred = m.split("-")
        planner = make_planner(pred, job["iters"], seed=1000 * job["noise_seed"] + 7)
        # t = 0 plan with the predictor (larger budget), then open loop or rolling
        ep0 = ep
        tasks, members, order = make_planner(pred, job["iters0"], seed=job["noise_seed"])(ep0, 0.0)
        ep.set_plan(tasks, members, order)
        res = ep.run(planner if kind == "RH" else None)
    return {**key, **res, "wall_s": time.perf_counter() - t0, "kappa": kap, "iters": job.get("iters"),
            "iters0": job.get("iters0")}


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--settings", nargs="+", default=["MA-AT-25-5-50", "MA-AT-50-5-50", "SA-AT-50-5-50",
                                                      "SA-BT-50-5-50"])
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--procs", type=int, default=28)
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--iters0", type=int, default=6000)
    ap.add_argument("--methods", nargs="+", default=["OL-cached", "OL-nom", "OL-ce", "OL-orc", "RH-nom", "RH-ce",
                                                     "RH-orc"])
    ap.add_argument("--scenarios", nargs="+", default=list(SCEN))
    ap.add_argument("--out", default=str(HERE / "out" / "rh_rows.jsonl"))
    args = ap.parse_args()
    sys.path.insert(0, str(SNAP))
    from cbba_sota.bench.configs import get

    plans = json.loads((TD / "out" / "alns_plans.json").read_text())
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
            path = str(st.instance_path("dev", i))
            for scn in args.scenarios:
                for ns in range(args.seeds):
                    for m in args.methods:
                        if scn.startswith("N2") and m.endswith("-ce"):
                            continue  # kappa = 1: identical to -nom
                        if (s, i, scn, ns, m) in done:
                            continue
                        jobs.append({"setting": s, "inst": i, "path": path, "inst_key": st.seed("dev", i),
                                     "scenario": scn, "noise_seed": ns, "method": m, "iters": args.iters,
                                     "iters0": args.iters0, "routes": plans.get(path, {}).get("routes")})
    jobs.sort(key=lambda j: j["method"].startswith("RH"), reverse=True)  # long jobs first
    print(f"{len(jobs)} jobs", flush=True)
    t0 = time.time()
    ctx = mp.get_context("spawn")
    with ctx.Pool(args.procs) as pool, out.open("a") as f:
        for k, r in enumerate(pool.imap_unordered(run_job, jobs)):
            f.write(json.dumps(r) + "\n")
            f.flush()
            if k % 50 == 0:
                print(f"{k}/{len(jobs)} {time.time() - t0:.0f}s", flush=True)
    print("done", time.time() - t0, flush=True)


if __name__ == "__main__":
    main()
