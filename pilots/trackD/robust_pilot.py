"""Track D robustness pilot: how much does re-planning buy under execution noise, and how much of that survives
range-limited communication?

Ground truth here is a plan-following discrete-event executor that uses the SAME noise realization as
``dyn_env.DynTaskEnv`` (realized durations, LoRR delay calendars, keyed by (noise seed, instance)); for a fixed
plan it reproduces the env's ``execute_by_route`` makespan (checked against pilots/trackD/out/rows.jsonl, ALNS-ol).
Coalitions are fixed minimal covers and every agent follows its route in global key order, so execution is
deadlock-free under any delay.

Methods (one run each per (setting, instance, scenario, noise seed); all start from the same nominal plan):
  OL            nominal ALNS plan (5 s, from out/alns_plans.json), open loop
  RS<N>         central event-triggered re-solve at every task finish: commit every task any agent is heading to /
                waiting at / working on, close that set over route prefixes, predict the committed part from the
                causal state (observed arrivals and starts, ETA = nominal + stall so far, nominal durations), then
                warm-started state-start ALNS (``rh_kernels``) for N iterations on the remaining tasks
  SAAk-OL/RS    best of 8 fresh nominal ALNS plans by mean makespan over 16 training noise draws (disjoint seeds)
  NOMk-OL/RS    best of the same 8 plans by nominal makespan (control for "more search")
  CRS<R>        central re-solve with local fallback: a station at (0.5, 0.5); only when the finishing coalition is
                in the station's multi-hop component (disk radius R among robots), and only over agents in that
                component; every task with a member outside it is frozen
  DRS<R>        component-local repair: the same computation in whatever component the finishing coalition is in
Knowledge simplification (favours CRS and DRS equally): predictions of frozen tasks use the true current state of
all agents. No message loss or latency is modelled; messages are not counted.

Usage: .venv/bin/python pilots/trackD/robust_pilot.py [--procs 32] [--part plans|main]
"""
from __future__ import annotations

import argparse
import dataclasses
import heapq
import importlib.util
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dyn_env  # noqa: E402  (puts the frozen _snap copy of cbba_sota on sys.path)
from dyn_env import Scenario, delay_calendar, realized_durations  # noqa: E402

import cbba_sota.solvers as _pkg  # noqa: E402

_spec = importlib.util.spec_from_file_location("cbba_sota.solvers._alns_kernels", HERE / "rh_kernels.py")
K = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = K
_spec.loader.exec_module(K)
_pkg._alns_kernels = K
from cbba_sota.hetero import Instance, Plan, evaluate  # noqa: E402
from cbba_sota.solvers import alns as ALNS  # noqa: E402

_orig_inst_arrays = ALNS._inst_arrays


def _inst_arrays(inst):
    base = _orig_inst_arrays(inst)
    A, T = inst.n_agents, inst.n_tasks
    ready = inst.__dict__.get("_ready", np.zeros(A))
    dout = inst.__dict__.get("_dout", inst.da)
    dhome = inst.__dict__.get("_dhome", np.zeros(A))
    f = lambda a: np.array(a, dtype=float, order="C")  # noqa: E731
    return base + (f(ready), f(dout), f(dhome))


ALNS._inst_arrays = _inst_arrays
CFG = dataclasses.replace(ALNS.ALNSConfig(), verify=False)

OUT = HERE / "out"
SETTINGS = ("MA-AT-25-5-50", "MA-AT-50-5-50", "SA-AT-50-5-50", "SA-BT-50-5-50")
MOD = dict(p_delay=0.05, min_delay=1, max_delay=10)
SCEN = {"N2": Scenario("N2-dur.3/causal", obs="causal", dur_sigma=0.3),
        "N3": Scenario("N3-mod/causal", obs="causal", dur_sigma=0.3, **MOD)}
RADII = (0.1, 0.15, 0.2, 0.3)
STATION = np.array([0.5, 0.5])
N_TRAIN = 16  # SAA training noise draws (noise seeds 1000..1015, disjoint from the test seeds 0..2)
K_PLANS = 8


# ---------------------------------------------------------------------------------------------------------------
def arrive(starts, ends, dep, tau):
    """dyn_env.DynTaskEnv._arrive (LoRR delay walk)."""
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


def stalled(starts, ends, a, b):
    if starts.size == 0 or b <= a:
        return 0.0
    return float(np.clip(np.minimum(ends, b) - np.maximum(starts, a), 0.0, None).sum())


class World:
    """Static data of one instance: points = tasks 0..T-1, then depots of agents T..T+A-1."""

    def __init__(self, inst: Instance):
        self.inst = inst
        T, A = inst.n_tasks, inst.n_agents
        self.T, self.A = T, A
        pts = np.vstack([inst.loc, inst.depot])
        self.pts = pts
        n = len(pts)
        D = np.zeros((n, n))
        for i in range(n):  # pairwise norm exactly as the env
            for j in range(n):
                D[i, j] = np.linalg.norm(pts[i] - pts[j]) / inst.speed
        self.D = D

    def home(self, i):
        return self.T + i


class Sim:
    """Plan-following executor. ``replan(sim, t, finishers)`` may rewrite routes/members of uncommitted tasks."""

    def __init__(self, world: World, plan: Plan, dur_real, delays, replan=None):
        self.w, self.dur_real, self.delays, self.replan = world, dur_real, delays, replan
        T, A = world.T, world.A
        self.members = [tuple(m) for m in plan.members]
        self.keys = plan.keys.astype(float).copy()
        self.route = [list(r) for r in plan.routes()]  # remaining queue after the current target
        self.target = [None] * A  # task id, -1 = home, None = idle (not moving)
        self.pos = [world.home(i) for i in range(A)]  # point index of last position
        self.dep = np.zeros(A)
        self.tau = np.zeros(A)
        self.arr_true = np.zeros(A)
        self.arrived = [True] * A
        self.home_arr = np.zeros(A)
        self.arrivals = [dict() for _ in range(T)]
        self.start = np.full(T, np.nan)
        self.finish = np.full(T, np.nan)
        self.done = np.zeros(T, bool)
        self.heap = []
        self.n = 0
        self.stats = {"events": 0, "replans": 0, "changed": 0, "solve_s": 0.0, "iters": 0, "frozen_frac": [],
                      "skipped": 0}

    def push(self, t, kind, x):
        self.n += 1
        heapq.heappush(self.heap, (t, kind, self.n, x))

    def depart(self, i, t):
        if self.route[i]:
            j = self.route[i].pop(0)
            dest = j
        else:
            j, dest = -1, self.w.home(i)
        tau = self.w.D[self.pos[i], dest]
        s, e = self.delays[i]
        self.target[i], self.dep[i], self.tau[i] = j, t, tau
        self.arr_true[i] = arrive(s, e, t, tau)
        self.arrived[i] = False
        self.push(self.arr_true[i], 1, i)

    def run(self):
        for i in range(self.w.A):
            self.depart(i, 0.0)
        while self.heap:
            t, kind, _, x = heapq.heappop(self.heap)
            if kind == 1:  # arrival of agent x
                i = x
                j = self.target[i]
                self.arrived[i] = True
                if j == -1:
                    self.pos[i] = self.w.home(i)
                    self.home_arr[i] = t
                    if self.route[i]:  # re-planned while heading home
                        self.depart(i, t)
                    else:
                        self.target[i] = None
                    continue
                self.pos[i] = j
                self.arrivals[j][i] = t
                if len(self.arrivals[j]) == len(self.members[j]):
                    self.start[j] = t
                    self.push(t + self.dur_real[j], 0, j)
            else:  # task x finished
                j = x
                self.done[j] = True
                self.finish[j] = t
                self.stats["events"] += 1
                fin = list(self.members[j])
                for m in fin:
                    self.target[m] = None
                if self.replan is not None and not self.done.all():
                    self.replan(self, t, fin)
                for m in fin:
                    self.depart(m, t)
                for i in range(self.w.A):  # idle at home and newly given work
                    if self.target[i] is None and self.route[i]:
                        self.depart(i, t)
        assert self.done.all(), "deadlock"
        return float(self.home_arr.max())

    # --- causal state ----------------------------------------------------------------------------------------
    def eta(self, i, t):
        if self.arrived[i]:
            return self.arr_true[i]
        s, e = self.delays[i]
        return max(self.dep[i] + self.tau[i] + stalled(s, e, self.dep[i], t), t)

    def position(self, i, t):
        if self.target[i] is None or self.arrived[i]:
            return self.w.pts[self.pos[i]]
        dest = self.target[i] if self.target[i] >= 0 else self.w.home(i)
        s, e = self.delays[i]
        moved = (t - self.dep[i]) - stalled(s, e, self.dep[i], t)
        frac = min(max(moved / self.tau[i], 0.0), 1.0) if self.tau[i] > 0 else 1.0
        p0, p1 = self.w.pts[self.pos[i]], self.w.pts[dest]
        return p0 + (p1 - p0) * frac


def components(sim: Sim, t, R, with_station):
    """Label of each agent's multi-hop component (and the station's label, or -1)."""
    A = sim.w.A
    P = np.array([sim.position(i, t) for i in range(A)])
    if with_station:
        P = np.vstack([P, STATION])
    n = len(P)
    adj = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=2) <= R
    lab = -np.ones(n, int)
    c = 0
    for s in range(n):
        if lab[s] >= 0:
            continue
        stack = [s]
        lab[s] = c
        while stack:
            u = stack.pop()
            for v in np.flatnonzero(adj[u] & (lab < 0)):
                lab[v] = c
                stack.append(v)
        c += 1
    return lab[:A], (lab[A] if with_station else -1)


# ---------------------------------------------------------------------------------------------------------------
class Replanner:
    def __init__(self, iters, seed=0, mode="ideal", R=None):
        self.iters, self.seed, self.mode, self.R = iters, seed, mode, R

    def __call__(self, sim: Sim, t, fin):
        w = sim.w
        A, T = w.A, w.T
        if self.mode == "ideal":
            S = set(range(A))
        else:
            lab, st = components(sim, t, self.R, self.mode == "central")
            c = lab[fin[0]]
            if self.mode == "central" and c != st:
                sim.stats["skipped"] += 1
                return
            S = set(np.flatnonzero(lab == c).tolist())
        # full remaining route per agent (current target first)
        full = []
        for i in range(A):
            r = []
            if self.mode and sim.target[i] is not None and sim.target[i] >= 0 and not sim.done[sim.target[i]]:
                r.append(sim.target[i])
            full.append(r + sim.route[i])
        open_tasks = [j for j in range(T) if not sim.done[j]]
        C = set(sim.target[i] for i in range(A) if sim.target[i] is not None and sim.target[i] >= 0
                and not sim.done[sim.target[i]])
        C |= {j for j in open_tasks if any(m not in S for m in sim.members[j])}
        pos_in = [{j: k for k, j in enumerate(full[i])} for i in range(A)]
        stack = list(C)
        while stack:  # prefix closure
            c = stack.pop()
            for m in sim.members[c]:
                for j in full[m][:pos_in[m][c]]:
                    if j not in C:
                        C.add(j)
                        stack.append(j)
        F = [j for j in open_tasks if j not in C]
        sim.stats["frozen_frac"].append(1.0 - len(F) / max(len(open_tasks), 1))
        if not F:
            return
        # predicted state after the committed part
        pfree = np.full(A, t)
        ppos = list(sim.pos)
        for i in range(A):
            if sim.target[i] == -1 and not sim.arrived[i]:
                pfree[i], ppos[i] = sim.eta(i, t), w.home(i)
        dnom = w.inst.dur
        for c in sorted(C, key=lambda j: (sim.keys[j], j)):
            if not np.isnan(sim.start[c]):
                s = sim.start[c]
                f = max(s + dnom[c], t)
            else:
                s = -np.inf
                for m in sim.members[c]:
                    if m in sim.arrivals[c]:
                        a = sim.arrivals[c][m]
                    elif sim.target[m] == c:
                        a = sim.eta(m, t)
                    else:
                        a = pfree[m] + w.D[ppos[m], c]
                    s = max(s, a)
                f = s + dnom[c]
            for m in sim.members[c]:
                pfree[m], ppos[m] = f, c
        Sl = sorted(S)
        loc = {i: k for k, i in enumerate(Sl)}
        inst = w.inst
        sub = Instance(req=inst.req[F], loc=inst.loc[F], dur=dnom[F], ab=inst.ab[Sl], depot=inst.depot[Sl],
                       species=inst.species[Sl], speed=inst.speed)
        Fi = np.array(F)
        object.__setattr__(sub, "tt", w.D[np.ix_(Fi, Fi)])
        object.__setattr__(sub, "da", w.D[np.ix_([w.home(i) for i in Sl], Fi)])
        object.__setattr__(sub, "_ready", pfree[Sl])
        object.__setattr__(sub, "_dout", w.D[np.ix_([ppos[i] for i in Sl], Fi)])
        object.__setattr__(sub, "_dhome", np.array([w.D[ppos[i], w.home(i)] for i in Sl]))
        warm = Plan([tuple(loc[m] for m in sim.members[j]) for j in F], sim.keys[F], len(Sl))
        t0 = time.perf_counter()
        plan, st = ALNS.solve(sub, 600.0, seed=self.seed + sim.stats["replans"], init_plan=warm, config=CFG,
                              max_iters=self.iters)
        sim.stats["solve_s"] += time.perf_counter() - t0
        sim.stats["iters"] += st.iterations
        sim.stats["replans"] += 1
        new_members = [tuple(Sl[k] for k in plan.members[x]) for x in range(len(F))]
        old_order = sorted(range(len(F)), key=lambda x: (sim.keys[F[x]], F[x]))
        changed = any(tuple(sorted(new_members[x])) != tuple(sorted(sim.members[F[x]])) for x in range(len(F))) \
            or list(plan.order()) != old_order
        sim.stats["changed"] += int(changed)
        if not changed:
            return
        base = float(np.max(sim.keys)) + 1.0
        for x, j in enumerate(F):
            sim.members[j] = new_members[x]
            sim.keys[j] = base + float(plan.keys[x])
        Fset = set(F)
        for i in Sl:
            keep = [j for j in sim.route[i] if j not in Fset]
            mine = sorted((j for j in F if i in sim.members[j]), key=lambda j: (sim.keys[j], j))
            sim.route[i] = keep + mine


# ---------------------------------------------------------------------------------------------------------------
def _setting(name):
    from cbba_sota.bench.configs import get

    return get(name)


def load_case(setting, i):
    st = _setting(setting)
    path = str(st.instance_path("dev", i))
    return path, st.seed("dev", i)


def noise_for(inst, sc, inst_key, noise_seed):
    return realized_durations(sc, inst.dur, inst_key, noise_seed), delay_calendar(sc, inst.n_agents, inst_key,
                                                                                    noise_seed)


def plans_job(args):
    path, k = args
    inst = Instance.from_pickle(path)
    plan, st = ALNS.solve(inst, 5.0, seed=100 + k)
    return path, k, plan.to_env_routes(), st.makespan


def main_job(job):
    inst = Instance.from_pickle(job["path"])
    w = World(inst)
    sc = SCEN[job["scen"]]
    T = inst.n_tasks
    dr, dl = noise_for(inst, sc, job["inst_key"], job["noise_seed"])
    plan0 = Plan.from_routes(job["plan0"], T, one_based=True)
    key = {k: job[k] for k in ("setting", "inst", "scen", "noise_seed")}
    rows = []

    def run(name, plan, rp):
        t0 = time.perf_counter()
        sim = Sim(w, plan, dr, dl, rp)
        ms = sim.run()
        s = sim.stats
        rows.append({**key, "method": name, "makespan": ms, "wall_s": time.perf_counter() - t0,
                     "events": s["events"], "replans": s["replans"], "changed": s["changed"],
                     "solve_s": s["solve_s"], "iters": s["iters"], "skipped": s["skipped"],
                     "frozen_frac": float(np.mean(s["frozen_frac"])) if s["frozen_frac"] else None})

    run("OL", plan0, None)
    for N in job["iters"]:
        run(f"RS{N}", plan0, Replanner(N))
    if job.get("saa"):
        for tag in ("SAA", "NOM"):
            p = Plan.from_routes(job["saa"][tag], T, one_based=True)
            run(f"{tag}k-OL", p, None)
            run(f"{tag}k-RS1000", p, Replanner(1000))
    for R in job.get("radii", ()):
        run(f"CRS{R}", plan0, Replanner(1000, mode="central", R=R))
        run(f"DRS{R}", plan0, Replanner(1000, mode="local", R=R))
    return rows


def saa_select(path, inst_key, cands):
    """cands: list of 1-based env routes. Returns {scen: {"SAA": routes, "NOM": routes, ...}}."""
    inst = Instance.from_pickle(path)
    w = World(inst)
    T = inst.n_tasks
    plans = [Plan.from_routes(r, T, one_based=True) for r in cands]
    nom = [evaluate(inst, p).makespan for p in plans]
    out = {}
    for scen, sc in SCEN.items():
        means = []
        for p in plans:
            ms = []
            for s in range(1000, 1000 + N_TRAIN):
                dr, dl = noise_for(inst, sc, inst_key, s)
                ms.append(Sim(w, p, dr, dl).run())
            means.append(float(np.mean(ms)))
        a, b = int(np.argmin(means)), int(np.argmin(nom))
        out[scen] = {"SAA": cands[a], "NOM": cands[b], "saa_idx": a, "nom_idx": b, "means": means, "nom": nom}
    return path, out


def _init():
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ[k] = "1"


def warm():
    """Compile the patched kernels once (disk cache) before spawning workers."""
    path, key = load_case(SETTINGS[0], 0)
    inst = Instance.from_pickle(path)
    ALNS.solve(inst, 1.0, seed=0, max_iters=50)
    w = World(inst)
    plan0 = json.loads((OUT / "alns_plans.json").read_text())
    p = Plan.from_routes(plan0[path]["routes"], inst.n_tasks, one_based=True)
    dr, dl = noise_for(inst, SCEN["N3"], key, 0)
    Sim(w, p, dr, dl, Replanner(20)).run()
    Sim(w, p, dr, dl, Replanner(20, mode="local", R=0.15)).run()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=32)
    ap.add_argument("--part", default="main", choices=("plans", "main"))
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--iters", type=int, nargs="+", default=[200, 1000, 5000])
    ap.add_argument("--out", default=str(OUT / "robust_rows.jsonl"))
    args = ap.parse_args()
    _init()
    ctx = mp.get_context("spawn")
    plans0 = json.loads((OUT / "alns_plans.json").read_text())
    cases = [(s, i, *load_case(s, i)) for s in SETTINGS for i in range(args.n)]
    cand_file = OUT / "robust_cands.json"
    sel_file = OUT / "robust_saa.json"
    if args.part == "plans":
        cands = json.loads(cand_file.read_text()) if cand_file.exists() else {}
        todo = [(p, k) for _, _, p, _ in cases for k in range(K_PLANS) if f"{p}|{k}" not in cands]
        print(len(todo), "plan jobs", flush=True)
        with ctx.Pool(args.procs, initializer=_init) as pool:
            for p, k, routes, ms in pool.imap_unordered(plans_job, todo):
                cands[f"{p}|{k}"] = {"routes": routes, "makespan": ms}
                cand_file.write_text(json.dumps(cands))
        sel = {}
        jobs = [(p, key, [cands[f"{p}|{k}"]["routes"] for k in range(K_PLANS)]) for _, _, p, key in cases]
        with ctx.Pool(args.procs, initializer=_init) as pool:
            for p, out in pool.starmap(saa_select, jobs):
                sel[p] = out
        sel_file.write_text(json.dumps(sel))
        print("selection written", flush=True)
        return
    warm()
    sel = json.loads(sel_file.read_text()) if sel_file.exists() else {}
    out = Path(args.out)
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            r = json.loads(line)
            done.add((r["setting"], r["inst"], r["scen"], r["noise_seed"]))
    jobs = []
    for s, i, path, key in cases:
        for scen in SCEN:
            for ns in range(args.seeds):
                if (s, i, scen, ns) in done:
                    continue
                jobs.append({"setting": s, "inst": i, "scen": scen, "noise_seed": ns, "path": path, "inst_key": key,
                             "plan0": plans0[path]["routes"], "iters": args.iters,
                             "saa": {k: sel[path][scen][k] for k in ("SAA", "NOM")} if path in sel else None,
                             "radii": RADII if scen == "N3" else ()})
    # longest first
    jobs.sort(key=lambda j: -len(j["radii"]))
    print(len(jobs), "jobs", flush=True)
    t0 = time.time()
    with ctx.Pool(args.procs, initializer=_init) as pool, out.open("a") as f:
        for k, rows in enumerate(pool.imap_unordered(main_job, jobs)):
            for r in rows:
                f.write(json.dumps(r) + "\n")
            f.flush()
            if k % 20 == 0:
                print(f"{k}/{len(jobs)} {time.time() - t0:.0f}s", flush=True)
    print("done", time.time() - t0, flush=True)


if __name__ == "__main__":
    main()
