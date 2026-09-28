"""Track D / robustness angle, pilot P0: how much can PLAN CHOICE (anticipation) buy under execution noise?

Open-loop question only (no re-planning): for each instance, build a pool of good plans
  nom[k]   ALNS on the nominal instance, seeds 0..7 (2 s, 1 core)
  q80      ALNS on the per-task 0.8-quantile durations of the noise model ("buffered durations" simple rule)
  msa[k]   ALNS on sampled duration scenarios (Bent & Van Hentenryck multiple-scenario pool), 8 per model
then execute every plan open loop (key order, wait for partners; = env execute_by_route for minimal covers) on
common random scenarios and compare
  E_nom    the nominally best nom plan
  E_saa    the plan picked from the whole pool by sample-average makespan on SELECTION scenarios
  E_saaN   same, picking among the nominal plans only
  E_q80    the quantile plan
  E_cv     clairvoyant ALNS on the realized durations (first N_CV test scenarios only)
all measured on held-out TEST scenarios. E_nom / E_saa is the open-loop anticipation lever; E_saa / E_cv what
anticipation cannot reach (value of information). A rolling re-planner recovers part of E_nom - E_cv without
anticipation, so the lever measured here is a (heuristic) ceiling for what SAA adds on top of rolling re-planning.

Noise models (per task realized duration; per (agent, destination) leg delay, common random numbers):
  logn.3   mean-preserving lognormal sigma 0.3 (the trackD pilot's N2)
  exp      exponential with mean d (stochastic RCPSP convention, Ballestin & Leus 2009; CV 1)
  u02d     uniform U(0, 2d) (same source; CV 0.577)
  hssrg    CS-HSSRG DELAYS analogue (Messing, Banfi et al., RSS 2023): d + U[0, dbar) always, dbar = 2.5 = mean
           nominal duration; each leg delayed w.p. 0.05 by U[0, 2.5)
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SNAP = HERE.parent / "trackD" / "_snap"
sys.path.insert(0, str(SNAP))

SETTINGS = ["MA-AT-25-5-50", "MA-AT-50-5-50", "SA-BT-50-5-50"]
N_INST = 10
MODELS = ["logn.3", "exp", "u02d", "hssrg"]
N_SEL, N_TEST, N_CV = 200, 200, 8
N_NOM, N_MSA = 8, 8
ALNS_S = 2.0
DBAR = 2.5


def durations(model, d, rng, n):
    T = len(d)
    if model == "logn.3":
        s = 0.3
        return d[None] * np.exp(s * rng.standard_normal((n, T)) - 0.5 * s * s)
    if model == "exp":
        return d[None] * rng.exponential(1.0, (n, T))
    if model == "u02d":
        return d[None] * rng.uniform(0.0, 2.0, (n, T))
    if model == "hssrg":
        return d[None] + rng.uniform(0.0, DBAR, (n, T))
    raise ValueError(model)


def leg_delays(model, A, T, rng, n):
    """[n, A, T+1] extra travel time of agent i's leg INTO task j (j == T: return to depot)."""
    if model != "hssrg":
        return np.zeros((n, A, T + 1))
    hit = rng.random((n, A, T + 1)) < 0.05
    return np.where(hit, rng.uniform(0.0, DBAR, (n, A, T + 1)), 0.0)


def q80_durations(model, d):
    if model == "logn.3":
        s = 0.3
        return d * np.exp(s * 0.8416 - 0.5 * s * s)
    if model == "exp":
        return d * -np.log(0.2)
    if model == "u02d":
        return d * 1.6
    if model == "hssrg":
        return d + 0.8 * DBAR
    raise ValueError(model)


def _fp():
    from numba import njit

    @njit(cache=True)
    def fp(order, mem, cnt, dur, tt, da, leg):
        A = da.shape[0]
        T = dur.shape[0]
        free = np.zeros(A)
        pos = np.full(A, -1, np.int64)
        for j in order:
            c = cnt[j]
            if c == 0:
                continue
            s = -np.inf
            for k in range(c):
                i = mem[j, k]
                a = free[i] + (da[i, j] if pos[i] < 0 else tt[pos[i], j]) + leg[i, j]
                s = max(s, a)
            f = s + dur[j]
            for k in range(c):
                i = mem[j, k]
                free[i] = f
                pos[i] = j
        ms = 0.0
        for i in range(A):
            if pos[i] >= 0:
                ms = max(ms, free[i] + da[i, pos[i]] + leg[i, T])
        return ms

    return fp


def evaluate_many(fp, inst, plan, D, L):
    order, mem, cnt = plan.to_arrays()
    tt, da = np.ascontiguousarray(inst.tt), np.ascontiguousarray(inst.da)
    return np.array([fp(order, mem, cnt, D[q], tt, da, L[q]) for q in range(len(D))])


def with_dur(inst, dur):
    from cbba_sota.hetero import Instance

    return Instance(req=inst.req, loc=inst.loc, dur=dur, ab=inst.ab, depot=inst.depot, species=inst.species,
                    speed=inst.speed, name=inst.name)


def solve_job(args):
    """One ALNS run; returns (key, plan members, keys)."""
    key, path, dur, seed = args
    from cbba_sota.hetero import Instance
    from cbba_sota.solvers.alns import solve

    inst = Instance.from_pickle(path)
    if dur is not None:
        inst = with_dur(inst, np.asarray(dur))
    plan, st = solve(inst, ALNS_S, seed=seed)
    return key, [list(m) for m in plan.members], plan.keys.tolist(), st.makespan


def main():
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ[k] = "1"
    procs = int(sys.argv[1]) if len(sys.argv) > 1 else 32
    from cbba_sota.bench.configs import get
    from cbba_sota.hetero import Instance

    out = HERE / "out"
    out.mkdir(exist_ok=True)
    jobs, meta = [], {}
    for s in SETTINGS:
        st = get(s)
        for i in range(N_INST):
            path = str(st.instance_path("dev", i))
            inst = Instance.from_pickle(path)
            ikey = st.seed("dev", i)
            meta[(s, i)] = (path, ikey)
            for k in range(N_NOM):
                jobs.append(((s, i, "nom", k), path, None, k))
            for m, model in enumerate(MODELS):
                jobs.append(((s, i, f"q80:{model}", 0), path, q80_durations(model, inst.dur).tolist(), 0))
                rng = np.random.default_rng([ikey, 77, m])
                D = durations(model, inst.dur, rng, N_MSA)
                for k in range(N_MSA):
                    jobs.append(((s, i, f"msa:{model}", k), path, D[k].tolist(), 100 + k))
                # clairvoyant on the first N_CV test scenarios (same stream as the test set below)
                rng_t = np.random.default_rng([ikey, 99, m])
                Dt = durations(model, inst.dur, rng_t, N_TEST)
                for q in range(N_CV):
                    jobs.append(((s, i, f"cv:{model}", q), path, Dt[q].tolist(), 0))
    cache = out / "lever_plans.jsonl"
    done = {}
    if cache.exists():
        for line in cache.read_text().splitlines():
            r = json.loads(line)
            done[tuple(r["key"])] = r
    todo = [j for j in jobs if tuple(j[0]) not in done]
    print(f"{len(jobs)} ALNS jobs, {len(todo)} to run", flush=True)
    t0 = time.time()
    if todo:
        ctx = mp.get_context("spawn")
        with ctx.Pool(procs) as pool, cache.open("a") as f:
            for n, (key, members, keys, ms) in enumerate(pool.imap_unordered(solve_job, todo)):
                r = {"key": list(key), "members": members, "keys": keys, "ms": ms}
                done[tuple(key)] = r
                f.write(json.dumps(r) + "\n")
                if n % 200 == 0:
                    print(f"{n}/{len(todo)} {time.time() - t0:.0f}s", flush=True)
    analyze(done, meta)


def analyze(done, meta):
    from cbba_sota.bench.configs import get
    from cbba_sota.hetero import Instance, Plan

    fp = _fp()
    rows = []
    for (s, i), (path, ikey) in meta.items():
        inst = Instance.from_pickle(path)
        A, T = inst.n_agents, inst.n_tasks

        def plan_of(key):
            r = done[key]
            return Plan([tuple(m) for m in r["members"]], np.array(r["keys"]), A)

        nom = [plan_of((s, i, "nom", k)) for k in range(N_NOM)]
        nom_ms = [done[(s, i, "nom", k)]["ms"] for k in range(N_NOM)]
        for m, model in enumerate(MODELS):
            rng_s = np.random.default_rng([ikey, 55, m])
            Ds, Ls = durations(model, inst.dur, rng_s, N_SEL), leg_delays(model, A, T, rng_s, N_SEL)
            rng_t = np.random.default_rng([ikey, 99, m])
            Dt = durations(model, inst.dur, rng_t, N_TEST)
            Lt = leg_delays(model, A, T, np.random.default_rng([ikey, 98, m]), N_TEST)
            pool = {f"nom{k}": p for k, p in enumerate(nom)}
            pool["q80"] = plan_of((s, i, f"q80:{model}", 0))
            for k in range(N_MSA):
                pool[f"msa{k}"] = plan_of((s, i, f"msa:{model}", k))
            sel = {n: evaluate_many(fp, inst, p, Ds, Ls).mean() for n, p in pool.items()}
            test = {n: evaluate_many(fp, inst, p, Dt, Lt) for n, p in pool.items()}
            best_nom = f"nom{int(np.argmin(nom_ms))}"
            pick_all = min(sel, key=sel.get)
            pick_nom = min((n for n in sel if n.startswith("nom")), key=sel.get)
            cv = np.array([done[(s, i, f"cv:{model}", q)]["ms"] for q in range(N_CV)])
            # clairvoyant plans evaluated with the realized leg delays too (they only knew durations)
            cvx = np.array([evaluate_many(fp, inst, plan_of((s, i, f"cv:{model}", q)), Dt[q:q + 1], Lt[q:q + 1])[0]
                            for q in range(N_CV)])
            rows.append({"setting": s, "inst": i, "model": model, "nominal_ms": min(nom_ms),
                         "E_nom": test[best_nom].mean(), "E_saa": test[pick_all].mean(),
                         "E_saaN": test[pick_nom].mean(), "E_q80": test["q80"].mean(),
                         "E_best_hindsight": min(v.mean() for v in test.values()),
                         "pick": pick_all,
                         "E_nom_cvset": test[best_nom][:N_CV].mean(), "E_saa_cvset": test[pick_all][:N_CV].mean(),
                         "E_cv": cvx.mean(), "E_cv_plan": cv.mean(),
                         "spread_nom_pool": float(np.ptp([test[f"nom{k}"].mean() for k in range(N_NOM)]))})
    out = HERE / "out" / "lever_rows.jsonl"
    out.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    report(rows)


def _ci(x):
    x = np.asarray(x)
    rng = np.random.default_rng(0)
    b = rng.choice(x, (4000, len(x))).mean(axis=1)
    return f"{x.mean():.3f} [{np.quantile(b, .025):.3f}, {np.quantile(b, .975):.3f}]"


def report(rows):
    print("ratios are per-instance means; lever = E_nom/E_saa (>1 means anticipation helps)")
    hdr = f"{'setting':15s} {'model':7s} {'E_nom/nomMS':>22s} {'lever E_nom/E_saa':>22s} {'E_nom/E_saaN':>22s} " \
          f"{'E_nom/E_q80':>22s} {'E_saa/E_cv (cv set)':>22s} {'E_nom/E_cv':>22s} picks"
    print(hdr)
    for s in SETTINGS:
        for model in MODELS:
            R = [r for r in rows if r["setting"] == s and r["model"] == model]
            if not R:
                continue
            f = lambda a, b: _ci([r[a] / r[b] for r in R])
            picks = {}
            for r in R:
                k = r["pick"].rstrip("0123456789")
                picks[k] = picks.get(k, 0) + 1
            print(f"{s:15s} {model:7s} {f('E_nom', 'nominal_ms'):>22s} {f('E_nom', 'E_saa'):>22s} "
                  f"{f('E_nom', 'E_saaN'):>22s} {f('E_nom', 'E_q80'):>22s} {f('E_saa_cvset', 'E_cv'):>22s} "
                  f"{f('E_nom_cvset', 'E_cv'):>22s} {picks}")


if __name__ == "__main__":
    main()
