"""Track D: the state-start kernel reduces to the pinned ALNS v2 kernel on static instances, bit for bit.

Loads ``cbba_sota/solvers/_alns_kernels.py`` at commit 4cf5e04 from git (``git show``, never the working tree, which
another workflow edits) and compares it with ``cbba_sota/dyn/rh_kernels.py`` on a static problem
(``PlanState.static``: every robot at its depot at time 0, nothing committed):
1. instance arrays 0-11 equal the pinned ``_inst_arrays`` (copied below from 4cf5e04 ``alns.py``);
2. ``schedule`` of random key-ordered minimal-cover plans: starts, finishes, arrivals, returns, tails, objective;
3. ``construct`` (seeded random-order best insertion): the same plan and schedule;
4. ``run_batch`` (seeded, one batch of ``--iters`` iterations from that plan): the same best plan, objective, SA
   statistics, operator weights and insertion counters.

Usage: .venv/bin/python scripts/trackD_kernel_check.py [--n 3] [--iters 1500]
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")

import numpy as np

from cbba_sota.bench.configs import get
from cbba_sota.dyn import rh_kernels as DK
from cbba_sota.dyn.planner import (
    PINNED_COMMIT,
    PlanState,
    Problem,
    SearchConfig,
    q_range,
    t_start_rule,
)
from cbba_sota.hetero import Instance, random_plan

ROOT = Path(__file__).resolve().parents[1]
SETTINGS = ["MA-AT-25-5-50", "MA-AT-50-5-50", "SA-AT-50-5-50", "SA-BT-50-5-50", "MA-AT-50-5-200"]


def load_pinned(tmp: Path):
    src = subprocess.check_output(["git", "show", f"{PINNED_COMMIT}:cbba_sota/solvers/_alns_kernels.py"], cwd=ROOT)
    path = tmp / "pinned_alns_kernels_4cf5e04.py"
    path.write_bytes(src)
    spec = importlib.util.spec_from_file_location("pinned_alns_kernels_4cf5e04", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def pinned_inst_arrays(inst: Instance) -> tuple:
    """Verbatim logic of ``_inst_arrays`` in ``cbba_sota/solvers/alns.py`` at 4cf5e04."""
    own = lambda a: np.array(a, dtype=float, order="C")
    T, A = inst.n_tasks, inst.n_agents
    capable = ((inst.ab[None, :, :] > 0) & (inst.req[:, None, :] > 0)).any(axis=2)
    ncap = capable.sum(axis=1).astype(np.int64)
    capl = np.full((T, A), -1, np.int64)
    for j in range(T):
        capl[j, :ncap[j]] = np.flatnonzero(capable[j])
    tt = inst.tt
    near = np.argsort(tt + np.diag(np.full(T, np.inf)), axis=1, kind="stable")[:, :T - 1].astype(np.int64)
    _, group = np.unique(inst.ab, axis=0, return_inverse=True)
    group = group.ravel().astype(np.int64)
    partners = np.argsort(group, kind="stable").astype(np.int64)
    pstart = np.searchsorted(group[partners], np.arange(group.max() + 2)).astype(np.int64)
    return (own(inst.req), own(inst.ab), own(inst.dur), own(tt), own(inst.da), capl, ncap, near, partners, pstart,
            group, capable)


def pinned_sol(T, A, W):
    return (np.zeros(T, np.int64), np.full((T, W), -1, np.int64), np.zeros(T, np.int64), np.zeros(1, np.int64),
            np.full(T, np.nan), np.full(T, np.nan), np.zeros((T, W)), np.full((T, W), -1, np.int64),
            np.zeros(A), np.full(A, -1, np.int64), np.zeros(3), np.full((T, W), -1, np.int64),
            np.full(A, -1, np.int64), np.zeros(A, np.int64), np.zeros(T))


def same(a, b) -> bool:
    a, b = np.asarray(a), np.asarray(b)
    return a.shape == b.shape and bool(np.all((a == b) | (np.isnan(a) & np.isnan(b)) if a.dtype.kind == "f"
                                              else a == b))


def load(sol, plan, lam, K, arrays):
    seq, mem, cnt, ni = sol[:4]
    order = [int(j) for j in plan.order() if plan.members[j]]
    cnt[:] = 0
    mem[:] = -1
    for j in order:
        mem[j, :len(plan.members[j])] = plan.members[j]
        cnt[j] = len(plan.members[j])
    seq[:len(order)] = order
    ni[0] = len(order)
    K.schedule(arrays, sol, lam)


def compare_sched(sa, sb, what) -> list[str]:
    errs = []
    for k, name in enumerate(("seq", "mem", "cnt", "ni", "start", "finish", "barr", "pred", "ret", "last")):
        if name in ("seq",):
            n = sa[3][0]
            if not same(sa[0][:n], sb[0][:n]):
                errs.append(f"{what}: {name}")
            continue
        if not same(sa[k], sb[k]):
            errs.append(f"{what}: {name}")
    if not same(sa[10][:3], sb[10][:3]):
        errs.append(f"{what}: fl {sa[10][:3]} vs {sb[10][:3]}")
    if not same(sa[14], sb[14]):
        errs.append(f"{what}: tail")
    return errs


def check(inst: Instance, P, iters: int, seed: int) -> list[str]:
    cfg = SearchConfig()
    state = PlanState.static(inst)
    prob = Problem(state)
    da, pa = prob.arrays, pinned_inst_arrays(inst)
    errs = [f"array {DK.INST[k]}" for k in range(12) if not (same(da[k], pa[k]) and da[k].dtype == pa[k].dtype)]
    T, A, W = prob.Tk, prob.A, prob.W
    rng = np.random.default_rng(seed)
    for r in range(5):
        plan = random_plan(inst, rng, greedy=0.0 if r % 2 else 2.0)
        sd, sp = prob.new_sol(), pinned_sol(T, A, W)
        load(sd, plan, cfg.lam, DK, da)
        load(sp, plan, cfg.lam, P, pa)
        errs += compare_sched(sd, sp, f"schedule random plan {r}")
    # construct
    sd, sp = prob.new_sol(), pinned_sol(T, A, W)
    wsd, wsp = prob.workspace(), prob.workspace()
    order = np.random.default_rng(seed).permutation(T).astype(np.int64)
    for K, sol, arrays, ws in ((DK, sd, da, wsd), (P, sp, pa, wsp)):
        K.seed(seed)
        sol[2][:] = 0
        sol[3][0] = 0
        K.schedule(arrays, sol, cfg.lam)
        K.construct(order, arrays, sol, ws, np.zeros(T + 1, np.bool_), np.full(T, -1, np.int64),
                    np.zeros(A, np.int64), 0.0, cfg.lam, cfg.max_slots)
    errs += compare_sched(sd, sp, "construct")
    # run_batch
    q_lo, q_hi = q_range(T)
    par = np.array([cfg.lam, cfg.noise, q_lo, q_hi, cfg.max_slots, cfg.shaw_p, cfg.worst_p, cfg.regret_max_q,
                    cfg.rho, cfg.seg_len, *cfg.sigma, cfg.restart_iters, cfg.q_stag, max(q_hi, min(cfg.q_big, T)),
                    0.0], float)
    temp0 = t_start_rule(T) * float(sd[10][2])
    out = {}
    for name, K, cur, arrays, ws, newsol in (("dyn", DK, sd, da, wsd, prob.new_sol),
                                             ("pinned", P, sp, pa, wsp, lambda: pinned_sol(T, A, W))):
        cand, best = newsol(), newsol()
        K.copy_sol(cur, cand)
        K.copy_sol(cur, best)
        wd = np.array([0.0 if op in cfg.ops_off else 1.0 for op in K.DESTROY])
        wr = np.ones(len(K.REPAIR))
        vec = [np.zeros(len(wd)), np.zeros(len(wr)), np.zeros(len(wd)), np.zeros(len(wr))]
        stats = np.zeros(len(K.STATS), np.int64)
        K.seed(seed + 1)
        K.run_batch(iters, temp0, cfg.t_end_ratio * temp0, arrays, cur, cand, best, ws, np.zeros(T + 1, np.bool_),
                    np.full(T, -1, np.int64), np.zeros(A, np.int64), np.zeros(A, np.int64), np.zeros(T, np.int64),
                    np.zeros(T, np.bool_), np.zeros(T, np.int64), wd, wr, *vec, par, stats)
        out[name] = (best, stats, wd, wr, ws[7].copy())
    (bd, std, wdd, wrd, cd), (bp, stp, wdp, wrp, cpn) = out["dyn"], out["pinned"]
    errs += compare_sched(bd, bp, f"run_batch({iters}) best")
    for nm, a, b in (("stats", std, stp), ("destroy weights", wdd, wdp), ("repair weights", wrd, wrp),
                     ("insertion counters", cd, cpn)):
        if not same(a, b):
            errs.append(f"run_batch {nm}: {a} vs {b}")
    return errs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--iters", type=int, default=1500)
    args = ap.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        P = load_pinned(Path(tmp))
        bad = 0
        for s in SETTINGS:
            st = get(s)
            for i in range(args.n):
                inst = Instance.from_pickle(st.instance_path("dev", i))
                errs = check(inst, P, args.iters, seed=1000 + i)
                bad += bool(errs)
                print(f"{s} dev {i}: {'OK' if not errs else 'MISMATCH ' + '; '.join(errs[:6])}", flush=True)
    print(f"pinned commit {PINNED_COMMIT}: {'all identical' if not bad else f'{bad} mismatching instances'}")
    return int(bad > 0)


if __name__ == "__main__":
    sys.exit(main())
