"""Run the comm-first pilot P1 (see comm_sim.py) and print paired summaries.

Usage (repo root): .venv/bin/python pilots/trackD_comm/run_p1.py [--procs 24] [--n 10] [--seeds 2] [--tag p1]
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
SNAP = HERE.parent / "trackD" / "_snap"

SETTINGS = ("MA-AT-25-5-50", "MA-AT-50-5-50", "SA-BT-50-5-50")
RADII = (0.3, 0.2, 0.15, 0.1)
ARMS = (("Rn", "idle"), ("Rn", "return"), ("Oh", "idle"), ("Oh", "return"), ("O", "return"),
        ("C", "idle"), ("C", "return"), ("C", "greedy"))


def job(args):
    setting, i, seed, R, arm, lf, dod, sigma, p_loss = args
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ[k] = "1"
    sys.path.insert(0, str(SNAP))
    sys.path.insert(0, str(HERE))
    from cbba_sota.hetero import Instance
    from comm_sim import HORIZON, run_episode

    inst = Instance.from_pickle(SNAP / "data" / "hetero" / setting / "dev" / f"env_{i}.pkl")
    t0 = time.perf_counter()
    r = run_episode(inst, i, R, arm, lf, dod, HORIZON[setting], sigma, seed, p_loss)
    return dict(setting=setting, inst=i, seed=seed, R=R, arm=arm, lf=lf, dod=dod, sigma=sigma, p_loss=p_loss,
                wall=time.perf_counter() - t0, **r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=24)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--tag", default="p1")
    ap.add_argument("--dod", type=float, default=0.5)
    ap.add_argument("--sigma", type=float, default=0.3)
    ap.add_argument("--p_loss", type=float, default=0.0)
    ap.add_argument("--radii", type=str, default=",".join(map(str, RADII)))
    ap.add_argument("--arms", type=str, default=",".join(f"{a}:{b}" for a, b in ARMS))
    ap.add_argument("--settings", type=str, default=",".join(SETTINGS))
    args = ap.parse_args()
    radii = [float(x) for x in args.radii.split(",")]
    arms = [tuple(x.split(":")) for x in args.arms.split(",")]
    jobs = []
    for s in args.settings.split(","):
        for i in range(args.n):
            for seed in range(args.seeds):
                jobs.append((s, i, seed, float("inf"), "O", "idle", args.dod, args.sigma, 0.0))  # G
                for R in radii:
                    for arm, lf in arms:
                        jobs.append((s, i, seed, R, arm, lf, args.dod, args.sigma, args.p_loss))
    out = HERE / "out" / f"{args.tag}.jsonl"
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            r = json.loads(line)
            done.add((r["setting"], r["inst"], r["seed"], r["R"], r["arm"], r["lf"], r["dod"], r["sigma"],
                      r["p_loss"]))
    todo = [j for j in jobs if j not in done]
    print(f"{len(jobs)} jobs, {len(todo)} to run", flush=True)
    t0 = time.time()
    with mp.get_context("spawn").Pool(args.procs) as pool, out.open("a") as f:
        for n, r in enumerate(pool.imap_unordered(job, todo), 1):
            f.write(json.dumps(r) + "\n")
            f.flush()
            if n % 100 == 0:
                print(f"  {n}/{len(todo)}  {time.time() - t0:.0f}s", flush=True)
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
