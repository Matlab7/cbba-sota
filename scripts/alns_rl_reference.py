"""Paired RL reference for ALNS dev comparisons: RL(g.) and RL(s.N) of the released checkpoint on a few instances.

Usage: alns_rl_reference.py --settings MA-AT-50-5-200 --split dev --indices 0 1 2 --samples 10 --procs 10
Rows go to runs/alns/rl_ref/<setting>-<split>.jsonl (replayed makespan of the best rollout). One rollout per pool
job, single-threaded; seeds follow scripts/run_rl.py (``rl.sample_seed(instance_seed, k)``), so sample k is the
same rollout as in the RL campaign.
"""
from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor

_net = None


def _init() -> None:
    global _net
    import torch

    from cbba_sota.solvers import rl

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    _net = rl.load_policy("cpu")


def _rollout(path: str, seed: int, sample: bool) -> dict:
    from pathlib import Path

    from cbba_sota.solvers import rl

    s = rl.sample_many(Path(path).read_bytes(), _net, [seed], sample=sample)
    ms, ok = rl.replay(Path(path).read_bytes(), s.best.routes)
    return {"makespan": s.best.makespan, "success": s.best.success, "replay_makespan": ms, "replay_success": ok,
            "wall_s": s.wall_s}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--settings", nargs="+", required=True)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--indices", nargs="+", type=int, required=True)
    ap.add_argument("--samples", type=int, default=10)
    ap.add_argument("--procs", type=int, default=10)
    args = ap.parse_args()
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ[var] = "1"

    from cbba_sota.bench import configs
    from cbba_sota.solvers import rl

    jobs = []
    for name in args.settings:
        s = configs.get(name)
        for i in args.indices:
            iseed = s.seed(args.split, i)
            path = str(s.instance_path(args.split, i))
            jobs.append((name, i, "RL(g.)", path, rl.sample_seed(iseed, 0), False))
            jobs += [(name, i, f"RL(s.{args.samples})", path, rl.sample_seed(iseed, k), True)
                     for k in range(args.samples)]
    with ProcessPoolExecutor(args.procs, initializer=_init) as ex:
        results = list(ex.map(_rollout, *zip(*[(j[3], j[4], j[5]) for j in jobs])))
    rows: dict[tuple, dict] = {}
    for (name, i, method, _, _, _), res in zip(jobs, results):
        key = (name, i, method)
        best = rows.get(key)
        ok = res["replay_success"]
        if best is None or (ok, -res["replay_makespan"]) > (best["success"], -best["makespan"]):
            rows[key] = {"setting": name, "split": args.split, "index": i, "method": method,
                         "makespan": res["replay_makespan"], "success": ok,
                         "rollout_wall_s": [], "name": f"{name}/{args.split}/{i}"}
        rows[key]["rollout_wall_s"].append(res["wall_s"])
    out_dir = configs.RUNS_DIR / "alns" / "rl_ref"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in args.settings:
        with (out_dir / f"{name}-{args.split}.jsonl").open("a") as f:
            for row in rows.values():
                if row["setting"] == name:
                    f.write(json.dumps(row) + "\n")
                    print(row["name"], row["method"], round(row["makespan"], 3), row["success"],
                          f"sum rollout wall {sum(row['rollout_wall_s']):.0f}s")


if __name__ == "__main__":
    main()
