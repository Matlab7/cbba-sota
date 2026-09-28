"""Resumable RL baseline campaign: runs/rl/<setting>/<split>.jsonl, one row per instance x method.

Usage: run_rl.py --split test --methods g s10 [--settings NAME ...] [--procs 36] [--workers 16]
       run_rl.py --split dev --methods s64 --lockstep --workers 1 --devices cuda:0 cuda:1 --out runs/rl_gpu

The N samples of an (instance, method) are split into min(N, --workers) contiguous blocks that run as separate
single-threaded pool jobs; wall_s = slowest block (the wall-clock with that many processes, excluding pool start-up
and model loading), cpu_s = total process CPU time, sample_s = per-rollout wall time. With --lockstep each block
runs its rollouts in lockstep with one batched forward pass per round on the pool process's device (--devices are
dealt out to the pool processes; sample_s is then empty). ``makespan``/``success`` are the best rollout's own env
run (the RL score); its routes are also replayed (pre_set_route + execute_by_route) and stored as
``replay_makespan``/``replay_success``, a diagnostic only. Every row records the instance ``fingerprint`` and the
``git`` code version; resuming skips only rows whose fingerprint matches the current instance.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time
from collections import defaultdict, deque
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict
from pathlib import Path
from typing import NamedTuple

import numpy as np

from cbba_sota.bench import runtime

METHODS = {"g": ("RL(g.)", 1, False), "s10": ("RL(s.10)", 10, True), "s64": ("RL(s.64)", 64, True),
           "s256": ("RL(s.256)", 256, True)}


class Job(NamedTuple):
    key: tuple[str, int, str]  # setting, instance, method code
    path: str
    seeds: list[int]
    sample: bool
    n_blocks: int  # blocks the key's samples are split into
    cost: float  # scheduling weight


_net = None
_device = "cpu"
_lockstep = False


def _init(devices, lockstep: bool) -> None:
    global _net, _device, _lockstep
    import torch

    from cbba_sota.solvers import rl

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    _device, _lockstep = devices.get(), lockstep
    _net = rl.load_policy(_device)


def _sample(path: str, seeds: list[int], sample: bool):
    """(samples, device, fingerprint of the instance they ran on)."""
    from cbba_sota.hetero import Instance
    from cbba_sota.solvers import rl

    data = Path(path).read_bytes()
    fp = runtime.fingerprint(Instance.from_env(rl.loads_env(data)))
    if _lockstep:
        return rl.sample_lockstep(data, _net, seeds, sample=sample, device=_device), _device, fp
    return rl.sample_many(data, _net, seeds, sample=sample, device=_device), _device, fp


def _replay(path: str, routes) -> tuple[float, bool]:
    from cbba_sota.solvers import rl

    return rl.replay(Path(path).read_bytes(), routes)


def out_path(root: Path, setting: str, split: str) -> Path:
    return root / setting / f"{split}.jsonl"


def done_keys(path: Path) -> set[tuple[int, str]]:
    """(instance, method) of the rows computed on the current instance (matching fingerprint)."""
    return {(row["instance"], row["method"]) for row in runtime.read_rows(path) if runtime.is_current(row)}


def main() -> None:
    from cbba_sota.bench import configs
    from cbba_sota.solvers.rl import merge, sample_seed

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--settings", nargs="*", default=[s.name for s in configs.SETTINGS])
    ap.add_argument("--split", required=True, choices=list(configs.SPLITS))
    ap.add_argument("--methods", nargs="+", default=["g", "s10"], choices=list(METHODS))
    ap.add_argument("--procs", type=int, default=36, help="pool processes (each single-threaded)")
    ap.add_argument("--workers", type=int, default=16, help="max processes one sampled method is spread over (CPU)")
    ap.add_argument("--lockstep", action="store_true", help="batch the rollouts of a block (one forward per round)")
    ap.add_argument("--devices", nargs="+", default=["cpu"], help="e.g. cuda:0 cuda:1 (with --lockstep)")
    ap.add_argument("--instances", type=int, default=None, help="only the first k instances")
    ap.add_argument("--out", type=Path, default=configs.RUNS_DIR / "rl")
    args = ap.parse_args()

    os.environ.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", NUMBA_NUM_THREADS="1")
    jobs: list[Job] = []
    for name in args.settings:
        setting = configs.get(name)
        if args.split not in setting.splits():
            continue
        done = done_keys(out_path(args.out, name, args.split))
        n_inst = setting.n_instances(args.split) if args.instances is None else args.instances
        for i in range(n_inst):
            path = str(setting.instance_path(args.split, i))
            for m in args.methods:
                method, n, sample = METHODS[m]
                if (i, method) in done:
                    continue
                seeds = [sample_seed(setting.seed(args.split, i), k) for k in range(n)]
                blocks = np.array_split(seeds, min(n, args.workers))
                cost = setting.n_tasks * setting.n_agents * n / len(blocks)
                jobs += [Job((name, i, m), path, [int(x) for x in b], sample, len(blocks), cost) for b in blocks]
    jobs.sort(key=lambda j: -j.cost)  # longest first; blocks of one instance stay adjacent
    print(f"{len(jobs)} jobs", flush=True)
    if not jobs:
        return

    ctx = mp.get_context("spawn")
    devices = ctx.Queue()
    for k in range(args.procs):
        devices.put(args.devices[k % len(args.devices)])
    queue, replays = deque(jobs), deque()
    parts: dict[tuple, list] = defaultdict(list)
    running: dict = {}
    t0, written = time.time(), 0
    args.git = runtime.code_version()
    with ProcessPoolExecutor(args.procs, mp_context=ctx, initializer=_init,
                             initargs=(devices, args.lockstep)) as ex:
        while queue or replays or running:
            while len(running) < args.procs and (replays or queue):  # replays first so rows land early
                if replays:
                    key, path, res = replays.popleft()
                    running[ex.submit(_replay, path, res[0].best.routes)] = ("replay", key, res)
                else:
                    job = queue.popleft()
                    running[ex.submit(_sample, job.path, job.seeds, job.sample)] = ("sample", job, None)
            completed, _ = wait(running, return_when=FIRST_COMPLETED)
            for fut in completed:
                kind, item, res = running.pop(fut)
                if kind == "replay":
                    _write(args, item, *res, fut.result())
                    written += 1
                    print(f"[{time.time() - t0:7.0f}s] {item} {res[0].best.makespan:.3f}", flush=True)
                    continue
                parts[item.key].append(fut.result())
                if len(parts[item.key]) == item.n_blocks:
                    results, devices_used, fps = zip(*parts.pop(item.key))
                    if len(set(fps)) > 1:
                        raise RuntimeError(f"{item.path} changed while {item.key} was sampled")
                    replays.append((item.key, item.path,
                                    (merge(results), ",".join(sorted(set(devices_used))), fps[0])))
    print(f"{written} rows in {time.time() - t0:.0f}s", flush=True)


def _write(args, key, res, device: str, fp: str, replayed: tuple[float, bool]) -> None:
    from cbba_sota.bench import configs

    name, i, m = key
    method, n, _ = METHODS[m]
    best = asdict(res.best)
    row = {"setting": name, "split": args.split, "instance": i, "seed": configs.get(name).seed(args.split, i),
           "method": method, "n_samples": n, "makespan": best["makespan"], "success": best["success"],
           "env_finished": best["env_finished"], "completion": best["completion"], "awt": best["awt"],
           "wall_s": res.wall_s, "cpu_s": res.cpu_s,
           "device": device, "lockstep": args.lockstep, "workers": min(n, args.workers),
           "replay_makespan": replayed[0],
           "replay_success": replayed[1], "depot_revisits": best["depot_revisits"],
           "sample_makespans": res.makespans, "sample_success": res.successes, "sample_s": res.sample_s,
           "routes": best["routes"], "fingerprint": fp, "git": args.git}
    path = out_path(args.out, name, args.split)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
