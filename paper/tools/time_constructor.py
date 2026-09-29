"""Time of one greedy construction (the step of the restart baseline and of ALNS's initial portfolio).

Usage: taskset -c <free cpu> time_constructor.py [--split val] [--out paper/tables/constructor_timing.json]
Times plain dispatch, noisy dispatch and random-order insertion (up to 100 tasks, as the restarts use it) on the
first instance of each setting, single-threaded, after one warm-up call (numba compilation excluded).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_tables import SETTINGS

from cbba_sota.bench import configs
from cbba_sota.hetero import load_instance
from cbba_sota.solvers import greedy


def timed(fn, reps: int) -> float:
    fn()
    t = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - t) / reps


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="val")
    ap.add_argument("--out", type=Path, default=ROOT / "paper" / "tables" / "constructor_timing.json")
    args = ap.parse_args()
    out = {}
    for name in SETTINGS:
        inst = load_instance(configs.get(name).instance_path(args.split, 0))
        rng = np.random.default_rng(0)
        reps = 5 if inst.n_tasks <= 50 else 2
        row = {"dispatch_s": timed(lambda inst=inst: greedy.dispatch(inst), reps),
               "noisy_dispatch_s": timed(lambda inst=inst: greedy.dispatch(inst, noise=0.3, seed=1), reps)}
        if inst.n_tasks <= 100:
            row["insertion_s"] = timed(lambda inst=inst, rng=rng: greedy.insertion(inst, rng), reps)
        out[name] = row
        print(name, {k: f"{v * 1000:.1f} ms" for k, v in row.items()})
    args.out.write_text(json.dumps(out, indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
