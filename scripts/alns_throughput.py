"""Search throughput of the ALNS (and optionally the pure-Python pilot LNS) on given instances, one process.

Usage: alns_throughput.py --time 10 RALTestSet/test/env_0.pkl MA-AT-25-5-50/dev/env_0.pkl ... [--pilot]
Paths are relative to data/hetero. ALNS iterations and insertions per second exclude setup and construction. The
pilot (pilots/heteromrta/coalition_lns_pilot.py, unchanged) removes 1..T/5 tasks per iteration and tries 8 random
slots x 2 covers per insertion with a full forward pass each; its insertions are counted by wrapping
``insert_best`` minus the T construction insertions (its clock, like ours, starts after construction).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _pilot_module() -> tuple[object, list[int]]:
    """The pilot module with ``insert_best`` wrapped to count calls (``lns`` looks it up as a module global)."""
    from cbba_sota.bench.configs import HETEROMRTA_DIR

    sys.path.insert(0, str(HETEROMRTA_DIR))
    path = os.path.join(ROOT, "pilots", "heteromrta", "coalition_lns_pilot.py")
    spec = importlib.util.spec_from_file_location("pilot", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    calls = [0]
    insert_best = mod.insert_best

    def counted(*a, **k):
        calls[0] += 1
        return insert_best(*a, **k)

    mod.insert_best = counted
    return mod, calls


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--time", type=float, default=10.0)
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--out", default=None, help="append JSON rows here")
    args = ap.parse_args()
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ[var] = "1"

    from cbba_sota.bench.configs import DATA_DIR
    from cbba_sota.hetero import Instance
    from cbba_sota.solvers.alns import _warmup, solve

    _warmup()
    if args.pilot:
        pilot, calls = _pilot_module()
    rows = []
    for rel in args.paths:
        inst = Instance.from_pickle(DATA_DIR / rel)
        inst.tt, inst.da  # noqa: B018
        row = {"instance": rel, "tasks": inst.n_tasks, "agents": inst.n_agents, "time": args.time,
               "load_avg": os.getloadavg()[0]}
        if args.pilot:
            before = calls[0]
            I = pilot.Inst(inst.req, inst.loc, inst.dur, inst.ab, inst.depot)
            ms, it = pilot.lns(I, args.time, 0)
            row.update(method="pilot", iterations=it, it_per_s=it / args.time,
                       insertions_per_s=(calls[0] - before - inst.n_tasks) / args.time, makespan=ms)
        else:
            _, st = solve(inst, time_limit_s=args.time, seed=0)
            row.update(method="alns", iterations=st.iterations, it_per_s=st.it_per_s,
                       insertions_per_s=st.insertions / st.search_s,
                       slots_per_insertion=st.candidate_slots / max(st.insertions, 1),
                       exact_per_insertion=st.exact_evals / max(st.insertions, 1), makespan=st.makespan)
        rows.append(row)
        print(json.dumps(row), flush=True)
    if args.out:
        with open(args.out, "a") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows)


if __name__ == "__main__":
    main()
