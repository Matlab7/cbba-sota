"""Track D C3 map runner (spec Section 8 days 4-5, Section 9.4): protocol arms x connectivity x families on dev.

Every episode runs in the ground-truth world ``cbba_sota.dyn.c3world.C3Env`` (``DynTaskEnvX`` + rally + per-replica
D7) with SPARC's planner (agent B's ``RHPlanner``, Light: 300 iterations) inside the T2 comm layer. Rows go to a
resumable JSONL: a case is done when a row with the same case id (configuration + C3 code hash) exists.

Usage:
  .venv/bin/python scripts/trackD_c3.py run --settings MA-AT-25-5-50 SA-BT-50-5-50 --instances 0-19 --seeds 0-1 \\
      --cells F1-R2 F3-pf0.2 --arms CEN-F HYB REP-clamp INF-r SPARC SPARC-V3 --rhos 2 1 0.5 \\
      --leases 1:30 --procs 22 --out runs/trackD/c3/eval.jsonl
  (FULL is added once per (setting, instance, seed, cell) whenever it is listed in --arms; its lease is the failure
  detector, so --leases and --rhos do not apply to it.)
  --leases accepts "g:L" pairs, "grid" (the shared 3x3 grid of spec 4.3) or "tuned:<json>" (per arm and rho, the
  frozen choice written by ``scripts/trackD_c3_analyze.py tune``).
  .venv/bin/python scripts/trackD_c3.py summary --rows runs/trackD/c3/eval.jsonl
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MAX_PROCS = 22  # the task's CPU cap is 24 processes (parent + workers)
GRID = [(g, L) for g in (0.3, 1.0, 3.0) for L in (10.0, 30.0, 60.0)]


def parse_range(spec: list[str]) -> list[int]:
    out: list[int] = []
    for s in spec:
        if "-" in s:
            a, b = s.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(s))
    return out


def lease_list(spec: list[str], arm: str, rho: float) -> list[tuple[float, float]]:
    out = []
    for s in spec:
        if s == "grid":
            out += GRID
        elif s.startswith("tuned:"):
            tuned = json.loads(Path(s[6:]).read_text())
            g, L = tuned["choice"][arm][f"{rho:g}"]
            out.append((float(g), float(L)))
        else:
            g, L = s.split(":")
            out.append((float(g), float(L)))
    return list(dict.fromkeys(out))


def build_cases(args) -> list[dict]:
    cases = []
    base = {"frozen_model": args.frozen_model, "keys": args.keys, "beacon": args.beacon, "iters": args.iters,
            "rally": not args.no_rally, "loss": args.loss, "adopt": args.adopt}
    for setting in args.settings:
        for i in parse_range(args.instances):
            for seed in parse_range(args.seeds):
                for cell in args.cells:
                    for arm in args.arms:
                        if arm == "FULL":
                            cases.append({"setting": setting, "inst": i, "seed": seed, "cell": cell, "arm": arm,
                                          "rho": math.inf, "lease_g": 0.0, "lease_L": 0.0, **base})
                            continue
                        for rho in args.rhos:
                            for g, L in lease_list(args.leases, arm, rho):
                                cases.append({"setting": setting, "inst": i, "seed": seed, "cell": cell, "arm": arm,
                                              "rho": float(rho), "lease_g": g, "lease_L": L, **base})
    return cases


def case_id(c: dict, h: str) -> str:
    keys = ("setting", "inst", "seed", "cell", "arm", "rho", "lease_g", "lease_L", "frozen_model", "keys", "beacon",
            "iters", "rally", "loss", "adopt")
    return "|".join(str(c[k]) for k in keys) + "|" + h


def _init() -> None:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["NUMBA_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"


def run_case(c: dict) -> dict:
    _init()
    from cbba_sota.dyn.c3world import C3Config, c3_hash, run_c3

    cfg = C3Config(arm=c["arm"], rho=c["rho"] if math.isfinite(c["rho"]) else 1.0, lease_g=c["lease_g"] or 1.0,
                   lease_L=c["lease_L"] or 30.0, iters=c["iters"], frozen_model=c["frozen_model"], keys=c["keys"],
                   beacon=c["beacon"], rally=c["rally"], loss=c["loss"], adopt=c.get("adopt", "keep"))
    try:
        row = run_c3(c["setting"], "dev", c["inst"], c["seed"], c["cell"], cfg)
        row["status"] = "ok"
    except Exception as e:  # noqa: BLE001  (recorded as an error row, never silently dropped)
        row = {"status": "error", "error": repr(e), "traceback": traceback.format_exc()[-3000:]}
    row.update({k: c[k] for k in ("setting", "inst", "seed", "cell", "arm", "rho", "lease_g", "lease_L",
                                  "frozen_model", "keys", "beacon", "iters", "rally", "loss", "adopt")})
    row["case_id"] = case_id(c, c3_hash())
    return row


def load_done(out: Path) -> set[str]:
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("status") == "ok":
                done.add(r["case_id"])
    return done


def cmd_run(args) -> None:
    import multiprocessing as mp

    _init()
    from cbba_sota.dyn.c3world import c3_hash

    h = c3_hash()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cases = build_cases(args)
    done = load_done(out)
    todo = [c for c in cases if case_id(c, h) not in done]
    # longest first: low rho and V1 arms plan the most
    todo.sort(key=lambda c: (c["rho"] if math.isfinite(c["rho"]) else 9, c["arm"] != "HYB", c["setting"]))
    procs = min(args.procs, MAX_PROCS)
    print(f"{len(cases)} cases, {len(cases) - len(todo)} done, {len(todo)} to run on {procs} processes, "
          f"c3 hash {h}", flush=True)
    if args.dry or not todo:
        return
    t0 = time.time()
    n = 0
    with mp.get_context("spawn").Pool(procs, initializer=_init, maxtasksperchild=50) as pool, out.open("a") as f:
        for row in pool.imap_unordered(run_case, todo, chunksize=1):
            f.write(json.dumps(row, default=float) + "\n")
            f.flush()
            n += 1
            if n % 20 == 0 or n == len(todo):
                el = time.time() - t0
                print(f"{n}/{len(todo)} rows, {el / 60:.1f} min, eta {(len(todo) - n) * el / n / 60:.1f} min",
                      flush=True)


def cmd_summary(args) -> None:
    import numpy as np

    rows = [json.loads(x) for p in args.rows for x in Path(p).read_text().splitlines() if x.strip()]
    rows = [r for r in rows if r.get("status") == "ok"]
    errs = sum(1 for p in args.rows for x in Path(p).read_text().splitlines() if '"status": "error"' in x)
    print(f"{len(rows)} ok rows, {errs} error rows")
    groups: dict = {}
    for r in rows:
        k = (r["setting"], r["cell"], r["arm"], r["rho"], r["lease_g"], r["lease_L"], r["frozen_model"], r["keys"])
        groups.setdefault(k, []).append(r)
    print(f"{'setting':15s} {'cell':9s} {'arm':10s} {'rho':>4s} {'g':>4s} {'L':>4s} {'model':6s} {'keys':8s} "
          f"{'n':>3s} {'GM ms':>7s} {'succ':>5s} {'wasted':>6s} {'aband':>5s} {'plans':>6s} {'B/r/t':>6s} "
          f"{'wall':>5s}")
    for k in sorted(groups, key=lambda k: tuple(str(x) for x in k)):
        g = groups[k]
        ms = np.array([r["makespan_or_cap"] for r in g])
        print(f"{k[0]:15s} {k[1]:9s} {k[2]:10s} {k[3]:4g} {k[4]:4g} {k[5]:4g} {k[6]:6s} {k[7]:8s} {len(g):3d} "
              f"{np.exp(np.log(ms).mean()):7.2f} {np.mean([r['success'] for r in g]):5.2f} "
              f"{np.mean([r['wasted_trips'] for r in g]):6.1f} {np.mean([r['abandons'] for r in g]):5.1f} "
              f"{np.mean([r['plan_calls'] for r in g]):6.0f} {np.mean([r['bytes_per_robot_time'] for r in g]):6.0f} "
              f"{np.mean([r['wall_s'] for r in g]):5.1f}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--settings", nargs="+", default=["MA-AT-25-5-50", "SA-BT-50-5-50"])
    r.add_argument("--instances", nargs="+", default=["0-19"])
    r.add_argument("--seeds", nargs="+", default=["0-1"])
    r.add_argument("--cells", nargs="+", default=["F1-R2", "F3-pf0.2"])
    r.add_argument("--arms", nargs="+", default=["FULL", "CEN-F", "HYB", "REP-clamp", "INF-r", "SPARC", "SPARC-V3"])
    r.add_argument("--rhos", nargs="+", type=float, default=[2.0, 1.0, 0.5])
    r.add_argument("--leases", nargs="+", default=["1:30"])
    r.add_argument("--frozen-model", default="anchor", choices=["anchor", "head"])
    r.add_argument("--keys", default="monotone", choices=["monotone", "relaxed"])
    r.add_argument("--beacon", type=float, default=None)
    r.add_argument("--iters", type=int, default=300)
    r.add_argument("--loss", default="ge")
    r.add_argument("--no-rally", action="store_true")
    r.add_argument("--adopt", default="keep", choices=["follow", "keep"])
    r.add_argument("--procs", type=int, default=MAX_PROCS)
    r.add_argument("--out", required=True)
    r.add_argument("--dry", action="store_true")
    r.set_defaults(func=cmd_run)
    s = sub.add_parser("summary")
    s.add_argument("--rows", nargs="+", required=True)
    s.set_defaults(func=cmd_summary)
    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
