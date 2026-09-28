"""ALNS campaign: one JSONL row per instance in runs/alns/<tag>/<setting>-<split>.jsonl, every makespan env-replayed.

Usage:
  run_alns.py --settings RALTestSet --split test --time 4 --tag ral4
  run_alns.py --settings MA-AT-50-5-200 --split dev --n 5 --time 30 --workers 4 --procs 8
  run_alns.py --json pilots/heteromrta/redteam/instances_scale50.json --time 13 --tag scale50
  run_alns.py ... --set lam=0.05 noise=0.3        (ALNSConfig overrides, for dev tuning; preset='"v1"': Phase 1)
  run_alns.py --settings MA-AT-50-5-50 --time B1 --workers 8   (B1 = paper RL(s.10) time, B2 = 2 x B1)

Each instance runs in its own single-threaded process (``--procs`` at a time) with ``--workers`` ALNS workers, so
at most procs * workers processes are busy. Rows record the instance ``fingerprint`` and the ``git`` code version;
rows already present in the output file are skipped (resumable) if their fingerprint matches the current instance.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from cbba_sota.bench import runtime

THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS")


def instances_from_json(path: str | Path) -> dict:
    """Instances of the red-team JSON export ({name: {req, loc, dur, ab, dep}}); species = distinct (traits, depot)."""
    from cbba_sota.hetero import Instance

    out = {}
    for name, d in json.loads(Path(path).read_text()).items():
        ab, dep = np.array(d["ab"], float), np.array(d["dep"], float)
        _, first, species = np.unique(np.hstack([ab, dep]), axis=0, return_index=True, return_inverse=True)
        rank = np.argsort(np.argsort(first))  # species ids in order of first appearance
        stem = Path(name).stem
        out[f"{Path(name).parent.name}/{stem}"] = Instance(req=d["req"], loc=d["loc"], dur=d["dur"], ab=ab,
                                                           depot=dep, species=rank[species.ravel()],
                                                           name=f"{Path(name).parent.name}/{stem}")
    return out


def parse_overrides(items: list[str]) -> dict:
    from cbba_sota.solvers.alns import ALNSConfig

    fields = {f.name: f for f in dataclasses.fields(ALNSConfig)} | {"preset": None}
    out = {}
    for item in items:
        key, value = item.split("=", 1)
        if key not in fields:
            raise SystemExit(f"unknown ALNSConfig field {key!r}")
        out[key] = json.loads(value)
        if isinstance(out[key], list):
            out[key] = tuple(out[key])
    return out


def budget(setting, spec: str) -> float:
    """Seconds for ``spec``: a number, or B1 / B2 (the paper's RL(s.10) time of ``setting``, times 1 / 2), optionally
    scaled (``0.5B1``, ``5B1``)."""
    spec = spec.upper()
    if spec.endswith(("B1", "B2")):
        scale = float(spec[:-2] or 1) * (2 if spec.endswith("B2") else 1)
        return setting.paper["RL(s.10)"].time_s * scale
    return float(spec)


def _warm() -> None:
    """Pool initializer: load the numba kernels before any budget starts."""
    from cbba_sota.solvers.alns import _warmup

    _warmup()


def _job(key: dict, inst, time_limit: float, seed: int, workers: int, overrides: dict) -> dict:
    from cbba_sota.hetero import evaluate, replay
    from cbba_sota.solvers.alns import ALNSConfig, solve

    fields = dict(overrides)
    base = ALNSConfig.v1() if fields.pop("preset", "v2") == "v1" else ALNSConfig()  # preset="v1": Phase 1 ALNS
    cfg = dataclasses.replace(base, **fields)
    inst.tt, inst.da  # noqa: B018  (travel matrices are instance loading, outside the time budget)
    t = time.monotonic()
    plan, st = solve(inst, time_limit_s=time_limit, seed=seed, n_workers=workers, config=cfg)
    wall = time.monotonic() - t
    rep = replay(inst, plan)
    ev = evaluate(inst, plan)
    return {**key, "time_limit": time_limit, "seed": seed, "workers": workers, "makespan": rep["makespan"],
            "success": rep["success"], "env_finished": rep["env_finished"], "awt": rep["awt"],
            "skipped": rep["skipped"], "eval_makespan": ev.makespan,
            "search_makespan": st.makespan, "init_makespan": st.init_makespan, "iterations": st.iterations,
            "insertions": st.insertions,
            "candidate_slots": st.candidate_slots, "exact_evals": st.exact_evals,
            "it_per_s": st.it_per_s, "wall_s": wall, "search_s": st.search_s, "trace": st.trace,
            "destroy_weights": st.destroy_weights, "repair_weights": st.repair_weights, "overrides": overrides,
            "routes": plan.to_env_routes(), "fingerprint": runtime.fingerprint(inst)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--settings", nargs="*", default=[])
    ap.add_argument("--split", default="dev")
    ap.add_argument("--json", help="red-team JSON export instead of settings")
    ap.add_argument("--n", type=int, default=None, help="first N instances per setting")
    ap.add_argument("--time", required=True, help="seconds, or B1 / B2 per setting")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--procs", type=int, default=30)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--set", nargs="*", default=[], dest="overrides")
    args = ap.parse_args()
    for var in THREAD_VARS:
        os.environ[var] = "1"

    from cbba_sota.bench import configs
    from cbba_sota.hetero import Instance

    overrides = parse_overrides(args.overrides)
    tag = args.tag or f"t{args.time}-w{args.workers}-s{args.seed}"
    root = configs.RUNS_DIR / "alns" / tag
    groups: dict[Path, list[tuple[dict, object]]] = {}
    if args.json:
        insts = list(instances_from_json(args.json).items())[:args.n]
        out = root / f"{Path(args.json).stem}.jsonl"
        groups[out] = [({"setting": Path(args.json).stem, "split": "json", "index": i, "name": name}, inst,
                        float(args.time)) for i, (name, inst) in enumerate(insts)]
    for name in args.settings:
        s = configs.get(name)
        n = min(args.n or s.n_instances(args.split), s.n_instances(args.split))
        out = root / f"{s.name}-{args.split}.jsonl"
        groups[out] = [({"setting": s.name, "split": args.split, "index": i, "name": f"{s.name}/{args.split}/{i}"},
                        s.instance_path(args.split, i), budget(s, args.time)) for i in range(n)]
    jobs = []
    for out, items in groups.items():
        out.parent.mkdir(parents=True, exist_ok=True)
        current = {key["name"]: runtime.fingerprint(inst) if isinstance(inst, Instance) else
                   runtime.instance_fingerprint(key["setting"], key["split"], key["index"]) for key, inst, _ in items}
        done = {r["name"] for r in runtime.read_rows(out) if r.get("fingerprint", 0) == current.get(r["name"])}
        jobs += [(out, key, inst, limit) for key, inst, limit in items if key["name"] not in done]
    print(f"{len(jobs)} jobs -> {root}", flush=True)
    version = runtime.code_version()
    import multiprocessing as mp

    with ProcessPoolExecutor(args.procs, mp_context=mp.get_context("spawn"), initializer=_warm) as ex:
        futures = {}
        for out, key, inst, limit in jobs:
            inst = inst if isinstance(inst, Instance) else Instance.from_pickle(inst)
            futures[ex.submit(_job, key, inst, limit, args.seed, args.workers, overrides)] = out
        for fut in as_completed(futures):
            row = fut.result() | {"git": version}
            with futures[fut].open("a") as f:
                f.write(json.dumps(row) + "\n")
    for out in groups:
        rows = [json.loads(line) for line in out.read_text().splitlines()]
        ms = np.array([r["makespan"] for r in rows])
        exact = all(r["makespan"] == r["eval_makespan"] and r["skipped"] == 0 for r in rows)
        print(f"{out.relative_to(configs.ROOT)}: n={len(rows)} mean makespan {ms.mean():.3f} "
              f"success {np.mean([r['success'] for r in rows]):.2f} replay==evaluator {exact} "
              f"it/s {np.mean([r['it_per_s'] for r in rows]):.0f} "
              f"init {np.mean([r['init_makespan'] for r in rows]):.3f}")


if __name__ == "__main__":
    main()
