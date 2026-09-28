"""B1 (published RL policy, online) on the Track D dev split: resumable JSONL rows plus a summary.

    OMP_NUM_THREADS=1 .venv/bin/python scripts/trackD_rl_online.py --variants x-decision x-arrival --procs 7
    .venv/bin/python scripts/trackD_rl_online.py --summary-only

Rows go to runs/trackD/rl_online/rows.jsonl, one per (setting, instance, cell, CRN seed, method, variant); a row's
key is its ``id``, so reruns skip finished work. Every row records the world that produced it (``world``) and the
git commit of the code. Methods: RL(g.) (greedy decode) and RL(s.1) (one sampled action per decision); both use
``policy_seed(instance key, CRN seed, method)``, so the variants of one method face the same policy randomness.
Dev split only; the test split is refused by ``make_world``.

Variants (``cbba_sota.dyn.baselines.rl_online``):
- ``x-decision``: DynTaskEnvX (ground truth, D1-D8, failures), native decision-order coalitions (the env's default
  for policies; bit-identical static reduction).
- ``x-arrival``: DynTaskEnvX with D2 arrival-order coalitions, the rule plan-following methods are scored under.
- ``pilot``: the day-1 prototype env (``pilots/trackD/dyn_env.py``), no failures; kept as the day-1 record.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "trackD" / "rl_online"
SETTINGS = ("MA-AT-25-5-50", "MA-AT-50-5-50", "SA-AT-50-5-50", "SA-BT-50-5-50")
CELLS = ("static", "F0-N1", "F0-N2", "F0-N3", "F1-R1", "F1-R2", "F1-R3", "F2-R2N3", "F3-pf0.1", "F3-pf0.2",
         "F3-pf0.2-R2")
METHODS = ("RL(g.)", "RL(s.1)")
VARIANTS = {"x-decision": ("x", "decision"), "x-arrival": ("x", "arrival"), "pilot": ("pilot", "decision")}

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_k, "1")


def git_commit() -> str:
    try:
        head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                              check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "cbba_sota/dyn", "scripts/trackD_rl_online.py"],
                               cwd=ROOT, capture_output=True, text=True, check=False).stdout.strip()
        return head + ("+dirty(dyn)" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def row_id(setting, inst, cell, seed, method, variant) -> str:
    return f"{setting}|{inst}|{cell}|{seed}|{method}|{variant}"


_NET = None
_ENV_HASH = None  # hash of cbba_sota/dyn/{env,perturb}.py as imported by this worker


def _init():
    global _NET, _ENV_HASH
    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    from cbba_sota.dyn import env as dyn_env
    from cbba_sota.dyn.baselines import rl_online

    # DynTaskEnvX.world hashes the files on disk when called, so it drifts if the env is edited during a run while
    # this worker keeps executing the code it imported. Rows record the import-time hash, and flag later edits.
    _ENV_HASH = dyn_env.code_hash()
    _NET = rl_online.load_policy("cpu")


def run_job(job: dict) -> list[dict]:
    from cbba_sota.bench import configs
    from cbba_sota.dyn.baselines import rl_online as B1

    key = configs.get(job["setting"]).seed("dev", job["inst"])
    out = []
    for method, variant in job["todo"]:
        world, coalition = VARIANTS[variant]
        env, label = B1.make_world(job["setting"], "dev", job["inst"], job["seed"], job["cell"], world=world,
                                   coalition=coalition)
        rz = env.realization
        base = {"id": row_id(job["setting"], job["inst"], job["cell"], job["seed"], method, variant),
                "setting": job["setting"], "inst": job["inst"], "inst_key": key, "cell": job["cell"],
                "family": rz.cell.family, "seed": job["seed"], "variant": variant, "H": rz.H,
                "n_released_at_0": int((rz.release <= 0).sum()), "excluded": bool(rz.excluded),
                "commit": job["commit"]}
        r = B1.run_b1(env, _NET, sample=B1.METHODS[method], seed=B1.policy_seed(key, job["seed"], method),
                      method=method, world=label)
        row = {**base, **r.row()}
        if world == "x":
            from cbba_sota.dyn.env import code_hash

            row["world"] = row["world"].rsplit("@", 1)[0] + f"@{_ENV_HASH}"
            row["env_code_hash"] = _ENV_HASH
            row["env_code_drift"] = code_hash() != _ENV_HASH  # the env files changed on disk after import
        out.append(row)
    return out


def load_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    for r in rows:  # rows of the first (pilot-only) run had no variant
        r.setdefault("variant", "pilot")
        if r["id"].count("|") == 4:
            r["id"] += "|pilot"
    return rows


def summarize(rows: list[dict]) -> str:
    import numpy as np

    from cbba_sota.bench import configs
    from cbba_sota.dyn import stats as S

    lines = [(f"rows: {len(rows)}; commits: {sorted({r['commit'] for r in rows})}; env code (import-time hash): "
              f"{sorted({r.get('env_code_hash', '-') for r in rows})}; rows with env files edited mid-run: "
              f"{sum(r.get('env_code_drift', False) for r in rows)}")]
    for w in sorted({(r["variant"], r["world"]) for r in rows}):
        lines.append(f"  variant {w[0]}: world {w[1]}")
    rows = [r for r in rows if not r.get("excluded")]
    variants = [v for v in VARIANTS if any(r["variant"] == v for r in rows)]
    cells = [c for c in CELLS if any(r["cell"] == c for r in rows)]
    by = defaultdict(list)
    for r in rows:
        by[(r["variant"], r["setting"], r["cell"], r["method"])].append(r)
    for v in variants:
        lines += ["", (f"[{v}] mean makespan (success rate) per setting x cell; dev instances x CRN seeds, excluded "
                   "failure realizations dropped")]
        lines.append(f"{'setting':<15} {'method':<8} " + " ".join(f"{c[:13]:>13}" for c in cells))
        for st in SETTINGS:
            for m in METHODS:
                vals = []
                for c in cells:
                    g = by.get((v, st, c, m), [])
                    if not g:
                        vals.append(f"{'-':>13}")
                        continue
                    vals.append(f"{np.mean([x['makespan'] for x in g]):7.2f} ({np.mean([x['success'] for x in g]):.2f})")
                lines.append(f"{st:<15} {m:<8} " + " ".join(vals))
    lines += ["", "paper RL(g.) static mean makespan (test split): " + ", ".join(
        f"{st} {configs.get(st).paper['RL(g.)'].makespan}" for st in SETTINGS)]

    def gm_line(label, d):
        if len(set(d["clusters"])) < 2:
            return None
        g = S.gm_ratio(d["log_r"], d["clusters"], strata=d["strata"])
        return (f"  {label:<12} {g.ratio:.3f} [{g.lo:.3f}, {g.hi:.3f}]  instances {g.n_clusters}, pairs {g.n_obs}, "
                f"wins {g.wins}/{g.n_clusters}, dropped pairs {int((~d['keep']).sum())}")

    for v in variants:
        lines += ["", (f"[{v}] cost of dynamics to RL(g.): GM makespan(cell) / makespan(static) on the same instance "
                   "(cluster bootstrap over instances, seeds nested, strata = settings; failures dropped)")]
        stat = {(r["setting"], r["inst"]): r for r in rows
                if r["variant"] == v and r["cell"] == "static" and r["method"] == "RL(g.)"}
        for c in cells:
            if c == "static":
                continue
            sel = [r for r in rows if r["variant"] == v and r["cell"] == c and r["method"] == "RL(g.)"
                   and r["success"] and (r["setting"], r["inst"]) in stat]
            if len({(r["setting"], r["inst"]) for r in sel}) < 2:
                continue
            lr = [float(np.log(r["makespan"] / stat[(r["setting"], r["inst"])]["makespan"])) for r in sel]
            cl = [(r["setting"], r["inst"]) for r in sel]
            g = S.gm_ratio(lr, cl, strata=[x[0] for x in cl])
            lines.append(f"  {c:<12} {g.ratio:.3f} [{g.lo:.3f}, {g.hi:.3f}]  instances {g.n_clusters}, "
                         f"episodes {g.n_obs}")
    for v in variants:
        lines += ["", f"[{v}] RL(s.1) / RL(g.), paired by instance, cell and seed, pooled over settings"]
        for c in cells:
            sub = [r for r in rows if r["variant"] == v and r["cell"] == c]
            ln = gm_line(c, S.pair_rows(sub, "RL(s.1)", "RL(g.)", pair_keys=("setting", "inst", "cell", "seed")))
            if ln:
                lines.append(ln)
    if "x-arrival" in variants and "x-decision" in variants:
        lines += ["", ("Coalition rule: RL under arrival order (D2) / RL under decision order, same realization and "
                   "policy seed (GM, pooled over settings; per setting in brackets)")]
        for m in METHODS:
            for c in cells:
                sub = [dict(r, arm=r["variant"]) for r in rows if r["method"] == m and r["cell"] == c
                       and r["variant"] in ("x-arrival", "x-decision")]
                d = S.pair_rows(sub, "x-arrival", "x-decision", method_key="arm",
                                pair_keys=("setting", "inst", "cell", "seed"))
                ln = gm_line(f"{m} {c}", d)
                if ln:
                    per = []
                    for st in SETTINGS:
                        ds = S.pair_rows([r for r in sub if r["setting"] == st], "x-arrival", "x-decision",
                                         method_key="arm", pair_keys=("setting", "inst", "cell", "seed"))
                        if ds["log_r"].size:
                            per.append(f"{st.split('-5-')[0]} {np.exp(ds['log_r'].mean()):.3f}")
                    lines.append(ln + "  [" + ", ".join(per) + "]")
    if "pilot" in variants and "x-decision" in variants:
        lines += ["", ("World check: pilot env vs DynTaskEnvX (decision order), RL(g.), same realization and seed: "
                   "share of episodes with identical makespan (1e-9) per cell")]
        idx = {(r["setting"], r["inst"], r["cell"], r["seed"], r["method"]): r for r in rows if r["variant"] == "pilot"}
        for c in cells:
            pairs = [(r, idx.get((r["setting"], r["inst"], r["cell"], r["seed"], r["method"]))) for r in rows
                     if r["variant"] == "x-decision" and r["cell"] == c]
            pairs = [(a, b) for a, b in pairs if b is not None]
            if pairs:
                same = np.mean([abs(a["makespan"] - b["makespan"]) < 1e-9 for a, b in pairs])
                lines.append(f"  {c:<12} {same:.3f} of {len(pairs)}")
    lines += ["", ("Per-decision CPU (observation + forward pass, 1 thread, loaded shared host; not a quiet-core "
               "timing), RL(g.)")]
    for v in variants:
        for st in SETTINGS:
            g = [r for r in rows if r["variant"] == v and r["setting"] == st and r["method"] == "RL(g.)"]
            if g:
                lines.append(f"  [{v}] {st:<15} p50 {np.median([r['cpu_ms_p50'] for r in g]):5.2f} ms, median "
                             f"episode p95 {np.median([r['cpu_ms_p95'] for r in g]):5.2f} ms, decisions/episode "
                             f"{np.mean([r['n_decisions'] for r in g]):6.1f}")
    lines += ["", "Unsuccessful episodes (excluded realizations dropped): " + ", ".join(
        f"{v} {sum(not r['success'] for r in rows if r['variant'] == v)}/{sum(r['variant'] == v for r in rows)}"
        for v in variants)]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--settings", nargs="+", default=list(SETTINGS))
    ap.add_argument("--cells", nargs="+", default=list(CELLS))
    ap.add_argument("--variants", nargs="+", default=["x-decision", "x-arrival"], choices=list(VARIANTS))
    ap.add_argument("--methods", nargs="+", default=list(METHODS), choices=list(METHODS))
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    ap.add_argument("--procs", type=int, default=7)
    ap.add_argument("--out", default=str(OUT / "rows.jsonl"))
    ap.add_argument("--summary-only", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not args.summary_only:
        done = {r["id"] for r in load_rows(out)}
        commit = git_commit()
        jobs = []
        for st in args.settings:
            for i in range(args.n):
                for c in args.cells:
                    for sd in ([0] if c == "static" else args.seeds):
                        todo = [(m, v) for v in args.variants for m in args.methods
                                if row_id(st, i, c, sd, m, v) not in done
                                and not (v == "pilot" and c.startswith("F3"))]
                        if todo:
                            jobs.append({"setting": st, "inst": i, "cell": c, "seed": sd, "todo": todo,
                                         "commit": commit})
        print(f"{len(jobs)} jobs, {args.procs} processes, commit {commit}", flush=True)
        t0 = time.time()
        with mp.get_context("spawn").Pool(args.procs, initializer=_init) as pool, out.open("a") as f:
            for k, rows in enumerate(pool.imap_unordered(run_job, jobs)):
                for r in rows:
                    f.write(json.dumps(r) + "\n")
                f.flush()
                if k % 100 == 0:
                    print(f"{k}/{len(jobs)} {time.time() - t0:.0f}s", flush=True)
        print(f"done in {time.time() - t0:.0f}s", flush=True)
    text = summarize(load_rows(out))
    (out.parent / "summary.txt").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    sys.exit(main())
