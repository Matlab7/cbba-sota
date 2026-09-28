"""ALNS variants side by side on the dev split: interleaved runs and the paired ablation table.

Usage:
  alns_ablation.py run SPEC.json --root v2abl/w1 [--settings ...] [--n 20] [--time B1] [--workers 1] [--procs 40]
  alns_ablation.py report --root v2abl/w1 REF [LABEL ...] [--csv FILE]
  alns_ablation.py anytime --root v2abl/w1 LABEL [LABEL ...] [--cpsat] --png FILE

SPEC maps a label to ALNSConfig overrides of the v2 defaults (``{"preset": "v1"}`` starts from the Phase 1
configuration). ``run`` shuffles the jobs of all variants (seed 0) into one pool, so every variant sees the same
host conditions, and writes run_alns rows (env-replayed makespans) to runs/alns/<root>/<label>/<setting>-dev.jsonl;
it resumes. ``report`` prints, per setting, each variant's mean makespan, its paired ratio to REF with a bootstrap
95% CI (10,000 resamples, seed 0) and wins/losses, and its paired ratio to CPSAT-8 at the same budget label from
runs/compare_dev (B1/B2 runs only). A failed run counts as 200. ``anytime`` plots, per setting, the mean ratio of each
variant's best-so-far makespan (from the stored traces) to CPSAT-8's final makespan against the share of the
budget; ``--cpsat`` adds the CP-SAT LNS traces of runs/alns/cpsat_ref at the same budget.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_alns

from cbba_sota.bench import configs, runtime
from cbba_sota.bench.configs import MAX_TIME, RUNS_DIR

SETTINGS = ("MA-AT-25-5-50", "SA-AT-50-5-50", "MA-AT-50-5-50", "SA-BT-50-5-50", "MA-AT-50-5-200", "SA-BT-25-5-50")


def _rows(path: Path) -> dict[int, dict]:
    """Rows by instance index, only those computed on the current instance (the last one wins)."""
    return {r["index"]: r for r in runtime.read_rows(path) if runtime.is_current(r)}


def run(args) -> None:
    for var in run_alns.THREAD_VARS:
        os.environ[var] = "1"
    from cbba_sota.hetero import Instance

    spec = json.loads(Path(args.spec).read_text())
    jobs = []
    for label, overrides in spec.items():
        overrides = {k: tuple(v) if isinstance(v, list) else v for k, v in overrides.items()}
        for name in args.settings:
            s = configs.get(name)
            out = RUNS_DIR / "alns" / args.root / label / f"{name}-{args.split}.jsonl"
            done = _rows(out)
            n = min(args.n or s.n_instances(args.split), s.n_instances(args.split))
            jobs += [(out, {"setting": name, "split": args.split, "index": i, "name": f"{name}/{args.split}/{i}",
                            "label": label}, s.instance_path(args.split, i), run_alns.budget(s, args.time),
                      overrides) for i in range(n) if i not in done]
    random.Random(0).shuffle(jobs)
    version = runtime.code_version()
    procs = max(1, args.procs // max(1, args.workers))
    print(f"{len(jobs)} jobs, {procs} at a time x {args.workers} workers", flush=True)
    with ProcessPoolExecutor(procs, mp_context=mp.get_context("spawn"), initializer=run_alns._warm) as ex:
        futures = {ex.submit(run_alns._job, key, Instance.from_pickle(path), limit, args.seed, args.workers,
                             overrides): out for out, key, path, limit, overrides in jobs}
        for fut in as_completed(futures):
            row = fut.result() | {"git": version, "load1": os.getloadavg()[0]}
            futures[fut].parent.mkdir(parents=True, exist_ok=True)
            with futures[fut].open("a") as f:
                f.write(json.dumps(row) + "\n")


def _score(r: dict) -> float:
    return r["makespan"] if r["success"] and r["makespan"] < MAX_TIME else MAX_TIME


def _boot(x: np.ndarray, reps: int = 10_000) -> tuple[float, float]:
    rng = np.random.default_rng(0)
    means = x[rng.integers(0, len(x), (reps, len(x)))].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _cpsat(budget_s: float) -> dict[tuple[str, int], float]:
    """CPSAT-8 of runs/compare_dev at the budget label (B1 or B2) whose seconds equal ``budget_s``."""
    out = {}
    for path in (RUNS_DIR / "compare_dev").glob("*.jsonl"):
        for r in map(json.loads, path.read_text().splitlines()):
            if r["method"] == "CPSAT-8" and abs(r["budget_s"] - budget_s) < 1e-6:
                out[r["setting"], r["instance"]] = _score(r)
    return out


def report(args) -> None:
    labels = [args.ref, *args.labels]
    root = RUNS_DIR / "alns" / args.root
    data = {lab: {p.name.rsplit("-", 1)[0]: _rows(p) for p in sorted((root / lab).glob("*.jsonl"))}
            for lab in labels}
    lines = ["setting,label,n,mean,ratio_ref,ci_lo,ci_hi,wins,losses,ratio_cpsat8,it_per_s,last_improvement_frac"]
    for setting in sorted({s for d in data.values() for s in d}):
        ref = data[args.ref].get(setting, {})
        print(setting)
        for lab in labels:
            rows = data[lab].get(setting, {})
            if not rows:
                continue
            idx = sorted(rows)
            ms = np.array([_score(rows[i]) for i in idx])
            text = f"  {lab:22s} n={len(idx):2d} mean {ms.mean():8.3f}"
            ratio_ref = lo = hi = wins = losses = np.nan
            common = [i for i in idx if i in ref]
            if lab != args.ref and common:
                r = np.array([_score(rows[i]) / _score(ref[i]) for i in common])
                ratio_ref, (lo, hi) = r.mean(), _boot(r)
                wins, losses = int((r < 1 - 1e-9).sum()), int((r > 1 + 1e-9).sum())
                text += f"  /ref {ratio_ref:.4f} [{lo:.4f}, {hi:.4f}] {wins:2d}-{losses:2d}"
            cp = _cpsat(rows[idx[0]]["time_limit"])
            cc = [i for i in idx if (setting, i) in cp]
            ratio_cp = np.mean([_score(rows[i]) / cp[setting, i] for i in cc]) if cc else np.nan
            if cc:
                text += f"  /CPSAT-8 {ratio_cp:.4f}"
            its = np.mean([rows[i]["it_per_s"] for i in idx])
            last = np.median([rows[i]["trace"][-1][0] / rows[i]["time_limit"] for i in idx])
            print(text + f"  it/s {its:6.0f}  last {last:.2f}")
            lines.append(f"{setting},{lab},{len(idx)},{ms.mean():.4f},{ratio_ref:.4f},{lo:.4f},{hi:.4f},{wins},"
                         f"{losses},{ratio_cp:.4f},{its:.0f},{last:.3f}")
    if args.csv:
        Path(args.csv).write_text("\n".join(lines) + "\n")


def _at(trace: list, t: float) -> float:
    """Best-so-far makespan of a trace at time ``t`` (nan before the first point)."""
    vals = [ms for x, ms in trace if x <= t]
    return vals[-1] if vals else np.nan


def anytime(args) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = RUNS_DIR / "alns" / args.root
    data = {lab: {p.name.rsplit("-", 1)[0]: _rows(p) for p in sorted((root / lab).glob("*.jsonl"))}
            for lab in args.labels}
    settings = sorted({s for d in data.values() for s in d})
    fracs = np.linspace(0.02, 1.0, 50)
    fig, axes = plt.subplots(1, len(settings), figsize=(4 * len(settings), 3.4), squeeze=False)
    for ax, setting in zip(axes[0], settings):
        curves = {lab: data[lab].get(setting, {}) for lab in args.labels}
        if args.cpsat:
            ref = {r["index"]: r for r in runtime.read_rows(RUNS_DIR / "alns" / "cpsat_ref" / f"{setting}-dev.jsonl")
                   if runtime.is_current(r)}
            limit = next(iter(next(iter(curves.values())).values()))["time_limit"]
            curves["CP-SAT LNS x8"] = {i: r | {"time_limit": r["budget_s"]} for i, r in ref.items()
                                       if r["workers"] == 8 and abs(r["budget_s"] - limit) < 1e-6}
        limit = next(iter(next(iter(curves.values())).values()))["time_limit"]
        cp = _cpsat(limit)
        for lab, rows in curves.items():
            idx = [i for i in sorted(rows) if (setting, i) in cp]
            if not idx:
                continue
            ys = [np.nanmean([_at(rows[i]["trace"], f * limit) / cp[setting, i] for i in idx]) for f in fracs]
            ax.plot(fracs, ys, label=f"{lab} (n={len(idx)})")
            print(setting, lab, " ".join(f"{f:.2f}:{y:.4f}" for f, y in zip(fracs[4::5], ys[4::5])))
        ax.axhline(1.0, color="grey", lw=0.8, ls=":")
        ax.set_title(f"{setting} ({limit:g} s)")
        ax.set_xlabel("share of budget")
        ax.set_ylim(top=1.1)
    axes[0][0].set_ylabel("best makespan / CPSAT-8 final")
    axes[0][-1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(args.png, dpi=120)
    print(f"wrote {args.png}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("spec")
    r.add_argument("--root", required=True)
    r.add_argument("--settings", nargs="*", default=list(SETTINGS))
    r.add_argument("--split", default="dev")
    r.add_argument("--n", type=int, default=20)
    r.add_argument("--time", default="B1")
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--workers", type=int, default=1)
    r.add_argument("--procs", type=int, default=40, help="busy processes (jobs x workers)")
    p = sub.add_parser("report")
    p.add_argument("--root", required=True)
    p.add_argument("ref")
    p.add_argument("labels", nargs="*")
    p.add_argument("--csv")
    a = sub.add_parser("anytime")
    a.add_argument("--root", required=True)
    a.add_argument("labels", nargs="+")
    a.add_argument("--cpsat", action="store_true")
    a.add_argument("--png", required=True)
    args = ap.parse_args()
    if args.cmd == "anytime":
        anytime(args)
    elif args.cmd == "run":
        if args.split == "test":
            raise SystemExit("ablations run on dev (or validation) splits only")
        run(args)
    else:
        report(args)


if __name__ == "__main__":
    main()
