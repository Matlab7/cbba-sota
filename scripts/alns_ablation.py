"""ALNS variants side by side on the dev split: interleaved runs and the paired ablation table.

Usage:
  alns_ablation.py run SPEC.json --root v2abl/w1 [--settings ...] [--n 20] [--time B1] [--workers 1] [--procs 40]
                       [--seed 0]
  alns_ablation.py report --root v2abl/w1 REF [LABEL ...] [--csv FILE]
  alns_ablation.py anytime --root v2abl/w1 LABEL [LABEL ...] [--cpsat] --png FILE
  alns_ablation.py budgets --roots v2e/t1 v2e/t2 ... --labels v1 v2 [--ref CPSAT-8] [--ref-budget B1] [--csv FILE]
                           [--png FILE]
  alns_ablation.py ttt --root v2e/B1 LABEL [LABEL ...] [--ref CPSAT-8]

SPEC maps a label to ALNSConfig overrides of the v2 defaults (``{"preset": "v1"}`` starts from the Phase 1
configuration). ``run`` shuffles the jobs of all variants into one pool, so every variant sees the same host
conditions, pins every pool process to its own ``--workers`` CPUs (``--cpus``, default the least busy physical
cores), refuses a dirty source tree unless ``--allow-dirty``, and writes run_alns rows (env-replayed makespans,
1-based routes) to runs/alns/<root>/<label>/<setting>-dev.jsonl; it resumes per (instance, seed), and reports
average the seeds of an instance. ``report`` prints, per setting, each
variant's mean makespan, its paired ratio to REF with a bootstrap 95% CI (10,000 resamples, seed 0) and
wins/losses, and its paired ratio to CPSAT-8 at the same budget. A failed run counts as 200. ``anytime`` plots,
per setting, the mean ratio of each variant's best-so-far makespan (from the stored traces) to CPSAT-8's final
makespan against the share of the budget; ``--cpsat`` adds the CP-SAT LNS traces of runs/alns/cpsat_ref.
``budgets`` compares runs at several budgets (one root per budget) with a competitor at one budget (default B1),
per label and core count. ``ttt`` reads the time at which each run's best-so-far first reached that competitor's
B1 makespan. Competitor rows come from runs/compare_dev, runs/alns/cpsat_ref and runs/baselines (``_reference``).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from concurrent.futures import as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_alns

from cbba_sota.bench import configs, runtime
from cbba_sota.bench.configs import MAX_TIME, RUNS_DIR

SETTINGS = ("MA-AT-25-5-50", "SA-AT-50-5-50", "MA-AT-50-5-50", "SA-BT-50-5-50", "MA-AT-50-5-200", "SA-BT-25-5-50")


def _runs(path: Path) -> dict[int, dict[int, dict]]:
    """Rows by instance index and seed, only those computed on the current instance (the last one wins)."""
    out: dict[int, dict[int, dict]] = {}
    for r in runtime.read_rows(path):
        if runtime.is_current(r):
            out.setdefault(r["index"], {})[r["seed"]] = r
    return out


def _rows(path: Path) -> dict[int, dict]:
    """One row per instance: that of the lowest seed, with ``makespan`` replaced by the mean score over its seeds
    (``seeds`` counts them) and ``success`` by all of them succeeding."""
    out = {}
    for i, by_seed in _runs(path).items():
        rows = [by_seed[k] for k in sorted(by_seed)]
        out[i] = rows[0] | {"makespan": float(np.mean([_score(r) for r in rows])), "seeds": len(rows),
                            "success": all(r["success"] for r in rows)}
    return out


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
            done = {i for i, by_seed in _runs(out).items() if args.seed in by_seed}
            n = min(args.n or s.n_instances(args.split), s.n_instances(args.split))
            jobs += [(out, {"setting": name, "split": args.split, "index": i, "name": f"{name}/{args.split}/{i}",
                            "label": label}, s.instance_path(args.split, i), run_alns.budget(s, args.time),
                      overrides) for i in range(n) if i not in done]
    random.Random(0).shuffle(jobs)
    version = runtime.require_clean(args.allow_dirty)
    procs = max(1, min(args.procs // max(1, args.workers), len(jobs)))
    print(f"{len(jobs)} jobs, {procs} at a time x {args.workers} workers", flush=True)
    cpus = runtime.parse_cpus(args.cpus) if args.cpus else None
    with runtime.pinned_pool(procs, max(1, args.workers), cpus, initializer=run_alns._warm) as ex:
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


def _reference(budget_s: float, method: str = "CPSAT-8") -> dict[tuple[str, int], float]:
    """Scores of a competitor per (setting, instance) at ``budget_s`` seconds: ``method`` rows of runs/compare_dev,
    then (CPSAT-8 only) the CP-SAT LNS rows of runs/alns/cpsat_ref (per setting the sub-solve time with the best
    mean), then ``method`` rows of runs/baselines (e.g. ``construct`` at 1 s) where the earlier sources have none.
    Stale rows are dropped."""
    out = {}
    for path in (RUNS_DIR / "compare_dev").glob("*.jsonl"):
        for r in runtime.read_rows(path):
            if r["method"] == method and abs(r["budget_s"] - budget_s) < 1e-6 and not runtime.is_stale(r):
                out[r["setting"], r["instance"]] = _score(r)
    if method == "CPSAT-8":  # per setting, the LNS sub-solve time with the best mean (the best variant on dev)
        variants: dict[tuple[str, float], dict[int, float]] = {}
        for path in (RUNS_DIR / "alns" / "cpsat_ref").glob("*-dev.jsonl"):
            for r in runtime.read_rows(path):
                if r["workers"] == 8 and abs(r["budget_s"] - budget_s) < 1e-6 and runtime.is_current(r):
                    variants.setdefault((r["setting"], r.get("sub_time", 2.0)), {})[r["index"]] = _score(r)
        for setting in {s for s, _ in variants}:
            runs = [scores for (s, _), scores in sorted(variants.items()) if s == setting]
            common = set.intersection(*(set(scores) for scores in runs))  # compare the variants on shared instances
            best = min(runs, key=lambda scores: (np.mean([scores[i] for i in common]) if common else 0.0, -len(scores)))
            for i, v in best.items():
                out.setdefault((setting, i), v)
    for path in (RUNS_DIR / "baselines").glob("*/dev.jsonl"):
        for r in runtime.read_rows(path):
            if (r["method"] == method and r.get("budget_s") is not None and abs(r["budget_s"] - budget_s) < 1e-6
                    and not runtime.is_stale(r)):
                out.setdefault((r["setting"], r["instance"]), _score(r))
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
            cp = _reference(rows[idx[0]]["time_limit"])
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
        cp = _reference(limit)
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


def budgets(args) -> None:
    """Quality against the budget: per setting, label and core count, the mean paired ratio of each budget's runs to
    ``args.ref`` at ``args.ref_budget`` (bootstrap 95% CI, wins), optionally plotted against seconds."""
    rows: dict[tuple, dict[int, dict]] = {}
    for root in args.roots:
        for lab in args.labels:
            for path in sorted((RUNS_DIR / "alns" / root / lab).glob("*-dev.jsonl")):
                for i, r in _rows(path).items():
                    rows.setdefault((r["setting"], lab, r["workers"], r["time_limit"]), {})[i] = r
    b1 = {s: run_alns.budget(configs.get(s), args.ref_budget) for s in {k[0] for k in rows}}
    lines = ["setting,label,workers,budget_s,n,mean,ratio,ci_lo,ci_hi,wins,losses"]
    curves: dict[tuple, list] = {}
    for key in sorted(rows):
        setting, lab, workers, limit = key
        ref = _reference(b1[setting], args.ref)
        idx = [i for i in sorted(rows[key]) if (setting, i) in ref]
        ms = np.array([_score(rows[key][i]) for i in sorted(rows[key])])
        text = f"{setting:18s} {lab:14s} x{workers} {limit:7.2f}s n={len(ms):2d} mean {ms.mean():8.3f}"
        ratio = lo = hi = np.nan
        wins = losses = 0
        if idx:
            r = np.array([_score(rows[key][i]) / ref[setting, i] for i in idx])
            ratio, (lo, hi) = r.mean(), _boot(r)
            wins, losses = int((r < 1 - 1e-9).sum()), int((r > 1 + 1e-9).sum())
            text += (f"  /{args.ref}@{args.ref_budget} {ratio:.4f} [{lo:.4f}, {hi:.4f}] {wins:2d}-{losses:2d} "
                     f"(n={len(idx)})")
            curves.setdefault((setting, lab, workers), []).append((limit, ratio, lo, hi))
        print(text)
        lines.append(f"{setting},{lab},{workers},{limit},{len(ms)},{ms.mean():.4f},{ratio:.4f},{lo:.4f},{hi:.4f},"
                     f"{wins},{losses}")
    if args.csv:
        Path(args.csv).write_text("\n".join(lines) + "\n")
    if args.png:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        settings = sorted({k[0] for k in curves})
        cols = min(4, len(settings))
        rows = -(-len(settings) // cols)
        fig, axes = plt.subplots(rows, cols, figsize=(3.8 * cols, 3.2 * rows), squeeze=False)
        for ax in axes.flat[len(settings):]:
            ax.axis("off")
        colors = {key: f"C{k}" for k, key in enumerate(sorted({(lab, w) for _, lab, w in curves}))}
        for ax, setting in zip(axes.flat, settings):
            for (s, lab, workers), pts in sorted(curves.items()):
                if s == setting:
                    x, y, lo, hi = map(np.array, zip(*sorted(pts)))
                    c = colors[lab, workers]
                    ax.plot(x, y, marker="o", ms=3, color=c, label=f"{lab} x{workers}")
                    ax.fill_between(x, lo, hi, color=c, alpha=0.15)
            ax.axhline(1.0, color="grey", lw=0.8, ls=":")
            ax.axvline(b1[setting], color="grey", lw=0.8, ls="--")
            ax.set_xscale("log")
            ax.set_title(setting, fontsize=9)
            ax.set_xlabel("budget (s)")
        for row in axes:
            row[0].set_ylabel(f"makespan / {args.ref} at {args.ref_budget}")
        handles = [plt.Line2D([], [], color=c, marker="o", ms=3, label=f"{lab} x{w}") for (lab, w), c in colors.items()]
        axes.flat[len(settings) - 1].legend(handles=handles, fontsize=7)
        fig.tight_layout()
        fig.savefig(args.png, dpi=120)
        print(f"wrote {args.png}")


def ttt(args) -> None:
    """Time to target: per setting and label, the seconds until the best-so-far makespan (run trace) first reaches
    ``args.ref``'s makespan at B1 on the same instance (median over instances, unreached counted as never)."""
    for lab in args.labels:
        for path in sorted((RUNS_DIR / "alns" / args.root / lab).glob("*-dev.jsonl")):
            rows = _rows(path)
            if not rows:
                continue
            setting = next(iter(rows.values()))["setting"]
            ref = _reference(run_alns.budget(configs.get(setting), "B1"), args.ref)
            times = []
            for i, r in rows.items():
                if (setting, i) in ref:
                    hit = [t for t, ms in r["trace"] if ms <= ref[setting, i] + 1e-9]
                    times.append(hit[0] if hit else np.inf)
            if times:
                t = np.array(times)
                limit = next(iter(rows.values()))["time_limit"]
                print(f"{setting:18s} {lab:14s} x{next(iter(rows.values()))['workers']} budget {limit:7.2f}s  "
                      f"reached {np.isfinite(t).sum():2d}/{len(t)}  median {np.median(t):7.2f}s  "
                      f"q25 {np.quantile(t, 0.25):7.2f}s  q75 {np.quantile(t, 0.75):7.2f}s")


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
    r.add_argument("--cpus", default=None, help="CPUs to pin the jobs to, e.g. 0-39 (default: least busy cores)")
    r.add_argument("--allow-dirty", action="store_true", help="run on uncommitted source (smoke tests only)")
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
    b = sub.add_parser("budgets")
    b.add_argument("--roots", nargs="+", required=True)
    b.add_argument("--labels", nargs="+", required=True)
    b.add_argument("--ref", default="CPSAT-8")
    b.add_argument("--ref-budget", default="B1", help="budget of the reference rows: B1 or seconds")
    b.add_argument("--csv")
    b.add_argument("--png")
    t = sub.add_parser("ttt")
    t.add_argument("--root", required=True)
    t.add_argument("labels", nargs="+")
    t.add_argument("--ref", default="CPSAT-8")
    args = ap.parse_args()
    if args.cmd == "ttt":
        ttt(args)
    elif args.cmd == "budgets":
        budgets(args)
    elif args.cmd == "anytime":
        anytime(args)
    elif args.cmd == "run":
        if args.split == "test":
            raise SystemExit("ablations run on dev (or validation) splits only")
        run(args)
    else:
        report(args)


if __name__ == "__main__":
    main()
