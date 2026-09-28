"""Write benchmark instances to data/hetero/<setting>/<split>/env_<i>.pkl (pickled from env.task_env).

Usage: gen_instances.py [--settings NAME ...] [--splits test dev val] [--procs 40] [--force] [--check]
The shipped RALTestSet is re-pickled (module mapping fixed) into the same layout as a test split. Every split uses
the same generator with its own seed base (``configs.SPLITS``). ``--check`` then verifies that no two instance
files of the selected settings (over all their splits) have the same data fingerprint (``runtime.fingerprint``)
and that every generated file equals a fresh generation from its seed.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp

from cbba_sota.bench import configs
from cbba_sota.bench.heteromrta import generate_env, load_env, save_env, slim


def _job(args: tuple[str, str, int, bool]) -> str:
    name, split, i, force = args
    setting = configs.get(name)
    path = setting.instance_path(split, i)
    if path.exists() and not force:
        return "skip"
    if setting.source is not None:
        env = slim(load_env(setting.source / f"env_{i}.pkl"))
    else:
        env = generate_env(setting, setting.seed(split, i))
    save_env(env, path)
    return "new"


def _fingerprint(args: tuple[str, str, int]) -> tuple[str, str, int, str, bool]:
    """(setting, split, index, fingerprint of the file, whether it equals a fresh generation from its seed)."""
    from cbba_sota.bench import runtime
    from cbba_sota.hetero import Instance

    name, split, i = args
    setting = configs.get(name)
    fp = runtime.fingerprint(Instance.from_pickle(setting.instance_path(split, i)))
    if setting.source is not None:
        return name, split, i, fp, True
    return name, split, i, fp, fp == runtime.fingerprint(Instance.from_env(generate_env(setting, setting.seed(split, i))))


def check(pool, settings: list[str]) -> None:
    jobs = [(name, split, i) for name in settings for split in configs.get(name).splits()
            for i in range(configs.get(name).n_instances(split))]
    rows = pool.map(_fingerprint, jobs, chunksize=4)
    first: dict[str, tuple] = {}
    clashes = []
    for name, split, i, fp, _ in rows:
        if first.setdefault(fp, (name, split, i)) != (name, split, i):
            clashes.append((first[fp], (name, split, i)))
    mismatched = [(n, s, i) for n, s, i, _, ok in rows if not ok]
    print(f"check: {len(rows)} instance files ({', '.join(sorted({r[1] for r in rows}))}), {len(first)} distinct "
          f"fingerprints, {len(clashes)} clashes, {len(mismatched)} files differ from their seed's generation")
    if clashes or mismatched:
        raise SystemExit(f"clashes {clashes[:5]}, mismatched {mismatched[:5]}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--settings", nargs="*", default=[s.name for s in configs.SETTINGS])
    ap.add_argument("--splits", nargs="*", default=list(configs.SPLITS), choices=list(configs.SPLITS))
    ap.add_argument("--procs", type=int, default=40)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--check", action="store_true", help="verify fingerprint disjointness over all splits")
    args = ap.parse_args()
    jobs = [(name, split, i, args.force) for name in args.settings for split in configs.get(name).splits()
            if split in args.splits for i in range(configs.get(name).n_instances(split))]
    jobs.sort(key=lambda j: -configs.get(j[0]).n_tasks)  # large instances first
    with mp.get_context("spawn").Pool(max(1, min(args.procs, len(jobs)))) as pool:
        status = pool.map(_job, jobs, chunksize=1) if jobs else []
        print(f"{status.count('new')} written, {status.count('skip')} existing, under {configs.DATA_DIR}")
        if args.check:
            check(pool, args.settings)


if __name__ == "__main__":
    main()
