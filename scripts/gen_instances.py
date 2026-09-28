"""Write benchmark instances to data/hetero/<setting>/<split>/env_<i>.pkl (pickled from env.task_env).

Usage: gen_instances.py [--settings NAME ...] [--procs 40] [--force]
The shipped RALTestSet is re-pickled (module mapping fixed) into the same layout as a test split.
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--settings", nargs="*", default=[s.name for s in configs.SETTINGS])
    ap.add_argument("--procs", type=int, default=40)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    jobs = [(name, split, i, args.force) for name in args.settings for split in configs.get(name).splits()
            for i in range(configs.get(name).n_instances(split))]
    jobs.sort(key=lambda j: -configs.get(j[0]).n_tasks)  # large instances first
    with mp.get_context("spawn").Pool(min(args.procs, len(jobs))) as pool:
        status = pool.map(_job, jobs, chunksize=1)
    print(f"{status.count('new')} written, {status.count('skip')} existing, under {configs.DATA_DIR}")


if __name__ == "__main__":
    main()
