"""Re-score stored campaign rows against the current instances.

Usage: check_rows.py [PATH ...] [--procs 16] [--move] [--stamp] [--json FILE]

Scans every JSONL file under runs/ (or the given files and directories) except ``_superseded`` trees. A row that
names an instance (``setting``, ``split``, ``instance`` or ``index``; split ``json`` means the red-team export
pilots/heteromrta/redteam/<setting>.json, keyed by ``name``) is checked against the current instance:
- a stored ``fingerprint`` must equal the current one;
- env-loop rows (RL, greedy and RL-chosen hybrid rows) re-run their routes through the env replay, which must give
  the stored ``replay_makespan``; greedy rows without one (a task never reached) re-run the greedy itself, which must
  reproduce ``makespan``;
- plan rows (every other row with routes, 0- or 1-based) must cover every task, and their env score (the exact
  evaluator for minimal covers, where it equals the replay bit for bit, else the replay) must equal ``makespan``;
- rows without routes cannot be re-scored (unverifiable).
Every comparison is exact. A row is ``ok``, ``stale`` (fingerprint or score differs, not a cover, invalid routes,
instance missing) or ``unverifiable``. Rows whose stored ``success`` contradicts the one success definition (all
tasks finished and makespan < 200) are counted separately.

--move rewrites each file without its stale rows and appends them, with the reason, to runs/_superseded/<path>;
--stamp adds the current ``fingerprint`` (and ``fingerprint_by: check_rows``) to verified rows that lack one. A
rewritten file is backed up first to runs/_superseded/_backup-<time>/<path>; files that change during the check,
or were written less than ``--min-age`` seconds before the scan (a campaign may still append to them), are left
alone.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from functools import cache
from pathlib import Path

from cbba_sota.bench import runtime
from cbba_sota.bench.configs import MAX_TIME, ROOT, RUNS_DIR

SUPERSEDED = RUNS_DIR / "_superseded"
REDTEAM = ROOT / "pilots" / "heteromrta" / "redteam"
GREEDY = ("greedy", "greedy_repo")
_ONE = {k: "1" for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS")}


def instance_key(row: dict) -> tuple | None:
    """Hashable reference to the row's instance, None if the row names none."""
    if not {"setting", "split"} <= row.keys():
        return None
    if row["split"] == "json":
        return "json", row["setting"], row.get("name")
    i = row.get("instance", row.get("index"))
    return ("bench", row["setting"], row["split"], i) if isinstance(i, int) else None


@cache
def _json_instances(stem: str) -> dict:
    sys.path.insert(0, str(ROOT / "scripts"))
    from run_alns import instances_from_json

    return instances_from_json(REDTEAM / f"{stem}.json")


def load(key: tuple):
    """(instance, env source for the env loops) or None if it no longer exists."""
    from cbba_sota.bench import configs
    from cbba_sota.hetero import Instance

    if key[0] == "json":
        inst = _json_instances(key[1]).get(key[2])
        return None if inst is None else (inst, inst)
    _, setting, split, i = key
    try:
        path = configs.get(setting).instance_path(split, i)
    except (KeyError, IndexError):
        return None
    return (Instance.from_pickle(path), path) if path.exists() else None


def _score_routes(inst, routes) -> tuple[str, float | None]:
    """Env score of stored plan routes (0- or 1-based), or the reason they are not a plan of ``inst``."""
    from cbba_sota.hetero import Plan, evaluate, replay

    ids = {j for r in routes for j in r}
    one_based = bool(ids) and 0 not in ids and max(ids) == inst.n_tasks
    try:
        plan = Plan.from_routes(routes, inst.n_tasks, one_based=one_based)
    except ValueError as e:
        return f"invalid routes ({e})", None
    if not plan.covers(inst).all():
        return "not a cover", None
    return "", evaluate(inst, plan).makespan if plan.is_minimal_cover(inst) else replay(inst, plan)["makespan"]


def check(row: dict, inst, source, fp: str, rescore: bool = True) -> tuple[str, str]:
    """(status, reason) of one row against the current instance."""
    from cbba_sota.hetero import replay_routes
    from cbba_sota.solvers import greedy

    if "fingerprint" in row:
        if row["fingerprint"] != fp:
            return "stale", "fingerprint differs"
        if not rescore:
            return "ok", ""
    routes = row.get("routes")
    if not routes:
        return "unverifiable", "no routes"
    if row.get("method") in GREEDY and row.get("replay_makespan") is None:  # env-loop greedy, re-run
        res = greedy.greedy_repo(source) if row["method"] == "greedy_repo" else greedy.greedy_nearest(source)
        got, want = res.makespan, row["makespan"]
    elif "replay_makespan" in row:
        try:
            got = replay_routes(source, [[j + 1 for j in r] for r in routes])["makespan"]
        except ValueError as e:
            return "unverifiable", f"routes leave tasks out ({e})"
        want = row["replay_makespan"]
    else:
        reason, got = _score_routes(inst, routes)
        if reason:
            return "stale", reason
        want = row["makespan"]
    return ("ok", "") if got == want else ("stale", f"score {got!r} != stored {want!r}")


def success_contradicts(row: dict) -> bool:
    ms = row.get("makespan")
    return bool(row.get("success")) and ms is not None and ms >= MAX_TIME


def check_group(key: tuple, rows: list[tuple[str, int, dict]], rescore: bool = True) -> list[tuple]:
    """[(file, line, status, reason, fingerprint)] for rows sharing one instance."""
    loaded = load(key)
    if loaded is None:
        return [(f, n, "stale", "instance missing", None) for f, n, _ in rows]
    inst, source = loaded
    fp = runtime.fingerprint(inst)
    out = []
    for f, n, row in rows:
        try:
            status, reason = check(row, inst, source, fp, rescore)
        except Exception as e:  # noqa: BLE001  (reported per row)
            status, reason = "stale", f"error {type(e).__name__}: {e}"
        out.append((f, n, status, reason, fp))
    return out


def scan(paths: list[Path]) -> tuple[dict[Path, list[str]], dict[tuple, list]]:
    """Lines of every file and the rows grouped by instance."""
    files: dict[Path, list[str]] = {}
    groups: dict[tuple, list] = defaultdict(list)
    for path in paths:
        files[path] = path.read_text().splitlines()
        for n, line in enumerate(files[path]):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = instance_key(row)
            if key is not None:
                groups[key].append((str(path), n, row))
    return files, groups


def jsonl_files(targets: list[Path]) -> list[Path]:
    out = []
    for t in targets:
        found = [t] if t.is_file() else sorted(t.rglob("*.jsonl"))
        out += [p.resolve() for p in found if not any(part.startswith("_superseded") for part in p.parts)]
    return sorted(set(out))


def _rewrite(path: Path, lines: list[str], results: dict[int, tuple], stat: os.stat_result, backup: Path,
             move: bool, stamp: bool) -> tuple[int, int] | None:
    """Drop stale rows (to runs/_superseded) and stamp verified ones; None if the file changed meanwhile."""
    keep, gone, stamped = [], [], 0
    for n, line in enumerate(lines):
        status, reason, fp = results.get(n, ("", "", None))
        if move and status == "stale":
            gone.append(json.dumps(json.loads(line) | {"superseded_by": "check_rows", "stale_reason": reason}))
            continue
        if stamp and status == "ok":
            row = json.loads(line)
            if "fingerprint" not in row:
                line, stamped = json.dumps(row | {"fingerprint": fp, "fingerprint_by": "check_rows"}), stamped + 1
        keep.append(line)
    if not gone and not stamped:
        return 0, 0
    now = path.stat()
    if (now.st_size, now.st_mtime_ns) != (stat.st_size, stat.st_mtime_ns):
        return None
    rel = path.relative_to(RUNS_DIR)
    (backup / rel).parent.mkdir(parents=True, exist_ok=True)
    (backup / rel).write_text("\n".join(lines) + "\n")
    if gone:
        (SUPERSEDED / rel).parent.mkdir(parents=True, exist_ok=True)
        with (SUPERSEDED / rel).open("a") as f:
            f.write("\n".join(gone) + "\n")
    tmp = path.with_suffix(".jsonl.tmp")
    tmp.write_text("\n".join(keep) + "\n" if keep else "")
    tmp.replace(path)
    return len(gone), stamped


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="*", type=Path, default=[RUNS_DIR])
    ap.add_argument("--procs", type=int, default=16)
    ap.add_argument("--move", action="store_true", help="move stale rows to runs/_superseded")
    ap.add_argument("--stamp", action="store_true", help="add the fingerprint to verified rows without one")
    ap.add_argument("--json", type=Path, default=None, help="write every row's status here")
    ap.add_argument("--min-age", type=float, default=600.0, help="only rewrite files idle for this many seconds")
    args = ap.parse_args()
    os.environ.update(_ONE)

    paths = jsonl_files(args.paths)
    stats = {p: p.stat() for p in paths}
    files, groups = scan(paths)
    t0 = time.time()
    print(f"{sum(map(len, files.values()))} lines in {len(files)} files, {len(groups)} instances", flush=True)
    order = sorted(groups, key=lambda k: -len(groups[k]))
    with ProcessPoolExecutor(args.procs, mp_context=mp.get_context("spawn")) as ex:
        done = list(ex.map(check_group, order, [groups[k] for k in order]))
    results: dict[Path, dict[int, tuple]] = defaultdict(dict)
    for group in done:
        for f, n, status, reason, fp in group:
            results[Path(f)][n] = (status, reason, fp)
    print(f"checked in {time.time() - t0:.0f}s\n")

    total, stale_rows, report = Counter(), [], []
    print(f"{'file':60s} {'rows':>6s} {'ok':>6s} {'stale':>6s} {'unver.':>6s} {'no inst':>7s} {'succ!':>5s}")
    for path, lines in files.items():
        c = Counter()
        for n, line in enumerate(lines):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                c["torn"] += 1
                continue
            status, reason, _ = results[path].get(n, ("no instance", "", None))
            c["rows"] += 1
            c[status] += 1
            c["success!"] += success_contradicts(row)
            report.append({"file": str(path.relative_to(RUNS_DIR)), "line": n, "status": status, "reason": reason,
                           "method": row.get("method"), "instance": row.get("instance", row.get("index")),
                           "success_contradicts": success_contradicts(row)})
            if status == "stale":
                stale_rows.append((path.relative_to(RUNS_DIR), n, row.get("method"),
                                   row.get("instance", row.get("index")), reason))
        total.update(c)
        print(f"{path.relative_to(RUNS_DIR)!s:60s} {c['rows']:6d} {c['ok']:6d} {c['stale']:6d} "
              f"{c['unverifiable']:6d} {c['no instance']:7d} {c['success!']:5d}")
    print(f"\n{'total':60s} {total['rows']:6d} {total['ok']:6d} {total['stale']:6d} {total['unverifiable']:6d} "
          f"{total['no instance']:7d} {total['success!']:5d}  (torn lines {total['torn']})")
    reasons = Counter(r["reason"].split(" (")[0] if not r["reason"].startswith("score") else "score differs"
                      for r in report if r["status"] in ("stale", "unverifiable"))
    print("reasons:", dict(reasons))
    for rel, n, method, i, reason in stale_rows[:40]:
        print(f"  stale {rel}:{n + 1} {method} instance {i}: {reason}")
    if args.json:
        args.json.write_text(json.dumps(report))
    if args.move or args.stamp:
        backup = SUPERSEDED / f"_backup-{time.strftime('%Y%m%d-%H%M%S')}"
        for path, lines in files.items():
            if t0 - stats[path].st_mtime < args.min_age:  # measured at the scan, not after the (long) check
                print(f"  {path.relative_to(RUNS_DIR)} was written {args.min_age:g}s or less before the scan, left alone")
                continue
            res = _rewrite(path, lines, results[path], stats[path], backup, args.move, args.stamp)
            if res is None:
                print(f"  {path.relative_to(RUNS_DIR)} changed during the check, left alone")
            elif any(res):
                print(f"  {path.relative_to(RUNS_DIR)}: moved {res[0]} stale rows, stamped {res[1]}")


if __name__ == "__main__":
    main()
