"""Compare ALNS run tags instance by instance: mean makespan per setting and paired mean ratio vs the first tag.

Usage: compare_alns.py TAG_REF TAG [TAG ...]   (reads runs/alns/<tag>/*.jsonl)
"""
from __future__ import annotations

import json
import sys

import numpy as np

from cbba_sota.bench.configs import RUNS_DIR


def load(tag: str) -> dict[str, dict[str, dict]]:
    out: dict[str, dict[str, dict]] = {}
    for path in sorted((RUNS_DIR / "alns" / tag).glob("*.jsonl")):
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        out[path.stem] = {r["name"]: r for r in rows}
    return out


def main() -> None:
    tags = sys.argv[1:]
    data = {t: load(t) for t in tags}
    ref = data[tags[0]]
    for group in sorted(ref):
        print(group)
        for t in tags:
            rows = data[t].get(group, {})
            common = sorted(set(rows) & set(ref[group]))
            if not common:
                continue
            ms = np.array([rows[n]["makespan"] for n in common])
            base = np.array([ref[group][n]["makespan"] for n in common])
            its = np.mean([rows[n]["it_per_s"] for n in common])
            print(f"  {t:28s} n={len(common):3d} mean {ms.mean():8.3f}  ratio vs ref {np.mean(ms / base):.4f}  "
                  f"wins {(ms < base - 1e-9).sum():2d} losses {(ms > base + 1e-9).sum():2d}  it/s {its:7.0f}")


if __name__ == "__main__":
    main()
