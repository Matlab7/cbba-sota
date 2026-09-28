"""Reproduction gate: our RL(g.)/RL(s.10) on the test split vs the paper's Tables I-IV.

Usage: rl_report.py [--split test] [--csv runs/rl/gate_test.csv]
Means and SDs are over successful instances (as in the paper). ``d/SD`` = (ours - paper) / paper SD.
Flags: ``band`` = |d| > 2 * SD_paper / sqrt(50) (rough 2-SE band of the paper mean);
``z`` = d / sqrt(SD_paper^2 / 50 + SD_ours^2 / n_ours) (two independent samples of instances).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from cbba_sota.bench import SETTINGS, configs


def load_rows(split: str, root: Path) -> pd.DataFrame:
    rows = []
    for s in SETTINGS:
        path = root / s.name / f"{split}.jsonl"
        if path.exists():
            rows += [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return pd.DataFrame(rows).drop(columns=["routes", "sample_makespans", "sample_success", "sample_s"])


def gate(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for s in SETTINGS:
        for method in ("RL(g.)", "RL(s.10)"):
            d = df[(df.setting == s.name) & (df.method == method)]
            if d.empty:
                continue
            ok = d[d.success]
            mean, sd = ok.makespan.mean(), ok.makespan.std(ddof=1)
            row = {"setting": s.name, "method": method, "n": len(d), "success": d.success.mean(),
                   "completion": d.completion.mean() if "completion" in d and d.completion.notna().all() else np.nan,
                   "mean": mean, "sd": sd, "wall_s": d.wall_s.mean(), "cpu_s": d.cpu_s.mean(),
                   "replay_eq": (np.isclose(d.replay_makespan, d.makespan, atol=1e-6)).mean()}
            paper = s.paper.get(method)
            if paper is not None:
                diff = mean - paper.makespan
                se2 = math.hypot(paper.makespan_sd / math.sqrt(50), sd / math.sqrt(len(ok)))
                band, z = abs(diff) > 2 * paper.makespan_sd / math.sqrt(50), diff / se2
                row |= {"paper_success": paper.success, "paper_mean": paper.makespan, "paper_sd": paper.makespan_sd,
                        "paper_time_s": paper.time_s, "diff": diff, "d_sd": diff / paper.makespan_sd, "z": z,
                        "flag": " ".join(f for f, on in (("band", band), ("z", abs(z) > 2)) if on) or "ok"}
            out.append(row)
    return pd.DataFrame(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="test")
    ap.add_argument("--root", type=Path, default=configs.RUNS_DIR / "rl")
    ap.add_argument("--csv", type=Path, default=None)
    args = ap.parse_args()
    g = gate(load_rows(args.split, args.root))
    if args.csv:
        g.to_csv(args.csv, index=False)
    cols = ["setting", "method", "paper_success", "paper_mean", "paper_sd", "n", "success", "completion", "mean", "sd",
            "diff", "d_sd", "z", "flag", "wall_s", "cpu_s", "paper_time_s", "replay_eq"]
    print("| " + " | ".join(cols) + " |")
    print("|" + "---|" * len(cols))
    for r in g.reindex(columns=cols).itertuples(index=False):
        print("| " + " | ".join(_fmt(v) for v in r) + " |")


def _fmt(v) -> str:
    if isinstance(v, str):
        return v
    if v is None or pd.isna(v):
        return "-"
    return f"{v:.3f}" if isinstance(v, float) else str(v)


if __name__ == "__main__":
    main()
