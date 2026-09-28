"""Figures of the paper from the campaign rows (matplotlib only).

Usage: make_figures.py [--split test|val] [--out paper/figures]
budget.pdf: per H1 setting, the mean gap to the best known plan (%) of ALNS on 1 core at 0.5, 1, 2 s and B1 and on
8 cores at B1, against each competitor on 8 cores at B1 (horizontal lines), over every instance of the
split (grid test / c1); runs are scored at their budget as in the reports (scripts/anytime.py at_budget).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import anytime as at
import bks

from cbba_sota.bench import configs

SETTINGS = ("SA-BT-25-5-50", "SA-BT-50-5-50", "SA-AT-50-5-50", "MA-AT-25-5-50", "MA-AT-50-5-50", "MA-AT-50-5-200",
            "MA-AT-150-10-500", "MA-AT-150-5-500")
# competitor on 8 cores at B1: (method, label, colour, line style); Okabe-Ito colours
LEVELS = (("CPSAT", "CP-SAT LNS", "#D55E00", "-"), ("PCPSAT", "Par. CP-SAT LNS", "#E69F00", "--"),
          ("CPFULL", "CP-SAT model", "#009E73", "-."), ("CONSTRUCT", "Restarts", "#999999", ":"),
          ("RL", "RL(s.N)", "#CC79A7", (0, (5, 1, 1, 1, 1, 1))))
ALNS = "#0072B2"


def mean_gap(by_i: dict[int, dict], bks_of: dict[int, float]) -> float | None:
    vals = [(i, at.at_budget(r)) for i, r in by_i.items() if i in bks_of]
    vals = [(i, v) for i, v in vals if v is not None]
    return float(np.mean([100 * (v - bks_of[i]) / bks_of[i] for i, v in vals])) if vals else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=["test", "val"], default="test")
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "figures")
    args = ap.parse_args()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, NullFormatter, NullLocator

    plt.rcParams.update({"font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7, "legend.fontsize": 6.5,
                         "xtick.labelsize": 6, "ytick.labelsize": 6, "pdf.fonttype": 42, "ps.fonttype": 42})
    at.SPLIT = args.split
    at.OUT = at.RUNS_DIR / ("anytime_test" if args.split == "test" else "anytime_c1")
    at.GRID = at.grid_test if args.split == "test" else at.grid_c1
    at.LATE_RULE = "score"
    bks_all = bks.load_bks(args.split)
    fig, axes = plt.subplots(2, 4, figsize=(7.0, 3.1))
    for ax, name in zip(axes.flat, SETTINGS):
        rows = at.load_rows(name)
        b = at.budgets(configs.get(name))
        bks_of = bks_all.get(name, {})
        xs, ys = [], []
        for label in ("0.5", "1", "2", "B1"):
            g = mean_gap(rows.get(("ALNS2", 1, label), {}), bks_of)
            if g is not None:
                xs.append(b[label])
                ys.append(g)
        ax.plot(xs, ys, "-o", color=ALNS, ms=3, lw=1.2, label="ALNS, 1 core", zorder=5)
        g = mean_gap(rows.get(("ALNS2", 8, "B1"), {}), bks_of)
        if g is not None:
            ax.plot([b["B1"]], [g], "s", color=ALNS, ms=3.5, mfc="white", mew=1.0, zorder=6,
                    label="ALNS, 8 cores at $B_1$")
        for method, name_, colour, ls in LEVELS:
            g = mean_gap(rows.get((method, 8, "B1"), {}), bks_of)
            if g is not None:
                ax.axhline(g, color=colour, ls=ls, lw=1.0, label=f"{name_}, 8 cores at $B_1$")
        ax.axvline(b["B1"], color="k", lw=0.5, ls=":", zorder=0)
        ax.set_xscale("log")
        ax.set_yscale("symlog", linthresh=1, linscale=0.5)
        ax.set_ylim(0, 60)
        ax.yaxis.set_major_locator(FixedLocator([0, 1, 2, 5, 10, 20, 40]))
        ax.yaxis.set_minor_locator(NullLocator())
        ax.set_yticklabels(["0", "1", "2", "5", "10", "20", "40"])
        ax.xaxis.set_minor_formatter(NullFormatter())
        family, a, s, t = name.rsplit("-", 3)
        ax.set_title(f"{family} {a}/{s}/{t} ($B_1$ = {b['B1']:.0f} s)")
        ax.grid(alpha=0.25, lw=0.4)
    for ax in axes[1]:
        ax.set_xlabel("wall-clock budget (s)")
    for ax in axes[:, 0]:
        ax.set_ylabel("gap to best (%)")
    handles, names = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, names, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.1, 1, 1), h_pad=0.8, w_pad=0.6)
    args.out.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out / "budget.pdf")
    print(f"wrote {args.out / 'budget.pdf'} ({args.split})")


if __name__ == "__main__":
    main()
