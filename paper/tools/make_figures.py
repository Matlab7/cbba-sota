"""Figures of the paper from the campaign rows (matplotlib only).

Usage: make_figures.py [--split test|val] [--out paper/figures]
budget.pdf: per H1 setting, the mean gap to the best known plan (%) of ALNS on 1 core at 0.5, 1, 2 s and B1 and on
8 cores at B1, against each competitor on 8 cores at B1 (horizontal lines), over every instance of the
split (grid test / c1); runs are scored at their budget as in the reports (scripts/anytime.py at_budget).
ratios.pdf (test): the paired makespan ratios ALNS / competitor with bootstrap 95% CIs of C2 (1 core, 2 s) and C1
(8 cores, B1), from docs/results/test/anytime_ratios_test.csv; hollow markers are not significant after Holm.
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
LEVELS = (("CPSAT", "CP-LNS", "#D55E00", "-"), ("PCPSAT", "Parallel CP-LNS", "#E69F00", "--"),
          ("CPFULL", "CP-SAT full model", "#009E73", "-."), ("CONSTRUCT", "Greedy restarts", "#999999", ":"),
          ("RL", "RL policy", "#CC79A7", (0, (5, 1, 1, 1, 1, 1))))
ALNS = "#0072B2"
FAMILY_NAME = {"SA-BT": "Single-skill robots, binary needs", "SA-AT": "Single-skill robots, additive needs",
               "MA-AT": "Multi-skill robots, additive needs"}


def size(name: str) -> str:
    """Robots and tasks in words; the species only for the two 500-task settings (all others have 5)."""
    _, a, sp, t = name.rsplit("-", 3)
    return f"{a} robots ({sp} species), {t} tasks" if t == "500" else f"{a} robots, {t} tasks"
CTAS_COLOUR = "#56B4E9"
RATIO_PANELS = (("C2 1 core 2 s vs 8 cores B1", "(a) C2: ALNS on 1 core for 2 s"),
                ("C1 8 cores B1", "(b) C1: ALNS on 8 cores at $B_1$"))


def ratios_figure(plt, csv_path: Path, out: Path) -> None:
    """Dot plot of the paired ratios (one row per setting, one marker per competitor, CI as whiskers)."""
    import csv

    rows = list(csv.DictReader(csv_path.open()))
    xmin, xmax = 0.6, 1.06
    ys, heads, y = [], [], 0.0  # top to bottom; each family gets a heading slot above its rows
    for k, name in enumerate(SETTINGS):
        fam = name.rsplit("-", 3)[0]
        if k == 0 or fam != SETTINGS[k - 1].rsplit("-", 3)[0]:
            heads.append((y, FAMILY_NAME[fam]))
            y -= 0.75
        ys.append(y)
        y -= 1.0
    ys = np.array(ys)
    offsets = np.linspace(-0.3, 0.3, len(LEVELS))
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), sharey=True)
    for ax, (tag, title) in zip(axes, RATIO_PANELS):
        by = {(r["setting"], r["other"].split("-")[0]): r for r in rows if r["comparison"] == tag}
        ax.axvspan(1.0, xmax, color="#EFEFEF", zorder=0, lw=0)
        ax.axvline(1.0, color="k", lw=0.7, zorder=1)
        for (method, name, colour, _), off in zip(LEVELS, offsets):
            for y, setting in zip(ys, SETTINGS):
                r = by.get((setting, method))
                if r is None:
                    continue
                ratio, lo, hi = float(r["ratio"]), float(r["lo"]), float(r["hi"])
                sig = float(r["p_holm"]) < 0.05 and ratio < 1
                ax.errorbar(ratio, y + off, xerr=[[ratio - lo], [hi - ratio]], fmt="o", ms=3.0, color=colour,
                            mfc=colour if sig else "white", mew=0.9, elinewidth=0.8, capsize=0, zorder=3)
        for y, setting in zip(ys, SETTINGS):  # CTAS-D: off the scale, value written next to the arrow
            r = by.get((setting, "CTAS"))
            if r is not None:
                ax.plot(xmin + 0.004, y, marker="<", color=CTAS_COLOUR, ms=4, zorder=3, clip_on=False)
                ax.text(xmin + 0.013, y, f"{float(r['ratio']):.2f}", fontsize=5.5, va="center", color="#2A7FB0")
        for hy, text in heads:
            ax.text(xmin + 0.004, hy - 0.05, text, fontsize=6, fontstyle="italic", color="#444444", va="center")
            ax.axhline(hy + 0.45, color="#BBBBBB", lw=0.5)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ys[-1] - 0.6, 0.5)
        ax.set_title(title)
        ax.set_xlabel("makespan ratio ALNS / competitor")
        ax.text(0.985, 0.5, "competitor better", rotation=90, fontsize=5.5, color="#777777", va="center", ha="right",
                transform=ax.transAxes)
        ax.grid(axis="x", alpha=0.25, lw=0.4)
    axes[0].set_yticks(ys)
    axes[0].set_yticklabels([size(n) for n in SETTINGS])
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, ms=3.5, label=n) for _, n, c, _ in LEVELS]
    handles.append(plt.Line2D([], [], marker="<", ls="", color=CTAS_COLOUR, ms=4, label="CTAS-D (off scale)"))
    handles.append(plt.Line2D([], [], marker="o", ls="", color="k", mfc="white", ms=3.5,
                              label="not significant (Holm)"))
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.13, 1, 1), w_pad=1.0)
    fig.savefig(out / "ratios.pdf")
    print(f"wrote {out / 'ratios.pdf'}")


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
        # x: compute used, CPU-seconds = cores x wall-clock budget (1 core for ALNS's line)
        ax.plot(xs, ys, "-o", color=ALNS, ms=3, lw=1.2, label="ALNS, 1 core", zorder=5)
        full = 8 * b["B1"]
        g = mean_gap(rows.get(("ALNS2", 8, "B1"), {}), bks_of)
        if g is not None:
            ax.plot([full], [g], "s", color=ALNS, ms=3.5, mfc="white", mew=1.0, zorder=6,
                    label="ALNS, 8 cores at $B_1$")
        for method, name_, colour, ls in LEVELS:
            g = mean_gap(rows.get((method, 8, "B1"), {}), bks_of)
            if g is not None:  # a competitor's level, drawn across so it can be read against ALNS's line
                ax.axhline(g, color=colour, ls=ls, lw=1.0, label=f"{name_}, 8 cores at $B_1$")
                ax.plot([full], [g], "D", color=colour, ms=2.6, zorder=4)
        ax.axvline(full, color="k", lw=0.5, ls=":", zorder=0)
        ax.set_xlim(0.3, full * 2.2)
        ax.set_xscale("log")
        ax.set_yscale("symlog", linthresh=1, linscale=0.5)
        ax.set_ylim(0, 90)
        ax.yaxis.set_major_locator(FixedLocator([0, 1, 2, 5, 10, 20, 40, 80]))
        ax.yaxis.set_minor_locator(NullLocator())
        ax.set_yticklabels(["0", "1", "2", "5", "10", "20", "40", "80"])
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_title(f"{FAMILY_NAME[name.rsplit('-', 3)[0]].replace(' robots', '')}\n{size(name)}", fontsize=6.3)
        ax.grid(alpha=0.25, lw=0.4)
    for ax in axes[1]:
        ax.set_xlabel("compute used (CPU-seconds)")
    for ax in axes[:, 0]:
        ax.set_ylabel("gap to best (%)")
    handles, names = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, names, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.1, 1, 1), h_pad=0.8, w_pad=0.6)
    args.out.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out / "budget.pdf")
    print(f"wrote {args.out / 'budget.pdf'} ({args.split})")
    ratios_csv = ROOT / "docs" / "results" / "test" / "anytime_ratios_test.csv"
    if args.split == "test" and ratios_csv.exists():
        ratios_figure(plt, ratios_csv, args.out)


if __name__ == "__main__":
    main()
