"""Figures of the paper from the campaign rows (matplotlib only).

Usage: make_figures.py [--split test|val] [--out paper/figures]
budget.pdf: per H1 setting, the mean gap to the best known plan (%) of ALNS on 1 core at 0.5, 1, 2 s and B1 and on
8 cores at B1, against each competitor on 8 cores at B1 (horizontal lines), over every instance of the
split (grid test / c1); runs are scored at their budget as in the reports (scripts/anytime.py at_budget).
ratios.pdf (test): per setting, how much shorter ALNS's plans are than the strongest competitor's and the RL policy's,
for C2 (1 core, 2 s) and C1 (8 cores, B1), from docs/results/test/anytime_ratios_test.csv.
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
    """Robots and tasks in words; the species, on a second line, only for the two 500-task settings (all others
    have 5)."""
    _, a, sp, t = name.rsplit("-", 3)
    return f"{a} robots, {t} tasks" + (f"\n({sp} species)" if t == "500" else "")
CTAS_COLOUR = "#56B4E9"


def ratios_figure(plt, csv_path: Path, out: Path) -> None:
    """Per setting, how much shorter ALNS's plans are than the strongest competitor's (the one with the highest ratio
    ALNS / competitor, named) and than the RL policy's: ALNS on 1 core for 2 s (hollow) and on 8 cores at B1 (filled),
    every competitor on 8 cores at B1."""
    import csv

    rows = list(csv.DictReader(csv_path.open()))
    short = {"CPSAT": "CP-LNS", "PCPSAT": "Parallel CP-LNS", "CPFULL": "CP-SAT full", "CONSTRUCT": "Greedy restarts"}
    levels = (("C2 1 core 2 s vs 8 cores B1", "white"), ("C1 8 cores B1", None))  # hollow, then filled

    def pick(tag: str, setting: str, rl: bool) -> dict:
        rs = [r for r in rows if r["comparison"] == tag and r["setting"] == setting]
        if rl:
            return next(r for r in rs if r["other"].startswith("RL"))
        return max((r for r in rs if r["other"].split("-")[0] in short), key=lambda r: float(r["ratio"]))

    ys, heads, y = [], [], 0.0
    for k, name in enumerate(SETTINGS):
        fam = name.rsplit("-", 3)[0]
        if k == 0 or fam != SETTINGS[k - 1].rsplit("-", 3)[0]:
            heads.append((y, FAMILY_NAME[fam]))
            y -= 0.8
        ys.append(y)
        y -= 1.0
    fig, ax = plt.subplots(figsize=(3.4, 3.9))
    ax.axvline(0, color="k", lw=0.7, ls="--", zorder=1)
    for y, setting in zip(ys, SETTINGS):
        for rl, colour, dy in ((False, "#333333", 0.17), (True, "#CC79A7", -0.17)):
            pts = []
            for tag, face in levels:
                r = pick(tag, setting, rl)
                gain = 100 * (1 - float(r["ratio"]))
                sig = float(r["p_holm"]) < 0.05 and float(r["ratio"]) < 1
                pts.append((gain, face, sig, r))
            ax.plot([pts[0][0], pts[1][0]], [y + dy] * 2, color=colour, lw=1.3, zorder=2)
            for gain, face, sig, r in pts:
                ax.plot(gain, y + dy, "o", ms=4.2, color=colour, mfc=face or colour, mew=1.1, zorder=3)
                if not sig:
                    ax.text(gain - 0.6, y + dy + 0.33, "n.s.", fontsize=6, ha="center", va="center", color=colour,
                            zorder=4, bbox=dict(facecolor="white", edgecolor="none", pad=0.6))
            if not rl:
                ax.text(pts[1][0] + 1.0, y + dy, short[pts[1][3]["other"].split("-")[0]], fontsize=6.2, va="center",
                        color="#555555", fontstyle="italic")
    for hy, text in heads:  # a centred heading per family, on white so the zero line does not cross it
        ax.text(0.5, hy - 0.05, text, fontsize=6.5, fontweight="bold", color="#444444", ha="center", va="center",
                transform=ax.get_yaxis_transform(), zorder=4,
                bbox=dict(facecolor="white", edgecolor="none", pad=1.2))
        ax.axhline(hy + 0.42, color="#CCCCCC", lw=0.5)
    ax.set_xlim(-3, 36)
    ax.set_ylim(ys[-1] - 0.6, 0.45)
    ax.set_yticks(ys)
    ax.set_yticklabels([size(n) for n in SETTINGS], fontsize=6.5)
    ax.set_xlabel("ALNS's plans shorter by (%)", fontsize=7)
    ax.grid(axis="x", alpha=0.25, lw=0.4)
    ax.tick_params(axis="y", length=0)
    handles = [plt.Line2D([], [], marker="o", ls="", color="k", mfc="white", ms=4, label="ALNS on 1 core, 2 s"),
               plt.Line2D([], [], marker="o", ls="", color="k", ms=4, label="ALNS on 8 cores, $B_1$"),
               plt.Line2D([], [], color="#333333", lw=1.3, label="vs. strongest competitor"),
               plt.Line2D([], [], color="#CC79A7", lw=1.3, label="vs. RL policy")]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=6.5, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.1, 1, 1))
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
    # the strongest competitor of each setting, as in the ratio figure: the highest ratio ALNS / competitor at B1
    strongest = {}
    ratios_csv = ROOT / "docs" / "results" / "test" / "anytime_ratios_test.csv"
    if args.split == "test" and ratios_csv.exists():
        import csv

        c1 = [r for r in csv.DictReader(ratios_csv.open()) if r["comparison"] == "C1 8 cores B1"]
        for name in SETTINGS:
            cand = [r for r in c1 if r["setting"] == name and r["other"].split("-")[0] in ("CPSAT", "PCPSAT", "CPFULL",
                                                                                            "CONSTRUCT")]
            if cand:
                strongest[name] = max(cand, key=lambda r: float(r["ratio"]))["other"].split("-")[0]
    ymax = 18.0
    fig, axes = plt.subplots(2, 4, figsize=(7.0, 3.7))
    for ax, name in zip(axes.flat, SETTINGS):
        rows = at.load_rows(name)
        b = at.budgets(configs.get(name))
        bks_of = bks_all.get(name, {})
        pts = {}
        for label in ("0.5", "1", "2", "B1"):
            g = mean_gap(rows.get(("ALNS2", 1, label), {}), bks_of)
            if g is not None:
                pts[label] = (b[label], g)  # x: compute used, CPU-seconds (1 core)
        full = 8 * b["B1"]
        for method, name_, colour, ls in LEVELS:
            g = mean_gap(rows.get((method, 8, "B1"), {}), bks_of)
            if g is None:
                continue
            if method == "RL":  # far above: an arrow at the top with its value
                ax.annotate(f"RL policy {g:.0f}%", xy=(full, ymax), xytext=(full / 1.4, ymax - 0.9), ha="right",
                            va="center", fontsize=6, color=colour,
                            arrowprops=dict(arrowstyle="->", color=colour, lw=0.9))
            elif method == strongest.get(name):
                ax.axhline(g, color="#333333", lw=1.3, zorder=3)
                ax.text(full / 1.6, g + 0.35, name_, fontsize=6, ha="right", va="bottom", color="#333333",
                        fontstyle="italic")
            else:
                ax.axhline(g, color="#BBBBBB", lw=0.8, zorder=2)
        xs, ys = zip(*pts.values())
        ax.plot(xs, ys, "-", color=ALNS, lw=1.5, zorder=5)
        ax.plot(xs, ys, "o", color=ALNS, ms=2.6, zorder=5)
        if "2" in pts:
            ax.plot(*pts["2"], "o", ms=5.2, mfc="white", mec=ALNS, mew=1.3, zorder=6)
        g = mean_gap(rows.get(("ALNS2", 8, "B1"), {}), bks_of)
        if g is not None:
            ax.plot([full], [g], "o", ms=5.2, color=ALNS, zorder=6)
        ax.axvline(full, color="k", lw=0.5, ls=":", zorder=0)
        ax.set_xscale("log")
        ax.set_xlim(0.3, full * 2.2)
        ax.set_ylim(0, ymax)
        ax.xaxis.set_minor_formatter(NullFormatter())
        _, a_, sp, t = name.rsplit("-", 3)
        extra = f", {sp} species" if t == "500" else ""
        ax.set_title(f"{FAMILY_NAME[name.rsplit('-', 3)[0]].replace(' robots', '')}\n{a_} robots, {t} tasks{extra}",
                     fontsize=6.6)
        ax.grid(alpha=0.25, lw=0.4)
    for ax in axes[1]:
        ax.set_xlabel("compute used (CPU-seconds)")
    for ax in axes[:, 0]:
        ax.set_ylabel("gap to best plan (%)")
    handles = [plt.Line2D([], [], color=ALNS, lw=1.5, marker="o", ms=2.6, label="ALNS, 1 core, 0.5 s to $B_1$"),
               plt.Line2D([], [], marker="o", ls="", ms=5.2, mfc="white", mec=ALNS, mew=1.3, label="ALNS, 1 core, 2 s"),
               plt.Line2D([], [], marker="o", ls="", ms=5.2, color=ALNS, label="ALNS, 8 cores, $B_1$"),
               plt.Line2D([], [], color="#333333", lw=1.3, label="strongest competitor, 8 cores, $B_1$ (named)"),
               plt.Line2D([], [], color="#BBBBBB", lw=0.8, label="other competitors, 8 cores, $B_1$"),
               plt.Line2D([], [], color="#CC79A7", lw=0.9, marker=r"$\uparrow$", ms=6, label="RL policy (off scale)")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=6.5, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.1, 1, 1), h_pad=0.9, w_pad=0.6)
    args.out.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out / "budget.pdf")
    print(f"wrote {args.out / 'budget.pdf'} ({args.split})")
    ratios_csv = ROOT / "docs" / "results" / "test" / "anytime_ratios_test.csv"
    if args.split == "test" and ratios_csv.exists():
        ratios_figure(plt, ratios_csv, args.out)


if __name__ == "__main__":
    main()
