"""Tables of the supplementary material from the result files, so that no number in it is typed by hand.

Usage: make_supplement.py RESULTS_DIR [--out paper/tables]
RESULTS_DIR holds the campaign CSVs (docs/results/test, or docs/results/val-c1 for a dry run). Also reads the dev
ablation (docs/results/alns-v2), the val references (docs/results/val-c1/bks_val*.csv), the val runs of the 20-task
settings (docs/results/phase1b) and the competitors' per-setting choices (scripts/bks.py). Writes supp_*.tex.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import bks
from make_tables import SETTINGS, label, read

from cbba_sota.bench import configs

DOCS = ROOT / "docs" / "results"
NAMES = {"ALNS2": "ALNS", "ALNS1": "ALNS v1", "CPSAT": "CP-SAT LNS", "PCPSAT": "Par.\\ CP-SAT LNS",
         "CPFULL": "CP-SAT model", "CTAS": "CTAS-D", "CONSTRUCT": "Restarts", "RL": "RL(s.$N$)"}
ORDER = ("ALNS2", "CPSAT", "PCPSAT", "CPFULL", "CTAS", "CONSTRUCT", "RL", "ALNS1")
BUDGETS = ("0.5", "1", "2", "5", "10", "B1", "2B1")
SMALL = ("SA-BT-25-5-20", "SA-AT-25-5-20", "MA-AT-25-5-20")
ABLATIONS = (("v1", "v1 (Phase 1)"), ("no-init", "no portfolio"), ("no-resort", "no re-sort"),
             ("alt", "delay-aware covers"))

CAPTION_GAPS = (r"\caption{Mean gap to the best known plan (\%) on the SPLIT split per method, number of cores and "
                r"budget ($^*$: some runs late, scored at the budget by their trace or as failures, as pre-registered). "
                r"CPU: CPU seconds of all processes and threads / (budget $\times$ cores) at $B_1$.}")
CAPTION_ABLATION = (r"\caption{Ablation on dev (1 worker): mean paired makespan ratio variant / final ALNS [bootstrap "
                    r"95\% CI]; above 1 means the variant is worse. v1 is the configuration of an earlier phase (own "
                    r"random-order construction, no re-sort, older kernels); ``no portfolio'' constructs by own "
                    r"insertion; ``no re-sort'' never re-sorts the order by start times; ``delay-aware covers'' adds a "
                    r"second cover per slot (not kept: it halves the iteration rate). Blank: not run.}")
CAPTION_REFS = (r"\caption{Val split: how far the best plan of any run without an ALNS component (competitor runs of "
                r"the timed campaigns and the long CP-SAT references) is above the best known plan, mean over the "
                r"instances; the 300-s references on 8 cores (parallel CP-SAT LNS and the full CP-SAT model, both "
                r"started from constructions only), their mean gap to the best known plan; and the distance of "
                r"CP-SAT's certified lower bound below the best known plan.}")
CAPTION_SMALL = (r"\caption{Val split, three 20-task settings (descriptive; earlier campaign on another host): mean gap "
                 r"to the best known plan (\%) at $B_1$ on 8 cores, and the instances that CP-SAT proved optimal in "
                 r"60\,s on 8 cores.}")
CAPTION_PARAMS = (r"\caption{Per-setting parameters of the competitors, chosen on dev: sub-solve length of CP-SAT LNS "
                  r"and of its parallel version (s), and the number of nearest candidate tasks per arc of the full "
                  r"CP-SAT model (all: every arc).}")


def budget_label(b: str) -> str:
    return {"B1": "$B_1$", "2B1": "$2B_1$"}.get(b, b)


def gap_cell(g: dict | None) -> str:
    if g is None:
        return ""
    late = "$^*$" if int(g.get("late") or 0) else ""
    return f"{float(g['gap_pct']):.2f}{late}"


def gaps_table(gaps: list[dict], split: str) -> str:
    """Mean gap to the best known plan (%) per method, cores and budget; CPU share at B1."""
    present = [b for b in BUDGETS if any(g["budget"] == b for g in gaps)]
    header = ["Setting & Method & " + " & ".join(budget_label(b) for b in present) + r" & CPU \\", r"\midrule"]
    lines = [r"\begin{longtable}{ll" + "r" * len(present) + "r}",
             CAPTION_GAPS.replace("SPLIT", split) + r"\label{tab:supp-gaps}\\", r"\toprule", *header,
             r"\endfirsthead", r"\toprule", *header, r"\endhead"]
    for name in SETTINGS:
        rows = [g for g in gaps if g["setting"] == name]
        if not rows:
            continue
        first = True
        for method in ORDER:
            for cores in ("1", "8"):
                sel = {g["budget"]: g for g in rows if g["method"] == method and g["cores"] == cores}
                if not sel:
                    continue
                cells = [gap_cell(sel.get(b)) for b in present]
                cpu = sel.get("B1", {}).get("cpu_share", "")
                cpu = f"{float(cpu):.2f}" if cpu else ""
                lines.append(f"{label(name) if first else ''} & {NAMES[method]}-{cores} & " + " & ".join(cells)
                             + f" & {cpu}" + r" \\")
                first = False
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{longtable}")
    return "\n".join(lines) + "\n"


def ratio_cell(r: dict | None) -> str:
    if r is None or r["ratio_ref"] == "nan":
        return ""
    return f"{float(r['ratio_ref']):.3f} {{\\scriptsize [{float(r['ci_lo']):.3f}, {float(r['ci_hi']):.3f}]}}"


def ablation_table() -> str:
    """Dev ablation of the ALNS components (1 worker; ratio to the final configuration, >1: the variant is worse)."""
    blocks = [("$B_1$", DOCS / "alns-v2" / "ablation_w1_B1.csv"), ("2\\,s", DOCS / "alns-v2" / "ablation_w1_2s.csv")]
    lines = [r"\begin{table}[h]", r"\centering", r"\small", CAPTION_ABLATION, r"\label{tab:supp-ablation}",
             r"\begin{tabular}{ll" + "c" * len(ABLATIONS) + "}", r"\toprule",
             "Budget & Setting & " + " & ".join(n for _, n in ABLATIONS) + r" \\", r"\midrule"]
    for budget, path in blocks:
        rows = read(path)
        first = True
        for name in SETTINGS:
            by = {r["label"]: r for r in rows if r["setting"] == name}
            if not by:
                continue
            cells = [ratio_cell(by.get(key)) for key, _ in ABLATIONS]
            lines.append(f"{budget if first else ''} & {label(name)} & " + " & ".join(cells) + r" \\")
            first = False
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines += [r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def references_table() -> str:
    """Val: the best known plans, the independent CP-SAT references and the certified bounds."""
    per = read(DOCS / "val-c1" / "bks_val.csv")
    runs = read(DOCS / "val-c1" / "bks_val_runs.csv")
    lines = [r"\begin{table}[h]", r"\centering", r"\small", CAPTION_REFS, r"\label{tab:supp-refs}",
             r"\begin{tabular}{lrrrrr}", r"\toprule",
             r"Setting & $n$ & best w/o ALNS & Par.\ CP-SAT LNS 300\,s & CP-SAT model 300\,s & bound below \\",
             r"\midrule"]
    for name in SETTINGS:
        sel = [r for r in per if r["setting"] == name and r["alns_free_best"]]
        if not sel:
            continue
        free = sum((float(r["alns_free_best"]) - float(r["bks"])) / float(r["bks"]) for r in sel) / len(sel) * 100
        by = {r["run"]: r for r in runs if r["setting"] == name}
        plns, full = by.get("PLNS-8x300s"), by.get("CPFULLc-8x300s")
        cells = [f"+{free:.1f}\\%",
                 f"+{float(plns['gap_to_bks_pct']):.1f}\\%" if plns else "--",
                 f"+{float(full['gap_to_bks_pct']):.1f}\\%" if full else "--",
                 f"{float(full['bks_to_lb_pct']):.0f}\\%" if full and full["bks_to_lb_pct"] else "--"]
        lines.append(f"{label(name)} & {len(sel)} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def small_table() -> str:
    """Val, three 20-task settings (descriptive): gap to the best known plan at B1 on 8 cores, and proofs."""
    gaps = read(DOCS / "phase1b" / "anytime_gaps_val.csv")
    runs = read(DOCS / "phase1b" / "bks_val_runs.csv")
    methods = ("ALNS2", "CPSAT", "CONSTRUCT", "RL")
    lines = [r"\begin{table}[h]", r"\centering", r"\small", CAPTION_SMALL, r"\label{tab:supp-small}",
             r"\begin{tabular}{l" + "r" * (len(methods) + 1) + "}", r"\toprule",
             "Setting & " + " & ".join(NAMES[m] for m in methods) + r" & proven optimal \\", r"\midrule"]
    for name in SMALL:
        cells = []
        for m in methods:
            g = [x for x in gaps if x["setting"] == name and x["method"] == m and x["cores"] == "8"
                 and x["budget"] == "B1"]
            cells.append(f"{float(g[0]['gap_pct']):.2f}" if g else "--")
        full = [r for r in runs if r["setting"] == name and r["run"].startswith("CPFULL-8x60s")]
        proven = full[0]["proven_optimal"] if full else ""
        cells.append(f"{proven}/{full[0]['n']}" if proven else "--")
        lines.append(f"{label(name)} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def competitor_table() -> str:
    """Per-setting competitor choices made on dev (scripts/bks.py)."""
    lines = [r"\begin{table}[h]", r"\centering", r"\small", CAPTION_PARAMS, r"\label{tab:supp-params}",
             r"\begin{tabular}{lrrrr}", r"\toprule",
             r"Setting & $B_1$ (s) & CP-SAT LNS & Par.\ CP-SAT LNS & CP-SAT model arcs \\", r"\midrule"]
    for name in SETTINGS:
        knn = bks.CPFULL_KNN[name]
        b1 = configs.get(name).paper["RL(s.10)"].time_s
        lines.append(f"{label(name)} & {b1:.2f} & {bks.sub_time(name):g} & {bks.PCPSAT_SUB_TIME[name]:g} & "
                     f"{knn or 'all'}" + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", type=Path)
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "tables")
    args = ap.parse_args()
    gaps = read(next(args.results.glob("anytime_gaps_*.csv")))
    split = "test" if "test" in args.results.name else "val"
    out = {"supp_gaps.tex": gaps_table(gaps, split), "supp_ablation.tex": ablation_table(),
           "supp_refs.tex": references_table(), "supp_small.tex": small_table(),
           "supp_params.tex": competitor_table()}
    args.out.mkdir(parents=True, exist_ok=True)
    for name, text in out.items():
        (args.out / name).write_text(text)
    print(f"wrote {', '.join(out)} to {args.out} from {args.results}")


if __name__ == "__main__":
    main()
