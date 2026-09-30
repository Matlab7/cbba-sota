"""LaTeX tables of the paper from the campaign CSVs (scripts/anytime.py report --csv-dir) and rows.

Usage: make_tables.py RESULTS_DIR [--runs RUNS_DIR] [--out paper/tables]
RESULTS_DIR holds anytime_ratios_*.csv and anytime_gaps_*.csv (docs/results/test for the paper, docs/results/val-c1
for the dry run); RUNS_DIR the campaign's rows (runs/anytime_test, runs/anytime_c1) for the RL sample counts and ALNS
iteration rates. Writes c1c2.tex (ALNS2 / competitor ratios, C1 at B1 on 8 cores and C2 at 2 s on 1 core),
methods.tex (gap to the best known plan and CPU share per method at B1 on 8 cores), settings.tex (budgets and RL
samples) and numbers.tex (macros for the numbers quoted in the text), so that no number in the paper is typed by hand.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import anytime as at  # noqa: E402  (the harness's budgets and its definition of a disturbed row)
from cbba_sota.bench import configs

SETTINGS = ("SA-BT-25-5-50", "SA-BT-50-5-50", "SA-AT-50-5-50", "MA-AT-25-5-50", "MA-AT-50-5-50", "MA-AT-50-5-200",
            "MA-AT-150-10-500", "MA-AT-150-5-500")
COMPETITORS = (("CPSAT", "CP-LNS"), ("PCPSAT", "Parallel CP-LNS"), ("CPFULL", "CP-SAT full"),
               ("CTAS", "CTAS-D"), ("CONSTRUCT", "Greedy restarts"), ("RL", "RL policy"))
CLASSICAL = ("CPSAT", "PCPSAT", "CPFULL", "CONSTRUCT")  # the competitors quoted as one range (CTAS-D fails, RL apart)
METHODS = (("ALNS2", "ALNS (ours)"), ("CPSAT", "CP-LNS"), ("PCPSAT", "Parallel CP-LNS"),
           ("CPFULL", "CP-SAT full model"), ("CTAS", "CTAS-D (exact MILP)"), ("CONSTRUCT", "Greedy restarts"),
           ("RL", "RL policy (published)"))
FAMILY_TEXT = {"SA-BT": ("Single-skill robots,", "binary needs"), "SA-AT": ("Single-skill robots,", "additive needs"),
               "MA-AT": ("Multi-skill robots,", "additive needs")}
FAMILIES = (("C1 8 cores B1", "ConeB"), ("C1 8 cores 2B1", "ConeBB"), ("C2 1 core 2 s vs 8 cores B1", "Ctwo"),
            ("C2 1 core 1 s vs 8 cores B1", "CtwoOne"), ("C2 1 core 0.5 s vs 8 cores B1", "CtwoHalf"))
ALPHA = 0.05
CAPTION_RATIOS = (r"\caption{Paired makespan ratio ALNS / competitor (mean over the instances with bootstrap 95\% CI; below "
                  r"1: ALNS better). $^\dagger$: not significant after Holm's correction (one-sided paired $t$-test on log "
                  r"ratios, $\alpha = 0.05$). CTAS-D failures count as makespan 200; the share of instances it solved "
                  r"within the budget follows its ratio in parentheses; it is not run on the 500-task settings.}")
CAPTION_METHODS = (r"\caption{Mean gap to the best plan found by any run (\%) and CPU used (CPU seconds of all processes and "
                   r"threads, as a share of 8 cores for $B_1$); range over the eight settings. Every method runs on 8 cores "
                   r"at $B_1$, except ALNS on 1 core for 2\,s. CTAS-D: share of the instances solved within $B_1$, up to "
                   r"200 tasks.}")
CAPTION_SETTINGS = (r"\caption{The eight benchmark settings with 50 or more tasks. $B_1$ is the time budget of every method: "
                    r"the computation time that the benchmark paper reports for its policy with ten sampled rollouts "
                    r"\cite{dai2025heterogeneous}. RL rollouts: how many rollouts the released policy completes within "
                    r"$B_1$ on our 8 cores (range over the instances). The exact MILP (CTAS-D) is not run on the two "
                    r"500-task settings (Section~\ref{sec:setup}).}")
PANELS = (("C2 1 core 2 s vs 8 cores B1", (r"\multicolumn{7}{l}{\emph{(a) C2, a fraction of the compute: ALNS on 1 "
                                             r"core for 2\,s ({share}\% or less of a competitor's CPU time) vs.\ each "
                                             r"competitor on 8 cores at $B_1$}} \\")),
          ("C1 8 cores B1", (r"\multicolumn{7}{l}{\emph{(b) C1, equal compute: ALNS on 8 cores vs.\ each competitor on "
                             r"8 cores, both at budget $B_1$}} \\")))


def b1_of(name: str) -> float:
    """The budget B1 of a setting, in seconds."""
    return at.budgets(configs.get(name))["B1"]


def share_pct(seconds: float, names=SETTINGS) -> tuple[float, float]:
    """Smallest and largest CPU time of 1 core for ``seconds`` as a share (%) of 8 cores for B1."""
    v = [100 * seconds / (8 * b1_of(n)) for n in names]
    return min(v), max(v)


def read(path: Path) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))


def size(name: str) -> str:
    """Robots / species / tasks of a setting, e.g. ``25/5/50``."""
    _, a, s, t = name.rsplit("-", 3)
    return f"{a}/{s}/{t}"


def family(name: str) -> str:
    return name.rsplit("-", 3)[0]


def label(name: str) -> str:
    """Readable name of a setting for running text, e.g. ``multi-skill additive 25/5/50``."""
    a, b = FAMILY_TEXT[family(name)]
    return f"{a.rstrip(',').lower().replace(' robots', '')} {b.replace(' needs', '')} {size(name)}"


def family_rows(names) -> list[tuple[str, list[str]]]:
    """Consecutive settings grouped by family, in the order given."""
    out: list[tuple[str, list[str]]] = []
    for n in names:
        if out and out[-1][0] == family(n):
            out[-1][1].append(n)
        else:
            out.append((family(n), [n]))
    return out


def family_cell(fam: str, rows: int) -> str:
    a, b = FAMILY_TEXT[fam]
    box = rf"\parbox{{2.1cm}}{{\raggedright\scriptsize {a} {b}}}"
    return box if rows == 1 else rf"\multirow{{{rows}}}{{*}}{{{box}}}"


def key(other: str) -> str:
    """Method of a ratio row's ``other`` column (e.g. ``CPSAT-8@B1`` -> ``CPSAT``)."""
    return other.split("-")[0]


def fmt_ratio(r: dict | None) -> str:
    if r is None:
        return "--"
    ratio, lo, hi = float(r["ratio"]), float(r["lo"]), float(r["hi"])
    mark = "" if float(r["p_holm"]) < ALPHA else "$^\\dagger$"
    return f"{ratio:.3f}{mark} {{\\scriptsize [{lo:.2f}, {hi:.2f}]}}"


def c1c2(ratios: list[dict], gaps: list[dict]) -> str:
    by = {(r["setting"], r["comparison"], key(r["other"])): r for r in ratios}
    solved = {g["setting"]: g for g in gaps if g["method"] == "CTAS" and g["cores"] == "8" and g["budget"] == "B1"}
    ncol = len(COMPETITORS) + 2
    lines = [r"\begin{table*}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{3.5pt}",
             CAPTION_RATIOS,
             r"\label{tab:ratios}", r"\begin{tabular}{ll" + "c" * len(COMPETITORS) + "}", r"\toprule",
             r"Robots & Robots / species / tasks & " + " & ".join(name for _, name in COMPETITORS) + r" \\",
             r"\midrule"]
    for p, (tag, title) in enumerate(PANELS):
        lines.append(title.replace("{7}", f"{{{ncol}}}").replace("{share}", f"{share_pct(2)[1]:.1f}"))
        for fam, names in family_rows(SETTINGS):
            for i, name in enumerate(names):
                cells = []
                for k, _ in COMPETITORS:
                    cell = fmt_ratio(by.get((name, tag, k)))
                    if k == "CTAS" and cell != "--" and name in solved:
                        cell += f" ({float(solved[name]['success']) * 100:.0f}\\%)"
                    cells.append(cell)
                first = family_cell(fam, len(names)) if i == 0 else ""
                lines.append(f"{first} & {size(name)} & " + " & ".join(cells) + r" \\")
            lines.append(r"\cmidrule(l){2-" + str(ncol) + "}")
        lines[-1] = r"\midrule" if p < len(PANELS) - 1 else r"\bottomrule"
    lines += [r"\end{tabular}", r"\end{table*}"]
    return "\n".join(lines) + "\n"


def methods(gaps: list[dict]) -> str:
    rows = [g for g in gaps if g["cores"] == "8" and g["budget"] == "B1"]
    lines = [r"\begin{table}[t]", r"\centering", r"\small",
             CAPTION_METHODS,
             r"\label{tab:methods}", r"\begin{tabular}{lcc}", r"\toprule", r"Method & Gap to best (\%) & CPU used (\%) \\",
             r"\midrule"]
    one = [g for g in gaps if g["method"] == "ALNS2" and g["cores"] == "1" and g["budget"] == "2"
           and g["setting"] in SETTINGS]
    for k, name in METHODS:
        sel = [g for g in rows if g["method"] == k and g["setting"] in SETTINGS]
        if not sel:
            continue
        gap = [float(g["gap_pct"]) for g in sel]
        cpu = [float(g["cpu_share"]) for g in sel]
        if k == "CTAS":
            ok = [float(g["success"]) * 100 for g in sel]
            gtxt = f"solves {min(ok):.0f}--{max(ok):.0f}\\%"
        else:
            gtxt = f"{min(gap):.1f}--{max(gap):.1f}"
        if k == "ALNS2":
            name = "ALNS, 8 cores"
        lines.append(f"{name} & {gtxt} & {100 * min(cpu):.0f}--{100 * max(cpu):.0f}" + r" \\")
        if k == "ALNS2" and one:  # the same search on one core for 2 s, CPU as a share of 8 cores for B1
            gap1 = [float(g["gap_pct"]) for g in one]
            cpu1 = [100 * float(g["cpu_share"]) * 2 / (8 * b1_of(g["setting"])) for g in one]
            lines.append(rf"ALNS, 1 core, 2\,s & {min(gap1):.1f}--{max(gap1):.1f} & {min(cpu1):.2f}--{max(cpu1):.1f} \\")
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


PROGRESS = (("C2 1 core 0.5 s vs 8 cores B1", r"1 core, 0.5\,s$^*$", 0.5),
            ("C2 1 core 1 s vs 8 cores B1", r"1 core, 1\,s$^*$", 1.0),
            ("C2 1 core 2 s vs 8 cores B1", r"1 core, 2\,s", 2.0),
            ("C1 8 cores B1", r"8 cores, $B_1$", None))
CAPTION_PROGRESS = (r"\caption{Settings (of 8; CTAS-D of 6) where ALNS with growing compute is significantly better than "
                    r"each competitor on 8 cores at $B_1$ (Holm per row); CPU: share of a competitor's CPU time. "
                    r"$^*$: descriptive.}")


def progress_table(ratios: list[dict]) -> str:
    """Settings won against each competitor as ALNS's compute grows (C2 at 0.5, 1, 2 s on 1 core; C1 on 8 cores)."""
    short = {"CPSAT": "CP-LNS", "PCPSAT": "P.\\ CP-LNS", "CPFULL": "CP-SAT", "CTAS": "CTAS-D", "CONSTRUCT": "Greedy",
             "RL": "RL"}
    lines = [r"\begin{table}[t]", r"\centering", r"\footnotesize", r"\setlength{\tabcolsep}{2.6pt}", CAPTION_PROGRESS,
             r"\label{tab:progress}", r"\begin{tabular}{@{}lr" + "c" * (len(COMPETITORS) + 1) + "@{}}", r"\toprule",
             r"ALNS & CPU (\%) & " + " & ".join(short[k] for k, _ in COMPETITORS) + r" & all \\", r"\midrule"]
    for tag, name, seconds in PROGRESS:
        rs = [r for r in ratios if r["comparison"] == tag and r["setting"] in SETTINGS]
        won = lambda r: float(r["p_holm"]) < ALPHA and float(r["ratio"]) < 1  # noqa: E731
        cells = [str(sum(won(r) for r in rs if key(r["other"]) == k)) for k, _ in COMPETITORS]
        every = sum(all(won(r) for r in rs if r["setting"] == n) for n in SETTINGS if any(r["setting"] == n for r in rs))
        cpu = "100" if seconds is None else rf"$\le$\,{share_pct(seconds)[1]:.1f}"
        lines.append(f"{name} & {cpu} & " + " & ".join(cells) + rf" & \textbf{{{every}}} \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def load_rows(runs: Path | None) -> list[dict]:
    out = []
    if runs is None:
        return out
    for name in SETTINGS:
        path = runs / f"{name}.jsonl"
        if not path.exists():
            continue
        for line in path.open():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not at.disturbed(r):
                out.append(r)
    return out


def rl_samples(rows: list[dict]) -> dict[str, tuple[int, int]]:
    """Per setting, the fewest and most rollouts RL(s.N) drew on 8 cores at B1."""
    out = {}
    for name in SETTINGS:
        n = [r["n_samples"] for r in rows if r["setting"] == name and r["method"] == "RL" and r["cores"] == 8
             and r["budget"] == "B1" and r.get("n_samples")]
        if n:
            out[name] = (min(n), max(n))
    return out


def settings_table(samples: dict[str, tuple[int, int]]) -> str:
    lines = [r"\begin{table}[t]", r"\centering", r"\small",
             CAPTION_SETTINGS,
             r"\label{tab:settings}", r"\begin{tabular}{lcrr}", r"\toprule",
             r"Robots & Robots / species / tasks & $B_1$ (s) & RL rollouts \\", r"\midrule"]
    for fam, names in family_rows(SETTINGS):
        for i, name in enumerate(names):
            b1 = configs.get(name).paper["RL(s.10)"].time_s
            lo, hi = samples.get(name, (None, None))
            n = f"{lo}--{hi}" if lo is not None else "--"
            first = family_cell(fam, len(names)) if i == 0 else ""
            lines.append(f"{first} & {size(name)} & {b1:.1f} & {n}" + r" \\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines += [r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


STEP_TEXT = {"ALNS2": "one remove and re-insert iteration", "CPSAT": "one CP-SAT solve of a freed window",
             "PCPSAT": "one CP-SAT solve of a freed window", "CPFULL": "one solve of the whole model",
             "CTAS": "one solve of the whole MILP", "CONSTRUCT": "one greedy construction",
             "RL": "one rollout of the policy"}
CAPTION_STEPS = (r"\caption{How the methods spend their budget: 8 cores for $B_1$ (first row), or 1 core for 2\,s (ALNS "
                 r"in C2, last row). Median steps per run, with the time of one step on one worker. CP-SAT full model and "
                 r"CTAS-D spend it all in one solve (runs proven optimal / runs with a plan).}")


def fmt_count(n: float) -> str:
    if n >= 1e6:
        return f"{n / 1e6:.1f}\\,M"
    if n >= 1e4:
        return f"{n / 1e3:.0f}\\,k"
    if n >= 1e3:
        return f"{n / 1e3:.1f}\\,k"
    return f"{n:.0f}"


def fmt_time(t: float) -> str:
    if t < 1e-3:
        return f"{t * 1e3:.2f}\\,ms"
    if t < 1:
        return f"{t * 1e3:.0f}\\,ms"
    return f"{t:.1f}\\,s" if t < 10 else f"{t:.0f}\\,s"


def cell2(count: str, t: str) -> str:
    """A count over its time per step, in one table cell."""
    return rf"\begin{{tabular}}[t]{{@{{}}c@{{}}}}{count}\\{{\scriptsize ({t})}}\end{{tabular}}"


def steps_table(rows: list[dict], timing: dict) -> str:
    """Steps per run and time per step (per worker) at B1 on 8 cores, median over instances per task count."""
    import statistics as st
    sizes = (50, 200, 500)
    lines = [r"\begin{table}[t]", r"\centering", r"\footnotesize", r"\setlength{\tabcolsep}{2.5pt}", CAPTION_STEPS,
             r"\label{tab:steps}", (r"\begin{tabular}{@{}>{\raggedright\arraybackslash}p{2.1cm}"
                                    r">{\raggedright\arraybackslash}p{2.3cm}ccc@{}}"), r"\toprule",
             r"Method & One step & 50 tasks & 200 tasks & 500 tasks \\", r"\midrule"]
    budget = []
    for m in sizes:
        b = sorted(b1_of(n) for n in SETTINGS if configs.get(n).n_tasks == m)
        budget.append(f"{b[0]:.0f}--{b[-1]:.0f}\\,s" if b[-1] - b[0] > 1 else f"{b[0]:.0f}\\,s")
    lines.append(r"\emph{Budget $B_1$} & {\scriptsize wall-clock, 8 cores} & " + " & ".join(budget)
                 + r" \\ \midrule")
    for k, name in METHODS:
        cells = []
        for m in sizes:
            names = [n for n in SETTINGS if configs.get(n).n_tasks == m]
            sel = [r for r in rows if r["setting"] in names and r["method"] == k and r["cores"] == 8
                   and r["budget"] == "B1"]
            if k == "CPFULL":
                opt = sum(r.get("status") == "OPTIMAL" for r in sel)
                cells.append(cell2(r"1 solve", f"{opt}/{len(sel)}") if sel else "--")
                continue
            if k == "CTAS":
                found = sum(bool(r.get("success")) for r in sel)
                cells.append(cell2(r"1 solve", f"{found}/{len(sel)}") if sel else "--")
                continue
            if k == "CONSTRUCT":
                t = [st.mean(v for v in timing[n].values()) for n in names if n in timing]
                b1 = [configs.get(n).paper["RL(s.10)"].time_s for n in names]
                if not t:
                    cells.append("--")
                    continue
                tt = st.median(t)
                cells.append(cell2(fmt_count(8 * st.median(b1) / tt), fmt_time(tt)))
                continue
            field = {"ALNS2": "iterations", "CPSAT": "iterations", "PCPSAT": "iterations", "RL": "n_samples"}[k]
            n = [r[field] for r in sel if r.get(field)]
            if not n:
                cells.append("--")
                continue
            workers = 1 if k == "CPSAT" else 8
            per = [workers * r["budget_s"] / r[field] for r in sel if r.get(field)]
            cells.append(cell2(fmt_count(st.median(n)), fmt_time(st.median(per))))
        lines.append(f"{name} & {{\\scriptsize {STEP_TEXT[k]}}} & " + " & ".join(cells) + r" \\ \addlinespace[2pt]")
    cells = []  # ALNS in C2: one core for 2 s
    for m in sizes:
        names = [n for n in SETTINGS if configs.get(n).n_tasks == m]
        sel = [r for r in rows if r["setting"] in names and r["method"] == "ALNS2" and r["cores"] == 1
               and r["budget"] == "2" and r.get("iterations")]
        cells.append(cell2(fmt_count(st.median(r["iterations"] for r in sel)),
                           fmt_time(st.median(r["budget_s"] / r["iterations"] for r in sel))) if sel else "--")
    lines.append(r"\midrule ALNS, 1 core & {\scriptsize the same iteration, for 2\,s} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def thousands(x: float) -> str:
    return f"{round(x, -2):,.0f}".replace(",", "{,}")


def numbers(ratios: list[dict], gaps: list[dict], rows: list[dict]) -> str:
    """Macros for the text: ranges of ratios and gains per competitor and claim, where claims hold, CPU use, CTAS-D
    solved shares, the worker ablation, RL sample counts and ALNS iteration rates."""
    out = []

    def macro(name: str, value: str) -> None:
        out.append(f"\\newcommand{{\\{name}}}{{{value}}}")

    def sel(tag: str, keys: tuple[str, ...] | None = None) -> list[dict]:
        return [r for r in ratios if r["comparison"] == tag and r["setting"] in SETTINGS
                and (keys is None or key(r["other"]) in keys)]

    def gain(rs: list[dict]) -> tuple[str, str]:
        """Range of 100 (1 - ratio): how much shorter ALNS's plans are, in %."""
        if not rs:
            return "--", "--"
        g = [100 * (1 - float(r["ratio"])) for r in rs]
        return f"{min(g):.1f}", f"{max(g):.1f}"

    for tag, pre in FAMILIES:
        rs = sel(tag)
        for k, _ in COMPETITORS:
            vals = [float(r["ratio"]) for r in rs if key(r["other"]) == k]
            macro(f"{pre}{k.lower()}lo", f"{min(vals):.3f}" if vals else "--")
            macro(f"{pre}{k.lower()}hi", f"{max(vals):.3f}" if vals else "--")
        lo, hi = gain(sel(tag, CLASSICAL))
        macro(f"{pre}classlo", lo)
        macro(f"{pre}classhi", hi)
        lo, hi = gain(sel(tag, ("RL",)))
        macro(f"{pre}rlgainlo", lo)
        macro(f"{pre}rlgainhi", hi)
        present = [s for s in SETTINGS if any(r["setting"] == s for r in rs)]
        fails = {s: [r for r in rs if r["setting"] == s and not (float(r["p_holm"]) < ALPHA and float(r["ratio"]) < 1)]
                 for s in present}
        held = [s for s in present if not fails[s]]
        names = dict(COMPETITORS)
        macro(f"{pre}held", str(len(held)))
        macro(f"{pre}verdict", "is met" if len(held) >= 6 else "is not met")
        macro(f"{pre}settings", str(len(present)))
        macro(f"{pre}heldlist", "; ".join(label(s) for s in held) or "none")
        macro(f"{pre}faillist", "; ".join(f"{label(s)} (against " + ", ".join(names[key(r['other'])] for r in f) + ")"
                                          for s, f in fails.items() if f) or "none")
        macro(f"{pre}tests", str(len(rs)))
        macro(f"{pre}sig", str(sum(float(r["p_holm"]) < ALPHA for r in rs)))
        macro(f"{pre}maxp", f"{max(float(r['p_holm']) for r in rs):.2g}" if rs else "--")
        macro(f"{pre}won", str(sum(int(r["wins"]) for r in rs)))
        macro(f"{pre}pairs", str(sum(int(r["n"]) for r in rs)))
    for seconds, pre in ((2, "ShareTwo"), (1, "ShareOne"), (0.5, "ShareHalf")):
        lo, hi = share_pct(seconds)
        macro(f"{pre}lo", f"{lo:.2g}")
        macro(f"{pre}hi", f"{hi:.1f}")
        macro(f"{pre}foldlo", f"{8 * min(map(b1_of, SETTINGS)) / seconds:,.0f}".replace(",", "{,}"))
        macro(f"{pre}foldhi", f"{8 * max(map(b1_of, SETTINGS)) / seconds:,.0f}".replace(",", "{,}"))
    macro("CpuBonelo", f"{8 * min(map(b1_of, SETTINGS)):,.0f}".replace(",", "{,}"))  # CPU-seconds of 8 cores at B1
    macro("CpuBonehi", f"{8 * max(map(b1_of, SETTINGS)):,.0f}".replace(",", "{,}"))
    for tag, pre in FAMILIES[2:]:  # C2 against the learned policy alone
        rl = [r for r in sel(tag, ("RL",)) if float(r["p_holm"]) < ALPHA and float(r["ratio"]) < 1]
        macro(f"{pre}RLheld", str(len(rl)))
    abl = sel("ablation 8 workers / 1 worker B1")
    macro("Workerslo", f"{min(float(r['ratio']) for r in abl):.3f}" if abl else "--")
    macro("Workershi", f"{max(float(r['ratio']) for r in abl):.3f}" if abl else "--")
    lo, hi = gain(abl)
    macro("Workersgainlo", lo)
    macro("Workersgainhi", hi)
    b1 = [g for g in gaps if g["cores"] == "8" and g["budget"] == "B1" and g["setting"] in SETTINGS]
    for k, _ in METHODS:
        m = k.lower().rstrip("2")  # macro names cannot hold digits (ALNS2 -> alns)
        cpu = [float(g["cpu_share"]) for g in b1 if g["method"] == k]
        macro(f"Cpu{m}lo", f"{100 * min(cpu):.0f}" if cpu else "--")  # % of 8 cores for B1, as in Table 4
        macro(f"Cpu{m}hi", f"{100 * max(cpu):.0f}" if cpu else "--")
        gap = [float(g["gap_pct"]) for g in b1 if g["method"] == k]
        macro(f"Gap{m}lo", f"{min(gap):.1f}" if gap else "--")
        macro(f"Gap{m}hi", f"{max(gap):.1f}" if gap else "--")
    one = {g["setting"]: float(g["gap_pct"]) for g in gaps if g["method"] == "ALNS2" and g["cores"] == "1"
           and g["budget"] == "B1"}
    ahead = [s for s in SETTINGS if s in one and all(one[s] < float(g["gap_pct"]) for g in b1
                                                     if g["setting"] == s and g["method"] != "ALNS2")]
    macro("OneCoreBoneAhead", str(len(ahead)))
    mean = {(g["setting"], g["method"]): float(g["mean_makespan"]) for g in b1}
    rr = [100 * (1 - mean[s, "CONSTRUCT"] / mean[s, "RL"]) for s in SETTINGS if (s, "CONSTRUCT") in mean
          and (s, "RL") in mean]
    macro("RestartsRLlo", f"{min(rr):.0f}" if rr else "--")
    macro("RestartsRLhi", f"{max(rr):.0f}" if rr else "--")
    ctas = [g for g in b1 if g["method"] == "CTAS"]
    macro("CtasSolved", str(sum(round(float(g["success"]) * int(g["n"])) for g in ctas)))
    macro("CtasRuns", str(sum(int(g["n"]) for g in ctas)))
    macro("LateBone", str(sum(int(g["late"]) for g in b1)))
    macro("RunsBone", str(sum(int(g["n"]) for g in b1)))
    samples = rl_samples(rows)
    if samples:
        macro("RLNlo", f"{min(lo for lo, _ in samples.values()) / 10:.0f}")
        macro("RLNhi", f"{max(hi for _, hi in samples.values()) / 10:.0f}")
    else:
        macro("RLNlo", "--")
        macro("RLNhi", "--")
    rate = {}
    for name in SETTINGS:
        its = [r["iterations"] / r["budget_s"] for r in rows if r["setting"] == name and r["method"] == "ALNS2"
               and r["cores"] == 1 and r["budget"] == "B1" and r.get("iterations")]
        if its:
            rate[name] = sum(its) / len(its)
    fifty = [v for s, v in rate.items() if configs.get(s).n_tasks == 50]
    big = [v for s, v in rate.items() if configs.get(s).n_tasks == 500]
    macro("ItsFiftylo", thousands(min(fifty)) if fifty else "--")
    macro("ItsFiftyhi", thousands(max(fifty)) if fifty else "--")
    macro("ItsFivehundred", thousands(sum(big) / len(big)) if big else "--")
    return "\n".join(out) + "\n"


def rl_sampling(rl_dir: Path | None, rows: list[dict]) -> str:
    """Macros \\RLsamplegainlo/hi: how much shorter RL(s.N)'s plans on 8 cores at B1 (campaign rows) are than
    RL(s.10)'s (``runs/rl/<setting>/test.jsonl``, the reproduction runs), mean over instances per setting, in %.
    Instances are matched by seed and fingerprint."""
    gains = []
    for name in SETTINGS:
        path = rl_dir / name / "test.jsonl" if rl_dir is not None else None
        if path is None or not path.exists():
            continue
        s10 = {}
        for line in path.open():
            r = json.loads(line)
            if r["method"] == "RL(s.10)" and r.get("split") == "test":
                s10[r["seed"], r.get("fingerprint")] = r["makespan"] if r["success"] else 200.0
        pairs = [(r["makespan"] if r["success"] else 200.0, s10[r["seed"], r.get("fingerprint")]) for r in rows
                 if r["setting"] == name and r["method"] == "RL" and r["cores"] == 8 and r["budget"] == "B1"
                 and (r["seed"], r.get("fingerprint")) in s10]
        if pairs:
            gains.append(100 * sum(1 - n / t for n, t in pairs) / len(pairs))
    lo = f"{min(gains):.1f}" if gains else "--"
    hi = f"{max(gains):.1f}" if gains else "--"
    return f"\\newcommand{{\\RLsamplegainlo}}{{{lo}}}\n\\newcommand{{\\RLsamplegainhi}}{{{hi}}}\n"


def bks_numbers(path: Path | None, pre: str) -> str:
    """Macros on the best known plans (``bks.py collect`` CSV): how far the best plan without any ALNS component is
    above the best known (mean per setting, range over the settings), on how many instances it matched the best
    ALNS-derived plan, and the certified lower bounds' distance below the best known, where CP-SAT gave one."""
    out = []

    def macro(name: str, value: str) -> None:
        out.append(f"\\newcommand{{\\{pre}{name}}}{{{value}}}")

    rows = [r for r in read(path) if r["setting"] in SETTINGS] if path is not None else []
    free, bound = [], []
    matched = 0
    for name in SETTINGS:
        sel = [r for r in rows if r["setting"] == name and r["alns_free_best"]]
        if sel:
            free.append(sum(100 * (float(r["alns_free_best"]) - float(r["bks"])) / float(r["bks"]) for r in sel)
                        / len(sel))
            matched += sum(float(r["alns_free_best"]) <= float(r["alns_best"]) + 1e-9 for r in sel)
        lb = [r for r in sel if r["lb"]]
        if lb:
            bound.append(sum(100 * (float(r["bks"]) - float(r["lb"])) / float(r["bks"]) for r in lb) / len(lb))
    macro("Freelo", f"{min(free):.1f}" if free else "--")
    macro("Freehi", f"{max(free):.1f}" if free else "--")
    macro("Freematched", str(matched))
    macro("Instances", str(sum(1 for r in rows if r["alns_free_best"])))
    macro("Boundlo", f"{min(bound):.0f}" if bound else "--")
    macro("Boundhi", f"{max(bound):.0f}" if bound else "--")
    return "\n".join(out) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", type=Path)
    ap.add_argument("--runs", type=Path, help="campaign rows (runs/anytime_test or runs/anytime_c1)")
    ap.add_argument("--bks-val", type=Path, help="bks.py collect CSV of val (independent references)")
    ap.add_argument("--bks-test", type=Path, help="bks.py collect CSV of the test split")
    ap.add_argument("--rl-s10", type=Path, help="RL reproduction rows (runs/rl) for RL(s.10) vs RL(s.N) on test")
    ap.add_argument("--timing", type=Path, default=Path(__file__).resolve().parents[1] / "tables" /
                    "constructor_timing.json", help="time_constructor.py output")
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "tables")
    args = ap.parse_args()
    ratios = read(next(args.results.glob("anytime_ratios_*.csv")))
    gaps = read(next(args.results.glob("anytime_gaps_*.csv")))
    rows = load_rows(args.runs)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "c1c2.tex").write_text(c1c2(ratios, gaps))
    (args.out / "methods.tex").write_text(methods(gaps))
    (args.out / "progress.tex").write_text(progress_table(ratios))
    (args.out / "settings.tex").write_text(settings_table(rl_samples(rows)))
    timing = json.loads(args.timing.read_text()) if args.timing.exists() else {}
    (args.out / "steps.tex").write_text(steps_table(rows, timing))
    (args.out / "numbers.tex").write_text(f"% generated from {args.results} and {args.runs} by paper/tools/make_tables.py\n"
                                          + numbers(ratios, gaps, rows) + bks_numbers(args.bks_val, "RefVal")
                                          + bks_numbers(args.bks_test, "RefTest") + rl_sampling(args.rl_s10, rows))
    print(f"wrote {args.out}/c1c2.tex, methods.tex, settings.tex, numbers.tex from {args.results} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
