"""Track D C3 analysis (spec 4.3 lease tuning, 4.5 variant selection, 5.3 DS, 8 gates K5-K7, 9.4 E3a) on dev rows.

Subcommands:
  design --rows R...                 station-outsider model x key rule (design pilot)
  tune   --rows R... --out J         per (arm, rho): the lease (g, L) of the shared grid with the lowest GM makespan
                                     (failures scored 200), pooled over settings, cells, instances and seeds; frozen
                                     into J before any evaluation row is read
  gates  --rows R... [--grid G...]   K5, K6 (E3a max regret, DS, variant selection), K7 (conclusions under every
                                     common lease of the grid rows G), secondary endpoints; --json writes the numbers

Conventions: the unit is the instance (setting, inst); log makespans are averaged over CRN seeds first; the value is
``makespan_or_cap`` (a failed episode counts 200, spec 9.1 sensitivity; success is reported separately, and a
both-succeed sensitivity is printed). Regret(arm, cell) = GM(arm) / min over the fixed architectures
A = {CEN-F, HYB, REP-clamp, INF-r} of GM, cells = (family cell, rho). SPARC's stranded variants: V1 = HYB,
V2 = SPARC, V3 = SPARC-V3 (theta 0.7).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FIXED = ["CEN-F", "HYB", "REP-clamp", "INF-r"]
VARIANTS = {"V1": "HYB", "V2": "SPARC", "V3": "SPARC-V3"}
PREF = ["V2", "V1", "V3"]  # spec 4.5 tie rule (within 1 point)
VAL = "makespan_or_cap"


def load(paths) -> list[dict]:
    rows = []
    for p in paths:
        for line in Path(p).read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                if r.get("status") == "ok":
                    rows.append(r)
    return rows


def dedup(rows: list[dict]) -> list[dict]:
    """One row per case id (the last one)."""
    out = {}
    for r in rows:
        out[r["case_id"]] = r
    return list(out.values())


def cluster_log_means(rows, arm, cell, rho, value=VAL, where=None):
    """{(setting, inst): mean over seeds of log value} for one arm and cell."""
    acc: dict = {}
    for r in rows:
        if r["arm"] != arm or r["cell"] != cell or (rho is not None and not _rho_eq(r["rho"], rho)):
            continue
        if where is not None and not where(r):
            continue
        v = r[value]
        if v is None:
            continue
        acc.setdefault((r["setting"], r["inst"]), []).append(math.log(float(v)))
    return {k: float(np.mean(v)) for k, v in acc.items()}


def _rho_eq(a, b) -> bool:
    return (math.isinf(a) and math.isinf(b)) or (not math.isinf(a) and not math.isinf(b) and abs(a - b) < 1e-9)


def gm_of(rows, arm, cell, rho, keys=None):
    d = cluster_log_means(rows, arm, cell, rho)
    ks = keys if keys is not None else list(d)
    return float(np.exp(np.mean([d[k] for k in ks]))) if ks else float("nan")


def ratio_ci(rows, a, cell_a, rho_a, b, cell_b, rho_b, reps=10000, seed=0):
    """GM ratio a/b over instances (seeds averaged in log), cluster bootstrap 95% CI (strata = settings)."""
    from cbba_sota.dyn.stats import gm_ratio

    da = cluster_log_means(rows, a, cell_a, rho_a)
    db = cluster_log_means(rows, b, cell_b, rho_b)
    ks = sorted(set(da) & set(db))
    if len(ks) < 2:
        return None
    lr = np.array([da[k] - db[k] for k in ks])
    return gm_ratio(lr, ks, strata=[k[0] for k in ks], reps=reps, seed=seed)


def ds_rows(rows, rho_star: float, name="DS") -> list[dict]:
    """The deployment switch as rows: CEN-F if rho >= rho*, else REP-clamp (spec 5.3)."""
    out = []
    for r in rows:
        if math.isinf(r["rho"]):
            continue
        src = "CEN-F" if r["rho"] >= rho_star - 1e-9 else "REP-clamp"
        if r["arm"] == src:
            out.append({**r, "arm": name})
    return out


def cells_of(rows):
    return sorted({(r["cell"], r["rho"]) for r in rows if not math.isinf(r["rho"])}, key=lambda c: (c[0], -c[1]))


def regret_table(rows, arms, cells):
    """{arm: {cell: regret}} and {arm: max regret} (paired instances: intersection over arms of each cell)."""
    reg, mx = {a: {} for a in arms}, {}
    for c in cells:
        per = {a: cluster_log_means(rows, a, c[0], c[1]) for a in arms + FIXED}
        ks = set.intersection(*(set(v) for v in per.values()))
        if not ks:
            continue
        ks = sorted(ks)
        gm = {a: float(np.mean([per[a][k] for k in ks])) for a in per}
        best = min(gm[a] for a in FIXED)
        for a in arms:
            reg[a][c] = float(np.exp(gm[a] - best))
    for a in arms:
        mx[a] = max(reg[a].values()) if reg[a] else float("nan")
    return reg, mx


def choose_ds(rows, cells):
    rhos = sorted({c[1] for c in cells})
    best = None
    res = {}
    for rs in rhos + [math.inf]:
        dsr = ds_rows(rows, rs, "DS")
        _, mx = regret_table(rows + dsr, ["DS"], cells)
        res[rs] = mx["DS"]
        if best is None or mx["DS"] < res[best] - 1e-12:
            best = rs
    return best, res


def choose_variant(mx: dict) -> str:
    vals = {v: mx[a] for v, a in VARIANTS.items() if a in mx and np.isfinite(mx[a])}
    lo = min(vals.values())
    for v in PREF:
        if v in vals and vals[v] <= lo + 0.01 + 1e-12:
            return v
    return min(vals, key=vals.get)


# ------------------------------------------------------------------------------------------------ design
def cmd_design(args) -> None:
    rows = dedup(load(args.rows))
    confs = sorted({(r["frozen_model"], r["keys"]) for r in rows})
    arms = sorted({r["arm"] for r in rows})
    cells = sorted({(r["cell"], r["rho"]) for r in rows}, key=lambda c: (c[0], -c[1]))
    print(f"{len(rows)} rows; configurations {confs}")
    print("GM makespan (failures 200) and success rate, per arm x cell x (frozen model, keys):")
    for a in arms:
        for c in cells:
            if a == "FULL" and not math.isinf(c[1]):
                continue
            if a != "FULL" and math.isinf(c[1]):
                continue
            line = f"  {a:8s} {c[0]:9s} rho={c[1]:<4g}"
            for fm, ky in confs:
                sub = [r for r in rows if r["frozen_model"] == fm and r["keys"] == ky]
                g = gm_of(sub, a, c[0], c[1])
                s = [r["success"] for r in sub if r["arm"] == a and r["cell"] == c[0] and _rho_eq(r["rho"], c[1])]
                if s:
                    line += f" | {fm}/{ky}: {g:6.2f} ({np.mean(s):.2f})"
            print(line)
    print("Paired GM ratios vs FULL of the same key rule (pooled over cells; instances x seeds of the design set):")
    for fm, ky in confs:
        sub = [r for r in rows if r["frozen_model"] == fm and r["keys"] == ky]
        full = [r for r in rows if r["arm"] == "FULL" and r["keys"] == ky]
        for a in arms:
            if a == "FULL":
                continue
            for rho in sorted({c[1] for c in cells if not math.isinf(c[1])}, reverse=True):
                lr = []
                for cell in sorted({c[0] for c in cells}):
                    da = cluster_log_means(sub, a, cell, rho)
                    db = cluster_log_means(full, "FULL", cell, math.inf)
                    lr += [da[k] - db[k] for k in set(da) & set(db)]
                if lr:
                    print(f"  {fm}/{ky:8s} {a:6s} rho={rho:<4g} {np.exp(np.mean(lr)):.3f} (n={len(lr)})")


# ------------------------------------------------------------------------------------------------ tune
def cmd_tune(args) -> None:
    rows = dedup(load(args.rows))
    rows = [r for r in rows if not math.isinf(r["rho"])]
    arms = sorted({r["arm"] for r in rows})
    rhos = sorted({r["rho"] for r in rows}, reverse=True)
    leases = sorted({(r["lease_g"], r["lease_L"]) for r in rows})
    choice, table = {}, {}
    for a in arms:
        choice[a], table[a] = {}, {}
        for rho in rhos:
            per = {}
            for gl in leases:
                acc: dict = {}
                for r in rows:
                    if r["arm"] == a and _rho_eq(r["rho"], rho) and (r["lease_g"], r["lease_L"]) == gl:
                        acc.setdefault((r["setting"], r["inst"], r["cell"]), []).append(math.log(r[VAL]))
                per[gl] = acc
            ks = set.intersection(*(set(v) for v in per.values())) if per else set()
            if not ks:
                continue
            gm = {gl: float(np.exp(np.mean([np.mean(per[gl][k]) for k in ks]))) for gl in leases}
            succ = {gl: float(np.mean([r["success"] for r in rows if r["arm"] == a and _rho_eq(r["rho"], rho)
                                       and (r["lease_g"], r["lease_L"]) == gl])) for gl in leases}
            best = min(leases, key=lambda gl: (gm[gl], gl))
            choice[a][f"{rho:g}"] = list(best)
            table[a][f"{rho:g}"] = {f"{g:g}:{L:g}": [gm[(g, L)], succ[(g, L)]] for g, L in leases}
            print(f"{a:10s} rho={rho:<4g} best g={best[0]:g} L={best[1]:g}  " +
                  "  ".join(f"{g:g}:{L:g}={gm[(g, L)]:.2f}" for g, L in leases))
    out = {"rule": "per (arm, rho): lowest GM of makespan_or_cap over (setting, inst, cell) clusters, seeds "
                   "averaged in log; ties -> smallest (g, L)", "rows": [str(p) for p in args.rows],
           "choice": choice, "table": table}
    Path(args.out).write_text(json.dumps(out, indent=1))
    print("wrote", args.out)


# ------------------------------------------------------------------------------------------------ gates
def conclusions(rows, full_rows, verbose=False, reps=10000):
    """K5 / K6 / variant / DS / secondary on one set of arm rows (one lease per arm and rho) plus FULL rows."""
    from cbba_sota.dyn.stats import max_regret_test

    cells = cells_of(rows)
    arms_present = sorted({r["arm"] for r in rows})
    tested = [a for a in ["CEN-F", "HYB", "REP-clamp", "INF-r", "SPARC", "SPARC-V3"] if a in arms_present]
    reg, mx = regret_table(rows, tested, cells)
    variant = choose_variant(mx)
    sp = VARIANTS[variant]
    rho_star, ds_mx = choose_ds(rows, cells)
    dsr = ds_rows(rows, rho_star, "DS")
    reg_ds, mx_ds = regret_table(rows + dsr, ["DS"], cells)
    best_fixed_mx = min(mx[a] for a in FIXED)
    best_fixed = min(FIXED, key=lambda a: mx[a])
    k6_margin = best_fixed_mx - mx[sp]
    k6_fire = (k6_margin < 0.02 - 1e-12) or (mx_ds["DS"] <= mx[sp] + 1e-12)
    out = {"cells": [list(c) for c in cells], "regret": {a: {f"{c[0]}@{c[1]:g}": v for c, v in reg[a].items()}
                                                         for a in tested},
           "regret_DS": {f"{c[0]}@{c[1]:g}": v for c, v in reg_ds["DS"].items()},
           "max_regret": mx, "max_regret_DS": mx_ds["DS"], "rho_star": rho_star,
           "ds_max_regret_by_rho_star": {f"{k:g}": v for k, v in ds_mx.items()},
           "variant": variant, "sparc_arm": sp, "best_fixed": best_fixed, "best_fixed_max_regret": best_fixed_mx,
           "k6_margin_points": 100 * k6_margin, "k6_fire": bool(k6_fire)}
    # E3a test (simultaneous cluster bootstrap)
    try:
        allrows = rows + dsr
        e3 = max_regret_test(allrows, arms_fixed=FIXED, test_arm=sp, comparators=FIXED + ["DS"],
                             cell_keys=("cell", "rho"), cluster_keys=("setting", "inst"), method_key="arm",
                             value=VAL, reps=reps)
        out["e3a"] = {"diff": e3.diff, "diff_hi95": e3.diff_hi, "diff_log": e3.diff_log,
                      "diff_log_hi95": e3.diff_log_hi, "comparator": e3.comparator, "n_clusters": e3.n_clusters,
                      "holds": bool(e3.diff_log_hi < 0)}
    except ValueError as e:
        out["e3a"] = {"error": str(e)}
    # K5: headroom at the lowest rho (0.5)
    lo = min(c[1] for c in cells)
    k5 = {}
    for cell in sorted({c[0] for c in cells}):
        gms = {a: gm_of(rows, a, cell, lo) for a in FIXED}
        bf = min(gms, key=gms.get)
        rr = ratio_ci(rows + full_rows, bf, cell, lo, "FULL", cell, math.inf, reps=reps)
        k5[cell] = {"best_fixed": bf, "ratio_to_full": rr.ratio if rr else None, "lo": rr.lo if rr else None,
                    "hi": rr.hi if rr else None}
    # pooled over families: the fixed architecture with the lowest pooled GM at rho_lo
    pooled = {}
    for a in FIXED:
        lr = []
        for cell in k5:
            da = cluster_log_means(rows, a, cell, lo)
            db = cluster_log_means(full_rows, "FULL", cell, math.inf)
            lr += [da[k] - db[k] for k in set(da) & set(db)]
        pooled[a] = float(np.exp(np.mean(lr))) if lr else float("nan")
    bfp = min(pooled, key=pooled.get)
    out["k5"] = {"rho": lo, "per_family": k5, "pooled_ratio_to_full": pooled, "pooled_best_fixed": bfp,
                 "fire": bool(pooled[bfp] <= 1.05)}
    # secondary: SPARC vs CEN-F
    sec = {}
    for c in cells:
        rr = ratio_ci(rows, sp, c[0], c[1], "CEN-F", c[0], c[1], reps=reps)
        if rr:
            sec[f"{c[0]}@{c[1]:g}"] = {"ratio": rr.ratio, "lo": rr.lo, "hi": rr.hi}
    out["sparc_vs_cenf"] = sec
    if sp != "SPARC" and "SPARC" in arms_present:  # the spec's default variant (V2) as well
        sec2 = {}
        for c in cells:
            rr = ratio_ci(rows, "SPARC", c[0], c[1], "CEN-F", c[0], c[1], reps=reps)
            if rr:
                sec2[f"{c[0]}@{c[1]:g}"] = {"ratio": rr.ratio, "lo": rr.lo, "hi": rr.hi}
        out["sparc_v2_vs_cenf"] = sec2
    return out


def summarize_costs(rows, full_rows):
    out = {}
    for r in rows + full_rows:
        k = (r["arm"], r["cell"], r["rho"])
        out.setdefault(k, []).append(r)
    res = {}
    for k, g in out.items():
        res[f"{k[0]}|{k[1]}|{k[2]:g}"] = {
            "n": len(g), "success": float(np.mean([r["success"] for r in g])),
            "gm": float(np.exp(np.mean([math.log(r[VAL]) for r in g]))),
            "wasted": float(np.mean([r["wasted_trips"] for r in g])),
            "abandons": float(np.mean([r["abandons"] for r in g])),
            "travel": float(np.mean([r["travel"] for r in g])),
            "plan_calls": float(np.mean([r["plan_calls"] for r in g])),
            "versions": float(np.mean([r["versions"] for r in g])),
            "bytes_rt": float(np.mean([r["bytes_per_robot_time"] for r in g])),
            "delta_bytes_rt": float(np.mean([r["delta_bytes_per_robot_time"] for r in g])),
            "frames_rt": float(np.mean([r["frames_per_robot_time"] for r in g])),
            "station_frac": float(np.mean([r["station_frac"] for r in g])),
            "plan_ms_p50": float(np.median([r["plan_cpu_ms_p50"] for r in g])),
            "plan_ms_p95": float(np.median([r["plan_cpu_ms_p95"] for r in g])),
            "wall_s": float(np.mean([r["wall_s"] for r in g]))}
    return res


def print_conclusions(tag, c):
    print(f"== {tag}")
    print(f"  variant selected: {c['variant']} ({c['sparc_arm']}); DS rho* = {c['rho_star']}")
    print("  max regret: " + ", ".join(f"{a} {v:.3f}" for a, v in c["max_regret"].items()) +
          f", DS {c['max_regret_DS']:.3f}")
    print(f"  best fixed {c['best_fixed']} {c['best_fixed_max_regret']:.3f}; SPARC margin "
          f"{c['k6_margin_points']:.1f} points; K6 fires: {c['k6_fire']}")
    if "e3a" in c:
        print(f"  E3a: {c['e3a']}")
    print(f"  K5 (rho={c['k5']['rho']}): pooled best fixed {c['k5']['pooled_best_fixed']} ratio to FULL "
          f"{c['k5']['pooled_ratio_to_full'][c['k5']['pooled_best_fixed']]:.3f}; fires: {c['k5']['fire']}")
    for cell, v in c["k5"]["per_family"].items():
        print(f"     {cell}: best fixed {v['best_fixed']} / FULL = {v['ratio_to_full']:.3f} "
              f"[{v['lo']:.3f}, {v['hi']:.3f}]")
    print("  regret by cell:")
    for a, d in c["regret"].items():
        print(f"     {a:10s} " + " ".join(f"{k}={v:.3f}" for k, v in d.items()))
    print(f"     {'DS':10s} " + " ".join(f"{k}={v:.3f}" for k, v in c["regret_DS"].items()))
    print("  SPARC(sel)/CEN-F: " + "; ".join(f"{k} {v['ratio']:.3f} [{v['lo']:.3f}, {v['hi']:.3f}]"
                                           for k, v in c["sparc_vs_cenf"].items()))


def cmd_gates(args) -> None:
    rows = dedup(load(args.rows))
    full = [r for r in rows if r["arm"] == "FULL"]
    arm_rows = [r for r in rows if r["arm"] != "FULL"]
    if args.settings:
        arm_rows = [r for r in arm_rows if r["setting"] in args.settings]
        full = [r for r in full if r["setting"] in args.settings]
    if args.rhos:
        arm_rows = [r for r in arm_rows if any(_rho_eq(r["rho"], x) for x in args.rhos)]
    res = {"n_rows": len(arm_rows) + len(full), "settings": sorted({r["setting"] for r in arm_rows}),
           "success": {}}
    main = conclusions(arm_rows, full, reps=args.reps)
    print_conclusions("tuned leases (evaluation rows)", main)
    res["main"] = main
    # both-succeed sensitivity: drop clusters where any arm failed in that cell? report success rates instead
    sel = arm_rows + full  # the analysed rows only (settings and rho filters applied)
    for a in sorted({r["arm"] for r in sel}):
        for c in sorted({(r["cell"], r["rho"]) for r in sel if r["arm"] == a}, key=lambda c: (c[0], -c[1])):
            s = [r["success"] for r in sel if r["arm"] == a and r["cell"] == c[0] and _rho_eq(r["rho"], c[1])]
            res["success"][f"{a}|{c[0]}|{c[1]:g}"] = [float(np.mean(s)), len(s)]
    bad = {k: v for k, v in res["success"].items() if v[0] < 1}
    print("  success < 100%: " + ("; ".join(f"{k} {v[0]:.3f} (n={v[1]})" for k, v in bad.items()) or "none"))
    res["costs"] = summarize_costs(arm_rows, full)
    if args.grid:
        grid = dedup(load(args.grid))
        gfull = [r for r in grid if r["arm"] == "FULL"] or full
        gar = [r for r in grid if r["arm"] != "FULL"]
        if args.settings:
            gar = [r for r in gar if r["setting"] in args.settings]
        if args.rhos:
            gar = [r for r in gar if any(_rho_eq(r["rho"], x) for x in args.rhos)]
        leases = sorted({(r["lease_g"], r["lease_L"]) for r in gar})
        k7 = {}
        for gl in leases:
            sub = [r for r in gar if (r["lease_g"], r["lease_L"]) == gl]
            c = conclusions(sub, gfull, reps=min(args.reps, 2000))
            k7[f"{gl[0]:g}:{gl[1]:g}"] = c
            print_conclusions(f"common lease g={gl[0]:g} L={gl[1]:g} (tuning rows)", c)
        if args.tuned:
            tuned = json.loads(Path(args.tuned).read_text())["choice"]
            sub = [r for r in gar if f"{r['rho']:g}" in tuned.get(r["arm"], {}) and
                   [r["lease_g"], r["lease_L"]] == tuned[r["arm"]][f"{r['rho']:g}"]]
            c = conclusions(sub, gfull, reps=min(args.reps, 2000))
            k7["tuned"] = c
            print_conclusions("tuned leases on the tuning rows", c)
        keys = ("k5_fire", "k6_fire", "variant", "best_fixed", "rho_star", "sparc_beats_cenf_lo")
        flips = {}
        for name, c in k7.items():
            lo_cells = [k for k in c["sparc_vs_cenf"] if k.endswith(f"@{c['k5']['rho']:g}")]
            flips[name] = {"k5_fire": c["k5"]["fire"], "k6_fire": c["k6_fire"], "variant": c["variant"],
                           "best_fixed": c["best_fixed"], "rho_star": c["rho_star"],
                           "sparc_beats_cenf_lo": all(c["sparc_vs_cenf"][k]["ratio"] < 1 for k in lo_cells)}
        res["k7"] = {"per_lease": flips,
                     "flips": {k: len({json.dumps(v[k]) for v in flips.values()}) > 1 for k in keys}}
        print("K7 flips across the lease grid:", res["k7"]["flips"])
    if args.json:
        Path(args.json).write_text(json.dumps(res, indent=1, default=float))
        print("wrote", args.json)


def _fmt_cell(k: str) -> str:
    cell, rho = k.split("@")
    return f"{cell} ρ={rho}"


def cmd_tables(args) -> None:
    """Markdown tables from a ``gates --json`` file (so the report never transcribes numbers by hand)."""
    g = json.loads(Path(args.json).read_text())
    m = g["main"]
    out = []
    arms = list(m["regret"])
    cells = list(next(iter(m["regret"].values())))
    out.append("| arm | " + " | ".join(_fmt_cell(c) for c in cells) + " | max regret |")
    out.append("|---" * (len(cells) + 2) + "|")
    for a in arms:
        out.append(f"| {a} | " + " | ".join(f"{m['regret'][a][c]:.3f}" for c in cells) +
                   f" | **{m['max_regret'][a]:.3f}** |")
    out.append(f"| DS (ρ*={m['rho_star']:g}) | " + " | ".join(f"{m['regret_DS'][c]:.3f}" for c in cells) +
               f" | **{m['max_regret_DS']:.3f}** |")
    out.append("")
    out.append("DS max regret by ρ*: " + ", ".join(f"ρ*={k}: {v:.3f}" for k, v in
                                                  m["ds_max_regret_by_rho_star"].items()))
    out.append("")
    e = m.get("e3a", {})
    if "diff" in e:
        out.append(f"E3a: MaxReg({m['sparc_arm']}) − min(MaxReg of A ∪ DS) = {e['diff']:+.3f} (one-sided 95% "
                   f"upper bound {e['diff_hi95']:+.3f}; log scale {e['diff_log']:+.3f}, upper {e['diff_log_hi95']:+.3f});"
                   f" comparator {e['comparator']}; {e['n_clusters']} instance clusters; holds: {e['holds']}")
    out.append("")
    out.append("| cell | best fixed arm at the lowest ρ | GM ratio to FULL | 95% CI |")
    out.append("|---|---|---|---|")
    for cell, v in m["k5"]["per_family"].items():
        out.append(f"| {cell} ρ={m['k5']['rho']:g} | {v['best_fixed']} | {v['ratio_to_full']:.3f} | "
                   f"[{v['lo']:.3f}, {v['hi']:.3f}] |")
    out.append("")
    out.append("Pooled over both families (ratio to FULL at the lowest ρ): " +
               ", ".join(f"{a} {v:.3f}" for a, v in m["k5"]["pooled_ratio_to_full"].items()))
    out.append("")
    v2 = m.get("sparc_v2_vs_cenf")
    out.append(f"| cell | {m['sparc_arm']} (selected) / CEN-F | 95% CI |" + (" SPARC (V2) / CEN-F | 95% CI |" if v2 else ""))
    out.append("|---|---|---|" + ("---|---|" if v2 else ""))
    for k, v in m["sparc_vs_cenf"].items():
        line = f"| {_fmt_cell(k)} | {v['ratio']:.3f} | [{v['lo']:.3f}, {v['hi']:.3f}] |"
        if v2 and k in v2:
            line += f" {v2[k]['ratio']:.3f} | [{v2[k]['lo']:.3f}, {v2[k]['hi']:.3f}] |"
        out.append(line)
    out.append("")
    c = g["costs"]
    out.append("| arm | cell | ρ | n | success | GM makespan | wasted trips | abandons | travel | plan calls | "
               "route versions | bytes / robot / t | delta bytes / robot / t | station fraction | plan CPU p50 ms |")
    out.append("|---" * 15 + "|")
    for k in sorted(c, key=lambda k: (k.split("|")[1], -float(k.split("|")[2]), k.split("|")[0])):
        a, cell, rho = k.split("|")
        v = c[k]
        out.append(f"| {a} | {cell} | {rho} | {v['n']} | {v['success']:.3f} | {v['gm']:.1f} | {v['wasted']:.1f} | "
                   f"{v['abandons']:.1f} | {v['travel']:.1f} | {v['plan_calls']:.0f} | {v['versions']:.0f} | "
                   f"{v['bytes_rt']:.0f} | {v['delta_bytes_rt']:.0f} | {v['station_frac']:.2f} | "
                   f"{v['plan_ms_p50']:.1f} |")
    if "k7" in g:
        out.append("")
        out.append("| lease (g:L) | K5 fires | K6 fires | variant | best fixed | DS ρ* | SPARC(sel) < CEN-F at lowest ρ |")
        out.append("|---|---|---|---|---|---|---|")
        for k, v in g["k7"]["per_lease"].items():
            out.append(f"| {k} | {v['k5_fire']} | {v['k6_fire']} | {v['variant']} | {v['best_fixed']} | "
                       f"{v['rho_star']:g} | {v['sparc_beats_cenf_lo']} |")
        out.append("")
        out.append("Flips across the grid: " + ", ".join(f"{k}: {v}" for k, v in g["k7"]["flips"].items()))
    text = "\n".join(out) + "\n"
    if args.out:
        Path(args.out).write_text(text)
    print(text)


def cmd_paired(args) -> None:
    """Secondary factors: GM ratio of rows A over base rows B, paired on (setting, inst, seed, cell, arm, rho) (or
    against FULL with --vs-full), grouped by arm x cell x rho; cluster bootstrap over instances."""
    from cbba_sota.dyn.stats import gm_ratio

    A = dedup(load(args.rows))
    B = dedup(load(args.base))
    if args.where:
        k, v = args.where.split("=")
        A = [r for r in A if str(r.get(k)) == v]
    key = (lambda r: (r["setting"], r["inst"], r["seed"], r["cell"])) if args.vs_full else \
        (lambda r: (r["setting"], r["inst"], r["seed"], r["cell"], r["arm"], r["rho"]))
    base = {key(r): r for r in B if (r["arm"] == "FULL") == bool(args.vs_full)}
    groups: dict = {}
    for r in A:
        if r["arm"] == "FULL":
            continue
        b = base.get(key(r))
        if b is None:
            continue
        groups.setdefault((r["arm"], r["cell"], r["rho"], str(r.get(args.label, ""))), []).append((r, b))
    print(f"{'arm':10s} {'cell':9s} {'rho':>5s} {args.label:>8s} {'n':>4s} {'GM A/B':>7s} {'95% CI':>17s} "
          f"{'succ A':>6s} {'succ B':>6s} {'wasted A-B':>10s} {'bytes A/B':>9s}")
    for k in sorted(groups, key=lambda k: (k[1], -k[2], k[0], k[3])):
        pairs = groups[k]
        lr = np.array([math.log(a[VAL]) - math.log(b[VAL]) for a, b in pairs])
        cl = [(a["setting"], a["inst"]) for a, _ in pairs]
        if len(set(cl)) < 2:
            continue
        g = gm_ratio(lr, cl, strata=[c[0] for c in cl], reps=args.reps)
        print(f"{k[0]:10s} {k[1]:9s} {k[2]:5g} {k[3]:>8s} {len(pairs):4d} {g.ratio:7.3f} [{g.lo:.3f}, {g.hi:.3f}] "
              f"{np.mean([a['success'] for a, _ in pairs]):6.2f} {np.mean([b['success'] for _, b in pairs]):6.2f} "
              f"{np.mean([a['wasted_trips'] - b['wasted_trips'] for a, b in pairs]):10.1f} "
              f"{np.mean([(a['bytes'] + a['beacon_bytes']) / max(b['bytes'] + b['beacon_bytes'], 1) for a, b in pairs]):9.2f}")


def cmd_merge(args) -> None:
    """Analysis input: the T2 rows of --t2 files and the FULL rows of --full files on the same (setting, instance,
    seed, cell) (superseded FULL rows are dropped)."""
    t2 = [r for r in load(args.t2) if r["arm"] != "FULL"]
    keys = {(r["setting"], r["inst"], r["seed"], r["cell"]) for r in t2}  # FULL only where the arms ran (paired)
    full = [r for r in load(args.full) if r["arm"] == "FULL" and (r["setting"], r["inst"], r["seed"], r["cell"]) in keys]
    with Path(args.out).open("w") as f:
        f.writelines(json.dumps(r, default=float) + "\n" for r in t2 + full)
    print(f"{len(t2)} T2 rows (hashes {sorted({r['c3_hash'] for r in t2})}) + {len(full)} FULL rows "
          f"(hashes {sorted({r['c3_hash'] for r in full})}) -> {args.out}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("design")
    d.add_argument("--rows", nargs="+", required=True)
    d.set_defaults(func=cmd_design)
    t = sub.add_parser("tune")
    t.add_argument("--rows", nargs="+", required=True)
    t.add_argument("--out", required=True)
    t.set_defaults(func=cmd_tune)
    g = sub.add_parser("gates")
    g.add_argument("--rows", nargs="+", required=True)
    g.add_argument("--grid", nargs="*", default=None)
    g.add_argument("--tuned", default=None)
    g.add_argument("--settings", nargs="*", default=None)
    g.add_argument("--rhos", nargs="*", type=float, default=None)
    g.add_argument("--reps", type=int, default=10000)
    g.add_argument("--json", default=None)
    g.set_defaults(func=cmd_gates)
    mg = sub.add_parser("merge")
    mg.add_argument("--t2", nargs="+", required=True)
    mg.add_argument("--full", nargs="+", required=True)
    mg.add_argument("--out", required=True)
    mg.set_defaults(func=cmd_merge)
    pr = sub.add_parser("paired")
    pr.add_argument("--rows", nargs="+", required=True)
    pr.add_argument("--base", nargs="+", required=True)
    pr.add_argument("--vs-full", action="store_true")
    pr.add_argument("--label", default="beacon")
    pr.add_argument("--where", default=None)
    pr.add_argument("--reps", type=int, default=10000)
    pr.set_defaults(func=cmd_paired)
    tb = sub.add_parser("tables")
    tb.add_argument("--json", required=True)
    tb.add_argument("--out", default=None)
    tb.set_defaults(func=cmd_tables)
    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
