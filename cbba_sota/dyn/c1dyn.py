"""C1-dyn pilot of Track D (docs/trackD-spec.md Section 8 days 2-3, Section 7 collapse checks T1-T3, gates K1/K2).

Two parts, both used by ``scripts/trackD_c1dyn.py``:

1. **B4 budget calibration** (spec 3.6: "Light = the CPU-ms equivalent of 300 SPARC iterations"). ``ProbeSPARC``
   is SPARC-Light whose every re-plan additionally *times* (without adopting) the rolling constructor B4 on the same
   belief state, at ``probe_restarts`` restart counts. The adopted plans are SPARC's own; the probe calls run after
   the adopted plan and every ``RHPlanner.plan`` call reseeds the kernel RNG, so the episode is identical to plain
   SPARC (tested). The per-event CPU ratios on identical states give the equal-CPU restart count per setting:
   ``r* = (mean SPARC CPU - a) / b`` with ``a + b r`` the mean B4 CPU at ``r`` restarts (ratio of totals over all
   probed events, i.e. equal CPU per episode). Ratios of process CPU on the same states in the same process are far
   less load-sensitive than absolute times, but the host is shared: the quiet-core calibration (K8) may move r*.

2. **Gate analysis** of runner rows (``scripts/trackD_run.py``): method labels (tier, B4 budget, coalition rule),
   paired geometric-mean ratios with the cluster bootstrap of ``cbba_sota.dyn.stats`` (clusters = instances,
   strata = settings, pairs = (setting, instance, cell, CRN seed)), and the decision rules of T1/K1 and K2.

Pinned kernel: the planners come from ``cbba_sota.dyn.planner`` / ``rh_kernels`` (ALNS v2 at commit 4cf5e04).
"""
from __future__ import annotations

import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from cbba_sota.dyn.planner import Problem, RHPlanner
from cbba_sota.dyn.sparc import SPARC, RollingPolicy, SPARCConfig

__all__ = [
    "B4_LEVELS",
    "PAIR_KEYS",
    "Hybrid",
    "ProbeSPARC",
    "b4_levels",
    "family_balanced",
    "hybrid",
    "k1_decision",
    "k2_decision",
    "labels",
    "restarts_equal_cpu",
]

PAIR_KEYS = ("setting", "instance", "cell", "seed")
CLUSTER_KEYS = ("setting", "instance")
B4_LEVELS = {"B4-eq": 1, "B4-2eq": 2, "B4-4eq": 4}  # multiples of the per-setting equal-CPU restart count


# ---------------------------------------------------------------------------------------------------------------
# 1. calibration probe
# ---------------------------------------------------------------------------------------------------------------
class ProbeSPARC(SPARC):
    """SPARC-Light that also times B4 (cold regret floor + ``r`` randomized constructions) on every belief it plans.

    ``probes`` holds one record per re-plan: the adopted SPARC plan's CPU (``sparc_s``), the B4 CPU at each probed
    restart count (``b4_s[r]``), the number of open tasks and whether it was the first (t = 0) plan."""

    name = "SPARC-probe"

    def __init__(self, iters: int = 300, probe_restarts: Sequence[int] = (0, 24), noise: float = 0.2):
        super().__init__(SPARCConfig(iters=iters))
        self.probe_restarts = tuple(int(r) for r in probe_restarts)
        self.noise = float(noise)
        self.b4 = RHPlanner()
        self.probes: list[dict] = []

    def solve(self, state, scope, seed, first):
        plan = super().solve(state, scope, seed, first)
        rec = {"first": bool(first), "n_open": int(plan.n_open), "sparc_s": float(plan.cpu_s), "b4_s": {}}
        for r in self.probe_restarts:
            p = self.b4.plan(state, None, scope, seed, iters=0, warm=False, restarts=r, restart_noise=self.noise)
            rec["b4_s"][r] = float(p.cpu_s)
        c0 = time.process_time()  # B4 rebuilds the kernel Problem for every restart: time one build (mean of 3)
        for _ in range(3):
            Problem(state, scope, False)
        rec["problem_s"] = (time.process_time() - c0) / 3.0
        self.probes.append(rec)
        return plan


def restarts_equal_cpu(sparc_s: Sequence[float], b4_s: Mapping[int, Sequence[float]],
                       overhead_s: Sequence[float] | None = None) -> dict:
    """Equal-CPU restart count from paired per-event timings: fit mean B4 CPU = a + b r over the probed restart
    counts (least squares on the per-r means), then r* = (mean SPARC CPU - a) / b. Also the median-based variant.
    ``overhead_s`` (per event): a per-restart cost not charged to B4 (its per-restart ``Problem`` rebuild), removed
    as ``b4_s[r] - r * overhead`` before the fit ("construction-only" charging)."""
    rs = np.array(sorted(b4_s), float)
    if rs.size < 2:
        raise ValueError("need B4 timings at two or more restart counts")
    if overhead_s is not None:
        ov = np.asarray(overhead_s, float)
        b4_s = {int(r): np.asarray(b4_s[int(r)], float) - r * ov for r in rs}
    means = np.array([np.mean(b4_s[int(r)]) for r in rs])
    meds = np.array([np.median(b4_s[int(r)]) for r in rs])
    b, a = np.polyfit(rs, means, 1)
    bm, am = np.polyfit(rs, meds, 1)
    s_mean, s_med = float(np.mean(sparc_s)), float(np.median(sparc_s))
    return {"a_ms": 1e3 * a, "b_ms": 1e3 * b, "sparc_mean_ms": 1e3 * s_mean, "sparc_median_ms": 1e3 * s_med,
            "r_star": float((s_mean - a) / b), "r_star_median": float((s_med - am) / bm), "n_events": len(sparc_s)}


class Hybrid(RollingPolicy):
    """Diagnostic arm: the first plan (t = 0) by ``first``, every later re-plan by ``then`` (the incumbent is handed
    over). Separates the quality of the initial plan from the quality of the in-loop repair (the pilots' RH-CA)."""

    def __init__(self, first: RollingPolicy, then: RollingPolicy, name: str = "hybrid"):
        super().__init__(then.triggers, then.seed_base)
        self.first_policy, self.then_policy, self.name = first, then, name

    def solve(self, state, scope, seed, first):
        pol = self.first_policy if first else self.then_policy
        pol.incumbent = self.incumbent
        return pol.solve(state, scope, seed, first)


def hybrid(env, realization, tier: str, seed: int, first: str = "SPARC", then: str = "B4", **params):
    """``scripts/trackD_run.py`` plug-in (``ctrl:cbba_sota.dyn.c1dyn:hybrid?first=SPARC&then=B4&restarts=47``):
    a ``PlanController`` whose t = 0 plan is planner ``first`` and whose re-plans are planner ``then`` (names of
    ``methods.PLANNERS`` at ``tier``; ``restarts`` / ``iters`` go to whichever of the two takes them)."""
    from cbba_sota.dyn.controller import PlanController
    from cbba_sota.dyn.methods import make_policy

    def build(name):
        keep = {"B4": ("restarts",)}.get(name, ("iters",))
        return make_policy(name, tier, **{k: v for k, v in params.items() if k in keep})

    pol = Hybrid(build(first), build(then), name=f"{first}>{then}")
    return PlanController(pol, env.nominal_instance(), realization.kappa())


# ---------------------------------------------------------------------------------------------------------------
# 2. labels and gates
# ---------------------------------------------------------------------------------------------------------------
def _parse(method: str) -> tuple[str, dict]:
    name, _, query = method.partition("?")
    params = {}
    for kv in filter(None, query.split("&")):
        k, _, v = kv.partition("=")
        params[k] = v
    return name, params


def b4_levels(budget: Mapping) -> dict[str, dict[str, int]]:
    """Per setting, the B4 restart count of every labelled budget level, from ``b4_budget.json``
    (``scripts/trackD_c1dyn.py calibrate``): B4-eq / -2eq / -4eq are multiples of the construction-only equal-CPU
    count, B4-impl charges the implementation as it runs."""
    out = {}
    for s, fit in budget["settings"].items():
        eq = int(fit["restarts"])
        out[s] = {**{lab: mult * eq for lab, mult in B4_LEVELS.items()}, "B4-impl": int(fit["restarts_impl"])}
    return out


def labels(row: Mapping, b4: Mapping[str, Mapping[str, int]]) -> list[str]:
    """Analysis labels of a runner row (a row may carry several, e.g. a B4 restart count that is both 'B4-impl' and
    'B4-r12'). SPARC-L / SPARC-H by tier; B4 as 'B4-r<k>' plus every level of ``b4[setting]`` (``b4_levels``) with
    that count; policies get '-arr' under the arrival-order coalition rule; other planners at the heavy tier '-H'."""
    name, params = _parse(row["method"])
    tier = row.get("tier")
    coal = (row.get("options") or {}).get("coalition", "auto")
    t0 = "+t0" if ("iters0" in params or "restarts0" in params) else ""  # a separate budget for the t = 0 plan
    if name == "SPARC":
        return [("SPARC-L" if tier == "light" else "SPARC-H") + t0]
    if name == "B4":
        r = int(params.get("restarts", 12))
        return [f"B4-r{r}{t0}"] + [lab + t0 for lab, k in (b4.get(row["setting"]) or {}).items() if k == r]
    if name in ("RL(g.)", "RL(s.1)", "greedy"):
        return [name + ("" if coal in ("auto", "decision") else "-arr")]
    if name == "ctrl:cbba_sota.dyn.c1dyn:hybrid":
        first, then = params.get("first", "SPARC"), params.get("then", "B4")
        r = params.get("restarts")
        lev = [lab for lab, k in (b4.get(row["setting"]) or {}).items() if r is not None and k == int(r)]
        tag = lev[0] if lev else (f"B4-r{r}" if r is not None else "B4")
        return [f"{first}>{then}".replace("B4", tag).replace("SPARC", "SPARC-L")]
    return [row["method"] if tier in ("light", "native") else f"{row['method']}-H"]


def family_balanced(d: dict, family_of: Mapping[str, str]) -> tuple[np.ndarray, list]:
    """Per-(instance, family) mean log r (families weighted equally inside an instance, whatever their number of
    cells): returns values and instance clusters for ``stats.gm_ratio`` (whose cluster mean then averages families).
    ``d`` is a ``stats.pair_rows`` result on ``PAIR_KEYS``."""
    kept = [k for k, kp in zip(d["keys"], d["keep"]) if kp]
    acc: dict = {}
    for k, v in zip(kept, d["log_r"]):
        setting, inst, cell, _seed = k
        acc.setdefault(((setting, inst), family_of[cell]), []).append(float(v))
    vals = np.array([np.mean(v) for v in acc.values()])
    clusters = [ci for ci, _ in acc]
    return vals, clusters


@dataclass(frozen=True)
class GateResult:
    name: str
    passed: bool  # the rule's condition holds (for K2: 'no separation' holds, i.e. the kill fires)
    detail: str


def k1_decision(pooled, per_setting: Mapping[str, object]) -> GateResult:
    """Spec 7 T1 / K1: SPARC vs B4 at the same Light budget, F1-F3 pooled. Pass = pooled <= 0.97, pooled cluster
    CI upper bound < 1, and ratio < 1 on >= 3 of 4 settings. K1 fires (kill) when T1 fails."""
    n_lt1 = sum(1 for g in per_setting.values() if g.ratio < 1.0)
    ok = pooled.ratio <= 0.97 and pooled.hi < 1.0 and n_lt1 >= 3
    detail = (f"pooled {pooled.ratio:.4f} [{pooled.lo:.4f}, {pooled.hi:.4f}] (need <= 0.97 and hi < 1); "
              f"settings < 1: {n_lt1}/{len(per_setting)} (need >= 3)")
    return GateResult("T1", bool(ok), detail)


def k2_decision(ins_f3, b4_f1) -> GateResult:
    """Spec 8 K2 'no dynamic separation': SPARC / insertion-only > 0.98 on F3 AND SPARC / B4 > 0.97 on F1 (point
    estimates). ``passed`` = the kill condition holds."""
    fire = ins_f3.ratio > 0.98 and b4_f1.ratio > 0.97
    detail = (f"SPARC/ins-only F3 {ins_f3.ratio:.4f} (> 0.98? {ins_f3.ratio > 0.98}); "
              f"SPARC/B4 F1 {b4_f1.ratio:.4f} (> 0.97? {b4_f1.ratio > 0.97})")
    return GateResult("K2", bool(fire), detail)


def episode_summary(rows: Iterable[Mapping]) -> dict:
    """Descriptive per-group numbers: success, mean / GM makespan (successful episodes), re-plans, route versions,
    CPU per structural event (median of per-episode p50 and p95, pooled p50/p95/max), travel, wasted trips."""
    rows = list(rows)
    ok = [r for r in rows if r.get("success")]
    ms = np.array([float(r["makespan"]) for r in ok]) if ok else np.array([])
    cpu_all = np.array([x for r in rows for x in (r.get("cpu_ms_event") or [])], float)
    p50 = [r["cpu_ms_p50"] for r in rows if r.get("cpu_ms_p50") is not None]
    p95 = [r["cpu_ms_p95"] for r in rows if r.get("cpu_ms_p95") is not None]

    def mean(key):
        v = [float(r[key]) for r in rows if r.get(key) is not None]
        return float(np.mean(v)) if v else float("nan")

    return {"n": len(rows), "success": len(ok), "success_rate": len(ok) / max(len(rows), 1),
            "mean_makespan": float(ms.mean()) if ms.size else float("nan"),
            "gm_makespan": float(np.exp(np.log(ms).mean())) if ms.size else float("nan"),
            "replans": mean("replans"), "route_versions": mean("route_versions"),
            "structural_events": mean("structural_events"), "travel": mean("travel"),
            "wasted_trips": mean("wasted_trips"), "abandons": mean("abandons"), "restarts": mean("restarts"),
            "cpu_ep_p50_med": float(np.median(p50)) if p50 else float("nan"),
            "cpu_ep_p95_med": float(np.median(p95)) if p95 else float("nan"),
            "cpu_ep_p95_max": float(np.max(p95)) if p95 else float("nan"),
            "cpu_p50": float(np.percentile(cpu_all, 50)) if cpu_all.size else float("nan"),
            "cpu_p95": float(np.percentile(cpu_all, 95)) if cpu_all.size else float("nan"),
            "cpu_max": float(cpu_all.max()) if cpu_all.size else float("nan"),
            "cpu_episode_s": mean("cpu_method_s")}
