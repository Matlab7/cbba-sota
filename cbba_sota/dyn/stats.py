"""Track D statistics (docs/trackD-spec.md Section 9.1): paired geometric-mean ratios with a cluster bootstrap over
instances (CRN seeds and families nested inside an instance), Holm, TOST, exact paired success tests and the
worst-case-regret test of E3a.

Conventions
- An *observation* is one paired episode: the same (setting, instance, family, CRN seed) run by method a and method b.
  Its value is ``log r = log(makespan_a / makespan_b)``; below 0 favours ``a``.
- A *cluster* is an instance, identified by any hashable label (e.g. ``(setting, instance)``). The unit of analysis is
  the cluster mean of ``log r`` over its observations (seeds, and families when pooling). Every cluster weighs the
  same, whatever its number of observations.
- The cluster bootstrap resamples clusters with replacement (optionally within strata, e.g. settings). With
  ``two_stage=True`` it also resamples the observations inside each drawn cluster.
- Defaults follow the spec: 10,000 resamples, generator seed 0, 95% percentile intervals, one-sided paired t-test of
  mean cluster log r < 0.

Only numpy and scipy; ``holm`` is re-exported from ``cbba_sota.stats`` (read-only reuse).
"""
from __future__ import annotations

from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass

import numpy as np
from scipy import stats as st

from cbba_sota.stats import holm

__all__ = [
    "MAX_TIME",
    "TOST_MARGIN_2PCT",
    "Comparison",
    "GMRatio",
    "MaxRegret",
    "SuccessTest",
    "Tost",
    "clear_margin",
    "cluster_bootstrap",
    "cluster_ids",
    "cluster_means",
    "compare",
    "exact_mcnemar",
    "gm_ratio",
    "hodges_lehmann",
    "holm",
    "holm_family",
    "holm_reject",
    "max_regret_test",
    "pair_rows",
    "success_test",
    "tost",
]

MAX_TIME = 200.0
TOST_MARGIN_2PCT = float(np.log(1.02))  # |mean log r| < 0.0198 (spec 9.3)


# ---------------------------------------------------------------------------------------------------------------
# clusters
# ---------------------------------------------------------------------------------------------------------------
def cluster_ids(clusters: Sequence[Hashable]) -> tuple[np.ndarray, list]:
    """Integer codes 0..k-1 in order of first appearance, and the labels."""
    labels: dict = {}
    codes = np.empty(len(clusters), dtype=np.int64)
    for i, c in enumerate(clusters):
        codes[i] = labels.setdefault(c, len(labels))
    return codes, list(labels)


def cluster_means(values, clusters: Sequence[Hashable]) -> tuple[np.ndarray, list]:
    """Per-cluster mean of ``values`` (clusters in order of first appearance) and the cluster labels."""
    v = np.asarray(values, float)
    if v.ndim != 1 or v.size != len(clusters):
        raise ValueError("values and clusters must be 1-D and of equal length")
    codes, labels = cluster_ids(clusters)
    k = len(labels)
    return np.bincount(codes, v, k) / np.bincount(codes, minlength=k), labels


def _resample_index(codes: np.ndarray, k: int, strata_codes: np.ndarray | None, rng: np.random.Generator,
                    reps: int) -> np.ndarray:
    """``reps x k`` cluster draws; within strata if given (each stratum keeps its cluster count)."""
    if strata_codes is None:
        return rng.integers(0, k, size=(reps, k))
    out = np.empty((reps, k), dtype=np.int64)
    col = 0
    for s in np.unique(strata_codes):
        members = np.flatnonzero(strata_codes == s)
        out[:, col:col + members.size] = members[rng.integers(0, members.size, size=(reps, members.size))]
        col += members.size
    return out


def _strata_of_clusters(codes: np.ndarray, k: int, strata: Sequence[Hashable] | None) -> np.ndarray | None:
    if strata is None:
        return None
    s_codes, _ = cluster_ids(strata)
    per_cluster = np.full(k, -1)
    for c, s in zip(codes, s_codes):
        if per_cluster[c] not in (-1, s):
            raise ValueError("a cluster spans several strata")
        per_cluster[c] = s
    return per_cluster


def cluster_bootstrap(values, clusters: Sequence[Hashable], *, stat: Callable[[np.ndarray], float] | None = None,
                      reps: int = 10_000, seed: int = 0, strata: Sequence[Hashable] | None = None,
                      two_stage: bool = False) -> np.ndarray:
    """Bootstrap distribution (length ``reps``) of ``stat`` applied to the vector of cluster means.

    ``stat`` defaults to the mean. Clusters are resampled with replacement (within ``strata``, given per
    observation, if set). ``two_stage`` also resamples observations within each drawn cluster.
    """
    v = np.asarray(values, float)
    codes, labels = cluster_ids(clusters)
    k = len(labels)
    if k < 2:
        raise ValueError("need at least two clusters")
    rng = np.random.default_rng(seed)
    draws = _resample_index(codes, k, _strata_of_clusters(codes, k, strata), rng, reps)
    if not two_stage:
        means = np.bincount(codes, v, k) / np.bincount(codes, minlength=k)
        boot = means[draws]
    else:
        order = np.argsort(codes, kind="stable")
        sizes = np.bincount(codes, minlength=k)
        starts = np.concatenate([[0], np.cumsum(sizes)[:-1]])
        sorted_v = v[order]
        width = int(sizes.max())
        boot = np.empty((reps, k))
        for lo in range(0, reps, 500):  # chunks bound the memory (reps x k x width)
            d = draws[lo:lo + 500]
            n_draw = sizes[d]
            u = rng.random((d.shape[0], k, width))
            pick = np.minimum((u * n_draw[..., None]).astype(np.int64), n_draw[..., None] - 1)
            vals = sorted_v[starts[d][..., None] + pick]
            valid = np.arange(width)[None, None, :] < n_draw[..., None]
            boot[lo:lo + 500] = np.where(valid, vals, 0.0).sum(-1) / n_draw
    if stat is None:
        return boot.mean(axis=1)
    return np.array([stat(row) for row in boot])


def hodges_lehmann(x) -> float:
    """One-sample Hodges-Lehmann location estimate: median of the Walsh averages (x_i + x_j) / 2, i <= j."""
    x = np.asarray(x, float)
    i, j = np.triu_indices(x.size)
    return float(np.median((x[i] + x[j]) / 2.0))


# ---------------------------------------------------------------------------------------------------------------
# geometric-mean paired ratio (E1, E2 and every descriptive ratio)
# ---------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class GMRatio:
    ratio: float  # exp(mean over clusters of the cluster-mean log r); < 1 favours a
    lo: float  # cluster-bootstrap percentile CI of the ratio
    hi: float
    median_ratio: float  # exp(median of cluster means)
    hl_ratio: float  # exp(Hodges-Lehmann of cluster means)
    p_less: float  # one-sided paired t-test on cluster means, H1: mean log r < 0
    p_two: float  # two-sided version
    n_clusters: int
    n_obs: int
    sd_log: float  # SD of the cluster means of log r
    wins: int  # clusters with mean log r < 0 (a better)
    losses: int  # clusters with mean log r > 0
    ties: int
    obs_wins: int  # observation-level counts
    obs_losses: int

    def as_dict(self) -> dict:
        return asdict(self)


def gm_ratio(log_r, clusters: Sequence[Hashable], *, reps: int = 10_000, seed: int = 0, alpha: float = 0.05,
             strata: Sequence[Hashable] | None = None, two_stage: bool = False, tie_tol: float = 1e-12) -> GMRatio:
    """Geometric-mean paired ratio with a cluster-bootstrap CI (spec 9.1). ``log_r`` per observation."""
    v = np.asarray(log_r, float)
    if not np.all(np.isfinite(v)):
        raise ValueError("log ratios must be finite (drop or impute failures first, see pair_rows)")
    means, _ = cluster_means(v, clusters)
    k = means.size
    boot = cluster_bootstrap(v, clusters, reps=reps, seed=seed, strata=strata, two_stage=two_stage)
    lo, hi = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
    if k >= 2 and np.ptp(means) > 0:
        p_less = float(st.ttest_1samp(means, 0.0, alternative="less").pvalue)
        p_two = float(st.ttest_1samp(means, 0.0).pvalue)
    else:  # degenerate: every cluster identical
        m = float(means.mean())
        p_less = 0.0 if m < 0 else 1.0
        p_two = 0.0 if m != 0 else 1.0
    return GMRatio(
        ratio=float(np.exp(means.mean())), lo=float(np.exp(lo)), hi=float(np.exp(hi)),
        median_ratio=float(np.exp(np.median(means))), hl_ratio=float(np.exp(hodges_lehmann(means))),
        p_less=p_less, p_two=p_two, n_clusters=int(k), n_obs=int(v.size),
        sd_log=float(means.std(ddof=1)) if k > 1 else 0.0,
        wins=int((means < -tie_tol).sum()), losses=int((means > tie_tol).sum()),
        ties=int((np.abs(means) <= tie_tol).sum()),
        obs_wins=int((v < -tie_tol).sum()), obs_losses=int((v > tie_tol).sum()))


def clear_margin(pooled: GMRatio, per_setting: Mapping[str, GMRatio] | Iterable[float], *, bound: float = 0.97,
                 min_settings: int = 3, require_ci: bool = False) -> bool:
    """Spec 9.2 'clear margin': pooled ratio <= ``bound`` and ratio < 1 on at least ``min_settings`` settings.
    T1 additionally needs the pooled CI upper bound < 1 (``require_ci=True``)."""
    vals = per_setting.values() if isinstance(per_setting, Mapping) else per_setting
    ratios = [v.ratio if isinstance(v, GMRatio) else float(v) for v in vals]
    ok = pooled.ratio <= bound and sum(r < 1.0 for r in ratios) >= min_settings
    return bool(ok and (pooled.hi < 1.0 if require_ci else True))


# ---------------------------------------------------------------------------------------------------------------
# equivalence (E2a, E2b, F0 control)
# ---------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Tost:
    mean_log: float
    margin_log: float
    p_lower: float  # H0: mean <= -margin
    p_upper: float  # H0: mean >= +margin
    p: float  # max of the two; equivalence at level alpha iff p < alpha
    ci90_lo: float  # cluster-bootstrap 90% CI of the mean log r (the bootstrap TOST)
    ci90_hi: float
    equivalent_t: bool
    equivalent_boot: bool
    n_clusters: int


def tost(log_r, clusters: Sequence[Hashable], margin_log: float = TOST_MARGIN_2PCT, *, alpha: float = 0.05,
         reps: int = 10_000, seed: int = 0, strata: Sequence[Hashable] | None = None) -> Tost:
    """Two one-sided paired t-tests on the cluster means of log r against +-``margin_log``, plus the bootstrap
    version (the (1 - 2 alpha) cluster-bootstrap CI lies inside the margin)."""
    means, _ = cluster_means(np.asarray(log_r, float), clusters)
    n = means.size
    m = float(means.mean())
    se = float(means.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    if se == 0.0:
        p_lo = 0.0 if m > -margin_log else 1.0
        p_hi = 0.0 if m < margin_log else 1.0
    else:
        p_lo = float(st.t.sf((m + margin_log) / se, n - 1))
        p_hi = float(st.t.cdf((m - margin_log) / se, n - 1))
    boot = cluster_bootstrap(log_r, clusters, reps=reps, seed=seed, strata=strata)
    lo, hi = np.quantile(boot, [alpha, 1 - alpha])
    p = max(p_lo, p_hi)
    return Tost(m, float(margin_log), p_lo, p_hi, p, float(lo), float(hi), bool(p < alpha),
                bool(-margin_log < lo and hi < margin_log), int(n))


def holm_reject(pvalues, alpha: float = 0.05) -> np.ndarray:
    """Boolean rejections of the Holm step-down procedure at family-wise level ``alpha``."""
    return holm(pvalues) <= alpha


def holm_family(pvalues: Mapping[str, float], alpha: float = 0.05) -> dict[str, dict]:
    """Holm over one named family of one-sided p-values (e.g. spec 9.1: E1 competitors + E2a + E2b + E3a).
    Returns name -> {p, p_holm, reject}."""
    names = list(pvalues)
    adj = holm([pvalues[n] for n in names])
    return {n: {"p": float(pvalues[n]), "p_holm": float(a), "reject": bool(a <= alpha)} for n, a in zip(names, adj)}


# ---------------------------------------------------------------------------------------------------------------
# success (analysed separately, exact paired tests)
# ---------------------------------------------------------------------------------------------------------------
def exact_mcnemar(success_a, success_b) -> dict:
    """Exact McNemar test on paired binary outcomes: binomial test of the discordant pairs.

    ``b10`` = a succeeded and b failed, ``b01`` = the reverse. Returns the two-sided p and the one-sided p for
    'a succeeds more often than b'."""
    a = np.asarray(success_a, bool)
    b = np.asarray(success_b, bool)
    b10 = int((a & ~b).sum())
    b01 = int((~a & b).sum())
    n = b10 + b01
    if n == 0:
        return {"b10": 0, "b01": 0, "p_two": 1.0, "p_a_better": 1.0, "p_a_worse": 1.0}
    return {"b10": b10, "b01": b01,
            "p_two": float(st.binomtest(b10, n, 0.5).pvalue),
            "p_a_better": float(st.binomtest(b10, n, 0.5, alternative="greater").pvalue),
            "p_a_worse": float(st.binomtest(b10, n, 0.5, alternative="less").pvalue)}


@dataclass(frozen=True)
class SuccessTest:
    rate_a: float
    rate_b: float
    diff: float  # rate_a - rate_b (mean over clusters of the cluster success-rate difference)
    diff_lo: float  # cluster-bootstrap percentile CI of diff
    diff_hi: float
    mcnemar_obs: dict  # exact McNemar over observations (ignores clustering)
    sign_clusters: dict  # exact sign test over clusters whose success rates differ (respects clustering)
    noninferior: bool | None  # lower bound of the (1 - 2 alpha) CI > -margin (None if no margin given)
    n_clusters: int
    n_obs: int


def success_test(success_a, success_b, clusters: Sequence[Hashable], *, margin: float | None = None,
                 alpha: float = 0.05, reps: int = 10_000, seed: int = 0) -> SuccessTest:
    """Paired success analysis: exact McNemar (observations), exact sign test (clusters), bootstrap CI of the
    success-rate difference and, if ``margin`` is set, non-inferiority of a (e.g. 0.02 = 2 pp, spec 9.4)."""
    a = np.asarray(success_a, bool)
    b = np.asarray(success_b, bool)
    d = a.astype(float) - b.astype(float)
    dm, _ = cluster_means(d, clusters)
    pos, neg = int((dm > 0).sum()), int((dm < 0).sum())
    if pos + neg:
        sign = {"pos": pos, "neg": neg, "p_two": float(st.binomtest(pos, pos + neg, 0.5).pvalue),
                "p_a_better": float(st.binomtest(pos, pos + neg, 0.5, alternative="greater").pvalue)}
    else:
        sign = {"pos": 0, "neg": 0, "p_two": 1.0, "p_a_better": 1.0}
    boot = cluster_bootstrap(d, clusters, reps=reps, seed=seed)
    lo, hi = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
    ni = None
    if margin is not None:
        ni = bool(np.quantile(boot, alpha) > -margin)
    ra, _ = cluster_means(a.astype(float), clusters)
    rb, _ = cluster_means(b.astype(float), clusters)
    return SuccessTest(float(ra.mean()), float(rb.mean()), float(dm.mean()), float(lo), float(hi),
                       exact_mcnemar(a, b), sign, ni, int(dm.size), int(a.size))


# ---------------------------------------------------------------------------------------------------------------
# rows -> paired observations
# ---------------------------------------------------------------------------------------------------------------
def pair_rows(rows: Iterable[Mapping], method_a: str, method_b: str, *, value: str = "makespan",
              pair_keys: Sequence[str] = ("setting", "inst", "family", "seed"),
              cluster_keys: Sequence[str] = ("setting", "inst"), method_key: str = "method",
              success_key: str = "success", failure: str = "drop", max_time: float = MAX_TIME,
              where: Callable[[Mapping], bool] | None = None) -> dict:
    """Match the rows of two methods on ``pair_keys`` and return paired arrays.

    ``failure='drop'`` (primary, spec 9.1) keeps only pairs where both succeeded; ``'impute'`` scores a failed
    episode as ``max_time`` (sensitivity). Duplicate rows for one (method, key) raise. Returns a dict with
    ``log_r``, ``clusters``, ``strata`` (the first cluster key), ``keys``, ``success_a``, ``success_b`` (over all
    matched pairs) and ``keep`` (mask of pairs used in ``log_r``)."""
    if failure not in ("drop", "impute"):
        raise ValueError(failure)
    side: dict[str, dict] = {method_a: {}, method_b: {}}
    for r in rows:
        m = r.get(method_key)
        if m not in side or (where is not None and not where(r)):
            continue
        k = tuple(r[x] for x in pair_keys)
        if k in side[m]:
            raise ValueError(f"duplicate row for {m} at {k}")
        side[m][k] = r
    keys = sorted(set(side[method_a]) & set(side[method_b]), key=repr)
    ra = [side[method_a][k] for k in keys]
    rb = [side[method_b][k] for k in keys]
    sa = np.array([bool(r[success_key]) for r in ra], bool)
    sb = np.array([bool(r[success_key]) for r in rb], bool)
    # a failed episode may carry no value (``scripts/trackD_run.py`` rows: makespan None); drop / impute handle it
    va = np.array([np.nan if r[value] is None else float(r[value]) for r in ra])
    vb = np.array([np.nan if r[value] is None else float(r[value]) for r in rb])
    if failure == "drop":
        keep = sa & sb
    else:
        keep = np.ones(len(keys), bool)
        va = np.where(sa, va, max_time)
        vb = np.where(sb, vb, max_time)
    idx = [tuple(k[pair_keys.index(c)] for c in cluster_keys) for k in keys]
    clusters = [c for c, kp in zip(idx, keep) if kp]
    return {"log_r": np.log(va[keep] / vb[keep]), "clusters": clusters, "strata": [c[0] for c in clusters],
            "keys": keys, "success_a": sa, "success_b": sb, "all_clusters": idx, "keep": keep,
            "unmatched_a": len(side[method_a]) - len(keys), "unmatched_b": len(side[method_b]) - len(keys)}


# ---------------------------------------------------------------------------------------------------------------
# E3a: worst-case regret over a grid of cells (connectivity levels x families)
# ---------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class MaxRegret:
    max_regret: dict[str, float]  # arm -> max over cells of GM(arm) / min over the fixed architectures of GM
    regret: dict[str, dict]  # arm -> cell -> regret
    test_arm: str
    comparator: str  # the comparator with the lowest point MaxReg
    diff: float  # MaxReg(test) - min over comparators MaxReg (ratio points); < 0 favours the test arm
    diff_hi: float  # one-sided (1 - alpha) bootstrap upper bound of diff; the hypothesis holds iff < 0
    diff_log: float  # same on the log scale
    diff_log_hi: float
    n_clusters: int


def _cell_gm_logs(logs: np.ndarray, present: np.ndarray, draws: np.ndarray | None) -> np.ndarray:
    """Mean over clusters of per-(arm, cell, cluster) mean log makespans; arrays are arm x cell x cluster."""
    if draws is None:
        with np.errstate(invalid="ignore"):
            return np.nansum(np.where(present, logs, 0.0), -1) / present.sum(-1)
    lg = logs[:, :, draws]  # arm x cell x reps x k
    pr = present[:, :, draws]
    return np.where(pr, lg, 0.0).sum(-1) / np.maximum(pr.sum(-1), 1)  # arm x cell x reps


def max_regret_test(rows: Iterable[Mapping], *, arms_fixed: Sequence[str], test_arm: str,
                    comparators: Sequence[str], cell_keys: Sequence[str] = ("family", "rho"),
                    cluster_keys: Sequence[str] = ("setting", "inst"), method_key: str = "method",
                    value: str = "makespan", reps: int = 10_000, seed: int = 0, alpha: float = 0.05) -> MaxRegret:
    """Spec 9.4 E3a. regret_arm(cell) = GM_arm(cell) / min over ``arms_fixed`` of GM(cell); MaxReg = max over cells.
    Tests MaxReg(test_arm) < min over ``comparators`` (the fixed architectures and DS) of MaxReg with one
    simultaneous cluster bootstrap (the same resampled instances for every arm and cell).

    Rows are episodes; within an (arm, cell, cluster) the log makespans are averaged over seeds first. Every
    (arm, cell) must cover the same clusters (a paired design); missing combinations raise."""
    arms = list(dict.fromkeys([*arms_fixed, test_arm, *comparators]))
    acc: dict = {}
    cells, clus = {}, {}
    for r in rows:
        a = r.get(method_key)
        if a not in arms:
            continue
        c = tuple(r[x] for x in cell_keys)
        k = tuple(r[x] for x in cluster_keys)
        cells.setdefault(c, len(cells))
        clus.setdefault(k, len(clus))
        acc.setdefault((a, c, k), []).append(np.log(float(r[value])))
    A, C, K = len(arms), len(cells), len(clus)
    logs = np.full((A, C, K), np.nan)
    for (a, c, k), v in acc.items():
        logs[arms.index(a), cells[c], clus[k]] = float(np.mean(v))
    present = np.isfinite(logs)
    if not present.all():
        missing = int((~present).sum())
        raise ValueError(f"{missing} (arm, cell, cluster) combinations have no rows; the design must be paired")
    fixed = [arms.index(a) for a in arms_fixed]

    def regrets(gm):  # gm: arm x cell [x reps] of mean log makespan
        return gm - gm[fixed].min(axis=0, keepdims=True)

    gm = _cell_gm_logs(logs, present, None)
    reg = regrets(gm)  # log regret
    maxreg = reg.max(axis=1)  # per arm
    ti = arms.index(test_arm)
    ci = [arms.index(a) for a in comparators]
    best = ci[int(np.argmin(maxreg[ci]))]
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, K, size=(reps, K))
    gmb = _cell_gm_logs(logs, present, draws)  # arm x cell x reps
    regb = regrets(gmb)
    maxb = regb.max(axis=1)  # arm x reps
    comp_min_log = maxb[ci].min(axis=0)
    d_log = maxb[ti] - comp_min_log
    d_ratio = np.exp(maxb[ti]) - np.exp(comp_min_log)
    cell_names = list(cells)
    return MaxRegret(
        max_regret={a: float(np.exp(maxreg[i])) for i, a in enumerate(arms)},
        regret={a: {cell_names[j]: float(np.exp(reg[i, j])) for j in range(C)} for i, a in enumerate(arms)},
        test_arm=test_arm, comparator=arms[best],
        diff=float(np.exp(maxreg[ti]) - np.exp(maxreg[ci].min())), diff_hi=float(np.quantile(d_ratio, 1 - alpha)),
        diff_log=float(maxreg[ti] - maxreg[ci].min()), diff_log_hi=float(np.quantile(d_log, 1 - alpha)),
        n_clusters=K)


# ---------------------------------------------------------------------------------------------------------------
# one competitor comparison as the gates use it (T1, K1-K4, E1)
# ---------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Comparison:
    method_a: str
    method_b: str
    pooled: GMRatio  # primary: pairs where both succeeded
    per_setting: dict[str, GMRatio]
    success: SuccessTest
    imputed: GMRatio | None  # sensitivity: failures scored as MAX_TIME
    clear_margin: bool  # spec 9.2 (pooled <= 0.97 and < 1 on >= 3/4 settings)
    t1_pass: bool  # spec 7 T1 (clear margin and pooled CI upper bound < 1)


def compare(rows: Iterable[Mapping], method_a: str, method_b: str, *,
            pair_keys: Sequence[str] = ("setting", "inst", "family", "seed"),
            cluster_keys: Sequence[str] = ("setting", "inst"), method_key: str = "method",
            where: Callable[[Mapping], bool] | None = None, reps: int = 10_000, seed: int = 0) -> Comparison:
    """Pooled and per-setting GM ratio of ``method_a`` / ``method_b`` (cluster bootstrap over instances, strata =
    settings), the paired success analysis and the failure-imputed sensitivity ratio. The setting is the first
    cluster key."""
    rows = list(rows)
    kw = {"pair_keys": pair_keys, "cluster_keys": cluster_keys, "method_key": method_key, "where": where}
    d = pair_rows(rows, method_a, method_b, **kw)
    pooled = gm_ratio(d["log_r"], d["clusters"], strata=d["strata"], reps=reps, seed=seed)
    per = {}
    for s_ in dict.fromkeys(d["strata"]):
        idx = [k for k, c in enumerate(d["clusters"]) if c[0] == s_]
        if len({d["clusters"][k] for k in idx}) >= 2:
            per[s_] = gm_ratio(d["log_r"][idx], [d["clusters"][k] for k in idx], reps=reps, seed=seed)
    succ = success_test(d["success_a"], d["success_b"], d["all_clusters"], reps=reps, seed=seed)
    imputed = None
    if not d["keep"].all():
        di = pair_rows(rows, method_a, method_b, failure="impute", **kw)
        imputed = gm_ratio(di["log_r"], di["clusters"], strata=di["strata"], reps=reps, seed=seed)
    cm = clear_margin(pooled, per) if len(per) >= 3 else False
    t1 = clear_margin(pooled, per, require_ci=True) if len(per) >= 3 else False
    return Comparison(method_a, method_b, pooled, per, succ, imputed, cm, t1)

