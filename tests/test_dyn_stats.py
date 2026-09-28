"""Tests for cbba_sota.dyn.stats (Track D analysis statistics)."""
from __future__ import annotations

import numpy as np
import pytest
from scipy import stats as st

from cbba_sota.dyn import stats as S


def _panel(n_inst=30, n_seed=3, effect=-0.05, sd_inst=0.06, sd_seed=0.02, seed=0):
    rng = np.random.default_rng(seed)
    u = rng.normal(effect, sd_inst, n_inst)
    vals, clus = [], []
    for i in range(n_inst):
        for s in range(n_seed):
            vals.append(u[i] + rng.normal(0, sd_seed))
            clus.append(("MA-AT-25-5-50", i))
    return np.array(vals), clus


def test_cluster_means_weigh_clusters_equally():
    m, labels = S.cluster_means([1.0, 3.0, 10.0], ["a", "a", "b"])
    assert labels == ["a", "b"]
    np.testing.assert_allclose(m, [2.0, 10.0])
    r = S.gm_ratio(np.log([0.5, 0.5, 2.0]), ["a", "a", "b"], reps=200)
    assert r.ratio == pytest.approx(1.0)  # mean of cluster means (log .5, log 2) = 0, not the observation mean
    assert (r.wins, r.losses, r.n_clusters, r.n_obs) == (1, 1, 2, 3)


def test_gm_ratio_constant_and_reproducible():
    clus = [(k, s) for k in range(10) for s in range(2)]
    c = [x[0] for x in clus]
    r = S.gm_ratio(np.full(20, np.log(0.9)), c, reps=500)
    assert r.ratio == pytest.approx(0.9) and r.lo == pytest.approx(0.9) and r.hi == pytest.approx(0.9)
    assert r.p_less == 0.0 and r.wins == 10
    v, cl = _panel()
    a = S.gm_ratio(v, cl, reps=2000)
    b = S.gm_ratio(v, cl, reps=2000)
    assert a == b
    assert S.gm_ratio(v, cl, reps=2000, seed=1).lo != a.lo
    means, _ = S.cluster_means(v, cl)
    assert a.p_less == pytest.approx(st.ttest_1samp(means, 0.0, alternative="less").pvalue)
    assert a.ratio == pytest.approx(np.exp(means.mean()))
    assert a.lo < a.ratio < a.hi
    with pytest.raises(ValueError):
        S.gm_ratio([np.inf, 0.0], ["a", "b"])


def test_cluster_ci_wider_than_naive_under_intra_cluster_correlation():
    v, cl = _panel(n_inst=20, n_seed=5, sd_inst=0.1, sd_seed=0.005)
    clustered = S.gm_ratio(v, cl, reps=4000)
    naive = S.gm_ratio(v, list(range(v.size)), reps=4000)
    assert (clustered.hi - clustered.lo) > 1.8 * (naive.hi - naive.lo)


def test_cluster_bootstrap_coverage():
    """Percentile CI of the GM ratio covers the true ratio at roughly the nominal rate."""
    hits, n_sim = 0, 300
    for k in range(n_sim):
        v, cl = _panel(n_inst=30, n_seed=3, effect=-0.03, seed=100 + k)
        r = S.gm_ratio(v, cl, reps=1000, seed=k)
        hits += r.lo <= np.exp(-0.03) <= r.hi
    assert 0.89 <= hits / n_sim <= 0.99


def test_two_stage_equals_one_stage_with_single_observations():
    rng = np.random.default_rng(3)
    v = rng.normal(size=15)
    cl = list(range(15))
    one = S.cluster_bootstrap(v, cl, reps=300, seed=4)
    two = S.cluster_bootstrap(v, cl, reps=300, seed=4, two_stage=True)
    np.testing.assert_allclose(one, two)
    v2, cl2 = _panel(n_inst=10, n_seed=4)
    b = S.cluster_bootstrap(v2, cl2, reps=1300, two_stage=True)  # > one 500-draw chunk
    assert b.shape == (1300,) and np.all(np.isfinite(b))
    means, _ = S.cluster_means(v2, cl2)
    assert abs(b.mean() - means.mean()) < 0.01
    assert b.std() >= 0.9 * S.cluster_bootstrap(v2, cl2, reps=300).std()


def test_stratified_bootstrap_keeps_strata_sizes():
    v = [0.0] * 6 + [1.0] * 6
    cl = list(range(12))
    strata = ["A"] * 6 + ["B"] * 6
    b = S.cluster_bootstrap(v, cl, reps=200, strata=strata)
    np.testing.assert_allclose(b, 0.5)
    assert S.cluster_bootstrap(v, cl, reps=200).std() > 0
    with pytest.raises(ValueError):
        S.cluster_bootstrap([0, 1, 2], ["x", "x", "y"], strata=["A", "B", "B"], reps=10)


def test_hodges_lehmann():
    assert S.hodges_lehmann([1, 2, 3]) == pytest.approx(2.0)
    assert S.hodges_lehmann([0, 0, 10]) == pytest.approx(2.5)


def test_holm():
    p = [0.01, 0.04, 0.03, 0.005]
    np.testing.assert_array_equal(S.holm_reject(p, 0.05), [True, False, False, True])
    np.testing.assert_allclose(S.holm(p), [0.03, 0.06, 0.06, 0.02])
    fam = S.holm_family({"B1": 0.01, "B4": 0.04, "E2a": 0.03, "E3a": 0.005})
    assert [fam[k]["reject"] for k in ("B1", "B4", "E2a", "E3a")] == [True, False, False, True]
    assert fam["B4"]["p_holm"] == pytest.approx(0.06)


def test_tost():
    rng = np.random.default_rng(0)
    clusters = list(range(60))
    eq = S.tost(rng.normal(0.002, 0.01, 60), clusters)
    assert eq.equivalent_t and eq.equivalent_boot and eq.p < 0.05
    far = S.tost(rng.normal(0.05, 0.01, 60), clusters)
    assert not far.equivalent_t and not far.equivalent_boot
    x = rng.normal(0.01, 0.03, 40)
    t = S.tost(x, list(range(40)), margin_log=0.02)
    se = x.std(ddof=1) / np.sqrt(40)
    assert t.p_lower == pytest.approx(st.t.sf((x.mean() + 0.02) / se, 39))
    assert t.p_upper == pytest.approx(st.t.cdf((x.mean() - 0.02) / se, 39))
    assert S.TOST_MARGIN_2PCT == pytest.approx(0.0198, abs=1e-4)


def test_exact_mcnemar_and_success_test():
    a = [True] * 8 + [False] * 1 + [True] * 5
    b = [False] * 8 + [True] * 1 + [True] * 5
    r = S.exact_mcnemar(a, b)
    assert (r["b10"], r["b01"]) == (8, 1)
    assert r["p_two"] == pytest.approx(20 / 512)
    assert r["p_a_better"] == pytest.approx(10 / 512)
    assert S.exact_mcnemar([True, False], [True, False])["p_two"] == 1.0
    clus = [i // 2 for i in range(14)]
    s = S.success_test(a, b, clus, margin=0.02, reps=500)
    assert s.rate_a > s.rate_b and s.diff > 0 and s.noninferior
    assert s.sign_clusters["pos"] >= 1
    s2 = S.success_test(b, a, clus, margin=0.02, reps=500)
    assert s2.noninferior is False


def _rows():
    rows = []
    for inst in range(4):
        for seed in range(2):
            key = {"setting": "S", "inst": inst, "family": "F1", "seed": seed}
            ok_b = not (inst == 0 and seed == 0)
            rows.append({**key, "method": "A", "makespan": 9.0, "success": True})
            rows.append({**key, "method": "B", "makespan": 10.0 if ok_b else 200.0, "success": ok_b})
    rows.append({"setting": "S", "inst": 9, "family": "F1", "seed": 0, "method": "A", "makespan": 5.0,
                 "success": True})
    return rows


def test_pair_rows_drop_and_impute():
    rows = _rows()
    d = S.pair_rows(rows, "A", "B")
    assert d["log_r"].size == 7 and d["unmatched_a"] == 1 and d["keep"].sum() == 7
    np.testing.assert_allclose(d["log_r"], np.log(0.9))
    assert d["clusters"][0] == ("S", 0) and set(d["strata"]) == {"S"}
    imp = S.pair_rows(rows, "A", "B", failure="impute")
    assert imp["log_r"].size == 8 and imp["log_r"].min() == pytest.approx(np.log(9 / 200))
    with pytest.raises(ValueError):
        S.pair_rows(rows + [rows[0]], "A", "B")


def test_clear_margin():
    pooled = S.GMRatio(0.95, 0.93, 0.97, 0.95, 0.95, 0.001, 0.002, 80, 160, 0.05, 70, 10, 0, 140, 20)
    assert S.clear_margin(pooled, [0.9, 0.95, 0.99, 1.01], require_ci=True)
    assert not S.clear_margin(pooled, [0.9, 1.02, 0.99, 1.01])
    worse = S.GMRatio(0.98, 0.96, 1.0, 0.98, 0.98, 0.01, 0.02, 80, 160, 0.05, 60, 20, 0, 120, 40)
    assert not S.clear_margin(worse, [0.9, 0.95, 0.99, 0.99])


def test_max_regret_test():
    rng = np.random.default_rng(0)
    # per-cell GM factors: two fixed architectures that each lose 10% in one cell, a robust test arm (2% everywhere)
    factors = {"CEN": {"r1": 1.0, "r2": 1.10}, "REP": {"r1": 1.10, "r2": 1.0},
               "SPARC": {"r1": 1.02, "r2": 1.02}, "DS": {"r1": 1.05, "r2": 1.05}}
    rows = []
    for inst in range(40):
        base = rng.lognormal(3.5, 0.1)
        for cell in ("r1", "r2"):
            for seed in range(2):
                noise = rng.normal(0, 0.01)
                for arm, f in factors.items():
                    rows.append({"setting": "S", "inst": inst, "family": "F1", "rho": cell, "seed": seed,
                                 "method": arm, "makespan": base * f[cell] * np.exp(noise + rng.normal(0, 0.005))})
    r = S.max_regret_test(rows, arms_fixed=["CEN", "REP"], test_arm="SPARC", comparators=["CEN", "REP", "DS"],
                          reps=2000)
    assert r.max_regret["SPARC"] == pytest.approx(1.02, abs=0.01)
    assert r.max_regret["CEN"] == pytest.approx(1.10, abs=0.01)
    assert r.comparator == "DS"
    assert r.diff < 0 and r.diff_hi < 0 and r.diff_log_hi < 0
    assert r.regret["CEN"][("F1", "r1")] == pytest.approx(1.0, abs=0.01)
    with pytest.raises(ValueError):
        holes = [r for r in rows if not (r["method"] == "DS" and r["inst"] == 39 and r["rho"] == "r2")]
        S.max_regret_test(holes, arms_fixed=["CEN", "REP"], test_arm="SPARC", comparators=["DS"], reps=10)


def test_compare_gate_view():
    rng = np.random.default_rng(1)
    rows = []
    for setting in ("A", "B", "C", "D"):
        for inst in range(15):
            base = rng.lognormal(3.5, 0.1)
            for fam in ("F1", "F2"):
                for sd in range(2):
                    key = {"setting": setting, "inst": inst, "family": fam, "seed": sd}
                    rows.append({**key, "method": "SPARC", "makespan": base * 0.94 * np.exp(rng.normal(0, 0.01)),
                                 "success": True})
                    ok = not (setting == "A" and inst == 0 and sd == 0)
                    rows.append({**key, "method": "B4", "makespan": base if ok else 200.0, "success": ok})
    c = S.compare(rows, "SPARC", "B4", reps=1000)
    assert c.pooled.ratio == pytest.approx(0.94, abs=0.01) and c.t1_pass and c.clear_margin
    assert set(c.per_setting) == {"A", "B", "C", "D"} and all(g.ratio < 1 for g in c.per_setting.values())
    assert c.imputed is not None and c.imputed.ratio < c.pooled.ratio
    assert c.success.mcnemar_obs["b10"] == 2 and c.success.rate_a == 1.0
