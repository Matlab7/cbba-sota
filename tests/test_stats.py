import numpy as np

from cbba_sota.stats import holm, paired, tost


def test_paired_detects_shift():
    rng = np.random.default_rng(1)
    b = rng.normal(0, 1, 40)
    r = paired(b + 1.0 + rng.normal(0, 0.1, 40), b)
    assert r.lo > 0.9 and r.hi < 1.1 and r.p < 1e-6 and r.n == 40


def test_holm_matches_hand_computation():
    adj = holm([0.01, 0.04, 0.03])
    assert np.allclose(adj, [0.03, 0.06, 0.06])


def test_tost_equivalence():
    rng = np.random.default_rng(2)
    b = rng.normal(0, 1, 60)
    assert tost(b + rng.normal(0, 0.05, 60), b, margin=0.1) < 0.05
    assert tost(b + 1.0, b, margin=0.1) > 0.5
