"""Paired comparison statistics for seed-matched experiments."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats as st


@dataclass(frozen=True)
class Paired:
    mean: float
    lo: float
    hi: float
    p: float
    n: int


def paired(a, b, reps: int = 10000, alpha: float = 0.05, seed: int = 0) -> Paired:
    """Mean of a - b over matched seeds, bootstrap CI and two-sided paired t p-value."""
    d = np.asarray(a, float) - np.asarray(b, float)
    n = d.size
    if n < 2:
        raise ValueError("need at least two pairs")
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, n, size=(reps, n))].mean(axis=1)
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    p = 1.0 if np.allclose(d, d[0]) and d[0] == 0 else float(st.ttest_1samp(d, 0.0).pvalue)
    return Paired(float(d.mean()), float(lo), float(hi), p, n)


def holm(pvalues) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values, same order as the input."""
    p = np.asarray(pvalues, float)
    order = np.argsort(p)
    m = p.size
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adj[idx] = min(1.0, running)
    return adj


def tost(a, b, margin: float) -> float:
    """Paired two one-sided tests for equivalence within +-margin; returns the larger p."""
    d = np.asarray(a, float) - np.asarray(b, float)
    n = d.size
    se = d.std(ddof=1) / np.sqrt(n)
    if se == 0:
        return 0.0 if abs(d.mean()) < margin else 1.0
    lower = st.t.sf((d.mean() + margin) / se, n - 1)
    upper = st.t.cdf((d.mean() - margin) / se, n - 1)
    return float(max(lower, upper))
