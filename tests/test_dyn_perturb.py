"""Perturbation families of DynHeteroMRTA-X (docs/trackD-spec.md Sections 3.3-3.4): CRN keys, the LoRR 2026
Start-Kit delay semantics, release models, durations and failures."""
from __future__ import annotations

import numpy as np
import pytest

from cbba_sota.bench import configs
from cbba_sota.dyn import perturb
from cbba_sota.dyn.env import make_env
from cbba_sota.dyn.perturb import N1, N3, N4, TICK, DelayModel, crn_rng, delay_calendar

PAPER_H = {"MA-AT-25-5-50": 25.079, "MA-AT-50-5-50": 14.812, "SA-AT-50-5-50": 19.7135, "SA-BT-50-5-50": 12.1165}


def test_cells_match_the_spec_families():
    assert perturb.FAMILIES["F0"] == ("F0-N1", "F0-N2", "F0-N3")
    assert perturb.FAMILIES["F1"] == ("F1-R1", "F1-R2", "F1-R3")
    assert perturb.FAMILIES["F2"] == ("F2-R2N3",)
    assert perturb.FAMILIES["F3"] == ("F3-pf0.1", "F3-pf0.2", "F3-pf0.2-R2")
    for name in perturb.FAMILIES["F1"] + perturb.FAMILIES["F3"]:  # every dynamic cell carries N12 = N1 + sigma 0.3
        c = perturb.cell(name)
        assert c.primary and c.delay == N1 and c.dur_model == "lognormal" and c.dur_sigma == 0.3
    c = perturb.cell("F2-R2N3")
    assert c.release == "R2" and c.delay == N3 and c.dur_sigma == 0.3
    assert (N1.p, N1.min_ticks, N1.max_ticks) == (0.01, 1, 4) and (N3.p, N3.max_ticks) == (0.05, 10)


def test_crn_keys_are_deterministic_and_separate():
    a = crn_rng(3, 503000, 1, perturb.STREAM_DELAY, 7).random(4)
    assert np.array_equal(a, crn_rng(3, 503000, 1, perturb.STREAM_DELAY, 7).random(4))
    for other in ((3, 503000, 1, perturb.STREAM_DUR, 7), (3, 503000, 2, perturb.STREAM_DELAY, 7),
                  (3, 503001, 1, perturb.STREAM_DELAY, 7), (3, 503000, 1, perturb.STREAM_DELAY, 8)):
        assert not np.array_equal(a, crn_rng(*other).random(4))


def test_realization_is_reproducible_and_shared_across_cells():
    r1 = perturb.realize_instance("MA-AT-25-5-50", "dev", 0, 1, "F1-R2")
    r2 = perturb.realize_instance("MA-AT-25-5-50", "dev", 0, 1, "F1-R2")
    assert np.array_equal(r1.release, r2.release) and np.array_equal(r1.dur_real, r2.dur_real)
    assert all(np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1]) for a, b in zip(r1.delays, r2.delays))
    # the delay and duration streams do not depend on the release model (paired across F1 cells)
    r3 = perturb.realize_instance("MA-AT-25-5-50", "dev", 0, 1, "F1-R3")
    assert np.array_equal(r1.dur_real, r3.dur_real)
    assert all(np.array_equal(a[0], b[0]) for a, b in zip(r1.delays, r3.delays))
    assert not np.array_equal(r1.release, r3.release)


# --- releases --------------------------------------------------------------------------------------------------------


def test_r1_batch_release():
    rel = perturb.r1_release_times(50)
    assert (rel[:21] == 0).all() and (rel[21:41] == 10).all() and (rel[41:] == 20).all()
    rz = perturb.realize_instance("SA-BT-50-5-50", "dev", 0, 0, "F1-R1")
    assert rz.H == 20.0 and np.array_equal(rz.release, rel)
    assert perturb.r1_release_times(200).max() == 90.0


@pytest.mark.parametrize("setting", sorted(PAPER_H))
@pytest.mark.parametrize("cell,dod", [("F1-R2", 0.5), ("F1-R3", 0.8)])
def test_degree_of_dynamism_release(setting, cell, dod):
    H = PAPER_H[setting]
    for seed in range(3):
        rz = perturb.realize_instance(setting, "dev", seed, seed, cell)
        assert rz.H == pytest.approx(H, abs=1e-12) and rz.H == 0.5 * perturb.published_rl_makespan(setting)
        dyn = rz.release > 0
        assert dyn.sum() == round(dod * rz.n_tasks) and (rz.release[dyn] < H).all()


def test_release_order_statistics_are_uniform():
    rng = crn_rng(0, 0, 0, perturb.STREAM_RELEASE)
    times = np.concatenate([perturb.dod_release_times(rng, 50, 0.5, 10.0) for _ in range(400)])
    times = times[times > 0] / 10.0
    assert abs(times.mean() - 0.5) < 0.01 and abs(np.mean(times < 0.25) - 0.25) < 0.01


# --- durations -------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("model,sigma", [("lognormal", 0.3), ("lognormal", 0.6), ("exponential", 0.0),
                                         ("uniform", 0.0)])
def test_durations_are_mean_preserving(model, sigma):
    c = perturb.Cell("x", "secondary", dur_model=model, dur_sigma=sigma)
    dur = np.ones(4000)
    real = perturb.realized_durations(c, dur, (0, 1, 2))
    assert abs(real.mean() - 1.0) < 0.05 and (real > 0).all()
    if model == "uniform":
        assert real.max() < 2.0
    assert np.array_equal(real, perturb.realized_durations(c, dur, (0, 1, 2)))


# --- LoRR Start-Kit delay semantics ---------------------------------------------------------------------------------


def startkit_ticks(model: DelayModel, n_ticks: int, n_agents: int, rng: np.random.Generator) -> np.ndarray:
    """Transcription of ``DelayGenerator::nextTick`` (Start-Kit v3.0.0; Bernoulli events, uniform lengths), all
    agents at once: [agents, ticks] bool, delayed (``remaining_delay > 0`` after ``nextTick``) at each tick."""
    remaining = np.zeros(n_agents, int)
    out = np.zeros((n_agents, n_ticks), bool)
    for k in range(n_ticks):
        remaining[remaining > 0] -= 1  # decrement active delays
        sel = (remaining == 0) & (rng.random(n_agents) < model.p)  # available agents, Bernoulli(pDelay)
        remaining[sel] = rng.integers(model.min_ticks, model.max_ticks + 1, sel.sum())  # uniform{min..max}
        out[:, k] = remaining > 0
    return out


def calendar_ticks(starts, ends, n_ticks: int) -> np.ndarray:
    out = np.zeros(n_ticks, bool)
    for s, e in zip(starts, ends):
        out[round(s / TICK):min(round(e / TICK), n_ticks)] = True
    return out


@pytest.mark.parametrize("model", [N1, N3, N4])
def test_delay_calendar_matches_the_startkit_generator_in_law(model):
    n_ticks, n = 4000, 300
    ref = startkit_ticks(model, n_ticks, n, crn_rng(9, 9, 9, 9))
    cal = np.array([calendar_ticks(*delay_calendar(model, crn_rng(1, 1, 1, 1, k), n_ticks * TICK), n_ticks)
                    for k in range(n)])

    def starts(x):  # delay onsets per tick (a delayed tick not preceded by a delayed tick, or a back-to-back start)
        return np.diff(np.concatenate([np.zeros((len(x), 1), bool), x], axis=1).astype(int), axis=1).clip(0).sum()

    frac_ref, frac_cal, expect = ref.mean(), cal.mean(), model.expected_stall_fraction()
    assert abs(frac_cal - expect) < 0.1 * expect + 0.002 and abs(frac_ref - expect) < 0.1 * expect + 0.002
    assert abs(frac_cal - frac_ref) < 0.12 * expect + 0.002
    # run lengths of delayed ticks are sums of back-to-back delays; compare the onset rates
    assert abs(starts(cal) - starts(ref)) < 0.12 * starts(ref) + 20


def test_delay_calendar_structure():
    for k in range(50):
        s, e = delay_calendar(N3, crn_rng(0, 0, 0, 1, k))
        ticks = np.round(s / TICK)
        assert np.allclose(s, ticks * TICK) and (np.diff(s) > 0).all()
        lengths = np.round((e - s) / TICK)
        assert lengths.min() >= N3.min_ticks and lengths.max() <= N3.max_ticks
        assert (s[1:] >= e[:-1] - 1e-9).all()  # a new delay may start on the tick where the previous ended
    assert delay_calendar(None, crn_rng(0, 0, 0, 1))[0].size == 0


def brute_arrival(starts, ends, dep, tau, dt=TICK / 200):
    t, moved = dep, 0.0
    while moved < tau - 1e-12:
        step = min(dt, tau - moved)
        if not np.any((starts <= t + 1e-12) & (t + 1e-12 < ends)):
            moved += step
        t += step
    return t


def test_continuous_arrival_equals_tick_integration():
    rz = perturb.realize_instance("MA-AT-25-5-50", "dev", 0, 2, "F0-N3")
    env = make_env(configs.get("MA-AT-25-5-50").instance_path("dev", 0), rz)
    rng = np.random.default_rng(0)
    worst, stalled = 0.0, 0
    for _ in range(30):
        i = int(rng.integers(len(env.agent_dic)))
        dep, tau = float(rng.uniform(0, 50)), float(rng.uniform(0.1, 5))
        arr = env._arrive(i, dep, tau)
        stalled += arr > dep + tau + 1e-9
        worst = max(worst, abs(arr - brute_arrival(*env.delays[i], dep, tau)))
    assert worst < 2 * TICK / 200 and stalled >= 10


# --- failures --------------------------------------------------------------------------------------------------------


def test_failure_draws_are_coupled_and_anchored():
    n_fail = {0.1: 0, 0.2: 0}
    total = 0
    for i in range(20):
        r1 = perturb.realize_instance("SA-BT-50-5-50", "dev", i, 0, "F3-pf0.1")
        r2 = perturb.realize_instance("SA-BT-50-5-50", "dev", i, 0, "F3-pf0.2")
        f1, f2 = np.isfinite(r1.fail_onset), np.isfinite(r2.fail_onset)
        assert (f2 | ~f1).all() and np.array_equal(r1.fail_onset[f1], r2.fail_onset[f1])  # pf0.1 set within pf0.2
        onset_max = 0.7 * perturb.published_rl_makespan("SA-BT-50-5-50")
        assert (r2.fail_onset[f2] >= 0).all() and (r2.fail_onset[f2] <= onset_max).all()
        assert r2.detect_after == pytest.approx(5 * TICK)
        n_fail[0.1] += f1.sum()
        n_fail[0.2] += f2.sum()
        total += r1.n_agents
    assert abs(n_fail[0.1] / total - 0.1) < 0.03 and abs(n_fail[0.2] / total - 0.2) < 0.04


def test_infeasible_failure_realizations_are_excluded():
    req = np.array([[1, 0], [0, 2]], float)
    ab = np.array([[1, 0], [0, 1], [0, 1]], float)
    assert perturb.survivors_cover(req, ab, np.array([np.inf, np.inf, np.inf]))
    assert not perturb.survivors_cover(req, ab, np.array([np.inf, 3.0, np.inf]))  # trait 1 drops to 1 < 2
    assert not perturb.survivors_cover(req, ab, np.array([5.0, np.inf, np.inf]))
    excluded = [perturb.realize_instance("MA-AT-25-5-50", "dev", i, 0, "F3-pf0.2").excluded for i in range(20)]
    assert not all(excluded)
