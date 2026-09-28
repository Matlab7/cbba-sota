"""Perturbations of DynHeteroMRTA-X (docs/trackD-spec.md Sections 3.3-3.4) with common-random-number calendars.

Every random quantity is drawn up front, before any method acts, from a generator keyed by
``(setting index, instance key, CRN seed, stream, entity)`` (``crn_rng``). The instance key is the generator seed of
the instance (``Setting.seed(split, i)``, e.g. 500000 + 1000 * setting index + i on dev), so keys never collide
across settings or splits. Delay calendars live in absolute time, so every method executed on the same
``(setting, instance, CRN seed, cell)`` faces the same realization whatever it decides.

Streams (entity in brackets):
- ``STREAM_DELAY`` [robot]: LoRR 2026 Start-Kit delay calendar (``delay_calendar``).
- ``STREAM_DUR`` [task]: realized duration factor (lognormal, exponential or uniform; ``realized_durations``).
- ``STREAM_RELEASE`` [0]: which tasks are dynamic and their release times (R2/R3).
- ``STREAM_FAIL`` [robot]: failure indicator uniform and onset (``failure_calendar``).
- ``STREAM_CHANNEL`` (5) is reserved for the T2 communication layer (not drawn here).

Models (anchor labels exactly as in the spec, Section 3.3):
- Travel delay (external model): LoRR 2026 Start-Kit ``DelayGenerator`` (MIT), verified in source on 2026-09-28 at
  MAPF-Competition/Start-Kit@9bc75d9 (``src/DelayGenerator.cpp``, last changed in 77bad9a; ``src/Simulator.cpp``).
  Bernoulli event model, uniform durations: every tick, first every active delay is decremented, then every agent
  with no remaining delay starts a delay with probability ``pDelay`` whose length is uniform on
  ``{minDelay..maxDelay}`` ticks; a delay drawn at tick k blocks the agent on ticks k..k+d-1 and the agent may start
  a new one at tick k+d. Idle agents are drawn too (a stall only matters while travelling). Here one tick is the env's
  own ``dt`` = 0.1 time units, a delay drawn at tick k is the stall interval ``[k*dt, (k+d)*dt)``, and travel is
  continuous at nominal speed outside stalls. Only the distribution is reproduced: our generator is numpy
  PCG64 per robot, not the Start-Kit's shared mt19937 stream. The Poisson event model and Gaussian durations of the
  Start-Kit are not implemented. N1 = {0.01, 1-4} (Start-Kit example), N3 = {0.05, 1-10}, N4 = {0.05, 10-50}.
- Durations (convention): mean-preserving lognormal ``d * exp(sigma Z - sigma^2 / 2)``; sensitivity models
  exponential with mean d and uniform on [0, 2d).
- Release R1 (weak anchor, *inspired by* the authors' GIF animation, ``task_env.py:686``, ``reactive_planning=False``
  by default, visual only; not an env feature): ids <= 20 at t = 0, then +20 ids every 10 time units. Its release
  window end H is the last batch epoch.
- Release R2 / R3 (convention: DVRP degree of dynamism 0.5 / 0.8): round(dod * T) tasks chosen uniformly are dynamic;
  their release times are uniform order statistics on [0, H] (the pilots called this "Poisson"; a Poisson process
  conditioned on its count gives the same law), H = 0.5 x the paper's published RL(g.) makespan of the setting.
  The pilots rounded H (25, 15, 20, 12); here H is exact (25.079, 14.812, 19.7135, 12.1165).
- Failures F (convention; event type after D-ITAGS, rates after Gosrich et al.): each robot fails with probability
  p_f, onset uniform on [0, 0.7 x published RL(g.) makespan], fail-stop, detected ``DETECT_TICKS`` = 5 ticks later.
  The indicator uniform and the onset are drawn for every robot whatever p_f, so the p_f = 0.1 failure set is a subset
  of the p_f = 0.2 set with the same onsets. A realization is *excluded* (for every method, paired) when the robots
  that never fail cannot cover some task (``Realization.excluded``): a method-independent, conservative rule.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from cbba_sota.bench import configs

TICK = 0.1  # env dt; one LoRR tick
T_CAL = 400.0  # delay calendars cover [0, T_CAL); twice the env's 200 time cap
STREAM_DELAY, STREAM_DUR, STREAM_RELEASE, STREAM_FAIL, STREAM_CHANNEL = 1, 2, 3, 4, 5
H_FRACTION = 0.5  # R2/R3 release window = 0.5 x published RL(g.) makespan
FAIL_ONSET_FRACTION = 0.7  # failure onset window = 0.7 x published RL(g.) makespan
DETECT_TICKS = 5  # heartbeat timeout h under good communication
R1_FIRST, R1_BATCH, R1_PERIOD = 20, 20, 10.0
PRIMARY_SETTINGS = ("MA-AT-25-5-50", "MA-AT-50-5-50", "SA-AT-50-5-50", "SA-BT-50-5-50")
SECONDARY_SETTINGS = ("MA-AT-50-5-200",)


def crn_rng(setting_index: int, instance_key: int, crn_seed: int, stream: int, entity: int = 0) -> np.random.Generator:
    """Generator for one (setting, instance, CRN seed, stream, entity) key; independent of method decisions."""
    return np.random.default_rng([int(setting_index), int(instance_key), int(crn_seed), int(stream), int(entity)])


# --- models ------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class DelayModel:
    """LoRR Start-Kit delay parameters: per-tick start probability and inclusive length range in ticks."""

    p: float
    min_ticks: int
    max_ticks: int

    def expected_stall_fraction(self) -> float:
        """Long-run fraction of ticks spent delayed: renewal E[d] / (E[available ticks before a start] + E[d])."""
        if self.p <= 0:
            return 0.0
        mean_d = 0.5 * (self.min_ticks + self.max_ticks)
        return mean_d / ((1.0 - self.p) / self.p + mean_d)

    def kappa(self) -> float:
        """Travel-time inflation 1 / (1 - expected stall fraction) (the spec's ETA predictor, Section 4.1)."""
        return 1.0 / (1.0 - self.expected_stall_fraction())


N1 = DelayModel(0.01, 1, 4)  # Start-Kit example config
N3 = DelayModel(0.05, 1, 10)
N4 = DelayModel(0.05, 10, 50)


@dataclass(frozen=True)
class Cell:
    """One scenario cell of a family (Section 3.4)."""

    name: str
    family: str  # static | F0 | F1 | F2 | F3 | secondary
    release: str = "none"  # none | R1 | R2 | R3
    delay: DelayModel | None = None
    dur_model: str = "none"  # none | lognormal | exponential | uniform
    dur_sigma: float = 0.0  # lognormal sigma
    p_fail: float = 0.0
    primary: bool = False  # part of the C1-dyn primary families F1-F3

    @property
    def dod(self) -> float:
        return {"R2": 0.5, "R3": 0.8}.get(self.release, 0.0)

    def as_dict(self) -> dict:
        d = asdict(self)
        d["delay"] = None if self.delay is None else asdict(self.delay)
        return d


def _cells() -> dict[str, Cell]:
    n12 = {"delay": N1, "dur_model": "lognormal", "dur_sigma": 0.3}
    n3s = {"delay": N3, "dur_model": "lognormal", "dur_sigma": 0.3}
    cells = [
        Cell("static", "static"),
        # F0 noise control, all tasks at t = 0 (listed cells of the spec table)
        Cell("F0-N1", "F0", delay=N1),
        Cell("F0-N2", "F0", dur_model="lognormal", dur_sigma=0.3),
        Cell("F0-N3", "F0", **n3s),
        # F1 release (each with N12), C1-dyn primary
        Cell("F1-R1", "F1", release="R1", primary=True, **n12),
        Cell("F1-R2", "F1", release="R2", primary=True, **n12),
        Cell("F1-R3", "F1", release="R3", primary=True, **n12),
        # F2 release + moderate noise
        Cell("F2-R2N3", "F2", release="R2", primary=True, **n3s),
        # F3 failures (N12)
        Cell("F3-pf0.1", "F3", p_fail=0.1, primary=True, **n12),
        Cell("F3-pf0.2", "F3", p_fail=0.2, primary=True, **n12),
        Cell("F3-pf0.2-R2", "F3", release="R2", p_fail=0.2, primary=True, **n12),
        # secondary / sensitivity (descriptive)
        Cell("S-N4", "secondary", delay=N4, dur_model="lognormal", dur_sigma=0.6),
        Cell("S-R2-durexp", "secondary", release="R2", delay=N1, dur_model="exponential"),
        Cell("S-R2-durunif", "secondary", release="R2", delay=N1, dur_model="uniform"),
    ]
    return {c.name: c for c in cells}


CELLS: dict[str, Cell] = _cells()
FAMILIES: dict[str, tuple[str, ...]] = {}
for _c in CELLS.values():
    FAMILIES.setdefault(_c.family, ())
    FAMILIES[_c.family] += (_c.name,)


def cell(name: str) -> Cell:
    try:
        return CELLS[name]
    except KeyError:
        raise KeyError(f"unknown cell {name!r}; known: {', '.join(CELLS)}") from None


# --- anchors -----------------------------------------------------------------------------------------------------


def published_rl_makespan(setting: str) -> float:
    """Mean makespan of RL(g.) in the paper's table row of ``setting`` (RA-L 2025, Tables I-IV)."""
    row = configs.get(setting).paper.get("RL(g.)")
    if row is None or row.makespan is None:
        raise KeyError(f"{setting}: no published RL(g.) makespan")
    return float(row.makespan)


def release_window(setting: str, c: Cell, n_tasks: int) -> float:
    """Release window end H (known to every method; 0 without release)."""
    if c.release == "none":
        return 0.0
    if c.release == "R1":
        return float(r1_release_times(n_tasks).max())
    return H_FRACTION * published_rl_makespan(setting)


# --- draws -------------------------------------------------------------------------------------------------------


def r1_release_times(n_tasks: int) -> np.ndarray:
    """Hidden iff id > clip(t // 10 * 20 + 20, 20, ...) (the animation's rule, without its 100-id visual clip)."""
    ids = np.arange(n_tasks)
    return np.where(ids <= R1_FIRST, 0.0, np.ceil((ids - R1_FIRST) / R1_BATCH) * R1_PERIOD)


def dod_release_times(rng: np.random.Generator, n_tasks: int, dod: float, horizon: float) -> np.ndarray:
    """round(dod * T) uniformly chosen tasks get uniform order statistics on [0, horizon]; the rest are known at 0."""
    n_dyn = round(dod * n_tasks)
    rel = np.zeros(n_tasks)
    dyn = rng.choice(n_tasks, n_dyn, replace=False)
    rel[dyn] = np.sort(rng.uniform(0.0, horizon, n_dyn))
    return rel


def realized_durations(c: Cell, dur_nom: np.ndarray, key: tuple[int, int, int]) -> np.ndarray:
    """Per-task realized durations (entity = task id)."""
    if c.dur_model == "none":
        return np.array(dur_nom, dtype=float)
    out = np.empty(len(dur_nom))
    for j, d in enumerate(dur_nom):
        rng = crn_rng(*key, STREAM_DUR, j)
        if c.dur_model == "lognormal":
            out[j] = d * np.exp(c.dur_sigma * rng.standard_normal() - 0.5 * c.dur_sigma ** 2)
        elif c.dur_model == "exponential":
            out[j] = d * rng.standard_exponential()
        elif c.dur_model == "uniform":
            out[j] = d * 2.0 * rng.random()
        else:
            raise ValueError(f"unknown duration model {c.dur_model!r}")
    return out


def delay_calendar(model: DelayModel | None, rng: np.random.Generator, t_max: float = T_CAL
                   ) -> tuple[np.ndarray, np.ndarray]:
    """(starts, ends) of one robot's stall intervals in env time, Start-Kit Bernoulli/uniform semantics.

    The number of available ticks before a delay starts is geometric (failures before the first success of the
    per-tick Bernoulli), so a delay starts at tick ``k + G - 1`` where k is the first available tick; the robot is
    available again at the tick where its delay ends (``nextTick`` decrements before sampling)."""
    if model is None or model.p <= 0:
        return np.zeros(0), np.zeros(0)
    n_ticks = round(t_max / TICK)
    starts, ends, k = [], [], 0
    while True:
        k += int(rng.geometric(model.p)) - 1
        if k >= n_ticks:
            break
        d = int(rng.integers(model.min_ticks, model.max_ticks + 1))
        starts.append(k * TICK)
        ends.append((k + d) * TICK)
        k += d
    return np.array(starts), np.array(ends)


def failure_calendar(p_fail: float, onset_max: float, n_agents: int, key: tuple[int, int, int]) -> np.ndarray:
    """[A] failure onset per robot (inf = never fails)."""
    out = np.full(n_agents, np.inf)
    for i in range(n_agents):
        rng = crn_rng(*key, STREAM_FAIL, i)
        u, onset = rng.random(), rng.uniform(0.0, onset_max)
        if u < p_fail:
            out[i] = onset
    return out


def survivors_cover(req: np.ndarray, ab: np.ndarray, fail_onset: np.ndarray) -> bool:
    """Robots that never fail can together cover every task (the spec's inclusion rule for failure realizations)."""
    alive = ~np.isfinite(fail_onset)
    total = ab[alive].sum(axis=0)
    return bool((req <= total[None, :] + 1e-9).all())


# --- realization -------------------------------------------------------------------------------------------------


@dataclass
class Realization:
    """Everything random about one episode, fixed before any method runs."""

    setting: str
    instance_key: int  # instance generator seed (e.g. 500000 + 1000 * setting index + i on dev)
    crn_seed: int
    cell: Cell
    release: np.ndarray  # [T] release time per task (0 = known at start)
    H: float  # release window end (known); 0 = native end of episode
    dur_nom: np.ndarray  # [T]
    dur_real: np.ndarray  # [T]
    delays: list[tuple[np.ndarray, np.ndarray]]  # per robot (starts, ends)
    fail_onset: np.ndarray  # [A] inf = never
    detect_after: float = DETECT_TICKS * TICK
    excluded: bool = False  # failure realization makes some task infeasible (excluded for all methods)
    meta: dict = field(default_factory=dict)

    @property
    def n_tasks(self) -> int:
        return len(self.release)

    @property
    def n_agents(self) -> int:
        return len(self.fail_onset)

    def kappa(self) -> float:
        return 1.0 if self.cell.delay is None else self.cell.delay.kappa()

    def summary(self) -> dict:
        return {"setting": self.setting, "instance_key": self.instance_key, "crn_seed": self.crn_seed,
                "cell": self.cell.name, "H": self.H, "n_released_at_0": int((self.release <= 0).sum()),
                "n_failures": int(np.isfinite(self.fail_onset).sum()), "excluded": self.excluded,
                "stall_intervals": int(sum(len(s) for s, _ in self.delays))}


def static_realization(req: np.ndarray, dur: np.ndarray, n_agents: int) -> Realization:
    """No perturbation: the static env (used for regression)."""
    T = len(dur)
    return Realization("", 0, 0, CELLS["static"], np.zeros(T), 0.0, np.array(dur, float), np.array(dur, float),
                       [(np.zeros(0), np.zeros(0)) for _ in range(n_agents)], np.full(n_agents, np.inf))


def realize(setting: str, instance_key: int, crn_seed: int, c: Cell | str, req: np.ndarray, ab: np.ndarray,
            dur_nom: np.ndarray) -> Realization:
    """Draw the realization of ``c`` for one instance (arrays in env order: tasks 0..T-1, robots 0..A-1)."""
    c = cell(c) if isinstance(c, str) else c
    s = configs.get(setting)
    key = (s.index, int(instance_key), int(crn_seed))
    T, A = len(dur_nom), len(ab)
    if c.release == "none":
        release = np.zeros(T)
    elif c.release == "R1":
        release = r1_release_times(T)
    elif c.release in ("R2", "R3"):
        release = dod_release_times(crn_rng(*key, STREAM_RELEASE, 0), T, c.dod, release_window(setting, c, T))
    else:
        raise ValueError(f"unknown release model {c.release!r}")
    H = release_window(setting, c, T)
    dur_real = realized_durations(c, np.asarray(dur_nom, float), key)
    delays = [delay_calendar(c.delay, crn_rng(*key, STREAM_DELAY, i)) for i in range(A)]
    if c.p_fail > 0:
        fail = failure_calendar(c.p_fail, FAIL_ONSET_FRACTION * published_rl_makespan(setting), A, key)
    else:
        fail = np.full(A, np.inf)
    excluded = not survivors_cover(np.asarray(req, float), np.asarray(ab, float), fail)
    return Realization(setting, int(instance_key), int(crn_seed), c, release, H, np.array(dur_nom, float), dur_real,
                       delays, fail, excluded=excluded, meta={"key": list(key)})


def realize_instance(setting: str, split: str, i: int, crn_seed: int, c: Cell | str) -> Realization:
    """``realize`` for instance ``i`` of ``split`` of ``setting`` (reads the env pickle)."""
    from cbba_sota.hetero.instance import Instance

    if split == "test":
        raise PermissionError("Track D pilots and tuning use dev (and validation) only; the test split is frozen")
    s = configs.get(setting)
    inst = Instance.from_pickle(s.instance_path(split, i))
    return realize(setting, s.seed(split, i), crn_seed, c, inst.req, inst.ab, inst.dur)
