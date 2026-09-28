"""HeteroMRTA benchmark settings (RA-L 2025, Tables I-IV), generator parameters, seeds and paths.

Generator conventions (TaskEnv of marmotlab/HeteroMRTA @ db51e29):
- ``kn`` agents split evenly over ``ks`` species; ``kb`` unique skills; 5-dim trait vectors for the network,
  settings with ``kb < 5`` (the SA settings with 3 species) are generated with ``traits_dim = kb`` and zero-padded
  to 5 ("unused skills set to zero"). MA-AT (9, 3, 20) uses kb = 5 although Table III prints 3 (see ``_T3``).
- SA (single-skill): species trait matrix ``np.diag(np.ones(kb))`` (so ``ks == kb``), the commented-out line of
  ``TaskEnv.generate_agent``; the shipped RALTestSet is reproduced bit-for-bit by it (seeds 0..49).
  MA (multi-skill): the repo default, unique non-zero random binary rows.
- BT (binary requirements): ``max_task_size = 2`` so requirements are in {0, 1}; AT (additive): ``max_task_size = 3``
  so requirements are in {0, 1, 2} (Section V-A). Durations ``U[0, 5)`` (``duration_scale = 5``).
- Table IV (up to 150 agents x 500 tasks) uses the same MA-AT generator and the unchanged 200 time cap; the released
  policy needs no padding or size overrides at test time (see ``cbba_sota.solvers.rl``).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HETEROMRTA_DIR = ROOT / "third_party" / "HeteroMRTA"
DATA_DIR = ROOT / "data" / "hetero"
RUNS_DIR = ROOT / "runs"

TRAIT_DIM = 5  # fixed by the released network (agent input 6 + 5, task input 5 + 2 * 5)
MAX_TIME = 200.0  # simulated-time cap of the env (execute_by_route) and of test.py; success = finished before it
SPLITS = {"test": (100_000, 50), "dev": (500_000, 20)}  # split -> (seed base, instances)


@dataclass(frozen=True)
class PaperRow:
    """One method row of the paper tables: success rate and mean (SD) over solved instances."""

    success: float
    makespan: float | None
    makespan_sd: float | None
    awt: float | None
    awt_sd: float | None
    time_s: float


@dataclass(frozen=True)
class Setting:
    index: int
    family: str  # SA-BT, SA-AT or MA-AT
    table: int  # paper table (1-4); 0 for the shipped RALTestSet
    n_agents: int
    n_species: int
    n_tasks: int
    n_skills: int
    paper: dict[str, PaperRow] = field(compare=False)
    source: Path | None = None  # shipped pickles instead of generation

    @property
    def name(self) -> str:
        if self.source is not None:
            return self.source.name
        return f"{self.family}-{self.n_agents}-{self.n_species}-{self.n_tasks}"

    @property
    def single_skill(self) -> bool:
        return self.family.startswith("SA")

    @property
    def binary(self) -> bool:
        return self.family.endswith("BT")

    @property
    def per_species(self) -> int:
        return self.n_agents // self.n_species

    def generator_kwargs(self) -> dict:
        """Keyword arguments of ``TaskEnv.__init__`` (before SA override and trait padding)."""
        n = self.per_species
        return {"per_species_range": (n, n), "species_range": (self.n_species, self.n_species),
                "tasks_range": (self.n_tasks, self.n_tasks), "traits_dim": self.n_skills, "decision_dim": 10,
                "max_task_size": 2 if self.binary else 3, "duration_scale": 5}

    def seed(self, split: str, i: int) -> int:
        base, n = SPLITS[split]
        if not 0 <= i < n:
            raise IndexError(f"{split} has {n} instances")
        return base + 1000 * self.index + i

    def n_instances(self, split: str) -> int:
        if self.source is not None:
            return 50 if split == "test" else 0
        return SPLITS[split][1]

    def splits(self) -> tuple[str, ...]:
        return ("test",) if self.source is not None else tuple(SPLITS)

    def instance_path(self, split: str, i: int) -> Path:
        return DATA_DIR / self.name / split / f"env_{i}.pkl"


_METHODS = {"TACO": "TACO", "SAS": "SAS", "CTAS_D": "CTAS-D", "Greedy": "Greedy", "RLg": "RL(g.)", "RLs10": "RL(s.10)"}


def _rows(**rows: tuple) -> dict[str, PaperRow]:
    return {_METHODS[k]: PaperRow(*v) for k, v in rows.items()}


# (success, makespan, sd, awt, sd, time_s) transcribed from docs/refs/heteromrta-ral2025.txt, Tables I-IV.
_T1 = [
    (9, 3, 20, 3, _rows(TACO=(1, 35.068, 5.857, 9.679, 4.905, 66.59), SAS=(1, 27.408, 3.918, 4.286, 1.78, 25.60),
                        CTAS_D=(1, 23.658, 2.918, 2.519, 1.068, 600), Greedy=(1, 32.733, 3.371, 5.377, 1.572, 0.11),
                        RLg=(1, 29.002, 3.073, 4.125, 1.414, 0.43), RLs10=(1, 27.193, 2.715, 3.687, 1.333, 4.33))),
    (25, 5, 20, 5, _rows(TACO=(1, 28.86, 5.561, 9.327, 4.385, 120.87), SAS=(1, 27.329, 4.686, 5.686, 1.982, 40.37),
                         CTAS_D=(1, 16.488, 1.744, 1.988, 0.519, 600), Greedy=(1, 21.879, 2.827, 3.163, 1.196, 0.34),
                         RLg=(1, 20.877, 2.467, 2.74, 0.954, 0.68), RLs10=(1, 20.071, 2.219, 2.539, 0.767, 6.54))),
    (25, 5, 50, 5, _rows(TACO=(1, 63.923, 7.051, 33.727, 5.949, 527.26), SAS=(1, 56.523, 7.221, 21.898, 4.973, 169.6),
                         CTAS_D=(0.04, None, None, None, None, 3600), Greedy=(1, 43.964, 3.749, 8.771, 1.573, 0.402),
                         RLg=(1, 36.996, 3.1, 6.496, 0.952, 1.52), RLs10=(1, 35.477, 2.842, 6.042, 1.053, 13.45))),
    (50, 5, 50, 5, _rows(TACO=(1, 38.244, 7.029, 15.858, 6.704, 909.67), SAS=(1, 52.682, 5.738, 13.469, 3.503, 201.41),
                         CTAS_D=(0.96, 18.649, 1.922, 2.479, 0.54, 3600), Greedy=(1, 27.153, 2.317, 3.406, 0.622, 0.74),
                         RLg=(1, 24.233, 2.071, 2.703, 0.494, 2.03), RLs10=(1, 22.871, 1.45, 2.703, 0.515, 19.14))),
]
_T2 = [
    (9, 3, 20, 3, _rows(CTAS_D=(0.40, 47.166, 11.686, 13.414, 7.247, 600), Greedy=(1, 64.449, 11.525, 22.895, 7.171, 0.07),
                        RLg=(1, 54.738, 10.073, 18.214, 5.915, 0.34), RLs10=(1, 50.648, 8.616, 14.532, 4.492, 3.37))),
    (15, 5, 20, 5, _rows(CTAS_D=(0.34, 58.665, 7.82, 21.045, 6.687, 1800), Greedy=(1, 74.179, 9.658, 31.493, 6.666, 0.15),
                         RLg=(0.91, 62.987, 11.596, 25.427, 12.836, 1.16),
                         RLs10=(1, 57.454, 6.681, 19.689, 4.025, 11.06))),
    (25, 5, 20, 5, _rows(CTAS_D=(0.68, 32.406, 8.347, 8.986, 5.59, 600), Greedy=(1, 41.121, 4.421, 12.22, 2.627, 0.218),
                         RLg=(1, 36.091, 3.343, 9.107, 2.187, 1.09), RLs10=(1, 33.439, 3.498, 7.476, 1.686, 10.2))),
    (50, 5, 50, 5, _rows(CTAS_D=(0.0, None, None, None, None, 3600), Greedy=(1, 46.269, 3.669, 10.433, 1.373, 1.185),
                         RLg=(1, 39.427, 3.124, 7.741, 1.452, 3.08), RLs10=(1, 37.664, 2.798, 7.253, 1.196, 29.8))),
]
# MA-AT (9, 3, 20): Table III prints kb = 3, but only kb = 5 reproduces its RL(g.) row. On our 50 test seeds, kb = 3
# gives makespan 44.58 (SD 11.62) and ability redundancy (env.get_efficiency) 1.60, kb = 5 gives 52.67 (12.06) and
# 2.00 vs the paper's 54.07 (11.30) and 2.007; the other MA-AT rows (kb = 5) report 1.91-1.94.
_T3 = [
    (9, 3, 20, 5, _rows(CTAS_D=(0.28, 42.244, 7.541, 5.003, 3.433, 600), Greedy=(1, 64.529, 13.293, 20.459, 8.546, 0.08),
                        RLg=(1, 54.066, 11.304, 17.472, 6.754, 0.3571),
                        RLs10=(1, 49.337, 11.126, 14.035, 6.378, 3.571))),
    (15, 5, 20, 5, _rows(CTAS_D=(0.36, 35.916, 9.082, 5.581, 2.364, 1800), Greedy=(1, 45.126, 11.124, 12.75, 8.713, 0.11),
                         RLg=(1, 38.495, 9.985, 10.917, 7.225, 0.86), RLs10=(1, 35.911, 8.985, 9.021, 6.038, 8.4))),
    (25, 5, 20, 5, _rows(CTAS_D=(0.90, 21.116, 5.439, 2.183, 0.98, 600), Greedy=(1, 29.087, 5.828, 5.945, 3.523, 0.173),
                         RLg=(1, 25.625, 5.006, 4.727, 2.894, 0.76), RLs10=(1, 23.674, 4.382, 4.044, 2.483, 7.6))),
    (25, 5, 50, 5, _rows(CTAS_D=(0.0, None, None, None, None, 3600), Greedy=(1, 63.214, 14.069, 15.986, 9.778, 0.529),
                         RLg=(1, 50.158, 11.886, 12.712, 7.893, 1.634),
                         RLs10=(1, 46.983, 10.711, 11.555, 7.12, 16.34))),
    (50, 5, 50, 5, _rows(CTAS_D=(0.14, 39.762, 5.752, 0.614, 0.253, 3600), Greedy=(1, 35.171, 5.701, 5.965, 2.827, 0.983),
                         RLg=(1, 29.624, 5.011, 4.732, 2.426, 2.343), RLs10=(1, 27.98, 4.581, 4.788, 2.698, 23.43))),
]
_T4 = [
    (50, 5, 200, 5, _rows(Greedy=(1, 114.98, 24.951, 54.075, 12.532, 9.94), RLg=(1, 82.698, 21.279, 36.564, 9.727, 7.32),
                          RLs10=(1, 81.213, 20.657, 36.11, 9.401, 75.2))),
    (150, 10, 500, 5, _rows(Greedy=(1, 86.776, 9.513, 11.588, 2.791, 62.6), RLg=(1, 56.728, 8.694, 8.694, 4.536, 55.4),
                            RLs10=(1, 56.093, 8.802, 8.934, 4.553, 560.2))),
    (150, 5, 500, 5, _rows(Greedy=(1, 97.735, 20.005, 14.08, 4.786, 109.62), RLg=(1, 67.35, 16.745, 15.673, 10.816, 54.6),
                           RLs10=(1, 66.199, 16.475, 15.655, 10.784, 560.5))),
]


def _build() -> list[Setting]:
    out: list[Setting] = []
    for family, table, rows in (("SA-BT", 1, _T1), ("SA-AT", 2, _T2), ("MA-AT", 3, _T3), ("MA-AT", 4, _T4)):
        for kn, ks, km, kb, paper in rows:
            out.append(Setting(len(out), family, table, kn, ks, km, kb, paper))
    # Shipped test set: SA-BT, 5 species x 3 agents, 20 tasks, seeds 0..49 (no paper table row).
    out.append(Setting(len(out), "SA-BT", 0, 15, 5, 20, 5, {}, HETEROMRTA_DIR / "RALTestSet"))
    return out


SETTINGS: list[Setting] = _build()
BY_NAME: dict[str, Setting] = {s.name: s for s in SETTINGS}


def get(name: str) -> Setting:
    try:
        return BY_NAME[name]
    except KeyError:
        raise KeyError(f"unknown setting {name!r}; known: {', '.join(BY_NAME)}") from None
