"""HeteroMRTA problem instance as plain arrays, with travel times computed exactly like the env."""
from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from os import PathLike
from pathlib import Path

import numpy as np

from cbba_sota.bench.heteromrta import TaskEnv, load_env

SPEED = 0.2
_DATA = ("req", "loc", "dur", "ab", "depot", "species")


def _travel(a: np.ndarray, b: np.ndarray, speed: float, symmetric: bool = False) -> np.ndarray:
    """``norm(a[i] - b[j]) / speed`` one pair at a time, as ``TaskEnv.calculate_eulidean_distance`` does.

    A 2-vector ``np.linalg.norm`` goes through BLAS ``dot`` (fused multiply-adds on this host), so a vectorized
    norm differs from the env in the last bit for ~8% of pairs. Pairwise calls keep the evaluator bit-exact.
    """
    out = np.zeros((len(a), len(b)))
    for i, x in enumerate(a):
        for j in range(i + 1 if symmetric else 0, len(b)):
            out[i, j] = np.linalg.norm(x - b[j]) / speed
    if symmetric:  # norm(x - y) == norm(y - x) bit for bit
        out += out.T
    return out


@dataclass(frozen=True, eq=False)
class Instance:
    req: np.ndarray  # [T, K] task requirements (integers stored as float)
    loc: np.ndarray  # [T, 2] task locations
    dur: np.ndarray  # [T] task durations
    ab: np.ndarray  # [A, K] agent traits
    depot: np.ndarray  # [A, 2] depot location of each agent (its species depot)
    species: np.ndarray  # [A] species index of each agent
    speed: float = SPEED
    name: str = ""
    source: str | None = None  # env pickle this instance was read from (ground truth for replay)

    def __post_init__(self) -> None:
        for field in _DATA:
            arr = np.array(getattr(self, field), dtype=np.int64 if field == "species" else float)
            arr.setflags(write=False)
            object.__setattr__(self, field, arr)
        object.__setattr__(self, "speed", float(self.speed))
        T, K = self.req.shape
        A = len(self.ab)
        if self.loc.shape != (T, 2) or self.dur.shape != (T,):
            raise ValueError("task arrays disagree on the number of tasks")
        if self.ab.shape != (A, K) or self.depot.shape != (A, 2) or self.species.shape != (A,):
            raise ValueError("agent arrays disagree on the number of agents or traits")

    def same_data(self, other: Instance) -> bool:
        """Same problem (arrays and speed), ignoring name and source."""
        return self.speed == other.speed and all(np.array_equal(getattr(self, f), getattr(other, f)) for f in _DATA)

    def __getstate__(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if k not in ("tt", "da")}  # derived, recomputed lazily

    @property
    def n_tasks(self) -> int:
        return len(self.req)

    @property
    def n_agents(self) -> int:
        return len(self.ab)

    @property
    def n_traits(self) -> int:
        return self.req.shape[1]

    @property
    def n_species(self) -> int:
        return int(self.species.max()) + 1 if self.n_agents else 0

    @cached_property
    def tt(self) -> np.ndarray:
        """[T, T] task-to-task travel time."""
        return _travel(self.loc, self.loc, self.speed, symmetric=True)

    @cached_property
    def da(self) -> np.ndarray:
        """[A, T] travel time between each agent's depot and each task (either direction)."""
        depots, index = np.unique(self.depot, axis=0, return_inverse=True)
        return _travel(depots, self.loc, self.speed)[index.ravel()]

    # --- loaders -------------------------------------------------------------------------------------------

    @classmethod
    def from_env(cls, env: TaskEnv, name: str = "", source: str | PathLike | None = None) -> Instance:
        tasks = [env.task_dic[j] for j in range(len(env.task_dic))]
        agents = [env.agent_dic[i] for i in range(len(env.agent_dic))]
        speeds = {float(a["velocity"]) for a in agents}
        if len(speeds) != 1:
            raise ValueError(f"agents move at different speeds {speeds}")
        return cls(req=[t["requirements"] for t in tasks], loc=[t["location"] for t in tasks],
                   dur=[t["time"] for t in tasks], ab=[a["abilities"] for a in agents],
                   depot=[a["depot"] for a in agents], species=[a["species"] for a in agents],
                   speed=speeds.pop(), name=name, source=None if source is None else str(source))

    @classmethod
    def from_pickle(cls, path: str | PathLike) -> Instance:
        path = Path(path).resolve()
        return cls.from_env(load_env(path), name=f"{path.parent.name}/{path.stem}", source=path)

    def save(self, path: str | PathLike) -> None:
        """Compact .npz (raw data only; travel times are recomputed on load)."""
        np.savez_compressed(path, req=self.req, loc=self.loc, dur=self.dur, ab=self.ab, depot=self.depot,
                            species=self.species, speed=self.speed, name=self.name, source=self.source or "")

    @classmethod
    def load(cls, path: str | PathLike) -> Instance:
        with np.load(path, allow_pickle=False) as z:
            arrays = {k: z[k] for k in _DATA}
            return cls(**arrays, speed=float(z["speed"]), name=str(z["name"]), source=str(z["source"]) or None)


def load_instance(path: str | PathLike) -> Instance:
    """Instance from an env pickle (.pkl) or our .npz."""
    return Instance.load(path) if Path(path).suffix == ".npz" else Instance.from_pickle(path)
