"""Glue for the HeteroMRTA env: import path, pickle load/save and instance generation."""
from __future__ import annotations

import io
import pickle
import sys
from pathlib import Path

import numpy as np

from cbba_sota.bench.configs import HETEROMRTA_DIR, TRAIT_DIM, Setting

if str(HETEROMRTA_DIR) not in sys.path:
    sys.path.insert(0, str(HETEROMRTA_DIR))

from env.task_env import TaskEnv

__all__ = ["TaskEnv", "dumps_env", "generate_env", "load_env", "loads_env", "save_env", "slim"]


class _EnvUnpickler(pickle.Unpickler):
    """The shipped RALTestSet pickles were dumped from task_env.py's ``__main__``."""

    def find_class(self, module: str, name: str):
        if module == "__main__" and name == "TaskEnv":
            return TaskEnv
        return super().find_class(module, name)


def loads_env(data: bytes) -> TaskEnv:
    return _EnvUnpickler(io.BytesIO(data)).load()


def load_env(path: str | Path) -> TaskEnv:
    return loads_env(Path(path).read_bytes())


def dumps_env(env: TaskEnv) -> bytes:
    if type(env) is not TaskEnv:
        raise TypeError(f"expected env.task_env.TaskEnv, got {type(env)}")
    return pickle.dumps(env, protocol=pickle.HIGHEST_PROTOCOL)


def save_env(env: TaskEnv, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(dumps_env(env))
    tmp.replace(path)


class _SingleSkillEnv(TaskEnv):
    """SA species: one-hot trait rows (the commented-out line in ``TaskEnv.generate_agent``)."""

    def generate_agent(self, species_num):
        return np.diag(np.ones(self.traits_dim))


def _pad_traits(env: TaskEnv, dim: int) -> None:
    """Zero-pad the trait dimension from ``env.traits_dim`` to ``dim`` (unused skills are zero)."""
    pad = dim - env.traits_dim
    abilities = np.pad(env.species_dict["abilities"], ((0, 0), (0, pad)))
    env.species_dict["abilities"] = abilities
    for agent in env.agent_dic.values():
        agent["abilities"] = abilities[agent["species"]]
    for task in env.task_dic.values():
        task["requirements"] = np.pad(task["requirements"], (0, pad))
        task["status"] = task["requirements"]
    env.traits_dim = dim


def slim(env: TaskEnv) -> TaskEnv:
    """Drop state no code path reads: the RNG and the per-species distance dicts (O(species * tasks^2) in the
    pickle, ~10^6 entries at 500 tasks); ``env.generate_distance_matrix()`` rebuilds the latter."""
    env.rng = None
    env.species_distance_matrix, env.species_neighbor_matrix = {}, {}
    return env


def generate_env(setting: Setting, seed: int) -> TaskEnv:
    """Instance of ``setting`` from the repo generator with ``seed``; a plain ``TaskEnv`` ready to pickle."""
    cls = _SingleSkillEnv if setting.single_skill else TaskEnv
    env = cls(**setting.generator_kwargs(), seed=seed)
    if env.traits_dim < TRAIT_DIM:
        _pad_traits(env, TRAIT_DIM)
    env.__class__ = TaskEnv  # drop the generator override so pickles only reference env.task_env
    return slim(env)
