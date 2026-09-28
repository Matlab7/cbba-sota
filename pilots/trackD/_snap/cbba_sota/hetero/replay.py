"""Ground-truth scoring: replay a plan in the HeteroMRTA env (``pre_set_route`` + ``execute_by_route``)."""
from __future__ import annotations

import contextlib
import copy
import io
import multiprocessing as mp
import os
from collections.abc import Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor
from os import PathLike

import numpy as np

from cbba_sota.bench.configs import MAX_TIME
from cbba_sota.bench.heteromrta import TaskEnv, load_env
from cbba_sota.hetero.instance import Instance
from cbba_sota.hetero.plan import Plan

Source = Instance | TaskEnv | str | PathLike
_ONE_THREAD = {k: "1" for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS")}


def succeeded(finished: bool, makespan: float | None) -> bool:
    """The one success definition: every task finished and the makespan (the env's final ``current_time``) below
    ``MAX_TIME``. The env loop only checks the cap before each decision round, so its own ``finished`` flag can be
    set after a round past the cap; that raw flag is reported separately as ``env_finished``."""
    return bool(finished) and makespan is not None and makespan < MAX_TIME


def env_from_instance(inst: Instance) -> TaskEnv:
    """Env holding exactly ``inst``, with the dicts laid out as ``TaskEnv.generate_env`` builds them."""
    env = TaskEnv(per_species_range=(1, 1), species_range=(1, 1), tasks_range=(1, 1), traits_dim=inst.n_traits,
                  seed=0)  # placeholder instance, replaced below
    first = [int(np.flatnonzero(inst.species == s)[0]) for s in range(inst.n_species)]
    abilities, depots = inst.ab[first], inst.depot[first]
    if not (np.array_equal(inst.ab, abilities[inst.species]) and np.array_equal(inst.depot, depots[inst.species])):
        raise ValueError("agents of one species must share traits and depot")
    req = inst.req.astype(np.int64) if np.array_equal(inst.req, np.round(inst.req)) else inst.req.copy()
    task_dic = {j: {"ID": j, "requirements": req[j], "members": [], "cost": [], "location": inst.loc[j].copy(),
                    "feasible_assignment": False, "finished": False, "time_start": 0, "time_finish": 0,
                    "status": req[j], "time": float(inst.dur[j]), "sum_waiting_time": 0, "efficiency": 0,
                    "abandoned_agent": [], "optimized_ability": None, "optimized_species": []}
                for j in range(inst.n_tasks)}
    agent_dic = {}
    for i, s in enumerate(inst.species.tolist()):
        agent_dic[i] = {"ID": i, "species": s, "abilities": abilities[s], "location": depots[s], "route": [-s - 1],
                        "current_task": -s - 1, "contributed": False, "arrival_time": [0.0], "cost": 0.0,
                        "travel_time": 0, "velocity": inst.speed, "next_decision": 0, "depot": depots[s],
                        "travel_dist": 0, "sum_waiting_time": 0, "current_action_index": 0, "decision_step": 0,
                        "task_waiting_ratio": 1, "trajectory": [], "angle": 0, "returned": False,
                        "assigned": False, "pre_set_route": None, "no_choice": False}
    species_dict = {"abilities": abilities.copy(), "number": np.bincount(inst.species).tolist()}
    depot_dic = {}
    for s in range(inst.n_species):
        species_dict[s] = np.flatnonzero(inst.species == s).tolist()
        depot_dic[s] = {"location": depots[s], "members": species_dict[s], "ID": -s - 1}
    env.reset(test_env=(task_dic, agent_dic, depot_dic, species_dict))
    return env


def make_env(source: Source) -> TaskEnv:
    """Fresh env (after ``init_state``) from an env object, an env pickle, or an ``Instance`` (its source pickle
    when it has one, else rebuilt from the arrays)."""
    if isinstance(source, TaskEnv):
        env = copy.deepcopy(source)
    elif isinstance(source, Instance) and source.source:
        env = load_env(source.source)
        if not source.same_data(Instance.from_env(env)):
            raise ValueError(f"{source.name} differs from its source pickle; drop the source to replay the arrays")
    elif isinstance(source, Instance):
        env = env_from_instance(source)
    else:
        env = load_env(source)
    env.init_state()
    return env


def replay_routes(source: Source, routes: Sequence[Sequence[int]]) -> dict:
    """Run 1-based per-agent routes (``Plan.to_env_routes``) through ``execute_by_route``.

    Returns ``makespan`` (env ``current_time``), ``success`` (``succeeded``), ``env_finished`` (the env's raw
    ``finished`` flag), ``awt`` (mean ``sum_waiting_time``, as the RA-L tables) and ``skipped``: planned visits the
    env never executed (members rejected because the task was already covered, or cut off by the cap).
    """
    env = make_env(source)
    if len(routes) != len(env.agent_dic):
        raise ValueError(f"{len(routes)} routes for {len(env.agent_dic)} agents")
    missing = set(range(1, len(env.task_dic) + 1)).difference(j for r in routes for j in r)
    if missing:  # the env would idle below the time cap forever
        raise ValueError(f"tasks without members (1-based): {sorted(missing)}")
    for i, route in enumerate(routes):
        env.pre_set_route([int(j) for j in route], i)
    with contextlib.redirect_stdout(io.StringIO()):
        env.execute_by_route(plot_figure=False)
    env.calculate_waiting_time()
    agents, tasks = env.agent_dic.values(), env.task_dic.values()
    executed = sum(sum(t >= 0 for t in a["route"]) for a in agents)
    makespan = float(env.current_time)
    return {"makespan": makespan, "success": succeeded(all(t["finished"] for t in tasks), makespan),
            "env_finished": bool(env.finished), "awt": float(np.mean([a["sum_waiting_time"] for a in agents])),
            "skipped": sum(map(len, routes)) - executed}


def replay(source: Source, plan: Plan) -> dict:
    return replay_routes(source, plan.to_env_routes())


def replay_batch(jobs: Iterable[tuple[Source, Plan]], workers: int = 8, context: str = "spawn") -> list[dict]:
    """``replay`` over ``(source, plan)`` jobs in single-threaded worker processes, results in job order.

    With the default ``spawn`` context the calling script needs the ``if __name__ == "__main__"`` guard."""
    jobs = list(jobs)
    if workers <= 1 or len(jobs) <= 1:
        return [replay(*job) for job in jobs]
    saved = {k: os.environ.get(k) for k in _ONE_THREAD}
    os.environ.update(_ONE_THREAD)  # inherited by the workers, which start on the first submits
    try:
        with ProcessPoolExecutor(min(workers, len(jobs)), mp_context=mp.get_context(context)) as ex:
            futures = [ex.submit(replay, *job) for job in jobs]
            return [f.result() for f in futures]
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
