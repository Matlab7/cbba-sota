"""Evaluate the released HeteroMRTA checkpoint on the shipped RALTestSet: RL(g.) and RL(s.10)."""
import glob
import pickle
import sys
import time

import numpy as np
import torch

from attention import AttentionNet
from env.task_env import TaskEnv  # noqa: F401  (pickles reference __main__.TaskEnv)
from parameters import EnvParams, TrainParams

EnvParams.TASKS_RANGE = (20, 20)
EnvParams.SPECIES_RANGE = (5, 5)
EnvParams.SPECIES_AGENTS_RANGE = (3, 3)
EnvParams.MAX_TIME = 200
EnvParams.TRAIT_DIM = 5
TrainParams.EMBEDDING_DIM = 128
TrainParams.AGENT_INPUT_DIM = 6 + EnvParams.TRAIT_DIM
TrainParams.TASK_INPUT_DIM = 5 + 2 * EnvParams.TRAIT_DIM

from worker import Worker  # noqa: E402


def run(f):
    torch.set_num_threads(1)
    import random
    random.seed(0)
    torch.manual_seed(0)
    np.random.seed(0)
    dev = torch.device("cpu")
    net = AttentionNet(TrainParams.AGENT_INPUT_DIM, TrainParams.TASK_INPUT_DIM, TrainParams.EMBEDDING_DIM).to(dev)
    import numpy.core.multiarray as _ma
    safe = [_ma.scalar, np.dtype] + [getattr(np.dtypes, n) for n in dir(np.dtypes) if n.endswith("DType")]
    with torch.serialization.safe_globals(safe):
        ck = torch.load("model/save/checkpoint.pth", map_location="cpu", weights_only=True)
    net.load_state_dict(ck["best_model"])
    w = Worker(0, net, net, 0, dev)
    out = {}
    for name, sampling, n in (("g", False, 1), ("s10", True, 10)):
        best = np.inf
        t0 = time.time()
        for _ in range(n):
            env = pickle.load(open(f, "rb"))
            env.init_state()
            w.env = env
            _, _, res = w.run_episode(False, sampling, False)
            if res["success_rate"][0] >= 1:
                best = min(best, float(res["makespan"][0]))
        out[name] = best
        out[name + "_t"] = time.time() - t0
    return f, out


if __name__ == "__main__":
    import multiprocessing as mp
    files = sorted(glob.glob("RALTestSet/env_*.pkl"), key=lambda s: int(s.split("_")[-1].split(".")[0]))
    with mp.get_context("fork").Pool(50) as p:
        res = p.map(run, files)
    import json
    json.dump({f: o for f, o in res}, open("/tmp/claude-0/-home-jovyan-dev-cbja/c781ca93-ceaa-4220-85a3-510892818ead/scratchpad/rl_eval.json", "w"), indent=1)
    g = np.array([o["g"] for _, o in res])
    s = np.array([o["s10"] for _, o in res])
    print(f"RL(g.) mean makespan {np.mean(g[np.isfinite(g)]):.3f} success {np.isfinite(g).mean():.2f}; "
          f"RL(s.10) mean {np.mean(s[np.isfinite(s)]):.3f} success {np.isfinite(s).mean():.2f}; "
          f"time s10 {np.mean([o['s10_t'] for _, o in res]):.2f}s")
