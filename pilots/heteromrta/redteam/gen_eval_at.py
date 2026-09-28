"""Red-team check: regenerate an MA-AT test family with the repo generator (requirements 0..2, binary multi-skill
agents, durations [0,5]) and evaluate the released checkpoint RL(g.)/RL(s.10). Usage: gen_eval_at.py OUTDIR KN_PER_SPECIES SPECIES TASKS N SEED0"""
import sys, os, pickle, time, json, glob
import numpy as np, torch
from env.task_env import TaskEnv
from attention import AttentionNet
from parameters import EnvParams, TrainParams
out, per, sp, T, N, seed0 = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6])
os.makedirs(out, exist_ok=True)
EnvParams.TRAIT_DIM = 5
TrainParams.EMBEDDING_DIM = 128
TrainParams.AGENT_INPUT_DIM = 6 + EnvParams.TRAIT_DIM
TrainParams.TASK_INPUT_DIM = 5 + 2 * EnvParams.TRAIT_DIM
EnvParams.MAX_TIME = 200
from worker import Worker

def gen(i):
    env = TaskEnv(per_species_range=(per, per), species_range=(sp, sp), tasks_range=(T, T), traits_dim=5,
                  decision_dim=10, max_task_size=3, duration_scale=5, seed=seed0 + i)
    env.rng = None
    f = f"{out}/env_{i}.pkl"
    pickle.dump(env, open(f, "wb"))
    return f

def run(f):
    torch.set_num_threads(1)
    import random
    random.seed(0); torch.manual_seed(0); np.random.seed(0)
    net = AttentionNet(TrainParams.AGENT_INPUT_DIM, TrainParams.TASK_INPUT_DIM, TrainParams.EMBEDDING_DIM)
    import numpy.core.multiarray as _ma
    safe = [_ma.scalar, np.dtype] + [getattr(np.dtypes, n) for n in dir(np.dtypes) if n.endswith("DType")]
    with torch.serialization.safe_globals(safe):
        ck = torch.load("model/save/checkpoint.pth", map_location="cpu", weights_only=True)
    net.load_state_dict(ck["best_model"])
    w = Worker(0, net, net, 0, torch.device("cpu"))
    o = {}
    for name, sampling, n in (("g", False, 1), ("s10", True, 10)):
        best = np.inf; t0 = time.time()
        for _ in range(n):
            env = pickle.load(open(f, "rb")); env.init_state(); w.env = env
            _, _, res = w.run_episode(False, sampling, False)
            if res["success_rate"][0] >= 1:
                best = min(best, float(res["makespan"][0]))
        o[name] = best; o[name + "_t"] = time.time() - t0
    return f, o

if __name__ == "__main__":
    import multiprocessing as mp
    files = [gen(i) for i in range(N)]
    with mp.get_context("fork").Pool(min(50, N)) as p:
        res = p.map(run, files)
    json.dump({f: o for f, o in res}, open(f"{out}/rl.json", "w"), indent=1)
    g = np.array([o["g"] for _, o in res]); s = np.array([o["s10"] for _, o in res])
    print(f"{out}: RL(g.) {np.mean(g[np.isfinite(g)]):.3f} succ {np.isfinite(g).mean():.2f} | RL(s.10) {np.mean(s[np.isfinite(s)]):.3f} succ {np.isfinite(s).mean():.2f} | t_s10 {np.mean([o['s10_t'] for _, o in res]):.1f}s")
