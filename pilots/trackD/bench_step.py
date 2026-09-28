"""Per-decision cost of running the released policy online: observation build (env, Python) and forward pass, for
the static env, the dynamic env with zeroed unreleased rows, and a compact observation (unreleased rows deleted).

Usage: .venv/bin/python pilots/trackD/bench_step.py [--device cpu] [--max_dec 400]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dyn_env import Scenario, make_dyn_env  # noqa: E402

import torch  # noqa: E402

from cbba_sota.bench.configs import get  # noqa: E402
from cbba_sota.bench.heteromrta import load_env  # noqa: E402
from cbba_sota.solvers import rl  # noqa: E402


def run(env, net, device, max_dec, dyn):
    env.init_state()
    tasks = env.task_dic.values()
    t_obs, t_full, t_cmp, n, hidden = [], [], [], 0, []
    while not env.finished and env.current_time < 200 and n < max_dec:
        released, env.current_time = env.next_decision()
        for aid in released[0] + released[1]:
            a = time.perf_counter()
            to, ao, m = env.agent_observe(aid, False)
            t_obs.append(time.perf_counter() - a)
            if m[0, 1:].all():
                if not all(t["feasible_assignment"] for t in tasks):
                    env.agent_dic[aid]["no_choice"] = True
                    continue
                if env.agent_dic[aid]["current_task"] < 0:
                    continue
            idx = torch.tensor([[[aid]]], device=device)
            with torch.no_grad():
                a = time.perf_counter()
                x = [torch.as_tensor(v, dtype=torch.float32, device=device) for v in (to, ao, m)]
                p, _ = net(*x, idx)
                act = int(torch.argmax(p, 1).item())
                t_full.append(time.perf_counter() - a)
                if dyn:
                    rel = env.released_mask()
                    hidden.append(int((~rel).sum()))
                    keep = np.concatenate([[True], rel])
                    a = time.perf_counter()
                    x = [torch.as_tensor(v[:, keep], dtype=torch.float32, device=device) for v in (to, m)]
                    pc, _ = net(x[0], torch.as_tensor(ao, dtype=torch.float32, device=device), x[1], idx)
                    int(torch.argmax(pc, 1).item())
                    t_cmp.append(time.perf_counter() - a)
            env.agent_step(aid, act, 0)
            n += 1
        env.finished = env.check_finished()
    ms = lambda v: 1e3 * float(np.mean(v)) if v else float("nan")  # noqa: E731
    return n, ms(t_obs), ms(t_full), ms(t_cmp), (float(np.mean(hidden)) if hidden else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max_dec", type=int, default=400)
    ap.add_argument("--settings", nargs="+", default=["MA-AT-25-5-50", "MA-AT-50-5-200", "MA-AT-150-5-500"])
    args = ap.parse_args()
    torch.set_num_threads(1)
    net = rl.load_policy(args.device)
    print(f"device {args.device}, torch threads {torch.get_num_threads()}")
    for s in args.settings:
        path = get(s).instance_path("dev", 0)
        st = get(s)
        for label, sc in (("static TaskEnv", None), ("dyn static", Scenario()),
                          ("dyn poisson .5 + N3/causal", Scenario(release="poisson", dod=0.5, horizon=20.0,
                                                                   dur_sigma=0.3, p_delay=0.05, max_delay=10,
                                                                   obs="causal"))):
            env = load_env(path) if sc is None else make_dyn_env(path, sc, inst_key=st.seed("dev", 0))
            n, to, tf, tc, hid = run(env, net, args.device, args.max_dec, sc is not None and sc.release != "none")
            print(f"{s:16s} {label:28s} decisions {n:4d}  obs {to:6.2f} ms  net(full) {tf:6.2f} ms  "
                  f"net(compact) {tc:6.2f} ms  mean hidden tasks {hid:5.1f}")


if __name__ == "__main__":
    main()
