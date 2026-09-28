"""Red-team: LNS pilot on a regenerated family; replay each plan in the HeteroMRTA env; compare to RL(s.10).
usage: rt_lns_eval.py DIR SECS NSEEDS"""
import sys, json, glob, io, contextlib, pickle, re
import numpy as np
DIR, SECS, NS = sys.argv[1], float(sys.argv[2]), int(sys.argv[3])
sys.argv = ['x']
src = open('coalition_lns_pilot.py').read().split("if __name__")[0].replace("    return best[0], it", "    return best[0], it, best")
exec(src)
from concurrent.futures import ProcessPoolExecutor
def replay(f, routes):
    env = pickle.load(open(f, 'rb')); env.init_state()
    for a, r in enumerate(routes): env.pre_set_route(list(r), a)
    with contextlib.redirect_stdout(io.StringIO()):
        env.execute_by_route('./', 'x', False)
    _, fin = env.get_episode_reward(100)
    return env.current_time if np.all(fin) else float('nan')
def greedy(f):
    env = pickle.load(open(f, 'rb')); env.init_state()
    with contextlib.redirect_stdout(io.StringIO()):
        env.execute_greedy_action('./', 'greedy', False)
    _, fin = env.get_episode_reward(100)
    return env.current_time if np.all(fin) else float('nan')
def one(a):
    f, seed = a
    env, req, loc, dur, ab, dep = load(f); I = Inst(req, loc, dur, ab, dep)
    ms, it, best = lns(I, SECS, seed)
    _, members, key = best
    routes = [[j + 1 for j in sorted([t for t in range(I.T) if a_ in members[t]], key=lambda t: key[t])] for a_ in range(I.A)]
    return f, seed, ms, replay(f, routes), it
if __name__ == '__main__':
    rl = json.load(open(f'{DIR}/rl.json'))
    files = sorted(glob.glob(f'{DIR}/env_*.pkl'), key=lambda s: int(s.split('_')[-1].split('.')[0]))
    jobs = [(f, s) for f in files for s in range(NS)]
    with ProcessPoolExecutor(90) as ex:
        res = list(ex.map(one, jobs))
        gr = dict(zip(files, ex.map(greedy, files)))
    key = lambda f: '../' + f.split('/', 0)[0] if False else f
    s10 = np.array([rl[[k for k in rl if k.endswith(f.split('/')[-2] + '/' + f.split('/')[-1])][0]]['s10'] for f in files])
    ev = {(f, s): (m, r, it) for f, s, m, r, it in res}
    d = np.array([abs(ev[(f, s)][1] - ev[(f, s)][0]) for f in files for s in range(NS)])
    print(f"{DIR} secs={SECS} seeds={NS} | replay-eval max|diff| {np.nanmax(d):.4f} nan-replays {int(np.isnan(d).sum())} | mean iters {np.mean([v[2] for v in ev.values()]):.0f}")
    single = np.array([ev[(f, 0)][1] for f in files])
    bestk = np.array([min(ev[(f, s)][1] for s in range(NS)) for f in files])
    g = np.array([gr[f] for f in files])
    rng = np.random.default_rng(0)
    for name, v in (("LNS seed0", single), (f"LNS best-of-{NS}", bestk)):
        r = v / s10; b = [rng.choice(r, len(r)).mean() for _ in range(4000)]
        print(f"  {name:14s} mean {np.nanmean(v):.3f} | RL(s.10) {s10.mean():.3f} | ratio {r.mean():.3f} CI[{np.percentile(b,2.5):.3f},{np.percentile(b,97.5):.3f}] | wins {(v < s10).sum()}/{len(v)}")
    print(f"  env Greedy mean {np.nanmean(g):.3f} (succ {np.isfinite(g).mean():.2f}); best-of-{NS} / seed0 ratio {np.mean(bestk/single):.4f}")
