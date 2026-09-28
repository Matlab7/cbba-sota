import sys, glob, io, contextlib, pickle, numpy as np
DIR=sys.argv[1]; SECS=float(sys.argv[2]); SEED=int(sys.argv[3])
sys.argv=['x']
MARK = "if __name_" + "_"
src=open('coalition_lns_pilot.py').read().split(MARK)[0].replace("    return best[0], it","    return best[0], it, best")
exec(src)
from concurrent.futures import ProcessPoolExecutor
def replay(f, routes):
    env = pickle.load(open(f, 'rb')); env.init_state()
    for a, r in enumerate(routes): env.pre_set_route(list(r), a)
    with contextlib.redirect_stdout(io.StringIO()):
        env.execute_by_route('./', 'x', False)
    _, fin = env.get_episode_reward(100)
    return env.current_time if np.all(fin) else float('nan')
def one(f):
    env, req, loc, dur, ab, dep = load(f); I = Inst(req, loc, dur, ab, dep)
    ms, it, best = lns(I, SECS, SEED)
    _, members, key = best
    routes = [[j + 1 for j in sorted([t for t in range(I.T) if a_ in members[t]], key=lambda t: key[t])] for a_ in range(I.A)]
    return ms, replay(f, routes), it
if __name__=='__main__':
    files=sorted(glob.glob(f'{DIR}/env_*.pkl'), key=lambda s:int(s.split('_')[-1].split('.')[0]))
    with ProcessPoolExecutor(50) as ex: res=list(ex.map(one,files))
    ev=np.array([r[0] for r in res]); rp=np.array([r[1] for r in res]); d=rp-ev; print("mean iters", np.mean([r[2] for r in res]))
    np.save(f'{DIR}/lns_{int(SECS)}s_seed{SEED}.npy', rp)
    print(f"secs={SECS} seed={SEED}: mean eval {ev.mean():.3f} mean replay {np.nanmean(rp):.3f} | n(|d|>1e-3)={int((abs(d)>1e-3).sum())}/50, replay>eval {int((d>1e-3).sum())}, replay<eval {int((d<-1e-3).sum())}, max {np.nanmax(d):.3f} min {np.nanmin(d):.3f}")
