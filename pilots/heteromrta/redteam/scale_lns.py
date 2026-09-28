import sys, json, glob, numpy as np
secs_list=[float(x) for x in sys.argv[1:]]
sys.argv=['x']
exec(open('coalition_lns_pilot.py').read().split("if __name__")[0].replace('    return best[0], it','    return best[0], it, best'))
from concurrent.futures import ProcessPoolExecutor
rl=json.load(open('rl_eval_scale50.json'))
files=sorted(glob.glob('HeteroMRTA/ScaleSet50/env_*.pkl'), key=lambda s:int(s.split('_')[-1].split('.')[0]))
def one(a):
    f,secs,seed=a
    env, req, loc, dur, ab, dep = load(f); I=Inst(req, loc, dur, ab, dep)
    ms,it,best=lns(I, secs, seed)
    _,members,key=best
    routes=[[j+1 for j in sorted([t for t in range(I.T) if a in members[t]], key=lambda t:key[t])] for a in range(I.A)]
    return f,(ms,it,routes)
if __name__=='__main__':
    src=open('coalition_lns_pilot.py').read()
    rng=np.random.default_rng(0)
    s=np.array([rl[f.replace('HeteroMRTA/','')]['s10'] for f in files]); g=np.array([rl[f.replace('HeteroMRTA/','')]['g'] for f in files])
    for secs in secs_list:
        with ProcessPoolExecutor(30) as ex:
            res=dict(ex.map(one,[(f,secs,3) for f in files]))
        json.dump({f:[res[f][0],res[f][2]] for f in files}, open(f'scale50_lns_{int(secs)}s.json','w'))
        l=np.array([res[f][0] for f in files]); r=l/s
        b=[rng.choice(r,len(r)).mean() for _ in range(5000)]
        print(f"LNS {secs}s mean {l.mean():.3f} iters {np.mean([res[f][1] for f in files]):.0f} | RL(s.10) {s.mean():.3f} RL(g.) {g.mean():.3f} | LNS/RL(s.10) {r.mean():.3f} CI[{np.percentile(b,2.5):.3f},{np.percentile(b,97.5):.3f}] wins {(l<s).sum()}/{len(l)}")
