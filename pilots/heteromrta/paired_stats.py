import sys, json, glob, numpy as np
sys.argv=['x']
exec(open('coalition_lns_pilot.py').read().split("if __name__")[0])
from concurrent.futures import ProcessPoolExecutor
rl=json.load(open('rl_eval.json'))
files=sorted(glob.glob('HeteroMRTA/RALTestSet/env_*.pkl'))
def one(a):
    f,secs,seed=a
    env, req, loc, dur, ab, dep = load(f); I=Inst(req, loc, dur, ab, dep)
    return f, lns(I, secs, seed)[0]
if __name__=='__main__':
    rng=np.random.default_rng(0)
    for secs in (4.0, 30.0):
        with ProcessPoolExecutor(50) as ex:
            res=dict(ex.map(one,[(f,secs,1) for f in files]))
        l=np.array([res[f] for f in files]); s=np.array([rl[f.replace('HeteroMRTA/','')]['s10'] for f in files]); g=np.array([rl[f.replace('HeteroMRTA/','')]['g'] for f in files])
        r=l/s; boots=[rng.choice(r,len(r)).mean() for _ in range(5000)]
        print(f"LNS {secs:>4}s: mean {l.mean():.3f} | RL(s.10) {s.mean():.3f} | ratio LNS/RL(s.10) {r.mean():.3f} 95%CI [{np.percentile(boots,2.5):.3f},{np.percentile(boots,97.5):.3f}] | LNS better on {(l<s).sum()}/50 | vs RL(g.) {np.mean(l/g):.3f}")
