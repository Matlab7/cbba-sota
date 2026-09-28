import sys as _s; _SECS=float(_s.argv[1]) if len(_s.argv)>1 else 4.0
import sys, json, glob, re, numpy as np
sys.argv=['x']
exec(open('coalition_lns_pilot.py').read().split("if __name__")[0])
from concurrent.futures import ProcessPoolExecutor
rl=json.load(open('rl_eval.json'))
files=[f'HeteroMRTA/RALTestSet/env_{i}.pkl' for i in range(50)]
def one(a):
    f,secs,seed=a
    env, req, loc, dur, ab, dep = load(f); I=Inst(req, loc, dur, ab, dep)
    return f, lns(I, secs, seed)[0]
if __name__=='__main__':
    secs=_SECS
    with ProcessPoolExecutor(50) as ex:
        res=dict(ex.map(one,[(f,secs,7) for f in files]))
    l=np.array([res[f] for f in files]); s=np.array([rl[f.replace('HeteroMRTA/','')]['s10'] for f in files])
    c=np.array([float(re.search(r'timeCost: ([\d.]+)',open(f.replace('.pkl','/results.yaml')).read()).group(1))/100 for f in files])
    print(f"LNS{secs}s(seed7) {l.mean():.3f} RL(s10) {s.mean():.3f} CTAS-D600s {c.mean():.3f}")
    print(f"LNS/RL {np.mean(l/s):.3f}  LNS/CTAS-D {np.mean(l/c):.3f} wins vs CTAS-D {(l<c).sum()}/50  CTAS-D/RL {np.mean(c/s):.3f}")
