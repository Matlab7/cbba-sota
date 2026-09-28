import sys, json, glob, random, numpy as np
sys.argv=['x']
exec(open('coalition_lns_pilot.py').read().split("if __name__")[0])
from concurrent.futures import ProcessPoolExecutor
rl=json.load(open('rl_eval.json'))
files=sorted(glob.glob('HeteroMRTA/RALTestSet/env_*.pkl'))
def one(f):
    env, req, loc, dur, ab, dep = load(f); I=Inst(req, loc, dur, ab, dep)
    out=[]
    for k in range(10):
        m,key=construct(I, random.Random(k)); out.append(evaluate(I,m,key)[0])
    return f, out[0], min(out)
if __name__=='__main__':
    with ProcessPoolExecutor(50) as ex: res=list(ex.map(one,files))
    c1=np.array([r[1] for r in res]); c10=np.array([r[2] for r in res]); s=np.array([rl[f.replace('HeteroMRTA/','')]['s10'] for f in files])
    print(f"construct x1: {c1.mean():.3f} (ratio to RL(s.10) {np.mean(c1/s):.3f}); construct best-of-10: {c10.mean():.3f} (ratio {np.mean(c10/s):.3f}); RL(s.10) {s.mean():.3f}")
