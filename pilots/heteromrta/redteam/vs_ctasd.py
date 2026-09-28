import sys, json, glob, re, numpy as np
sys.argv=['x']
exec(open('coalition_lns_pilot.py').read().split("if __name__")[0])
from concurrent.futures import ProcessPoolExecutor
rl=json.load(open('rl_eval.json'))
files=sorted(glob.glob('HeteroMRTA/RALTestSet/env_*.pkl'))
def ctasd(f):
    t=open(f.replace('.pkl','/results.yaml')).read()
    return float(re.search(r'timeCost: ([\d.]+)',t).group(1))/100
def one(a):
    f,secs,seed=a
    env, req, loc, dur, ab, dep = load(f); I=Inst(req, loc, dur, ab, dep)
    ms,it=lns(I, secs, seed)
    return f, (ms,it)
if __name__=='__main__':
    rng=np.random.default_rng(0)
    c=np.array([ctasd(f) for f in files]); s=np.array([rl[f.replace('HeteroMRTA/','')]['s10'] for f in files])
    print(f"CTAS-D(600s) mean {c.mean():.3f}; RL(s.10) {s.mean():.3f}; CTAS-D/RL {np.mean(c/s):.3f}; CTAS-D better than RL on {(c<s).sum()}/50")
    for secs in [float(x) for x in sys.argv[1:]] or (4.0,30.0):
        with ProcessPoolExecutor(25) as ex:
            res=dict(ex.map(one,[(f,secs,3) for f in files]))
        l=np.array([res[f][0] for f in files]); its=np.mean([res[f][1] for f in files]); r=l/c
        b=[rng.choice(r,len(r)).mean() for _ in range(5000)]
        print(f"LNS {secs}s mean {l.mean():.3f} iters {its:.0f} | LNS/CTAS-D {r.mean():.3f} CI[{np.percentile(b,2.5):.3f},{np.percentile(b,97.5):.3f}] wins {(l<c-1e-6).sum()}/50 | LNS/RL {np.mean(l/s):.3f}")
