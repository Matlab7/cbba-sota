import sys, json, glob, random, pickle, io, contextlib, math, time
import numpy as np
sys.argv=['x']
src=open('coalition_lns_pilot.py').read().split("if __name__")[0]
# variant of lns() that also returns the best plan
src=src.replace("    return best[0], it","    return best[0], it, best")
exec(src)
from concurrent.futures import ProcessPoolExecutor
rl=json.load(open('rl_eval.json'))
files=sorted(glob.glob('HeteroMRTA/RALTestSet/env_*.pkl'))
def replay(f, routes):
    env=pickle.load(open(f,'rb')); env.init_state()
    for a,r in enumerate(routes): env.pre_set_route(list(r), a)
    with contextlib.redirect_stdout(io.StringIO()):
        env.execute_by_route('./','x',False)
    _,fin=env.get_episode_reward(100)
    return env.current_time if np.all(fin) else float('nan')
def one(f):
    out={}
    for m in ('sas','taco'):
        sol=pickle.load(open(f.replace('.pkl','/')+m+'.solution','rb'))
        out[m]=replay(f,[r[1:] for r in sol]) if sol is not None else float('nan')
    env, req, loc, dur, ab, dep = load(f); I=Inst(req, loc, dur, ab, dep)
    ms,_,best=lns(I,4.0,1)
    _,members,key=best
    routes=[[j+1 for j in sorted([t for t in range(I.T) if a in members[t]], key=lambda t:key[t])] for a in range(I.A)]
    out['lns_eval']=ms; out['lns_replay']=replay(f,routes)
    return f,out
if __name__=='__main__':
    with ProcessPoolExecutor(50) as ex: res=dict(ex.map(one,files))
    s10=np.array([rl[f.replace('HeteroMRTA/','')]['s10'] for f in files])
    for k in ('sas','taco','lns_eval','lns_replay'):
        v=np.array([res[f][k] for f in files]); ok=np.isfinite(v)
        print(f"{k:10s} mean makespan {np.nanmean(v):.3f} success {ok.mean():.2f}  ratio to RL(s.10) {np.nanmean(v/s10):.3f}")
    d=np.array([res[f]['lns_replay']-res[f]['lns_eval'] for f in files])
    print("replay - evaluator: max abs diff", np.nanmax(np.abs(d)).round(4), " nan replays:", int(np.isnan(d).sum()))
