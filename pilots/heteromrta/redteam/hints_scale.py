import sys, json, glob, random, numpy as np
sys.argv=['x']
exec(open('coalition_lns_pilot.py').read().split("if __name__")[0])
files=sorted(glob.glob('HeteroMRTA/ScaleSet50/env_*.pkl'), key=lambda s:int(s.split('_')[-1].split('.')[0]))
out={}
for f in files:
    env, req, loc, dur, ab, dep = load(f); I=Inst(req, loc, dur, ab, dep)
    m,key=construct(I, random.Random(0)); ms,st=evaluate(I,m,key)
    out[f]=dict(members=[sorted(x) for x in m], start=st.tolist(), ms=ms)
json.dump(out,open('hints_scale50.json','w')); print('construct mean', np.mean([v['ms'] for v in out.values()]).round(3))
