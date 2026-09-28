import sys, glob, json, re, numpy as np
sys.argv=['x']
exec(open('coalition_lns_pilot.py').read().split("if __name__")[0])
files=sorted(glob.glob('HeteroMRTA/RALTestSet/env_*.pkl'))
out={}
for f in files:
    env, req, loc, dur, ab, dep = load(f)
    out[f]=dict(req=req.tolist(), loc=loc.tolist(), dur=dur.tolist(), ab=ab.tolist(), dep=dep.tolist())
json.dump(out, open('instances.json','w'))
print(len(out))
