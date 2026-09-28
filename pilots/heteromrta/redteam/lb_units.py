import sys, glob, re, json, numpy as np
sys.argv=['x']
exec(open('coalition_lns_pilot.py').read().split("if __name__")[0])
files=sorted(glob.glob('HeteroMRTA/RALTestSet/env_*.pkl'))
out=[]
for f in files:
    env, req, loc, dur, ab, dep = load(f); I=Inst(req, loc, dur, ab, dep)
    # simple LB: each task needs some agent to come from a depot, do it and return
    lb1=max(min(I.dt[i,j]+I.dur[j]+I.dt[i,j] for i in range(I.A) if (np.minimum(I.req[j],I.ab[i]).sum()>0)) for j in range(I.T))
    # workload LB per skill: total required skill-duration / number of agents with skill
    lb2=0
    for s in range(ab.shape[1]):
        n=(ab[:,s]>0).sum(); w=sum(I.dur[j]*req[j][s] for j in range(I.T))
        if n: lb2=max(lb2,w/n)
    t=open(f.replace('.pkl','/results.yaml')).read()
    c=float(re.search(r'timeCost: ([-\d.]+)',t).group(1))/100
    out.append((lb1,lb2,c))
o=np.array(out); print('A',I.A,'T',I.T,'skills',ab.shape[1], 'req max', req.max(), 'agent ab per row', ab.sum(1)[:6])
print('LB travel mean',o[:,0].mean().round(2),'LB work mean',o[:,1].mean().round(2),'CTAS mean',o[:,2].mean().round(2),'#CTAS<max(LB)', (o[:,2]<np.maximum(o[:,0],o[:,1])-1e-6).sum())
