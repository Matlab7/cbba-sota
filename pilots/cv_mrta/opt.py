import numpy as np, pandas as pd, sys
from multiprocessing import Pool
S=[(0,0),(0,6),(0,12),(0,18)]
sc=pd.read_csv('CV_MRTA_Benchmark/scenarios/known_visit_g19_t10_n500.csv',comment='#')
def md(a,b): return abs(a[0]-b[0])+abs(a[1]-b[1])
def solve(row):
    T=[(int(row[f'target{i}_x']),int(row[f'target{i}_y'])) for i in range(1,11)]
    n=10; F=1<<n; INF=10**9
    D=np.array([[md(a,b) for b in T] for a in T])
    cost=np.full((4,F),INF,dtype=np.int64)
    for r,s in enumerate(S):
        dp=np.full((F,n),INF,dtype=np.int64)
        for j in range(n): dp[1<<j,j]=md(s,T[j])
        for m in range(1,F):
            for j in range(n):
                v=dp[m,j]
                if v>=INF or not (m>>j)&1: continue
                for k in range(n):
                    if (m>>k)&1: continue
                    nm=m|(1<<k); nv=v+D[j,k]
                    if nv<dp[nm,k]: dp[nm,k]=nv
        cost[r]=dp.min(axis=1); cost[r,0]=0
    # partition DP over robots
    full=F-1
    bestmax=np.full(F,INF,dtype=np.int64); bestsum=np.full(F,INF,dtype=np.int64)
    bestmax[:]=cost[0]; bestsum[:]=cost[0]
    for r in range(1,4):
        nmx=np.full(F,INF,dtype=np.int64); nsm=np.full(F,INF,dtype=np.int64)
        for m in range(F):
            sub=m
            while True:
                a=bestmax[m^sub]; c=cost[r,sub]
                v=max(a,c)
                if v<nmx[m]: nmx[m]=v
                v2=bestsum[m^sub]+c
                if v2<nsm[m]: nsm[m]=v2
                if sub==0: break
                sub=(sub-1)&m
        bestmax,bestsum=nmx,nsm
    return int(row['trial_id']),int(bestmax[full]),int(bestsum[full])
if __name__=='__main__':
    rows=[r for _,r in sc.iterrows()]
    with Pool(48) as p: res=p.map(solve,rows)
    pd.DataFrame(res,columns=['trial_id','opt_minmax','opt_minsum']).to_csv('opt.csv',index=False)
    print('done',len(res))
