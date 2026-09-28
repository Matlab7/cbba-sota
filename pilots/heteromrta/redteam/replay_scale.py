import sys, json, numpy as np
sys.argv=[sys.argv[0]]+sys.argv[1:]
exec(open('replay_cpsat.py').read().split("if __name__")[0])
from concurrent.futures import ProcessPoolExecutor
if __name__=='__main__':
    for fn in sys.argv[1:]:
        d=json.load(open(fn)); fs=list(d)
        with ProcessPoolExecutor(30) as ex: rp=np.array(list(ex.map(replay,[(f,d[f][1]) for f in fs])))
        ev=np.array([d[f][0] for f in fs])
        print(fn,'replay mean',np.nanmean(rp).round(3),'success',np.isfinite(rp).mean(),'max|replay-eval|',np.nanmax(np.abs(rp-ev)).round(4))
