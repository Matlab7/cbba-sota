import sys, json, pickle, io, contextlib, numpy as np
sys.path.insert(0,'HeteroMRTA')
from env.task_env import TaskEnv  # noqa
from concurrent.futures import ProcessPoolExecutor
def replay(a):
    f, routes = a
    env=pickle.load(open(f,'rb')); env.init_state()
    for i,r in enumerate(routes): env.pre_set_route(list(r), i)
    with contextlib.redirect_stdout(io.StringIO()):
        env.execute_by_route('./','x',False)
    _,fin=env.get_episode_reward(100)
    return env.current_time if np.all(fin) else float('nan')
if __name__=='__main__':
    rl=json.load(open('rl_eval.json'))
    for fn in sys.argv[1:]:
        res=json.load(open(fn))
        with ProcessPoolExecutor(25) as ex:
            rp=list(ex.map(replay, [(r[0], r[6]) for r in res]))
        ev=np.array([r[1] for r in res]); rp=np.array(rp)
        s10=np.array([rl[r[0].replace('HeteroMRTA/','')]['s10'] for r in res])
        print(fn, 'replay mean', np.nanmean(rp).round(3), 'success', np.isfinite(rp).mean(), 'max|replay-eval|', np.nanmax(np.abs(rp-ev)).round(4), 'ratio to RL(s.10)', np.nanmean(rp/s10).round(3))
