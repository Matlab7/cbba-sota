import sys, json, math, time, numpy as np
from concurrent.futures import ProcessPoolExecutor
from ortools.sat.python import cp_model
V = 0.2
SC = 1000  # time scale

def evaluate(tt, dt, dur, A, T, members, key):
    order = sorted([t for t in range(T) if members[t]], key=lambda j: key[j])
    free_t = np.zeros(A); pos = [None]*A
    for j in order:
        s = max(free_t[i] + (dt[i, j] if pos[i] is None else tt[pos[i], j]) for i in members[j])
        f = s + dur[j]
        for i in members[j]:
            free_t[i] = f; pos[i] = j
    return max([free_t[i] + dt[i, pos[i]] for i in range(A) if pos[i] is not None] + [0.0])

HINTS=json.load(open('hints_scale50.json'))
def solve(args):
    name, d, secs, workers = args
    req = np.array(d['req']); loc = np.array(d['loc']); dur = np.array(d['dur']); ab = np.array(d['ab']); dep = np.array(d['dep'])
    T, A, S = len(req), len(ab), req.shape[1]
    tt = np.linalg.norm(loc[:, None]-loc[None], axis=2)/V
    dt = np.linalg.norm(dep[:, None]-loc[None], axis=2)/V
    ci = lambda x: int(math.ceil(x*SC))
    H = ci(dt.max()*2 + dur.sum() + tt.max()*T)
    m = cp_model.CpModel()
    s = [m.NewIntVar(0, H, f's{j}') for j in range(T)]
    ms = m.NewIntVar(0, H, 'ms')
    x = {}
    for a in range(A):
        arcs = []
        end = m.NewIntVar(0, H, f'e{a}')
        m.Add(ms >= end)
        # node 0 = depot, node j+1 = task j
        idle = m.NewBoolVar(f'idle{a}'); arcs.append((0, 0, idle)); m.Add(end == 0).OnlyEnforceIf(idle)
        for j in range(T):
            xa = m.NewBoolVar(f'x{a}_{j}'); x[a, j] = xa
            arcs.append((j+1, j+1, xa.Not()))
            if np.minimum(req[j], ab[a]).sum() == 0:
                m.Add(xa == 0)
            l = m.NewBoolVar(''); arcs.append((0, j+1, l)); m.Add(s[j] >= ci(dt[a, j])).OnlyEnforceIf(l)
            l = m.NewBoolVar(''); arcs.append((j+1, 0, l)); m.Add(end >= s[j] + ci(dur[j]) + ci(dt[a, j])).OnlyEnforceIf(l)
            for k in range(T):
                if k == j: continue
                l = m.NewBoolVar(''); arcs.append((j+1, k+1, l))
                m.Add(s[k] >= s[j] + ci(dur[j]) + ci(tt[j, k])).OnlyEnforceIf(l)
        m.AddCircuit(arcs)
    for j in range(T):
        for k in range(S):
            if req[j][k] > 0:
                m.Add(sum(int(ab[a][k]) * x[a, j] for a in range(A) if ab[a][k] > 0) >= int(req[j][k]))
    h=HINTS.get(name)
    if h:
        for (a,j),v in x.items(): m.AddHint(v, int(a in h['members'][j]))
        for j in range(T): m.AddHint(s[j], int(round(h['start'][j]*SC)))
    m.Minimize(ms)
    sol = cp_model.CpSolver(); sol.parameters.max_time_in_seconds = secs; sol.parameters.num_workers = workers
    t0 = time.time(); st = sol.Solve(m); el = time.time()-t0
    if st not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return name, float('nan'), float('nan'), float('nan'), el, sol.StatusName(st), None
    members = [set(a for a in range(A) if sol.Value(x[a, j])) for j in range(T)]
    key = [sol.Value(s[j]) + 1e-6*j for j in range(T)]
    exact = evaluate(tt, dt, dur, A, T, members, key)
    routes=[[j+1 for j in sorted([t for t in range(T) if a in members[t]], key=lambda t:key[t])] for a in range(A)]
    return name, exact, sol.ObjectiveValue()/SC, sol.BestObjectiveBound()/SC, el, sol.StatusName(st), routes

if __name__ == '__main__':
    secs = float(sys.argv[1]); workers = int(sys.argv[2]); par = int(sys.argv[3])
    inst = json.load(open('instances_scale50.json'))
    jobs = [(k, v, secs, workers) for k, v in sorted(inst.items())]
    with ProcessPoolExecutor(par) as ex:
        res = list(ex.map(solve, jobs))
    json.dump(res, open(f'scale50_cpsat_hint_{int(secs)}s_{workers}w.json', 'w'))
    e = np.array([r[1] for r in res]); b = np.array([r[3] for r in res])
    print(f'CP-SAT {secs}s x{workers}w: exact-eval mean {np.nanmean(e):.3f} feasible {np.isfinite(e).sum()}/{len(e)} bound mean {np.nanmean(b):.3f} status', {s: sum(r[5]==s for r in res) for s in set(r[5] for r in res)})
