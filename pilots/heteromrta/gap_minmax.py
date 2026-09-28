"""MinMax (makespan) gap decomposition on CV_MRTA-distribution instances.

SGA-mm   : greedy insertion minimising the resulting max route length (tie: smallest increase)
SGA-mm+TSP: same assignment, each route re-sequenced near-optimally (simple per-robot rule)
REF      : near-optimal min-max via binary search on a per-vehicle max_distance with PyVRP
"""
import random
import sys
from concurrent.futures import ProcessPoolExecutor

from pyvrp import Model
from pyvrp.stop import MaxRuntime

from gap_decomp import BIG, inst, md, path_len, pyvrp_open


def sga_mm(st, tg):
    routes = [[] for _ in st]
    lens = [0] * len(st)
    left = set(range(len(tg)))
    while left:
        best = None
        for i, r in enumerate(routes):
            pts = [st[i]] + [tg[t] for t in r]
            others = max([lens[j] for j in range(len(st)) if j != i], default=0)
            for t in left:
                p = tg[t]
                for pos in range(len(r) + 1):
                    a = pts[pos]
                    inc = md(a, p) + (md(p, pts[pos + 1]) - md(a, pts[pos + 1]) if pos < len(r) else 0)
                    key = (max(others, lens[i] + inc), inc)
                    if best is None or key < best[0]:
                        best = (key, i, t, pos, inc)
        _, i, t, pos, inc = best
        routes[i].insert(pos, t)
        lens[i] += inc
        left.remove(t)
    return routes


def feasible_with(st, tg, D, secs, seed):
    m = Model()
    sl = m.add_location(-50, -50)
    sink = m.add_depot(sl)
    dl = []
    for (x, y) in st:
        loc = m.add_location(x, y)
        d = m.add_depot(loc)
        dl.append(loc)
        m.add_vehicle_type(1, start_depot=d, end_depot=sink, max_distance=D)
    cl = []
    for (x, y) in tg:
        loc = m.add_location(x, y)
        m.add_client(loc)
        cl.append(loc)
    pts = [(sl, None)] + list(zip(dl, st)) + list(zip(cl, tg))
    dset = set(id(x) for x in dl)
    for a, pa in pts:
        for b, pb in pts:
            if a is b:
                continue
            if b is sl:
                dist = 0
            elif a is sl or id(b) in dset:
                dist = BIG
            else:
                dist = md(pa, pb)
            m.add_edge(a, b, distance=dist)
    res = m.solve(stop=MaxRuntime(secs), seed=seed, display=False)
    if not res.is_feasible():
        return None
    return max(r.distance() for r in res.best.routes())


def job(args):
    nr, nt, seed = args
    st, tg = inst(nr, nt, seed)
    routes = sga_mm(st, tg)
    mm_sga = max(path_len(st[i], [tg[t] for t in r]) for i, r in enumerate(routes))
    mm_tsp = max((pyvrp_open([st[i]], [tg[t] for t in r], 0.5, seed) if len(r) > 2 else path_len(st[i], [tg[t] for t in r])) for i, r in enumerate(routes))
    lo, hi = 1, mm_tsp
    best = mm_tsp
    while lo <= hi:
        mid = (lo + hi) // 2
        v = feasible_with(st, tg, mid, 1.5, seed)
        if v is not None:
            best = min(best, v)
            hi = mid - 1
        else:
            lo = mid + 1
    return nr, nt, mm_sga, mm_tsp, best


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    jobs = [(r, t, 1000 * t + 17 * r + k) for t in (25, 50, 100) for r in (6, 16) for k in range(n)]
    import collections
    agg = collections.defaultdict(list)
    with ProcessPoolExecutor(48) as ex:
        for nr, nt, a, b, c in ex.map(job, jobs):
            agg[(nt, nr)].append((a, b, c))
    print("targets robots  SGAmm/ref  SGAmm+perRobotTSP/ref  ref_minmax")
    for k in sorted(agg):
        v = agg[k]
        ref = sum(x[2] for x in v) / len(v)
        print(f"{k[0]:7d} {k[1]:6d}  {sum(x[0] for x in v)/len(v)/ref:9.3f}  {sum(x[1] for x in v)/len(v)/ref:21.3f}  {ref:8.1f}")
