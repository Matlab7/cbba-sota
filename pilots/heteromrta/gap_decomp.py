"""Decompose the CBBA-style gap into sequencing vs assignment on CV_MRTA-distribution instances.

SGA = centralized sequential greedy insertion (the solution CBBA converges to under DMG-like
conditions). SGA+TSP = keep SGA's assignment, re-sequence every robot's route near-optimally
(PyVRP single-vehicle open path). FULL = PyVRP multi-robot open-path MinSum (reference).
If SGA+TSP ~= FULL, the gap is sequencing only and a per-robot 2-opt rule would close it
(CBJA-style collapse). If a large gap remains, assignment (inter-robot) moves are needed.
"""
import random
import sys
from concurrent.futures import ProcessPoolExecutor

from pyvrp import Model
from pyvrp.stop import MaxRuntime

G = 19
BIG = 10**6


def starts(n):
    return [(0, round(i * (G - 1) / (n - 1))) for i in range(n)]


def inst(nr, nt, seed):
    st = starts(nr)
    s = set(st)
    el = [(x, y) for y in range(G) for x in range(G) if (x, y) not in s]
    return st, random.Random(seed).sample(el, nt)


def md(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def path_len(start, route):
    L, cur = 0, start
    for p in route:
        L += md(cur, p)
        cur = p
    return L


def sga(st, tg):
    routes = [[] for _ in st]
    left = set(range(len(tg)))
    while left:
        best = None
        for i, r in enumerate(routes):
            pts = [st[i]] + [tg[t] for t in r]
            for t in left:
                p = tg[t]
                for pos in range(len(r) + 1):
                    a = pts[pos]
                    if pos < len(r):
                        b = pts[pos + 1]
                        inc = md(a, p) + md(p, b) - md(a, b)
                    else:
                        inc = md(a, p)
                    if best is None or inc < best[0]:
                        best = (inc, i, t, pos)
        _, i, t, pos = best
        routes[i].insert(pos, t)
        left.remove(t)
    return routes


def pyvrp_open(st, tg, secs, seed):
    m = Model()
    sl = m.add_location(-50, -50)
    sink = m.add_depot(sl)
    dl = []
    for (x, y) in st:
        loc = m.add_location(x, y)
        d = m.add_depot(loc)
        dl.append(loc)
        m.add_vehicle_type(1, start_depot=d, end_depot=sink)
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
    return sum(r.distance() for r in res.best.routes())


def job(args):
    nr, nt, seed = args
    st, tg = inst(nr, nt, seed)
    routes = sga(st, tg)
    s_sga = sum(path_len(st[i], [tg[t] for t in r]) for i, r in enumerate(routes))
    s_tsp = 0
    for i, r in enumerate(routes):
        if len(r) <= 2:
            s_tsp += path_len(st[i], [tg[t] for t in r])
        else:
            s_tsp += pyvrp_open([st[i]], [tg[t] for t in r], 1.0, seed)
    s_full = pyvrp_open(st, tg, 4.0, seed)
    return nr, nt, s_sga, s_tsp, s_full


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    jobs = [(r, t, 1000 * t + 17 * r + k) for t in (25, 50, 100) for r in (6, 16) for k in range(n)]
    import collections
    agg = collections.defaultdict(list)
    with ProcessPoolExecutor(48) as ex:
        for nr, nt, a, b, c in ex.map(job, jobs):
            agg[(nt, nr)].append((a, b, c))
    print("targets robots  SGA/ref  SGA+perRobotTSP/ref  ref")
    for k in sorted(agg):
        v = agg[k]
        ref = sum(x[2] for x in v) / len(v)
        print(f"{k[0]:7d} {k[1]:6d}  {sum(x[0] for x in v)/len(v)/ref:7.3f}  {sum(x[1] for x in v)/len(v)/ref:19.3f}  {ref:6.1f}")
