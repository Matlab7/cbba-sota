"""Near-optimal open-path MinSum reference for CV_MRTA-distribution scale instances.

Same distribution as the benchmark's scale campaign (19x19 grid, edge_even starts,
targets sampled uniformly from non-start cells), but NOT the same seeds (the campaign
ran in an unreleased dev repo). Collision-free Manhattan relaxation, so this is a
lower-bound-ish reference for realized steps, solved with PyVRP (HGS).
"""
import random
import sys
from concurrent.futures import ProcessPoolExecutor

from pyvrp import Model
from pyvrp.stop import MaxRuntime

G = 19
BIG = 10**6


def starts(n):
    if n == 1:
        return [(0, (G - 1) // 2)]
    return [(0, round(i * (G - 1) / (n - 1))) for i in range(n)]


def instance(n_rob, n_tgt, seed):
    st = starts(n_rob)
    s = set(st)
    elig = [(x, y) for y in range(G) for x in range(G) if (x, y) not in s]
    return st, random.Random(seed).sample(elig, n_tgt)


def solve(args):
    n_rob, n_tgt, seed, secs = args
    st, tg = instance(n_rob, n_tgt, seed)
    m = Model()
    sink_loc = m.add_location(-50, -50, name="sink")
    sink = m.add_depot(sink_loc)
    dlocs = []
    for i, (x, y) in enumerate(st):
        loc = m.add_location(x, y)
        d = m.add_depot(loc)
        dlocs.append(loc)
        m.add_vehicle_type(1, start_depot=d, end_depot=sink)
    clocs = []
    for (x, y) in tg:
        loc = m.add_location(x, y)
        m.add_client(loc)
        clocs.append(loc)
    pts = [(sink_loc, None)] + list(zip(dlocs, st)) + list(zip(clocs, tg))
    for a, pa in pts:
        for b, pb in pts:
            if a is b:
                continue
            if b is sink_loc:
                dist = 0
            elif a is sink_loc or pb is None:
                dist = BIG
            elif b in dlocs:
                dist = BIG
            else:
                dist = abs(pa[0] - pb[0]) + abs(pa[1] - pb[1])
            m.add_edge(a, b, distance=dist)
    res = m.solve(stop=MaxRuntime(secs), seed=seed, display=False)
    sol = res.best
    routes = sol.routes()
    lens = [r.distance() for r in routes]
    return n_rob, n_tgt, seed, sum(lens), max(lens) if lens else 0, res.is_feasible()


if __name__ == "__main__":
    trials = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    secs = float(sys.argv[2]) if len(sys.argv) > 2 else 5
    jobs = [(r, t, 1000 * t + 17 * r + k, secs) for t in (25, 50, 75, 100) for r in (6, 8, 12, 16) for k in range(trials)]
    import collections
    agg = collections.defaultdict(list)
    with ProcessPoolExecutor(32) as ex:
        for n_rob, n_tgt, seed, s, mx, feas in ex.map(solve, jobs):
            agg[(n_tgt, n_rob)].append(s)
    for k in sorted(agg):
        v = agg[k]
        print(f"targets={k[0]:3d} robots={k[1]:2d} ref_minsum_mean={sum(v)/len(v):6.1f} n={len(v)}")
