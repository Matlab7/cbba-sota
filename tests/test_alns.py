"""Coalition ALNS: kernels are exact (incremental insertion == full forward pass == core evaluator) and every returned
plan is a key-ordered minimal cover whose evaluator makespan equals the env replay."""
from itertools import combinations, pairwise

import numpy as np
import pytest

from cbba_sota.bench.configs import HETEROMRTA_DIR, Setting
from cbba_sota.bench.heteromrta import generate_env, save_env
from cbba_sota.hetero import Instance, Plan, evaluate, forward_pass, random_plan, replay
from cbba_sota.solvers import _alns_kernels as K
from cbba_sota.solvers._alns_pool import Exchange
from cbba_sota.solvers.alns import ALNSConfig, _Search, portfolio, solve

RAL = HETEROMRTA_DIR / "RALTestSet"
SPECS = [("SA-BT", 9, 3, 20, 3), ("SA-AT", 15, 5, 20, 5), ("MA-AT", 25, 5, 50, 5), ("MA-AT", 50, 5, 120, 5)]


@pytest.fixture(scope="module")
def instances(tmp_path_factory) -> list[Instance]:
    root = tmp_path_factory.mktemp("alns")
    out = [Instance.from_pickle(RAL / "env_7.pkl")]
    for k, (family, A, S, T, Kd) in enumerate(SPECS):
        path = root / f"{family}-{A}-{T}.pkl"
        save_env(generate_env(Setting(0, family, 0, A, S, T, Kd, {}), seed=7_000_000 + k), path)
        out.append(Instance.from_pickle(path))
    return out


def _load(search: _Search, plan: Plan) -> None:
    """Put a (complete or partial) minimal-cover plan into ``search.cur``."""
    seq, mem, cnt, ni = search.cur[:4]
    order = [int(j) for j in plan.order() if plan.members[j]]
    cnt[:] = 0
    for j in order:
        mem[j, :len(plan.members[j])] = plan.members[j]
        cnt[j] = len(plan.members[j])
    seq[:len(order)] = order
    ni[0] = len(order)
    K.schedule(search.arrays, search.cur, search.cfg.lam)


def _plan(search: _Search, sol=None) -> Plan:
    seq, mem, cnt, ni = (sol or search.cur)[:4]
    return Plan.from_arrays(seq[:ni[0]], mem, cnt, search.A)


def _check_plan(inst: Instance, plan: Plan, makespan: float | None = None) -> None:
    assert plan.is_minimal_cover(inst)
    Plan.from_routes(plan.routes(), inst.n_tasks)  # acyclic
    ev = evaluate(inst, plan)
    rep = replay(inst, plan)
    assert ev.success and rep["success"] and rep["skipped"] == 0
    assert abs(rep["makespan"] - ev.makespan) <= 1e-9  # one 1-ulp difference seen in ~7000 solves (not reproduced)
    if makespan is not None:
        assert ev.makespan == makespan


def test_cover_is_minimal_and_earliest():
    rng = np.random.default_rng(0)
    for _ in range(300):
        A, Kd = rng.integers(2, 12), rng.integers(1, 6)
        ab = (rng.random((A, Kd)) < 0.5).astype(float)
        ab[np.arange(A), rng.integers(0, Kd, A)] = 1.0
        req = np.minimum(rng.integers(0, 3, (1, Kd)).astype(float), ab.sum(axis=0))
        if not req.any():
            continue
        arr = rng.random(A) * 10
        capl = np.flatnonzero((ab[:, req[0] > 0] > 0).any(axis=1))[None].astype(np.int64)
        chosen = np.zeros(A, np.int64)
        c = K.cover(0, req, ab, capl, np.array([capl.shape[1]]), arr, np.zeros(A), np.zeros(Kd), chosen, np.ones(A))
        m = chosen[:c]
        total = ab[m].sum(axis=0)
        assert (total >= req[0]).all() and len(set(m.tolist())) == c
        assert all(((total - ab[i]) < req[0]).any() for i in m)  # minimal
        # no cover can start earlier: the earliest start is the first arrival threshold that covers
        tau = min(t for t in np.sort(arr) if (ab[arr <= t].sum(axis=0) >= req[0]).all())
        assert arr[m].max() == tau
        order = capl[0][np.argsort(arr[capl[0]])]  # the sweep's variant on agents sorted by arrival
        again = np.zeros(A, np.int64)
        c2 = K.cover_sorted(0, req, ab, order, len(order), arr, np.zeros(Kd), again)
        assert sorted(again[:c2].tolist()) == sorted(m.tolist())


def test_schedule_matches_core_forward_pass(instances):
    for inst in instances:
        s = _Search(inst, ALNSConfig(), 0)
        for r in range(5):
            plan = random_plan(inst, np.random.default_rng(r), greedy=[0.0, 2.0][r % 2])
            _load(s, plan)
            order, mem, cnt = plan.to_arrays()
            T, A = inst.n_tasks, inst.n_agents
            start, finish, ret, wait = np.empty(T), np.empty(T), np.empty(A), np.empty(A)
            ms = forward_pass(order, mem, cnt, inst.dur, inst.tt, inst.da, start, finish, ret, wait)
            assert s.cur[10][0] == ms and np.array_equal(s.cur[8], ret) and np.array_equal(s.cur[5], finish)
            assert np.isclose(s.cur[10][1], finish.sum(), rtol=1e-12)


def test_incremental_insertion_is_exact(instances):
    """Every slot's incremental makespan equals a full forward pass of the plan with the task inserted there, with
    and without the alternative cover (which must be a minimal cover too)."""
    rng = np.random.default_rng(1)
    for inst in instances[:4]:
        s = _Search(inst, ALNSConfig(), 0)
        T = inst.n_tasks
        for trial in range(4):
            alt = trial % 2 == 1  # with the delay-aware alternative cover
            _load(s, random_plan(inst, rng, greedy=1.0))
            removed = rng.choice(T, size=rng.integers(1, max(2, T // 4)), replace=False)
            s.flag[:] = False
            s.flag[removed] = True
            seq, _, cnt, ni = s.cur[:4]
            keep = [t for t in seq[:ni[0]] if not s.flag[t]]
            cnt[removed] = 0
            seq[:len(keep)] = keep
            ni[0] = len(keep)
            K.schedule(s.arrays, s.cur, s.cfg.lam)
            j = int(removed[0])
            for p in range(ni[0] + 1):
                s.slot_ok[:] = False
                s.slot_ok[p] = True
                bp, obj, ms, c, _ = K.insertion(j, s.arrays, s.cur, s.ws, s.slot_ok, 0.0, s.cfg.lam, False,
                                                s.best_c, alt)
                assert bp == p
                total = inst.ab[s.best_c[:c]].sum(axis=0)
                assert (total >= inst.req[j]).all() and all(((total - inst.ab[i]) < inst.req[j]).any()
                                                             for i in s.best_c[:c])
                K.copy_sol(s.cur, s.cand)
                K.apply_insertion(j, p, s.best_c, c, s.cand)
                full = K.schedule(s.arrays, s.cand, s.cfg.lam)
                assert ms == full
                assert np.isclose(obj, s.cand[10][2], rtol=1e-12)


@pytest.mark.parametrize("op", range(len(K.DESTROY)))
def test_each_destroy_operator_keeps_plans_exact(instances, op):
    for inst in instances[1:4]:
        s = _Search(inst, ALNSConfig(), 3)
        s.construct(None)
        s.wd[:] = 0.0
        s.wd[op] = 1.0
        s.par[K.P_SEG] = 1e9  # keep the one-hot weights
        for r in range(len(K.REPAIR)):
            s.wr[:] = 0.0
            s.wr[r] = 1.0
            s.batch(15, 0.05 * s.cur[10][0])
            for sol in (s.cur, s.best):
                assert sol[3][0] == inst.n_tasks
                plan = _plan(s, sol)
                assert plan.is_minimal_cover(inst)
                assert evaluate(inst, plan).makespan == sol[10][0]


def test_tail_swap_keeps_trait_multisets(instances):
    inst = instances[3]
    s = _Search(inst, ALNSConfig(), 0)
    s.construct(None)
    before = [sorted(map(tuple, inst.ab[list(m)].tolist())) for m in _plan(s).members]
    routes = _plan(s).routes()
    for _ in range(40):
        assert K.tail_swap(s.arrays, s.cur, s.cur, s.par)
        plan = _plan(s)
        assert [sorted(map(tuple, inst.ab[list(m)].tolist())) for m in plan.members] == before
        assert evaluate(inst, plan).makespan == s.cur[10][0]
    assert plan.routes() != routes


def test_sort_by_start_keeps_the_schedule(instances):
    """Reordering by start time is another topological order of the same plan: same routes, same times."""
    for inst in instances[1:]:
        s = _Search(inst, ALNSConfig(), 0)
        s.construct(None)
        s.batch(200, 0.05 * s.cur[10][0])
        before = _plan(s)
        start, ms = s.cur[4].copy(), s.cur[10][0]
        K.sort_by_start(s.cur)
        after = _plan(s)
        assert after.routes() == before.routes() and after.members == before.members
        assert np.all(np.diff(start[s.cur[0]]) >= 0)
        assert K.schedule(s.arrays, s.cur, s.cfg.lam) == ms and np.array_equal(s.cur[4], start)


def test_exchange_publishes_and_adopts(instances):
    inst = instances[2]
    s = _Search(inst, ALNSConfig(), 0)
    ex = Exchange(s.T, s.W)
    rng = np.random.default_rng(4)
    plans = []
    for _ in range(2):
        _load(s, random_plan(inst, rng, greedy=1.0))
        plans.append(tuple(a.copy() for a in s.cur))
    good, bad = sorted(plans, key=lambda sol: (sol[10][0], sol[10][2]))
    assert not ex.sync(good, s.cand, s.arrays, s.cfg.lam)  # published
    assert ex.sync(bad, s.cand, s.arrays, s.cfg.lam)  # adopted the shared plan
    assert (s.cand[10][0], s.cand[10][2]) == (good[10][0], good[10][2])
    assert not ex.sync(good, s.cand, s.arrays, s.cfg.lam)  # equal: nothing to do


def test_portfolio_is_minimal_and_not_worse_than_dispatch(instances):
    from cbba_sota.solvers.greedy import dispatch

    for inst in instances[1:]:
        plan = portfolio(inst, until=0.0, builds=6)
        _check_plan(inst, plan)
        assert evaluate(inst, plan).makespan <= evaluate(inst, dispatch(inst)).makespan


@pytest.mark.parametrize("config", [ALNSConfig(), ALNSConfig(alt_cover=True), ALNSConfig.v1()],
                         ids=["v2", "v2-alt", "v1"])
def test_solve_returns_exact_plans(instances, config):
    for k, inst in enumerate(instances):
        plan, st = solve(inst, time_limit_s=1.0, seed=k, config=config)
        _check_plan(inst, plan, st.makespan)
        assert st.iterations > 0 and st.it_per_s > 0 and st.makespan <= st.init_makespan
        assert 0 < st.insertions and st.insertions <= st.exact_evals <= st.candidate_slots
        times, values = zip(*st.trace)
        assert list(times) == sorted(times) and all(b < a for a, b in pairwise(values))
        assert values[-1] == st.makespan and st.time_s < 1.5


def test_iteration_limited_runs_are_reproducible(instances):
    inst = instances[2]
    cfg = ALNSConfig(restart_iters=100)  # several restarts within the run
    a, sa = solve(inst, time_limit_s=60, seed=5, max_iters=1500, config=cfg)
    b, sb = solve(inst, time_limit_s=60, seed=5, max_iters=1500, config=cfg)
    assert sa.iterations == sb.iterations == 1500
    assert a.members == b.members and np.array_equal(a.order(), b.order())


def test_init_plan_is_improved_and_completed(instances):
    inst = instances[3]
    rng = np.random.default_rng(2)
    init = random_plan(inst, rng)
    padded = Plan([tuple(set(m) | {int(rng.integers(inst.n_agents))}) for m in init.members], init.keys,
                  inst.n_agents)  # non-minimal: pruned first
    start = evaluate(inst, padded.prune_to_minimal(inst)).makespan
    plan, st = solve(inst, time_limit_s=0.5, seed=0, init_plan=padded)
    assert st.init_makespan == start and st.makespan <= start
    _check_plan(inst, plan, st.makespan)
    partial = Plan([m if j % 3 else () for j, m in enumerate(init.members)], init.keys, inst.n_agents)
    plan, st = solve(inst, time_limit_s=0.5, seed=0, init_plan=partial)
    _check_plan(inst, plan, st.makespan)


def test_parallel_workers(instances):
    inst = instances[3]
    for cfg in (ALNSConfig.v1(exchange_s=0.1), ALNSConfig.v1(exchange_s=None), ALNSConfig(exchange_s=0.1)):
        plan, st = solve(inst, time_limit_s=1.5, seed=0, n_workers=3, config=cfg)
        _check_plan(inst, plan, st.makespan)
        assert st.n_workers == 3 and len(st.workers) == 3
        assert st.makespan == min(w["makespan"] for w in st.workers)
        assert st.time_s < 2.5


def test_width_bounds_minimal_covers(instances):
    """With binary traits a minimal cover has at most sum(req) members; the coalition buffers rely on it."""
    for inst in instances:
        for j in range(5):
            need = inst.req[j]
            caps = [i for i in range(inst.n_agents) if (inst.ab[i] * (need > 0)).any()][:8]
            for size in range(1, len(caps) + 1):
                for m in combinations(caps, size):
                    total = inst.ab[list(m)].sum(axis=0)
                    if (total >= need).all() and all(((total - inst.ab[i]) < need).any() for i in m):
                        assert size <= need.sum()
