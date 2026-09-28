"""Coalition ALNS for HeteroMRTA: anytime, exact w.r.t. the env, numba inner loop.

A plan is a global task order plus one minimal-cover coalition per task; every agent visits its tasks in that
order, so plans are deadlock-free and the forward pass equals the env replay (``cbba_sota.hetero``). Each
iteration removes a few tasks (random, Shaw-related, longest-waiting, critical chain, route segment) and reinserts
them at their best slot with a greedy minimal cover by earliest arrival (random / noisy / largest-first / regret-2
order), or moves visits between identical agents (route tails, single visits). Moves are accepted by simulated
annealing on ``makespan + lam * mean finish time``; operator weights adapt by segment scores (Ropke & Pisinger
2006). The best plan is kept by (makespan, objective).

v2 (the default) starts from a constructor portfolio run for a share of the budget and restarts stagnating
searches from an elite pool shared by all workers, half of the time from a crossover child of two elites. v1
(``ALNSConfig.v1()``) starts from its own insertion, restarts from its best plan and lets forked workers share
their incumbent. Settings were tuned on dev instances only. Kernels compile on first use (~30 s, cached on disk by
numba).
"""
from __future__ import annotations

import dataclasses
import multiprocessing as mp
import pickle
import time
from dataclasses import asdict, dataclass, field

import numpy as np

from cbba_sota.hetero import Instance, Plan, evaluate
from cbba_sota.solvers import _alns_kernels as K
from cbba_sota.solvers import greedy
from cbba_sota.solvers._alns_pool import ElitePool, Exchange

__all__ = ["ALNSConfig", "ALNSStats", "portfolio", "solve"]

INIT_BUILDS = 8  # portfolio constructions of iteration-limited (reproducible) runs


@dataclass(frozen=True)
class ALNSConfig:
    lam: float = 0.1  # weight of the mean finish time in the search objective
    noise: float = 0.2  # multiplicative arrival noise of the noisy repair
    q: tuple[int, int] | None = None  # removed tasks per iteration (None: 1 to q_max(T), see ``q_range``)
    max_slots: int = 64  # insertion slots tried per task (all of them for smaller plans)
    shaw_p: float = 6.0  # randomization power of related/worst removal (larger = greedier)
    worst_p: float = 3.0
    regret_max_q: int = 10  # regret-2 repair only for up to this many removed tasks (else random order)
    rho: float = 0.1  # weight reaction factor
    seg_len: int = 100  # iterations per weight segment
    sigma: tuple[float, float, float] = (33.0, 9.0, 13.0)  # scores: new best, better than current, accepted
    restart_iters: int = 3000  # restart after this many iterations without improving the best plan of the cycle
    t_start: float | None = None  # SA start temperature relative to the initial objective (None: ``t_start_rule``)
    t_end_ratio: float = 0.02  # end temperature / start temperature (geometric cooling over the budget)
    batch_s: float = 0.02  # wall time between returns to Python (deadline and trace resolution)
    exchange_s: float | None = 1.0  # incumbent exchange period between workers without a pool (None: independent)
    verify: bool = True  # check the returned plan (minimal covers, evaluator == search objective)
    # v2
    init: str = "portfolio"  # "insert": own random-order best insertion (v1); "portfolio": see ``portfolio``
    init_share: float = 0.05  # budget share of the portfolio construction (at least one dispatch runs)
    init_noise: tuple[float, ...] = (0.01, 0.03, 0.1, 0.3)  # noise levels of the randomized dispatches
    ops_off: tuple[str, ...] = ("member_swap",)  # destroy operators never used
    q_stag: int = 0  # grow the removal cap by one every q_stag iterations without improvement (0: off)
    q_big: int = 12  # limit of that growth
    pool: int = 0  # elite pool size (0: restart from the best plan, v1)
    xover: float = 0.5  # share of pool restarts from a crossover child (else from a random elite)
    xover_frac: tuple[float, float] = (0.3, 0.7)  # range of the share of tasks inherited from the first parent
    kick_q: int = 0  # tasks removed and reinserted when restarting from an elite
    worker_temp: tuple[float, ...] = (1.0,)  # start temperature factor of worker k: worker_temp[k % len]

    @classmethod
    def v1(cls, **changes) -> ALNSConfig:
        """The Phase 1 configuration (for ablations)."""
        return dataclasses.replace(cls(init="insert", ops_off=("member_swap",), q_stag=0, pool=0), **changes)


@dataclass
class ALNSStats:
    iterations: int
    time_s: float  # wall time of the whole call
    search_s: float  # wall time spent iterating (after construction)
    it_per_s: float  # iterations per second of search time, summed over workers
    init_makespan: float
    makespan: float
    trace: list[tuple[float, float]]  # (seconds since the call, best makespan so far)
    insertions: int = 0  # task insertions during the search (all workers)
    candidate_slots: int = 0  # slots whose cover and bound were computed
    exact_evals: int = 0  # slots evaluated exactly (the rest were pruned by the bound)
    improvements: int = 0
    accepted: int = 0
    restarts: int = 0
    crossovers: int = 0
    destroy_weights: dict[str, float] = field(default_factory=dict)
    repair_weights: dict[str, float] = field(default_factory=dict)
    n_workers: int = 1
    seed: int = 0
    workers: list[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)


# --- construction and instance arrays ---------------------------------------------------------------------


def q_range(n_tasks: int) -> tuple[int, int]:
    """Default removal size: 1 to 6 tasks at 20 tasks, 4 at 50, 2 from 200 on (tuned on dev instances at 4-30 s:
    larger instances are iteration-starved and gain from many small moves)."""
    return 1, int(np.clip(round(4 * np.sqrt(50 / n_tasks)), 2, 6))


def t_start_rule(n_tasks: int) -> float:
    """Default relative start temperature: 0.01 up to 50 tasks, then ~T^-1.5 (single-move objective changes shrink
    with the number of tasks; tuned on dev instances)."""
    return 0.01 * min(1.0, (50 / n_tasks) ** 1.5)


def portfolio(inst: Instance, until: float, seed: int = 0, noise: tuple[float, ...] = (0.01, 0.03, 0.1, 0.3),
              builds: int | None = None) -> Plan:
    """Best of the plain dispatch and randomized constructions until ``until`` (monotonic) or, if given, for
    ``builds`` constructions: dispatches with the ``noise`` levels in turn and, up to 100 tasks, every third one a
    random-order insertion (``greedy``)."""
    rng = np.random.default_rng(seed)
    best, best_key = None, None
    k = 0
    while k == 0 or (k < builds if builds is not None else time.monotonic() < until):
        if k == 0:
            plan = greedy.dispatch(inst)
        elif k % 3 == 0 and inst.n_tasks <= 100:
            plan = greedy.insertion(inst, rng)
        else:
            plan = greedy.dispatch(inst, noise=noise[k % len(noise)], seed=int(rng.integers(2**31)))
        sch = evaluate(inst, plan)
        key = (sch.makespan, float(sch.finish.sum()))
        if best_key is None or key < best_key:
            best, best_key = plan, key
        k += 1
    return best


def _width(inst: Instance) -> int:
    """Upper bound on a minimal cover's size: with binary traits every member is the only provider of one unit."""
    binary = np.isin(inst.ab, (0.0, 1.0)).all() and np.array_equal(inst.req, np.round(inst.req))
    return int(min(inst.n_agents, np.ceil(inst.req).sum(axis=1).max())) if binary else inst.n_agents


def _own(a: np.ndarray) -> np.ndarray:
    """Writable float copy, so numba sees one array type whatever the source (unpickled arrays are writable,
    fresh ``Instance`` arrays are not; mixing them compiles the kernels twice)."""
    return np.array(a, dtype=float, order="C")


def _inst_arrays(inst: Instance) -> tuple:
    """Instance tuple of the kernels: req, ab, dur, tt, da, capable lists, nearest tasks, identical-agent groups."""
    T, A = inst.n_tasks, inst.n_agents
    capable = ((inst.ab[None, :, :] > 0) & (inst.req[:, None, :] > 0)).any(axis=2)  # [T, A]
    if not (inst.ab.sum(axis=0) >= inst.req).all():
        raise ValueError("some task cannot be covered by all agents together")
    ncap = capable.sum(axis=1).astype(np.int64)
    capl = np.full((T, A), -1, np.int64)
    for j in range(T):
        capl[j, :ncap[j]] = np.flatnonzero(capable[j])
    tt = inst.tt
    near = np.argsort(tt + np.diag(np.full(T, np.inf)), axis=1, kind="stable")[:, :T - 1].astype(np.int64)
    _, group = np.unique(inst.ab, axis=0, return_inverse=True)
    group = group.ravel().astype(np.int64)
    partners = np.argsort(group, kind="stable").astype(np.int64)
    pstart = np.searchsorted(group[partners], np.arange(group.max() + 2)).astype(np.int64)
    return (_own(inst.req), _own(inst.ab), _own(inst.dur), _own(tt), _own(inst.da), capl, ncap, near, partners,
            pstart, group, capable)


def _new_sol(T: int, A: int, W: int) -> tuple:
    return (np.zeros(T, np.int64), np.full((T, W), -1, np.int64), np.zeros(T, np.int64), np.zeros(1, np.int64),
            np.full(T, np.nan), np.full(T, np.nan), np.zeros((T, W)), np.full((T, W), -1, np.int64),
            np.zeros(A), np.full(A, -1, np.int64), np.zeros(K.N_FL), np.full((T, W), -1, np.int64),
            np.full(A, -1, np.int64), np.zeros(A, np.int64), np.zeros(T))


# --- one worker -------------------------------------------------------------------------------------------


class _Search:
    """One worker: numba state plus the Python batch loop. ``best`` is the best plan of the current cycle (since
    the last restart), ``gbest`` the worker's best overall."""

    def __init__(self, inst: Instance, cfg: ALNSConfig, seed: int, rank: int = 0):
        self.inst, self.cfg, self.seed, self.rank = inst, cfg, seed, rank
        T, A = inst.n_tasks, inst.n_agents
        self.T, self.A, self.W = T, A, _width(inst)
        self.arrays = _inst_arrays(inst)
        self.cur, self.cand, self.best, self.gbest = (_new_sol(T, A, self.W) for _ in range(4))
        W = self.W
        self.ws = (np.zeros(A), np.zeros(A), np.zeros(inst.n_traits), np.zeros(A, np.int64),  # arr key need chosen
                   np.zeros(A, np.int64), np.zeros(A), np.full(A, -1, np.int64),  # overlay: stamp free last
                   np.zeros(len(K.COUNTERS), np.int64),
                   np.zeros(T + 1, np.int64), np.zeros(T + 1), np.zeros(T + 1), np.zeros(T + 1),  # slot p f bound ms
                   np.zeros(T + 1, np.int64), np.zeros((T + 1, W), np.int64),  # slot cover
                   np.full(A, -1, np.int64), np.zeros(A, np.int64), np.zeros(A, np.int64))  # route position
        self.slot_ok = np.zeros(T + 1, np.bool_)
        self.posn = np.full(T, -1, np.int64)
        self.best_c, self.regret_c = np.zeros(A, np.int64), np.zeros(A, np.int64)
        self.removed, self.flag, self.buf = np.zeros(T, np.int64), np.zeros(T, np.bool_), np.zeros(T, np.int64)
        self.keys = np.zeros(T)
        nd, nr = len(K.DESTROY), len(K.REPAIR)
        self.wd = np.array([0.0 if op in cfg.ops_off else 1.0 for op in K.DESTROY])
        self.wr = np.ones(nr)
        self.sd, self.sr, self.ud, self.ur = np.zeros(nd), np.zeros(nr), np.zeros(nd), np.zeros(nr)
        q_lo, q_hi = cfg.q or q_range(T)
        q_hi = max(1, min(q_hi, T))
        q_lo = max(1, min(q_lo, q_hi))
        self.par = np.array([cfg.lam, cfg.noise, q_lo, q_hi, cfg.max_slots, cfg.shaw_p, cfg.worst_p,
                             cfg.regret_max_q, cfg.rho, cfg.seg_len, *cfg.sigma, cfg.restart_iters, cfg.q_stag,
                             max(q_hi, min(cfg.q_big, T)), cfg.pool > 0], float)
        assert len(self.par) == len(K.PARAMS)
        self.stats = np.zeros(len(K.STATS), np.int64)
        self.rng = np.random.default_rng(seed)
        self.crossovers = 0
        K.seed(seed)

    def construct(self, init: Plan | None, until: float = -np.inf, builds: int | None = None) -> None:
        """Start from ``init`` (pruned to minimal covers; tasks without members are inserted) or, without it, from
        the ``portfolio`` construction until ``until`` or for ``builds`` constructions (``cfg.init ==
        "portfolio"``), or by own insertion."""
        cur, cfg = self.cur, self.cfg
        if init is None and cfg.init == "portfolio":
            init = portfolio(self.inst, until, self.seed, cfg.init_noise, builds)
        seq, mem, cnt, ni = cur[:4]
        cnt[:] = 0
        if init is not None:
            if init.n_tasks != self.T or init.n_agents != self.A:
                raise ValueError("init_plan does not match the instance")
            init = init.prune_to_minimal(self.inst)
            order = [int(j) for j in init.order() if init.members[j]]
            for j in order:
                m = init.members[j]
                mem[j, :len(m)] = m
                cnt[j] = len(m)
            seq[:len(order)] = order
            ni[0] = len(order)
        todo = np.flatnonzero(cnt == 0)
        todo = todo[np.random.default_rng(self.seed).permutation(len(todo))].astype(np.int64)
        K.schedule(self.arrays, cur, cfg.lam)
        K.construct(todo, self.arrays, cur, self.ws, self.slot_ok, self.posn, self.best_c, 0.0, cfg.lam,
                    cfg.max_slots)
        K.copy_sol(cur, self.best)
        K.copy_sol(cur, self.gbest)
        self.counters0 = self.ws[7].copy()  # search counters exclude the construction

    def batch(self, n_iter: int, temp_a: float, temp_b: float | None = None) -> int:
        temp_b = temp_a if temp_b is None else temp_b
        return K.run_batch(n_iter, temp_a, temp_b, self.arrays, self.cur, self.cand, self.best, self.ws, self.slot_ok,
                           self.posn, self.best_c, self.regret_c, self.removed, self.flag, self.buf, self.wd,
                           self.wr, self.sd, self.sr, self.ud, self.ur, self.par, self.stats)

    @property
    def best_ms(self) -> float:
        return float(self.gbest[10][0])

    def _new_best(self) -> bool:
        """Carry an improvement of the cycle's best over to ``gbest``."""
        if K.better(self.best[10][0], self.best[10][2], self.gbest[10][0], self.gbest[10][2]):
            K.copy_sol(self.best, self.gbest)
            return True
        return False

    def restart(self, pool: ElitePool) -> None:
        """End the cycle: offer its best plan to the pool and continue from a crossover child of two elites or from
        a random elite (perturbed by ``kick_q`` removals)."""
        cfg = self.cfg
        pool.offer(self.best)
        elites = pool.filled()
        if len(elites) >= 2 and self.rng.random() < cfg.xover:
            a, b = self.rng.choice(elites, 2, replace=False)
            pool.load(a, self.best, self.arrays, cfg.lam)  # the parents go through best and cand
            pool.load(b, self.cand, self.arrays, cfg.lam)
            K.crossover(self.arrays, self.best, self.cand, self.cur, int(self.rng.integers(2)),
                        self.rng.uniform(*cfg.xover_frac), self.keys, self.flag, cfg.lam)
            self.crossovers += 1
        else:
            pool.load(int(self.rng.choice(elites)), self.cur, self.arrays, cfg.lam)
            if cfg.kick_q > 0:
                K.kick(self.arrays, self.cur, min(cfg.kick_q, self.T), self.ws, self.slot_ok, self.posn, self.best_c,
                       self.regret_c, self.removed, self.flag, self.buf, self.par)
        K.copy_sol(self.cur, self.best)
        self.stats[K.S_SINCE] = 0
        self.stats[K.S_RESTART] += 1

    def run(self, t0: float, deadline: float, max_iters: int | None, exchange: Exchange | None,
            pool: ElitePool | None) -> dict:
        cfg = self.cfg
        obj0 = float(self.cur[10][2])
        temp0 = (cfg.t_start if cfg.t_start is not None else t_start_rule(self.T)) * obj0
        temp0 *= cfg.worker_temp[self.rank % len(cfg.worker_temp)]
        temp1 = cfg.t_end_ratio * temp0
        init_ms = self.best_ms
        trace = [(time.monotonic() - t0, init_ms)]
        s0 = time.monotonic()
        horizon = max(deadline - s0, 1e-9)
        per_it = None
        n = 1
        next_exchange = s0 + (cfg.exchange_s or np.inf)

        def temp(frac: float) -> float:
            return temp0 * (temp1 / temp0) ** min(frac, 1.0) if temp0 > 0 else 0.0

        while True:
            now = time.monotonic()
            done = int(self.stats[K.S_IT])
            if now >= deadline or (max_iters is not None and done >= max_iters):
                break
            if per_it is not None:
                n = int(min(cfg.batch_s, deadline - now) / per_it) + 1
            if max_iters is not None:  # temperature by iteration count, independent of the batching
                n = min(n, max_iters - done)
                ta, tb = temp(done / max_iters), temp((done + n) / max_iters)
            else:
                ta, tb = temp((now - s0) / horizon), temp((now + n * (per_it or 0.0) - s0) / horizon)
            improved = self.batch(n, ta, tb)
            end = time.monotonic()
            per_it = max((end - now) / max(int(self.stats[K.S_IT]) - done, 1), 1e-7)
            if improved and self._new_best():
                trace.append((end - t0, self.best_ms))
            if pool is not None and self.stats[K.S_SINCE] >= cfg.restart_iters:
                self.restart(pool)
            if exchange is not None and end >= next_exchange:
                if exchange.sync(self.gbest, self.cur, self.arrays, cfg.lam):
                    K.copy_sol(self.cur, self.best)
                    K.copy_sol(self.cur, self.gbest)
                    trace.append((time.monotonic() - t0, self.best_ms))
                next_exchange = time.monotonic() + cfg.exchange_s
        if pool is not None:
            pool.offer(self.best)
        search_s = time.monotonic() - s0
        it = int(self.stats[K.S_IT])
        counters = dict(zip(K.COUNTERS, (self.ws[7] - self.counters0).tolist()))
        return {"seed": self.seed, "iterations": it, "insertions": counters["insertions"],
                "candidate_slots": counters["candidate_slots"], "exact_evals": counters["exact_evals"],
                "search_s": search_s, "it_per_s": it / max(search_s, 1e-9),
                "init_makespan": init_ms, "makespan": self.best_ms, "trace": trace,
                "improvements": int(self.stats[K.S_IMP]), "accepted": int(self.stats[K.S_ACC]),
                "restarts": int(self.stats[K.S_RESTART]), "crossovers": self.crossovers,
                "destroy_weights": dict(zip(K.DESTROY, self.wd.round(3).tolist())),
                "repair_weights": dict(zip(K.REPAIR, self.wr.round(3).tolist()))}


def _worker(inst, cfg, seed, rank, t0, deadline, max_iters, init, exchange, pool, queue) -> None:
    s = _Search(inst, cfg, seed, rank)
    s.construct(init, t0 + cfg.init_share * (deadline - t0), None if max_iters is None else INIT_BUILDS)
    out = s.run(t0, deadline, max_iters, exchange, pool)
    queue.put((seed, s.gbest[0].copy(), s.gbest[1].copy(), s.gbest[2].copy(), out))


_warm = False


def _warmup() -> None:
    """Compile (or load from cache) every kernel on a toy instance, so forked workers inherit them (once per
    process)."""
    global _warm
    if _warm:
        return
    rng = np.random.default_rng(0)
    toy = Instance(req=rng.integers(0, 2, (6, 2)) + np.eye(2)[[0, 1, 0, 1, 0, 1]], loc=rng.random((6, 2)),
                   dur=rng.random(6), ab=np.eye(2)[[0, 0, 1, 1]], depot=np.zeros((4, 2)), species=[0, 0, 1, 1])
    cfg = ALNSConfig(init="insert", ops_off=(), pool=2, kick_q=2, restart_iters=5, q_stag=2)
    s = _Search(toy, cfg, 0)
    s.construct(None)
    pool = ElitePool(2, s.T, s.W)
    s.run(time.monotonic(), time.monotonic() + 1.0, 60, Exchange(s.T, s.W), pool)
    s.par[K.P_REGQ] = 0
    s.batch(5, 0.1)
    for inst in (toy, pickle.loads(pickle.dumps(toy))):  # fresh arrays are read-only, unpickled ones writable
        portfolio(inst, until=0.0, builds=4)
    _warm = True


def solve(instance: Instance, time_limit_s: float = 10.0, seed: int = 0, n_workers: int = 1,
          init_plan: Plan | None = None, config: ALNSConfig | None = None,
          max_iters: int | None = None) -> tuple[Plan, ALNSStats]:
    """Best plan found within ``time_limit_s`` wall seconds (setup and construction included).

    ``max_iters`` (per worker) additionally stops each worker after that many iterations and fixes the portfolio
    construction to ``INIT_BUILDS`` builds, which makes single-worker runs reproducible for a given ``seed``. ``n_workers > 1`` forks worker processes with seeds ``seed + k``;
    kernels are compiled first in the calling process so the workers inherit them.
    """
    cfg = config or ALNSConfig()
    t0 = time.monotonic()
    deadline = t0 + time_limit_s
    instance.tt, instance.da  # noqa: B018  (cached before any fork)
    W = _width(instance)
    pool = ElitePool(cfg.pool, instance.n_tasks, W) if cfg.pool > 0 else None
    if n_workers <= 1:
        s = _Search(instance, cfg, seed)
        s.construct(init_plan, t0 + cfg.init_share * time_limit_s, None if max_iters is None else INIT_BUILDS)
        runs = [s.run(t0, deadline, max_iters, None, pool)]
        best = s.gbest[:3]
    else:
        _warmup()
        ctx = mp.get_context("fork")
        queue = ctx.Queue()
        exchange = Exchange(instance.n_tasks, W) if cfg.exchange_s and pool is None else None
        procs = [ctx.Process(target=_worker, args=(instance, cfg, seed + k, k, t0, deadline, max_iters, init_plan,
                                                   exchange, pool, queue), daemon=True) for k in range(n_workers)]
        for p in procs:
            p.start()
        results = sorted((queue.get(timeout=time_limit_s + 600) for _ in procs), key=lambda r: r[0])
        for p in procs:
            p.join()
        runs = [r[4] for r in results]
        k = min(range(len(results)), key=lambda k: (runs[k]["makespan"], k))
        best = results[k][1:4]
    seq, mem, cnt = best
    plan = Plan.from_arrays(seq, mem, cnt, instance.n_agents)
    makespan = min(r["makespan"] for r in runs)
    if cfg.verify:
        if not plan.is_minimal_cover(instance):
            raise AssertionError("ALNS returned a plan that is not a minimal cover")
        if evaluate(instance, plan).makespan != makespan:
            raise AssertionError("ALNS makespan disagrees with the evaluator")
    trace = _merge_traces([r["trace"] for r in runs])
    lead = runs[min(range(len(runs)), key=lambda k: (runs[k]["makespan"], k))]
    total = {key: sum(r[key] for r in runs) for key in ("iterations", "improvements", "accepted", "restarts",
                                                         "crossovers", "insertions", "candidate_slots",
                                                         "exact_evals")}
    stats = ALNSStats(time_s=time.monotonic() - t0, search_s=max(r["search_s"] for r in runs),
                      it_per_s=sum(r["it_per_s"] for r in runs), init_makespan=min(r["init_makespan"] for r in runs),
                      makespan=makespan, trace=trace, destroy_weights=lead["destroy_weights"],
                      repair_weights=lead["repair_weights"], n_workers=max(1, n_workers), seed=seed,
                      workers=runs if n_workers > 1 else [], **total)
    return plan, stats


def _merge_traces(traces: list[list[tuple[float, float]]]) -> list[tuple[float, float]]:
    """Best-so-far over all workers, one point per improvement."""
    out: list[tuple[float, float]] = []
    for t, ms in sorted(p for tr in traces for p in tr):
        if not out or ms < out[-1][1]:
            out.append((round(t, 4), ms))
    return out
