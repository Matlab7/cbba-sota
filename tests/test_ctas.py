"""CTAS-D port (solvers.ctas): tiny instances against the exact CP-SAT model and the env, a plan's encoding, the
shipped Gurobi runs plugged into the model and replayed, infeasible and out-of-time runs, hints."""
import math
import time
from itertools import pairwise

import numpy as np
import pytest

from cbba_sota.bench.configs import HETEROMRTA_DIR
from cbba_sota.hetero import Instance, evaluate, random_plan, replay
from cbba_sota.hetero.replay import replay_routes
from cbba_sota.solvers import cpsat, ctas

RAL = HETEROMRTA_DIR / "RALTestSet"


def _tiny(seed: int, per: int, n_tasks: int) -> Instance:
    """Three species with traits (1, 0), (0, 1) and (1, 1), ``per`` agents each; requirements in {0, .., per}^2."""
    rng = np.random.default_rng(seed)
    species = np.repeat(np.arange(3), per)
    req = rng.integers(0, per + 1, (n_tasks, 2)).astype(float)
    req[req.sum(axis=1) == 0, 0] = 1
    depots = rng.random((3, 2))
    return Instance(req=req, loc=rng.random((n_tasks, 2)), dur=5 * rng.random(n_tasks),
                    ab=np.array([[1, 0], [0, 1], [1, 1]], dtype=float)[species], depot=depots[species],
                    species=species)


@pytest.mark.parametrize("backend", ctas.BACKENDS)
def test_tiny_optimum_is_the_true_optimum_as_a_minimal_cover(backend):
    """The CTAS optimum neither beats the exact CP-SAT optimum (certified bound) nor misses its encoding (objective
    at most the encoded optimum's; CP_SAT on its 1e-3 grid up to one grid step per arc more), and it converts to a
    minimal cover whose evaluate equals the env replay; here its makespan is the optimum itself."""
    for seed in range(2):
        inst = _tiny(seed, per=2, n_tasks=3)
        opt = cpsat.solve_full(inst, 10, workers=1)
        assert opt.status == "OPTIMAL"
        model = ctas.Model(inst)
        _, opt_objective = model.check(model.assignment(opt.plan))
        res = ctas.solve(inst, 20, backend=backend, threads=1)
        assert res.status == "OPTIMAL" and res.plan.is_minimal_cover(inst)
        rep = replay(inst, res.plan)
        assert rep["success"] and rep["skipped"] == 0 and rep["makespan"] == res.makespan
        assert res.makespan >= opt.bound - (inst.n_tasks + 1) / cpsat.SCALE - 1e-9
        slack = ctas.TIME_PENALTY * (inst.n_tasks + 1) / ctas.CPSAT_SCALING if backend == "CP_SAT" else 1e-6
        assert res.objective <= opt_objective + slack
        assert res.makespan <= res.qmax + 1e-6
        assert res.makespan == pytest.approx(opt.makespan, abs=1e-6)
        assert all(res.plan.keys[a] < res.plan.keys[b] for r in res.plan.routes() for a, b in pairwise(r))


def test_plan_encoding_is_feasible_with_its_objective():
    """A plan's encoding satisfies every row; its objective is 100 x makespan + travelled distance, plus
    1e-4 x (longest energy path to every visited node) with the energy rows."""
    inst = Instance.from_pickle(RAL / "env_3.pkl")
    plan = random_plan(inst, np.random.default_rng(3), greedy=2.0)
    ms = evaluate(inst, plan).makespan
    dist = sum(inst.da[a, r[0]] + sum(inst.tt[i, j] for i, j in pairwise(r)) + inst.da[a, r[-1]]
               for a, r in enumerate(plan.routes()) if r) * inst.speed
    for energy in (False, True):
        model = ctas.Model(inst, "SCIP", energy)
        viol, objective = model.check(model.assignment(plan))
        assert viol < 1e-7  # one ulp of the EngEdge big-M (1e8) is 1.5e-8
        extra = objective - ctas.TIME_PENALTY * ms - dist
        if energy:
            assert 0 < extra < ctas.G_COST * len(model.g) * dist
        else:
            assert extra == pytest.approx(0, abs=1e-9)


@pytest.mark.parametrize("i", [0, 7])
def test_shipped_gurobi_run_replays_and_fits_the_model(i):
    """The benchmark's Gurobi solution: its routes (CTAS-D.py's assignment) replay to qMax = timeCost / 100, our graph
    equals graph.yaml, its values satisfy our model to print precision (energy rows: see ``check_shipped``) and give
    its objVal."""
    inst = Instance.from_pickle(RAL / f"env_{i}.pkl")
    shipped = ctas.shipped_solution(inst, RAL / f"env_{i}")
    rep = replay_routes(inst.source, [[j + 1 for j in r] for r in shipped.routes])
    assert rep["success"] and rep["skipped"] == 0
    assert rep["makespan"] == pytest.approx(shipped.qmax, abs=1e-5)
    assert shipped.result["timeCost"] == pytest.approx(ctas.TIME_PENALTY * shipped.qmax, abs=1e-4)
    assert evaluate(inst, shipped.plan).makespan == pytest.approx(rep["makespan"], abs=1e-9)
    chk = ctas.check_shipped(inst, shipped)
    assert chk["graph_diff"] < 1e-12 and chk["dropped"] == 0
    assert chk["violation"] < 1e-4  # 6-decimal printing; big-M rows with xr within Gurobi's tolerance of 1
    assert chk["eng_violation"] < ctas.MAXENG * 1e-5  # an xr within Gurobi's IntFeasTol (1e-5) of 1 frees g
    assert chk["objective"] == pytest.approx(chk["objVal"], abs=1e-4)


@pytest.mark.parametrize("backend", ctas.BACKENDS)
def test_infeasible_requirement(backend):
    inst = Instance(req=[[2, 0], [0, 1]], loc=[[0.2, 0.3], [0.7, 0.1]], dur=[1, 2], ab=[[1, 0], [0, 1]],
                    depot=[[0.5, 0.5], [0.5, 0.5]], species=[0, 1])
    res = ctas.solve(inst, 10, backend=backend, threads=1)
    assert res.plan is None and res.status == "INFEASIBLE" and res.makespan == math.inf


def test_out_of_time():
    inst = Instance.from_pickle(RAL / "env_0.pkl")
    res = ctas.solve(inst, 5, t0=time.perf_counter() - 10)  # budget spent before building
    assert res.plan is None and res.status == "TIMEOUT"
    rng = np.random.default_rng(0)
    big = Instance(req=rng.integers(0, 2, (80, 5)) + np.eye(5)[np.arange(80) % 5], loc=rng.random((80, 2)),
                   dur=rng.random(80), ab=np.eye(5)[np.arange(15) % 5], depot=rng.random((15, 2)),
                   species=np.arange(15) % 5)
    res = ctas.solve(big, 1e-3)  # building runs out of time
    assert res.plan is None and res.status == "TIMEOUT" and res.wall_s < 1
    res = ctas.solve(inst, 2, backend="SCIP", threads=1)  # SCIP finds no incumbent on env_0 in seconds
    assert res.plan is None and res.status == "NOT_SOLVED" and res.makespan == math.inf


def test_highs_incumbent_after_time_limit():
    """OR-Tools drops HiGHS's incumbent at the time limit; the improving-solution file recovers it."""
    inst = _tiny(0, per=2, n_tasks=7)
    res = ctas.solve(inst, 2, backend="HIGHS", threads=1)
    assert res.status == "FEASIBLE" and math.isnan(res.bound) and res.plan.is_minimal_cover(inst)
    assert replay(inst, res.plan)["makespan"] == res.makespan <= res.qmax + 1e-6


def test_hint_is_a_complete_solution():
    """SCIP finds nothing on env_0 in seconds alone, so its incumbent is the plan hint (objective at most the
    hint's); HIGHS refuses hints."""
    inst = Instance.from_pickle(RAL / "env_0.pkl")
    hint = random_plan(inst, np.random.default_rng(0), greedy=2.0)
    model = ctas.Model(inst, "SCIP")
    viol, hint_objective = model.check(model.assignment(hint))
    assert viol < 1e-9
    res = ctas.solve(inst, 3, backend="SCIP", threads=1, hint=hint)
    assert res.plan is not None and res.objective <= hint_objective + 1e-6
    assert res.makespan <= res.qmax + 1e-6 <= res.objective / ctas.TIME_PENALTY + 2e-6
    with pytest.raises(ValueError):
        ctas.solve(inst, 1, backend="HIGHS", hint=hint)
