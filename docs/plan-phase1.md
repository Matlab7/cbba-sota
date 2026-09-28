# Phase 1 plan: coalition ALNS vs the strongest baselines (kill gate)

Decision: [research/decision-2026-09-27.md](research/decision-2026-09-27.md), option A.

## Problem (HeteroMRTA semantics, marmotlab/HeteroMRTA @ db51e29)

- Agents `i` in species `s`, binary trait vector `c_s ∈ {0,1}^K` (K=5), depot per species, speed 0.2, Euclidean travel in the unit square.
- Task `j`: requirement `r_j ∈ Z_{≥0}^K`, location, duration `d_j`. Starts when the summed traits of its coalition cover `r_j` (additive), at the last member's arrival.
- Makespan: time when all tasks are finished and every agent is back at its depot. The env caps simulated time at 200 (`success` = finished before the cap).
- **Ground-truth scorer is the env itself** (`TaskEnv.execute_by_route` after `pre_set_route`). The env forms coalitions in *decision order*: once joined members cover the requirement, later members are rejected (the task is skipped for them). Our fast evaluator therefore only produces **minimal covers** (no member can be removed without breaking coverage); for minimal covers and deadlock-free (key-ordered) plans, the forward pass equals the env exactly. Every reported number still comes from env replay.

## Settings (paper Tables I–IV; 50 test instances each)

| Family | (kn agents, ks species, km tasks) |
| --- | --- |
| SA-BT (single-skill agents, binary req.) | (9,3,20) (25,5,20) (25,5,50) (50,5,50) |
| SA-AT (single-skill, additive req. 0–2) | (9,3,20) (15,5,20) (25,5,20) (50,5,50) |
| MA-AT (multi-skill, additive) | (9,3,20) (15,5,20) (25,5,20) (25,5,50) (50,5,50) |
| Large MA-AT (Table IV) | (50,5,200) (150,10,500) (150,5,500) |

Test instances are regenerated with the repo generator from fixed seeds; the RL rows must reproduce the paper within CI before any comparison. Dev instances use disjoint seeds and are the only data used for tuning.

## Baselines

| Baseline | Source | Notes |
| --- | --- | --- |
| RL(g.), RL(s.N) | released checkpoint | N ∈ {10, 64, 256}; parallel sampling; report wall-clock and CPU/GPU time |
| RL(s.N) + ALNS polish | ours | equal-infrastructure hybrid |
| CTAS-D | paper Table I–IV, shipped RALTestSet results | no Gurobi licence here; published rows only (selection-biased: solved instances only) |
| CP-SAT | ours, OR-Tools 9.15 | interval/circuit model, warm-start hints, 8–96 workers, matched wall-clock |
| Greedy (fixed) | ours | the repo greedy never updates `dist` (bug); implement the described nearest-contributable rule |
| SAS / TACO | shipped RALTestSet solutions | BT only |
| ITAGS-style, CBTA | reimplementation (later) | disclosed as reimplementations |

## Primary endpoint and kill gate (to be frozen in `prereg-phase1.md` before test runs)

Per setting: paired mean ratio `ALNS / best baseline` at matched wall-clock budget, where best baseline = the per-setting best of {RL(s.N) at the same wall-clock, RL + polish, CP-SAT, published CTAS-D}. **Kill** if the ratio on the large and 50-task settings is not ≤ 0.92 (8% better) after 2 weeks.

## Code layout

```
cbba_sota/
  hetero/instance.py    Instance dataclass + load from env pickle
  hetero/plan.py        Plan (routes / members / keys), minimal-cover utilities, env route export (1-based)
  hetero/evaluate.py    numba forward pass: makespan, starts, waiting; exact vs env for minimal covers
  hetero/replay.py      ground-truth env replay scorer (subprocess-safe)
  solvers/greedy.py, cpsat.py, rl.py, alns.py
  bench/configs.py      settings table, seeds
scripts/                generation, campaign runner (resumable JSONL), analysis
tests/                  evaluator == env replay on random minimal-cover plans for every family
```
