# Stronger baselines and the same-host C1/C2 dry run on val (2026-09-28, second server)

This is step 3.1 of `docs/HANDOFF.md` ("harden the static claims"): the must-do items of the Track S review
(`docs/plan-phase1b.md`, section "Before freezing the static claims"). Every timed run here ran on the new server,
`dhcho-dev-2gpus-0` (Section 10 names the one untimed check that did not). No test-split instance was touched; tuning
used `dev`, measurement used `val`.

## Summary

- **Stronger competitors.** Added a properly parallel CP-SAT LNS (PCPSAT: 8 single-thread workers, one shared
  incumbent, CPU share 1.00), the full CP-SAT model with CP-SAT's own portfolio (CPFULL), both hinted with the best of
  8 construction streams and tuned on dev, and the benchmark paper's exact method CTAS-D, rebuilt on an open solver
  and verified against the paper's own Gurobi runs (50/50 shipped solutions reproduce its objective and makespan).
- **Parallelism was not what held CP-SAT back.** PCPSAT keeps 8 cores busy and runs 5-32 times as many sub-solves as
  the threaded CP-SAT LNS, but it is at most 2.8% better on dev and level on 200 tasks; the full model rarely
  improves on its hint within B1 at 200 and 500 tasks (13 of 16 dev instances unchanged). CTAS-D solves 12 of 60 val
  instances within B1 (10 of them on SA-BT-50-5-50, 87% above the best known) and neither 500-task dev pilot.
- **C1 holds on all 8 H1 settings**, on one host, with every competitor re-run in the same lanes: ALNS v2 on 8 cores
  at B1 is 3-13% better than every CP-SAT variant and constructor restarts, 20-32% better than RL, every instance won,
  all 46 tests Holm-significant (500-task settings at n = 5). The 500-task settings, the missing piece, behave like the
  others (11-13% ahead of the CP-SAT variants and restarts).
- **C2 holds on 4 of 8 settings at 2 s** (1 core against 8-core competitors at B1). It fails on SA-AT-50-5-50 and
  SA-BT-50-5-50 (ratios 0.977-0.995, not significant) and, at n = 5, on the 500-task settings, although ALNS2 won 39 of
  the 40 pairs there against the CP-SAT variants and restarts. Against RL it holds everywhere from 0.5 s.
- **The best known plans are not only ALNS's own:** no run without an ALNS component (competitors of both hosts, and
  new 300-s x 8-core CP-SAT references) matches the best ALNS-derived plan on any 50- to 500-task instance; the best of
  them is 2.8-12.0% worse. Certified bounds stay 12-58% below, so the distance to the optimum remains unknown.
- **The harness items of the review are done:** pinning in every runner, no timed run on a dirty tree, routes base in
  rows and plan files, RL's first batch of 1 at low budgets, CPU seconds per method, and a reviewer's ten findings on
  the new code fixed (Section 9). The same cells measured on both hosts agree within 2%.
- **Not done here:** the pre-registration freeze (the user's; proposed edits in Section 8) and the test run.


## 1. Host and provenance

- **Host.** `dhcho-dev-2gpus-0`: 2 x Intel Xeon Platinum 8480C (112 physical cores), a cgroup CPU quota of 57.6 CPUs
  and a memory cap of 330 GB for the container, 2 x H100. The first server (Phase 1 and 1b) had a 96-CPU quota shared
  with other tenants' jobs. Here no other tenant was seen; the only other load was the user's GPU keep-alive process
  (`use-gpu.py`, about 1.9 CPUs and both GPUs, running since 2026-09-16). RL runs on CPU in this harness, so the busy
  GPUs do not affect it.
- **Code.** Every timed row ran from a clean git worktree of a fixed commit (`scripts/campaign_worktree.sh`), so edits
  in the main checkout could not change what a running campaign imported. Rows record that commit (`git`), the host,
  the CPU affinity, the load and the cgroup's throttling counters. Tuning: `3fa7e7e`. Val campaign and independent
  references: `80e643e`.
- **Lanes.** 6 lanes of 8 pinned CPUs (one CPU per physical core) at most, i.e. 48 CPUs, so the container stayed
  under its quota; the tuning ran on 4 lanes while the CTAS-D port was being validated, then on 6.
- **Setup check** (HANDOFF Section 2): `gen_instances.py --check` found 1490 instance files with 1490 distinct
  fingerprints, all equal to a fresh generation from their seeds; the run archive's sha256 matches; the test suite
  passed (333 tests before this work).

## 2. What changed in the harness (review items 2 and 5)

- **Pinning everywhere.** `run_alns.py`, `alns_ablation.py` and `cpsat_lns_ref.py` now pin every pool process to its
  own block of CPUs (`runtime.pinned_pool`) and write host conditions into every row.
- **No timed campaign on a dirty tree.** `anytime.py`, `bks.py`, `compare_dev.py`, `run_alns.py`,
  `alns_ablation.py`, `cpsat_lns_ref.py` and `run_baselines.py` refuse to start when the source has uncommitted
  changes (`runtime.require_clean`; `--allow-dirty` for smoke tests).
- **Route convention stored.** Rows and the published BKS plan files carry `routes_base` (0: 0-based task ids,
  the convention of runs/anytime, runs/bks, runs/compare_dev and runs/baselines; 1: env routes, runs/alns).
- **RL at low budgets.** In `anytime.py`, RL(s.N) starts with a lockstep batch of 1 sample at budgets up to 10 s
  (else 4), so it has a solution from about 1 s on 50-task instances.
- **CPU seconds reported.** Rows of multi-process methods record the CPU seconds of all their processes and threads;
  the reports print CPU used / (budget x cores) per method.

## 3. Competitors (review item 1)

The review found that the only non-RL competitor, CP-SAT LNS (CPSAT), used about 53% of its 8 cores and that no
established external method was compared. Three competitors were added. Every competitor runs in the pinned lanes of
`scripts/anytime.py`; its budget starts after the instance is loaded and includes construction and model building;
its plan is replayed in the env.

| Label | Method | On 8 cores |
| --- | --- | --- |
| CPSAT | CP-SAT LNS (`cpsat.solve_lns`) from the `greedy.construct` hint built in min(10% of B, 3 s); sub-solves of 2 s (10 s on MA-AT-150-10-500), the Phase 1b dev choice | one process; 8 CP-SAT threads per sub-solve; model building is serial |
| **PCPSAT** (new) | parallel CP-SAT LNS (`cpsat.solve_lns_parallel`): 8 forked processes, one CP-SAT thread each, improving one shared incumbent (makespan, then summed depot returns); they start from the best of 8 `greedy.construct` restart streams built in min(10% of B, 3 s); sub-solve time per setting from dev | 8 processes |
| **CPFULL** (new) | the full CP-SAT model (`cpsat.solve_full`) with CP-SAT's own portfolio of complete and LNS workers, hinted with the best of 8 forked restart streams; arcs to the k nearest candidate tasks where dev chose so | 8 CP-SAT threads; model building and presolve serial |
| **CTAS** (new, external) | CTAS-D (Fu et al., T-RO 2022), the exact MILP the benchmark paper compared against, rebuilt from the MIT-licensed C++/Gurobi code (`cbba_sota/solvers/ctas.py`) and solved by CP-SAT as a MIP backend, no warm start; up to 200 tasks (Section 3.1) | 8 CP-SAT threads |
| CONSTRUCT | the regret-insertion constructor, 8 independent restart streams | 8 processes |
| RL | RL(s.N), the released policy, 8 single-threaded CPU processes sampling lockstep batches | 8 processes |

### 3.1 CTAS-D on an open solver

- **Faithfulness.** The model is planner mode `TEAMPLANNER_CONDET` of the C++ code: species-level continuous flows
  with binary arc indicators, big-M synchronisation of task start times, cumulative capability coverage, and the
  FlowConverter rounding and path split into per-agent routes. The benchmark authors' makespan objective is not in the
  public code; it was recovered from their 50 shipped RALTestSet Gurobi runs as energy + 100 x makespan.
  - Plugged into our model, the 50 shipped Gurobi solutions satisfy every row to print precision and reproduce the
    shipped objective value on 50/50.
  - Their routes replay in the env to the shipped makespan on 50/50.
  - Deviations, each with its reason, are listed in the module docstring (irrelevant species-task pairs get no
    variables; the energy rows, which constrain nothing at an energy cap of 1e6, are off by default; CP-SAT solves a
    1e-3 time grid; the budget includes model building).
- **Backend** (RALTestSet env 0-9, 60 s including model building, 8 threads, commit `368469b`;
  `docs/results/ctas/ralt_check.txt`): CP-SAT solved 10/10 with a mean makespan of 23.27, 4.1% below the shipped
  Gurobi 600 s runs (24.31) and 0.9% above the best known; HiGHS solved 10/10 at 2.2 x Gurobi; SCIP solved 3/10. An
  earlier run of the uncommitted port on the 2-GPU host gave 23.58 (2.8% below Gurobi) and SCIP 5/10. CP-SAT is used.
  RALTestSet is non-blind, so this choice used no dev, val or test instance.
- **Scale.** On SA-BT-50-5-50 val 0 in 30 s only the CP-SAT backend found a solution (30.19 in this run, 24.98 in the
  earlier one, against a best known of 16.06). On both 500-task settings (dev 0, B1 = 560 s,
  8 threads) it found no incumbent and the solver held 55 and 87 GB (`docs/results/ctas/ctas500_dev_pilot.jsonl`);
  several such jobs at once would exceed the container's memory, so CTAS is not run on the 500-task settings and
  counts as failed there.

## 4. Dev tuning of the new CP-SAT competitors

`anytime.py tune`, dev 0-9 (500 tasks: dev 0-2), 8 cores at B1, pinned lanes, commit `3fa7e7e`. Variants: PCPSAT
sub-solve time 0.2, 0.5, 2 and 5 s (500 tasks: 0.5, 2, 5, 10 s); CPFULL with all arcs or the 10 nearest (200 tasks:
10 or 20; 500 tasks: 5 or 10). Per setting the variant with the lowest mean was chosen (`bks.PCPSAT_SUB_TIME`,
`bks.CPFULL_KNN`); full table in `docs/results/val-c1/tune_dev.txt`. Dev is the ALNS tuning split, so these ratios
are optimistic for ALNS; but each competitor's variant was also chosen on these very instances, which favours the
competitor.

| Setting | n | ALNS2 | CPSAT | PCPSAT (dev choice) | CPFULL (dev choice) | ALNS2/CPSAT | ALNS2/PCPSAT | ALNS2/CPFULL | CPU share CPSAT / PCPSAT / CPFULL |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SA-BT-25-5-50 | 10 | 27.311 | 29.937 | 29.093 (0.5 s) | 29.433 (all arcs) | 0.912 [0.904, 0.919] 10-0 | 0.940 [0.928, 0.951] 10-0 | 0.928 [0.920, 0.936] 10-0 | 0.80 / 1.00 / 0.74 |
| SA-BT-50-5-50 | 10 | 17.342 | 17.979 | 17.938 (5 s) | 18.220 (k=10) | 0.965 [0.951, 0.978] 10-0 | 0.967 [0.954, 0.978] 10-0 | 0.952 [0.942, 0.964] 10-0 | 0.58 / 1.00 / 0.83 |
| SA-AT-50-5-50 | 10 | 28.533 | 30.771 | 30.097 (0.5 s) | 30.505 (all arcs) | 0.928 [0.920, 0.934] 10-0 | 0.948 [0.940, 0.956] 10-0 | 0.935 [0.926, 0.944] 10-0 | 0.72 / 1.00 / 0.72 |
| MA-AT-25-5-50 | 10 | 36.307 | 41.259 | 40.467 (0.5 s) | 40.849 (all arcs) | 0.877 [0.857, 0.897] 10-0 | 0.897 [0.879, 0.912] 10-0 | 0.887 [0.870, 0.904] 10-0 | 0.76 / 1.00 / 0.53 |
| MA-AT-50-5-50 | 10 | 20.158 | 22.250 | 21.864 (0.5 s) | 22.376 (k=10) | 0.904 [0.888, 0.918] 10-0 | 0.918 [0.904, 0.933] 10-0 | 0.897 [0.882, 0.910] 10-0 | 0.66 / 1.00 / 0.76 |
| MA-AT-50-5-200 | 10 | 53.003 | 59.645 | 59.575 (2 s) | 60.420 (k=20) | 0.888 [0.879, 0.898] 10-0 | 0.890 [0.876, 0.903] 10-0 | 0.877 [0.864, 0.890] 10-0 | 0.66 / 1.00 / 0.78 |
| MA-AT-150-10-500 | 3 | - | - | 42.819 (5 s) | 44.545 (k=10) | - | - | - | - / 1.00 / 0.87 |
| MA-AT-150-5-500 | 3 | - | - | 45.472 (2 s) | 46.785 (k=5) | - | - | - | - / 1.00 / 0.91 |

Mean makespan (env replay) per method; paired ratios with bootstrap 95% CI and wins-losses. The 500-task tuning ran
the variants only (no ALNS2 or CPSAT reference rows).

Observations:
- **Parallelism is not what limits CP-SAT here.** PCPSAT keeps all 8 cores busy (CPU share 1.00, against 0.58-0.80
  for CPSAT) and runs 5-32 times as many sub-solves, but it is only 0.2-2.8% better than CPSAT on the 50-task
  settings and level on 200 tasks (59.58 vs 59.65). CP-SAT's own portfolio on the full model (CPFULL) is no better.
- **Short sub-solves stall on large instances.** At 200 tasks, PCPSAT with 0.2 or 0.5 s sub-solves ran about 1000
  sub-solves per instance without a single improvement (building and presolving the sub-model uses up the time); from
  2 s on it improves. On the 50-task settings 0.5 s was best and 0.2 s worse everywhere, so the optimum is interior.
- **The full model does not scale.** CPFULL ends with its hint (the best of 8 construction streams) on 9 of 10
  instances at 200 tasks and 4 of 6 at 500 tasks (0-4 of 10 on the 50-task settings): within B1 CP-SAT does not
  improve on it.
- **ALNS v2 stays ahead of every CP-SAT variant on dev:** ALNS2/PCPSAT 0.890-0.967, ALNS2/CPFULL 0.877-0.952,
  ALNS2/CPSAT 0.877-0.965, every pair won (60-0 in each column).
- Two process incidents, both handled: the first tuning launch was stopped after 277 rows to extend the sub-solve grid
  below 0.5 s (0.5 s was best on 4 of 5 settings at the time) and to add 0.5 s at 500 tasks; and for about 4 minutes
  two launches ran at once (a kill that missed its session), which throttled 10 rows of MA-AT-50-5-200 in 16-26% of
  their CPU periods. Those rows were dropped (`runs/tune_dev/NOTE.txt`) and re-run.

## 5. The same-host C1/C2 dry run on val (grid `c1`)

The Phase 1b val study ran on the first server, and its 500-task settings were never measured. Comparing new
competitor rows from this host with those rows would mix hardware and host conditions, so every method was run again
here, in one campaign, with jobs of all methods shuffled into the same lanes (`anytime.py run --grid c1`, commit
`80e643e`, rows in runs/anytime_c1):
- **Instances:** val 0-9 of the six 50- and 200-task H1 settings; val 0-4 of the two 500-task settings.
- **Every instance:** ALNS2 on 1 core at 0.5, 1, 2 s and B1; ALNS2, CPSAT, PCPSAT, CPFULL, CTAS (up to 200 tasks) and
  RL on 8 cores at B1; the 8 constructor restart streams (read off at every budget); ALNS2, PCPSAT and CPFULL on 8
  cores at 2B1 (up to 200 tasks).
- **Val 0-4 of the 50/200-task settings:** also ALNS2 on 1 core at 5 and 10 s, and ALNS2, PCPSAT, CPFULL and RL on 8
  cores at 0.5-10 s (anytime curves). 500 tasks: ALNS2 on 1 core at 0.5-10 s.
- **Rules.** A row whose job saw more than 40% throttled CPU periods is disturbed, re-run and left out. A run over its
  budget by more than 10% + 0.25 s is late; it is scored by the best plan its trace had reached by the budget, or as a
  failure (200) if it keeps no trace (RL, CTAS). (The Phase 1b report left late runs out; that rule is kept for its
  own tables.) A failed plan counts as 200. Paired ratios ALNS2 / competitor: bootstrap 95% CI (10,000 resamples,
  seed 0), wins-losses, one-sided paired t-test of mean log ratio < 0, Holm over settings x competitors per row type.
- The ALNS v1 ablation and the 20-task settings (descriptive, no headroom) were not re-run; their Phase 1b numbers
  stand.

### 5.1 C1: 8 cores at B1

ALNS v2 is better than every competitor on all 8 H1 settings: 46 paired tests, every one Holm-significant (largest
p_holm 0.013; the 500-task settings have n = 5), every instance won. At 2B1 (ALNS2 vs PCPSAT, CPFULL and constructor
restarts, 6 settings up to 200 tasks) the same holds: 17 of 17 tests, largest p_holm 9e-5.

| Setting | n | ALNS2 gap | ALNS2/CPSAT | ALNS2/PCPSAT | ALNS2/CPFULL | ALNS2/CONSTRUCT | ALNS2/RL | CTAS-D solved |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SA-BT-25-5-50 | 10 | 1.53% | 0.911 [0.895, 0.924] | 0.925 [0.915, 0.935] | 0.910 [0.902, 0.917] | 0.919 [0.907, 0.930] | 0.799 [0.782, 0.815] | 1/10 |
| SA-BT-50-5-50 | 10 | 0.27% | 0.968 [0.957, 0.976] | 0.951 [0.938, 0.964] | 0.930 [0.917, 0.942] | 0.918 [0.909, 0.927] | 0.758 [0.740, 0.774] | 10/10 (87% above the best known) |
| SA-AT-50-5-50 | 10 | 0.90% | 0.923 [0.915, 0.930] | 0.926 [0.919, 0.933] | 0.922 [0.919, 0.924] | 0.934 [0.925, 0.941] | 0.770 [0.755, 0.784] | 0/10 |
| MA-AT-25-5-50 | 10 | 0.80% | 0.900 [0.885, 0.914] | 0.889 [0.874, 0.904] | 0.888 [0.872, 0.904] | 0.903 [0.887, 0.919] | 0.733 [0.711, 0.757] | 0/10 |
| MA-AT-50-5-50 | 10 | 0.73% | 0.890 [0.872, 0.907] | 0.907 [0.896, 0.918] | 0.877 [0.862, 0.893] | 0.890 [0.873, 0.909] | 0.727 [0.693, 0.760] | 1/10 |
| MA-AT-50-5-200 | 10 | 1.36% | 0.900 [0.882, 0.916] | 0.904 [0.889, 0.920] | 0.886 [0.871, 0.902] | 0.885 [0.871, 0.899] | 0.722 [0.707, 0.737] | 0/10 |
| MA-AT-150-10-500 | 5 | 0.00% | 0.885 [0.875, 0.902] | 0.890 [0.877, 0.906] | 0.870 [0.859, 0.886] | 0.888 [0.880, 0.896] | 0.680 [0.673, 0.687] | not run |
| MA-AT-150-5-500 | 5 | 0.07% | 0.890 [0.846, 0.935] | 0.890 [0.847, 0.931] | 0.867 [0.823, 0.909] | 0.888 [0.859, 0.917] | 0.713 [0.695, 0.729] | not run |

Paired ratio ALNS2 / competitor (bootstrap 95% CI); every pair was won (10-0 or 5-0). "ALNS2 gap" is the mean gap to the
best of all our runs (Section 6). CTAS-D failures count as makespan 200, so ALNS2/CTAS is 0.13-0.54
(`docs/results/val-c1/c1_val.md`); the table gives its solved share instead. On the 500-task settings the best known
plans are ALNS2's own 8-core B1 runs, hence the near-zero gaps.

- **The 500-task settings**, the C1 evidence that was missing, show the same picture as the smaller ones: ALNS2 is
  11-13% better than every CP-SAT variant and constructor restarts, and 29-32% better than RL.
- **CPU used** at B1 on 8 cores (CPU seconds of all processes and threads / (B1 x 8), `c1_val.md`): ALNS2, PCPSAT,
  constructor restarts and RL 0.94-1.00; CPSAT 0.28-0.79; CPFULL 0.43-0.90; CTAS 0.43-0.97. PCPSAT uses its cores
  fully and is not better than CPSAT for it (ALNS2/PCPSAT and ALNS2/CPSAT are within 0.02 of each other everywhere).
- **Eight workers vs one** (ALNS2-8 / ALNS2-1 at B1): 0.973-0.996, i.e. 0.4-2.7% from 8 workers.

### 5.2 C2: ALNS v2 on 1 core against 8-core competitors at B1

| Budget of ALNS2 on 1 core | Settings where ALNS2-1 beats every competitor on 8 cores at B1 (Holm) |
| --- | --- |
| 0.5 s | 1 of 8 |
| 1 s | 3 of 8 |
| 2 s | 4 of 8: SA-BT-25-5-50, MA-AT-25-5-50, MA-AT-50-5-50, MA-AT-50-5-200 |

At 2 s the claim fails on:
- **SA-AT-50-5-50** against every CP-SAT variant and against constructor restarts: ratios 0.982-0.995, 7-3 to 9-1,
  p_holm 0.07-0.23. This is the setting where C2 already failed against constructor restarts on the first host.
- **SA-BT-50-5-50** against CPSAT (0.995, 6-4) and PCPSAT (0.977, 8-2).
- **The 500-task settings** at n = 5: ALNS2-1 wins 30 of 30 pairs against the CP-SAT variants and 9 of 10 against
  constructor restarts (ratios 0.935-0.986), but three comparisons miss Holm significance (p_holm 0.12-0.13).
Against RL on 8 cores at B1, ALNS2 on 1 core wins every pair at every budget from 0.5 s (0.75-0.85).


## 6. Is the best known solution only ALNS's own? (review item 4)

The review asked for a reference that ALNS did not produce. Two long runs without any ALNS component were added on val
0-4 of the five 50-task settings (`bks.py run --kinds PLNS CPFULLc --seconds 300 --width 8`, 10-20 times the 8-core
B1 compute): parallel CP-SAT LNS (PLNS) and the full CP-SAT model with all arcs, which also gives a certified lower
bound (CPFULLc); both start from constructions only. `bks.py collect` now compares, per instance, the best plan of
every run with an ALNS component (ALNS runs, and CPFULL BKS runs hinted with the best plan so far) with the best plan
of every run without one (the two references, the CP-SAT LNS BKS runs and every competitor row of both hosts' timed
campaigns; `docs/results/val-c1/bks_val.txt`).

| Setting | best known (mean) | best without ALNS: above it | PLNS 8 x 300 s | CPFULLc 8 x 300 s | certified bound below the best known |
| --- | --- | --- | --- | --- | --- |
| SA-BT-25-5-50 | 26.431 | +7.6% | +8.06% | +6.10% | 46% |
| SA-BT-50-5-50 | 16.734 | +2.8% | +3.66% | +4.98% | 12% |
| SA-AT-50-5-50 | 26.848 | +6.7% | +7.18% | +7.04% | 45% |
| MA-AT-25-5-50 | 30.243 | +9.4% | +13.30% | +11.95% | 58% |
| MA-AT-50-5-50 | 19.088 | +8.1% | +9.68% | +13.88% | 30% |
| MA-AT-50-5-200 | 60.174 | +11.3% | - | - | - |
| MA-AT-150-10-500 | 39.219 | +12.0% | - | - | - |
| MA-AT-150-5-500 | 39.883 | +11.5% | - | - | - |

"Best without ALNS: above it" is the mean over instances of (best plan without ALNS - best known) / best known. The
reference columns are their mean gap to the best known on val 0-4.
- On no instance of any 50- to 500-task setting did a run without ALNS find a plan as good as the best ALNS-derived
  one. The best of all of them, including 300-s runs on 8 cores, is 2.8-12.0% worse on average (0.3-18% per
  instance). On the 20-task settings the two families meet (SA-BT: the same plans, proven optimal).
- ALNS2 on 8 cores at B1 ends 0.0-1.5% above the best known (Section 5.1), much closer than any independent run with
  10-20 times its compute.
- This still does not bound the distance to the optimum: the certified CP-SAT bounds are 12-58% below the best known.

## 7. Cross-host check

`anytime.py report --compare runs/anytime` pairs the cells that the first host's Phase 1b campaign also ran (ALNS2 on
1 and 8 cores, CPSAT, RL, constructor restarts) over the instances both ran on time. All 52 shared cells at B1 and 2B1
agree: ratio (this host / first host) 0.982-1.010, mean 0.999 (end of `docs/results/val-c1/c1_val.txt`). The only
cell outside 0.99-1.01 is CPSAT-8 at B1 on SA-BT-50-5-50 (0.982): CPSAT is the method whose CPU use depends most on
the host (Python model building between multi-threaded sub-solves). RL at budgets up to 10 s is left out, because its
first lockstep batch changed from 4 samples to 1 between the two campaigns.

## 8. Proposed changes to the pre-registration (for the freeze, HANDOFF step 3.3; not applied)

`docs/prereg-phase1.md` is still a draft. The user freezes it. Proposed edits from this work:
- **Competitors** (each compared separately, all env-scored): CPSAT, PCPSAT, CPFULL (per-setting dev choices in
  `bks.PCPSAT_SUB_TIME` / `bks.CPFULL_KNN`), CTAS-D on CP-SAT up to 200 tasks (a failure on the 500-task settings, on
  the dev pilot's evidence), CONSTRUCT-8 and RL(s.N) on 8 cores. The published Gurobi CTAS-D rows stay descriptive.
- **Late runs:** scored at the budget by their trace, else as failures (Section 5), in place of leaving them out.
- **Host:** `dhcho-dev-2gpus-0`, at most 6 lanes of 8 pinned CPUs, runs from a clean worktree of the frozen commit;
  rows with more than 40% throttled CPU periods are re-run. The Phase 1b val numbers came from another host.
- **Test-split cost:** one 500-task test instance costs about 3,400 lane-seconds (six 8-core jobs of 560 s), so 50
  instances of both 500-task settings take about 16 hours on 6 lanes, against about 5.5 hours for 50 instances of the
  six other H1 settings (B1 and 2B1 points). Either plan about a day for the test run or reduce the 500-task n (20
  instances: about 6.5 hours).
- **Disclosures:** constructor rows on the test split were seen in Phase 1; RALTestSet is non-blind (and chose the
  CTAS-D backend); the competitors were tuned on dev, where ALNS was also tuned.
- **Decision rule:** with six competitors the Holm family per row type grows to 8 x 6 = 48 tests; C3 is descriptive
  (Track D appendix, K1 and K6 fired).

## 9. Review of the new code

A read-only review of commits `3fa7e7e` and `80e643e` reported ten findings, none critical; all were fixed in
`368469b`:
- late runs were left out rather than scored (now: trace at the budget, else a failure; Section 5);
- CTAS-D's flow split could overrun the budget (now bounded by the deadline);
- resume keys ignored the split and the code version (now both; a campaign resumed on other code recomputes);
- CPFULLc would build about 30M arc literals at 500 tasks (now up to 200 tasks);
- one dead lane process would have failed every later unit of its lane (now the lane requeues its jobs and stops);
- tune variants accepted parameters the runners reject; the cross-host comparison mixed RL's first-batch change;
  the BKS report's "own" group was not ALNS-only; `require_clean` accepted an unknown version and a worktree run
  without `PYTHONPATH`.
None of them affected the val campaign's C1/C2 numbers: no C1/C2 cell had a late run (every B1 and 2B1 run ended
within 1.07 x its budget; the 105 late runs are all at the 0.5-10 s anytime points on 8 cores: RL 66, CPFULL 30,
PCPSAT 7, ALNS2 2), CTAS-D rows ended within 1.05 x B1, and no lane process died.

## 10. Caveats

- **Val, not test.** Nobody tuned on val, but these are dry runs: n = 10 (500 tasks: n = 5) per setting, one seed per
  (instance, method, budget).
- **Dev tuning favours the competitors' variants**, chosen per setting on the instances they are then compared on in
  Section 4; on val the same variants were run unchanged.
- **CTAS-D is not the paper's Gurobi run.** An open backend (CP-SAT on a 1e-3 time grid), a budget that includes model
  building, and no Gurobi licence. Its 60-s runs beat the shipped 600-s Gurobi runs on the 20-task RALTestSet, so the
  port is not weak at small scale; at 50+ tasks it rarely finds a solution within B1, as the paper's Gurobi runs rarely
  did within an hour.
- **The session moved hosts.** The tuning, the val campaign and the independent references ran on
  `dhcho-dev-2gpus-0`; after 18:18 UTC the controlling session continued on `dhcho-4gpu-0`, where only analysis and
  the CTAS-D port check (a correctness check, not a timed comparison) ran. Rows record their host; every timed row
  here is from `dhcho-dev-2gpus-0`.
- **One flaky test.** `tests/test_alns.py::test_solve_returns_exact_plans[v2]` failed once in four full runs (the
  last trace point and the final makespan differed in the last bit after a 1-s run) and passed five targeted reruns;
  the ALNS code was not changed here.

## Reproduce

```
scripts/campaign_worktree.sh <commit>           # clean worktree at /home/jovyan/dev/cbba-sota-run
cd /home/jovyan/dev/cbba-sota-run && export PYTHONPATH=$PWD
PY=/home/jovyan/dev/cbba-sota/.venv/bin/python
# dev tuning (commit 3fa7e7e; runs/tune_dev), per stage:
$PY scripts/anytime.py tune --settings SA-BT-25-5-50 SA-BT-50-5-50 SA-AT-50-5-50 MA-AT-25-5-50 MA-AT-50-5-50 --n 10 \
  --variants PCPSAT:sub_time=0.2 PCPSAT:sub_time=0.5 PCPSAT:sub_time=2 PCPSAT:sub_time=5 CPFULL:knn=0 CPFULL:knn=10 \
  --refs ALNS2 CPSAT --lanes 6
$PY scripts/anytime.py tune --settings MA-AT-50-5-200 --n 10 --variants PCPSAT:sub_time=0.2 PCPSAT:sub_time=0.5 \
  PCPSAT:sub_time=2 PCPSAT:sub_time=5 CPFULL:knn=10 CPFULL:knn=20 --refs ALNS2 CPSAT --lanes 6
$PY scripts/anytime.py tune --settings MA-AT-150-10-500 MA-AT-150-5-500 --n-large 3 --variants PCPSAT:sub_time=0.5 \
  PCPSAT:sub_time=2 PCPSAT:sub_time=5 PCPSAT:sub_time=10 CPFULL:knn=5 CPFULL:knn=10 --refs --lanes 6
# val campaign and independent references (commit 80e643e; runs/anytime_c1, runs/bks)
$PY scripts/anytime.py run --grid c1 --out runs/anytime_c1 --settings SA-BT-25-5-50 SA-BT-50-5-50 SA-AT-50-5-50 \
  MA-AT-25-5-50 MA-AT-50-5-50 MA-AT-50-5-200 MA-AT-150-10-500 MA-AT-150-5-500 --n 10 --n-large 5 --lanes 6
$PY scripts/bks.py run --kinds PLNS CPFULLc --settings SA-BT-25-5-50 SA-BT-50-5-50 SA-AT-50-5-50 MA-AT-25-5-50 \
  MA-AT-50-5-50 --n 5 --width 8 --lanes 6 --seconds 300
# CTAS-D checks (any host; not timed comparisons)
.venv/bin/python scripts/ctas_check.py
.venv/bin/python scripts/ctas500_pilot.py MA-AT-150-5-500 <8 CPUs>
# tables (from the main checkout)
bash docs/results/val-c1/analyze.sh
```
The exact launch scripts and logs are in runs/logs (`tune_dev_2026-09-28*.sh`, `val_c1_2026-09-28.sh`).

## Files

- Code: `cbba_sota/solvers/cpsat.py` (`solve_lns_parallel`, `construct_parallel`), `cbba_sota/solvers/ctas.py`,
  `cbba_sota/bench/runtime.py` (`require_clean`, `pinned_pool`, `Probe`), `scripts/anytime.py` (grid `c1`, `tune`,
  new competitors, late rule), `scripts/bks.py` (dev choices, PLNS/CPFULLc, ALNS-free comparison),
  `scripts/campaign_worktree.sh`, `scripts/ctas_check.py`, `scripts/ctas500_pilot.py`.
- Tests: `tests/test_baselines.py`, `tests/test_headroom.py`, `tests/test_ctas.py`, `tests/test_adversarial.py`.
- Results: `docs/results/val-c1/` (tuning, BKS, C1/C2 tables, anytime curves), `docs/results/ctas/` (port check,
  500-task pilot).
