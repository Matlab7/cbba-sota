# Pre-registration of the confirmatory test-split run

Status: FROZEN on 2026-09-28, before any ALNS or CP-SAT variant ran on the test split.

The frozen solver and harness are the source trees recorded under "Frozen artefacts". `scripts/anytime.py` refuses
the test split unless this file is frozen and the checkout's committed `cbba_sota/`, `scripts/` and `pyproject.toml`
are exactly those trees (`runtime.require_frozen`). The test split is run once, from a clean worktree of the freeze
commit. The draft this replaces is in the git history (last draft: commit `7d1fc62`); the changes follow
`docs/baselines-2026-09.md`, Section 8.

## Hypotheses

Target venue: AAMAS 2027, main track (replaces RA-L; decided by the user on 2026-09-28). No minimum margin: the claim
is "consistently and significantly better at equal compute", with effect sizes reported.

- **C1 (primary).** On each of the 8 H1 settings, the mean makespan of ALNS v2 on 8 cores at B1 is lower than that of
  each competitor on 8 cores at B1. H1 settings: SA-BT (25,5,50), SA-BT (50,5,50), SA-AT (50,5,50), MA-AT (25,5,50),
  MA-AT (50,5,50), MA-AT (50,5,200), MA-AT (150,10,500), MA-AT (150,5,500). Competitors: CPSAT, PCPSAT, CPFULL,
  CTAS-D (up to 200 tasks), constructor restarts, RL(s.N). One family of 46 tests (8 settings x 6 competitors, without
  CTAS-D on the two 500-task settings). C1 holds on a setting if every competitor's test is significant after Holm.
- **C1 at 2B1 (secondary).** ALNS v2 vs PCPSAT, CPFULL and constructor restarts, 8 cores at 2B1, the six settings up
  to 200 tasks; its own Holm family (17 tests: the restart streams of the 200-task setting stop at B1).
- **C2 (secondary).** ALNS v2 on 1 core at 2 s vs each competitor on 8 cores at B1; its own Holm family (46 tests).
  ALNS v2 on 1 core at 0.5 and 1 s, and at B1, is descriptive.
- **C3 and the dynamic claims (C1-dyn, C2-dyn)** are descriptive only: the Track D dev gates K1 and K6 fired
  (`docs/trackD-week1-memo.md`). Track D is reported as supplementary analysis; nothing of it is confirmatory.
- **Descriptive:** the 20-task settings (CP-SAT proves many optimal; no headroom), the shipped RALTestSet (not blind),
  anytime curves (from val), and the gap to the best of all test-split runs of this campaign.

## Competing methods

Every competitor is compared separately (no "best of" selection), runs on 8 cores in the same lanes as ALNS v2, and
is scored by the env: plans are replayed with `pre_set_route` + `execute_by_route`; RL is scored by its own rollout.
Budgets include construction and model building. Parameters are the dev choices recorded in `scripts/bks.py`.
- **CPSAT:** CP-SAT LNS (`cpsat.solve_lns`), 8 CP-SAT threads per sub-solve, hint `greedy.construct` built in
  min(10% of the budget, 3 s), sub-solves of `bks.sub_time` seconds (2 s; 10 s on MA-AT-150-10-500).
- **PCPSAT:** parallel CP-SAT LNS (`cpsat.solve_lns_parallel`), 8 forked single-thread workers on one shared
  incumbent, started from the best of 8 `greedy.construct` restart streams built in min(10% of the budget, 3 s),
  sub-solves of `bks.pcpsat_sub_time` seconds.
- **CPFULL:** the full CP-SAT model (`cpsat.solve_full`, CP-SAT's portfolio), 8 threads, hinted with the best of 8
  restart streams built in min(10% of the budget, 3 s), arcs restricted by `bks.cpfull_knn`.
- **CTAS-D:** the MILP of Fu et al. (T-RO 2022) as the benchmark paper ran it (`cbba_sota/solvers/ctas.py`), CP-SAT
  backend, 8 threads, no warm start; up to 200 tasks. On the two 500-task settings it is not run: on the dev pilots it
  found no incumbent within B1 and held 55-87 GB per solve (`docs/results/ctas/ctas500_dev_pilot.jsonl`); it is
  reported as failing there and is not part of the test families. A budget without a converted plan is a failure.
- **Constructor restarts:** 8 independent restart streams (seeds 0-7) of `greedy.construct`; the best plan any stream
  finished by the budget.
- **RL(s.N):** the released HeteroMRTA policy (RA-L 2025) in 8 single-threaded CPU processes, each sampling lockstep
  batches until the budget is used (first batch 1 sample at budgets up to 10 s, else 4); the best rollout.
- The published Gurobi CTAS-D rows (paper Tables I-IV, solved instances only) are descriptive.
- Ablations, not competitors: ALNS v2 on 1 vs 8 workers; ALNS v1 and RL(s.64) + ALNS polish (reported from val/dev).

## Budgets, grid and instances

- B1 = the benchmark paper's RL(s.10) computation time of the setting; 2B1 = 2 x B1.
- Grid `test` (`scripts/anytime.py`, `grid_test`), per instance: ALNS v2 on 1 core at 0.5, 1, 2 s and B1; ALNS v2 and
  every competitor on 8 cores at B1; ALNS v2, PCPSAT and CPFULL on 8 cores at 2B1 (up to 200 tasks); the 8 restart
  streams, read off at every budget up to 2B1 (up to B1 from 200 tasks on).
- Instances: all 50 test instances of each H1 setting (seeds `100000 + 1000 * setting_index + i`, `i < 50`).

## Execution

- Host `dhcho-dev-2gpus-0` (2 x Intel Xeon Platinum 8480C, 57.6-CPU cgroup quota), at most 6 lanes of 8 CPUs pinned
  one per physical core; jobs of all methods shuffled into the same lanes; no other campaign on the host (the user's
  GPU keep-alive process may run, about 2 CPUs). Command: `anytime.py run --split test --grid test --n 50 --n-large 50
  --lanes 6 --out runs/anytime_test` from a clean worktree of the freeze commit (`scripts/campaign_worktree.sh`).
- A lane starts a unit only when at most 40% of the cgroup's CPU periods were throttled over 1 s. A row whose job saw
  more than 40% throttled periods is disturbed: it is re-run and left out.
- A run is late if its wall time exceeds the budget by more than 10% + 0.25 s; it is scored at the budget by the best
  plan in its trace, or as a failure if it keeps no trace. A failure (no plan, makespan >= 200 or unfinished tasks)
  counts as makespan 200.
- The run is resumable; a resume reuses only rows of the frozen code version.

## Endpoint and analysis

- Per instance and competitor: r = makespan(ALNS v2) / makespan(competitor), same budget and cores.
- Mean r with a paired bootstrap 95% CI (10,000 resamples, seed 0), wins-losses, and the one-sided paired t-test of
  mean log r < 0; Holm within each family (C1 at B1, C1 at 2B1, C2 at 2 s), alpha 0.05.
- Success rates (CTAS-D's solved share) are reported separately.
- Tables: `docs/results/test/analyze.sh` (`scripts/anytime.py report --split test --grid test`).

## Decision rule

- The paper claims C1 if it holds on at least 6 of the 8 H1 settings, and names the settings where it fails.
- C2 is claimed only for the settings where it holds; the others are reported as not holding.
- Whatever the outcome, every family is reported in full.

## Disclosures (known at the freeze)

- The val dry run on this host (`docs/baselines-2026-09.md`): C1 held on 8/8 settings, C2 at 2 s on 4/8. Nobody tuned
  on val.
- Constructor and RL rows on the test split were run in Phase 1 (`docs/results/phase1`), so their test-split quality
  was seen. No ALNS or CP-SAT variant and no CTAS-D port has run on the test split.
- The RALTestSet is not blind; it also chose CTAS-D's backend.
- The competitors were tuned on dev, where ALNS was tuned too.

## Seeds

- Test split: `100000 + 1000 * setting_index + i`, `i < 50`; dev: `500000 + ...`, `i < 20`; val: `900000 + ...`,
  `i < 20` (`cbba_sota/bench/configs.py`).
- ALNS v2: seed 0 (8 workers: seeds 0-7); PCPSAT: seed 0 (workers 0-7); CPSAT and CPFULL: seed 0; constructor
  streams: 0-7; RL samples: `rl.sample_seed(instance seed, k)`.

## Frozen artefacts

- Code (the solver and harness of this run), git object ids at the frozen code commit `c8a736173ddffcf8fc1101920119d7899ab497ec`:
- `cbba_sota` tree: `88e870a1c6084cc75d75f48de2273a9525de783a`
- `scripts` tree: `4be8cb5dfefe43a4cbbf756940706b209c07e827`
- `pyproject.toml` blob: `0f89205a1cfc2d79a920a527708a5437af5aa1e9`
- ALNS configuration: `ALNSConfig()` defaults at that commit (v2: portfolio construction for 5% of the budget,
  start-time re-sort with probability 0.1, no delay-aware alternative cover; `cbba_sota/solvers/alns.py`).
- Competitor settings: `bks.SUB_TIME = {MA-AT-150-10-500: 10 s}` (else 2 s); `bks.PCPSAT_SUB_TIME`: 0.5 s on
  SA-BT-25-5-50, SA-AT-50-5-50, MA-AT-25-5-50, MA-AT-50-5-50; 5 s on SA-BT-50-5-50 and MA-AT-150-10-500; 2 s on
  MA-AT-50-5-200 and MA-AT-150-5-500; `bks.CPFULL_KNN`: all arcs on SA-BT-25-5-50, SA-AT-50-5-50, MA-AT-25-5-50;
  10 on SA-BT-50-5-50, MA-AT-50-5-50, MA-AT-150-10-500; 20 on MA-AT-50-5-200; 5 on MA-AT-150-5-500.
- Instances: `data/hetero/<setting>/test/env_<i>.pkl`, regenerated from their seeds; `gen_instances.py --check` found
  all 1490 test, dev and val files equal to their seeds' generation, with distinct fingerprints (2026-09-28).

## Corrections after the freeze

- 2026-09-29: removed an inaccurate disclosure bullet on AI assistance (the frozen text is in commit `2e53eab`). The
  authors proposed the experimental design; an AI coding assistant wrote code and reviewed the design. This correction
  changes no hypothesis, competitor, budget, instance, execution rule or analysis.
- 2026-09-30, after the test run: `scripts/bks.py collect --split test` read the validation rows, because `rows_of`
  bound its default split when the module was imported, before `--split` set it. The descriptive gap to the best
  known plans (and the tables built on it: per-method gaps, CPU shares and CTAS-D's solved share, which the report
  computes on the instances that have a best known plan) was therefore computed against validation plans on 10-20
  instances. The fix passes the split at call time (`rows_of`, `load_bks`) and names the report's CSV files by split;
  the test tables were regenerated from the unchanged campaign rows. The paired ratios, the C1 and C2 tests and every
  row of the campaign are unaffected; the frozen trees above still identify the code that produced the rows.
