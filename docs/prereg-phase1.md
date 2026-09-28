# Pre-registration, Phase 1 test comparison (DRAFT — freeze before any test-split ALNS run)

Status: draft written 2026-09-27 while components were being built. Only dev-split results may inform the open items marked TBD. Once frozen, the commit hash of this file and of the solver code is recorded here, and the test split is run once.

## Hypotheses

- **H1 (primary, SOTA claim).** On each large or 50-task setting, the mean makespan of ALNS at a matched wall-clock budget is at most 0.92 of the best competing method at the same budget and core count.
  - Settings: SA-BT (25,5,50), SA-BT (50,5,50), SA-AT (50,5,50), MA-AT (25,5,50), MA-AT (50,5,50), and Large (50,5,200), (150,10,500), (150,5,500).
- **H2 (secondary).** On the 20-task settings, ALNS at the matched budget is no worse than the best competing method, within a ±2% equivalence margin (TOST). Published CTAS-D rows are compared descriptively only, because CTAS-D means are over solved instances only.
- The shipped RALTestSet is **not blind**: pilots, the red team and ALNS development all ran on it. Its results are descriptive only and are excluded from H1 and H2.

## Competing methods at each budget

The best competing method is chosen per setting and budget. All are env-replayed.
- RL(s.N), with N filling the budget on the same cores.
- RL(s.64) + ALNS polish.
- CP-SAT (hinted, best variant from dev).
- Fixed greedy.

## Budgets

- B1 = the paper's RL(s.10) computation time for that setting; B2 = 2 × B1. For RALTestSet, B1 = 4 s.
- Core counts: 1 and 8.
- Hardware: this host, isolated core sets, no other campaign running.

## Endpoint and analysis

- Per instance: the ratio r = makespan(ALNS) / makespan(best competing method), where "best" is fixed per setting from the dev split (TBD after integration), not chosen per instance on test.
- Report the mean of r with a paired bootstrap 95% CI (10,000 resamples, seed 0), the one-sided paired t-test of log r < log 0.92, and Holm correction across the 8 H1 settings.
- Success rate is reported separately. A failed instance (makespan ≥ 200 or incomplete) counts as makespan 200 in r.

## Decision rule (kill gate)

- **Continue** to the paper if H1 holds, i.e. the Holm-adjusted p < 0.05, in at least 5 of the 8 settings, at B1 with 8 cores.
- **Stop** otherwise, and report the null result.

## Seeds

- Test split: 100000 + 1000·setting_index + i, i < 50.
- Dev split: 500000 + 1000·setting_index + i, i < 20.
- ALNS seeds: 0–7.
- Defined in `cbba_sota/bench/configs.py`.

## Frozen artefacts (fill at freeze time)

- Commit: TBD
- Best competing method per setting: TBD
- ALNS configuration: TBD
