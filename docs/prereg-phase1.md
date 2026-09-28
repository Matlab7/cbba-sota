# Pre-registration (DRAFT — freeze before any test-split ALNS run)

Status: draft written 2026-09-27 while components were being built. Only dev-split results may inform the open items marked TBD. Once frozen, the commit hash of this file and of the solver code is recorded here, and the test split is run once.

## Hypotheses

Target venue: RA-L (decided 2026-09-28; AAMAS optional). The earlier draft's fixed 8% margin gate is dropped:
the claim is "consistently and significantly better at equal compute, cheap enough for real-time replanning,
and robust on degraded networks", with effect sizes reported, not a minimum margin.

- **C1 (quality at matched compute, static).** On each of the 8 H1 settings, ALNS mean makespan is lower than
  that of every competing method at the same wall-clock budget and core count. One-sided paired t-test on
  log-ratios per competitor, Holm correction across settings x competitors; report mean ratio, bootstrap 95% CI and
  win counts. H1 settings: SA-BT (25,5,50), SA-BT (50,5,50), SA-AT (50,5,50), MA-AT (25,5,50), MA-AT (50,5,50), and
  Large (50,5,200), (150,10,500), (150,5,500). The 20-task settings are reported descriptively (CP-SAT proves many of
  them optimal; there is no headroom).
- **C2 (low compute).** Anytime curves at 0.5, 1, 2, 5 s and B1 on 1 and 8 cores; primary summary: ALNS on 1 core at
  2 s versus each competitor at B1 on 8 cores, and ALNS time-to-reach each competitor's B1 quality.
- **C3 (robust on degraded networks, conditional).** Defined in `docs/trackD-spec.md` (Sections 2 and 9.4): on a
  prespecified connectivity grid, SPARC's worst-case regret is lower than that of every fixed architecture sharing its
  planner and of a deployment-time switch. Confirmatory only if the Track D dev gates K5 and K6 pass; otherwise
  descriptive. No message-count claim.
- **C1-dyn / C2-dyn.** Defined in `docs/trackD-spec.md` Section 2: SPARC vs online competitors under release, noise
  and failures at equal per-event budget, and light-budget saturation.
- The shipped RALTestSet is **not blind**: pilots, the red team and ALNS development all ran on it. Its results are
  descriptive only.

## Competing methods at each budget

Every competitor is compared separately (no "best of" selection). All are env-scored.
- Published RL policy: RL(s.N), with N filling the budget on the same cores (CPU lockstep), scored by its own rollout.
- Published CTAS-D rows (paper Tables I-IV; solved instances only): descriptive, because no Gurobi licence is available here.
- CP-SAT-LNS warm-started by our constructor (best variant on dev).
- Our regret-insertion constructor with restarts filling the budget.
- RL(s.64) + ALNS polish is reported as an ablation only (it contains ALNS, so it is not a competitor).

## Budgets

- B1 = the paper's RL(s.10) computation time for that setting; B2 = 2 × B1. For RALTestSet, B1 = 4 s.
- Core counts: 1 and 8.
- Hardware: this host, isolated core sets, no other campaign running.

## Endpoint and analysis

- Per instance and competitor: r = makespan(ALNS) / makespan(competitor) at the same budget and cores.
- Report the mean of r with a paired bootstrap 95% CI (10,000 resamples, seed 0), win counts, and the one-sided
  paired t-test of mean log r < 0; Holm correction over the 8 H1 settings x competitors.
- Success rate is reported separately. A failed instance (makespan >= 200 or incomplete) counts as makespan 200 in r.

## Decision rule

- **Continue** to the paper if C1 holds after Holm correction on at least 6 of the 8 H1 settings against every
  competitor, and C3 holds on the primary degraded-network condition.
- **Stop** (or fall back to a static-only empirical paper) if C1 fails broadly or C3 shows no advantage over the
  central re-solve with local fallback in any degraded condition.

## Seeds

- Test split: 100000 + 1000·setting_index + i, i < 50.
- Dev split: 500000 + 1000·setting_index + i, i < 20.
- ALNS seeds: 0–7.
- Defined in `cbba_sota/bench/configs.py`.

## Frozen artefacts (fill at freeze time)

- Commit: TBD
- Best competing method per setting: TBD
- ALNS configuration: TBD
