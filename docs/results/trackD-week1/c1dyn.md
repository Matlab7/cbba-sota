# Track D week 1: C1-dyn pilot, gates K1 and K2 (agent B, days 2-3)

Date: 2026-09-28. Dev split only (dev instances 0-19 of each setting). The test split was not touched.

**Where the numbers come from.** Every number below comes from the ground-truth env `DynTaskEnvX`, never a hand-written executor. The env code hash is `6fabb0570f` (env.py + perturb.py, fixed at import). It is the hash K0 verified, with controller `80bf534371`. Worlds:
- plan-following methods: `DynTaskEnvX[arrival,causal,L=200,det]@6fabb0570f`;
- policies (RL, greedy): `DynTaskEnvX[decision,causal,L=200,det]@6fabb0570f`, plus the same arms under the arrival-order rule (`-arr`).

**Code.** Every file the runner imports (env, perturb, controller, planner, rh_kernels, sparc, methods, the baselines, `scripts/trackD_run.py`) is identical to commit `1f0f046`, checked by sha1 before and after the campaigns. The rows carry `git_dirty_dyn = true` only because other agents have untracked or modified modules in `cbba_sota/dyn` that the runner does not import, plus my own `c1dyn.py`. The kernel is ALNS v2 pinned at `4cf5e04` (`cbba_sota/dyn/rh_kernels.py`).

**Host.** Shared, load 60-90. Every CPU number is process time on this loaded host and is descriptive only. Workers were pinned to CPUs 8-39, clear of the concurrent `anytime.py` cores and their hyperthread siblings. At most 22 workers ran at once.

## 1. Decisions

| Gate | Rule (spec) | Result (dev, seeds 0-1, 80 instance clusters) | Decision |
|---|---|---|---|
| **T1 / K1** planner vs rule | SPARC-L / B4 (same Light budget), F1-F3 pooled ≤ 0.97, cluster-CI upper < 1, < 1 on ≥ 3/4 settings | **0.9847 [0.9806, 0.9890]**; < 1 on 4/4 settings (0.978, 0.982, 0.990, 0.989); p = 8.5e-10 | **T1 fails, so K1 fires** |
| **K2** no dynamic separation | SPARC / ins-only > 0.98 on F3 **and** SPARC / B4 > 0.97 on F1 | F3: 0.8279 [0.8167, 0.8390] (≤ 0.98); F1: 0.9881 (> 0.97) | **does not fire** (the first condition fails) |
| T1b | SPARC / insertion-only, descriptive | 0.8598 [0.8510, 0.8687]; 80/80 instances | search pays on structural events |
| T2 | SPARC / every-event RH, descriptive | 1.0098 [1.0052, 1.0145]; F3 1.0191 | every-event is 1% better (contrary to the S8 prior) |
| T3 | F0: SPARC vs open loop, TOST ±2%; decisions identical | GM 1.0000; 480/480 pairs identical; 1 re-plan per episode | pass (identity) |
| E2a preview (C2) | SPARC-L vs SPARC-H, TOST ±2% | 1.0229, 90% CI [1.0189, 1.0269] | **not equivalent**: Light is 2.3% behind Heavy |
| K3 preview | B1 within 5% of SPARC (SPARC / B1 > 0.952) | RL(g.) 0.9144; RL(g.) arrival order 0.9209; RL(s.1) 0.9104 | does not fire |

**Reading.**
- SPARC-Light is better than every competitor run here, with every cluster CI below 1:
  - the rolling constructor B4 at every budget tested;
  - D-ITAGS-style repair B5;
  - insertion-only;
  - open loop;
  - the published RL online (B1), under both coalition rules;
  - the paper greedy (B7).
- Against the strongest cheap rule, B4 at equal CPU, the margin is only **1.5%**. That is half the 3% that T1 requires.
- The failure is not borderline: the whole pooled CI [0.9806, 0.9890] lies above 0.97. It holds at every B4 budget from the as-implemented equal-CPU count up to 4x the generous one (Section 5). It also holds with families weighted equally (0.9855).
- Per spec Section 8, K1 means dropping the dynamic-method claim, and Track D becomes an analysis appendix: S2, S6 and S8 as findings, with this pilot's numbers.

The exploratory results in Section 6 are the only routes to a planner claim, and none is gate evidence:
- The margin comes entirely from SPARC's in-loop repairs, not from its t = 0 plan.
- Relaxed keys (SPARC-rk) add about 1.5%. That puts SPARC-rk / B4 at 0.9717 on seeds 0-1 and 0.9665 on fresh seeds 2-3, straddling the 0.97 bar.
- The Heavy tier passes the T1 rule (0.9627), but with 10x SPARC-L's iterations per event: 3.69 s CPU per episode against 1.26 s for B4-eq.

Choosing relaxed keys now, after seeing T1 fail on these instances, is a forking path. It would need a re-freeze and a confirmation on data not used to choose it: the validation split, which is not generated yet.

## 2. Design

| Item | Value |
|---|---|
| Settings | MA-AT-25-5-50, MA-AT-50-5-50, SA-AT-50-5-50, SA-BT-50-5-50 (primary); MA-AT-50-5-200 (secondary, descriptive) |
| Instances x CRN seeds | dev 0-19 x seeds {0, 1}; exploratory replication on seeds {2, 3} |
| Cells | F1-R1, F1-R2, F1-R3, F2-R2N3, F3-pf0.1, F3-pf0.2, F3-pf0.2-R2 (C1-dyn primary); F0-N1/N2/N3 for T3 |
| Episodes | 32,240 rows: 32,234 completed, 6 excluded, 0 errors. 1,120 paired episodes per method on F1-F3 (seeds 0-1, 4 settings); 0 F3 realizations excluded on seeds 0-1 |
| Env options | spec defaults: causal observations; lease = failure detector (L = 200, detection 5 ticks); abandon on failure detection; D7 termination |
| Light tier (1 core) | SPARC-L 300 ALNS iterations per structural event (first plan 300 too; `iters0` defaults to the per-event budget); B5 300; every-event 300; open-loop 300 at t = 0 then insertion; insertion-only floor |
| B4 (collapse control) | rolling constructor: regret-2 floor plus r randomized restarts, cold at every structural event; r from the calibration (Section 3) |
| Heavy tier | SPARC-H 3000 iterations per event (first plan 3000) |
| Native | RL(g.) and RL(s.1) (released checkpoint, B1) and greedy (B7), each under the decision-order rule (native, `auto`) and the arrival-order rule (`-arr`) |
| Statistics | `cbba_sota/dyn/stats.py`: pairs are (setting, instance, cell, seed); unit = instance mean of log r over its cells and seeds; cluster bootstrap over the 80 instances within settings (10,000 resamples, seed 0); one-sided t on instance means; pairs where both succeed (failures = 200 as sensitivity) |

The spec lists "B7" as the paper greedy; the task text used that name for D-ITAGS-style repair (B5). Both were run.

**Pre-specification.** Before any campaign row was analysed, the header of `runs/trackD/c1dyn/campaign.sh` fixed three things:
- the T1/K1 comparator: B4-eq, the construction-only equal-CPU count;
- the T1 and K2 rules;
- the sensitivity arms: B4-2eq, B4-4eq, B4-impl, B4-r12.

The frozen budget file is `runs/trackD/c1dyn/b4_budget_primary_frozen.json`. Everything in Section 6 was added after a snapshot of seeds 0-1 was seen, and is labelled exploratory.

## 3. B4 budget calibration (spec 3.6: Light = CPU-ms of 300 SPARC iterations)

**Method.**
- `ProbeSPARC` is SPARC-L that, at every re-plan, also times B4 on the *same* belief state. The adopted plans are SPARC's own, and a test shows the episode is identical to plain SPARC.
- Sample: 240 SPARC-L episodes (4 settings x dev 0-19 x {F1-R2, F2-R2N3, F3-pf0.2}, seed 0), giving 6,555 probed events.
- Fit: mean B4 CPU = a + b·r (probed at r = 0 and r = 24), then r* = (mean SPARC CPU − a) / b.

**Why two counts.** B4 as implemented rebuilds the kernel `Problem` for every restart, and that rebuild is about 70% of its per-restart CPU. So two counts were computed:
- *as implemented* (B4-impl);
- *construction-only*, which does not charge the rebuild.

The construction-only count is generous to B4 and is the pre-specified comparator, B4-eq.

| Setting | SPARC-L CPU per event (mean) | B4 as implemented, a + b·r | r*, as implemented | Construction-only b | r*, construction-only (median-based) | B4-eq restarts |
|---|---|---|---|---|---|---|
| MA-AT-25-5-50 | 15.54 ms | 1.51 + 0.818 r ms | 17.2 | 0.300 ms | 46.8 (51.5) | 47 |
| MA-AT-50-5-50 | 12.29 ms | 1.35 + 0.880 r | 12.4 | 0.251 | 43.6 (47.1) | 44 |
| SA-AT-50-5-50 | 23.07 ms | 2.40 + 1.120 r | 18.5 | 0.424 | 48.8 (55.2) | 49 |
| SA-BT-50-5-50 | 9.61 ms | 1.26 + 0.889 r | 9.4 | 0.216 | 38.6 (39.2) | 39 |
| MA-AT-50-5-200 (secondary) | 42.86 ms | 36.89 + 3.404 r | 1.8 | 2.147 | 2.8 (11.8) | 3 |

**Check in the campaign itself** (CPU per episode, all controller calls with structural events): SPARC-L 0.464 s, B4-impl 0.490 s, B4-eq 1.260 s. The as-implemented calibration therefore reproduces equal CPU in the real runs to within 6%. B4-eq spends 2.7x SPARC-L's CPU.

Commands: `scripts/trackD_c1dyn.py calibrate --instances 0-19 --seeds 0 --procs 20` (log `runs/trackD/c1dyn/calibrate.log`). For 200 tasks: `--settings MA-AT-50-5-200 --instances 0-4 --tag 200`.

## 4. Results: SPARC-L / method (geometric-mean paired ratio, < 1 favours SPARC-L)

F1-F3 pooled, 1,120 paired episodes per method (seeds 0-1). The CI is the cluster bootstrap over 80 instances. Wins count instances whose mean log r is below 0. The last column is the competitor's success rate (SPARC-L succeeded in 1120/1120).

| Method | Pooled [95% CI] | F1 | F2 | F3 | MA-AT-25 | MA-AT-50 | SA-AT-50 | SA-BT-50 | Wins /80 | Success |
|---|---|---|---|---|---|---|---|---|---|---|
| B4-eq (T1 comparator) | **0.9847 [0.9806, 0.9890]** | 0.9881 | 0.9880 | 0.9803 | 0.9784 | 0.9818 | 0.9897 | 0.9890 | 59 | 1120 |
| B4-impl (equal CPU as implemented) | 0.9740 [0.9695, 0.9785] | 0.9808 | 0.9725 | 0.9677 | 0.9659 | 0.9732 | 0.9819 | 0.9748 | 73 | 1120 |
| B4-r12 (K0 preview budget) | 0.9723 [0.9679, 0.9767] | 0.9776 | 0.9742 | 0.9663 | 0.9573 | 0.9732 | 0.9775 | 0.9812 | 72 | 1120 |
| B4-2eq | 0.9884 [0.9838, 0.9930] | 0.9920 | 0.9878 | 0.9850 | 0.9810 | 0.9951 | 0.9929 | 0.9848 | 56 | 1120 |
| B4-4eq | 0.9913 [0.9867, 0.9959] | 0.9973 | 0.9937 | 0.9846 | 0.9909 | 0.9921 | 0.9957 | 0.9865 | 50 | 1120 |
| B5 D-ITAGS-style | 0.9914 [0.9877, 0.9950] | 0.9953 | 0.9828 | 0.9904 | 0.9913 | 0.9991 | 0.9837 | 0.9915 | 55 | 1120 |
| insertion-only | 0.8598 [0.8510, 0.8687] | 0.8962 | 0.8509 | 0.8279 | 0.8092 | 0.8772 | 0.8315 | 0.9261 | 80 | 1120 |
| open loop (R-c) | 0.8827 [0.8753, 0.8899] | 0.9221 | 0.8880 | 0.8432 | 0.8725 | 0.8923 | 0.8397 | 0.9286 | 80 | 1120 |
| every-event RH (R-b) | 1.0098 [1.0052, 1.0145] | 1.0013 | 1.0078 | 1.0191 | 1.0068 | 1.0132 | 1.0148 | 1.0045 | 26 | 1120 |
| SPARC-H (3000 it) | 1.0229 [1.0181, 1.0277] | 1.0094 | 1.0167 | 1.0387 | 1.0244 | 1.0237 | 1.0353 | 1.0085 | 12 | 1120 |
| B1 RL(g.) | 0.9144 [0.9074, 0.9214] | 0.8987 | 0.8830 | 0.9415 | 0.8770 | 0.9001 | 0.9490 | 0.9332 | 80 | 1117 |
| B1 RL(g.), arrival order | 0.9209 [0.9141, 0.9275] | 0.9039 | 0.8922 | 0.9481 | 0.8899 | 0.9124 | 0.9490 | 0.9332 | 80 | 1119 |
| B1 RL(s.1) | 0.9104 [0.9036, 0.9171] | 0.8969 | 0.8871 | 0.9322 | 0.8711 | 0.8969 | 0.9404 | 0.9349 | 80 | 1118 |
| B1 RL(s.1), arrival order | 0.9168 [0.9100, 0.9234] | 0.9026 | 0.8923 | 0.9397 | 0.8836 | 0.9092 | 0.9404 | 0.9349 | 80 | 1119 |
| B7 greedy | 0.7878 [0.7748, 0.8006] | 0.8070 | 0.7933 | 0.7894 | 0.6302 | 0.8329 | 0.8044 | 0.9122 | 80 | 1038 |
| B7 greedy, arrival order | 0.7940 [0.7808, 0.8070] | 0.8117 | 0.8004 | 0.7923 | 0.6394 | 0.8470 | 0.8044 | 0.9122 | 80 | 1039 |

**Per cell** (SPARC-L / method):

| Method | F1-R1 | F1-R2 | F1-R3 | F2-R2N3 | F3-pf0.1 | F3-pf0.2 | F3-pf0.2-R2 |
|---|---|---|---|---|---|---|---|
| B4-eq | 0.9885 | 0.9901 | 0.9858 | 0.9880 | 0.9817 | 0.9834 | 0.9758 |
| B4-impl | 0.9814 | 0.9822 | 0.9787 | 0.9725 | 0.9698 | 0.9689 | 0.9644 |
| B5 | 0.9981 | 0.9951 | 0.9928 | 0.9828 | 0.9845 | 0.9893 | 0.9975 |
| insertion-only | 0.9151 | 0.8707 | 0.9034 | 0.8509 | 0.8605 | 0.8109 | 0.8131 |
| open loop | 0.9505 | 0.8923 | 0.9245 | 0.8880 | 0.8632 | 0.8175 | 0.8496 |
| every-event | 0.9960 | 1.0045 | 1.0035 | 1.0078 | 1.0292 | 1.0310 | 0.9974 |
| SPARC-H | 1.0147 | 1.0123 | 1.0013 | 1.0167 | 1.0450 | 1.0498 | 1.0214 |
| RL(g.) | 0.8900 | 0.9043 | 0.9020 | 0.8830 | 0.9290 | 0.9583 | 0.9387 |
| greedy | 0.8172 | 0.8090 | 0.8166 | 0.7933 | 0.7936 | 0.7800 | 0.8109 |

**Significance.** One-sided p on instance means for the E1 competitors run here: B4-eq 8.5e-10, B5 1.9e-05, RL(g.) 1.8e-28. All survive Holm over these three. B2, B3 and B6 were not run in this pilot. With failures scored as 200 (sensitivity): SPARC-L / RL(g.) 0.9128 [0.9058, 0.9196], RL(g.) arrival order 0.9204, RL(s.1) 0.9095, greedy 0.7564 [0.7326, 0.7792]. Every planner has 100% success, so for the planners the two analyses coincide.

**Weak cells.** Per setting x family, SPARC-L / B4-eq is not distinguishable from 1 in:
- SA-BT F1: 1.0006 [0.9952, 1.0059];
- SA-AT F2: 0.9964;
- MA-AT-50 F2: 0.9910;
- SA-AT F3: 0.9887 [0.9705, 1.0068].

RL(g.) beats SPARC-L on SA-AT F3: 1.0207 [1.0022, 1.0410], with SPARC-L winning 6 of 20 instances.

## 5. T1 and K2 in detail

T1 at every B4 budget (pooled F1-F3; each row is < 1 on 4/4 settings; the rule fails whenever pooled > 0.97):

| B4 budget (restarts per setting) | CPU per episode | SPARC-L / B4 | T1 |
|---|---|---|---|
| B4-r12 (12 on all settings) | 0.437 s | 0.9723 [0.9679, 0.9767] | fail |
| B4-impl (17 / 12 / 18 / 9) | 0.490 s | 0.9740 [0.9695, 0.9785] | fail |
| **B4-eq (47 / 44 / 49 / 39), pre-specified** | 1.260 s | **0.9847 [0.9806, 0.9890]** | **fail** |
| B4-2eq | 2.377 s | 0.9884 [0.9838, 0.9930] | fail |
| B4-4eq | 4.596 s | 0.9913 [0.9867, 0.9959] | fail |
| B4-eq, families weighted equally | | 0.9855 [0.9807, 0.9902] | fail |

- B4 keeps improving with budget, but slowly: B4-4eq / B4-eq = 0.9934 [0.9893, 0.9975].
- The B4-r12 row reproduces the K0 agent's post-fix preview (0.9723 [0.9679, 0.9767]) exactly, on the same code and seeds.
- T1 fails at every B4 budget tested: from 12 restarts and the as-implemented equal-CPU counts (9-18 restarts) up to 4x the construction-only count (156-196 restarts, about 10x SPARC-L's CPU per episode). So the K1 decision does not depend on the calibration choice or on the pending quiet-core measurement (K8).

**K2.** SPARC-L / insertion-only on F3 is 0.8279 [0.8167, 0.8390]. Per cell: pf0.1 0.8605, pf0.2 0.8109 [0.7945, 0.8276], pf0.2-R2 0.8131. By setting: MA-AT-25 0.8132, MA-AT-50 0.8367, SA-AT 0.7912, SA-BT 0.8724. SPARC-L / B4-eq on F1 is 0.9881, which is > 0.97. K2 needs both conditions, and the first is far from met, so K2 does not fire. Dynamic separation from *local insertion* is large (14%, 17% on F3). Separation from *re-planning by restarts* is 1.5%.

## 6. Where the margin comes from (EXPLORATORY: added after the seeds 0-1 snapshot; no gate uses these)

**Decomposition** (`cbba_sota.dyn.c1dyn.hybrid`, the pilots' RH-CA idea; 1,120 paired episodes each). A hybrid of a planner with itself reproduces that planner's episode bit for bit (tested).
- *Same t = 0 plan (SPARC-L's), different repairs:* SPARC-L / (SPARC-L plan, then B4-eq repairs) = 0.9849 [0.9810, 0.9889]. The SPARC-plan-then-B4 hybrid is no better than B4 itself: 0.9998 [0.9950, 1.0047].
- *Same repairs (SPARC-L's), different t = 0 plan:* SPARC-L / (B4-eq plan, then SPARC-L repairs) = 1.0000 [0.9950, 1.0050].
- So all of SPARC-L's margin over B4-eq comes from the warm ALNS repair in the loop. At 300 iterations, SPARC-L's t = 0 plan is no better than the best of B4-eq's 40-50 constructions (39-49 restarts plus the regret floor).

**t = 0 budget.** Both SPARC-L and B4 plan at t = 0 with the per-event budget; `iters0` is unspecified in the spec and defaults to the per-event budget. An ablation gives the first plan 10x for both methods: SPARC iters0 = 3000, B4 restarts0 = 10 x eq.
- SPARC-L+t0 / SPARC-L = 0.9915 [0.9875, 0.9953].
- B4-eq+t0 / B4-eq = 0.9995 [0.9954, 1.0037]: restarts do not scale, ALNS does.
- SPARC-L+t0 / B4-eq+t0 = 0.9768 [0.9721, 0.9815]: the T1 rule still fails.

**Relaxed keys** (SPARC-rk: committed relative order plus head-first, instead of spec 4.1's "every repaired key above every committed key"; day-1 finding):

| | Seeds 0-1 | Seeds 2-3 (fresh realizations of the same instances) |
|---|---|---|
| SPARC-L / SPARC-rk | 1.0135 [1.0081, 1.0189] | 1.0166 [1.0119, 1.0212] |
| SPARC-rk / B4-eq | 0.9717 [0.9665, 0.9768] (T1 rule fails by 0.0017) | 0.9665 [0.9614, 0.9720] (T1 rule passes; < 1 on 4/4 settings) |
| SPARC-L / B4-eq | 0.9847 [0.9806, 0.9890] | 0.9825 [0.9786, 0.9865] |

Relaxed keys help most under failures (seeds 2-3: SPARC-L / SPARC-rk on F3 = 1.0249) and on MA-AT-25 (1.036-1.038). SA-BT does not benefit (0.9899 and 1.0017). The day-1 claim that the relaxed rule preserves G1's precondition is agent B's (day 1) and has not been re-proved here. Seeds 2-3: 2 realizations were excluded (MA-AT-25 dev 13 seed 2, F3-pf0.2 and F3-pf0.2-R2; survivors cannot cover), paired for all methods.

**Heavy tier.**
- SPARC-H / B4-eq = 0.9627 [0.9579, 0.9674], passing the T1 rule, but at 3.69 s vs 1.26 s CPU per episode. SPARC-H is not an equal-budget comparator.
- SPARC-L / SPARC-H = 1.0229 overall, 1.0387 on F3 and 1.0559 on SA-AT F3. Of this, only 0.85 points come from the t = 0 plan: SPARC-L+t0 / SPARC-H = 1.0142. So the Light budget does not saturate on dev, and E2a (Light within 2% of Heavy) fails in this preview.

**Trigger rule.** Every-event RH beats structural-only SPARC by 1.0% on F1-F3 (1.9% on F3). Even on F0, where only noise events happen, every-event re-planning beats plan-once-and-follow: SPARC-L / every-event = 1.0250 [1.0151, 1.0349] on 480 pairs. This is not evidence against S2 ("noise alone does not justify re-planning"). With a 300-iteration first plan, extra re-plans mostly add search to a weak initial plan; the pilots behind S2 used a 20,000-iteration first plan (`iters0` = 20000 in every row of `pilots/trackD_robust/out/c2/*.jsonl`).

## 7. F3 (failures): the first real-env run

F3, 480 episodes per method (4 settings x 20 instances x 2 seeds x 3 cells). Per-episode means; abandons are partners leaving at a failure detection (lease = failure detector); restarts are aborted tasks restarting from scratch.

| Method | Success | Mean makespan | Re-plans | Structural events | Abandons | Aborted-task restarts | Wasted trips | Travel |
|---|---|---|---|---|---|---|---|---|
| SPARC-L | 480/480 | 42.81 | 19.7 | 38.1 | 14.67 | 2.35 | 0.19 | 85.92 |
| SPARC-H | 480/480 | 41.21 | 19.8 | 38.1 | 14.54 | 2.45 | 0.15 | 83.87 |
| B4-eq | 480/480 | 43.70 | 19.5 | 38.0 | 14.78 | 2.32 | 0.09 | 85.67 |
| B5 | 480/480 | 43.18 | 19.6 | 38.2 | 14.87 | 2.39 | 0.20 | 86.14 |
| every-event | 480/480 | 42.23 | 66.6 | 38.2 | 14.80 | 2.44 | 0.19 | 84.74 |
| insertion-only | 480/480 | 52.74 | 20.4 | 38.6 | 15.03 | 2.23 | 0.06 | 89.83 |
| open loop | 480/480 | 51.52 | 16.6 | 38.7 | 15.10 | 2.15 | 0.19 | 89.64 |
| RL(g.) | 477/480 | 44.93 | n/a | 29.4 | 12.79 | 2.46 | 0.00 | 101.47 |
| RL(g.), arrival order | 479/480 | 44.78 | n/a | 29.2 | 12.61 | 2.46 | 3.09 | 101.99 |
| greedy | 430/480 | 50.87 | n/a | 31.7 | 15.07 | 2.10 | 0.00 | 80.05 |

- All planners complete every F3 episode.
- The RL failures (3 decision-order, 1 arrival-order for RL(g.); 2 and 1 for RL(s.1)) are all MA-AT-25 dev 12 or 19, seed 1, pf0.2 or pf0.2-R2. They end at the 200 cap with completion 0.38-0.82: the livelocks agents A and D reported.
- Greedy fails 82 of its 1,120 F1-F3 episodes (50 of 480 on F3): 75 on MA-AT-25 and 7 on SA-AT F3.
- On seeds 0-1, no F3 realization was excluded (0 of 480).

## 8. Re-plans, churn and CPU per event (descriptive; loaded shared host)

F1-F3 pooled, 1,120 episodes per method. CPU is process time per controller call with a structural event (planners) or per decision (RL, greedy). It includes belief building and plan adoption. The per-event maximum includes one-time numba cache loading in a fresh worker.

| Method | Re-plans per episode | Route versions per episode | CPU p50 (ms) | CPU p95 (ms) | Median per-episode p95 (ms) | CPU per episode (s) |
|---|---|---|---|---|---|---|
| SPARC-L | 28.1 | 334.5 | 8.53 | 31.30 | 25.68 | 0.464 |
| SPARC-H | 28.2 | 328.1 | 68.36 | 287.53 | 223.71 | 3.692 |
| B4-eq | 27.7 | 403.2 | 34.57 | 70.27 | 59.22 | 1.260 |
| B4-impl | 27.7 | 411.3 | 11.21 | 29.18 | 23.03 | 0.490 |
| B5 | 28.0 | 245.0 | 7.95 | 29.43 | 23.82 | 0.427 |
| insertion-only | 29.1 | 132.2 | 2.06 | 4.37 | 3.81 | 0.123 |
| open loop | 20.8 | 131.5 | 1.84 | 4.36 | 3.60 | 0.140 |
| every-event | 70.8 | 597.6 | 8.56 | 31.38 | 25.44 | 0.969 |
| SPARC-rk (exploratory) | 27.4 | 317.7 | 10.32 | 38.65 | 30.76 | 0.557 |
| RL(g.) (per decision) | n/a | n/a | 7.32 | 12.28 | 9.25 | 3.124 |
| greedy (per decision) | n/a | n/a | 0.04 | 0.28 | 0.26 | 0.062 |

**Per setting.** SPARC-L p50 / p95 per event:
- MA-AT-25: 13.5 / 27.9 ms;
- MA-AT-50: 7.4 / 27.2 ms;
- SA-AT: 18.9 / 41.8 ms;
- SA-BT: 5.6 / 18.8 ms.

The largest per-episode p95 is 81.2 ms (F3). The single t = 0 plan in F0 (50 open tasks) costs 63.9 ms at p50. The published RL spends more CPU per episode than SPARC-L (3.1 s vs 0.46 s), because it pays about 7 ms on each of its roughly 400 decisions.

## 9. Secondary setting MA-AT-50-5-200 (descriptive, 20 instances x 2 seeds x F1-F3)

At 200 tasks B4's cold regret floor alone costs 36.9 ms per event, against 42.9 ms for all of SPARC-L. The equal-CPU count is therefore 3 restarts (2 as implemented; median-based 12).

| SPARC-L / method | Ratio [95% CI], 20 clusters |
|---|---|
| B4-eq (3 restarts) | 0.9873 [0.9787, 0.9964] |
| B4-impl (2) | 0.9807 [0.9744, 0.9874] |
| B4 at 12 restarts | 0.9969 [0.9892, 1.0054] |
| B5 | 0.9833 [0.9776, 0.9891] |
| insertion-only | 0.7855 [0.7695, 0.8019] |
| open loop | 0.8190 [0.8077, 0.8296] |
| every-event | 1.0358 [1.0260, 1.0451] |
| SPARC-H | 1.0436 [1.0384, 1.0489] |
| RL(g.) | 0.9565 [0.9396, 0.9758] |
| greedy | 0.8327 [0.8168, 0.8508] |

**Success.** SPARC-L 278/280, SPARC-H 279/280, B4 279/280, RL(g.) 279/280, insertion-only 270/280. The planner failures hit the 200 time cap. The shared failing case is dev 8, seed 1, F3-pf0.2 (17 of 50 robots fail), where every planner reaches the cap.

**CPU per event.** SPARC-L p50 34.4 ms, p95 65.6 ms; SPARC-H p50 300 ms, p95 604 ms.

At 200 tasks, SPARC-L is only 4.4% better than RL(g.), close to the K3 threshold (5%). Heavy is 4.4% better than Light, so the Light budget saturates even less.

## 10. Caveats

- **Dev only, two CRN seeds** (spec 9.1 plans three). Every exploratory arm was chosen after seeing the seeds 0-1 snapshot. The seeds 2-3 replication uses the same 80 instances, so it guards against realization noise, not instance selection.
- **Untuned baselines.** Neither B4 nor SPARC is tuned. The spec's ≤ 12 configurations per method are week-2 work. Tuning B4, for example its restart noise or cover rule, could only narrow SPARC's margin further.
- **Loaded-host CPU.** The budget calibration uses CPU ratios on identical states on a loaded host; the quiet-core calibration (K8) is pending. T1 fails over the whole range of B4 budgets (9 to 196 restarts), so K1 does not depend on it.
- **Implementation overhead in B4.** B4 rebuilds the kernel `Problem` for every restart (about 70% of its per-restart cost). B4-eq removes this from its charge, which is generous. A copy-based rewrite of B4 would change its CPU, not its plans.
- **B5** is an in-house reading of D-ITAGS (no public code).
- **B1 coalition rule.** On B1, the arrival-order rule helps RL by 0.7% (RL(g.)-arr / RL(g.) = 0.9928 [0.9905, 0.9950]). Both variants are reported, and K3 does not fire under either.
- **T4 (SPARC = R-a under good communication)** holds by construction: it is the same code path, with seeds from the belief digest. No separate R-a arm was run.
- **B3 (CP-SAT-LNS) was not run here.** It is day 3-4 work, and its deterministic-time calibration is pending.
- **K2 uses point estimates**, as the spec states it.

## 11. Reproduce

```bash
cd /home/jovyan/dev/cbba-sota
# 1. B4 calibration (writes runs/trackD/c1dyn/b4_budget.json; primary entries frozen in b4_budget_primary_frozen.json)
OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 taskset -c 8-39 .venv/bin/python scripts/trackD_c1dyn.py calibrate \
    --instances 0-19 --seeds 0 --procs 20
# 2. campaigns (scripts/trackD_run.py underneath; resumable by case id and env code hash)
runs/trackD/c1dyn/campaign.sh        # pre-specified arms: light planners, B4 levels, SPARC-H, F0, RL/greedy x 2 rules
runs/trackD/c1dyn/campaign2.sh x     # 200-task calibration, SPARC-rk, hybrids, 200-task planners
runs/trackD/c1dyn/campaign2.sh y     # 200-task B4, RL(g.), greedy
runs/trackD/c1dyn/campaign3.sh       # exploratory t = 0 ablation (seeds 0-1) and seeds 2-3 replication
# 3. analysis (tables above; CSVs copied here)
.venv/bin/python scripts/trackD_c1dyn.py analyze --copy-to docs/results/trackD-week1
```

Wall time: campaign 1 took 09:31-10:24 UTC; all four finished at 10:49 UTC. Total over all rows: 94,380 s wall, 78,907 s CPU.

## 12. Files

- Code:
  - `cbba_sota/dyn/c1dyn.py`: `ProbeSPARC`, `restarts_equal_cpu`, `Hybrid`/`hybrid` (runner plug-in), `b4_levels`, `labels`, `k1_decision`, `k2_decision`, `family_balanced`, `episode_summary`;
  - `scripts/trackD_c1dyn.py` (`calibrate`, `analyze`);
  - `tests/test_dyn_c1dyn.py` (9 tests).
- Rows (git-ignored): `runs/trackD/c1dyn/rows_*.jsonl` (13 files, 32,240 rows); `calib.jsonl`, `calib_200.jsonl`; logs in `runs/trackD/c1dyn/logs/`.
- Tables in this directory:
  - `c1dyn_ratios.csv`: every comparison x scope, with CI, median, HL, p-values, wins and McNemar;
  - `c1dyn_methods.csv`: per method x setting x family: success, makespan, re-plans, churn, CPU, travel, wasted trips, abandons, restarts;
  - `c1dyn_gates.csv`;
  - `c1dyn_summary.txt`;
  - `c1dyn_b4_budget.json`.
