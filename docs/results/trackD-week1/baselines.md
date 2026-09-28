# Track D week 1: RL-MPC, rolling CP-SAT-LNS tiers, auctions; gates K3 and K4 (agent D, days 3-4)

Date: 2026-09-28. Dev split only (dev instances 0-19 of the four primary settings). The test split was not touched.

**Where the numbers come from.** Every number below comes from the ground-truth env `DynTaskEnvX` (env code hash `6fabb0570f`, the hash gate K0 verified), never from a hand-written executor. Worlds: plan-following methods `DynTaskEnvX[arrival,causal,L=200,det]@6fabb0570f`; policies (RL) `DynTaskEnvX[decision,causal,L=200,det]@6fabb0570f`, and the same under the arrival-order rule where marked. The reference SPARC-Light, SPARC-Heavy, B4, B5 and B1 rows are agent B's C1-dyn pilot rows (`runs/trackD/c1dyn`, same code hash, same cases), reused as the task asked; this workflow's rows are in `runs/trackD/baselines_d`. Where B1 was run by both workflows (RL(g.) under the arrival rule, 1,120 cases), the rows are bit-identical (makespans equal case by case; `load_rows` checks this).

**Host.** Shared, load 80-180 on 224 CPUs; two shared H100s (cuda:0 and cuda:1) that other users were driving at 66-100%. Every CPU number is process time on this host and is descriptive; the tier budgets below are calibrated on it and are provisional until the quiet-core measurement (gate K8). At most 16 of my processes ran at once.

**Statistics.** `cbba_sota/dyn/stats.py` via `scripts/trackD_baselines.py report`: pairs are (setting, instance, cell, CRN seed); the unit is the instance mean of log r; cluster bootstrap over instances within settings (10,000 resamples, seed 0); r = makespan(SPARC-Light) / makespan(method), so < 1 favours SPARC-Light; pairs where both succeed (every method here had 100% success on these cases unless stated).

## 1. Decisions

| Gate | Rule (spec Section 8) | Result (dev) | Decision |
|---|---|---|---|
| **K3** RL anomaly, B1 | B1 within 5% of SPARC pooled (SPARC / B1 > 0.95) | RL(g.) 0.9144 [0.9074, 0.9214]; RL(g.) arrival rule 0.9209 [0.9141, 0.9275]; RL(s.1) 0.9104 [0.9036, 0.9171]; RL(s.1) arrival 0.9168 [0.9100, 0.9234] (1,120 pairs each, 80 clusters) | **does not fire** |
| **K3** RL anomaly, B2 | B2 within 5% of SPARC pooled | SPARC-L / B2-Heavy = **0.9061 [0.8934, 0.9191]** (171 pairs, 25 clusters, CRN seed 0); < 1 on 4/4 settings (0.880, 0.897, 0.919, 0.934) and 7/7 cells (0.882-0.922); CI upper 0.919 | **does not fire** |
| **K4** heavy beats light | B3-Heavy beats SPARC-Light by > 2% with the CI excluding 0 (SPARC-L / B3-H > 1.02, CI lower > 1) | SPARC-L / B3-Heavy = **0.9192 [0.8980, 0.9376]** (72 pairs, 24 clusters); B3-Heavy is 8% *worse* | **does not fire**: K4 does not drop the C2 "light" claim (E2a, SPARC-L vs SPARC-H, fails on its own in agent B's preview: 1.0229) |

**Reading.**
- Neither RL baseline comes within 5% of SPARC-Light. RL-MPC (B2) at the Heavy tier is no better than the published policy run online under structural dynamics, although it is 5-8% better than it when all tasks are known at t = 0 (Section 4.4).
- Rolling CP-SAT-LNS (B3) is far behind at both tiers (Light 0.885, Heavy 0.919): its per-event LNS on 12-40-task windows barely improves on the insertion floor it starts from, and it cannot fit the Light budget at all.
- **Flag for the K1/T1 decision (agent B found K1 fires).** The CBTA-style earliest-start auction (B6, about 3 ms per event, no search) is as strong as the rolling constructor B4-eq and as B5: SPARC-L / B6-CBTA(start) = **0.9864 [0.9796, 0.9932]**, B6-CBTA(start) / B4-eq = 0.9983 [0.9914, 1.0052], B6-CBTA(start) / B5 = 1.0051 [0.9984, 1.0119]. On SA-AT-50-5-50 it *beats* SPARC-Light: 1.0148 [1.0015, 1.0295]; on F3-pf0.2 1.0055 [0.9879, 1.0236]. So a cheap re-auction is within 1.4% of SPARC-Light pooled, like B4-eq (1.5%). It replicates on fresh CRN seeds 2-3 (run after seeing seeds 0-1, so a replication, not a pre-specified test): SPARC-L / B6-CBTA(start) = 0.9837 [0.9779, 0.9895], SA-AT 1.0061 [0.9957, 1.0169], F3 0.9921 [0.9802, 1.0042] (1,118 pairs; 2 realizations excluded for all methods). The spec expected B6 ≤ 0.95. No E1 "clear margin" (pooled ≤ 0.97) exists against B4, B5 or B6-CBTA(start).

## 2. What was built (`cbba_sota/dyn/baselines/`)

All four run through the shared plan-following controller (`controller.PlanController` in `DynTaskEnvX.run_plan`: trigger rule of spec 4.2, causal belief, kappa-scaled predictors, commit at departure, arrival-order coalitions, lease = failure detector), so infrastructure is equal by construction (spec 5.1). Nothing under `cbba_sota/solvers`, `hetero`, `bench` or `third_party` was edited. Runner plug-ins: `ctrl:cbba_sota.dyn.baselines.<module>:make_controller`.

### 2.1 B2 RL-MPC (`rl_mpc.py`, new)

- **Nominal state clone** (`NominalClone`, a `TaskEnv` subclass, so the released policy runs in its own env with native decision-order coalitions and native observations). Built only from causal information: the env's `StateView` plus the controller's `PlanState`.
  - Unreleased tasks are all-zero, masked, done rows (D1); finished tasks are finished rows; started tasks run to `max(start + nominal duration, now)`; other released tasks are open, with the robots that physically departed to them as members.
  - Known-failed robots are removed (rows deleted, never polled, as B1); an undetected failed robot is believed alive.
  - Travelling robots arrive at the belief's kappa-scaled predicted arrival; waiting and working robots stay; idle and home robots are free now; robots heading home or released from their task are free on arrival. Future travel in the clone is at `speed / kappa` (the spec 4.1 predictor shared by all methods).
  - Commitments (spec 4.1, shared rule): frozen members that have not departed go to their committed tasks first, in key order (forced steps, not policy decisions); a committed task their traits cover is masked for other robots until they have departed; a residual committed task stays open to the policy.
- **Rollouts.** The transcription of `Worker.run_episode` / `solvers/rl.py _episode` (pinned 4cf5e04: `git diff 4cf5e04 -- cbba_sota/solvers/rl.py` is empty) plus the forced steps; forward passes of a batch are batched on the GPU (lockstep, batch 16). Candidate 0 is the greedy decode (declared strong-baseline choice); the rest sample (RL(s.N)). Seeds: `hash(belief digest, event index)` (spec 4.1), rollout k seeded `SeedSequence([seed, k])`.
- **Budget (deterministic).** A number of policy decisions per event, spent in lockstep batches (the first always runs; a started batch finishes; at most 256 rollouts). Heavy tier = 8 cores x 1 s = 8 CPU-s per event at the measured CPU per decision, with the GPU forward passes on top ("8 cores *and* a GPU", the generous reading). Calibration (`calibrate-b2`, lockstep batch 16 on 6 mid-episode clones of one dev episode per setting): 1.375, 1.796, 1.336, 1.418 ms per decision, so `HEAVY_DECISIONS` = 5820 / 4450 / 5990 / 5640 (MA-AT-25 / MA-AT-50 / SA-AT / SA-BT). That is about 15-40 full 50-task rollouts at t = 0 and up to 256 late in an episode. Realized CPU per re-plan in the campaign: 12.6 / 10.2 / 10.8 / 9.8 s (MA-AT-25 / MA-AT-50 / SA-AT / SA-BT), 254-344 s per episode (above 8 s: clone building, pickling and batch overheads on the loaded host).
- **Plan.** The best rollout (every released task finished, lowest nominal makespan incl. return home) is converted to key-ordered routes: each released unstarted task gets the coalition that started it in the rollout, pruned to a minimal cover latest-arrival-first (robots physically committed to it are never pruned), keyed by its start rank. Start times along one robot's visits increase, so the keys are one consistent order (G1's precondition) and every physically committed robot has its task first. If no rollout finishes every released task, the plan is SPARC's insertion floor warm-started from the previous plan (counted; the campaign rows do not record the count; on 4 greedy-only (N = 1) diagnostic episodes it happened at 1 of 123 re-plans, and with 15-256 rollouts per event it needs every rollout to fail).
- **Speed without changing results.** `NominalClone` re-implements the observation builders, `task_update`, `agent_update` and `get_arrival_time` with the native float operations (distances precomputed per location with the native `np.linalg.norm(a - b)`, loop-invariant `get_matrix` calls hoisted). This is 2.3x faster per decision on the GPU path. Rollouts are bit-identical to the native methods: tests on static and mid-episode clones, and 40 mid-episode clones x 3 rollouts in `/tmp/agentD3/ident_mid.py`.
- **Bug found and fixed during calibration.** A robot whose pending start was taken kept its pending decision time. When every visible task was already covered, that made the clone loop forever at one epoch (SA-BT-50-5-50 dev 10, seed 1, F1-R3 at t = 0: 10 released tasks, 50 robots). Fix: restore the native `next_decision` when the pending start is taken (an idempotent `agent_update`). The fix was re-verified bit-identical on the static and mid-episode identity checks. Every B2 row in this document was produced after the fix.

### 2.2 B3 tiers (`cpsat_tiers.py`, new; `cpsat_rh.py` by agent B, unchanged)

B's `CPSATRH` bounds an event by the *summed CP-SAT deterministic time* of its sub-solves. That does not bound CPU: sub-solves that prove optimality report tiny deterministic times, so the loop builds many models in Python. Measured on MA-AT-25-5-50 dev 0 seed 0 F1-R2 (`scripts/trackD_run.py`, one episode each):
- `dtime=0.02`: per-event CPU p50 0.12 s, p95 7.8 s;
- `dtime=2.0` with 8 workers: p50 5.5 s, p95 70.5 s.

The tiers are therefore a fixed number of sub-solves per event, each with its own `max_deterministic_time`:

| Tier | Definition | Calibration | Realized CPU per re-plan (campaign) |
|---|---|---|---|
| Light (1 core; SPARC-L spends 16-31 ms per re-plan here) | 1 sub-solve of 0.02 dtime, 1 worker | A sub-model cannot be built in the Light budget (70-130 ms per sub-solve), so B3-Light *overspends* by 2-6x (declared, generous) | 52 / 92 / 161 / 140 ms (SA-BT / MA-AT-50 / SA-AT / MA-AT-25) |
| Heavy (8 cores x 1 s = 8 CPU-s) | sequential single-worker sub-solves of 0.05 dtime, 32 / 47 / 100 / 98 per event (MA-AT-25 / MA-AT-50 / SA-AT / SA-BT) | 8 s / measured CPU per sub-solve on one dev episode per setting (0.247, 0.171, 0.080, 0.082 s) | 7.4 / 6.8 / 19.8 / 5.4 s (MA-AT-25 / MA-AT-50 / SA-AT / SA-BT) |

Why single-worker sub-solves for Heavy:
- Interleaved 8-worker CP-SAT kept only about 2 of its 8 threads busy on these small sub-models (static MA-AT-25 dev 0: 7.1 CPU-s in 3.1 s wall).
- At roughly equal CPU per event, 24 single-worker sub-solves found at least as many improvements as 4 eight-worker ones (4 calibration episodes, `/tmp/agentD3/b3cal.py`: 4 / 14 / 27 / 33 improvements vs 1 / 9 / 25 / 25).
- It keeps the process count within the cap.

The realized per-re-plan CPU misses the 8 s target by setting: 0.7-0.9x on MA-AT and SA-BT, 2.5x on SA-AT. The single-episode calibration was imprecise. K4 is not sensitive to this: B3-Heavy is 8% behind SPARC-Light overall and 11% behind on SA-AT, where it got 2.5x the budget.

### 2.3 B5 D-ITAGS-style repair (agent B's `ditags.py`, unchanged)

B5 matches the spec's definition ("insertion plus ALNS restricted to coalitions touched by the event"). Its event region is the floor-placed tasks, plus the 5 nearest open tasks of each idle robot, closed once under shared membership. No rerun was needed: the C1-dyn pilot rows are reused, SPARC-L / B5 = 0.9914 [0.9877, 0.9950].

### 2.4 B6 coalition auctions (`cbta.py`, new; week-1 start)

Disclosed reimplementations (no public code for CBTA or CBGA). Under good communication a CBBA-family auction re-run to consensus reaches its sequential-greedy fixed point (Choi, Brunet, How 2009, for scores with diminishing marginal gain). Coalition timing breaks that condition, which is why CBTA adds timetables. Week 1 therefore implements the fixed points directly, one award per round. The message-level asynchronous version (bundle release, CBBA-PR partial reset) is needed only for the C3 arm CBTA-dec and is week-2 work.

All rules share SPARC's belief, anchors, residual repair (SPARC's insertion floor) and trigger rule. Only open tasks are auctioned, from scratch at every structural event.

| Rule | Definition |
|---|---|
| `cbta`, objective `start` (CBTA's own objective) | Append-only timetables. A robot's bid is its arrival time after its last timetabled task. A task's offer is its earliest start, from the earliest-arriving cover pruned to minimal. Each round the earliest start is awarded. |
| `cbta`, objective `makespan` (our variant) | The same, awarding the earliest finish |
| `cbga` (fidelity check only) | Travel-distance bids; the cheapest-travel group is awarded; timing is not bid |
| `seq` (the spec's declared fallback) | Coalition sequential auction with our insertion-kernel bids (`rh_kernels.insertion`, any slot after the anchored prefix); the cheapest is awarded (cheapest insertion) |

Deviation: the CBTA rules bid arrival times from the planner's state-start timing arrays, appended at the end of each timetable (CBTA's timetable semantics as we read them), not the insertion kernel's any-slot insertion. The insertion kernel is used by `seq`.

**Fidelity check** (`scripts/trackD_baselines.py fidelity-b6`, static dev 0-4 x 4 settings = 20 instances; mean of the t = 0 plan's model schedule):

| Rule | Mean start time | Mean makespan |
|---|---|---|
| CBTA-style (start) | 9.005 | 30.289 |
| CBTA-style (makespan) | 8.700 | 30.982 |
| CBGA-style | 44.414 | 105.687 |
| seq (cheapest insertion) | 11.726 | 33.091 |
| SPARC-L t = 0 plan (300 iterations), for reference | | 28.887 |
| insertion floor (regret-2), for reference | | 31.273 |

- CBTA-style has the lower average start time on **20 of 20** instances, so the check passes as specified.
- But our CBGA-style reading (bid on travel, ignore timing) is very weak. The check is therefore weak evidence of fidelity.

Status for the freeze: B6 is runnable and tested in its good-comms fixed-point form. Still to do: the CBTA-vs-CBGA check against a stronger CBGA reading, and the message-level version (week 2). If these are not ready at the freeze, the spec's rule makes B6 secondary and gives `seq` the confirmatory slot. **Caution:** `seq` (0.950) is a much weaker competitor than CBTA-style (start) (0.986), so that swap would flatter SPARC. We recommend reporting CBTA-style (start) as a competitor whatever its slot.

## 3. Design of the runs

| Item | Value |
|---|---|
| Cases | the C1-dyn pilot design: 4 primary settings x dev 0-19 x CRN seeds {0, 1} x F1-F3 (F1-R1, F1-R2, F1-R3, F2-R2N3, F3-pf0.1, F3-pf0.2, F3-pf0.2-R2) = 1,120 episodes per method |
| Full design | B3-Light, B6 (cbta-start, cbta-makespan, seq), B1 RL(g.) under the arrival rule (duplicate of B's, identical) |
| Subsamples (Heavy methods cost 8 CPU-s per event) | B2: dev 0-5, seed 0, all 7 cells = 168 episodes (24 clusters) (plus 3 MA-AT-25 dev 6 episodes that finished before the scope was cut from dev 0-7 to dev 0-5 for time; all kept: 171 episodes, 25 clusters); B3-Heavy: dev 0-5, seed 0, one cell per family (F1-R2, F2-R2N3, F3-pf0.2) = 72 episodes (24 clusters) |
| Replication (after seeing seeds 0-1) | B6-CBTA(start) and B6-seq on CRN seeds 2-3 (1,120 cases each; SPARC-L rows: `runs/trackD/c1dyn/rows_seeds23.jsonl`) |
| Diagnostic (descriptive) | B2 vs RL(g.) vs SPARC-L on static, F0-N1, F0-N3 (dev 0-3, seed 0; 48 episodes each) |
| Reference rows reused | SPARC-L, SPARC-H, B4, B5, B1 (both rules): `runs/trackD/c1dyn` |

## 4. Results: SPARC-L / method (< 1 favours SPARC-Light)

### 4.1 Pooled, per family, per setting

| Method | Pairs (clusters) | Pooled [95% CI] | F1 | F2 | F3 | MA-AT-25 | MA-AT-50 | SA-AT-50 | SA-BT-50 | CPU per re-plan |
|---|---|---|---|---|---|---|---|---|---|---|
| B2 RL-MPC, Heavy | 171 (25) | 0.9061 [0.8934, 0.9191] | 0.9049 | 0.9138 | 0.9029 | 0.8798 | 0.8967 | 0.9186 | 0.9344 | 9.8-12.6 s |
| B3 CP-SAT-LNS, Light | 1120 (80) | 0.8849 [0.8770, 0.8929] | 0.9149 | 0.8879 | 0.8549 | 0.8290 | 0.9085 | 0.8550 | 0.9523 | 52-161 ms |
| B3 CP-SAT-LNS, Heavy | 72 (24) | 0.9192 [0.8980, 0.9376] | 0.9579 | 0.9329 | 0.8692 | 0.8877 | 0.9261 | 0.8872 | 0.9789 | 5.4-19.8 s |
| B6 CBTA-style (start) | 1120 (80) | 0.9864 [0.9796, 0.9932] | 0.9793 | 0.9798 | 0.9958 | 0.9703 | 0.9799 | **1.0148** | 0.9812 | 2.9 ms (p50 per structural event) |
| B6 CBTA-style (makespan) | 1120 (80) | 0.9569 [0.9498, 0.9636] | 0.9497 | 0.9525 | 0.9656 | 0.9231 | 0.9600 | 0.9710 | 0.9743 | 2.8 ms |
| B6 seq (declared fallback) | 1120 (80) | 0.9497 [0.9447, 0.9546] | 0.9497 | 0.9443 | 0.9515 | 0.9271 | 0.9416 | 0.9610 | 0.9696 | 2.4 ms |
| B5 D-ITAGS-style (B's rows) | 1120 (80) | 0.9914 [0.9877, 0.9950] | 0.9953 | 0.9828 | 0.9904 | | | | | |
| B1 RL(g.) (B's rows) | 1117 (80) | 0.9144 [0.9074, 0.9214] | 0.8987 | 0.8830 | 0.9415 | | | | | |
| B1 RL(g.), arrival rule | 1119 (80) | 0.9209 [0.9141, 0.9275] | 0.9039 | 0.8922 | 0.9481 | | | | | |

One-sided p on instance means (H1: SPARC-L better): < 1e-3 for every pooled row above (B6-CBTA(start): 5.9e-4). B6-CBTA(start) is not significant on F3 (p = 0.27) and is reversed on SA-AT (p = 0.97).

### 4.2 B6 per cell (SPARC-L / method)

| Method | F1-R1 | F1-R2 | F1-R3 | F2-R2N3 | F3-pf0.1 | F3-pf0.2 | F3-pf0.2-R2 |
|---|---|---|---|---|---|---|---|
| CBTA-style (start) | 0.9794 | 0.9876 | 0.9709 | 0.9798 | 0.9988 | 1.0055 | 0.9833 |
| CBTA-style (makespan) | 0.9499 | 0.9505 | 0.9486 | 0.9525 | 0.9734 | 0.9747 | 0.9489 |
| seq | 0.9405 | 0.9530 | 0.9556 | 0.9443 | 0.9540 | 0.9504 | 0.9501 |
| B3-Light | 0.9292 | 0.8974 | 0.9184 | 0.8879 | 0.8792 | 0.8289 | 0.8574 |

### 4.3 B3 tiers against each other and against SPARC-Heavy (the 72 B3-Heavy cases)

- B3-Light / B3-Heavy = 1.0571 [1.0389, 1.0769]: Heavy is 5.7% better than Light (F3: 7.8%).
- SPARC-H / B3-Heavy = 0.9069 [0.8860, 0.9259]: at the same tier, SPARC is 9.3% better.

### 4.4 B2 against B1, and the F0 diagnostic (descriptive)

On the 171 B2 cases (pairs with B1 rows of the C1-dyn pilot):

| Ratio | Pooled [95% CI] | F1 | F2 | F3 | MA-AT-25 | MA-AT-50 | SA-AT-50 | SA-BT-50 |
|---|---|---|---|---|---|---|---|---|
| B2-Heavy / RL(g.) | 1.0172 [0.9995, 1.0339] | 1.0004 | 0.9904 | 1.0428 [1.0185, 1.0663] | 1.0307 | 0.9996 | 1.0385 | 0.9986 |
| B2-Heavy / RL(g.), arrival rule | 1.0240 [1.0074, 1.0394] | 1.0056 | 0.9953 | 1.0517 | 1.0415 | 1.0153 | 1.0385 | 0.9986 |
| B2-Heavy / RL(s.1) | 1.0112 [0.9949, 1.0270] | 0.9969 | 0.9834 | 1.0341 | 1.0222 | 0.9854 | 1.0237 | 1.0123 |

RL-MPC at the Heavy tier (10-13 CPU-s per re-plan, 250-340 CPU-s per episode) is no better than the released policy run online (about 7 ms per decision, 3 CPU-s per episode), and 4-5% worse under failures (F3).

F0 diagnostic (every task known at t = 0, so one plan per episode; dev 0-3 x 4 settings, seed 0, 16 pairs per cell; `runs/trackD/baselines_d/f0diag`):

| Ratio | static | F0-N1 | F0-N3 | pooled |
|---|---|---|---|---|
| B2 / RL(g.) | 0.9171 [0.9013, 0.9333] | 0.9420 [0.9226, 0.9647] | 0.9538 [0.9214, 0.9866] | 0.9375 [0.9266, 0.9474] |
| SPARC-L / B2 | 0.8778 [0.8536, 0.9004] | 0.8643 [0.8390, 0.8897] | 0.8936 [0.8695, 0.9189] | 0.8785 [0.8578, 0.8979] |
| SPARC-L / RL(g.) | 0.8050 | 0.8142 | 0.8522 | 0.8236 [0.8037, 0.8418] |

- Without structural events, RL-MPC's best-of-N plan beats the online policy by 5-8%, shrinking with noise.
- With structural events (F1-F3) the advantage is gone.
- The RL rollouts re-plan the whole team from a state the policy did not produce (frozen commitments, reserved tasks, robots stranded mid-plan), and the plan is then executed rigidly between structural events. The online policy instead reacts at every decision epoch.
- This is a property of the spec's B2 definition, not a defect we found. The clone is tested for causality, for consistency with the belief, and for bit-identity with the static and native paths. The risk-register item "RL-MPC seen as a strawman" (risk 6) applies, and the paper should say so.

### 4.5 B2 budget sensitivity (descriptive)

B2 at 3x the Heavy decision budget (`?scale=3`; 25 CPU-s per event p50, 593 CPU-s per episode) on 4 settings x dev 0-2 x seed 0 x {F1-R2, F3-pf0.2} = 24 episodes (12 clusters):
- SPARC-L / B2-3x = 0.9174 [0.9014, 0.9344];
- B2-3x / B2-1x = 0.9890 [0.9634, 1.0125] (F1-R2 1.0191, F3-pf0.2 0.9597);
- B2-3x / RL(g.) = 1.0269 [1.0103, 1.0439].

Three times the rollouts do not move RL-MPC measurably, so the K3 decision is not an artefact of the budget calibration.

## 5. Verification

- `tests/test_dyn_rl_mpc.py` (8 tests):
  - a static clone rolls out bit-identically to `solvers.rl.rollout` (greedy and sampled) and, in sampled batches, to `rl.lockstep`, with fast and native methods;
  - fast == native on mid-episode clones of an F3-pf0.2-R2 episode;
  - the clone is causal and consistent: hidden = unreleased, known-failed robots never move and are nobody's member, heads are members with the belief's arrival, and plans put heads first and leave failed robots unrouted;
  - the static open-loop plan is never later than the greedy RL rollout it came from;
  - episodes are deterministic and complete;
  - the policy refuses to plan without the `StateView`, and refuses a non-Heavy tier.
- `tests/test_dyn_cbta.py` (9 tests):
  - every auction plan satisfies `planner.check_plan` at every re-plan (F1-R2, F3-pf0.2-R2), and the episodes complete;
  - the CBTA offer is the earliest achievable start;
  - CBTA beats CBGA on average start time;
  - B3 tiers: per-setting sub-solve counts; one sub-solve per event at Light.
- 17 passed in 144 s. Full suite with these files: `OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 .venv/bin/python -m pytest tests/ -q` gave 332 passed in 644 s after the campaigns, and 333 passed in 672 s at the end (another workflow had added a test); `ruff check cbba_sota scripts tests`: all checks passed.

## 6. Caveats

- All budgets are calibrated on a loaded shared host and a shared GPU. Realized CPU per re-plan misses the 8 s Heavy target by setting: B2 got 1.2-1.6x the tier everywhere; B3-Heavy got 0.7-0.9x on three settings and 2.5x on SA-AT, where it is still 11% behind SPARC-Light. K3 and K4 are far from their thresholds, and B2 at 3x its budget is unchanged (Section 4.5), so the quiet-core recalibration (K8) is unlikely to flip them.
- B2 and B3-Heavy are subsamples (25 and 24 clusters, CRN seed 0 only). B3-Heavy covers one cell per family.
- B3-Light is not at the Light budget; nothing CP-SAT-based fits 16-31 ms per event in Python.
- B6 is a fixed-point implementation, not message-level; the fidelity check uses a weak CBGA reading. B6 and B5 are in-house reimplementations; wins over them are not external SOTA evidence (spec 10).
- The CBTA-style result is the strongest simple-rule evidence so far against a planner claim: it has no search, and on SA-AT it beats SPARC-Light.
- B2's greedy candidate 0 and the "8 cores *and* a GPU" budget reading are generous choices, declared.
- Spec 5.1 asks for per-method tuning with ≤ 12 configurations each. Not done: all methods ran at defaults.

## 7. Commands and files

```
# calibration
.venv/bin/python scripts/trackD_baselines.py calibrate-b2 --procs 4 --clones 6     # (run per case; see Section 2.1)
.venv/bin/python /tmp/agentD3/b3cal.py {heavy8|heavy1|light1|sparc}                # B3 sub-solve CPU (scratch)
# campaigns (dev), rows in runs/trackD/baselines_d
runs/trackD/baselines_d/campaign_cheap.sh     # B6 x3, B3-Light, RL(g.) arrival: 1,120 cases each
runs/trackD/baselines_d/campaign_b3heavy.sh   # B3-Heavy, 72 cases
INST=0-5 runs/trackD/baselines_d/campaign_b2.sh   # B2 Heavy
runs/trackD/baselines_d/campaign_f0diag.sh    # F0 diagnostic
.venv/bin/python scripts/trackD_baselines.py report runs/trackD/baselines_d/f0diag --families F0 static --seeds 0 \
    --ref 'B2[heavy]' --methods 'RL(g.)'          # (and --ref 'SPARC[light]' --methods 'B2[heavy]' 'RL(g.)')
runs/trackD/baselines_d/campaign_b6seeds23.sh # B6 replication on CRN seeds 2-3
runs/trackD/baselines_d/campaign_b2x3.sh      # B2 at 3x budget, 24 cases
.venv/bin/python scripts/trackD_baselines.py report runs/trackD/c1dyn runs/trackD/baselines_d --seeds 0 \
    --ref 'B2?scale=3[heavy]' --methods 'B2[heavy]' 'RL(g.)'
.venv/bin/python scripts/trackD_baselines.py report runs/trackD/c1dyn runs/trackD/baselines_d --seeds 2 3 \
    --methods 'B6?rule=cbta&objective=start' 'B6?rule=seq'
# analysis
.venv/bin/python scripts/trackD_baselines.py fidelity-b6 --instances 5
.venv/bin/python scripts/trackD_baselines.py report runs/trackD/c1dyn runs/trackD/baselines_d --methods ...
# docs/results/trackD-week1/baselines_summary.txt concatenates the outputs of the report commands above (its
# section headers give each command) and of fidelity-b6
```

Rows (git-ignored): `runs/trackD/baselines_d/rows_{b6,b3light,b1arrival,b3heavy,b2,b2x3,b6_seeds23}.jsonl` and `runs/trackD/baselines_d/f0diag/rows_*.jsonl`; each row carries its world (`DynTaskEnvX[...]@6fabb0570f`), env code hash, git commit (`1f0f0462bafb`, with `git_dirty_dyn = true`: the baseline modules are uncommitted) and the method string with its parameters. `rl_mpc.py` changed after the main B2 campaign only by the `scale` parameter (default 1, the identity); `cbta.py` and `cpsat_tiers.py` are unchanged since their campaigns.

Code: `cbba_sota/dyn/baselines/{rl_mpc,cpsat_tiers,cbta}.py`, `scripts/trackD_baselines.py`, `tests/test_dyn_{rl_mpc,cbta}.py`.
