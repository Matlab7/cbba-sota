# Track D week 1: decision memo (spec Section 8, day 7)

Date: 2026-09-28. For the user's decision; nothing here is committed by the memo author. This memo replaces the interim
memo of commit `02ccc51`, which was written from partial result files after a host restart. The interim memo said that
`c3.md` had not been written; the C3 agent has since finished it.

**Scope and provenance**
- **Data.** Dev split only: dev instances 0-19 of the four primary settings, with MA-AT-50-5-200 as a secondary
  setting. No test-split instance was touched. Track D has never used the validation split (`val`, seed base 900000,
  created by Track S in `0cb50d5`): every one of the 86,075 Track D rows that records a split says `dev` **(m)**.
- **World.** Every number comes from the ground-truth env `DynTaskEnvX` at env code hash `6fabb0570f`. That is the hash
  K0 verified, and it is still the hash of today's working tree. The worlds are:
  - plan-following methods: `DynTaskEnvX[arrival,causal,L=200,det]@6fabb0570f`;
  - policies: `DynTaskEnvX[decision,...]`, plus arrival-order variants;
  - C3: `C3Env<DynTaskEnvX[...]@6fabb0570f>@5bc2c415b9` for the T2 arms and `@26d92033cd` for FULL and the
    ρ ∈ {1.25, 0.75} extension.

  No hand-written executor produced any reported number.
- **Planner kernel.** The kernel is pinned at `4cf5e04`. That is the work-in-progress "v2" kernel of `103da3a`, not the
  committed ALNS v2 of Track S (`0e449fb`); see open problem 1.
- **Ratios.** r = makespan(SPARC-L) / makespan(method), a geometric-mean paired ratio with a cluster-bootstrap 95% CI
  over instances; r < 1 favours SPARC-L. "X% behind" or "X% worse" means 1 − r (or r − 1 when r > 1), as in the agents'
  reports.
- **CPU numbers** are process time on a loaded shared host (load 60-180) and are descriptive only.
- **Verification.** I recomputed the gate numbers from the raw rows with an independent script,
  `scripts/trackD_week1_verify.py`. It does its own pairing and its own bootstrap and does not import
  `cbba_sota.dyn.stats` or the agents' analysis scripts.
  - Every recomputed point estimate matches the agents' reports to the printed precision.
  - Bootstrap CI ends differ by at most 0.0004, because the RNG differs.
  - Section 8 lists what was recomputed. Other secondary numbers are quoted from the named result files: the t = 0
    ablation, the connected-T2 and beacon runs, and the C3 design pilots.
  - **(m)** marks numbers computed or recounted for this memo. Section 8 names the ones that appear in no agent
    report.

## 0. Decision summary

| Outcome on dev | Consequence (spec Section 8 and 9.5) |
|---|---|
| K0 passes | Numbers on env `6fabb0570f` are not provisional for fidelity reasons. Day-1 rows on `4b70840e7e` are superseded. |
| **K1 fires.** T1 fails: SPARC-L / B4-eq = 0.9847 [0.9806, 0.9890] | Drop the dynamic-method claim (C1-dyn). Track D becomes an analysis appendix to the static paper. |
| **E2a fails.** SPARC-L / SPARC-H = 1.0229, 90% CI [1.0189, 1.0269] | C2-dyn ("the Light budget saturates") is not supported. Under 9.5, T1 and E2a both fail, so Track D is not method evidence. |
| K2, K3 and K4 do not fire | No further kill and no RL anomaly to investigate. K4 does not drop C2 on its own; E2a already does. |
| K5 does not fire | C3 has headroom: the best fixed architecture is about 1.8 × FULL at ρ = 0.5. Keep the benchmark. |
| **K6 fires on both clauses** | C3 is descriptive. The spec says to recommend DS, but on dev DS is dominated by the fixed rule INF-r (Section 3). |
| K7 fires for rankings | Report the lease grid. Claim only what holds across it: C3 has headroom, and SPARC does not separate. |
| K8 is deferred | No compute claim until it is measured on a quiet pinned host after the other workflow finishes. |
| K9 passes with a narrowed claim | Only the narrow "first evaluation of the HeteroMRTA benchmark under dynamics" wording survives (Section 4). |

**Bottom line.** Track D yields an analysis appendix. It supports no method claim and no confirmatory C3 claim. The
paper's contribution remains the static planner (Track S).

## 1. Gates K0-K9

| Gate | Rule (spec 8) | Dev result | Status | Evidence |
|---|---|---|---|---|
| **K0** fidelity | Executor vs env replay on ≥ 20 episodes per family, to 1e-6 for minimal covers. Arrival order = decision order at ideal comms. Static reduction bit-identical. | Executor == env on **12,760 / 12,760** episodes (static 1,160; F0 3,480; F1 3,480; F2 1,160; F3 3,480), with max \|dt\| = 0 and equal counters. Planners: 7 Light planners, SPARC-H and B3. Decision == arrival order on **12,368 / 12,368** minimal-cover episodes. The 392 F3 episodes with non-minimal coalitions from residual repair are descriptive (269 identical). RL static reduction: the stored Phase-1 RL(s.64) samples are reproduced on 80/80 instances (5,120 rollouts); runner RL(g.) and RL(s.1) == `rl.rollout` on 80/80. | **pass** | `runs/trackD/k0/summary.txt`, `rows.jsonl`, `rl_static.jsonl`, all recounted **(m)**; the RL(s.64) multisets were re-compared with `runs/rl/<setting>/dev.jsonl` and are equal on 80/80, with the best equal on 80/80 **(m)**. Hashes: env `6fabb0570f`, executor `56a085a816`, controller `80bf534371`. The env and executor hashes are unchanged today, and `git diff 1f0f046` of every file the runner imports is empty **(m)**. |
| **K1** planner collapse | T1: SPARC-L / B4 at the same Light budget, F1-F3 pooled ≤ 0.97, cluster-CI upper bound < 1, and < 1 on ≥ 3/4 settings | **0.9847 [0.9806, 0.9890]** against the pre-specified B4-eq (construction-only equal CPU: 47 / 44 / 49 / 39 restarts). < 1 on 4/4 settings (0.978, 0.982, 0.990, 0.989); p = 8.5e-10. The whole CI lies above 0.97. T1 fails at every B4 budget: B4-r12 0.9723, B4-impl 0.9740, B4-2eq 0.9884, B4-4eq 0.9913. Family-balanced: 0.9855. Fresh CRN seeds 2-3: 0.9825 [0.9786, 0.9865]. | **fires** | `docs/results/trackD-week1/c1dyn.md` §1, §5; `c1dyn_gates.csv`; recomputed **(m)** |
| **K2** no dynamic separation | SPARC / insertion-only > 0.98 on F3 **and** SPARC / B4 > 0.97 on F1 | F3: **0.8279 [0.8167, 0.8390]**, 80/80 instances. F1: 0.9881. Only the second condition holds. | does not fire | `c1dyn.md` §5; recomputed **(m)** |
| **K3** RL anomaly | B1 or B2 within 5% of SPARC, pooled | B1 RL(g.) **0.9144 [0.9074, 0.9214]**; arrival order 0.9209; RL(s.1) 0.9104 and 0.9168 (1,117-1,119 pairs, 80 clusters). B2 RL-MPC Heavy **0.9061 [0.8934, 0.9191]** (171 episodes, 25 clusters, CRN seed 0 subsample); B2 at 3× budget 0.9174 (24 episodes). | does not fire | `baselines.md` §1, §4; recomputed **(m)** |
| **K4** heavy beats light | B3-Heavy better than SPARC-L by > 2%, CI excluding 0 | SPARC-L / B3-Heavy **0.9192 [0.8980, 0.9376]** (72 episodes, 24 clusters, one cell per family, seed 0): B3-Heavy is 8% *worse*. B3-Light: 0.8849. | does not fire | `baselines.md` §4; recomputed **(m)** |
| **K5** C3 headroom | At ρ = 0.5, the best fixed architecture within 5% of FULL | Best fixed is INF-r in both families: F1-R2 **1.853 [1.815, 1.894]** and F3-pf0.2 **1.794 [1.754, 1.835]** × FULL (80 clusters). Pooled over families 1.823. | does not fire | `c3.md` §7.2; `c3_gates_4settings.txt`; recomputed **(m)** |
| **K6** C3 separation | SPARC's max regret not ≥ 2 points below the best fixed architecture's, **or** DS ≤ SPARC | See the note below. | **fires** (both clauses) | `c3.md` §7.1, §7.3, §7.10; `c3_gates_*.txt`; recomputed, plus the both-succeed sensitivity **(m)** |
| **K7** lease fragility | Any C3 conclusion flips within the lease grid | See the note below. | **fires for rankings** | `c3_gates_4settings.txt` (per-lease blocks); `c3.md` §7.4 |
| **K8** compute | Light p95 > 100 ms at 50 tasks, or Heavy SPARC > 1 s at 200 tasks, on one quiet core | See the note below. | **deferred** | recomputed from `runs/trackD/c1dyn` rows **(m)** |
| **K9** novelty | An unread HeteroMRTA citer already evaluates dynamic HeteroMRTA against a rolling optimizer | 62 citers screened: 48 by abstract, 14 by title or snippets. A Google Scholar full-text search for "HeteroMRTA" returns 2 documents (SADCHER and the original paper). No hits. DODRL-HMRTA is prior art for dynamic heterogeneous MRTA on its own instances (full text unread). | pass, narrowed claim | `docs/trackD-novelty.md` |

**K6 in detail.**
- The dev-selected SPARC variant is V1, which is HYB. Its max regret is **1.192**; the best fixed architecture, INF-r,
  has **1.006**. SPARC is therefore **18.6 points worse**, not 2 points better.
- DS (ρ\* = 2) has 1.170, which is below SPARC's 1.192.
- E3a preview: difference +0.186, one-sided upper bound +0.217.
- The verdict is the same in three other views:
  - on the five-level ρ grid over the first two settings: V1 1.177, INF-r 1.086, DS 1.120;
  - in every setting separately;
  - on episodes where every arm succeeds **(m)**: HYB 1.161, INF-r 1.006, DS 1.150.

**K7 in detail.**
- The K5 and K6 verdicts are the same under all 9 common leases and under the tuned set. SPARC's K6 margin ranges from
  −7.9 to +1.1 points and never reaches +2.
- Four rankings flip:
  - the selected variant (V1, V2 or V3);
  - the best fixed architecture (INF-r, REP-clamp or HYB);
  - DS's ρ\* (2 or ∞);
  - whether SPARC beats CEN-F at ρ = 0.5 (it does not under L = 10).
- These per-lease results come from the tuning rows, which have only 10 clusters.

**K8 in detail.** K8 has not been measured. It is deferred to a quiet pinned host after the other workflow finishes.
Loaded-host process CPU per structural event **(m)**:
- SPARC-L at 50 tasks: p50 8.5 ms, p95 31.3 ms (41,196 events); the largest per-episode p95 is 81.2 ms.
- SPARC-H at 200 tasks: p50 300 ms, p95 604 ms; per-episode p95 up to **1.52 s**.

## 2. Collapse checks (spec Section 7) and endpoint previews

| Test | Dev result | Status |
|---|---|---|
| T1 planner vs rule | fails (K1 above) | fail, so K1 fires |
| T1b SPARC vs insertion-only | 0.8598 [0.8510, 0.8687] (F1 0.896, F2 0.851, F3 0.828); 80/80 instances | descriptive: search pays on structural events, confirming S6 in the real env |
| T2 trigger rule, SPARC vs every-event RH | 1.0098 [1.0052, 1.0145] (F3 1.0191); 28.1 vs 70.8 re-plans per episode | descriptive: every-event re-planning is 1% *better*, reversing the pilot prior S8 |
| T3 F0 noise control | 480/480 pairs identical; GM 1.0000; 1 re-plan per episode | pass (identity) |
| T4 decision difference | Good comms: holds by construction (same code path, digest seeds); no separate R-a arm was run. The C3 log was not run. | partial |
| T5 factorial, T6 planner under partitions, T7 consistency machinery, T8 CBBA layer | not run | open |
| E1 preview | SPARC-L is below 1 with CI upper < 1 against every competitor run. B4-eq 0.9847; B5 0.9914 [0.9877, 0.9950]; B6 CBTA-style (start) 0.9864 [0.9796, 0.9932]; B3-Light 0.8849; B1 0.9144; B2 0.9061. Every one-sided p ≤ 5.9e-4. | significant, but no "clear margin" (≤ 0.97) against B4, B5 or B6 |
| E2a preview | SPARC-L / SPARC-H 1.0229, TOST p = 0.86; F3 1.0387; 200 tasks 1.0436 | fails |
| E2b preview | SPARC-L / B3-Heavy 0.9192 (subsample) | superiority holds, but not on the confirmatory design |
| E3a preview | +0.186 (upper bound +0.217) | fails |

## 3. What each outcome implies (spec Section 8)

**K0 passes.**
- Pilot numbers on `6fabb0570f` stand.
- Day-1 rows on `4b70840e7e` are superseded: A's campaign, D's B1 rows and B's day-1 pilot. On a 440-episode sample,
  policy trajectories are identical across the two envs on F0-F2 but not on F3.
- B's day-1 T1 value (0.9559) came from an env that missed release epochs and must not be quoted.
- The executor may be used for speed only while the env, executor and controller hashes are those verified.

**K1 fires, so the dynamic-method claim is dropped.** The spec's fallback is an appendix "with S2, S6 and S8 as
findings". The real env changes that list:
- **S6 is confirmed.** SPARC-L / insertion-only = 0.860 pooled and 0.828 on F3.
- **S8 is reversed.** Every-event re-planning is 1.0% better than structural-only re-planning (1.9% better on F3).
- **S2 was not re-tested at its own budget.** On F0, every-event re-planning beats plan-once-and-follow by 2.5%
  (SPARC-L / every-event = 1.0250 [1.0151, 1.0349]). Here the first plan gets only 300 iterations, while the S2 pilots
  used 20,000-iteration first plans. The appendix must either state S2 with that budget condition or re-test it.

The K1 failure is not borderline:
- The whole CI is above 0.97.
- It holds for B4 budgets from 9 to 196 restarts, under family weighting, and on fresh CRN seeds.
- A CBTA-style re-auction with no search (B6-start, about 3 ms per event) comes equally close (0.9864; B6-start /
  B4-eq = 0.9983). It beats SPARC-L on SA-AT-50: 1.0148 [1.0015, 1.0295].
- Exploratory decomposition: all of SPARC-L's margin over B4 comes from warm ALNS repair in the loop. The t = 0 plans
  are equal (1.0000).

**K2 does not fire.** Separation from local insertion repair is large, 17% on F3, so "search pays on structural
events" is supported. Separation from re-planning by restarts is only 1.5%.

**K3 does not fire, so there is no bug to chase.**
- The released RL policy online is 8.6% behind SPARC-L.
- RL-MPC is no better than RL online under structural events: B2 / RL(g.) = 1.0172 [0.9995, 1.0339], and 1.0428 on
  F3. It helps (by 5-8%) only when every task is known at t = 0.
- Near misses to watch:
  - RL(g.) beats SPARC-L on SA-AT F3: 1.0207 [1.0022, 1.0410].
  - At 200 tasks, SPARC-L / RL(g.) = 0.9565 [0.9396, 0.9758].

**K4 does not fire, but E2a fails on its own.**
- Light (300 iterations) is 2.3% behind Heavy (3000 iterations): 3.9% behind on F3 and 4.4% at 200 tasks.
- Only 0.85 points of the gap come from the t = 0 plan: SPARC-L with a 10× first plan / SPARC-H = 1.0142.
- C2-dyn is therefore not supported.
- SPARC-L does beat rolling CP-SAT-LNS at the Heavy tier (8%), but on a 24-cluster subsample.

**K5 does not fire, so keep the C3 benchmark.** T2 comms cost 1.8-2.3 × FULL at ρ = 0.5.

**K6 fires, so C3 is descriptive.**
- The spec says "DS is recommended". On dev, DS (1.170) is dominated by INF-r (1.006):
  - in every setting;
  - on the five-level grid;
  - on the both-succeed subset.

  INF-r is the rule "outsiders keep only their physically committed head; unreachable robots' stale commitments are
  not honoured", which is S12's mechanism.
- **Structural note.** The variant tie rule selected V1, which *is* HYB, a member of the fixed set A. With V1 selected,
  the first K6 clause cannot pass: SPARC's max regret equals HYB's, which is ≥ the minimum over A. Here the choice
  does not matter, because no variant (V1 1.192, V2 1.205, V3 1.184) is within 17 points of INF-r. The spec should
  still exclude fixed architectures from the variant set, or say explicitly that selecting one ends C3.

**K7 fires.** Report the grid, and claim only "C3 has headroom (no K5 kill)" and "SPARC does not separate (K6)".

**K8 is deferred.** Make no compute claim.
- The loaded-host numbers suggest the Light threshold would hold at 50 tasks: p95 31 ms against 100 ms.
- The Heavy threshold at 200 tasks is not safely met even before the quiet-host measurement: per-episode p95 reaches
  1.52 s.

**K9 passes.** Use the narrow wording only (Section 4).

**Spec 9.5.** Track D enters the paper as method evidence only if E1 (against every competitor), T1 and E2a all hold.
T1 and E2a fail. C3 is claimed only if E3a holds, and it fails. So Track D contributes no confirmatory endpoint.

## 4. Claims that survive

| Claim | Status | What remains |
|---|---|---|
| C1-dyn (primary) | **dropped** (K1) | descriptive appendix findings, listed below |
| C2-dyn | **not supported** (E2a fails; K8 deferred) | descriptive: Light is 2.3% behind Heavy, and beats rolling CP-SAT-LNS Heavy by 8% on a subsample |
| C3 (conditional) | **descriptive** (K6 fires, E3a fails; K5 does not kill) | benchmark plus diagnosis |
| Static C1 / C2 (Track S) | not affected by Track D | the paper's contribution |

**Descriptive findings the dev data support** (for an appendix; dev only, provisional until any frozen run):
1. **Search in the loop pays on structural events.**
   - SPARC-L / insertion-only = 0.860 pooled and 0.828 on F3.
   - SPARC-L / open loop = 0.883.
2. **A cheap rolling rule at equal CPU comes within 1.5%.**
   - B4-eq 0.985, B5 0.991, B6 CBTA-style (start) 0.986.
   - B6 beats SPARC-L on SA-AT-50.
3. **The published RL policy run online is 8-9% behind.**
   - SPARC-L / RL(g.) = 0.914 (0.921 under arrival order).
   - Dynamics cost RL(g.) 10-34% against its own static makespan **(m)**: F1-R2 1.102, F1-R1 1.219, F2-R2N3 1.336,
     F3-pf0.2 1.227, F3-pf0.2-R2 1.309. Env `6fabb0570f`, CRN seed 0, same policy seed as the static run.
   - RL-MPC does not close the gap.
4. **Rolling CP-SAT-LNS is 8-12% behind at both tiers** (Light 0.885, Heavy 0.919).
5. **Trigger rule.**
   - Every-event re-planning is 1% better than structural-only.
   - On F0 (noise only), SPARC reduces exactly to open loop.
6. **The Light budget does not saturate.** Light / Heavy = 1.023.
7. **C3.**
   - The headroom is large (1.8 × FULL at ρ = 0.5).
   - On F3, the cost comes mostly from failure suspicion (rule 4.6), not from partitions: fully connected T2 is
     1.32-1.59 × FULL on F3, against 1.02-1.04 on F1.
   - A one-line rule (INF-r) has the lowest max regret, 1.006, and pays for it in wasted trips (111 per episode on
     F1-R2 at ρ = 0.5).
   - Leases and a beacon move results as much as the architecture does on F1. K7; beacon ratios 0.805-1.036 on
     F1-R2.

**Allowed wording (K9).** "The first evaluation of the HeteroMRTA benchmark (its instance generator, settings and
released policy) under online task release, execution delays and robot failures, against rolling optimizers at equal
per-event compute."

**Not allowed:**
- "SPARC is the best online policy";
- "clear margin";
- "first dynamic heterogeneous or coalition MRTA study";
- "the Light budget saturates";
- "SPARC's protocol has the lowest worst-case regret";
- any message-count claim;
- recommending SPARC for degraded networks.

The spec's own "explicitly not claimed" list still applies.

## 5. Proposed frozen parameters (spec 4.7 and the conventions found in week 1)

These matter only if a Track D number goes into the paper from a validation or test run.

**Planner, env and baselines**

| Parameter | Proposed frozen value | Basis |
|---|---|---|
| ALNS iterations per event, Light / Heavy | 300 / 3000 | spec 4.7, not tuned. E2a failed, so Light is not claimed to saturate. |
| First-plan budget `iters0` | equal to the per-event budget (300 / 3000) | The spec is silent. Every week-1 run used this value. The 10× t = 0 ablation is a sensitivity (SPARC-L+t0 / SPARC-L = 0.9915). |
| Key rule | monotone (spec 4.1, G1 precondition) | Relaxed keys (1.4-1.7% better) are exploratory; adopting them now is a forking path (open problem 2). |
| Kernel | `4cf5e04` as run, **or** re-pin to `0e449fb` and re-run the kernel check, K0 and T1 before reporting | open problem 1 |
| Trigger rule | structural events only (release, orphan, idle, membership); no noise triggers | spec 4.2; report T2's reversal |
| Predictors | κ = 1 / (1 − expected stall fraction); nominal residual durations | spec 4.1 |
| Heartbeat; failure detector under good comms | 1.0; h = 5 ticks (detection 0.5 after onset) | spec 4.7 |
| Lease under good comms | lease = failure detector (L = 200, abandon at detection) | spec 4.3 |
| Release window H | 0.5 × the published RL(g.) makespan, unrounded: 25.079 / 14.812 / 19.7135 / 12.1165 (MA-AT-25 / MA-AT-50 / SA-AT / SA-BT). R1: H = 20 at 50 tasks, 90 at 200 tasks. | spec 3.3; used in every week-1 run |
| Failures | p_f ∈ {0.1, 0.2}; onset U[0, 0.7 × published RL(g.)]; detection 0.5 after onset | spec 3.3 |
| Failure exclusion | a realization is excluded when the robots that never fail cannot cover every task (method-independent, paired) | 0 exclusions on seeds 0-1; 2 on seeds 2-3 |
| Delay and duration models | N1 {0.01, 1-4 ticks}; N3 {0.05, 1-10}; lognormal durations σ = 0.3; N4 and the duration shapes secondary | spec 3.3 |
| Coalition rule | Plan-following: arrival order (D2). Policies (B1, B7): report both rules, primary = arrival order. | Spec 5.1's shared executor; arrival order favours RL by 0.7% (0.9928 [0.9905, 0.9950]). **User decision.** |
| Failed robots in RL observations | rows **deleted**, not zeroed | A zero agent row 0 makes the released network return NaN; spec 3.1 and 5.2 wording must change |
| B4 (Light) | B4-eq restarts 47 / 44 / 49 / 39; 3 at 200 tasks (`runs/trackD/c1dyn/b4_budget_primary_frozen.json`) | Loaded-host calibration; re-derive at K8. The T1 verdict is insensitive (9-196 restarts). |
| B3 tiers | Light: 1 sub-solve × 0.02 dtime, 1 worker (overspends Light 2-6×, declared). Heavy: 32 / 47 / 100 / 98 single-worker sub-solves × 0.05 dtime. | Summed dtime does not bound CPU; re-derive at K8 |
| B2 (Heavy) | 5820 / 4450 / 5990 / 5640 policy decisions per event; lockstep batch 16; candidate 0 greedy; ≤ 256 rollouts | "8 cores *and* a GPU", the generous reading; re-derive at K8 |
| B6 slot | CBTA-style (start) as the competitor whatever the fidelity rule decides; `seq` and CBTA-style (makespan) secondary | The spec's fallback `seq` (0.950) would flatter SPARC. **User decision.** |

**C3 (T2)**

| Parameter | Proposed frozen value | Basis |
|---|---|---|
| Channel | tick 0.1; one tick per hop; GE p = 0.2, ρ_GE = 0.8; 1500 B per node per tick; digest = the full version vector; CRN stream 5; range ρ·r_c(n) with n = robots + 1; station at (0.5, 0.5); heartbeat 1.0 | agent C's conventions, cross-checked against cbja `network.py` |
| Leases (g, L) per arm and ρ | `runs/trackD/c3/tuned.json` (md5 `24071d2685ee01687dfd76d5880e8c16`) for ρ ∈ {2, 1, 0.5}; `tuned_ext.json` (md5 `72f49b0a6ed7a1f51b40050b9068024a`) for ρ ∈ {1.25, 0.75} | Frozen before evaluation, and the evaluation rows use exactly these leases **(m)**. Tuned on MA-AT-25 and SA-BT-50 only. |
| Stranded-component variant | V1 | dev tie rule: V3 1.184, V1 1.192, V2 1.205 (see the K6 note) |
| DS ρ\* | ∞ (REP-clamp at every ρ) | Ties ρ\* = 2 on 4 settings × 3 ρ levels (1.170) and is better on the five-level grid (1.120 vs 1.129). **User decision**; either way INF-r dominates DS. |
| h_f; silence; member window; membership debounce; θ (V3) | 10; 2.0; 1.5; 0.5; 0.7 | Spec 4.7 lists only h_f. The debounce is new and must be declared. |
| Liveness conventions (each found on a dev episode and regression-tested) | See the list below. | `c3.md` §4 |
| Adoption | T2 keeps its commitments; FULL follows the version | design seed 2: keep 82.4 vs follow 83.9 GM |
| Station outsiders | anchor model (spec literal) with monotone keys | Design seed 2; anchor vs head differs by 5-16%, not consistent in sign |

The C3 liveness conventions:
- Termination uses a global 1-bit mission-complete signal. The per-replica D7 rule has no liveness under partitions.
- Rule 4.6(c): an overdue start, arrival, finish or work record of a silent robot counts as abort, absent and failure
  evidence.
- Absent records are superseded by newer knowledge of the member.
- The lease names declared-failed planned members absent.
- Rally is stall-aware and stops when the robot hears the station.
- Idle robots are polled at comm ticks.

**Families.** Freeze them as run (`perturb.CELLS`):
- F0 {N1, N2, N3}; F1 {R1, R2, R3}; F2 {R2N3}; F3 {pf0.1, pf0.2, pf0.2-R2};
- secondary: S-N4, S-R2-durexp, S-R2-durunif;
- C3 grid: {F1-R2, F3-pf0.2} × ρ ∈ {2, 1.25, 1, 0.75, 0.5}.

**Endpoints (spec Section 9).**
- Convert Section 9 into a descriptive appendix plan: the same statistics (9.1), every family reported, and no
  confirmatory Track D family.
- If the user wants any confirmatory Track D statement, the only candidates the dev data support with a margin are:
  - SPARC vs insertion-only on F3 (0.828);
  - SPARC vs B1 online (0.914).

  Either one would have to be pre-specified before any validation or test run.

**Prereg.** The spec's "fewer messages" clause is already gone: `docs/prereg-phase1.md` C3 reads "No message-count
claim" (commit `4cf5e04`). Proposed edit, for the user to make and commit:
- Replace the C3 and C1-dyn / C2-dyn bullets with one sentence: the Track D dev gates K1 and K6 fired and E2a failed,
  so C1-dyn, C2-dyn and C3 are descriptive (an analysis appendix).
- Point that sentence to this memo.

## 6. Open problems

1. **Kernel pin.**
   - `cbba_sota/dyn` runs the kernel pinned at `4cf5e04`: the work-in-progress "v2" of `103da3a`, even though the
     module docstrings call it "ALNS v2". The static paper's ALNS v2 is `0e449fb`: portfolio construction, start-time
     re-sort, and kernels that are 1.4-2.3× faster.
   - Any Track D number printed next to Track S results must either disclose the older kernel or be re-run after
     re-pinning.
   - Re-pinning changes the Light-tier CPU conversion: faster iterations mean fewer equal-CPU B4 restarts. It could move
     T1. T1 held across 9-196 restarts, but B4-r12 (0.9723) and B4-impl (0.9740) are closest to the 0.97 bar.
   - A re-pinned T1 is a new, pre-specified test on the validation split, not a re-run of K1.
2. **Forking paths.** These arms were added after T1 failed. They are exploratory, and none is gate evidence:
   - SPARC-rk / B4-eq = 0.9717 on seeds 0-1 and 0.9665 on seeds 2-3 (where the rule passes);
   - SPARC-H / B4-eq = 0.9627, at 2.9× B4-eq's CPU.

   The validation split now exists, so a pre-specified confirmation is possible. It would be a new hypothesis and must
   be labelled as one.
3. **K8 is not measured.** All tier budgets (B2, B3, B4-eq) come from loaded-host calibrations.
4. **No per-method tuning.** Spec 5.1 allows up to 12 configurations per method. Tuning B4 or B6 could only narrow
   SPARC's margin.
5. **Heavy baselines are subsamples.**
   - B2: 171 episodes (25 clusters, seed 0). B3-Heavy: 72 episodes (24 clusters, one cell per family). B2 at 3×: 24
     episodes.
   - The rows do not record B2's fallback count.
   - Realized budgets vs the tier: B2 got 1.2-1.6×; B3-Heavy got 0.7-0.9× on three settings and 2.5× on SA-AT.
6. **B6 is a good-comms fixed point only.**
   - The CBTA-vs-CBGA fidelity check is weak: the CBGA reading has makespan 105.7.
   - Message-level CBTA, needed for CBTA-dec, is not built.
7. **C3 open issues.**
   - FULL is slower than the K0-verified controller on SA-AT-50 F3: 8-41% on 3 episodes, undiagnosed.
   - A robot that dies while idle can be given clamped solo work (a residual liveness hole).
   - T2 failures concentrate on SA-AT-50: 364 of 422.
   - Leases were tuned on 2 settings (10 clusters) and applied to 4.
   - ρ ∈ {1.25, 0.75} was run on 2 settings only.
   - Not run: RL-dec, CBTA-dec, the oracle arms, the T4 log, T5-T8, the corner station, and the Bernoulli and Rayleigh
     channels.
8. **Spec text to correct at freeze.**
   - D2's regression ("15 RL rollouts identical under arrival order") is false for the live RL policy on MA-AT. It holds
     only for plans made from RL rollouts.
   - "Zero rows for failed robots" must become deleted rows.
   - D7 under T2 needs the global signal, and rule 4.6(c) must be added.
   - The K6 variant set contains a fixed architecture (V1 = HYB).
   - `iters0` is unspecified.
   - B3 budgets need sub-solve counts, because deterministic time does not bound CPU.
   - Citations: degree of dynamism is Lund et al. 1996, and the effective degree is Larsen 2001.
   - S8 is reversed, and S2 needs its budget condition.
9. **K9 residual.** The full texts of DODRL-HMRTA and of the Demiray et al. journal version are unread. Confirm that
   HMRTA-TO is not HeteroMRTA-generated before any novelty sentence is printed.
10. **Provenance hygiene.**
    - The C1-dyn and baseline rows record git `1f0f0462bafb` with `git_dirty_dyn = true` and no planner hash. Agent B
      verified the sha1 of every imported file before and after.
    - The C3 rows' hashes (`5bc2c415b9`, `26d92033cd`) differ from the working tree's `40ea3eec4e` only in docstrings.
      Code snapshots are in `runs/trackD/c3/code_<hash>/`.
    - Uncommitted:
      - the `protocols.py` docstring line;
      - the test-count line in `baselines.md`;
      - untracked C3 result files: `c3.md`, `c3_design_adopt.txt`, `c3_gates_5rho.*`, `c3_rho_ext.txt`,
        `c3_tuned_leases_ext.json`;
      - this memo and `scripts/trackD_week1_verify.py`.
    - `runs/` is git-ignored.

## 7. Remaining week-2 work

The spec's week-2 plan (tune, validation dry run, freeze, one test run of a confirmatory family) no longer serves a
Track D method claim.

**User decisions first:**

| # | Decision | Proposal |
|---|---|---|
| U1 | Accept K1 and K6, and the appendix framing, or stop Track D here | accept |
| U2 | C3 recommendation | Report DS as the pre-declared recommendation, and INF-r as the dev-best rule, labelled post hoc |
| U3 | Whether any Track D number goes to the validation or test split, and which descriptive endpoints | the minimum needed for the appendix |
| U4 | Kernel: re-pin to `0e449fb`, or disclose `4cf5e04` | re-pin if the appendix prints SPARC numbers next to Track S results |
| U5 | B1 coalition rule and the B6 slot | arrival order primary; CBTA-style (start) |

**Then, only if U3 is yes:**
1. Re-pin (if U4), then:
   - `scripts/trackD_kernel_check.py` against the new commit;
   - `scripts/trackD_k0.py run` (executor parity depends on the planner);
   - T1 as a pre-specified validation-split test.
2. K8 on a quiet pinned host, after the other workflow finishes:
   - per-event CPU p50 / p95 for SPARC-L / H, B4, B5 and B6 at 50 and 200 tasks;
   - 150 × 500, descriptive;
   - milliseconds per iteration, to fix the tiers;
   - recalibrate B4-eq, the B3 sub-solve counts and the B2 decision budget.
3. Freeze Section 5 and the families. Rewrite Section 9 as a descriptive plan. The user edits `prereg-phase1.md` and
   commits with the solver and `dyn` commit hashes.
4. An untuned dry run on the validation split at the frozen parameters.
5. One test-split run on the quiet host: descriptive, every family reported.

**Optional** (appendix completeness; no gate depends on these):
- B6 message-level and a stronger CBGA reading;
- the C3 day-6 items: RL-dec, T4 log, T5 factorial, corner station, channels;
- diagnosing the SA-AT-50 F3 failures and the FULL gap.

## 8. Verification and test status (this memo)

- **Recomputation.** `.venv/bin/python scripts/trackD_week1_verify.py` (sections c1, c3, c3bs, k0, cpu, rlcost; one
  process, under a minute). From the raw rows it reproduced:
  - every C1-dyn and baseline ratio in Sections 1-4 (B4-eq, B4-r12, B5, B6 ×3, B3-Light / Heavy, B2, B2 ×3,
    insertion-only, open loop, every-event, SPARC-H, RL ×4, greedy, the K2 views, T3, and the seeds 2-3
    replication);
  - the C3 regret tables for 4 settings × 3 ρ and 2 settings × 5 ρ, DS for every ρ\*, the K5 ratios, and the failure
    counts (422 of 5,760 by setting);
  - the K0 counts, including the RL(s.64) sample multisets against the stored Phase-1 rows (80/80);
  - the loaded-host CPU percentiles.

  New in this memo, and in no agent report **(m)**:
  - the both-succeed C3 sensitivity (`c3bs`);
  - the RL cost of dynamics on the K0 env (`rlcost`);
  - the split audit;
  - the hash, lease and RL(s.64) checks.
- **Hashes.** Today `env.code_hash()` is `6fabb0570f` and `executor.code_hash()` is `56a085a816`. `git diff 1f0f046`
  of every runner-imported file (env, perturb, controller, executor, planner, `rh_kernels`, sparc, methods,
  `constructor_rh`, `ditags`, `trackD_run.py`) is empty. The leases in `eval_final.jsonl` equal the frozen
  `tuned.json` choices.
- **Kernel pin.** `scripts/trackD_kernel_check.py --n 1 --iters 1500` was re-run today on dev instance 0 of all five
  settings. The state-start kernel is identical to the `4cf5e04` kernel on every instance: instance arrays, schedules,
  construction, and a 1500-iteration `run_batch`.
- **Full test suite.** `OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 .venv/bin/python -m pytest tests/ -q -p
  no:cacheprovider` gave **333 passed in 626.8 s**, exit 0, with no other cbba-sota process running.
- **Lint.** `ruff check cbba_sota scripts tests`: all checks passed, including the new `scripts/trackD_week1_verify.py`.
