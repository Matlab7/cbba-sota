# Track D spec: dynamic and decentralized execution (DynHeteroMRTA-X, SPARC)

Status: draft for freeze, written 2026-09-28 from three designs (KORE, SCOPE-RALNS, HALO), their three red-team kills, and the dev-split pilots. Nothing here has touched the test split. Section 9 becomes the Track D pre-registration once the week-1 gates in Section 8 are decided. Until then only dev and validation seeds may be used.

Sources: `docs/research/trackD-scouts-2026-09-27.json`, `docs/research/trackD-proposals-2026-09-28.json`, `docs/research/trackD-redteam-2026-09-28.json`, and pilots under `pilots/trackD*` (Section 1 names the files).

## 0. Summary

**What Track D claims.** SPARC is the Phase-1 coalition ALNS run as a warm, event-sparse repair loop inside the real HeteroMRTA event loop. Under online release, execution noise and robot failures, it is the best online policy at equal per-event compute and under the same execution protocol. The comparison set is: the published RL policy run online, RL used as a rollout planner (RL-MPC), rolling CP-SAT-LNS, a rolling constructor with restarts, targeted (D-ITAGS-style) repair, and a CBBA-family coalition auction.

**What it does not claim.**
- A margin over a central rolling re-solve that uses the same engine and the same triggers. Under good communication that baseline *is* SPARC; the spec states this as an identity and does not test it.
- A new decentralized coordination mechanism.
- Fewer messages than central.

**The communication claim (C3) is conditional.** SPARC's protocol has the lowest worst-case regret across a prespecified connectivity grid. The comparison set is the fixed architectures that share its planner, plus a deployment-time switch. C3 is confirmatory only if the dev gates in Section 8 pass. Otherwise it is reported as benchmark and diagnosis: teammate-state staleness is the bottleneck, and leases and a low-rate beacon dominate the choice of architecture.

**What the static result contributes and what is new.** Most of any margin over the published RL is the known static quality gap, and the paper says so. The dynamic-specific evidence is two findings:
- Search pays on structural events. Insertion-only repair is 7-13% worse.
- Noise alone does not justify re-planning.

## 1. Evidence this spec rests on (dev split, pilots, loaded shared host)

All numbers are geometric-mean paired ratios with bootstrap 95% CIs. Below 1 favours the first-named method. "Surrogate" means a hand-written executor or comm world, not yet regression-tested against DynTaskEnv.

| # | Finding | Numbers | Source |
|---|---|---|---|
| S1 | Static gap (Phase 1) | ALNS-8/RL-8 = 0.734-0.789; ALNS-8/CP-SAT-LNS-8 = 0.890-0.982; ALNS-8/(RL64+ALNS polish) = 0.983-0.999, a tie | `runs/compare_dev/report.txt` |
| S2 | Noise alone does not justify re-planning | open-loop / RH-ALNS(3000 it) = 0.993 [0.978, 1.009], 0.998 [0.981, 1.014], 1.022 [1.001, 1.041] (N2, N3, N4) | `pilots/trackD_robust/summarize_c2.py` |
| S3 | Cheap re-planning on noise is nervous | RH@30 it / RH@3000 it = 1.034, 1.049, 1.019 | same |
| S4 | Perfect duration foresight is worth a lot, but no causal policy can get it | RH / RH-oracle = 1.101, 1.023, 1.016 | same |
| S5 | Anticipation is small; q80 buffers hurt | nominal / SAA plan = 0.999-1.022; nominal / q80 = 0.950-1.017 | `pilots/trackD_robust/out/lever.log` |
| S6 | Structural repair needs search | insertion-only / ALNS repair = 1.070, 1.122, 1.113, 1.132 (R1, R2, R3, R2+N3); SA-BT only 1.009-1.041 | `summarize_rel.py` (surrogate) |
| S7 | Budget saturates early | ALNS 300 it / 3000 it = 1.018, 1.018, 1.007, 0.998; per re-plan 13 ms vs 122 ms | same |
| S8 | Structural-only triggers beat every-event re-planning | every-event RH / structural-only (3000 it) = 1.019, 1.011, 1.012, 0.993; 57 vs 23 re-plans per episode | same |
| S9 | Published RL online is far behind | RL(g.) / structural ALNS@30 = 1.151-1.167 under release; RL / RH-ALNS = 1.205-1.230 under noise; RL is better in 0-7 of 80 pairs (all 7 in SA-BT R1) | same, and `summarize_c2.py` |
| S10 | Rolling ALNS vs rolling constructor (noise only) | 0.906 [0.890, 0.923], 0.904 [0.888, 0.919]; rolling ALNS vs open-loop ALNS = 1.016, 1.003 | `pilots/trackD_comm2/out/planner_collapse_summary.txt` (surrogate) |
| S11 | Consistency machinery does not help under partitions | equal patient lease (3/30), vs CEN-F: LOC-F 0.997-1.055; CAS-INF 0.968-1.124; OPEN 1.26-1.51; staleness gate SG null; dead reckoning harmful (1.20-1.28 vs REP) | `pilots/trackD_integ/out/summary.txt` P2, P4, P7 (surrogate, list-scheduling planner) |
| S12 | Stranded groups re-planning is the lever at 0.5-0.63 r_c | vs CEN-F at R=0.1: CLAMP 0.839 / 0.824; HYB 0.838 / 0.817; HALO 0.865 / 0.813; INF-r / HALO = 0.972-0.991 (a tie) | P7, P8; red-team rerun `/tmp/halo_rt/analyze2.py` |
| S13 | Near and above r_c, central with fallback ties or wins | HYB / CEN-F at R=0.3 = 1.032 / 1.075; INF-r / CEN-F at R=0.3 = 1.13-1.14 | same |
| S14 | Worst-case regret across 8 cells (vs per-cell best fixed architecture) | HALO 1.034, CLAMP 1.065, HYB 1.075, INF-r 1.157-1.188, CEN-F 1.26; the gate θ was picked post hoc | `/tmp/halo_rt/analyze2.py` |
| S15 | The bottleneck is teammate staleness, not news latency | teammate-state oracle / REP = 0.904, 0.845, 0.719; release oracle / REP = 0.988-1.032; task-status oracle is worse | P3 |
| S16 | Infrastructure swamps architecture | global beacon (period 1, loss 0.2) / REP = 0.924, 0.871, 0.771; patient lease / REP = 0.83-0.95; FULL(3/30) / FULL(1/10) = 0.929; the ranking flips with the lease | P5, P6; `/tmp/redteam_trackD/bestbest.py` |
| S17 | Makespan-only scoring hides the cost of redundancy | wasted trips per episode at R=0.1: HALO 42, CLAMP 43, INF-r 51, CEN-F 3.4 | red team 3 |
| S18 | Connectivity geometry | at R ≈ r_c the station reaches 28-57% of robots; the median latency to current coalition partners is 0 even at 0.5 r_c | `pilots/trackD/out/conn.txt` |
| S19 | The env prototype is faithful | DynTaskEnv with all releases at 0 and no noise is bit-identical to the static env (RL, greedy and ALNS replay); a zero row equals a deleted row (max \|dp\| 7.8e-7) | `pilots/trackD/check_equivalence.py` |

Findings with no pilot yet: robot failures (F3); anything with the real ALNS inside a comm world; the T2 channel; true 0.5 r_c for 50 robots; CPU timings on a quiet core.

## 2. Claims

- **C1-dyn (primary).** With good communication, on families F1-F3 (Section 3.4), SPARC at the Light budget (1 core, 300 ALNS iterations per structural event) gives lower makespan than each online competitor in Section 5.2 at the same budget tier or a heavier one.
- **C2-dyn.** The Light budget saturates: it is within 2% of SPARC at 3000 iterations, and within 2% of (or better than) rolling CP-SAT-LNS at the Heavy tier (8 cores, 1 s per event). This is a per-robot CPU and energy claim. It is not a reaction-time claim, except under the LoRR time-mapping sensitivity (Section 3.6).
- **C3 (conditional).** On the prespecified connectivity grid, SPARC's worst-case regret is lower than that of every fixed architecture sharing its planner, and than a deployment-time switch. It is confirmatory only if gates K5 and K6 pass on dev.
- **Explicitly not claimed.**
  - Parity or superiority vs central RH-ALNS with structural triggers (an identity; checked as an implementation test).
  - Novelty of any single component (Section 4.9).
  - Fewer messages. The draft clause "uses fewer messages" in `docs/prereg-phase1.md` C3 must be deleted at freeze.
  - Optimality.

## 3. Benchmark: DynHeteroMRTA-X

### 3.1 Held identical to HeteroMRTA (marmotlab @ db51e29, Apache-2.0)

- Instance generator, species traits, additive requirements, depots, speed 0.2 in the unit square, env dt 0.1, MAX_TIME 200.
- Coalition semantics: a task starts when present members cover the requirement, and members are held until it finishes. Plans use minimal covers.
- The released RL checkpoint, unmodified. Unknown tasks and robots are all-zero rows, which the network treats as padding.
- Settings:
  - Primary: MA-AT-25-5-50, MA-AT-50-5-50, SA-AT-50-5-50, SA-BT-50-5-50.
  - Secondary (scale, C1-dyn and C2 only): MA-AT-50-5-200.
  - 150x500: compute only, descriptive.
- Splits: test = 100000 + 1000·idx + i (i < 50); dev = 500000 + ... (i < 20); validation = 900000 + ... (i < 20). Same bases as `docs/prereg-phase1.md`.
- **Ground truth is the env.** Implementation: `DynTaskEnvX`, a subclass of `TaskEnv`, with no edits to `third_party`. Any faster executor used for search must match env replay (gate K0).

### 3.2 Declared semantic changes (each has a regression test)

| # | Change | Why | Regression |
|---|---|---|---|
| D1 | Release masking: unreleased task rows are zeroed and masked | online arrivals | all releases at 0 → bit-identical to the static env |
| D2 | Arrival-order coalitions: a task starts when present members cover it; late or redundant arrivals leave (a wasted trip) | stale beliefs under C3 | ideal comms + consistent minimal-cover plans → identical to the env's decision order (plan replay and 15 RL rollouts) |
| D3 | Causal observations: ETA = departure + nominal travel + stall so far; nominal durations until observed | removes the oracle leak | pilot RL oracle/causal = 0.98-1.02 |
| D4 | Departure back-dating capped at the last release epoch | causality | no-op when static |
| D5 | Causal lease replaces the native `max_waiting_time`, whose covered branch uses the realized arrival spread | causality | lease = ∞ and no failures → identical |
| D6 | `fail_agent(i, t)`: robot halts; its in-progress task is aborted and restarts from scratch when re-covered | failures (F3) | no failures → identical |
| D7 | Termination: the release window end H is known, the total task count is not revealed. A robot returns to its depot when t ≥ H and its replica shows every known task finished. Makespan is the physical time at which all tasks are finished and every live robot is home. With H = 0 the native end-of-episode behaviour runs unchanged. | removes the task-count leak and the hindsight-makespan artefact | H = 0 → bit-identical |
| D8 | Idle robots wait in place while t < H | online release | n/a |

### 3.3 Perturbations and anchors

CRN keys are (setting, instance, CRN seed, stream, entity). Each stream is drawn independently of method decisions: delay calendars are in absolute time, and per-link, per-send-tick uniforms drive the channel.

| Stressor | Model and parameters | Anchor | Status |
|---|---|---|---|
| Travel delay N1 | LoRR 2026 Start-Kit `DelayGenerator`. Each tick, each moving robot starts a stall with probability pDelay; stall length U{min..max} ticks. N1 = {0.01, 1-4} | Start-Kit example config, MIT, verified in source | external |
| Travel delay N3 | same generator, {0.05, 1-10} | same model; the parameter choice is ours | external model, convention parameters |
| Durations | mean-preserving lognormal, σ = 0.3. Sensitivity: exponential and U(0, 2d) | stochastic-RCPSP convention | convention |
| Release R1 | ids ≤ 20 at t = 0, +20 every 10 time units | *inspired by* the authors' GIF animation (`task_env.py:686`, `reactive_planning=False` by default, visual only). Not an env feature | weak |
| Release R2 / R3 | degree of dynamism 0.5 / 0.8; arrival times are uniform order statistics on [0, H], with H = 0.5 × the paper's published RL(g.) makespan for the setting; the station at (0.5, 0.5) is the dispatch point | DVRP degree-of-dynamism convention (verify the citation) | convention |
| Failures | each robot fails with p_f ∈ {0.1, 0.2}, onset U[0, 0.7 × published RL(g.) makespan], fail-stop, detected by heartbeat timeout h = 5 ticks. Realizations that make any task infeasible are excluded for all methods, paired | event type after D-ITAGS (RA-L 2023); rates after Gosrich et al. (T-RO 2025) | convention |
| Range | unit disk, R = ρ · r_c(n), r_c(n) = sqrt(ln n / (π n)), n = robots + 1 | random-geometric-graph connectivity threshold (Penrose; Gupta-Kumar) | external theory |
| Link loss | per message: Bernoulli p = 0.2; Gilbert-Elliott with ρ = 0.8, pGG = 1-(1-ρ)p, pBB = ρ+(1-ρ)p; Rayleigh (L0 40 dB, η 3, Ptx 30 dBm, h ~ Exp(1)) | CV_MRTA protocol (arXiv 2609.13711); code has no licence, so the protocol is reimplemented | external protocol |
| Beacon (factor) | LoRa-class global channel, period 1 or 5, loss 0.2, per-node byte budget; symmetric payload (any method may send anything, including central route pushes) | convention | infrastructure factor |
| Station location | (0.5, 0.5); corner-station sensitivity (C3 only) | convention | declared |

Do not present loss models as stressors that separate methods. CC-OPI and our pilots both show that loss barely matters once range limits dominate. They are reported to show robustness, and range is the stressor.

### 3.4 Scenario families

Every dynamic cell includes the base noise N12 = N1 + σ 0.3.

| Family | Cells | Role |
|---|---|---|
| F0 noise control | N1, N2 (σ 0.3), N3 (N3 + σ 0.3), all tasks at t = 0 | control; parity with open loop is expected and reported |
| F1 release | R1, R2, R3 (each with N12) | C1-dyn primary |
| F2 release + moderate noise | R2 with N3 + σ 0.3 | C1-dyn primary |
| F3 failures | p_f 0.1 and p_f 0.2 (all tasks at t = 0, N12); p_f 0.2 with R2 | C1-dyn primary |
| C3 grid | {F1-R2, F3 p_f 0.2} × ρ ∈ {2, 1.25, 1, 0.75, 0.5}, GE loss, tier T2 | C3 |
| Secondary | N4 severe; Bernoulli and Rayleigh channels; beacon factor; corner station; MA-AT-50-5-200 | descriptive |

Families are frozen by this document. Every family is reported, F0 included.

### 3.5 Communication tier T2 (C3 only; good comms = instant and lossless)

- **Links.** A link is up when the analytic distance between the two robots' straight-line legs is at most R, and the message survives its loss draw.
- **Latency.** One tick per hop. Multi-hop relaying needs one tick per hop, with no instant flooding.
- **Gossip.** Per-node byte budget; delta gossip by version digests.
- **Oracle.** Cross-check against `/home/jovyan/dev/cbja/cbja/network.py` as a small-case oracle only (5 cases).

### 3.6 Time and compute accounting

- **Primary: compute outside the loop.** Decision latency is zero, and budgets are deterministic: ALNS iterations, CP-SAT `max_deterministic_time`, constructor restart counts. Budgets are converted once to CPU milliseconds on a quiet, pinned core at freeze time. This makes runs deterministic and preserves CRN. Wall-clock deadlines on the shared host would break both.
- **Sensitivity T.** The LoRR two-rate loop (1 tick = dt 0.1 = 100 ms, so 1 time unit = 1 s; plan limit 1000 ms). Each planner or policy call is charged its measured CPU time *unrounded*, and robots execute the last staged plan meanwhile. A second mapping (1 time unit = 100 s) is reported, under which compute is effectively free.
- **Budget tiers**, identical for every method that can use them:
  - Light: 1 core, the CPU-ms equivalent of 300 SPARC iterations.
  - Heavy: 8 cores or 1 GPU, 1 s per event.

### 3.7 Metrics

- Makespan (physical, D7) and success. A failure is scored 200 only in sensitivity analyses (Section 9.1).
- P90 makespan.
- Total travel distance, wasted trips and abandons. These are co-reported with makespan as a Pareto view in C3.
- CPU per event (p50/p95), CPU per episode, re-plans per episode, route versions (churn).
- Messages and bytes per robot per time unit (C3).
- Station-component fraction (C3).
- Gap to references: FULL (the same method with ideal comms) and hindsight ALNS (releases, realized durations and delay calendars known; start ≥ release). Hindsight ALNS is a reference, not a bound, and is not defined for failures.

## 4. Algorithm: SPARC (Structural-event Planning with Anchored Replication of Coalitions)

SPARC is kept from the three killed designs only where their pilots supported it. It runs per robot and at the station on CPU, with no GPU.

### 4.1 Planner (the only claimed algorithmic content)

- **Kernel.** The Phase-1 ALNS (v2, frozen commit) with the state-start kernel: robot i is free at `ready_i` at `pos_i`, its first leg is `ready_i + travel(pos_i, t)`, and task start ≥ release. Minimal covers and a global key order are kept. Source: `pilots/trackD/rh_kernels.py` and `pilots/trackD_robust/robust_rh_kernels.py`, ported into `cbba_sota/dyn/`.
- **Anchors.** A task is *committed* once any member has departed to it. Its members and key are then frozen. A committed task still missing traits (after an abandon or failure) is re-planned with its residual requirement.
- **Repair region.** All released, uncommitted tasks plus orphans.
- **Floor.** Regret insertion of new and orphaned tasks first, which gives a feasible plan in about 2 ms (G4).
- **Search.** Warm-started ALNS from the incumbent restricted to open tasks, 300 iterations (Light), with a lexicographic objective (tasks placed, then makespan).
- **Keys.** Every repaired task gets a key greater than every committed key (G1).
- **Seed.** Deterministic: `hash(replica digest, event index)`. Identical replicas therefore compute identical plans (G3).
- **Predictors.** ETA = nominal travel × κ, with κ = 1/(1 − expected stall fraction) of the declared delay model; residual duration = nominal. All baselines get the same predictors.

### 4.2 Trigger rule (a declared design rule, S2, S3, S8)

- **Re-plan only on structural events:**
  - a task release becomes known;
  - an orphan (abandon or failure detection);
  - a change of component membership that changes who the planner can reach (C3);
  - a live robot idle with an empty route while open tasks exist.
- **Never re-plan on noise events** (finishes, stalls, early or late arrivals). There is no periodic trigger.
- **Consequence.** Under good communication, SPARC equals central RH-ALNS with structural triggers. The decision-difference log must show 0% (Section 7, T4).

### 4.3 Execution

- Follow the adopted route in key order. Depart when the previous task finishes. Coalitions start in arrival order.
- **Good comms.** The lease equals the failure detector: a robot waits while its partners heartbeat, and abandons only when a partner is suspected failed (silent for h ticks). This configuration is what G5 needs.
- **C3.** A patient causal lease with parameters g and L:
  - A waiting robot abandons if it has waited ≥ g and knows of no partner still coming, or if it has waited ≥ L.
  - (g, L) is chosen per method and per ρ on dev from the shared grid g ∈ {0.3, 1, 3} × L ∈ {10, 30, 60}.
  - An abandon publishes an abandon record and the robot continues with its route.
- **Rally.** An idle robot that is outside the station component and has an empty route moves toward the station. Shared by all methods.

### 4.4 Replicated knowledge (the CBBA-lineage consensus layer)

This is a delta-state CRDT, merged on contact:
- released tasks: grow-only set with the task data;
- start and finish: earliest time wins;
- abandons: grow-only set of (robot, task, epoch);
- per-robot state record (mode, target, ETA or finish, position, adopted route version): last-writer-wins on the robot's own counter; heartbeat every 1.0 time unit;
- per-robot route: last-writer-wins on (Lamport epoch, author id). This is ACBBA's timestamp rule.

A robot that adopts a new route drops its old suffix, which plays the role of CBBA's bundle release. Gossip carries deltas only. Under good communication this layer is the identity.

### 4.5 Planning scope under partitions (anchored replication)

- **Leader** per connected component: the station if present, otherwise the lowest live robot id.
- **Station component.**
  - The leader plans only its members.
  - Robots outside it keep their believed routes, and their traits count toward those coalitions.
  - This is exactly central re-solve with frozen fallback (CEN-F), the arm that ties or wins at ρ ≥ 1 (S13).
- **Any other component.**
  - The leader re-plans *all* robots on its replica.
  - A robot b outside the component is modelled by a clamped snapshot: free at max(recorded ETA or finish, now), located at its recorded target, with no dead reckoning (S11).
  - The leader writes route versions for all robots. Versions for absent robots travel by store-carry-forward and are adopted on contact. Writing only local routes was worse in every pilot cell (HYB2).
  - Re-planning happens only under the Section 4.2 trigger rule.
- **Variant selection (dev only, frozen before test).** Three stranded-component variants run on dev:
  - V1: re-plan on every knowledge change plus a 1.0 period (= HYB);
  - V2: structural-only (the default);
  - V3: HALO's θ-gate.
  The variant with the lowest dev worst-case regret is chosen. If variants are within 1 point, prefer V2, then V1, then V3; the parameter-free rule wins ties. All three are reported on test.

### 4.6 Failure suspicion under partitions (fix for a red-team finding)

A heartbeat timeout cannot tell a crashed robot from one that is out of range. SPARC therefore treats a silent robot as *unreachable* (clamped), not as failed. Two rules:

- **(a) On-site evidence.** A robot waiting on site whose lease expires with a silent member publishes `absent(member, task, t)`. Every planner then removes that member from the task's cover and re-plans the residual requirement. If the "absent" robot later arrives, the redundant arrival leaves (a wasted trip, D2). This trades redundancy for liveness, and the paper states that.
- **(b) Declaring failure.** A robot is declared failed only after silence ≥ h_f = 10 time units and at least one `absent` record. This affects its future route, not its already-committed head task.

### 4.7 Parameters (frozen at the end of week 1; dev only)

| Parameter | Default | Tuned? |
|---|---|---|
| ALNS iterations per event, Light / Heavy | 300 / 3000 | no (C2 is the anytime sweep {0, 30, 300, 3000}) |
| Heartbeat period; failure detector h (good comms) | 1.0; 5 ticks | no |
| Lease (g, L), C3 | per method, per ρ, shared grid | yes, equal budget for every method |
| Stranded variant V1 / V2 / V3 (θ) | V2 | selected on dev (4.5) |
| h_f (C3 failure declaration) | 10 time units | no |

### 4.8 Analytical collapse view

- **Noise.** For a fixed key-ordered plan, realized makespan is the longest path in a fixed DAG with random weights (a max-plus system). LoRR stalls are memoryless per tick and durations are independent across tasks. Noise events therefore reveal only second-order information, so re-planning on noise can at best recover second-order effects (S2), and cheap re-planning adds nervousness (S3). *Declared rule: no re-planning on noise.*
- **Structural events** (release, failure) change the problem by a first-order amount. A local rule (insertion) repairs them badly (S6). *Claimed: search in the loop.* The collapse test compares against the strongest cheap rule, a constructor with restarts at the same budget, not against insertion-only (T1).
- **Partitions.** The gains come from not honouring stale commitments of unreachable robots, and from letting stranded groups keep planning (S12). Both are one-line rules. *Declared, not claimed as mechanisms.* The only C3 content that is not a simple rule is whether a single fixed protocol has lower worst-case regret than every fixed alternative and than a deployment switch (E3a). The prior for that is weak (S14: a 3-point margin over CLAMP, with θ chosen post hoc).

### 4.9 What was dropped, and why

| Component | From | Status | Evidence |
|---|---|---|---|
| Frozen closure-local repair (rule 6), LOC-F, FRZk | KORE, SCOPE | dropped | lost or tied to CEN-F (S11); freezing more is monotonically worse |
| Delivery-aware ready time (DAR, DTH-θ) | SCOPE | dropped | collapses to REP+ / INF-r; DTH1 / INF-r ≈ 1 |
| Optimistic dead reckoning | HALO | dropped | predicted the current task 28% vs 52%; 20-28% worse |
| Coalition locks (CAS), staleness gate (SG) | pilots | dropped | null or harmful (S11) |
| Stability gate, SAA, q80 buffers, spares, phantom tasks | KORE | ablation only | S5; gate no help |
| θ regime gate | HALO | dev variant V3 only | null vs HYB in 7/8 cells |
| "Fewer messages" | prereg draft | dropped | HALO-like arms send 2-170x central's route versions |
| Wall-clock 25 ms deadlines | KORE | replaced by iteration budgets | non-deterministic on a shared host; breaks CRN |
| Lease cap as fixed timeout under good comms | KORE, SCOPE | replaced by the failure detector | a fixed 10-unit cap cascaded (49.6 → 69.0) |
| Kept | all | planner (4.1), structural triggers (4.2), commit-at-departure and key monotonicity, CRDT layer (4.4), clamp, station anchoring, stranded re-plan, patient leases, rally | S6-S8, S11-S13 |

### 4.10 CBBA lineage, stated plainly

- CBBA contributes the consensus layer: timestamped max-consensus on per-robot routes, and suffix release on adoption. CBBA's bundle-building phase is replaced by coalition ALNS.
- Under good communication the consensus layer is the identity. Under partitions, the pilots found every added consistency step null or harmful.
- The paper therefore places SPARC in the replicated-centralized family (HIPC, DGA, DMCHBA). The CBBA family appears as baselines (CBTA-style, coupled-constraint CBBA) and as ablation T8. The user's preference for CBBA lineage is honoured in structure, not in a margin claim.

## 5. Baselines with equal infrastructure

### 5.1 Shared by every method

- The same `DynTaskEnvX` and CRN realization.
- The same event notifications: release known at the station; failure at heartbeat timeout.
- The same causal observations, predictors (κ-scaled ETA, nominal residuals), and plan-following executor (key order, arrival order).
- The same lease grid, tuned per method on dev with an equal budget.
- The same rally and termination rule (D7).
- The same CRDT gossip layer and beacon factor under C3, with symmetric payloads.
- The same budget tiers (Section 3.6).
- Each method's own hyperparameters are tuned on dev with the same number of configurations (≤ 12).

### 5.2 Competitors for C1-dyn (good comms, F1-F3)

| Id | Method | Tier | Notes |
|---|---|---|---|
| B1 | Published RL(g.), online | native, about 6 ms per decision | checkpoint unmodified; zero rows for unknown tasks and failed robots; receives failure notices. Also RL(s.1) |
| B2 | RL-MPC | Heavy | at each structural event, clone the state into a nominal model (released tasks only, nominal durations, live robots), run RL(s.N) with N filling 1 s on 8 cores or a GPU (lockstep), convert the best rollout to key-ordered routes, and execute under 4.3. This is the learned counterpart of SPARC's plan-then-execute. RL(s.N) over realized episodes is not a legal online policy and is not used |
| B3 | RH-CP-SAT-LNS | Light and Heavy | the Phase-1 CP-SAT-LNS with state start (ready times, start positions, releases, fixed anchors), warm-started from the incumbent, same triggers |
| B4 | RH-constructor with restarts | Light | regret insertion rebuilt from the current state, with randomized restarts filling the budget, same triggers. **The simple-rule re-planner and collapse control T1** |
| B5 | D-ITAGS-style targeted repair | Light | insertion plus ALNS restricted to coalitions touched by the event. In-house reimplementation (no official code) |
| B6 | CBTA-style coalition timetable auction, re-run to consensus on structural events | native | makespan variant plus the native average-start objective; bids from our insertion kernel. Disclosed reimplementation; fidelity check: it must beat a CBGA-style auction on average start time, as CBTA's paper reports. If it is not ready at freeze, it moves to secondary and a coalition sequential auction with our bids takes its confirmatory slot |
| B7 | Paper greedy (dist bug fixed), online | native | descriptive |

References, not competitors:
- R-a: central RH-ALNS with structural triggers. Identical to SPARC under good comms; implementation check.
- R-b: every-event RH-ALNS at the same budget. Design-rule comparison, expected 0.98-0.99.
- R-c: open loop (t = 0 plan plus insertion of releases).
- R-d: hindsight ALNS.

Ablations:
- SPARC with an RL(s.64) warm start;
- SPARC with q80 durations, and with SAA over 8 scenarios;
- insertion-only repair;
- anytime sweep {0, 30, 300, 3000}.

### 5.3 Protocol arms for C3 (all use SPARC's planner, tier T2)

| Arm | Definition |
|---|---|
| FULL | SPARC with ideal comms (reference) |
| CEN-F | station plans its component; outsiders follow their last route, then rally |
| HYB | CEN-F plus stranded components re-plan everyone on every knowledge change and every 1.0 (V1) |
| REP-clamp | every component leader re-plans all robots with clamped snapshots; newest version wins |
| INF-r | leaders plan only their component; outsiders keep only their physically committed head task |
| DS (deployment switch) | CEN-F if ρ ≥ ρ\*, else REP-clamp; ρ\* chosen on dev. Realistic because range and site size are known at deployment and connectivity is stationary within an episode |
| SPARC | anchored replication with V2 (or the dev-selected variant) |
| RL-dec | B1 on each robot's own belief (obs_from_belief), plus RL with oracle global state as a reference |
| CBTA-dec | B6 over the same gossip, if ready |
| Oracles | REP with a teammate-state oracle, and REP with a knowledge oracle (diagnosis only) |

A beacon factor {none, period 1, period 5} is applied to all arms (secondary).

## 6. Guarantees (provable) and non-guarantees

Definitions:
- A plan is a set of routes sorted by a global key and minimal covers.
- *Consistent* means every robot follows the same plan.
- Travel times, stalls and durations are finite.

- **G1 (deadlock freedom).** Every task in a consistent key-ordered minimal-cover plan starts and finishes in finite time.
  - Proof: take the unfinished task j\* with the minimum key. Each member's preceding route tasks have smaller keys, so they finish (by minimality). Each member therefore departs and arrives in finite time. The cover forms, and j\* starts and finishes. Contradiction.
  - Repair preserves the precondition: committed tasks keep their keys and members, and every repaired task gets a larger key. So each route is a sorted committed prefix followed by a sorted repaired suffix.
  - Prior art: RCPSP serial SGS; ADG (Hönig et al., RA-L 2019). Not novel.
- **G2 (bounded waiting, C3).** No robot waits at a task longer than L plus one heartbeat period. True by construction of the lease.
- **G3 (replica convergence).**
  - The replica is a product of join-semilattices (union, min, and last-writer-wins with a total tie-break). Merge is therefore commutative, associative and idempotent.
  - Two replicas that have received the same set of deltas are identical (strong eventual consistency; Shapiro et al. 2011).
  - With the digest-seeded planner, they also compute identical plans.
  - Not claimed: that plans computed by different leaders on different replicas are compatible.
- **G4 (anytime feasibility).**
  - Suppose every released task's residual requirement can be covered by live robots known to the planner. Then the insertion floor returns a feasible key-ordered plan: append the task to the ends of a covering set of routes, give it a key larger than all current keys, and prune to a minimal cover.
  - ALNS accepts only feasible candidates, so every iteration count ≥ 0 returns a feasible plan.
- **G5 (completion under good communication).**
  - Assumptions: instant, reliable messaging; finitely many releases in [0, H]; finitely many fail-stop failures, each detected within h; the lease abandons only on failure suspicion.
  - Claim: every released task coverable by the surviving robots finishes in finite time.
  - Proof: there are finitely many structural events. After the last one, the plan is consistent and feasible (G4) and contains every coverable task, so G1 applies.
- **Not guaranteed under C3:**
  - completion (abandons can recur, and the rally rule does not imply recurrent contact);
  - coalition atomicity (Two Generals);
  - absence of duplicate service (tolerated by design);
  - any optimality or approximation bound. Hindsight and FULL gaps are reported instead.

## 7. Prespecified collapse checks (all reported whatever the outcome)

| Test | Comparison | Pass rule | If it fails |
|---|---|---|---|
| T1 planner vs rule | SPARC vs B4 (constructor with restarts, same Light budget), F1-F3 | pooled ≤ 0.97 with the cluster CI upper bound < 1, and < 1 on ≥ 3/4 settings | no dynamic-method claim (K1) |
| T1b | SPARC vs insertion-only | descriptive | — |
| T2 trigger rule | SPARC vs every-event RH (R-b) | descriptive design rule | — |
| T3 noise control | F0: SPARC vs open loop | TOST ±2%; decisions identical on 100% of noise events (implementation check) | bug |
| T4 decision diff | good comms: SPARC vs R-a. C3: share of structural events where SPARC's authored versions differ from DS / HYB / REP-clamp | good comms 0%. C3: if SPARC differs from DS on < 5% of events at every ρ, any C3 gain is DS's | C3 claim dropped |
| T5 factorial | at ρ = 1 and 0.5: {clamp, station anchoring, stranded re-plan, V1/V2} = 2^4, plus gossip and rally knockouts | report main effects; if one factor explains ≥ 80% of SPARC vs CEN-F, the paper states "collapses to rule X" | stated |
| T6 planner under partitions | SPARC vs SPARC-with-B4-planner at ρ = 0.5 | descriptive (checks whether the planner margin survives small components) | — |
| T7 consistency machinery | CAS locks, LOC-F inside SPARC | if either improves SPARC by > 2% on dev, drop "redundancy beats consistency" | stated |
| T8 CBBA layer | per-robot last-writer-wins vs per-task winner table with bundle release | ideal comms identical; at ρ ≤ 1, inconsistent coalitions and wasted trips must drop (CI excludes 0), otherwise the lineage is described as nominal | stated |

## 8. Plan: week 1 pilots with kill criteria, week 2 confirmation

Four agents: A (env), B (planner and optimisation baselines), C (comm and protocols), D (RL baselines, auctions, statistics, novelty).

Code goes in a new package `cbba_sota/dyn/`:
- `env.py`, `perturb.py`, `rh_kernels.py`, `sparc.py`, `replica.py`, `comm.py`, `protocols.py`;
- `baselines/{rl_online, rl_mpc, cpsat_rh, constructor_rh, ditags, cbta}.py`;
- `scripts/trackD_run.py` (resumable JSONL; reuse the pattern of `/home/jovyan/dev/cbja/scripts/run_v5.py`) and `scripts/trackD_analyze.py`;
- tests in `tests/test_dyn_*.py`.

Pin the ALNS v2 commit before porting, because another agent is editing `cbba_sota/solvers`. Pilots use dev only: 4 settings × 20 instances × 2 CRN seeds unless stated.

| Day | Work | Gate |
|---|---|---|
| 1 | A: `DynTaskEnvX` with D1-D8 and the regression tests. B: port the state-start kernel onto the pinned v2, with anchors, key monotonicity, insertion floor and deterministic budgets; unit test that a noise-free static run is within 1% of open-loop ALNS. C: T2 CommLayer, CRN channel calendars, CRDT, heartbeats, leader election; cross-check against cbja `network.py` on 5 cases. D: statistics module (cluster bootstrap, Holm, TOST, exact paired success test); RL online on `DynTaskEnvX`; read DODRL-HMRTA (Neurocomputing 2026, doi 10.1016/j.neucom.2026.135060, abstract not accessible on 2026-09-28), the CSCWD 2026 fault-recovery paper, the RA-L 2026 LLM-MILP paper, Demiray et al. (arXiv 2309.09321; confirmed centralized event-triggered ALNS with synchronization) and Calvo & Capitan (T-RO 2025) | — |
| 2 | A+B: executor vs env replay on 20 episodes per family, to 1e-6 for minimal covers; arrival order equals decision order at ideal comms; static reduction bit-identical | **K0** |
| 2-3 | B: C1-dyn pilot at good comms on F1, F2 and F3 (F3 for the first time): SPARC-Light/Heavy, insertion-only, every-event RH, B4, B1, B7 | **K1, K2** |
| 3-4 | B: B3 (state-start CP-SAT-LNS) and B5. D: B2 RL-MPC (state clone, GPU lockstep); start B6 CBTA-style (continues into week 2) | **K3, K4** |
| 4-5 | C: C3 map with the real planner and T2: arms of Section 5.3 × ρ ∈ {2, 1, 0.5} × {F1-R2, F3}, on MA-AT-25 and SA-BT-50 first, then all 4. Lease grid per arm. Variants V1/V2/V3 | **K5, K6, K7** |
| 5 | B: quiet pinned-core CPU per event (p50/p95) at 50 and 200 tasks; 150x500 descriptive; ms-per-iteration calibration for the budget tiers | **K8** |
| 6 | C+D: RL-dec, beacon factor, corner station, ρ ∈ {1.25, 0.75}; decision-difference logs (T4) | — |
| 7 | All: decision memo against K0-K9; freeze parameters (4.7), families and endpoints; convert Section 9 into the frozen prereg; delete the "fewer messages" clause from `prereg-phase1.md` C3; the user commits | **K9** |
| Week 2 | D: finish B6 and its CBTA-vs-CBGA fidelity check. All baselines tuned on dev with equal configuration budgets. Untuned dry run on the validation split. Freeze the commit. Run the test split once on a quiet pinned host. Analyse per Section 9 | — |

Kill criteria (dev):

- **K0 fidelity.** Executor-vs-env equivalence or a D2/D7 regression fails and cannot be fixed within 2 days. Stop; every pilot number is provisional until fixed.
- **K1 planner collapse.** T1 fails. Drop the dynamic-method claim; Track D becomes an analysis appendix to the static paper (S2, S6, S8 as findings).
- **K2 no dynamic separation.** SPARC / insertion-only > 0.98 on F3 *and* SPARC / B4 > 0.97 on F1. Same outcome as K1.
- **K3 RL anomaly.** B1 or B2 comes within 5% of SPARC pooled. Stop and investigate (a bug, or a genuinely strong RL) before any further build.
- **K4 heavy beats light.** B3-Heavy beats SPARC-Light by > 2% with the CI excluding 0. Not a kill: the C2 "light" claim is dropped and C1-dyn is stated per tier.
- **K5 C3 headroom.** At ρ = 0.5, the best fixed architecture is within 5% of FULL. No C3 method claim; keep the benchmark.
- **K6 C3 separation.** Either SPARC's dev worst-case regret is not at least 2 points below the best fixed architecture's, or DS ≤ SPARC. C3 becomes descriptive, DS is recommended, and the diagnosis (S15-S17) is the C3 section.
- **K7 lease fragility.** Any C3 conclusion flips within the lease grid. Report the grid and claim only what holds across it.
- **K8 compute.** Light p95 > 100 ms at 50 tasks, or Heavy SPARC > 1 s at 200 tasks, on one quiet core. Retune the tiers on dev, or restrict claims to ≤ 50 tasks.
- **K9 novelty.** Any unread HeteroMRTA citer already evaluates dynamic HeteroMRTA against a rolling optimizer. Drop the "first dynamic HeteroMRTA study" framing, and add it as a baseline if code exists.

## 9. Draft pre-registration endpoints

### 9.1 Common rules

- **Unit of analysis.** The instance: mean over its 3 CRN seeds of log r, where r = makespan(SPARC) / makespan(competitor).
- **Estimate and CI.** Geometric-mean ratio with a cluster bootstrap over instances (10,000 resamples, seed 0). Also report the median, the Hodges-Lehmann estimate and win counts.
- **Tests.** One-sided paired t-test on instance means.
- **Failures.** Success is analysed separately (exact paired test). The primary ratio uses pairs where both methods succeed. Imputing failure = 200 is a sensitivity analysis. With 100% success, as in all good-comms pilots, the two coincide.
- **Multiplicity.** One confirmatory family: Holm over E1 (6 competitors) + E2a + E2b + E3a. Everything else is secondary or descriptive, with Holm within its own family.
- **Power.** Pilot per-instance SD of log r is about 0.06 under good comms, so 200 instance clusters give a CI of about ±1%. Under C3 the SD is 0.07-0.31, giving a pooled CI of about ±3%. Per-setting TOST in C3 is therefore not powered and is reported descriptively only.

### 9.2 C1-dyn (primary)

- **E1.** For each competitor c ∈ {B1 RL(g.), B2 RL-MPC-Heavy, B3 RH-CP-SAT-LNS-Light, B4 RH-constructor-Light, B5 D-ITAGS-style-Light, B6 CBTA-style (or its declared replacement)}: mean log r < 0, pooled over the 4 primary settings and families F1-F3.
  - A "clear margin" is stated only if the pooled ratio ≤ 0.97 and the ratio is < 1 on ≥ 3/4 settings.
  - Expected: B1 0.84-0.87; B2 ≤ 0.85; B3-Light ≤ 0.95; B4 0.90-0.97; B5 0.95-0.99; B6 ≤ 0.95.
- **Secondary.**
  - Per-setting and per-family ratios.
  - MA-AT-50-5-200.
  - B3-Heavy and B2 at other budgets.
  - B7.
  - R-b (every-event RH).
  - F0 control: TOST ±2% vs open loop.
  - Hindsight and FULL gaps.
  - Travel distance.
- **Identity check.** SPARC vs R-a: identical decisions on 100% of events. This is not a hypothesis.

### 9.3 C2-dyn

- **E2a.** SPARC Light vs SPARC Heavy (3000 iterations): TOST ±2% (|mean log r| < 0.0198), pooled over F1-F3.
- **E2b.** SPARC-Light (1 core) vs B3-Heavy (8 cores, 1 s per event): TOST ±2%, or superiority.
- **Secondary.**
  - Anytime curve at {0, 30, 300, 3000} iterations.
  - Per-event CPU p50/p95 on a quiet pinned core (thresholds 50 ms at 50 tasks and 0.5 s at 200 tasks; descriptive).
  - CPU per episode.
  - Under sensitivity T (LoRR mapping): SPARC-Light not worse than any Heavy method by more than 1% (one-sided).

### 9.4 C3 (confirmatory only if K5 and K6 passed on dev; otherwise all descriptive)

Conditions: {F1-R2, F3 p_f 0.2} × ρ ∈ {2, 1.25, 1, 0.75, 0.5} × tier T2 with GE loss. Each arm uses its dev-tuned lease.

- **E3a (worst-case regret).**
  - For each arm, regret(ρ) = GM(arm) / min over fixed architectures A = {CEN-F, HYB, REP-clamp, INF-r} of GM at ρ.
  - MaxReg = max over the 5 ρ levels and 2 families.
  - Hypothesis: MaxReg(SPARC) < min(min over A of MaxReg, MaxReg(DS)).
  - Test: simultaneous cluster bootstrap of the difference; the one-sided 95% bound must exclude 0.
  - Pilot analogue: SPARC-like 1.034 vs 1.065, which is marginal.
- **Secondary.**
  - SPARC vs CEN-F at ρ = 0.5 (superiority; pilot about 0.84).
  - Non-inferiority vs CEN-F at ρ ≥ 1 (margin 3%).
  - Success non-inferiority vs every arm (2 pp).
  - Wasted trips, travel, messages and bytes (costs; no "fewer" claim).
  - RL-dec and CBTA-dec (superiority).
  - Beacon factor.
  - Corner station.
  - Bernoulli and Rayleigh channels.

### 9.5 Decision rule

- Track D enters the RA-L paper as method evidence only if all of the following hold: E1 holds against every competitor after Holm; T1 passes; E2a holds.
- C3 is claimed only if E3a holds. Otherwise C3 is reported as benchmark plus diagnosis, with DS as the recommended deployment rule.
- No endpoint, family, margin or competitor may change after the test run.

## 10. Why this is not CBJA again

| CBJA failure | Countermeasure here | Residual risk |
|---|---|---|
| Self-defined world, self-built testbed | HeteroMRTA instances, generator and published RL checkpoint unchanged; static reduction bit-identical; LoRR delay generator and CV_MRTA channel protocol; conventions labelled as such (3.3) | R1/R2/R3, failures and the station are conventions; no published dynamic HeteroMRTA numbers exist |
| Weak, in-house baselines | the published RL online, plus RL-MPC, rolling CP-SAT-LNS at a heavier tier, a constructor with restarts, targeted repair, a CBTA-style auction; one infrastructure list (5.1); equal tuning budgets; the central re-solve with the same engine is declared an identity, not beaten | CBTA and D-ITAGS are reimplementations; wins over them are not counted as external SOTA evidence |
| Method equal to a simple rule | the simple rules are declared as rules (4.2, 4.5, 4.8); the claim is the planner, and T1 tests it against the strongest cheap rule at equal budget; C3 is gated on beating a deployment switch | C3 may collapse to DS (prior: likely) |
| No headroom or upper bound | Phase-1 static gaps (S1); hindsight and FULL references; F0 control with parity expected and reported | the hindsight reference is not a bound |
| Mock-review loop, forking paths | families, endpoints, gates and one test run frozen here; every family reported | families were chosen after pilots, which is disclosed |
| Underpowered cells | 50 instances × 3 seeds × 4 settings, cluster bootstrap, power from pilot SDs | C3 per-setting cells stay underpowered |
| Engine changes mid-campaign | pinned solver commit, regression tests, frozen snapshot, deterministic budgets | concurrent edits to `cbba_sota` |
| Infrastructure asymmetry (the no-text shortcut) | leases, rally, predictors, gossip, beacon payload and budgets shared by all | per-method lease tuning may still favour someone; K7 |

## 11. Risk register

| # | Risk | Likelihood | Impact | Mitigation / trigger |
|---|---|---|---|---|
| 1 | The C1-dyn margin over non-RL planners is small (SA-BT flat; B4/B5 within 3%), so the result reads as "Phase-1 ALNS in an event loop" (prior art: Demiray et al., C&IE 2026; Calvo & Capitan, T-RO 2025) | medium-high | high | T1/K1; frame as empirical SOTA on coalition makespan, not as a framework novelty |
| 2 | C3 does not separate: the stranded re-plan rule or DS matches SPARC; leases and the beacon dominate | high | medium | K5-K7; fall back to the diagnosis section; DS reported |
| 3 | Fidelity: pilot numbers come from surrogate executors and list-scheduling planners; the ranking shifts once in `DynTaskEnvX` with ALNS and T2 | medium | high | K0 first; re-derive every gate on the real stack |
| 4 | F3 (failures) never piloted; suspicion rule 4.6 untested | medium | medium | K2 on day 2-3; declare 4.6 as a convention |
| 5 | Novelty pre-emption by DODRL-HMRTA or the CSCWD 2026 paper (abstracts inaccessible) | low-medium | high | K9 on day 1 (library access or email the authors) |
| 6 | RL-MPC or RL-dec seen as a strawman (zero-row patterns outside training; nominal model) | medium | medium | report RL with oracle state; RL-MPC at the Heavy tier; optional retraining with the released Apache code is out of scope for 2 weeks and listed as a limitation |
| 7 | CBTA reimplementation disputed or late | medium | low-medium | fidelity check vs CBGA; declared replacement slot |
| 8 | Weak anchors (R1 is only animation code; conventions) | certain | medium | label as "inspired by" or "convention"; LoRR delay and CV_MRTA protocol carry the external weight |
| 9 | Timing invalid on the shared, overloaded host; concurrent ALNS edits | high | medium | deterministic budgets; one calibration on a quiet pinned core; pinned commit |
| 10 | Redundancy cost: the makespan win at ρ = 0.5 is paid in wasted travel | high | medium | travel and wasted trips co-reported as a Pareto view; no claim that ignores them |
| 11 | Scale: RL-MPC at 200-500 tasks is too costly; per-event time at 150x500 unmeasured | medium | low | 200 tasks secondary; 500 compute-only (K8) |
| 12 | Hub-favouring world choices (central station, rally to station) | medium | low | corner-station sensitivity; stated in limitations |

## 12. Open items before freeze

- Verify the citations marked "verify" (degree-of-dynamism source). Read the four unread papers (K9).
- Decide whether SA-BT-25-5-50 (a Phase-1 H1 setting) joins C1-dyn as a fifth primary setting. The default is no: descriptive only.
- The user decides the C3 wording in `docs/prereg-phase1.md` and commits the frozen spec together with the solver commit hash.
