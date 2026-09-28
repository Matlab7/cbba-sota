# Track D week 1: decision memo (2026-09-28)

Written from the result files in `docs/results/trackD-week1/` after the week-1 workflow was cut off by a host restart
(the memo agent never ran; the C1-dyn pilot agent hung after writing its results). Every number below is quoted from
those files; all are dev split, ground-truth env `DynTaskEnvX@6fabb0570f` (the hash K0 verified). Spec:
`docs/trackD-spec.md`.

## Gates

| Gate | Result | Status | Source |
| --- | --- | --- | --- |
| K0 fidelity | executor vs env 12,760/12,760 episodes bit-identical; RL static reduction bit for bit | pass | commit 1f0f046 |
| **K1 planner collapse (T1)** | SPARC-Light / rolling constructor B4 at equal CPU = **0.9847 [0.9806, 0.9890]** pooled (rule: <= 0.97); < 1 on 4/4 settings; holds at every B4 budget tested and with families weighted equally | **fires** | c1dyn.md |
| K2 no dynamic separation | SPARC / insertion-only on F3 = 0.828 (search pays under failures); SPARC / B4 on F1 = 0.988 | does not fire | c1dyn.md |
| K3 RL anomaly | SPARC-L / RL(g.) online 0.914; / RL-MPC Heavy (B2) 0.906 | does not fire | baselines.md |
| K4 heavy beats light | SPARC-L / rolling CP-SAT-LNS Heavy (B3-H) 0.919: B3-H is 8% worse | does not fire | baselines.md |
| C2-dyn preview (E2a) | SPARC-L / SPARC-H = 1.0229, 90% CI [1.0189, 1.0269]: Light is 2.3% behind Heavy | light claim fails | c1dyn.md |
| K5 C3 headroom | best fixed architecture at rho = 0.5 is 1.78x FULL | does not fire (headroom exists) | c3_gates_4settings.txt |
| **K6 C3 separation** | max regret over the connectivity grid: INF-r 1.006, REP-clamp 1.170, DS 1.170, SPARC-V3 1.184, HYB 1.192, SPARC 1.205, CEN-F 1.382; SPARC margin -18.6 points | **fires** | c3_gates_4settings.txt |
| K7 lease fragility | the best fixed architecture changes across the common-lease grid (INF-r, REP-clamp, HYB) | fires (rankings flip with the lease) | c3_gates_4settings.txt |
| K8 compute | not run (needs a quiet pinned host) | deferred | — |
| K9 novelty | no HeteroMRTA citer evaluates it under release/noise/failures against a rolling optimizer | pass | trackD-novelty.md |

Additional finding (baselines.md): a CBTA-style earliest-start re-auction (B6, about 3 ms per event, no search) is as
strong as the rolling constructor: SPARC-L / B6 = 0.9864 [0.9796, 0.9932]; on SA-AT-50-5-50 B6 beats SPARC-L (1.0148).
It replicates on fresh CRN seeds (0.9837).

## What the gates imply (spec Section 8)

- **K1 fires: the dynamic-method claim is dropped.** Under structural dynamics a cheap rolling rule (constructor
  restarts, or a CBBA-style re-auction) is within 1.5% of SPARC at equal CPU. The large static search advantage does
  not carry over to the online setting, where information arrives piecemeal and each re-plan only touches a few tasks.
- **K6 fires: C3 is descriptive.** Under partitioned links the simplest rule, INF-r (do not assign work to unreachable
  robots and ignore their stale commitments), has the lowest worst-case regret; SPARC's anchored replication is 19
  points worse. K7: rankings depend on the lease setting.
- What survives as findings for an analysis section:
  - Search pays on structural events that break coalitions: under robot failures SPARC beats insertion-only repair by
    17% (0.828), and rolling CP-SAT-LNS and RL-based planners are 8-9% worse than SPARC-Light.
  - Noise alone does not justify re-planning (S2), and every-event re-planning is 1% better than structural-only
    triggers here (T2 = 1.0098), contrary to the pilot prior S8.
  - The published RL policy degrades by 11-34% under release and noise and is 9% behind SPARC online.
  - Under partitions, staleness of teammate state dominates, and a one-line protocol rule is the most robust choice.

## Consequence for the project

The contribution is the **static planner** (Track S): ALNS v2 is significantly better than every competitor at matched
compute on the untuned validation split and cheap enough for real-time use. Track D becomes an analysis appendix: where
search matters online (failures) and where cheap rolling rules and simple protocols suffice (release, partitions).

Before any freeze, the Track S must-do list in `docs/plan-phase1b.md` applies: a properly parallel CP-SAT-LNS and an
external method, pinned runs including the 500-task settings, honest C2 wording, and an independent reference for the
50-task instances. Track D's SPARC still pins the pre-v2 kernel (4cf5e04); re-pinning to v2 would not change K1 (the gap
is to cheap rules, not to a weaker kernel), but must be done before any Track D number is reported.

## Open items

- The C1-dyn relaxed-key variant (SPARC-rk / B4 = 0.9717 / 0.9665 on fresh seeds) and SPARC-Heavy (0.9627) were found
  after T1 failed; using either would be a forking path and needs a fresh split to confirm.
- The C3 design3 follow-up campaign (adopt keep vs follow) was killed by the host restart; c3.md was not written.
- K8 compute on a quiet host.
