# Phase 1b plan (after Phase 1 build, 2026-09-27)

## What Phase 1 showed (dev + non-blind RALTestSet only)

- Exact env-consistent evaluator (bit-identical on 354 + 480 adversarial plans), RL reproduction within ±2 SE on 13/16 settings (the rest favour RL), CP-SAT feasible, CTAS-D references replay exactly.
- The published RL SOTA is weak: our 1 s regret-insertion constructor beats RL(s.10) on 50/50 instances in all 11 settings checked (ratio 0.78–0.86).
- RALTestSet (15×20): CP-SAT proves 21/50 instances optimal in 30 s; every competent method is within ~0.5% of the best known. No headroom at 20 tasks.
- 50-task dev, matched compute (30 s × 8): ALNS v1 beats CP-SAT-LNS 20/20 on every setting, ratio 0.911 (MA-AT) to 0.968 (SA-BT).
- Large dev: cold ALNS v1 loses to the 1 s constructor; warm-started ALNS beats it by 5–6%.
- ALNS v1 stagnates after ~1–2 s on small/medium instances.

## Open question for the kill gate

The draft gate (≤ 0.92 vs the best competitor at matched budget) is now against our own strong classical baselines (constructor, CP-SAT-LNS). Whether 8% is even achievable depends on the remaining headroom, which is unknown. Phase 1b measures it before the gate is frozen.

## Work items

1. **Fix verifier defects**: instance fingerprint + git commit in every row and in resume keys; one success definition (all tasks finished and makespan < 200) everywhere, env flag kept separately; CP-SAT keys from arc literals; replayed pilot references; RL scored by its own rollout; CPU affinity pinning, load and throttling recorded per row.
2. **Fresh validation split** (seed base 900000 + 1000·idx + i, 20 per setting) for untuned dry runs; dev stays the tuning split.
3. **ALNS v2**: warm start from the constructor (budget share); anti-stagnation (elite pool with restarts and larger destroy, or HGS-style population with order crossover on keys and coalition inheritance); faster large-instance search (neighbourhood-limited slots, removal scaled with T). Tuned on dev only.
4. **Best-known solutions (BKS)** on the validation split: portfolio of ALNS v2 and CP-SAT-LNS, long runs (≥10 min × 16 cores) per instance; report every method's gap to BKS and the share of the constructor-to-BKS gap it closes.
5. **Anytime curves** (0.1–100 s, 1 and 8 cores) for ALNS v2, CP-SAT-LNS, constructor restarts, RL(s.N) GPU lockstep; quality vs wall-clock and vs CPU-seconds.
6. Decide the gate with the user from 4–5; then freeze the prereg, commit, and run the test split once on a quiet, pinned host.

## Before freezing the static claims (Track S review, 2026-09-28)

Source: `docs/research/trackS-2026-09-28.json` (key `review`). Must-do before the prereg freeze:

1. **Stronger, properly parallel competitor.** CP-SAT-LNS used about 53% of its 8 cores (13-36% on 500 tasks). Add a parallel CP-SAT-LNS (8 concurrent single-thread sub-solves or CP-SAT's own LNS workers on the full model, hinted with the best of 8 constructions) and report CPU-seconds used. Add at least one established external method (CTAS-D with an open MIP solver, or a published VRP-with-synchronization / coalition metaheuristic).
2. **Pinned runs everywhere.** run_alns, alns_ablation and cpsat_lns_ref do not pin; all ALNS v2 dev rows are unpinned. Run the two 500-task settings through the pinned anytime harness on val before any C1/C2 claim there.
3. **Honest C2.** At 2 s on 1 core ALNS does not beat 8-core constructor restarts on SA-AT-50-5-50 (1.015 [1.001, 1.028]); reported in docs/headroom-2026-09.md.
4. **No self-referential headline.** "Gap to BKS" is "gap to the best of our own runs"; add an independent reference on 50-task instances (hinted CP-SAT full model with bound, or another algorithm family).
5. Minor: routes base convention in stored plans; RL first batch of 1 at low budgets; refuse timed campaigns on a dirty tree; disclose that constructor rows on the test split were seen before the freeze; report that 8 workers add only 0.6-2.1% over 1 worker.

Status 2026-09-28 (second server; `docs/baselines-2026-09.md`):
1. Done. PCPSAT (parallel CP-SAT LNS, CPU share 1.00), CPFULL (full model, CP-SAT portfolio) and CTAS-D on an open
   solver (verified against the paper's Gurobi runs) added and tuned on dev; CPU seconds reported per method. ALNS v2
   beats every one of them on all 8 H1 settings at B1 on val (46/46 Holm-significant).
2. Done. Every runner pins; the two 500-task settings ran through the pinned anytime harness on val (n = 5).
3. Reported: on this host C2 at 2 s holds on 4 of 8 settings; it fails on SA-AT-50-5-50 and SA-BT-50-5-50 and, at n = 5,
   on the 500-task settings.
4. Done. No run without an ALNS component (incl. 300-s x 8-core CP-SAT references) matches the best ALNS-derived plan
   on any 50- to 500-task val instance (2.8-12.0% worse on average); certified bounds stay 12-58% below.
5. Done except the disclosure, which belongs to the prereg freeze: routes base stored, RL first batch of 1 at budgets
   up to 10 s, dirty trees refused, 8 workers vs 1 reported (0.4-2.7%).

