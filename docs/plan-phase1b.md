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
