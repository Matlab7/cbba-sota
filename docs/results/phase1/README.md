# Phase 1 results (dev split, 2026-09-27)

Copied from `runs/` (gitignored) so the numbers are versioned.

- `report.txt`, `summary.csv`, `summary_ratios.csv`: matched-budget comparison on 5 settings x 20 dev instances (`scripts/compare_dev.py`), budgets B1 = paper RL(s.10) time and B2 = 2 x B1, 1 and 8 cores, every makespan env-scored. Produced on a shared host (load average about 62), cold-start ALNS v1.
- `rl_reproduction_gate_test.csv`: released HeteroMRTA RL checkpoint on our regenerated test split vs the paper's Tables I-IV (`scripts/rl_report.py`).

Summary and interpretation: `docs/research/decision-2026-09-27.md`, `docs/plan-phase1b.md`, and the full agent reports in `docs/research/phase1-build-2026-09-27.json`.
