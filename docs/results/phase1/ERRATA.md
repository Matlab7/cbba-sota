# Errata for the Phase 1 record

- `docs/research/phase1-build-2026-09-27.json`, key `cpsat`: the MA-AT-9-3-20 greedy numbers (greedy_repo 51.11 on test, fixed greedy success 0.44) and the "generator difference" remark were computed on superseded kb=3 instances. On the regenerated kb=5 instances greedy_repo is 63.448 (test) vs the paper's 64.529, and there is no generator difference. The stale rows are in `runs/baselines/_superseded/`.
- `docs/results/phase1/report.txt` uses the dropped 0.92-gate framing and a "best competing method" that includes RL + ALNS polish. The C1 re-analysis (every competitor separately, Holm) is in `docs/results/phase1b/report_c1.txt`.
