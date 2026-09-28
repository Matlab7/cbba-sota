# Pilots (2026-09-27)

Feasibility pilots written by design-phase agents, re-checked by hand. Not the proposed algorithm.

## HeteroMRTA RALTestSet headroom (`heteromrta/`)

Benchmark: marmotlab/HeteroMRTA @ db51e29 (Apache-2.0, RA-L 2025), shipped RALTestSet, 50 instances,
15 agents (5 species x 3), 20 tasks, makespan. `heteromrta_numpy2.patch` makes the env run on
numpy 2.5 (scalar conversions only, no semantic change).

| Method | Mean makespan | Success | Source |
| --- | ---: | ---: | --- |
| TACO (shipped solutions, replayed) | 38.92 | 1.00 | replay_check.py |
| Greedy (repo) | 32.42 | 1.00 | coalition_lns_pilot.py |
| SAS (shipped solutions, replayed) | 29.74 | 1.00 | replay_check.py |
| RL greedy decode (released checkpoint) | 29.02 / 28.80 | 1.00 | run_rl_eval.py, seed 0 / seed 7 |
| RL(s.10) (released checkpoint, ~3.9 s/inst CPU) | 27.07 / 27.09 | 1.00 | run_rl_eval.py, seed 0 / seed 7 |
| Centralized coalition LNS, 4 s, 1 core, Python | 23.48 | 1.00 | replay_check.py (plans replayed in the env, 0.0 evaluator discrepancy) |

LNS / RL(s.10) = 0.867 (agent-reported 95% bootstrap CI [0.856, 0.879], better on 50/50).
Construction heuristic alone 29.70; best of 10 constructions 26.68 (simple_rule_check.py), so the gain
comes from iterative improvement, not from the constructor.

Re-run: from a dir containing `HeteroMRTA/` (patched) and `rl_eval.json`:
`<venv>/bin/python replay_check.py`. Env: `.venv-het` (torch 2.8.0, numpy 2.5.3). The released
checkpoint needs `weights_only=False` under numpy 2 (official repo file, trusted).

## CV_MRTA scale headroom (`cv_mrta/`)

`opt.py` / `opt.csv`: collision-free Manhattan lower bound on the 500 core instances (mean 19.65 MinMax,
49.33 MinSum). `cv_scale_ref.py`: near-optimal PyVRP reference on CV_MRTA-distribution scale instances.
