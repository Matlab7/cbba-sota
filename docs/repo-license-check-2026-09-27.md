# Candidate benchmark repos: license and activity (checked 2026-09-27 via GitHub API)

| Repo | License | Last push | Stars | Note |
| --- | --- | --- | --- | --- |
| jlott22/CV_MRTA_Benchmark | none ("no reuse license selected") | 2026-09-12 | 0 | run locally only; do not redistribute adapted code |
| jakbichler/Sadcher | none | 2025-07-12 | 9 | 250K dataset itself is CC BY 4.0 (4TU DOI) |
| inmo-jang/space-simulator | GPL-3.0 | 2026-04-10 | 29 | |
| MAPF-Competition/Start-Kit (LoRR) | MIT | 2026-07-15 | 38 | |
| MAPF-Competition/Benchmark-Archive | none | 2025-05-09 | 17 | data license to confirm |
| zhanglixuan0720/TWPC-MRTA | none | 2023-09-18 | 5 | |
| proroklab/VectorizedMultiAgentSimulator | GPL-3.0 | 2026-05-19 | 614 | |
| facebookresearch/BenchMARL | MIT | 2026-02-07 | 667 | |
| PyVRP/PyVRP | MIT | 2026-09-26 | 701 | centralized OR reference solver |

## CV MRTA Benchmark published numbers (docs/RESULTS.md, main @ 2026-09-12)

Static known targets on a grid, Manhattan cost, 6 non-learning decentralized allocators (CBAA, ACBBA, PI, HIPC, DMCHBA, DGA), 24 impaired comm conditions + ideal.

- MinSum total-team steps, ideal / impaired mean: HIPC(h=8) 52.62 / 66.95; DGA 71.79 / 85.18; DMCHBA 76.45 / 89.99; ACBBA 72.78 / 95.73; CBAA 76.83 / 96.42; PI(h=5) 62.54 / 105.92. HIPC beats all 5 in 120/120 Holm-adjusted Wilcoxon tests.
- MinMax max-robot steps, impaired mean: DGA 24.49, DMCHBA 24.78, HIPC 25.12.
- HIPC h=8 trade-offs: +39% max-robot travel, +52% allocation publications, +208% allocator calls vs h=2.
- Headroom observation: ideal->impaired degradation of the MinSum leader is ~27% (52.6 -> 67.0); no centralized optimum reported in the summary.
