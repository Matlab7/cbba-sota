# Team briefing slides

Animated reveal.js deck (Korean) on the research so far: direction changes, problem, ALNS method, the
compute-matched pre-registered protocol and the results of the paper draft.

**To share, send `team-briefing.html`**: one self-contained file (figures, reveal.js, KaTeX and its fonts inlined;
works offline, only the Pretendard font falls back to system fonts). Rebuild it after editing `index.html` with
`python docs/slides/build_single.py`. `index.html` is the editable source (loads its libraries from CDNs).
Keys: arrows, `F` full screen, `O` overview, `S` speaker notes; append `?print-pdf` to the URL and print for a PDF.

Numbers come from `paper/tables/numbers.tex`, which is currently generated from the val dry run
(`docs/results/val-c1`). Replace them after the test campaign (`docs/results/test/analyze.sh`, `paper/tools/make_tables.py`).

Figures in `fig/`: `overview.svg` (paper Figure 1, compiled standalone from `paper/figures/overview.tex`),
`budget.svg` (from `paper/figures/budget.pdf`), `c1_val.png` (from `docs/results/val-c1`).
