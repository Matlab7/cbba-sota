#!/bin/bash
# After the confirmatory test campaign: check that it is complete, then build the reports, every table, number macro and
# figure of the paper from the test rows, and the PDFs. Usage: paper/tools/post_campaign.sh [--wait]
#   --wait  first wait until the campaign process (anytime.py run --split test) has exited.
# Exits 3 without touching anything if the campaign is incomplete (resume it with runs/logs/test_2026-09-28.sh).
set -e
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
if [ "$1" = "--wait" ]; then
  # only the campaign's python process (a shell whose command line merely mentions it does not count)
  while pgrep -f '^[^ ]*python[0-9.]* scripts/anytime[.]py run --split test' > /dev/null; do sleep 60; done
fi
remaining=$($PY - <<'EOF'
import random, sys
sys.path.insert(0, "scripts")
import anytime as at
at.SPLIT, at.GRID = "test", at.grid_test
at.OUT = at.RUNS_DIR / "anytime_test"
S = ["SA-BT-25-5-50", "SA-BT-50-5-50", "SA-AT-50-5-50", "MA-AT-25-5-50", "MA-AT-50-5-50", "MA-AT-50-5-200",
     "MA-AT-150-10-500", "MA-AT-150-5-500"]
print(len(at._units(S, 50, 50, at._done("2e53eab577d876ea1e39e948303dc50984262d20"), random.Random(0))))
EOF
)
echo "$(date -u +%FT%TZ) units left: $remaining"
if [ "$remaining" != "0" ]; then
  echo "campaign incomplete: resume with runs/logs/test_2026-09-28.sh from the campaign worktree"
  exit 3
fi
bash docs/results/test/analyze.sh
$PY paper/tools/make_tables.py docs/results/test --runs runs/anytime_test --bks-val docs/results/val-c1/bks_val.csv \
  --bks-test docs/results/test/bks_test.csv --rl-s10 runs/rl
$PY paper/tools/make_supplement.py docs/results/test
$PY paper/tools/sensitivity.py --split test
$PY paper/tools/make_figures.py --split test
$PY paper/tools/abstract_txt.py
bash paper/build.sh
$PY paper/tools/make_archive.py
echo "$(date -u +%FT%TZ) done: paper/main.pdf, supplement.pdf, openreview.txt and supplementary.zip from the test rows"
