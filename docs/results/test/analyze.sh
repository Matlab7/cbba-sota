#!/bin/bash
# Regenerates the confirmatory test-split tables of the paper from runs/anytime_test (gitignored), the campaign of the
# frozen pre-registration (docs/prereg-phase1.md): grid "test", 8 H1 settings x 50 instances, host dhcho-dev-2gpus-0.
set -e
cd "$(dirname "$0")/../../.."
OUT=docs/results/test
H1="SA-BT-25-5-50 SA-BT-50-5-50 SA-AT-50-5-50 MA-AT-25-5-50 MA-AT-50-5-50 MA-AT-50-5-200 MA-AT-150-10-500 MA-AT-150-5-500"
# best of every test-split run (the timed campaign only: nothing else ran on test)
.venv/bin/python scripts/bks.py collect --split test --settings $H1 --csv $OUT/bks_test.csv > $OUT/bks_test.txt
gzip -c runs/bks/test_bks.json > $OUT/bks_test_plans.json.gz
.venv/bin/python scripts/anytime.py report --split test --grid test --out runs/anytime_test --settings $H1 \
  --csv-dir $OUT --md $OUT/c1_test.md > $OUT/c1_test.txt
.venv/bin/python scripts/anytime.py report --split test --grid test --out runs/anytime_test --settings $H1 \
  --keep-disturbed --md $OUT/c1_test_all.md > $OUT/c1_test_all.txt
