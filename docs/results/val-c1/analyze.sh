#!/bin/bash
# Regenerates the tables of docs/baselines-2026-09.md from the rows in runs/tune_dev, runs/bks, runs/anytime and
# runs/anytime_c1 (gitignored). Campaign commands: see docs/baselines-2026-09.md, section "Reproduce".
set -e
cd "$(dirname "$0")/../../.."
OUT=docs/results/val-c1
H1="SA-BT-25-5-50 SA-BT-50-5-50 SA-AT-50-5-50 MA-AT-25-5-50 MA-AT-50-5-50 MA-AT-50-5-200 MA-AT-150-10-500 MA-AT-150-5-500"
# dev tuning of the new CP-SAT competitors (8 cores at B1)
.venv/bin/python scripts/anytime.py tune-report --out runs/tune_dev > $OUT/tune_dev.txt
# best known solutions: every run so far (both hosts, BKS runs, independent references)
.venv/bin/python scripts/bks.py collect --settings SA-BT-25-5-20 SA-AT-25-5-20 MA-AT-25-5-20 $H1 \
  --csv $OUT/bks_val.csv > $OUT/bks_val.txt
gzip -c runs/bks/val_bks.json > $OUT/bks_val_plans.json.gz
# the same-host C1/C2 campaign (grid c1), with the cells the first host also ran
.venv/bin/python scripts/anytime.py report --grid c1 --out runs/anytime_c1 --settings $H1 --csv-dir $OUT \
  --md $OUT/c1_val.md --png $OUT/c1_val.png --compare runs/anytime > $OUT/c1_val.txt
.venv/bin/python scripts/anytime.py report --grid c1 --out runs/anytime_c1 --settings $H1 --keep-disturbed \
  --md $OUT/c1_val_all.md > $OUT/c1_val_all.txt
