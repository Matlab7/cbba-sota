#!/bin/bash
# Regenerates the headroom / anytime tables of docs/headroom-2026-09.md from the rows in runs/bks and runs/anytime
# (gitignored). Campaign commands: see docs/headroom-2026-09.md, section "Reproduce".
set -e
cd "$(dirname "$0")/../../.."
OUT=docs/results/phase1b
S="SA-BT-25-5-20 SA-AT-25-5-20 MA-AT-25-5-20 SA-BT-25-5-50 SA-BT-50-5-50 SA-AT-50-5-50 MA-AT-25-5-50 MA-AT-50-5-50
   MA-AT-50-5-200"  # the settings measured on val (docs/headroom-2026-09.md)
.venv/bin/python scripts/bks.py collect --settings $S --csv $OUT/bks_val.csv > $OUT/bks_val.txt
.venv/bin/python scripts/anytime.py report --settings $S --csv-dir $OUT --png $OUT/anytime_val.png --md $OUT/anytime_val.md \
  > $OUT/anytime_val.txt
gzip -c runs/bks/val_bks.json > $OUT/bks_val_plans.json.gz
.venv/bin/python scripts/anytime.py report --settings $S --keep-disturbed --md $OUT/anytime_val_all.md > $OUT/anytime_val_all.txt
