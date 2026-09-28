#!/bin/bash
# Tables and figures of the ALNS v2 dev campaign (runs/alns/v2g) -> docs/results/alns-v2
cd /home/jovyan/dev/cbba-sota
OUT=docs/results/alns-v2; mkdir -p $OUT
P=".venv/bin/python scripts/alns_ablation.py"
W1B="v2g/w1-t0.5 v2g/w1-t1 v2g/w1-t2 v2g/w1-t5 v2g/w1-B1"
W8B="v2g/w8-t0.5 v2g/w8-t1 v2g/w8-t2 v2g/w8-t5 v2g/w8-B1"
{
echo "== Ablation, 1 worker, B1 (ref v2): paired ratio [bootstrap 95% CI] wins-losses"
$P report --root v2g/w1-B1 v2 no-init no-resort alt v1 v1-p1 --csv $OUT/ablation_w1_B1.csv
echo; echo "== Ablation, 1 worker, 2 s (ref v2)"
$P report --root v2g/w1-t2 v2 no-init no-resort alt v1 --csv $OUT/ablation_w1_2s.csv
echo; echo "== Ablation, 8 workers, B1 (ref v2)"
$P report --root v2g/w8-B1 v2 no-init no-resort alt v1 --csv $OUT/ablation_w8_B1.csv
} > $OUT/ablation.txt 2>&1
{
echo "== Quality vs budget: makespan / CP-SAT-LNS (8 threads) at B1"
$P budgets --roots $W1B $W8B --labels v2 v1 --csv $OUT/budgets_vs_cpsat8.csv --png $OUT/budgets_vs_cpsat8.png
echo; echo "== Quality vs budget: makespan / RL(s.N) on 8 cores at B1 (compare_dev)"
$P budgets --roots $W1B $W8B --labels v2 v1 --ref RL-8 --csv $OUT/budgets_vs_rl8.csv
echo; echo "== Quality vs budget: makespan / constructor at 1 s (runs/baselines)"
$P budgets --roots $W1B $W8B --labels v2 v1 --ref construct --ref-budget 1 --csv $OUT/budgets_vs_construct1s.csv
} > $OUT/budgets.txt 2>&1
{
echo "== Time until the run's best reaches CP-SAT-LNS (8 threads) at B1"
$P ttt --root v2g/w1-B1 v2 v1
$P ttt --root v2g/w8-B1 v2 v1
} > $OUT/ttt.txt 2>&1
$P anytime --root v2g/w1-B1 v2 v1 --png $OUT/anytime_w1_B1.png > $OUT/anytime_w1_B1.txt 2>&1
$P anytime --root v2g/w8-B1 v2 v1 --png $OUT/anytime_w8_B1.png > $OUT/anytime_w8_B1.txt 2>&1
ls -la $OUT
