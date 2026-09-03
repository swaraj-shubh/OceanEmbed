#!/usr/bin/env bash
# docs/10 tasks 1 and 4-7: M4 val baseline, then six screening runs at one seed.
set -u
cd ~/OceanEmbed
PY=/opt/pytorch/bin/python
export PYTHONUNBUFFERED=1
SCREENS="m4_anomaly m4_clim m4_aux m4_clim_aux m4_dw m4_grad"

# Task 0: the sync is not optional. Day 2's checkpoints survived only because the box
# outlived the session; that is luck, not a backup.
pkill -f sync_checkpoints.sh || true
BUCKET=oceanembed-sih26-data PREFIX=oceanembed nohup bash deploy/sync_checkpoints.sh > sync.log 2>&1 &
echo "checkpoint sync running (pid $!)"

echo "############ PHASE A: M4 val baseline (task 1) ############"
for s in 1 2 3; do
  echo "--- m4_convlstm_s${s} val ---"
  $PY src/predict_cube.py --ckpt checkpoints/m4_convlstm_s${s}_best.pt --split val 2>&1 | tail -2
done

echo "############ PHASE B: screening runs, seed 1 (tasks 4-7) ############"
for cfg in $SCREENS; do
  echo "======== train $cfg  ($(date -u +%H:%M:%S)) ========"
  $PY src/train.py configs/${cfg}.yaml --seed 1 2>&1 | tail -3
done

echo "############ PHASE C: score screens on val ############"
for cfg in $SCREENS; do
  echo "--- $cfg val ---"
  $PY src/predict_cube.py --ckpt checkpoints/${cfg}_s1_best.pt --split val 2>&1 | tail -2
done

echo "############ VAL ABLATION ############"
$PY src/ablation.py --split val
aws s3 sync checkpoints s3://oceanembed-sih26-data/oceanembed/checkpoints --quiet
aws s3 sync results s3://oceanembed-sih26-data/oceanembed/results --quiet
echo "############ PROGRAMME PHASE 1 DONE $(date -u) ############"
