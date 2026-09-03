#!/usr/bin/env bash
# docs/10 task 8: three seeds for the one promoted config, then val cubes for ensembling.
set -u
cd ~/OceanEmbed
PY=/opt/pytorch/bin/python
export PYTHONUNBUFFERED=1

echo "############ seeds 2 and 3 for m4_dw ############"
for s in 2 3; do
  echo "======== m4_dw seed $s  ($(date -u +%H:%M:%S)) ========"
  $PY src/train.py configs/m4_dw.yaml --seed $s 2>&1 | tail -2
done

echo "############ val cubes for every m4_dw seed ############"
for s in 1 2 3; do
  $PY src/predict_cube.py --ckpt checkpoints/m4_dw_s${s}_best.pt --split val 2>&1 | tail -2
done

echo "############ test cubes (needed for the final read) ############"
for s in 1 2 3; do
  $PY src/predict_cube.py --ckpt checkpoints/m4_dw_s${s}_best.pt --split test 2>&1 | tail -2
done

aws s3 sync checkpoints s3://oceanembed-sih26-data/oceanembed/checkpoints --quiet
echo "############ PHASE 2 DONE $(date -u) ############"
