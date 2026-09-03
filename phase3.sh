#!/usr/bin/env bash
# docs/10 tasks 8 and 10: pick the ensemble on VAL (uncorrected, so the offset is never
# selected in-sample), then fit the correction on the winner's val cube and read test ONCE.
set -u
cd ~/OceanEmbed
PY=/opt/pytorch/bin/python
export PYTHONUNBUFFERED=1
R=results

echo "############ STEP 1: ensemble composition, chosen on VAL, uncorrected ############"
echo "--- A: M4 baseline x3 ---"
$PY src/predict_cube.py --split val --run ens_base --ensemble \
  $R/m4_convlstm_s1_best_val_cube.nc $R/m4_convlstm_s2_best_val_cube.nc $R/m4_convlstm_s3_best_val_cube.nc 2>&1 | tail -1
echo "--- B: m4_dw x3 ---"
$PY src/predict_cube.py --split val --run ens_dw --ensemble \
  $R/m4_dw_s1_best_val_cube.nc $R/m4_dw_s2_best_val_cube.nc $R/m4_dw_s3_best_val_cube.nc 2>&1 | tail -1
echo "--- C: mixed, all six ---"
$PY src/predict_cube.py --split val --run ens_mix6 --ensemble \
  $R/m4_convlstm_s1_best_val_cube.nc $R/m4_convlstm_s2_best_val_cube.nc $R/m4_convlstm_s3_best_val_cube.nc \
  $R/m4_dw_s1_best_val_cube.nc $R/m4_dw_s2_best_val_cube.nc $R/m4_dw_s3_best_val_cube.nc 2>&1 | tail -1

echo "############ STEP 2: build the matching TEST ensembles (not yet scored) ############"
$PY src/predict_cube.py --split test --run ens_base --ensemble \
  $R/m4_convlstm_s1_best_test_cube.nc $R/m4_convlstm_s2_best_test_cube.nc $R/m4_convlstm_s3_best_test_cube.nc 2>&1 | tail -1
$PY src/predict_cube.py --split test --run ens_dw --ensemble \
  $R/m4_dw_s1_best_test_cube.nc $R/m4_dw_s2_best_test_cube.nc $R/m4_dw_s3_best_test_cube.nc 2>&1 | tail -1
$PY src/predict_cube.py --split test --run ens_mix6 --ensemble \
  $R/m4_convlstm_s1_best_test_cube.nc $R/m4_convlstm_s2_best_test_cube.nc $R/m4_convlstm_s3_best_test_cube.nc \
  $R/m4_dw_s1_best_test_cube.nc $R/m4_dw_s2_best_test_cube.nc $R/m4_dw_s3_best_test_cube.nc 2>&1 | tail -1

echo "############ STEP 3: fit the correction on each winner-candidate's VAL cube ############"
for e in ens_base ens_dw ens_mix6; do
  $PY src/bias_correct.py --cube $R/${e}_val_cube.nc --split val 2>&1 | head -1
done

echo "############ VAL SUMMARY (this is what chooses) ############"
$PY src/ablation.py --split val
echo "############ PHASE 3 DONE $(date -u) ############"
