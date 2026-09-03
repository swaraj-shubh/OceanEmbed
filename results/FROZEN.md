# Frozen final model (docs/10 task 10)

FINAL = 6-member ensemble + Argo bias correction. Blended RMSE 0.786 degC against 6,056
independent Argo casts, test split 2023-24. Selected on val; test read once.

Members (all in s3://oceanembed-sih26-data/oceanembed/checkpoints/):
  m4_convlstm_s1_best.pt  m4_convlstm_s2_best.pt  m4_convlstm_s3_best.pt   configs/m4_convlstm.yaml
  m4_dw_s1_best.pt        m4_dw_s2_best.pt        m4_dw_s3_best.pt         configs/m4_dw.yaml

Post-processing:
  results/ens_mix6_offset.json   depth-wise offset, fitted on 2022 (val) Argo, SUBTRACTED

Artifacts the model needs:
  data/processed/norm_stats.json        train-split channel stats
  data/processed/nio_daily.clim.npy     train-split monthly climatology
  data/processed/nio_daily.bathy.npy    train-split bathymetry (only for extra=[aux] runs)

Reproduce:
  python src/predict_cube.py --split test --run ens_mix6 --ensemble \
    results/m4_convlstm_s{1,2,3}_best_test_cube.nc results/m4_dw_s{1,2,3}_best_test_cube.nc \
    --offset results/ens_mix6_offset.json
