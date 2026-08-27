---
title: "05 · Training & Evaluation"
nav_order: 6
---

# 05 — Training & Evaluation Protocol

## 1. Loss

**Masked MSE** (baseline for all stages):

```
L = Σ_d Σ_xy  m(xy) · (Ŷ_d(xy) − Y_d(xy))²  /  Σ m
```

- `m` = land mask ∧ valid-target mask.
- Optional experiment after M4 works: **depth-weighted MSE** with weights peaking at 50–200 m (thermocline, where the error concentrates — doc 02 §3). Only keep it if the depth-wise table improves.
- Normalize targets per depth channel (train-stats), predict in normalized space, denormalize for metrics. This stops the surface layers (high variance) from dominating the loss.

## 2. Training recipe (defaults — override via config)

| Item | Value |
|---|---|
| Optimizer | AdamW, lr 3e-4, weight decay 1e-4 |
| Schedule | Cosine decay, 5% warmup |
| Batch | 16 (M1–M3), 8 (M4, sequences) |
| Epochs | 50–100 with early stopping (patience 10 on val RMSE) |
| Augmentation | Random 96×96 crops; no flips/rotations (breaks Coriolis/geography physics — winds and currents are directional) |
| Precision | AMP (fp16) — free 2× on T4/4090 |
| Seed | Fixed, logged in config |
| Checkpoints | Every epoch: model+optimizer+epoch+config+norm-stats path; keep best + last; auto-resume for Spot instances |

**Sanity gate before any real run:** overfit 10 samples to ~zero loss. If a component can't, it's mis-wired — debug before scaling.

## 3. Evaluation tracks

### Track A — dense, vs GLORYS test year (2022)
Fast, per-pixel, per-depth. Used for model development and the ablation table.

### Track B — independent, vs held-out Argo (the headline)
For each Argo profile in the region, 2021–2022:
1. Match date → the model's daily prediction; nearest 0.25° grid cell (≤ ~18 km mismatch — same order as OceanDepths' tolerance).
2. Interpolate profile to the 15 SIH depths (acceptance rule in doc 04 §3).
3. Compute per-depth errors; aggregate over all profiles.

Argo profiles are **never** inputs or targets at any stage. If asked "but GLORYS assimilated those Argo floats" — correct, and that's why Track B is still the strictest available test (industry-standard protocol per OceanBench/OceanForecastBench); truly assimilation-free validation would require withheld cruise data we don't have. Honest answer, judges respect it.

### Metrics (all reported per depth)

| Metric | Formula / note |
|---|---|
| RMSE | primary |
| MAE | interpretability |
| Bias | mean(Ŷ−Y); detects systematic warm/cold offsets |
| Correlation (Pearson) | pattern skill |
| **Anomaly correlation** | correlation after removing monthly climatology — the anti-"you just learned the seasons" metric; include it |
| R² | optional |

### The headline table (fill per model stage)

| Depth (m) | RMSE | MAE | Bias | Corr | AnomCorr |
|---|---|---|---|---|---|
| 0 / 30 / 50 / 100 / 150 / 200 / 300 / 500 / 1000 | … | | | | |

Plus one figure: RMSE-vs-depth curves for M0…M4 on one plot (the single most persuasive figure we can make — expect the M4 curve to pull below M0 mainly in the 50–300 m band).

## 4. Ablation table (deliverable)

Same test set, same metrics, rows = M0 climatology · M1 tiny CNN · M2 U-Net 7-var · M3 +attention · M4 +ConvLSTM · (optional) M2 with SST-only vs 7-var — the last one directly proves the multivariate claim.

## 5. Experiment hygiene

- One YAML per run in `configs/`; run name = `m3_attn_lr3e4_2026xxxx`.
- Log train/val curves + final metrics to CSV (or W&B free tier) — every number in the deck must be traceable to a run.
- Frozen final artifact for the demo: `checkpoints/final/{model.pt, config.yaml, norm_stats.json, metrics.json}`.

## 6. Compute plan

1. **All of M0–M3 on Kaggle T4 (free).** The model is ~10M params on 96×176 grids — hours, not days, per run.
2. M4 + ablation sweep: still likely Kaggle-feasible; if the 30h/week GPU quota binds, RunPod 4090 (~₹65–100/hr) or AWS g6.2xlarge Spot (~₹37/hr). Total paid budget ceiling ₹5–10k.
3. Never train on the Windows dev box; never rent A100/H100; never leave cloud instances running idle.
