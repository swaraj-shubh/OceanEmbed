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
- **Depth-weighted MSE — tried, and the reasoning above turned out to be backwards.** The
  blended score is an n-weighted RMS across depths with roughly equal n per level, i.e.
  almost exactly the mean per-depth MSE in °C — so **plain masked MSE is already the loss
  matched to the reported metric**, and any re-weighting is a trade rather than a free win.
  What was measured (`configs/m4_dw.yaml`, weights = 1/σ² from the frozen train stats, which
  run 0.38 at 100 m up to 2.53 at 1000 m): blended slightly *worse* (0.865 vs 0.860 on val),
  50 m worse (1.142 vs 1.040), and 500 m / 1000 m clearly **better** (0.292 vs 0.347, 0.234
  vs 0.257). Weighting *toward* the thermocline as originally proposed would have made the
  known-weak deep levels worse still. It is kept not as a replacement but as an **ensemble
  member**, because it is complementary to the plain-MSE model. See
  [doc 10](10-experiment-programme.html).
- A **vertical-gradient loss** (penalising error in level-to-level differences) was also
  tried, on both M2 and M4, and is negative both times. Do not revisit.
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

### Track B — independent observations (the headline)

The PS names **INCOIS LAS Gridded ARGO** as the in-situ dataset, and it also explicitly permits interpolation where resolutions differ (requirement 7). Since that product is 1° / 10-day while we output 0.25° / daily, we run **two sub-tracks** — B1 for compliance, B2 for rigour.

#### B1 — vs INCOIS LAS Gridded ARGO *(PS-mandated)*
1. Pull the gridded Argo T field for the test period (1°, 10-day or monthly).
2. **Aggregate our prediction up to the comparison grid** — average our 0.25° daily field over each 1° cell and over the 10-day window. Aggregate *our* output up; never interpolate the coarse observation down. Downscaling the reference invents detail it does not have and flatters the score.
3. Interpolate the gridded product's depth levels onto the 15 SIH depths (or compare on the intersection of levels — state which).
4. Depth-wise metrics over ocean cells only.

#### B2 — vs raw Argo profiles *(stricter, our scientific claim)*
1. Match each profile's date → that day's prediction; nearest 0.25° cell (≤ ~18 km mismatch, same order as OceanDepths' tolerance).
2. Interpolate the profile to the 15 SIH depths (acceptance rule in doc 04 §3 — no extrapolation).
3. Depth-wise metrics over all valid (profile, depth) pairs.

Report both tables. B1 is what the PS asked for; B2 is the one that proves we resolve structure the 1° product cannot. Where they differ, that gap quantifies what objective analysis smooths away — a result, not an embarrassment.

Argo data is **never** an input or a training target at any stage, in either sub-track. If asked *"but GLORYS assimilated those Argo floats"* — correct, and it's why we report Track B at all rather than resting on Track A; this is the standard protocol (OceanBench, OceanForecastBench, OceanDepths). Genuinely assimilation-free validation would need withheld cruise data we don't have. State the limitation plainly; judges reward that more than an overclaim.

#### B3 — post-processing stages (added in [doc 10](10-experiment-programme.html))

Two stages sit between the network and the reported number. Both are model-agnostic, both
are **always reported as their own table rows**, and neither puts Argo into training.

1. **Ensemble.** Average N checkpoints' prediction cubes. Members may differ in seed *and*
   in config; cubes are aligned on the intersection of their time axes, because a window=7
   model cannot predict a split's first six days. Worth −3.5%.
2. **Depth-wise bias correction.** Subtract a 15-number offset — mean(prediction −
   observation) per depth — **fitted on validation-split Argo only**. `src/bias_correct.py`
   refuses to fit on test; `predict_cube.py --offset` refuses to apply an offset fitted on
   the split being scored. Worth −8.5%. Depth × month was tried and is worse (180 bins over
   ~3,400 casts overfits). The bias **drifts** with time, so refit annually against the most
   recent year.

### Selection discipline (non-negotiable)

Interventions are screened at **one seed on the val split (2022 Argo)**, promoted to three
seeds only if they pass, and the **test split is read once**, at the end, on the frozen
winner. Doc 09 selected on test; doc 10 does not. Ensemble composition is also chosen on
val, *uncorrected*, so the bias offset is never fitted and selected on the same data.

### Statistical testing

Model comparisons use `argo_eval.paired_bootstrap`: 1,000 paired resamples **blocking by
Argo float, not by profile**. The 6,448 test casts come from only 147 floats, and two casts
from one float ten days apart in the same water mass are not independent samples — a
profile-level bootstrap reports intervals roughly √(6448/147) ≈ 6.6× too narrow, which is
enough to turn seed noise into a publishable result. This is what finally showed that M3
attention's advantage has a 95% interval of [−0.011, +0.011], i.e. nothing.

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

Filled in, for the frozen final model (6-member ensemble + Argo bias correction), against
6,056 independent Argo casts on the 2023–24 test split:

| Depth (m) | RMSE | MAE | Bias | Corr | R² |
|---|---|---|---|---|---|
| 0 | 0.426 | 0.292 | −0.108 | 0.965 | 0.925 |
| 50 | 1.011 | 0.772 | −0.253 | 0.889 | 0.774 |
| 100 | 1.372 | 1.056 | +0.299 | 0.819 | 0.653 |
| 200 | 0.680 | 0.523 | +0.176 | 0.953 | 0.902 |
| 500 | 0.245 | 0.185 | −0.018 | 0.982 | 0.963 |
| 1000 | 0.216 | 0.165 | −0.014 | 0.973 | 0.945 |
| **blended** | **0.786** | | | | |

Full 15-depth version in [doc 10](10-experiment-programme.html); regenerate with
`python src/ablation.py --split test`.

Plus one figure: RMSE-vs-depth curves for M0…M4 plus the final model on one plot (the single
most persuasive figure we can make). The final curve sits below M0 at **every** depth —
including 500–1000 m, where every single model lost to climatology until the bias correction
was added.

## 4. Ablation table (deliverable)

Same test set, same metrics, rows = M0 climatology · M1 tiny CNN · M2 U-Net 7-var · M3 +attention · M4 +ConvLSTM · (optional) M2 with SST-only vs 7-var — the last one directly proves the multivariate claim.

**Generate it, don't hand-maintain it:** `python src/ablation.py --split test` reads every
`results/*_test_argo.csv`, averages across seeds, reports the spread, and writes
`results/ablation_test.md`. The `--split val` form is what selection decisions are read from.

**Include the failures.** As of [doc 10](10-experiment-programme.html) the table carries
seven model-side interventions — attention, ConvLSTM, gradient loss, anomaly formulation,
climatology-as-input, auxiliary channels, depth weighting — **none of which improved the
observational score on its own**, against two output-side stages that both did. That
asymmetry *is* the research finding, and it only reads as a finding if the failures are on
the page. Keep the `M0 climatology` and `GLORYS12V1 target` rows in every version of the
table: the model's number is only interpretable between that floor and that ceiling.

## 5. Experiment hygiene

- One YAML per run in `configs/`; run name = `m3_attn_lr3e4_2026xxxx`.
- Log train/val curves + final metrics to CSV (or W&B free tier) — every number in the deck must be traceable to a run.
- Frozen final artifact for the demo: **`results/FROZEN.md`** is the manifest — six
  checkpoints (`m4_convlstm_s{1,2,3}_best.pt`, `m4_dw_s{1,2,3}_best.pt`), their two configs,
  `norm_stats.json`, the climatology and bathymetry caches, and
  `results/ens_mix6_offset.json` (the bias offset). All mirrored to
  `s3://oceanembed-sih26-data/oceanembed/checkpoints/`.
- **Never train without the checkpoint sync running.** `deploy/setup.sh` now starts it
  automatically; it used to be a separate manual command, and a day's worth of checkpoints
  existed only on one instance's root volume because nobody ran it.

## 6. Compute plan

1. **All of M0–M3 on Kaggle T4 (free).** The model is ~10M params on 96×176 grids — hours, not days, per run.
2. M4 + ablation sweep: still likely Kaggle-feasible; if the 30h/week GPU quota binds, RunPod 4090 (~₹65–100/hr) or AWS g6.2xlarge Spot (~₹37/hr). Total paid budget ceiling ₹5–10k.
3. Never train on the Windows dev box; never rent A100/H100; never leave cloud instances running idle.
