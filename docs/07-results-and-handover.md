---
title: "Results & Handover"
nav_order: 8
---

# 07 — Results & Engineering Handover

Everything that was actually built, measured, and learned. Written for an ML engineer
picking this up cold: what the system does, what the numbers are, what went wrong on the
way, and what I would do next.

---

## 1. What the system does

Satellites see only the ocean surface. Argo floats see the subsurface but are desperately
sparse (~0.01% coverage per depth level). This model learns the mapping between them.

**Input** `X = [7, 96, 176]` — SST, SSS, SLA, current U/V, wind U/V, daily, 0.25°
**Output** `Y = [15, 96, 176]` — temperature at 0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000 m
**Region** Arabian Sea + Bay of Bengal, 0–25°N, 55–100°E
**Record** 2015-04-01 → 2024-12-31, 3,563 consecutive days, no gaps

Trained against GLORYS12V1 reanalysis. **Validated against raw Argo profiles**, which were
never an input and never a training target.

---

## 2. Headline result

**OceanEmbed (M2) reconstructs subsurface temperature to a blended RMSE of 0.908 °C,
21.7% better than a monthly-climatology baseline, measured against 6,093 independent Argo
profiles in years the model never saw.**

Mean correlation across depths: **0.913** (climatology: 0.842).

### Depth-wise, test split 2023–24, vs raw Argo

| Depth (m) | M0 clim. | **M2** | M3 attn. | M2 MAE | M2 bias | M2 corr | n profiles |
|---|---|---|---|---|---|---|---|
| 0 | 0.745 | **0.505** | 0.514 | 0.362 | −0.100 | 0.952 | 5940 |
| 5 | 0.746 | **0.457** | 0.494 | 0.338 | −0.142 | 0.961 | 5957 |
| 10 | 0.751 | **0.492** | 0.528 | 0.354 | −0.109 | 0.952 | 5961 |
| 20 | 0.844 | 0.711 | **0.699** | 0.423 | +0.012 | 0.902 | 6016 |
| 30 | 0.949 | 0.871 | **0.851** | 0.548 | +0.126 | 0.880 | 6018 |
| 50 | 1.308 | 1.033 | **1.031** | 0.725 | +0.201 | 0.880 | 6021 |
| 75 | 1.946 | 1.426 | **1.418** | 1.064 | +0.541 | 0.836 | 6021 |
| **100** | 2.163 | **1.642** | 1.661 | 1.300 | **+0.867** | 0.801 | 6024 |
| 125 | 1.821 | 1.475 | **1.455** | 1.184 | +0.817 | 0.839 | 6028 |
| 150 | 1.385 | **1.113** | 1.113 | 0.886 | +0.505 | 0.896 | 6027 |
| 200 | 0.881 | 0.736 | **0.718** | 0.564 | +0.152 | 0.944 | 6025 |
| 300 | 0.558 | **0.533** | 0.537 | 0.363 | +0.006 | 0.956 | 6018 |
| 500 | **0.273** | 0.305 | 0.303 | 0.227 | −0.020 | 0.972 | 6007 |
| 700 | **0.263** | 0.305 | 0.296 | 0.233 | −0.118 | 0.970 | 5970 |
| 1000 | **0.237** | 0.270 | 0.289 | 0.208 | −0.055 | 0.961 | 5375 |
| **blended** | **1.160** | **0.908** | **0.908** | | | | |

Read it as: error peaks at the thermocline (75–125 m), which is exactly where the physics
is hard and where the model earns its keep (−24% vs climatology at 100 m). Below 300 m the
ocean is nearly static and climatology is genuinely hard to beat — we lose there by
10–15%, and both methods correlate at ~0.97, so neither is doing real work.

---

## 3. The three findings that matter

### 3.1 Validating against the reanalysis would have given the wrong answer

Scored against **GLORYS**, M2 looked **38% worse than climatology at the surface** — a
serious-looking regression. Scored against **Argo**, the same model is **32% better** at
the surface.

The model was disagreeing with the reanalysis, and the independent observations sided with
the model. GLORYS is itself a model with its own biases; Argo is instrumentation in the
water. This is the single strongest argument in the project for the two-track validation
design, and it is empirical rather than rhetorical.

**Implication for anyone extending this:** never report a headline number against GLORYS.

### 3.2 Attention bought nothing (a real null result)

M3 adds multi-head self-attention over the U-Net bottleneck (264 spatial tokens,
+331k parameters). That attended bottleneck is the "OceanEmbed" latent.

| | GLORYS val RMSE | Argo blended RMSE |
|---|---|---|
| M2 (1.93M params) | 0.757 | **0.908** |
| M3 (2.26M params) | 0.729 | **0.908** |

Identical to three decimals against observations, and per-depth the differences are small
and unsigned. Early stopping did not change this, so it is not overfitting — attention
genuinely does not improve agreement with reality here.

**Carry M2 forward.** It is 331k parameters lighter for the same observational skill.

### 3.3 Run-to-run variance is larger than the effects being measured

Same config, same seed, two runs: M3's best validation RMSE came out **0.660** and
**0.729** — a 10% spread from nondeterministic cuDNN kernel selection and dataloader
worker ordering alone.

That spread is **larger than the M2/M3 gap it was initially used to argue for**. An
earlier version of this analysis reported "M3 improves GLORYS RMSE by 11%" as a real
result; it was noise.

**Any architecture claim on this setup requires multiple seeds and a reported spread.**
This is the most important methodological lesson in the project.

---

## 4. Data pipeline

Raw provider files → QC → subset → regrid 0.25° → daily alignment → masks → normalise
(train-split stats only) → Zarr store. **The model never touches provider files.**

| Field | Product | Notes |
|---|---|---|
| SST | NOAA OISST v2.1 | already on the target grid, no interpolation |
| SSS | SMAP RSS L3 **V6** | 8-day running mean at its centre date; 81 missing days |
| SLA | CMEMS DUACS L4 | 0.125° → 0.25° |
| Currents U/V | NASA OSCAR v2.0 | total current (not geostrophic), 0–30 m mean |
| Winds U/V | CMEMS ASCAT L3 | MetOp-A 2015–2021 + MetOp-B 2019–2024, asc+desc |
| **Target** | **GLORYS12V1** daily | 1/12°, 36 levels 0–1100 m → 15 SIH depths |
| **Validation** | Raw Argo (Ifremer ERDDAP) | 14,313 profiles, 5.4M levels, 2021–2024 |

Store: `X (3563, 7, 100, 180)`, `Y (3563, 15, 100, 180)`, float32, Zarr v2, chunked
`time=1`. 3.1 GB. Centre-cropped to 96×176 at load time so the U-Net's 3 pooling levels
divide evenly.

Splits are strictly chronological: **train 2015-04→2021-12 (2467 d) · val 2022 (365 d) ·
test 2023–24 (731 d)**. No random mixing, no overlap.

### Pre-flight audit

An 8-section audit (`ALL CHECKS PASSED`) covers: contract and dtypes, time-axis
continuity, split disjointness, per-channel QC ranges, per-depth monotonicity, dead days,
land-mask agreement, and normalisation provenance. The leakage check is explicit —
normalisation statistics recompute from the train split (28.7476) and differ from the
whole-record mean (28.7856). Land masks derived independently from GLORYS and OISST agree
to 98.8%.

---

## 5. Bugs that would have silently corrupted results

These are documented because each produced *plausible-looking* output, and an ML engineer
inheriting this should know the traps are real.

**The target silently vanished.** Spatial dims were named `y`/`x`, so the data variable
`Y` and the coordinate `y` collided as directory names on a case-insensitive filesystem.
The written store contained `X` and `y` and **no target at all**, with no error. Works on
Linux, fails only on Windows. Dims are `lat`/`lon` now.

**The surface level was entirely NaN.** GLORYS's shallowest level is 0.494 m, so
interpolating to exactly 0 m fell outside the source range. 0 m is a headline metric depth
and the demo's first map.

**The 1000 m level was nearly extrapolated.** `maximum_depth=1000` returns levels topping
out at **902.3 m** — the next GLORYS level is 1062.4 m. Interpolating the 1000 m target
onto a 902 m ceiling is extrapolation. Ceiling raised to 1100 m.

Both depth-end bugs are now guarded by an assert on the report depths, because an all-NaN
level is invisible until the metrics table comes back empty.

**Argo surface rejection.** The acceptance rule initially refused to interpolate outside
the observed range, which rejected **every** profile at 0 m — Argo floats surface at
2–5 dbar, never exactly 0. Acceptance went 0% → 87%.

**SSS QC floor.** A 25 PSU floor would have masked the Ganga–Brahmaputra freshwater plume
(observed 15.56 PSU) as bad data — precisely the barrier-layer signal SSS is an input for.
Floor is 5.0.

**Stale product version.** SMAP SSS V4 was retired at 2022-07-11, which would have left
the validation year half empty. Switched to V6.

**Silent re-download.** When the GLORYS chunk size changed mid-download, the new filenames
stopped matching the old ones and the "already downloaded?" check missed every time — 42
chunks re-fetched days already on disk while the success counter climbed and real coverage
stood still. Skipping is now by date coverage, not filename.

---

## 6. Infrastructure

**Download.** GLORYS is 45 GB / ~19 h. Repeated "job killed" events turned out to be
**out-of-memory crashes**, not the scheduler: a monthly chunk decodes to ~1.5 GB in
float64, and the dev box had a system commit charge of 15.16 GB against a 16.15 GB limit.
3-day chunks (~37 MB) run clean. Final: 967 chunks, **zero failures**, no date gaps.

**Storage.** `s3://oceanembed-sih26-data/oceanembed/` — `processed/` (the store, 3.27 GB,
object count verified 7,148 = 7,148), `interim/` (raw archive), `results/`,
`checkpoints/`. Private bucket; share via bucket policy to specific AWS accounts, never
public-read.

**Training.** g4dn.xlarge (Tesla T4 16 GB), Deep Learning AMI, IAM instance role for S3 so
no keys live on the box. **23–24 s/epoch**, 30 epochs ≈ 12 min, ~$0.20 per run.
`train.py` auto-resumes from checkpoint; `deploy/sync_checkpoints.sh` mirrors to S3 every
5 minutes, because on a preemptible instance a checkpoint that only exists locally is not
a checkpoint.

Note: `pip install torch` on the DLAMI replaces the CUDA build with a CPU one. `setup.sh`
deliberately does not.

---

## 7. Known limitations

- **+0.87 °C warm bias at 100 m.** Systematic, not random — the model places the
  thermocline too deep or too diffuse. The clearest improvement target.
- **Worse than climatology below 500 m** by 10–15%.
- **SSS missing 81 days** (2.3%) — real SMAP outages; those days become the train mean
  after normalisation.
- **Winds are gap-filled** with a centred 3-day mean (raw swath coverage 55–86%). The
  window looks 1 day ahead — this is reconstruction, not forecasting, and it is strictly
  less lookahead than the 8-day centred SSS composite the problem statement specifies.
- **Argo covers 2021–2024 only**, so observational validation exists for val/test years
  but not for training years.
- **Single-seed results.** See §3.3.

---

## 8. What I would do next, in order

1. **Multi-seed everything.** 3 seeds per config, report mean ± spread. Nothing below is
   measurable without this — a 10% run-to-run spread hides any realistic improvement.
2. **Predict the anomaly, not the absolute temperature.** Have the model output the
   departure from climatology; final answer = climatology + Δ. This makes climatology the
   floor (directly fixing the sub-500 m regression) and spends all capacity on the hard
   part. ~20 lines. Highest expected value.
3. **Ensemble 3–5 seeds** — free once (1) is running.
4. **Attack the 100 m bias** — depth-weighted loss, or a penalty on the vertical gradient
   so the thermocline is placed sharply rather than smeared.
5. **Cosine LR decay + longer training.** Validation flattens ~epoch 20 while train loss
   keeps falling.
6. **M4 ConvLSTM (7-day context) — only after (1).** Added capacity has so far bought
   GLORYS fit rather than physical skill; without multi-seed you cannot tell a real M4 gain
   from noise.

Not recommended: ViT / transformers / foundation models. The M3 null result says capacity
is not the binding constraint, and 2,467 training days is far too little for them.

---

## 9. Reproducing

```bash
git clone https://github.com/swaraj-shubh/OceanEmbed.git && cd OceanEmbed
aws s3 sync s3://oceanembed-sih26-data/oceanembed/processed/nio_daily.zarr \
            data/processed/nio_daily.zarr
aws s3 cp   s3://oceanembed-sih26-data/oceanembed/processed/norm_stats.json \
            data/processed/norm_stats.json

python src/train.py configs/m2_unet.yaml
python src/evaluate.py --model climatology --split test
python src/predict_cube.py --ckpt checkpoints/m2_unet_best.pt --split test
```

Every module has a runnable `__main__` self-check (`python src/models/unet.py`,
`python src/argo_eval.py`, `python src/preprocess/regrid.py`, …). The U-Net check asserts
it can overfit a single smooth sample; the attention check asserts that perturbing one
cell moves a distant one, against an identical-input control that must be exactly zero —
otherwise attention has collapsed to a per-pixel transform and M3 is M2 with extra steps.

To rebuild the store from raw (only needed if preprocessing changes):

```bash
python src/download/cmems.py glorys
python src/preprocess/build_store.py
python -c "import sys;sys.path.append('src');import datasets;datasets.compute_stats()"
```

---

## 10. Repository map

```
src/
  config.py            frozen constants — grid, depths, splits, QC ranges
  datasets.py          Zarr contract, train-only normalisation, NIODataset
  models/unet.py       M2 UNet, M3 OceanEmbed (attention), masked_mse
  train.py             config-driven, auto-resume, best-val checkpointing
  evaluate.py          depth-wise metrics vs GLORYS
  predict_cube.py      prediction cube + Argo (B2) scoring
  argo_eval.py         profile matching and the acceptance rule
  baselines.py         M0 monthly climatology
  metrics.py           streaming DepthStats
  download/            oisst, podaac (SMAP/OSCAR), cmems, argo
  preprocess/          regrid, consolidate, build_store
configs/               one YAML per experiment
results/               metric tables (committed)
deploy/                EC2 bootstrap, S3 checkpoint sync
```

Data, checkpoints and `.env` are gitignored. Credentials live in `.env` and
`~/.copernicusmarine/` and have never been committed.
