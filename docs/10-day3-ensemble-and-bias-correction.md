---
title: "10 — Day 3: Ensembles, Bias Correction, and the Attention Verdict"
nav_order: 11
---

# 10 — Day 3: The Best Model, and Two Closed Questions

Doc 09 ended with a ceiling: 0.890 °C against Argo, with four interventions unable to
move it and a target (GLORYS) that is itself 0.728 off Argo. Day 3 attacks that ceiling
directly instead of tuning around it, and closes the attention question doc 09 left open.

**Read §1 and §5 if you read nothing else.**

---

## 1. The new best model: 0.786 °C

```
6× ConvLSTM models (3 seeds m4_convlstm + 3 seeds m4_dw), averaged
  -> per-depth bias correction, fitted on 2022 Argo, applied to 2023-24
  -> 0.786 degC blended RMSE, 5,980 independent Argo profiles
```

That is **32% better than climatology** and closes 65% of the gap between the old M4
(0.890) and the GLORYS-vs-Argo floor (0.728).

| Depth (m) | M0 clim. | M4 (old best) | **ens_mix6_bc** | GLORYS (floor) |
|---|---|---|---|---|
| 0 | 0.745 | 0.468 | **0.426** | 0.357 |
| 50 | 1.308 | 1.031 | **1.011** | 0.856 |
| 100 | 2.163 | 1.637 | **1.372** | 1.343 |
| 150 | 1.385 | (n/a) | **0.923** | 0.895 |
| 200 | 0.881 | 0.730 | **0.680** | 0.603 |
| 500 | 0.273 | 0.279 | **0.245** | 0.226 |
| 700 | 0.263 | 0.280 | **0.224** | 0.246 |
| 1000 | 0.237 | 0.254 | **0.216** | 0.243 |
| **blended** | 1.160 | 0.890 | **0.786** | 0.728 |

Two things worth stating in the deck:

- **It now beats climatology at every depth.** The old M4 lost below 500 m; that
  weakness is gone.
- **It beats GLORYS itself at 700 m and 1000 m** (0.224 vs 0.246; 0.216 vs 0.243). This
  is not a contradiction — the model is trained on GLORYS, but the bias correction is
  fitted against independent Argo, so at depths where GLORYS itself is biased the
  correction pulls the prediction past its own training target.

## 2. What it is made of

**Ingredient 1 — six models, not one.** `m4_convlstm` (plain ConvLSTM) seeds 1–3, plus
`m4_dw` (same architecture, inverse-variance depth-weighted loss) seeds 1–3. Averaged with
a plain mean, deliberately not `nanmean`:

```python
cube = sum(cubes) / len(cubes)   # NOT np.nanmean -- see docs/09 sec.2
```

`nanmean` would invent a value on any cell where even one of the six members happened to
be NaN, which is exactly the land-contamination bug doc 09 §2 found. Every member is NaN
on the same land mask, so a plain mean keeps land NaN, which is what the metrics need.

**Ingredient 2 — a per-depth bias correction**, fit as the mean `(prediction - observed)`
over matched 2022 Argo casts:

| Depth | Offset subtracted | Depth | Offset subtracted |
|---|---|---|---|
| 0 m | +0.003 | 150 m | +0.334 |
| 50 m | +0.444 | 200 m | +0.051 |
| 75 m | +0.494 | 300 m | −0.025 |
| **100 m** | **+0.590** | 500 m | +0.002 |
| 125 m | +0.561 | 1000 m | −0.052 |

Peaks exactly at the thermocline, near-zero at the surface and in the deep — physically
coherent, not a fitting artefact.

**Ingredient 3 — done without touching test.** Composition chosen on val (three
candidates scored, `ens_mix6` won), correction fit on val, test read once
(`phase3.sh`). `src/bias_correct.py` refuses to fit on test at the code level:

```python
assert a.split != "test", "refusing to fit an offset on the test split (CLAUDE.md rule 3)"
```

## 3. Leakage audit — all 8 checks, run live

`src/audit_leakage.py` turns every rule in CLAUDE.md sec.6 into a runnable assertion.
Run before trusting any number in this doc:

```bash
python src/audit_leakage.py
```

1. Splits ordered, non-overlapping (train ≤2021 < val 2022 < test 2023-24)
2. No sample window crosses a split boundary (checked for window=1 and window=7)
3. Normalisation stats are a train-only fit — checked *against a control*: the stats
   must reproduce the train-only mean AND differ from an all-years mean, so a stats file
   secretly fitted on everything cannot pass by coincidence
4. Climatology cache is a train-only fit
5. Argo never appears in the Zarr store (`{X, Y}` only, no Argo variable)
6. Argo used for scoring falls inside its own split window
7. **GLORYS/Argo circularity, stated precisely**: GLORYS12V1 assimilates Argo, so Argo
   is *not* statistically independent of the target in general. The defensible claim is
   narrower: the model trains on GLORYS 2015–2021 only, so no 2023–24 Argo cast — nor
   the GLORYS state it informed — was ever seen in training. Say it that way, never
   "Argo is independent of GLORYS."
8. Effective sample size is floats, not profiles — 6,448 test profiles come from only
   147 physical instruments, so quoting n=6,448 overstates statistical power by ~6.6x.
   Model comparisons use `argo_eval.paired_bootstrap`, which blocks by float.

## 4. Extra questions asked of the 0.786 result specifically

Beyond the 8 standing checks, two things specific to the bias correction were verified:

**Float overlap between the val fit and the test score.** 71 of the 86 floats used to
fit the offset also appear in the 147 test floats — so 3,411 of 5,980 test profiles come
from an instrument that also contributed to the correction. This is **not** leakage: the
offset is 15 constants (one per depth), far too low-capacity to memorise a cast, and it
is meant to capture a basin-scale bias shared across floats. But state the number rather
than let a judge find it unprompted.

**Where the 468 dropped profiles went.** 6,448 test profiles → 5,980 scored:

| | Count |
|---|---|
| Outside the 96×176 model grid | 354 |
| No prediction within ±1 day (first 6 days of the split; the 7-day window can't start before it) | 37 |
| Rejected by the depth-acceptance rule (no measurement within `max(0.1z, 10 m)`) | 77 |
| **Scored** | **5,980** |

## 5. The attention question — now closed, with the wrong-metric trap caught in the act

Doc 09 left one cell of the ablation grid untested: attention *combined with* the
ConvLSTM, rather than either alone. `configs/m4_attn.yaml` = `m4_convlstm` + `attn: true`
(attention runs once per day, inside the encoder, before the recurrence). 3 seeds, 20
epochs, 87 s/epoch (32% slower than plain M4 — attention runs 7x per sample at this
window length, so it stops being hidden behind data loading).

**On validation RMSE, attention looked like a clean win:**

| | Val RMSE (best epoch) |
|---|---|
| m4_attn s1/s2/s3 | 0.6507 / 0.6493 / 0.6492 |
| m4_convlstm s1/s2/s3 | 0.6617 / 0.6551 / 0.6696 |

Every attention seed beat every baseline seed. If Day 3 had stopped there, the
conclusion would have been "attention helps, ship it."

**On Argo — the metric that counts — it is worse, outside the noise band:**

| Model | Argo blended RMSE |
|---|---|
| m4_attn (3 seeds) | **0.917 ± 0.008** |
| m4_convlstm (3 seeds) | **0.890 ± 0.010** |

A 3% gap, larger than either model's own seed spread. This is the exact scenario
doc 09 §3's benchmark rule exists to prevent: GLORYS validation loss swings ~8% between
seeds and Argo swings ~1.4%, and here the two metrics don't just disagree on magnitude,
they disagree on **direction**. Attention fit the training distribution slightly better
and generalised to independent observations worse.

**The full ablation grid is now complete:**

| Combination | Argo RMSE |
|---|---|
| encoder → decoder (M2) | 0.901 ± 0.013 |
| encoder → attention → decoder (M3) | 0.907 |
| encoder → ConvLSTM → decoder (M4) | **0.890 ± 0.010** ✅ |
| encoder → attention → ConvLSTM → decoder | 0.917 ± 0.008 |

Attention loses in both places it was tried, once inside the noise band (M3) and once
clearly outside it (M4+attn). **Final architecture: CNN encoder → ConvLSTM → U-Net
decoder. No attention.**

## 6. Extra input channels — tried, also negative

Three additional input representations were built and tested, extending doc 09 §5's
"four interventions moved nothing" finding to seven:

| Extra channels | Val RMSE | vs m4_convlstm baseline (0.860 ± 0.004) |
|---|---|---|
| Monthly climatology (15 ch, `extra=("clim",)`) | 0.899 | worse |
| Bathymetry + lat/lon + day-of-year sin/cos (12 ch, `extra=("aux",)`) | 0.889 | worse |
| Both combined (27 ch) | 0.889 | worse |

Single seed each — not airtight — but the gap is ~7x the baseline's own seed spread, so
this reads as a real loss, not noise. Consistent with the doc 09 finding: **the seven
satellite surface fields already carry the available signal; the binding constraint is
target quality, not missing inputs.**

## 7. Full ablation table (test split, all runs)

```bash
python src/ablation.py --split test --out results/ablation_test.md
```

| run | seeds | blended | rmse@0m | rmse@50m | rmse@100m | rmse@200m | rmse@500m | rmse@1000m | bias@100m | corr@100m |
|---|---|---|---|---|---|---|---|---|---|---|
| GLORYS_target | 1 | 0.728 | 0.357 | 0.856 | 1.343 | 0.603 | 0.226 | 0.243 | 0.723 | 0.874 |
| **ens_mix6_bc** | 1 | **0.786** | 0.426 | 1.011 | 1.372 | 0.680 | 0.245 | 0.216 | 0.299 | 0.819 |
| ens_mix6 | 1 | 0.859 | 0.417 | 0.999 | 1.609 | 0.693 | 0.244 | 0.226 | 0.893 | 0.819 |
| ens_base | 1 | 0.862 | 0.440 | 1.005 | 1.595 | 0.700 | 0.255 | 0.238 | 0.850 | 0.815 |
| ens_dw | 1 | 0.869 | 0.406 | 1.004 | 1.641 | 0.698 | 0.242 | 0.223 | 0.937 | 0.816 |
| m4_convlstm | 3 | 0.890 ± 0.010 | 0.468 | 1.031 | 1.637 | 0.730 | 0.279 | 0.254 | 0.850 | 0.800 |
| m4_dw | 3 | 0.892 ± 0.022 | 0.434 | 1.030 | 1.669 | 0.726 | 0.257 | 0.234 | 0.937 | 0.805 |
| m2_unet | 4 | 0.903 ± 0.014 | 0.476 | 1.040 | 1.643 | 0.737 | 0.293 | 0.268 | 0.853 | 0.798 |
| m3_oceanembed (attn only) | 1 | 0.907 | 0.514 | 1.031 | 1.662 | 0.708 | 0.301 | 0.287 | 0.920 | 0.804 |
| **m4_attn** (attn + ConvLSTM) | 3 | **0.917 ± 0.008** | 0.472 | 1.046 | 1.708 | 0.736 | 0.280 | 0.261 | 0.999 | 0.804 |
| m2_grad | 3 | 0.918 ± 0.005 | 0.492 | 1.049 | 1.690 | 0.739 | 0.312 | 0.286 | 0.939 | 0.798 |
| m2_anomaly | 3 | 0.975 ± 0.024 | 0.605 | 1.035 | 1.807 | 0.799 | 0.252 | 0.229 | 0.984 | 0.766 |
| M0_climatology | 1 | 1.160 | 0.745 | 1.308 | 2.163 | 0.881 | 0.273 | 0.237 | 0.811 | 0.514 |

## 8. What is left

**1. Streamlit demo.** Written (`app/streamlit_app.py`), not yet run end-to-end against
a live checkpoint + store. Point it at `ens_mix6_bc`, not a single M4 checkpoint, since
that is now the frozen best model. Needs a saved `_bc` cube or the offset applied at
inference time — `predict_cube.py`'s `--offset` flag does this.

**2. INCOIS LAS Gridded ARGO (track B1).** PS-named product. Their OPeNDAP service was
found non-responsive during this work (every Argo dataset times out with zero bytes
after 4+ minutes, while non-existent paths on the same server 404 in under a second —
their Ferret backend for this category is down, not a network or auth issue on our
side). The aggregation code is written and self-tested (`src/incois_eval.py`, 7 checks:
footprint-mean spatial aggregation, end-stamped 10-day windows, depth intersection only,
NaN-safety, irregular-axis rejection). It needs only the file. B2 (this doc's entire
result set) remains the primary, stricter track in the meantime.

**3. Monthly or spatially-varying bias correction.** The current offset is one flat
constant per depth for the whole basin and the whole year. `bias_correct.py` already
supports `--by-month` and was never used with it; a basin split (Arabian Sea vs Bay of
Bengal) was never tried either. This is the only lever that has produced a large gain so
far (0.890 → 0.786) and is the highest-value remaining direction.

**4. More `ens_mix6_bc`-style seeds**, if time allows — the current result is a single
ensemble build, not an averaged-and-spread one like the individual models are.

Not recommended: bigger models, more input channels, attention. Seven interventions
(four from doc 09, three from §5–6 of this doc) have now moved the blended number by
less than one architecture-level effect each; the constraint is target quality, not
capacity or missing signal.

## 9. Reproducing Day 3

```bash
python src/train.py configs/m4_dw.yaml   --seed 1   # repeat for seeds 2, 3
python src/train.py configs/m4_attn.yaml --seed 1   # repeat for seeds 2, 3

bash phase3.sh   # ensemble composition on val, offset fit on val, test cubes built

python src/audit_leakage.py                          # all 8 checks must pass
python src/ablation.py --split test --out results/ablation_test.md
```

Checkpoints and result CSVs are mirrored to
`s3://oceanembed-sih26-data/oceanembed/{checkpoints,results}/`. `src/`, the configs, and
the phase scripts were **not** covered by that sync and existed only on the training
instance until this doc was written — `deploy/sync_checkpoints.sh` now also syncs `src/`
and `configs/` for exactly this reason.
