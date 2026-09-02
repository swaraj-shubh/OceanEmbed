---
title: "Day 3 — Handover"
nav_order: 12
---

# 11 — Day 3: The Ceiling Moved

Day 2 ([doc 09](09-day2-handover.html)) ended with M4 ConvLSTM at 0.890 °C, four failed
interventions, and one measurement that reframed the problem: **the training target itself
carries a +0.723 °C warm bias at 100 m against Argo.** It concluded that the remaining
levers were on the data side, not the model side, and named bias-correcting the target as
the highest-value direction left.

Day 3 took that seriously and the number moved from **0.890 to 0.786 °C**. Almost none of
that came from training anything.

Read §1, §3 and §4 if you read nothing else.

---

## 1. What changed, in one table

Blended RMSE against independent Argo casts, test split 2023–24:

| | Argo blended RMSE | |
|---|---|---|
| **FINAL — 6-member ensemble + Argo bias correction** | **0.786** | ⭐ new |
| GLORYS12V1 target itself | 0.728 | ceiling for an *uncorrected* model — see §3 |
| M4 3-seed ensemble + correction | 0.792 | ⭐ new |
| M2+M3 ensemble + correction | 0.818 | ⭐ new, from the two Aug-31 checkpoints alone |
| M2 + correction, depth-only | 0.844 | ⭐ new |
| M2 + correction, depth × month | 0.850 | ⭐ new, worse — as predicted out-of-sample |
| 6-member ensemble, uncorrected | 0.859 | ⭐ new |
| M4 3-seed ensemble | 0.862 | ⭐ new |
| M4 ConvLSTM, 3 seeds | 0.890 | Day 2's headline |
| M2 U-Net, 4 seeds | 0.902 ± 0.014 | was 0.901 ± 0.013 at 3 seeds |
| M3 attention | 0.907 | unchanged |
| M0 climatology | 1.160 | unchanged |

**Headline: 0.786 °C — 32% better than climatology, better than climatology at every one
of the 15 depths, against ~6,000 independent Argo casts in years the model never saw.**

The final system is three seeds of M4 ConvLSTM plus three seeds of the same network trained
with a depth-weighted loss, averaged, minus a fifteen-number depth-wise offset fitted on
2022 validation Argo. Manifest: `results/FROZEN.md`.

**Six more interventions were screened. One was promoted, and only as an ensemble member.**
That brings the running total to seven model-side interventions, none of which improved the
observational score on its own, against two output-side steps that both did.

---

## 2. Day 2 claims that need correcting

**"No model trained on GLORYS can score better than 0.728 °C."** Wrong as stated, and the
correction is the most interesting result of the day. See §3.

**"Checkpoints are in `s3://oceanembed-sih26-data/oceanembed/checkpoints/`."** Only four
were — `m2_unet` and `m3_oceanembed`, dated Aug 31. Every Day 2 checkpoint (all seeds of
M2, M4, anomaly and gradient) was missing, and Day 3 opened by planning a retrain. They
turned out to be **alive on the running g4dn.xlarge, unsynced**: `sync_checkpoints.sh`
existed but had to be started by hand in a second shell, and nobody had. All 28 recovered
and mirrored. `deploy/setup.sh` now starts the sync itself, so the only way to train
without one is to not use the script.

**"Bias-correct GLORYS against Argo before training."** Right diagnosis, unnecessarily
expensive prescription. Correcting the *output* captures most of the same gain for no GPU
time at all, and can be re-fitted in seconds when the bias drifts. Retraining on a
corrected target is still untried and is now a lower priority, not a higher one.

**"More M4 seeds (5–7) would settle whether ConvLSTM is a real improvement."** Superseded
by a better instrument. The problem was never the seed count — it was that **every error
bar in this project had been computed as though the 6,448 test casts were independent
samples. They come from 147 floats.** Two casts from one float ten days apart in the same
water mass are not independent, so a profile-level bootstrap reports intervals roughly
√(6448/147) ≈ 6.6× too narrow. Blocking by float instead settles the question directly:
M3 attention − M2 is **−0.0009 with a 95% interval of [−0.0110, +0.0105]**. The null result
holds, now on a proper footing.

**Selection discipline changed.** Day 2 scored every intervention on the test split, which
quietly makes its M4-vs-M2 comparison test-selected. Day 3 screens on **val (2022 Argo)**
and reads test once, on the frozen winner.

---

## 3. The finding: the ceiling bounds an *uncorrected* model

Day 2 measured GLORYS12V1's own error against Argo at 0.728 °C and concluded no model
trained on it could do better. That is true only while the model inherits the target's
bias. Once a correction fitted on *independent observations* is applied, the inheritance
stops — and at the depths where GLORYS' bias dominates GLORYS' error, we pass it:

| Depth (m) | **FINAL** | GLORYS | |
|---|---|---|---|
| 125 | **1.158** | 1.178 | model wins |
| 700 | **0.224** | 0.246 | model wins |
| 1000 | **0.216** | 0.243 | model wins |

Blended, the model is still behind (0.786 vs 0.728) because GLORYS is far ahead in the
upper 50 m, where its bias is small and its resolution advantage is real. But the framing
must change. **Quote the ceiling as "the bound for an uncorrected model", not as a law.**
Stated the old way, a judge who notices the 125 m row has caught us in a contradiction;
stated correctly, the same row is the strongest evidence the correction does what we claim.

The gap the model adds on top of its target fell from **0.162 °C to 0.058 °C — a 64%
reduction.**

### Every depth now beats climatology

Day 2's standing weakness was losing to a monthly climatology at 500, 700 and 1000 m — the
one honest hole in the story. It is closed.

| Depth (m) | M0 clim. | M4 | **FINAL** |
|---|---|---|---|
| 500 | 0.273 | 0.279 | **0.245** |
| 700 | 0.263 | 0.280 | **0.224** |
| 1000 | 0.237 | 0.254 | **0.216** |

**15 of 15.**

---

## 4. What actually worked, and what it cost

| Step | Gain | Cost |
|---|---|---|
| Ensembling six existing checkpoints | −3.5% | zero — an array mean |
| Depth-wise Argo bias correction | −8.5% | zero — fifteen numbers |
| Six training experiments (screened) | one promoted, as an ensemble member | ~3.5 GPU-hours |

Both output-side steps are significant under the float-blocked bootstrap:

| Comparison | Δ blended | 95% CI | |
|---|---|---|---|
| M3 attention − M2 | −0.0009 | [−0.0110, +0.0105] | not significant |
| 6-member ensemble − M4 single seed | −0.0354 | [−0.0427, −0.0278] | **significant** |
| correction, on top of the ensemble | −0.0730 | [−0.0863, −0.0601] | **significant** |
| **FINAL − M4 single seed** | **−0.1084** | **[−0.1208, −0.0972]** | **significant** |

### The bias drifts, and it costs us

Fitted on 2022 val, the final ensemble's offset at 100 m is **+0.590 °C**. Its actual
2023–24 bias is **+0.893**. The correction under-shoots, which is why the model gains ~8%
where a GLORYS-only probe gained 9.3%. An operational version must refit annually against
the most recent Argo. This is a limitation to state, not to hide — it is also evidence the
correction is doing physical work rather than fitting noise.

Depth × month scored 0.850 against depth-only's 0.844. 180 bins over ~3,400 casts overfits.
The form was chosen on a prior out-of-sample GLORYS probe (0.671 vs 0.675), not on an
in-sample val comparison that would have flattered the more flexible model.

---

## 5. The six screens, and what each taught

Screened at one seed against an M4 val baseline of **0.860 ± 0.004**. Promote at ≤ 0.870,
reject above 0.880.

| Config | Val blended | vs baseline | Decision |
|---|---|---|---|
| **M4 depth-weighted loss** | 0.865 → **0.854 ± 0.010** at 3 seeds | +0.005 | **promoted** |
| M4 gradient loss | 0.875 | +0.015 | rejected — no depth-wise win |
| M4 + auxiliary channels | 0.889 | +0.029 | rejected |
| M4 + climatology + aux | 0.889 | +0.029 | rejected |
| M4 + climatology channels | 0.899 | +0.039 | rejected |
| M4 anomaly | 0.928 | +0.068 | rejected |

### 5.1 Depth-weighted loss — promoted, and the prediction was written down first

[Doc 10](10-experiment-programme.html) Task 7 predicted this in writing *before the run*:
the blended score is an n-weighted RMS across depths with roughly equal n per level, so
**plain masked MSE already is the loss matched to the reported metric**, and inverse-variance
weighting is a trade rather than a win — expect blended slightly worse, 500–1000 m better.

Measured at one seed: 50 m **1.142 vs 1.040** (worse), 500 m **0.292 vs 0.347** and 1000 m
**0.234 vs 0.257** (better). Exactly the predicted shape.

It is not a better model, it is a *complementary* one. That is why it earns an ensemble slot
instead of replacing anything — and why the six-member mix (0.823 on val) beats either
family alone (0.829 depth-weighted, 0.831 baseline).

Note also that doc 05 had proposed weighting *toward* the thermocline. That reasoning was
backwards and would have made the known-weak deep levels worse still. Corrected in place.

### 5.2 Auxiliary channels — the cleanest lesson of the day

Bathymetry + day-of-year sin/cos + lat/lon. Standard in this literature, and free:
bathymetry comes from the store's own NaN structure below the sea floor, needing no GEBCO
download and no regrid.

**`m4_aux` produced the best GLORYS validation RMSE of the three channel experiments —
0.659, better than the baseline typically manages — and scored *worse* against Argo (0.889
vs 0.860).** Fitting the reanalysis better made agreement with observations worse.

That is doc 09 §3's benchmark rule demonstrated live rather than argued, and it is the
single most useful slide we have for explaining why we report against Argo.

### 5.3 Climatology as input — a good idea that fails for a knowable reason

Doc 09 §5.1 found the right formulation is depth-dependent and proposed a hand-picked
crossover; feeding climatology as 15 input channels lets the network learn it instead.
[TS-Cast (Ocean Science, 2026)](https://os.copernicus.org/articles/22/2161/2026/) does
exactly this. It scored 0.899 — clearly worse.

The likely mechanism is the same as §5.2: handing the model the climatology lets it lean
harder on GLORYS' climatological bias structure, so it reproduces the target's errors more
faithfully. In a regime where the target is the binding constraint, that is the wrong
direction. Combining it with aux channels (0.889) was no better than aux alone.

### 5.4 Anomaly and gradient loss — both re-confirmed negative on M4

Anomaly 0.928, gradient 0.875, against 0.860. Day 2 measured both as negative on M2; they
are negative on M4 too. Do not revisit.

---

## 6. Full depth-wise table

| Depth (m) | M0 clim. | M4 | **FINAL** | GLORYS | MAE | Bias | Corr | R² |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.745 | 0.468 | **0.426** | 0.357 | 0.292 | −0.108 | 0.965 | 0.925 |
| 5 | 0.746 | 0.457 | **0.405** | 0.326 | 0.287 | −0.119 | 0.968 | 0.931 |
| 10 | 0.751 | 0.455 | **0.425** | 0.367 | 0.290 | −0.122 | 0.964 | 0.924 |
| 20 | 0.844 | 0.682 | **0.656** | 0.505 | 0.408 | −0.092 | 0.918 | 0.839 |
| 30 | 0.949 | 0.846 | **0.810** | 0.614 | 0.560 | −0.151 | 0.897 | 0.797 |
| 50 | 1.308 | 1.031 | **1.011** | 0.856 | 0.772 | −0.253 | 0.889 | 0.774 |
| 75 | 1.946 | 1.411 | **1.274** | 1.206 | 0.973 | +0.051 | 0.850 | 0.717 |
| 100 | 2.163 | 1.637 | **1.372** | 1.343 | 1.056 | +0.299 | 0.819 | 0.653 |
| 125 | 1.821 | 1.423 | **1.158** | 1.178 | 0.883 | +0.241 | 0.863 | 0.732 |
| 150 | 1.385 | 1.086 | **0.923** | 0.895 | 0.713 | +0.201 | 0.913 | 0.824 |
| 200 | 0.881 | 0.730 | **0.680** | 0.603 | 0.523 | +0.176 | 0.953 | 0.902 |
| 300 | 0.558 | 0.511 | **0.482** | 0.408 | 0.339 | +0.097 | 0.965 | 0.928 |
| 500 | 0.273 | 0.279 | **0.245** | 0.226 | 0.185 | −0.018 | 0.982 | 0.963 |
| 700 | 0.263 | 0.280 | **0.224** | 0.246 | 0.172 | −0.022 | 0.981 | 0.963 |
| 1000 | 0.237 | 0.254 | **0.216** | 0.243 | 0.165 | −0.014 | 0.973 | 0.945 |
| **blended** | **1.160** | **0.890** | **0.786** | **0.728** | | | | |

At 100 m, the hardest layer: R² goes 0.502 → **0.653**, bias +0.850 → **+0.299**.

---

## 7. What was built

Six new or substantially changed modules. Every one carries an `if __name__ == "__main__"`
assert block, per CLAUDE.md §13 — no test framework.

| File | What it does |
|---|---|
| `src/bias_correct.py` | **new.** Fit a depth (or depth × month) offset from a cube + Argo; apply it. Refuses to fit on the test split; `predict_cube --offset` refuses to apply an offset fitted on the split being scored. Two guards, both in code rather than in discipline. |
| `src/audit_leakage.py` | **new.** CLAUDE.md §6 as eight runnable assertions, including a control that catches normalisation stats fitted on all years rather than train only. Passes 8/8. |
| `src/ablation.py` | **new.** Builds the ablation table from every `results/*_argo.csv`, averaging seeds and reporting spread. The table is regenerated, never hand-maintained. |
| `src/argo_eval.py` | `match_profiles()` extracted (returns matched predictions, observations, **float ids** and times); `paired_bootstrap()` added — float-blocked, paired, 1,000 resamples. |
| `src/datasets.py` | `extra=` channel sets — `clim` (15 monthly climatology channels) and `aux` (day-of-year sin/cos, lat, lon, bathymetry). `build_bathymetry()` derives a shelf map from the store's own NaN structure: no download, no regrid. |
| `src/models/unet.py`, `src/train.py` | `depth_weight` on the loss; `in_ch` now **derived** from the channel set and stored in the checkpoint rather than hand-typed in YAML. |
| `src/predict_cube.py` | `--ensemble` (aligns members on the intersection of their time axes), `--offset`, and `score_cube()` extracted for reuse. |

Plus six `configs/m4_*.yaml`, and repo hygiene: `.env.example`, `LICENSE`, `.gitattributes`
(a CRLF shell script fails on the GPU box with an unreadable error — that cost time twice),
a CI workflow running the seven self-checks that build their own synthetic data, and a
rewritten README.

---

## 8. Bugs found on Day 3

**The anomaly residual base leaked into climatology-as-input mode.** `__getitem__` keyed the
residual base off `self.clim is not None`, which became true for `extra=("clim",)` as well.
Tasks 4 and 5 would silently have been the same experiment. Caught by a self-check
assertion written before the run, not by a confusing result afterwards.

**`--ensemble` tried to overwrite a cube it had open for reading.** The input path was
relative and the output path absolute, so the guard comparing them never matched.

**Ensembling a window=7 model with a window=1 model.** A 7-day model cannot predict a
split's first six days, so its cube is six frames shorter. The strict shape assertion caught
it; the fix aligns members on the intersection of their time axes and reports how many days
were dropped, rather than silently averaging misaligned days.

**A `.gitignore` trailing comment staged 1.5 GB of prediction cubes.** `results/*.nc  # …`
— `.gitignore` has no trailing comments, so the pattern became the whole line and matched
nothing. Caught on the status check before committing.

**And the one that wasn't a bug:** Day 2's checkpoints were never lost. See §2.

---

## 9. What is left

**1. The Streamlit demo — still not built.** The largest outstanding item and an explicit PS
requirement. Design agreed in [doc 06](06-demo-and-roadmap.html): a committed ~68 MB bundle
(int16-packed, 0.0002 °C round-trip error) covering **2023-10-01 → 2023-12-31** — a window
chosen because it contains Cyclone Tej in the Arabian Sea and Cyclone Michaung in the Bay of
Bengal, both inside the test split. Predictions are precomputed, so the app needs neither
torch nor cartopy, which makes it deployable to Streamlit Cloud as well as running offline.

**2. Track B1 — INCOIS LAS Gridded ARGO.** PS-named, lower value: raw Argo (B2) is the
stricter test and is done.

**3. Refit the offset annually** if this ever runs operationally. §4.

**Not recommended:** more capacity, more input channels, ViT, foundation models. Seven
interventions now say the same thing. Retraining on a bias-corrected target remains the one
untried model-side idea worth keeping on the list, and it is a *lower* priority than it was
on Day 2, because the post-hoc version already captured most of the gain.

---

## 10. Reproducing Day 3

```bash
python src/audit_leakage.py                                     # 8/8 before anything else
python src/datasets.py --clim                                   # climatology + bathymetry caches

# screens (GPU, ~3.5 h total)
for c in m4_anomaly m4_clim m4_aux m4_clim_aux m4_dw m4_grad; do
    python src/train.py configs/$c.yaml --seed 1
    python src/predict_cube.py --ckpt checkpoints/${c}_s1_best.pt --split val
done
for s in 2 3; do python src/train.py configs/m4_dw.yaml --seed $s; done

# ensemble chosen on val, uncorrected; then fit the offset on the winner's val cube
python src/predict_cube.py --split val --run ens_mix6 --ensemble \
    results/m4_convlstm_s{1,2,3}_best_val_cube.nc results/m4_dw_s{1,2,3}_best_val_cube.nc
python src/bias_correct.py --cube results/ens_mix6_val_cube.nc --split val

# the single test read
python src/predict_cube.py --split test --run ens_mix6 --ensemble \
    results/m4_convlstm_s{1,2,3}_best_test_cube.nc results/m4_dw_s{1,2,3}_best_test_cube.nc \
    --offset results/ens_mix6_offset.json

python src/ablation.py --split test
```

M4 is 65–90 s/epoch on a T4. All metric CSVs are committed under `results/`; checkpoints are
in `s3://oceanembed-sih26-data/oceanembed/checkpoints/` — **and this time the sync that put
them there starts itself.**
