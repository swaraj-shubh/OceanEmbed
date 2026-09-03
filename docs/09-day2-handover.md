---
title: "Day 2 — Results & Handover"
nav_order: 10
---

# 09 — Day 2: Multi-Seed, Four Interventions, and a Measured Ceiling

Day 1 ([doc 07](07-results-and-handover.html)) ended with one model, one seed, and a
headline of 0.908 °C. **Every number on that page is superseded by this one.** Day 2 added
error bars, three new interventions, a temporal architecture, and one measurement that
reframes the whole problem.

**[Doc 10](10-day3-ensemble-and-bias-correction.html) supersedes this page's headline.**
The current best model is 0.786 °C (an ensemble + bias correction, not a single
architecture), and the attention question §8 leaves open below is now closed — attention
combined with the ConvLSTM was built and tested, and it lost. Read this page for the
ceiling-finding methodology (§4) and the multi-seed discipline (§2–3), which still hold;
read doc 10 for the current numbers.

Read §1 and §4 if you read nothing else here.

---

## 1. What changed, in one table

Blended RMSE against **6,093 independent Argo profiles**, test split 2023–24, three seeds
each where marked:

| | Argo blended RMSE | vs Day 1 |
|---|---|---|
| **GLORYS12V1 — the training target itself** | **0.728** | ⭐ new — this is the ceiling |
| **M4 ConvLSTM** (3 seeds) | **0.890 ± 0.008** | ⭐ new |
| **M2 U-Net** (3 seeds) | **0.901 ± 0.013** | was "0.908", single seed |
| M3 attention | 0.907 | unchanged, now with a real error bar to compare against |
| M2 + vertical-gradient loss (3 seeds) | 0.918 ± 0.004 | ⭐ new |
| M2 anomaly formulation (3 seeds) | 0.975 ± 0.020 | ⭐ new |
| M0 monthly climatology | 1.160 | unchanged |

**Headline: 0.890 ± 0.008 °C, 23% better than climatology, against independent
observations, in years the model never saw.**

Four interventions were tried. **None of them moves the blended number by more than one
standard deviation.** That is the day's real finding, and §4 explains why.

---

## 2. Day 1 numbers that were wrong, and why

Three corrections. All were caused by measurement, not by the model changing.

**"0.908" was a single seed with no error bar.** The same config run three times gives
0.890, 0.919, 0.893 — **0.901 ± 0.013**. Any Day 1 claim smaller than ~3% was
unfalsifiable.

**"M3 improves GLORYS RMSE by 11%" was noise.** Reported on Day 1 as a real result. The
same config with the same seed produced 0.660 on one run and 0.729 on another — a ~10%
spread from nondeterministic cuDNN kernel selection and dataloader worker ordering, larger
than the effect being claimed. Corrected in doc 08.

**Every Argo number was contaminated by land cells.** A network emits *some* value on
land, and Argo matching takes the nearest grid cell, so **42 of 6,093 coastal profiles
(0.7%)** were scored against output the loss never constrained. Small for M2
(0.891 → 0.890) but not zero, and catastrophic for the anomaly model — see §5. Prediction
cubes now carry NaN on unsupervised cells.

---

## 3. Which benchmark to trust

The same three M2 runs, measured two ways:

| Metric | Spread across seeds |
|---|---|
| GLORYS validation RMSE | 0.672 → 0.729 (**8%**) |
| Argo blended RMSE | 0.890 → 0.919 (**1.4%**) |

The validation loss is 365 days of one gridded product; the Argo score is 6,093
independent casts. **Report architecture claims against Argo, never against validation
loss.** Several Day 1 conclusions came from the noisy metric.

**R² was added** and immediately earned its place. It is *not* correlation squared — it
charges for bias, and can go negative. At 100 m, correlation reads a healthy 0.797 while
R² is **0.502**, because correlation is blind to the +0.848 °C bias sitting there.

---

## 4. The finding: the ceiling is the training target, not the model

The 100 m warm bias survived every attempt to remove it. So we measured the target itself
against the same 6,093 Argo profiles.

**GLORYS12V1 carries a +0.723 °C warm bias at 100 m.**

| Depth | GLORYS vs Argo | | M2 vs Argo | |
|---|---|---|---|---|
| | RMSE | bias | RMSE | bias |
| 0 m | 0.357 | +0.022 | 0.467 | −0.061 |
| 50 m | 0.856 | +0.283 | 1.043 | +0.184 |
| 75 m | 1.206 | **+0.528** | 1.420 | +0.511 |
| **100 m** | 1.343 | **+0.723** | 1.644 | **+0.848** |
| 125 m | 1.178 | +0.615 | 1.452 | +0.790 |
| 200 m | 0.603 | +0.194 | 0.738 | +0.211 |
| 500 m | 0.226 | −0.005 | 0.289 | −0.024 |
| 1000 m | 0.243 | −0.064 | 0.267 | −0.087 |
| **blended** | **0.728** | | **0.901** | |

The model is not *creating* the thermocline bias — it is faithfully reproducing its
target's. GLORYS runs warm through the thermocline in this basin, and the network learned
that correctly.

**Consequences.**

No model trained on GLORYS can score better than **0.728 °C** against Argo. M2 is at 0.901
and M4 at 0.890, so roughly **0.16 °C of the remaining error is the model** and the rest is
the target.

It explains three failed interventions at once. You cannot sharpen, re-anchor or
re-architect your way out of a thermocline that sits in the wrong place in the training
data.

**Removing it is a data problem, not an architecture problem** — bias-correct GLORYS
against Argo before training. That is now the highest-value direction left, and it is a
genuine research contribution rather than a tuning exercise.

---

## 5. The four interventions, and what each taught

### 5.1 Anomaly formulation — 0.975 ± 0.020 (worse overall, wins in the deep)

The model predicts the departure from train-split monthly climatology; the final answer is
`climatology + Δ`. Built as a residual *inside the model path*, so loss and metrics stay in
absolute °C and remain directly comparable — training on anomalies instead would produce
anomaly-space numbers that cannot be lined up against the rest of the table.

| Depth | M0 | M2 | Anomaly |
|---|---|---|---|
| 0 m | 0.745 | **0.467** | 0.605 |
| 100 m | 2.163 | **1.644** | 1.807 |
| **500 m** | 0.273 | 0.289 | **0.252** |
| **1000 m** | 0.237 | 0.267 | **0.229** |

Worse overall, but it did exactly what it was designed for: 500 m and 1000 m are the two
levels where M2 has always lost to climatology, and **the anomaly model is the first thing
to beat climatology there**. Above 200 m it is clearly worse.

**Lesson: the right formulation is depth-dependent.** The upper ocean wants the absolute
target; the deep wants the climatology anchor. A hybrid with the crossover chosen on
validation is an untried, cheap experiment.

This run also exposed the land-cell bug in §2 — with land included it scored 1.292, which
looked absurd rather than merely bad, and chasing that discrepancy is what found the bug.

### 5.2 Vertical-gradient loss — 0.918 ± 0.004 (negative)

Penalises error in level-to-level differences — the *shape* of the profile — on the theory
that the 100 m bias was the model smearing the thermocline.

It made things slightly worse everywhere, and **the 100 m bias got worse** (+0.939 vs
+0.848) while its spread collapsed to ±0.010. Sharpening a profile cannot help when the
target's thermocline is in the wrong place; it just sharpens it in the wrong place, more
consistently. This result is what prompted the measurement in §4.

### 5.3 M3 attention — 0.907 (null, now with a proper error bar)

Day 1 could not tell whether attention helped. With M2 at 0.901 ± 0.013, M3's 0.907 sits
**0.5 σ** away. The null result holds under statistics rather than eyeball.

### 5.4 M4 ConvLSTM — 0.890 ± 0.008 (best, but not significant)

Encoder per day → ConvLSTM over 7 daily frames at the bottleneck → U-Net decoder. The
recurrence sits at the bottleneck (12×22) rather than full resolution: the question being
asked — how the last week of surface forcing set up today's subsurface — is basin-scale,
not per-pixel. Skips come from the last frame, so fine spatial detail is today's while the
bottleneck carries the week. 6.65M parameters vs M2's 1.93M.

**0.890 ± 0.008 vs M2's 0.901 ± 0.013 — a difference of 0.011 °C, or 0.80 pooled standard
deviations. Within noise.** Consistently in M4's favour and better at nearly every depth,
but three seeds cannot separate it. Reported as a non-significant improvement, not a win.

Where it looks real is **the deep**: 500 m 0.279 vs 0.289, 1000 m 0.254 vs 0.267. That is
physically sensible — the deep ocean integrates forcing over time, so a week of history
should matter more there than at the surface — and it is the same region where the anomaly
model won. Two independent experiments pointing at the same place.

---

## 6. Full depth-wise table

Three-seed means. GLORYS column is the target's own error against Argo — the floor for any
model trained on it.

| Depth (m) | M0 clim. | M2 | **M4** | GLORYS | M2 bias | M4 bias | M2 R² |
|---|---|---|---|---|---|---|---|
| 0 | 0.745 | 0.467 | 0.468 | 0.357 | −0.061 | −0.091 | 0.911 |
| 5 | 0.746 | 0.456 | 0.457 | 0.326 | −0.113 | −0.132 | 0.913 |
| 10 | 0.751 | 0.462 | **0.455** | 0.367 | −0.070 | −0.087 | 0.911 |
| 20 | 0.844 | 0.693 | **0.682** | 0.505 | +0.058 | +0.040 | 0.822 |
| 30 | 0.949 | 0.847 | 0.846 | 0.614 | +0.124 | +0.108 | 0.778 |
| 50 | 1.308 | 1.043 | **1.031** | 0.856 | +0.184 | +0.198 | 0.759 |
| 75 | 1.946 | 1.420 | **1.411** | 1.206 | +0.511 | +0.521 | 0.648 |
| 100 | 2.163 | 1.644 | **1.637** | 1.343 | +0.848 | +0.850 | **0.502** |
| 125 | 1.821 | 1.452 | **1.423** | 1.178 | +0.790 | +0.774 | 0.578 |
| 150 | 1.385 | 1.116 | **1.086** | 0.895 | +0.524 | +0.521 | 0.742 |
| 200 | 0.881 | 0.738 | **0.730** | 0.603 | +0.211 | +0.222 | 0.884 |
| 300 | 0.558 | **0.506** | 0.511 | 0.408 | +0.050 | +0.070 | 0.921 |
| 500 | **0.273** | 0.289 | 0.279 | 0.226 | −0.024 | −0.015 | 0.949 |
| 700 | **0.263** | 0.308 | 0.280 | 0.246 | −0.112 | −0.105 | 0.929 |
| 1000 | **0.237** | 0.267 | 0.254 | 0.243 | −0.087 | −0.071 | 0.916 |
| **blended** | **1.160** | **0.901** | **0.890** | **0.728** | | | |

Bold marks the best of M2/M4, or M0 where climatology still wins.

---

## 7. New bugs found on Day 2

**Argo profiles scored against land.** §2. 0.7% of profiles, invisible for M2, fatal for
the anomaly model. Found by chasing a result that was implausible rather than merely
disappointing.

**Fork-after-threads deadlock.** The first anomaly runs sat at zero epochs for twelve
minutes with four DataLoader workers asleep and load average 0.00. Fitting the climatology
runs a dask reduction that leaves a live thread pool in the parent; the DataLoader then
forks workers onto that broken lock state. The cache is now built in its own process
(`python src/datasets.py --clim`) and the dataset only ever `np.load`s it.

**A global cache path that the self-check poisoned.** The climatology cache was first
written to a fixed location, so the self-check — which builds a tiny *fake* store — wrote
its climatology where real runs would silently pick it up. Caught before any run used it.
The cache is now keyed to the store it was fitted from.

**`predict_cube` ignored the checkpoint's window.** It always built 1-day samples, so M4
received `[B, C, H, W]` where it wanted `[B, T, C, H, W]`.

---

## 8. What is left

*(§8 as originally written on Day 2, kept for history. See [doc 10](10-day3-ensemble-and-bias-correction.html) §8 for the current state of each item.)*

**1. The Streamlit demo — not built.** The largest outstanding item and an explicit PS
requirement: date + depth → map, the 7 surface inputs, click → 0–1000 m profile, nearby
Argo overlay, live metrics. Runs offline from a frozen checkpoint and precomputed samples,
so it needs no GPU. *(Written as of Day 3, not yet run end-to-end — doc 10 §8.)*

**2. Bias-correct GLORYS against Argo before training.** The only remaining direction that
attacks the 0.728 ceiling rather than the 0.16 °C above it. *(Done, differently than
proposed here: rather than bias-correcting GLORYS before training, the correction is
fit post-hoc against Argo and applied to the ensemble output — 0.890 → 0.786. See doc 10
§1–2. The monthly/spatially-varying version of this is the next open direction.)*

**3. More M4 seeds (5–7).** 0.890 ± 0.008 vs 0.901 ± 0.013 is 0.80 σ. Four more seeds
would settle whether ConvLSTM is a real improvement. *(Superseded: M4 is now the
uncontested best single architecture — see doc 10 §5, which instead answers the open
attention question with 3 seeds of attn+ConvLSTM.)*

**4. Hybrid depth formulation.** Absolute above ~300 m, anomaly below, crossover chosen on
validation. Cheap, evidence-backed by §5.1, and would make every depth beat the baseline.
*(Not done. Still open — the ensemble+bias-correction result in doc 10 achieves the same
goal, beating climatology at every depth, by a different route, but the hybrid-depth
idea itself was never tried.)*

**5. INCOIS LAS gridded Argo (track B1).** PS-named but lower value — raw Argo (B2) is the
stricter test and is done. *(Attempted on Day 3: INCOIS's OPeNDAP service was found
non-responsive for every Argo dataset. Aggregation code is written and self-tested and
needs only the file — doc 10 §8.)*

Not recommended: bigger models, ViT, foundation models. Four interventions and 0.16 °C of
headroom say capacity is not the binding constraint.

---

## 9. Reproducing Day 2

```bash
python src/datasets.py --clim                              # build the climatology cache
python src/train.py configs/m2_unet.yaml      --seed 1     # repeat for seeds 2, 3
python src/train.py configs/m4_convlstm.yaml  --seed 1
python src/train.py configs/m2_anomaly.yaml   --seed 1
python src/train.py configs/m2_grad.yaml      --seed 1
python src/predict_cube.py --ckpt checkpoints/m4_convlstm_s1_best.pt --split test
```

M2 is 24 s/epoch on a T4, M4 is 66 s/epoch. Every result above is three seeds except M3.
All metric CSVs are committed under `results/`, and checkpoints are in
`s3://oceanembed-sih26-data/oceanembed/checkpoints/`.
