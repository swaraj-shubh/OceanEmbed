---
title: "Day 4 — Audit"
nav_order: 13
---

# 12 — Day 4: Three Directions Closed

Day 3 ([doc 11](11-day3-handover.html)) ended at **0.786 °C** with two output-side steps
that worked and seven model-side interventions that did not, and named three things left:
the demo (built), track B1 (not built), and an annual refit of the offset.

**Day 4 produced no improvement to that number, and that is the finding.** Four
measurements were run against the frozen system. Three of them close directions that would
otherwise have consumed a week each, and the fourth makes training about ten times cheaper
without making it better.

The headline is unchanged: **0.786 °C, 6-member ensemble + depth-wise Argo offset.**

Read §2 and §3 if you read nothing else.

---

## 1. What was measured

| | Result | What it means |
|---|---|---|
| Six alternative correction forms | all null | the correction is finished at fifteen numbers |
| Per-depth ensemble weights | −0.1%, null | the flat mean already has the diversity |
| Error budget by depth | **62% in 75–150 m**, and it is **variance, not bias** | post-hoc correction cannot reach what is left |
| Training hygiene (intervention #8) | GLORYS val −3.2%, Argo **not significant** | take it for its cost, not its score |

Nothing here changes `results/FROZEN.md`. Two of the four are new *rules* — things not to
try — and the repo now carries the instrument that produced them rather than the claim.

---

## 2. Six correction forms, and why they are all null

The shipped correction is fifteen numbers, one mean residual per depth. The obvious next
move is to condition it on something the residual might depend on: which basin, which
season, which latitude, or the prediction itself.

**The measurement has to be cross-validated or it is worthless.** Fitted in-sample, a form
with more bins always scores better on the data it was fitted to, so an in-sample val
comparison selects the most flexible form regardless of truth. Doc 11 §4 avoided that trap
by hand — choosing depth-only over depth × month on a *prior out-of-sample probe* rather
than on the in-sample comparison that would have flattered the bigger model. Day 4 makes
that discipline mechanical: `src/correction_forms.py` runs repeated half-splits over val
**floats**, fits on half and scores the other half.

Blocking by float, not by cast, for the same reason `argo_eval.paired_bootstrap` blocks by
float: 3,107 val casts come from 83 floats. A cast-level split puts casts from the same
float on both sides, so a form that has merely memorised that float's water mass scores
well "out of sample" and looks like it generalises.

Held-out blended RMSE, 40 × 2 folds, on the six-member val cube:

| Correction form | Held-out blended | vs shipped | |
|---|---|---|---|
| depth × latitude band | 0.7657 ± 0.0180 | −0.6% | inside fold noise |
| **depth only — shipped** | **0.7704 ± 0.0191** | — | **keep** |
| depth × basin (AS / BoB) | 0.7705 ± 0.0188 | +0.0% | null |
| depth, linear a + b·pred | 0.7711 ± 0.0202 | +0.1% | null |
| depth × season (monsoon) | 0.7715 ± 0.0193 | +0.1% | null |
| depth × basin × season | 0.7719 ± 0.0187 | +0.2% | null |
| no correction | 0.8230 ± 0.0156 | +6.8% | the gain being kept |

Every alternative sits within ±0.6% of the shipped form against **±1.9% fold-to-fold
noise**. The latitude-band form is nominally best and is not distinguishable from the
others; promoting it on a −0.6% reading against ±1.9% noise would be exactly the mistake
this project has spent three days learning not to make.

**Fifteen numbers is the right answer, and it already was.** Do not stratify further.

### Per-depth ensemble weights, same verdict

The six members come from two families whose strengths doc 11 §5.1 showed are
complementary. Fitting one blending weight per depth instead of a flat average recovers
exactly that shape — the ConvLSTM family favoured near the surface, the depth-weighted
family favoured below 500 m, α running 0.61 → 0.83 → 0.33 → 0.02 down the column — and
moves the held-out score by **−0.1%**. The plain mean is already collecting the diversity.
Do not optimise the weights.

---

## 3. The finding: what is left is variance, not bias

Run the depth-wise correction in-sample on the val cube and the bias at *every* level goes
to exactly zero, by construction. Then look at what is still there:

| Depth (m) | RMSE after zeroing bias | Share of squared error (test) |
|---|---|---|
| 0 | 0.408 | 2.0% |
| 30 | 0.831 | 7.1% |
| 50 | 0.914 | 11.1% |
| **75** | **1.137** | **17.7%** |
| **100** | **1.287** | **20.5%** |
| **125** | **1.163** | **14.6%** |
| **150** | **0.936** | **9.3%** |
| 300 | 0.619 | 2.5% |
| 1000 | 0.223 | 0.5% |

**Four of fifteen levels carry 62% of the squared error, and 100 m still scores 1.287 °C
with its bias removed.**

That single number closes the whole post-hoc route. The residual in the thermocline is not
an offset any lookup table can subtract — it is the model putting the thermocline in the
wrong place, or making it too gradual, on individual days. §2's six null results are not a
run of bad luck; they are what that fact predicts.

It also sets the exchange rate for every future experiment: **an improvement at 75–150 m is
worth roughly thirty times the same improvement at 1000 m.** Screen on the thermocline
band, not only on the blended number.

---

## 4. Intervention #8: the models were undertrained

Every one of the six final ensemble members reached its best val epoch at **14–19 of a
20-epoch budget**; two peaked on the very last epoch. None had plateaued and then
overfitted, which is the shape a converged run makes. Two causes are visible in the loss
curves:

1. **The head starts at zero and the target is raw °C.** `m4_convlstm_s1` opens at train
   loss 434 °C², i.e. RMSE 20.8 — exactly "predict zero" against a 15–29 °C ocean. The
   first five epochs buy nothing but a constant.
2. **A constant learning rate of 1e-3.** Val RMSE bounced ±0.02 across the last five
   epochs, so "best epoch" was partly a draw from that noise.

Both are one line. Seed the head bias with the train-split per-depth means; decay the
learning rate on a cosine. Both are behind config keys and **default off**, so every
committed result reproduces unchanged.

| Run (seed 1) | Opening train loss | Best GLORYS val | Reached at | Argo val |
|---|---|---|---|---|
| `m4_convlstm` (as shipped) | 434.1 | 0.6617 | epoch 19 | 0.8607 |
| **`m4_sched`** (bias init + cosine, 40 ep) | **0.76** | **0.6405** | **epoch 2** ≈ old best | 0.8550 |

**Epoch 2 of the fixed recipe beats twenty epochs of the old one.** Best GLORYS validation
improves 3.2%, and the run converges properly — best at epoch 17 of 40, then a plateau.

### And against Argo it is a null

    m4_sched_s1 - m4_convlstm_s1 (val Argo):  -0.0060   95% CI [-0.0316, +0.0207]

Not significant under the project's own float-blocked paired bootstrap. A real 3.2%
improvement in fitting GLORYS bought essentially nothing observationally.

This is the same lesson `m4_aux` taught in doc 11 §5.2, arriving from the opposite
direction and on a model that is now genuinely converged rather than merely stopped. There
it was "fit GLORYS better, score worse"; here it is "fit GLORYS better, score the same".
Both say the target, not the optimiser and not the architecture, is the binding constraint.

**Adopt it for its cost.** It reaches shipped-model quality roughly ten times cheaper, so
the six-member ensemble can be rebuilt in a third of the GPU time, or three times the seeds
bought for the same money — and more seeds is the one thing that has reliably moved the
number.

---

## 5. Day 3 statements that need updating

**"Refit the offset annually."** Still right, and now known to be the *only* remaining
headroom in the correction. §2 shows no form change helps, so the entire residual gain in
that route is drift — fitted +0.590 at 100 m on 2022, actual +0.893 in 2023–24.

**"Retraining on a bias-corrected target remains the one untried model-side idea worth
keeping on the list, and it is a lower priority than it was."** Raise it back. Day 3
downgraded it because the post-hoc version was cheaper; §2 and §3 have now closed every
alternative around it, which makes it the highest-value untried lever rather than a
redundant one.

**"Do not add capacity. Do not add input channels. Work on the output side."** Refine it.
"Output side" was read as post-processing, and post-processing is now measured out. The
live reading is **change what the network is asked to emit** — see §7 item 06.

**The M3 attention null.** Worth one caveat now that the loss curves have been read
properly: M3's val curve was still descending at epoch 29 when training stopped, so it was
scored while less converged than M2. The bootstrap interval [−0.011, +0.011] almost
certainly still holds — and §4 shows convergence buys nothing observationally anyway — but
it was not a fair fight, and if anyone re-runs M3 it should be under the `m4_sched` recipe.

---

## 6. Where this sits in the literature

Published RMSEs in this field are almost never comparable, and the reason matters for the
viva. [DORS 0.25°](https://www.sciencedirect.com/science/article/abs/pii/S0924271624003617)
reports 0.579 °C with R² 0.980 — but over **0–2000 m**. Half of that range is nearly
errorless water, and averaging over it dilutes the thermocline heavily. This project's
fifteen levels are packed into 0–1000 m and are thermocline-dominated by construction,
which makes 0.786 a *harder* number than it looks.

| Restricted to | FINAL | GLORYS | M0 clim. |
|---|---|---|---|
| all 15 levels, 0–1000 m | **0.786** | 0.728 | 1.160 |
| 13 levels ≤ 500 m | 0.837 | 0.774 | 1.237 |
| 12 levels ≤ 300 m | 0.868 | 0.802 | 1.285 |
| the six PS report depths | 0.788 | 0.730 | 1.155 |

**Always quote the depth range with the number.** A reviewer who assumes 0–2000 m reads
0.786 as mediocre; one who knows it is fifteen thermocline-weighted levels reads it
correctly. [TS-Cast](https://os.copernicus.org/articles/22/2161/2026/) (Ocean Science,
2026) reports under 1 °C in the upper 500 m at the Kuroshio Extension; on the same
restriction this model scores **0.837**, in a basin whose barrier layer makes the
surface-to-subsurface link weaker than the Kuroshio's.

### Three papers to read before the next experiment

- [**NeSPReSO**](https://www.sciencedirect.com/science/article/abs/pii/S1463500325000538)
  (Ocean Modelling, 2025) — runs PCA over Argo profiles and has the network predict the
  **principal components**, not the levels, with absolute dynamic topography among its
  inputs. Beats GEM, MLR and ISOP. This is the output parametrisation we have not tried,
  and it targets profile *shape*, which §3 says is the whole remaining problem.
- [**Adaptive spatiotemporal clustering**](https://arxiv.org/abs/2605.00860) (2026) —
  clusters the water column into thermodynamically coherent depth bands and trains per
  band, reporting 12.4–27.2% RMSE reductions across six architectures including an Indian
  Ocean box. A further step along the road the depth-weighted loss started down.
- [**Attention-enhanced thermocline depth in the tropical Indian
  Ocean**](https://www.sciencedirect.com/science/article/pii/S146350032500040X) — same
  basin, predicting thermocline *depth* as a scalar rather than temperature on a grid.
  RMSE 5.29 m, r 0.87. A useful bound on how well the thermocline can be located at all.

---

## 7. The ranked plan

### Before the internal round — near-zero GPU

**01 · Ship an uncertainty band.** Six members give a per-cell, per-depth spread for free.
Publish it as an envelope on the demo's profile plot plus one spread–skill reliability plot
showing that where the members disagree is where the error is. "How confident is it?" is
asked in every viva and currently has no answer. *One afternoon, no training.*

**02 · Derive the products INCOIS actually uses.** From the existing cube: depth of the
20 °C isotherm, mixed-layer depth, and 0–700 m ocean heat content — which is tropical
cyclone heat potential, the thing that makes this project matter for the Bay of Bengal.
Interpolation and a depth integral over arrays we already have. *Half a day, and the
strongest story-per-hour on this list.*

**03 · Close track B1.** INCOIS LAS gridded Argo is named in the PS and is the one
deliverable still absent. Scientifically weaker than the raw-profile track already done, so
this is compliance, not discovery — but it is the gap a judge holding the PS will find.

**04 · Adopt the `m4_sched` recipe for cost.** Rebuild the ensemble on the same budget with
more seeds. See §4.

### The two open research directions — 1–4 GPU-hours each

**05 · Train on a bias-corrected target.** Correct GLORYS against Argo *before* training,
so the network learns the observed relationship rather than the reanalysis's, instead of
learning the bias and having it subtracted afterwards. Now the highest-value untried lever,
because §2 and §3 closed everything around it. Risk: the offset drifts, so the corrected
target is itself approximate. Screen one seed on val against 0.860.

**06 · Predict a vertical EOF basis, not fifteen independent maps.** Fit PCA over the
train-split profiles and have the decoder emit six or eight coefficients. The thermocline
stops being fifteen separately-guessed numbers and becomes a coherent structure that moves
up and down — exactly the failure mode §3 identifies. Note this is **fewer** output
dimensions, not more capacity, so it is not a ninth repeat of an experiment that has failed
eight times. Precedent: NeSPReSO. Screen on the 75–150 m band, not only on blended.

### Second wave, if 05 and 06 stall

**07 · Absolute dynamic topography alongside SLA.** Dynamic height is *the* physical proxy
for thermocline depth and the model currently sees only its anomaly. Categorically unlike
the aux channels that failed — bathymetry, day-of-year and lat/lon are static or
deterministic and carry nothing about today's ocean. ADT comes from the DUACS product
already being downloaded.

**08 · Train on GLORYS and train-period Argo jointly**, as DORS does. Compatible with the
credibility story if and only if the split is **both by year and by float**: train on
2015–2021 casts, validate on 2022, test on 2023–24, no float crossing a boundary. Do not
attempt it without writing that split into `audit_leakage.py` first.

### Do not

- **More capacity, ViT, foundation models, GNN.** Eight interventions, one answer. The most
  recent improved the training objective by 3.2% and moved the observational score by
  nothing.
- **Any further bias-correction stratification.** Six forms, cross-validated, all null (§2).
- **Optimised ensemble weights.** −0.1% (§2).

---

## 8. What was built

| File | What it does |
|---|---|
| `src/correction_forms.py` | **new.** Float-blocked repeated half-split CV over seven correction forms. Refuses to run on the test split. Its self-check injects a purely depth-wise bias and asserts the extra bins do *not* win, then injects a genuinely basin-dependent one and asserts they *do* — the control that stops the first assertion being vacuous. |
| `src/train.py` | `init_head_bias` seeds the output head with the train-split depth means; `sched: cosine` decays the learning rate and is restored correctly on resume. Both default off. |
| `configs/m4_sched.yaml` | **new.** M4 under the fixed recipe, 40 epochs. |
| `src/bias_correct.py` | docstring records the six null forms and points at the instrument. |
| `.github/workflows/selfchecks.yml` | runs `correction_forms` too — eight modules now. |

---

## 9. Housekeeping

**The GPU box was stopped mid-write-up**, so three artefacts from §4 are not in `results/`:
`m4_sched_s1.csv`, `m4_sched_s1_best_val_argo.csv` and `m4_sched_s1_best_val_depthwise.csv`,
plus the checkpoint. The numbers quoted in §4 were read from the live run and are recorded
here; **re-sync them from the instance before quoting them anywhere else**, or regenerate
with the two commands below. Nothing in `FROZEN.md` depends on them.

**The offset still drifts.** §5. Refit annually against the most recent Argo.

## 10. Reproducing Day 4

```bash
# the six correction forms (no GPU; needs the val cube and the Argo parquet)
python src/correction_forms.py --cube results/ens_mix6_val_cube.nc --split val

# intervention #8 (GPU, ~45 min on a T4)
python src/train.py configs/m4_sched.yaml --seed 1
python src/predict_cube.py --ckpt checkpoints/m4_sched_s1_best.pt --split val
```

The significance test in §4 is `argo_eval.paired_bootstrap` on the two val cubes, 1,000
resamples, blocked by float — the same instrument as every other comparison in this
project.
