---
title: "Day 5 — Track B1"
nav_order: 14
---

# 13 — Track B1: INCOIS Gridded Argo

Doc 12 §7 listed **03 · Close track B1** as "the one deliverable still absent — the gap a
judge holding the PS will find." It is now closed.

**The headline number does not change. 0.786 °C, track B2, raw Argo profiles, stands.**
What this document adds is a second, independent reference product, the result of scoring
against it, and — more usefully — the reason its ordering of the same four systems comes
out *backwards*, which is a fact about the reference, not about the models.

Read §3 if you read nothing else.

---

## 1. What B1 is and why it exists

The problem statement names **INCOIS LAS Gridded ARGO** by name. It is an objective
analysis: raw Argo casts interpolated onto a **1° × 1° grid in 10-day windows**, 24 depth
levels to 2000 m. It is not an observation; it is a smoothed *field* fitted to
observations, filling every cell including those no float visited that window.

That makes it categorically weaker evidence than track B2 (raw profiles, doc 11), and the
project has always said so. It is here because the PS names it, and because a second
reference with different failure modes is a real cross-check on the one result Day 3
produced that mattered.

**Data:** `INCOIS_Argo_VAM_10d_{2022,2023,2024}.nc`, concatenated time-sorted with
duplicate windows dropped into `data/interim/argo_10d.nc` (109 windows, 2021-12-30 →
2024-12-30, `TEMP` only). Mirrored at
`s3://oceanembed-sih26-data/oceanembed/interim/argo_10d.nc`.

**Method:** `src/incois_eval.py`, unchanged in substance since it was written — the
comparison **aggregates our 0.25° daily field up to the reference's 1°/10-day footprint**,
never the reverse (CLAUDE.md rule 3). One fix was needed to run it: INCOIS names its
vertical axis `ZAX`, which was missing from the dimension rename map, so `("ZAX", "depth")`
was added. The module's synthetic self-check passes unchanged.

**Scope:** 72 ten-day windows inside the test split (2023–24), 784,152 scored cell-levels,
14 of our 15 depths. **Depth 0 is not scored** — INCOIS starts at 5 m.

---

## 2. The result

Blended RMSE against INCOIS gridded Argo, test split 2023–24, all four systems evaluated
identically:

| System | **B1** (INCOIS gridded) | B2 (raw Argo casts) |
|---|---|---|
| **ens_mix6 + bias correction — shipped** | **1.232** | **0.786** |
| M0 climatology | 1.278 | 1.160 |
| ens_mix6, no correction | 1.320 | 0.859 |
| GLORYS12V1 — the training target | 1.437 | 0.728 |

Depth-wise RMSE, the levels the PS asks to report:

| Depth (m) | FINAL | M0 clim. | GLORYS |
|---|---|---|---|
| 5 | 1.368 | **1.299** | 1.475 |
| 50 | **1.158** | 1.297 | 1.522 |
| 100 | **1.825** | 2.033 | 2.177 |
| 200 | 0.975 | **0.918** | 1.106 |
| 500 | 0.472 | **0.471** | 0.508 |
| 1000 | **0.523** | 0.502 | 0.556 |

Two things carry over from B2 unchanged: **the thermocline is where the error lives**
(100 m is the worst level for every system, by a wide margin), and **the correction
helps** — it moves the blended score 6.7% and cuts the 100 m bias from +1.016 to +0.422.

**Blended MAE, Bias, Correlation and R² — GLORYS vs FINAL**, same `n`-weighted pooling as
the RMSE above (weighted RMS for RMSE, weighted mean for the other four). Computed by
`metrics.blend_all()` — the same function the demo's Skill tab uses for the B2 comparison
(`app/loader.final_vs_glorys()`), applied here to the two `*_test_incois_b1.csv` files
instead; not wired into the demo UI, since B1 is out of scope for what the demo shows:

| Metric | GLORYS | FINAL | Δ |
|---|---|---|---|
| RMSE | 1.437 | **1.232** | −0.205 |
| MAE | 0.993 | **0.851** | −0.142 |
| Bias | +0.622 | **+0.359** | −0.263 |
| Corr | 0.678 | **0.700** | +0.022 |
| R² | **−0.230** | **+0.070** | +0.299 |

FINAL wins every metric here, including correlation — not just bias as on B2. **This is
not evidence the model is more accurate than GLORYS.** GLORYS's R² is *negative*: worse
than predicting the mean at every point, which is not a plausible statement about a serious
reanalysis product and is not what B2 (0.877) says about the same GLORYS field. It is a
statement about this reference's smoothing, exactly as §3 below argues from RMSE alone —
the correlation and R² numbers are further symptoms of the same artefact, not independent
confirmation of a real accuracy win. Do not quote any of this row as "beats GLORYS."

---

## 3. The ordering inverts, and that is the finding

In B2, GLORYS is the **ceiling**: 0.728, better than anything trained on it, exactly as
doc 09 predicted. In B1, GLORYS is the **worst of the four** at 1.437 — worse than
climatology.

GLORYS did not get worse. **The reference changed.**

INCOIS VAM is a 1°, 10-day objective analysis. Its own smoothing removes the mesoscale —
eddies, fronts, filaments — that GLORYS at 1/12° actually resolves. When a sharp field is
scored against a smooth one, the sharpness is counted as error. Our prediction and the
climatology are both smooth fields to begin with, so they are penalised far less. The
ranking is measuring **how much each product looks like a smoothed analysis**, which is
not the same question as how accurate it is.

This has a direct consequence for the viva:

> **Do not cite "we beat GLORYS by 14%" from track B1.** It is a representativeness
> artefact. The honest statement is that B1 and B2 rank the same four systems differently,
> that the reason is the reference product's smoothing, and that B2 — actual measurements —
> is the one to believe.

### B1 barely separates the model from climatology

The second consequence is about discriminating power:

| | model vs climatology |
|---|---|
| B2, raw casts | **32% better** |
| B1, gridded analysis | **3.6% better** |

Ten-day, 1° averaging removes most of what the model adds. This is not a weakness of the
model; it is a ceiling on what B1 can resolve, and it is why B1 could never have been the
primary track. A reference that cannot separate a neural network from a monthly mean is a
compliance artifact, not a benchmark.

---

## 4. What B1 genuinely bought

Two things, and they are worth having:

**1. The bias correction transfers to a reference it was never fitted against.** The 15
offsets were fitted on raw 2022 Argo casts. Applied unchanged, they improve the score
against a *different product built by a different institution with a different method* by
6.7%, and the 100 m bias falls from +1.016 to +0.422. Doc 12 §2 established that no
*fancier form* of the correction helps; this establishes that the correction is not an
artefact of the B2 metric it was tuned on. That is the strongest single sentence available
about the correction, and it did not exist before today.

**2. The thermocline diagnosis reproduces against an independent reference.** Doc 12 §3
found 62% of the squared error in 75–150 m and argued it was variance, not bias. B1 shows
the same shape — 100 m the worst level for every system including the reanalysis, and the
only level where the model beats climatology by a wide margin (10%). A finding that
reproduces against a reference with entirely different failure modes is much harder to
dismiss as an artefact of the Argo matching rule.

### And one thing to be honest about

At **5 m the model is worse than climatology** (1.368 vs 1.299), the only level where that
is true in either track. The near-surface is where the 10-day window hurts most — SST
varies on shorter timescales than the reference averages over — and where our own input is
strongest, so the model has the least excuse. Worth a sentence if asked, not worth an
experiment: it is one level out of fourteen and the blended number already includes it.

---

## 5. Reproduce

```bash
# on the GPU box
aws s3 cp s3://oceanembed-sih26-data/oceanembed/interim/argo_10d.nc data/interim/argo_10d.nc

python src/incois_eval.py --cube results/ens_mix6_bc_test_cube.nc \
                          --incois data/interim/argo_10d.nc --split test
```

`results/*_test_incois_b1.csv` holds the depth-wise table for each of the four systems.
The bias-corrected test cube is built by applying `results/ens_mix6_offset.json` to
`results/ens_mix6_test_cube.nc` with `bias_correct.apply_offset`; the offset file's
`depths` list is asserted against the cube's depth coordinate first, because a silent
reorder there would corrupt every number above.

---

## 6. What this changes in the plan

**Doc 12 §7 item 03 is done.** Nothing else on that list moves. The ranked plan stands:
01 uncertainty band, 02 INCOIS-facing derived products (D20, MLD, OHC), 04 the `m4_sched`
recipe for cost, then 05 bias-corrected target and 06 the EOF basis as the two research
directions.

One line in the constitution needed updating: CLAUDE.md §2 described B1 as a validation
track to be built. It is built, and its result is recorded here rather than in the frozen
results table — because **B1 is not the number this project reports.** B2 is.
