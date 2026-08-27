---
title: "02 · Research Review"
nav_order: 3
---

# 02 — Research Review: What the Literature Says

The point of this review: every architecture decision we made is backed by a published result. Use these citations in the presentation.

## The task family: surface → subsurface reconstruction

| Work | Region | Method | Inputs | Key takeaway for us |
|---|---|---|---|---|
| Su et al. 2022 (**DORS**) | Global 1° monthly | ConvLSTM | SST, ADT, wind + EN4 | ConvLSTM works for this task; coarse resolution though |
| Su et al. 2024 (**DORS-0.25°**) | Global 0.25° monthly | Deep Forest | same | 0.25° is achievable globally |
| Smith et al. 2023 (Frontiers Mar. Sci.) | — | CNN | satellite + in-situ | The point-wise CNN baseline our PoC beats |
| **NeSPReSO** 2025 (Ocean Modelling) | Gulf of Mexico | PCA + NN | ADT, SST, SSS + Argo | Compressing the vertical profile (PCA / few depths) makes learning easier — supports our 15-depth output head |
| **TS-Cast** 2026 (Ocean Science) | NW Pacific | deep learning | satellite obs | Regional satellite-only reconstruction is publishable, current SOTA task framing |
| **Attention 3D-U-Net++** 2026 (ESSD 18:4617) | NW Pacific, 1/4°, daily, 26 layers 5–2000 m | attention-enhanced U-Net++ + transfer learning | SST, SSH | **Closest published system to our design.** Daily, 0.25°, attention + U-Net. Validated vs WOD; captures thermocline gradient at 200 m. Our plan = this idea, Indian Ocean, +5 more input variables |
| EBAM-CNN 2025 (Sci. Direct) | **Tropical Indian Ocean** | block-attention CNN | ADT, SST, wind | Attention-enhanced CNN beats plain CNN *in our region* (thermocline depth RMSE 5.29 m, R 0.87) |
| Zhao et al. 2025 (Remote Sens. 17:2954) | South China Sea | DL + physical guidance | satellite | Physics-informed loss is a later upgrade path, not needed for PoC |
| **OceanDepths** 2026 (arXiv 2608.16373, ESA Φ-lab) | Global 0.1° weekly | dataset + baselines | SST, SSS, ADT + EN4 + GLORYS12 | Our bootstrap dataset. Its baseline table is a warning (below) |

## Three findings that shape our design

### 1. CNN > Transformer at this data scale
Reconstruction of 3D equatorial T/S (Remote Sens. 17:2005, 2025) explicitly compared architectures: **Transformer performed worse than CNN** — self-attention is good at global features but poor at the locally small-scale structures (fronts, eddies) that dominate subsurface reconstruction, and ViTs are data-hungry. This is our documented justification for CNN-first, and for using attention only as a *fusion* mechanism inside a convolutional backbone (as EBAM-CNN and the 3D-U-Net++ paper do). **Do not build a ViT for the internal round.**

### 2. Climatology is embarrassingly strong — respect the baseline
OceanDepths baseline table (temperature, vs held-out EN4):

| Method | RMSE (°C) | R² |
|---|---|---|
| Climatology | **0.974** | 0.964 |
| IDW interpolation | 0.979 | 0.964 |
| Point-wise LSTM | 2.420 | 0.701 |
| Point-wise 1D CNN | 3.073 | 0.370 |
| 2D U-Net | 1.092 | 0.941 |
| 3D U-Net | 1.101 | 0.937 |

Naive ML **loses to climatology**. Spatial U-Nets are competitive; point-wise models are not. Two consequences: (a) M0 climatology baseline is mandatory and must be reported honestly; (b) our model must use spatial context (U-Net-style), never per-pixel MLPs. A judge-winning claim is "we beat climatology at thermocline depths", not a bare RMSE.

### 3. Error concentrates at the thermocline — report depth-wise
All papers show RMSE peaks around 50–200 m (thermocline) where vertical gradients are sharpest, and is small at surface and below 500 m. So: a single averaged RMSE hides everything; the depth-wise table/curve is the honest and standard reporting format. Also motivates the optional depth-weighted loss (upweight 50–200 m) as an M4+ experiment.

## Why GLORYS-as-target + Argo-as-validation is the right split

Reanalyses (GLORYS12) are dense but are *model outputs* with documented biases (seasonal T bias above 100 m; heat-content discrepancies at 700–2000 m — Verezemskaya et al. 2021). Training on GLORYS is fine (dense supervision), but **evaluating only on GLORYS risks rewarding reproduction of its errors**. OceanDepths, OceanBench, and OceanForecastBench all converge on the same protocol we use: train on reanalysis, validate on independent in-situ profiles. This is the single most defensible methodological choice in our stack — lead with it when questioned.

## Where OceanEmbed is novel (our pitch)

1. **North Indian Ocean focus** — most published regional systems are NW Pacific / Gulf of Mexico / South China Sea. Bay of Bengal has unique physics (huge freshwater input → salinity-stratified barrier layer → SSS matters far more than elsewhere), which justifies our 7-variable input over the SST+SSH-only SOTA.
2. **Full 7-variable multivariate fusion with attention** — published Indian Ocean work (EBAM-CNN) uses 3 inputs; we fuse 7.
3. **Observation-validated** — depth-wise metrics against INCOIS-relevant Argo floats, not just reanalysis self-consistency.
4. **Temporal context** — ~7-day ConvLSTM window, which the daily-snapshot papers omit.

## Sources

- [OceanDepths (arXiv 2608.16373)](https://arxiv.org/abs/2608.16373) · [dataset](https://huggingface.co/datasets/ESA-philab/OceanDepths)
- [Attention 3D-U-Net++ NW Pacific (ESSD)](https://essd.copernicus.org/articles/18/4617/2026/)
- [TS-Cast (Ocean Science)](https://os.copernicus.org/articles/22/2161/2026/)
- [EBAM-CNN thermocline Indian Ocean](https://www.sciencedirect.com/science/article/pii/S146350032500040X)
- [NeSPReSO (Ocean Modelling)](https://www.sciencedirect.com/science/article/abs/pii/S1463500325000538)
- [Equatorial 3D T/S reconstruction — CNN vs Transformer](https://doi.org/10.3390/rs17122005)
- [DORS-0.25° Deep Forest (ISPRS)](https://www.sciencedirect.com/science/article/abs/pii/S0924271624003617)
- [South China Sea physics-guided DL](https://doi.org/10.3390/rs17172954)
- [Adaptive spatiotemporal clustering 3D reconstruction (arXiv 2605.00860)](https://arxiv.org/abs/2605.00860)
- [SIH 2026 PS catalogue (26066)](https://github.com/vedantchalke36/sih-2026-problem-statements)
