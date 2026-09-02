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

### 1. CNN-first is right *for our data scale* — but do not overclaim it

The defensible claim, and the one to make on stage:

> At our data scale (~2,800 daily samples, one basin, a free T4 GPU), a convolutional backbone with attention where it demonstrably helps is the highest-skill-per-unit-compute choice. Transformers are data-hungry, and the structures that dominate subsurface reconstruction — fronts, eddies, thermocline gradients — are local.

Supporting evidence: every top-performing regional system we found is convolutional with attention added inside it, not a pure ViT — [EBAM-CNN](https://www.sciencedirect.com/science/article/pii/S146350032500040X) (block attention CNN, tropical Indian Ocean) and [attention 3D-U-Net++](https://essd.copernicus.org/articles/18/4617/2026/) (NW Pacific). OceanDepths' own baselines also show spatial U-Nets beating point-wise models decisively.

**⚠️ Counter-evidence — know this before you claim "Transformers don't work":** **DUViT** (dual U-Vision-Transformer) reconstructs eddy-resolving 3D T/S/currents to 2000 m in the South China Sea from multi-resolution satellite data. Transformers demonstrably *can* do this task well. A judge who knows the field may raise exactly this.

**So the honest framing is a resource argument, not a capability argument.** "A ViT is not the best use of our data and compute budget" is defensible and true. "Transformers lose at this task" is not, and would collapse under one informed question. Use the former.

> **Citation confidence note.** An earlier draft of this doc asserted that Remote Sens. 17:2005 ran an explicit CNN-vs-Transformer head-to-head with CNN winning. That claim traced only to a search-engine snippet we could not attribute to a specific paper (MDPI blocks automated fetching), so it has been removed rather than cited. The paper itself exists and is real — *Reconstruction of Three-Dimensional Temperature and Salinity in the Equatorial Ocean with Deep-Learning* — but **read it before quoting any head-to-head result from it.** Do not put an unverified comparison in the deck.

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

### Bonus finding — significant wave height is a cheap accuracy win (upgrade path)

A 2025 JMSE study ([10.3390/jmse13050910](https://doi.org/10.3390/jmse13050910)) added **significant wave height (SWH)** as an input to U-Net / VI-U-Net T-S reconstruction and reported NRMSE reductions of **up to 40% for temperature** in the thermocline band (100–300 m), with the largest gains in *"tropical regions with active wind-waves (such as the Indian Ocean…)"* — i.e. exactly our basin, exactly the depths where our error will concentrate.

We are **not** adding SWH for the internal round: the PS specifies 7 surface variables, and an 8th channel is scope creep before M4 works. But this is the single best-evidenced "what's next" slide we have — an 8th channel, one line in the config, targeted at our known weak band. Mention it as future work; it shows we read past the requirements.

## Why GLORYS-as-target + Argo-as-validation is the right split

Reanalyses (GLORYS12) are dense but are *model outputs* with documented biases (seasonal T bias above 100 m; heat-content discrepancies at 700–2000 m — Verezemskaya et al. 2021). Training on GLORYS is fine (dense supervision), but **evaluating only on GLORYS risks rewarding reproduction of its errors**. OceanDepths, OceanBench, and OceanForecastBench all converge on the same protocol we use: train on reanalysis, validate on independent in-situ profiles. This is the single most defensible methodological choice in our stack — lead with it when questioned.

## Where OceanEmbed is novel (our pitch)

> **Revised after measurement.** The original pitch led with attention and ConvLSTM. We
> built both, measured both, and neither improves the observational score by a significant
> margin (attention's 95% bootstrap interval is [−0.011, +0.011]). Claiming them as
> contributions would hand a judge the counter-question. The real contribution turned out
> to be somewhere else, and it is a better one. See [doc 10](10-experiment-programme.html).

1. **We measured the training target's own error, and corrected it.** GLORYS12V1 carries a
   **+0.723 °C warm bias at 100 m** in this basin against the same independent Argo casts we
   validate on. That is a *ceiling*: four architectural interventions failed to remove an
   error the architecture never created. Correcting it with a fifteen-number depth-wise
   offset fitted on validation-year Argo is worth **−8.5%**, more than every architecture
   change in the project combined, and at 125/700/1000 m it takes the model **past GLORYS
   itself**. This is the paper.
2. **North Indian Ocean focus** — most published regional systems are NW Pacific / Gulf of
   Mexico / South China Sea. Bay of Bengal has unique physics (huge freshwater input →
   salinity-stratified barrier layer → SSS matters far more than elsewhere), which justifies
   our 7-variable input over the SST+SSH-only SOTA.
3. **Observation-validated, with honest statistics** — depth-wise metrics against
   INCOIS-relevant Argo floats, not reanalysis self-consistency; and comparisons tested with
   a paired bootstrap that **blocks by float rather than by profile**, because 6,448 casts
   come from only 147 floats and a naive bootstrap reads ~6.6× too narrow. We have not seen
   this done in the regional reconstruction literature, and it is what let us retract our own
   attention result rather than publish noise.
4. **A negative result worth reporting.** Seven model-side interventions — attention,
   ConvLSTM, vertical-gradient loss, anomaly formulation, climatology-as-input, auxiliary
   channels (bathymetry / day-of-year / lat-lon), depth-weighted loss — none of which
   improved the observational score on its own, against two output-side stages that both
   did. Notably `m4_aux` fit the reanalysis **better** (val GLORYS RMSE 0.659) while scoring
   **worse** against Argo. In a sparse-observation regime the binding constraint is target
   quality, not model capacity, and the field's default move — more architecture — is the
   wrong one.
5. **Beats climatology at all 15 depths.** A stronger claim than it sounds: climatology is a
   hard baseline in the deep ocean, and the published work we build on (OceanDepths) reports
   it beating naive ML. Every model we trained lost to it at 500–1000 m until the bias
   correction was added.

*Temporal context (~7-day ConvLSTM) is still in the delivered system and is still absent
from the daily-snapshot papers — but on our data it is worth 0.011 °C at 0.80 σ, so it is
reported as a non-significant improvement, not sold as a contribution.*

## Sources

- [OceanDepths (arXiv 2608.16373)](https://arxiv.org/abs/2608.16373) · [dataset](https://huggingface.co/datasets/ESA-philab/OceanDepths)
- [Attention 3D-U-Net++ NW Pacific (ESSD)](https://essd.copernicus.org/articles/18/4617/2026/)
- [TS-Cast (Ocean Science)](https://os.copernicus.org/articles/22/2161/2026/)
- [EBAM-CNN thermocline Indian Ocean](https://www.sciencedirect.com/science/article/pii/S146350032500040X)
- [NeSPReSO (Ocean Modelling)](https://www.sciencedirect.com/science/article/abs/pii/S1463500325000538)
- [Equatorial 3D T/S reconstruction with deep learning](https://doi.org/10.3390/rs17122005) — *read before citing any head-to-head result; see confidence note above*
- [DORS-0.25° Deep Forest (ISPRS)](https://www.sciencedirect.com/science/article/abs/pii/S0924271624003617)
- [South China Sea physics-guided DL](https://doi.org/10.3390/rs17172954)
- [Adaptive spatiotemporal clustering 3D reconstruction (arXiv 2605.00860)](https://arxiv.org/abs/2605.00860)
- [SIH 2026 PS catalogue (26066)](https://github.com/vedantchalke36/sih-2026-problem-statements)
