---
title: "01 · Problem Statement"
nav_order: 2
---

# 01 — Problem Statement Analysis (SIH PS 26066)

## Official PS (verified against the SIH 2026 catalogue)

- **ID:** SIH26066
- **Title:** *OceanEmbed — Satellite Embedding-Based Deep Learning Framework for Reconstruction of Subsurface Ocean Temperature from Surface Satellite Observations*
- **Organization:** Ministry of Earth Sciences (MoES) — Indian National Centre for Ocean Information Services (INCOIS)
- **Category:** Software · **Theme:** Space Technology

### Expected outcomes (verbatim requirements)

1. Complete **preprocessing pipeline** harmonizing multi-source satellite datasets.
2. A **satellite embedding engine** leveraging deep learning architectures (CNN, Vision Transformers, autoencoders, GNNs are *mentioned*, not all required).
3. A **reconstruction model** estimating temperature at **15 standard depth levels**.
4. **Daily outputs at 0.25° spatial resolution**.
5. **Validation using independent Argo observations**.
6. **Functional proof-of-concept over the Bay of Bengal / Arabian Sea** (North Indian Ocean).

Input variables named by the PS: SST, sea surface salinity, sea surface height, surface currents, surface winds — **7 channels** total (SST, SSS, SSH/SLA, U-current, V-current, U-wind, V-wind).

## What the PS is really asking (interpretation)

The ocean absorbs >90% of excess anthropogenic heat and ~30% of CO₂ emissions. Subsurface thermal structure controls:

- **Cyclone intensification** in the Bay of Bengal (upper-ocean heat content, not just SST, feeds cyclones — directly in INCOIS's mandate).
- **Ocean heat content** monitoring for climate.
- **Fisheries advisories** (thermocline depth ↔ fish aggregation — INCOIS PFZ service).
- **Navy/underwater acoustics** (sound speed depends on the T-profile).

Satellites can't see below the skin of the ocean. Argo floats can, but one float covers ~3°×3° and surfaces every ~10 days — ~99.9% of grid cells at any depth have no observation in a given week (OceanDepths paper). The physics that makes reconstruction possible: **surface expressions of subsurface dynamics** — SSH integrates the density (and hence temperature) structure of the whole water column; SST fronts, eddies and salinity signatures correlate with the thermocline. A neural network can learn this mapping much cheaper than a physics-based data-assimilation system.

## What "OceanEmbed" means

The PS name signals: **learn an embedding of the surface state** that captures information relevant to the subsurface. In our design, OceanEmbed is the latent representation after the CNN encoder + temporal ConvLSTM + attention fusion — not a separate magic model. The decoder reads this embedding into 15 temperature maps.

## Judging criteria we optimize for (internal round)

| Criterion | Our answer |
|---|---|
| Understanding of the problem | This doc + the research review (doc 02) |
| Technical soundness | Progressive M0→M4 model plan, time-based splits, independent Argo validation |
| Working demo | Streamlit PoC — map + profile + Argo overlay + live metrics |
| Feasibility | Regional 0.25° scope, free/cheap compute, staged milestones |
| Innovation | Attention-fused multivariate embedding + temporal context, validated against observations (most published Indian Ocean work validates only against reanalysis) |
| Impact | Cyclone heat-content, PFZ, INCOIS operational relevance |

## Deliverables checklist for the internal round

- [ ] Preprocessing pipeline producing model-ready `X=[7,H,W]`, `Y=[15,H,W]` daily samples
- [ ] Trained OceanEmbed model (M3 minimum, M4 target)
- [ ] Depth-wise metrics table vs GLORYS *and* vs held-out Argo
- [ ] Ablation table M0 → M4
- [ ] Streamlit demo running offline from frozen checkpoint
- [ ] Presentation deck + this documentation set
