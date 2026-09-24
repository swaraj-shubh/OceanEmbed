@

---
title: "14 · Research & References"
nav_order: 15
---
# 14 — Research & References

Consolidated list of every paper, dataset, and link cited across the project (source: docs/02, docs/04, and cross-doc citation grep). For the full narrative, read docs/02-research-review.md (papers) and docs/04-data.md (datasets).

## Research Papers Cited

| Paper                                                                 | Region/Scope                         | Method                                                     | Why we cited it                                                                                                                                                 |
| --------------------------------------------------------------------- | ------------------------------------ | ---------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **OceanDepths** — Donike et al., arXiv 2608.16373 (ESA Φ-lab) | Global, 0.1°, weekly                | Dataset + baselines (climatology, IDW, LSTM, 1D/2D/3D CNN) | Our bootstrap dataset; its baseline table (climatology RMSE 0.974 beats point-wise ML) shaped M0's importance and the "spatial not per-pixel" architecture call |
| **DORS** — Su et al. 2022                                      | Global, 1° monthly                  | ConvLSTM                                                   | Proof ConvLSTM works for this task (coarse)                                                                                                                     |
| **DORS-0.25°** — Su et al. 2024, ISPRS                        | Global, 0.25° monthly               | Deep Forest                                                | Proof 0.25° is achievable globally                                                                                                                             |
| Smith et al. 2023, Frontiers Mar. Sci.                                | —                                   | CNN                                                        | The point-wise CNN baseline we frame ourselves against                                                                                                          |
| **NeSPReSO** 2025, Ocean Modelling                              | Gulf of Mexico                       | PCA + NN                                                   | Justifies compressing the vertical profile → supports our 15-depth output head                                                                                 |
| **TS-Cast** 2026, Ocean Science                                 | NW Pacific                           | Deep learning                                              | Current SOTA framing for regional satellite-only reconstruction                                                                                                 |
| **Attention 3D-U-Net++** 2026, ESSD 18:4617                     | NW Pacific, 0.25°, daily, 26 layers | Attention U-Net++ + transfer learning                      | **Closest published system to our design** — our plan is this idea + Indian Ocean + 5 more input vars                                                    |
| **EBAM-CNN** 2025, Science Direct                               | Tropical Indian Ocean                | Block-attention CNN                                        | Attention beats plain CNN*in our exact region* (thermocline RMSE 5.29 m, R 0.87)                                                                              |
| Zhao et al. 2025, Remote Sens. 17:2954                                | South China Sea                      | DL + physical guidance                                     | Physics-informed loss = later upgrade path, not PoC                                                                                                             |
| Equatorial 3D T/S reconstruction, Remote Sens. 17:2005                | Equatorial ocean                     | Deep learning                                              | ⚠️ Retracted head-to-head claim — cited only with the caveat "read before quoting"                                                                           |
| Adaptive spatiotemporal clustering, arXiv 2605.00860                  | —                                   | 3D reconstruction                                          | Background/related work                                                                                                                                         |
| JMSE 10.3390/jmse13050910                                             | incl. Indian Ocean                   | U-Net / VI-U-Net + SWH input                               | Bonus finding: adding significant wave height cut thermocline NRMSE up to 40% — noted as future work, not adopted (scope creep beyond PS's 7 vars)             |
| **DUViT** (dual U-ViT)                                          | South China Sea                      | Transformer                                                | Counter-evidence we deliberately kept — Transformers*can* do this task; shaped our honest "resource argument, not capability argument" framing               |
| Verezemskaya et al. 2021                                              | —                                   | GLORYS bias analysis                                       | Basis for "reanalysis has documented biases → validate on independent Argo"                                                                                    |
| SIH 2026 PS catalogue#26066                                           | —                                   | —                                                         | The problem statement itself                                                                                                                                    |

## Datasets / Data Products

| Role                    | Product                                                                             | Resolution            | Access                            |
| ----------------------- | ----------------------------------------------------------------------------------- | --------------------- | --------------------------------- |
| SST                     | NOAA OISST v2.1                                                                     | 0.25°, daily         | NCEI THREDDS, no account          |
| SSS                     | NASA SMAP RSS L3 SSS**V6** (8-day running mean)                               | 0.25°                | PO.DAAC, Earthdata login          |
| SSH/SLA                 | Copernicus DUACS L4`SEALEVEL_GLO_PHY_L4_MY_008_047`                               | 0.125°               | `copernicusmarine`, CMEMS login |
| Currents U/V            | NASA OSCAR v2.0 (`u`,`v`, not `ug`,`vg`)                                    | 0.25°, daily         | PO.DAAC                           |
| Winds U/V               | Copernicus`WIND_GLO_PHY_L3_MY_012_005` (fallback L4_012_006)                      | 0.125°, daily        | `copernicusmarine`              |
| **Target**        | GLORYS12V1`GLOBAL_MULTIYEAR_PHY_001_030` (`doi:10.48670/moi-00021`) — PS-named | 1/12°, 50 levels     | `copernicusmarine`              |
| **Validation B1** | INCOIS LAS Gridded ARGO — PS-named                                                 | 1°, 10-day           | INCOIS LAS OPeNDAP, no account    |
| **Validation B2** | Raw Argo profiles (Ifremer ERDDAP GDAC, not argopy in the end)                      | point obs             | no account                        |
| Bootstrap               | ESA Φ-lab**OceanDepths** (HF `ESA-philab/OceanDepths`)                     | 0.1°, weekly, global | `huggingface_hub`               |

## All Links Referenced in the Repo

- OceanDepths paper: https://arxiv.org/abs/2608.16373 · dataset: https://huggingface.co/datasets/ESA-philab/OceanDepths
- Attention 3D-U-Net++: https://essd.copernicus.org/articles/18/4617/2026/
- TS-Cast: https://os.copernicus.org/articles/22/2161/2026/
- EBAM-CNN: https://www.sciencedirect.com/science/article/pii/S146350032500040X
- NeSPReSO: https://www.sciencedirect.com/science/article/abs/pii/S1463500325000538
- Equatorial 3D T/S DL: https://doi.org/10.3390/rs17122005
- DORS-0.25°: https://www.sciencedirect.com/science/article/abs/pii/S0924271624003617
- South China Sea physics-guided DL: https://doi.org/10.3390/rs17172954
- SWH bonus finding: https://doi.org/10.3390/jmse13050910
- Adaptive spatiotemporal clustering: https://arxiv.org/abs/2605.00860
- GLORYS DOI: https://doi.org/10.48670/moi-00021
- INCOIS data holdings: https://incois.gov.in/site/dataholdings.jsp
- SIH PS catalogue: https://github.com/vedantchalke36/sih-2026-problem-statements
- Repo: https://github.com/swaraj-shubh/OceanEmbed.git

## The 3 Big Research Findings That Shaped Design Decisions

1. **Climatology is brutally strong** (OceanDepths: 0.974°C RMSE, beats point-wise LSTM/CNN) → M0 baseline is mandatory, spatial U-Net not per-pixel MLP.
2. **Error concentrates at the thermocline (50–200m)** across every paper reviewed → depth-wise metrics, not one blended number, is the primary result.
3. **Reanalysis-as-target + independent-in-situ-as-validation** is the field-standard protocol (OceanDepths, OceanBench, OceanForecastBench all converge on it) → our GLORYS-train / Argo-validate split.
