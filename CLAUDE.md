# CLAUDE.md — OceanEmbed (SIH PS 26066)

**Project:** Satellite Embedding-Based Deep Learning Framework for Reconstruction of Subsurface Ocean Temperature
**Team goal:** Working demo + solid research story to clear the SIH internal round, then nationals.
**Codename:** OceanEmbed — the learned latent representation that maps observable surface ocean state → hidden subsurface temperature structure.

---

## 1. The Problem in One Paragraph

Satellites see only the ocean surface (SST, SSS, SSH, currents, winds) at high resolution; Argo floats measure subsurface temperature down to ~2000 m but are extremely sparse (~4000 floats globally, ~0.01% coverage per depth level). The PS asks us to learn the surface→subsurface relationship: given 7 surface fields, reconstruct the 3D temperature structure (0–1000 m) as gridded maps. Training target is GLORYS12 reanalysis (dense); **final validation is against held-out Argo observations** (independent, observational — this is the credibility story judges care about).

## 2. The Committed Design (do not re-litigate)

| Decision | Choice |
|---|---|
| Region (PoC) | Arabian Sea + Bay of Bengal |
| Grid | 0.25° × 0.25°, daily |
| Input X | `[7, H, W]` — SST, SSS, SSH/SLA, Current U, Current V, Wind U, Wind V |
| Output Y | `[15, H, W]` — T at depths 0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000 m |
| Architecture | CNN encoder → ConvLSTM (~7-day window) → Attention fusion (= OceanEmbed latent) → U-Net decoder |
| Training target | **GLORYS12V1** (PS-named, `doi:10.48670/moi-00021` = `GLOBAL_MULTIYEAR_PHY_001_030`), regridded + interpolated to the 15 SIH depths |
| Validation | Two tracks: **B1 INCOIS LAS Gridded ARGO** (PS-named, 1°/10-day — aggregate our output up to it) and **B2 raw Argo profiles** (stricter). Depth-wise RMSE, MAE, Bias, Correlation |
| Loss | MSE first; depth-weighted loss only if results justify it |
| Demo | Streamlit: pick date/depth → reconstructed map; click location → 0–1000 m profile + nearby Argo overlay + metrics |
| DL framework | PyTorch |
| Data stack | xarray, NumPy, SciPy, xESMF (regridding), NetCDF/Zarr, Dask if needed |
| Viz | Matplotlib, Cartopy, Plotly |

**Why not ViT/GNN/foundation model:** PS mentions them but doesn't require them. CNN is data/compute-efficient for gridded fields; ViT needs more data, GNN adds graph-construction complexity. Attention is included as the fusion layer — that satisfies the "attention-based" checkbox honestly.

## 3. Model Stages (build progressively — never skip ahead)

- **M0** — Climatology / mean-profile baseline (non-AI). *Must exist before any DL claim; the OceanDepths paper shows climatology beats naive ML.*
- **M1** — Tiny CNN, SST-only or few channels, small subset. Prove the mapping is learnable.
- **M2** — CNN/U-Net with all 7 variables. Meets the full SIH input requirement.
- **M3** — + attention fusion → this *is* OceanEmbed.
- **M4** — + ConvLSTM 7-day temporal context. Final PoC model.

Report every stage against M0. If a stage doesn't beat the previous one, investigate before adding complexity.

## 4. Data Sources (exact products)

| Variable | Product | Notes |
|---|---|---|
| SST | NOAA OISST v2.1 | daily, 0.25° — already on target grid |
| SSS | NASA SMAP L3 SSS (RSS 8-day running mean V4) | 0.25°; document the composite window |
| SSH/SLA | Copernicus Marine DUACS altimetry L4 | regrid to 0.25° |
| Currents U/V | NASA OSCAR v2.0 (PO.DAAC) | daily 0.25° |
| Wind U/V | Copernicus scatterometer/ASCAT | pick a gridded product, regrid |
| Target T | Copernicus GLORYS12V1 (`GLOBAL_MULTIYEAR_PHY_001_030`, `doi:10.48670/moi-00021`) — **named by the PS** | 1/12°, 50 levels → regrid to 0.25°, interp to 15 depths |
| Validation B1 | **INCOIS LAS Gridded ARGO** — **named by the PS** | 1°×1°, 10-day/monthly objective analysis; aggregate our 0.25° output up to compare |
| Validation B2 | Raw Argo profiles (argopy / EN4) | point observations; stricter test. Neither track ever used as a training target |
| Bootstrap | **ESA Φ-lab OceanDepths** (HuggingFace `ESA-philab/OceanDepths`) | ~120 GiB global; use its patches for M0/M1 before the 7-source pipeline exists |

Copernicus Marine requires a (free) account — use `copernicusmarine` Python client. PO.DAAC needs NASA Earthdata login. Register both early; credentials go in `.env` / `~/.netrc`, **never committed**.

### OceanDepths facts worth remembering
0.1° global, weekly, 2000–2024; 9.5M EN4 profiles interpolated to 50 GLORYS depth levels; 128×128 patches; eval year = 2018; comes with PyTorch DataLoaders. Its baselines: 2D U-Net beat LSTM/1D-CNN, climatology is a strong baseline. Use it to get a model training on day 1 without building any pipeline.

## 5. Data Pipeline Contract

Raw products → QC → subset Arabian Sea + Bay of Bengal → regrid 0.25° → daily alignment → unit standardization → land/missing masks → normalize **using TRAIN-split stats only** → save model-ready `(X=[7,H,W], Y=[15,H,W])` samples (NetCDF/Zarr).

The model never touches raw provider files. **Frozen region bbox (docs/04): lat 0–25°N, lon 55–100°E → 100×180 at 0.25°, padded to 96×176 model grid. Period 2015–2022 (SMAP SSS constraint); split: train 2015–20 / val 2021 / test 2022.**

## 6. Non-Negotiable Methodology Rules

1. **Time-based split.** Train on earlier years, validate on a later year, test on later still (e.g. train ≤2020, val 2021, test 2022+). Never random pixel/patch mixing; no overlapping patches across split boundaries.
2. **Normalization stats from train split only** — computed once, saved as an artifact, applied everywhere.
3. **Argo is sacred.** Argo data is never a model input or target, in either validation track. Match by date/location, interpolate to common depths, compute depth-wise metrics. PS requirement 7 explicitly permits the regridding this needs — when resolutions differ, **aggregate our prediction up to the coarser reference**, never downscale the reference.
4. **Depth-wise metrics table** (RMSE/MAE/Bias/Corr at 0, 50, 100, 200, 500, 1000 m) is the primary result, not one blended number.
5. **Masks:** land and missing-data masked out of the loss. Never let the model be scored on land pixels.
6. **Checkpoint everything** (model + optimizer + epoch + config + norm stats) — required for Spot/preemptible GPUs and demo reproducibility.
7. Every experiment gets a config file (YAML) and a run name; results logged (CSV or W&B free tier). No untracked "it worked yesterday" runs.

## 7. Compute Strategy

- **Phase 1 (free):** Kaggle (T4 16 GB) / Colab for all development, M0–M2. This region at 0.25° is small — a 16 GB GPU is plenty.
- **Phase 2 (paid, only when needed):** RunPod RTX 4090 (~$0.74–1.15/hr) or AWS g6.2xlarge Spot (~$0.43/hr, L4 24 GB). ~₹5–10k covers 100 GPU-hours. Do NOT rent A100/H100; do NOT run anything 24/7.
- **Free credits to chase:** Azure for Students ($100, no card), GitHub Student Pack, college institutional AWS/GCP/Azure credits, AWS Activate only if a legit startup route exists.
- Training code must survive preemption: auto-resume from latest checkpoint.

## 8. Repo Layout (create as needed, keep flat)

```
sih26/
  CLAUDE.md              # this file — project constitution
  docs/                 # 01-06: PS analysis, research, architecture, data, training, demo
  configs/               # YAML per experiment
  data/                  # gitignored; raw/ interim/ processed/
  src/
    download/            # per-source download scripts
    preprocess/          # QC, regrid, align, normalize, sample builder
    datasets.py          # PyTorch Dataset/DataLoader
    models/              # cnn.py, unet.py, convlstm.py, attention.py, oceanembed.py
    train.py             # single entrypoint, config-driven
    evaluate.py          # depth-wise metrics vs GLORYS and vs Argo
  notebooks/             # exploration only, never pipeline logic
  app/                   # Streamlit demo
  checkpoints/           # gitignored
  results/               # metric tables, figures
```

Rules: pipeline logic lives in `src/`, notebooks only explore. `data/`, `checkpoints/`, `.env` in `.gitignore`. Small, boring, readable modules — no premature abstraction, no framework-of-frameworks.

## 9. Demo Requirements (what the judge must be able to do)

1. Select region (Arabian Sea / Bay of Bengal) and a date.
2. View the 7 surface input fields.
3. Pick any of the 15 depths → reconstructed temperature map.
4. Click a location → full 0–1000 m predicted profile.
5. Overlay a nearby independent Argo profile where available.
6. See RMSE / Bias / Correlation numbers on screen.
7. (Optional flex) embedding visualization of what OceanEmbed learned.

Demo runs **offline from a frozen checkpoint + precomputed samples**. No live data ingestion, no 3D web engines.

## 10. Explicitly Out of Scope (internal round)

Global operational model · satellite foundation model from scratch · giant ViT/Transformer before CNN works · global GNN · physics-based simulator · real-time ingestion · complicated 3D visualization. If tempted, re-read §3.

## 11. Implementation Order (the checklist)

1. OceanDepths → load one sample → visualize ✅ = first milestone
2. Tiny CNN on a small subset → prove reconstruction
3. Build SIH preprocessing → produce X=[7,H,W], Y=[15,H,W]
4. Train CNN/U-Net with all seven variables
5. Add attention → OceanEmbed
6. Add ~7-day ConvLSTM temporal context
7. Validate against held-out Argo
8. Ablations + depth-wise metric tables
9. Streamlit demo
10. Freeze checkpoint, document exact data products/versions, prepare presentation

## 12. Presentation / Research Story (for the internal round)

- Motivation: ocean absorbs >90% of excess heat; subsurface structure drives heat content, cyclone intensification (very relevant for Bay of Bengal), fisheries, INCOIS services.
- Honesty points that win credibility: climatology baseline reported; GLORYS is a reanalysis (has biases) so we validate on independent Argo; time-based splits prevent leakage.
- Cite: OceanDepths (Donike et al., arXiv 2608.16373), DORS (Su et al. 2022), NeSPReSO (Gulf of Mexico), Smith et al. 2023 CNN reconstruction, OceanBench/OceanForecastBench — position OceanEmbed as regional, observation-validated, Indian-Ocean-focused.
- Keep an ablation table: M0 → M4, same test set, depth-wise RMSE. That single table is the core evidence.

## 13. Working Conventions for Claude Sessions

- Python 3.10+, PyTorch. Windows dev machine (this box) is for code/small tests; GPU training happens on Kaggle/cloud — keep code environment-agnostic (pathlib, no hardcoded Windows paths in `src/`).
- xESMF needs conda/linux — if regridding locally on Windows is painful, use `xarray.interp`/`scipy` bilinear for 0.25° targets (all sources are ≤0.25° native or close; document the method used).
- Prefer small runnable scripts with `if __name__ == "__main__"` self-checks over test frameworks at this stage.
- When adding any model component, wire it behind the config, train the smallest version that can overfit 10 samples first, then scale.
- Update this file when a decision is frozen (bbox, split years, product versions, hyperparameters that worked).

## 14. Key Links

- OceanDepths: https://huggingface.co/datasets/ESA-philab/OceanDepths
- NOAA OISST: https://www.ncei.noaa.gov/products/optimum-interpolation-sst
- SMAP SSS: https://podaac.jpl.nasa.gov/dataset/SMAP_RSS_L3_SSS_SMI_8DAY-RUNNINGMEAN_V4
- OSCAR currents: https://podaac.jpl.nasa.gov/dataset/OSCAR_L4_OC_FINAL_V2.0
- GLORYS12: https://data.marine.copernicus.eu/product/GLOBAL_MULTIYEAR_PHY_001_030/description
- Copernicus winds: https://data.marine.copernicus.eu/product/WIND_GLO_PHY_L3_MY_012_005/description
- INCOIS data: https://incois.gov.in/site/dataholdings.jsp
- Argo via python: https://argopy.readthedocs.io
