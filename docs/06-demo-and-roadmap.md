---
title: "06 · Demo & Roadmap"
nav_order: 7
---

# 06 — Demo Spec, Roadmap, Risks

## 1. Streamlit demo (what the judge does)

```mermaid<br/>flowchart LR<br/>    U["Judge"] --> UI["Streamlit app"]<br/>    UI --> S1["Sidebar:\ndate picker · depth selector ·\nregion (AS / BoB / full)"]<br/>    UI --> V1["Tab 1 — Inputs:\n7 surface fields for the date\n(Cartopy maps)"]<br/>    UI --> V2["Tab 2 — Reconstruction:\ntemperature map at chosen depth\n+ colorbar + GLORYS side-by-side toggle"]<br/>    UI --> V3["Tab 3 — Profile:\nclick map → 0–1000 m predicted profile\n+ nearest held-out Argo overlay\n+ RMSE/Bias at that point"]<br/>    UI --> V4["Tab 4 — Skill:\ndepth-wise metrics table,\nRMSE-vs-depth curve M0…M4,\nOceanEmbed embedding PCA-RGB view"]<br/>```

Implementation rules:

- **Fully offline.** Loads the frozen checkpoint + the processed Zarr + a precomputed Argo evaluation parquet. No internet at demo time (venue Wi-Fi always fails).
- Inference is a CPU forward pass (<1 s full region). Precompute nothing that a click can compute; precompute everything that needs the pipeline (all inputs/targets for the demo date range shipped in the Zarr).
- `streamlit` + `plotly` for interactive maps (`st.plotly_chart` with clickable heatmap → profile), `matplotlib+cartopy` for the pretty coastline figures in Tab 1.
- Keep a 90-second scripted demo path: pick a cyclone-season date in the Bay of Bengal → show depth-100 m map → click the eddy → profile matches Argo → skill tab. Rehearse it.

## 2. Roadmap (adjust dates to the internal round deadline)

| Week | Milestone | Definition of done |
|---|---|---|
| 1 | Environment + OceanDepths bootstrap | One patch loaded and visualized; both data accounts registered |
| 1–2 | M0 + M1 on OceanDepths | Our metrics code reproduces climatology RMSE ≈ 0.97 °C; tiny CNN trains |
| 2–3 | Own pipeline | `nio_daily.zarr` exists for 2015–2022; X/Y/M sample plots look physical |
| 3–4 | M2 (7-var U-Net) | Beats M0 on anomaly correlation at ≥ some depths |
| 4–5 | M3 (attention) + M4 (ConvLSTM) | Ablation rows filled; M4 = best model |
| 5 | Argo Track-B evaluation | Headline depth-wise table vs independent Argo |
| 5–6 | Streamlit demo + freeze | Offline demo runs from checkout on a clean machine |
| 6 | Deck + rehearsal | 90-second demo path timed; Q&A prep from docs 01–05 |

Parallelize: one member owns pipeline (doc 04), one owns model/training (docs 03/05), one owns evaluation+Argo, one owns demo+deck. The Zarr contract (`X=[7,96,176]`, `Y=[15,96,176]`, masks) is the interface — agree on it first, then work independently.

## 3. Risks & mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Copernicus/PO.DAAC download slowness or quota | High | Start downloads week 1, year-by-year, process-and-delete; OceanDepths as fallback data |
| Model doesn't beat climatology | Medium | Expected at raw-RMSE for M1–M2 (doc 02 §2); the claim to win is anomaly correlation + thermocline-band RMSE; if even M4 fails, present honest ablation — methodological rigor still scores |
| Kaggle GPU quota binds late | Medium | Checkpoint/resume everywhere; ₹5–10k RunPod/AWS-Spot escape hatch |
| SMAP SSS gaps/coastal noise | Certain | Missing masks in loss; document as limitation, spin as "handling real observational sparsity" |
| Team pipeline/model integration breaks late | Medium | Freeze the Zarr tensor contract week 2; integration test = one end-to-end train step in CI-style script |
| Demo-day machine issues | Medium | Demo runs CPU-only from a `requirements.txt` + one `streamlit run` command; test on a second laptop; screen-record a backup video |

## 4. Q&A ammunition (one-liners)

- *Why not ViT/GNN (the PS mentions them)?* — Published head-to-head (RS 17:2005): Transformer loses to CNN at this data scale; attention is included where it demonstrably helps — as fusion inside the CNN. GNNs suit irregular point data; our inputs are regular grids.
- *Why 7 variables?* — Bay of Bengal barrier layer: salinity stratification decouples SST from subsurface heat, so SSS/currents/winds carry real signal here; ablation SST-only vs 7-var quantifies it.
- *Isn't GLORYS circular with Argo?* — Partially; that's exactly why we hold out Argo and report anomaly correlation. Same protocol as OceanBench/NeurIPS benchmarks.
- *Operational use?* — All inputs are near-real-time products; the same weights run daily in <1 s CPU — a plausible INCOIS service component (cyclone heat content, PFZ support).
- *Scale beyond the region?* — Fully convolutional model: retrain/fine-tune on other basins; transfer learning precedent in ESSD 3D-U-Net++ paper.
