<div align="center">

# OTER — Ocean Thermal Embedding Reconstruction

**Satellite Embedding-Based Deep Learning Framework for Reconstruction of Subsurface Ocean Temperature**

**Smart India Hackathon 2026 · Problem Statement ID: 26066**
**Ministry of Earth Sciences · INCOIS · Space Technology**

[![Live Demo](https://img.shields.io/badge/Live_Demo-oter.shubhh.xyz-1b4f72)](https://oter.shubhh.xyz)
[![Documentation](https://img.shields.io/badge/Docs-oceanembed--sih26.vercel.app-2a78d6)](https://oceanembed-sih26.vercel.app)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Reconstructs ocean temperature at **15 depths (0–1000 m)** from **7 satellite surface fields** over the Arabian Sea and Bay of Bengal, at **0.25° daily** resolution.

</div>

---

## Problem & Approach

Argo floats measure subsurface temperature but cover ~0.01% of the ocean per depth level. Satellites see the surface (SST, SSS, SSH, currents, winds) continuously and at high resolution, and that surface state carries an indirect signature of what's underneath.

**OTER** learns that surface → subsurface mapping:

1. Harmonizes 7 surface variables from 5 public products onto a common 0.25° daily grid
2. **CNN encoder → ConvLSTM (7-day window) → U-Net decoder** reconstructs all 15 depths at once
3. A 6-member ensemble is averaged and corrected with a depth-wise bias offset fit on 2022 Argo
4. Validated against **held-out Argo profiles never used in training**

```
7 surface fields  →  CNN encoder  →  ConvLSTM (7-day)  →  U-Net decoder  →  15 depth maps
                                                                                  │
                                                    Argo-fitted bias correction ◀─┘
```

---

## Results

Validated against **6,056 held-out Argo casts (2023–24)**:

| Method | Blended RMSE (↓) |
|---|---|
| M0 Monthly Climatology (baseline) | 1.160 °C |
| **OTER (ours)** | **0.786 °C** |
| GLORYS12V1 reanalysis (training target) | 0.728 °C |

- 32% improvement over climatology, at **every one of the 15 depths**
- Beats the GLORYS reanalysis it was trained on at 700 m and 1000 m, by correcting its known cold bias at depth
- Second, independent validation track (INCOIS LAS Gridded Argo): **1.232 °C** vs 1.278 °C climatology and 1.437 °C GLORYS

Full experiment log, including negative results (attention, gradient loss): [`docs/10-experiment-programme.md`](docs/10-experiment-programme.md).

---

## Key Features

- Satellite-only inference — no in-situ data needed at prediction time
- ~0.5 s per reconstruction on a 4-core CPU, 6.65M parameters, no GPU required
- Interactive Streamlit demo — pick a date/depth, click the map for a 0–1000 m profile with the nearest Argo cast overlaid
- Reproducible: time-based train/val/test splits, leakage audit, float-blocked bootstrap for error bars

---

## Quick Start

**Live demo:** <https://oter.shubhh.xyz>

**Run the demo locally** (no GPU, no network, no credentials):
```bash
pip install -r app/requirements.txt
streamlit run app/streamlit_app.py
```

**Reproduce end-to-end** (needs free CMEMS + NASA Earthdata accounts):
```bash
pip install -r requirements.txt
cp .env.example .env

python src/download/oisst.py && python src/download/argo.py \
  && python src/download/podaac.py && python src/download/cmems.py
python src/preprocess/build_store.py

python src/train.py configs/m4_convlstm.yaml --seed 1
python src/predict_cube.py --split test --run ens_mix6 \
    --ensemble results/m4_convlstm_s{1,2,3}_best_test_cube.nc \
               results/m4_dw_s{1,2,3}_best_test_cube.nc \
    --offset results/ens_mix6_offset.json
python src/audit_leakage.py
```

---

## Repository Structure

```
configs/        YAML configs (one per experiment)
docs/           Methodology, experiments, handover notes
app/            Streamlit demo + precomputed offline bundle
src/
  download/     one script per data source
  preprocess/   QC, regrid, align → Zarr store
  models/       U-Net, attention, ConvLSTM
  train.py, predict_cube.py, bias_correct.py, argo_eval.py, incois_eval.py
  ablation.py, audit_leakage.py
results/        Metric CSVs and ablation tables
deploy/         demo_ec2.sh (hosting), setup.sh (training)
```

---

## Datasets

| Variable | Product | Resolution · Cadence |
|---|---|---|
| SST | NOAA OISST v2.1 | 0.25° · daily |
| SSS | SMAP RSS L3 V6 | 0.25° · 8-day |
| SSH / SLA | Copernicus DUACS L4 | 0.125° · daily |
| Currents (U, V) | NASA OSCAR v2.0 | 0.25° · daily |
| Winds (U, V) | Copernicus ASCAT L3 | 0.25° · daily |

**Target:** GLORYS12V1 reanalysis (`doi:10.48670/moi-00021`). **Validation:** raw Argo profiles (Ifremer ERDDAP) and INCOIS LAS Gridded Argo — both PS-named, neither ever used as a training input or target.

---

## Documentation

Full methodology, docs site: **<https://oceanembed-sih26.vercel.app>**

- [Problem Statement Interpretation](docs/01-problem-statement.md)
- [Architecture](docs/03-architecture.md)
- [Data Pipeline](docs/04-data.md)
- [Experiment Programme & Final Results](docs/10-experiment-programme.md)

---

## Acknowledgements

Ministry of Earth Sciences (MoES) and INCOIS for the problem statement · Copernicus Marine Service, NASA PO.DAAC, NOAA NCEI, Ifremer Argo for open data · Smart India Hackathon 2026 organisers.

## License

[MIT](LICENSE) for the code. Datasets retain their original terms (Copernicus Marine, NASA PO.DAAC, NOAA, Argo programme).
