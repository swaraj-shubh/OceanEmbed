<div align="center">

# OTER — Ocean Thermal Embedding Reconstruction

**Satellite Embedding-Based Deep Learning Framework for Reconstruction of Subsurface Ocean Temperature**

**Smart India Hackathon 2026 · Problem Statement ID: 26066**
**Ministry of Earth Sciences · INCOIS · Space Technology**

[![Live Demo](https://img.shields.io/badge/Live_Demo-oter.shubhh.xyz-1b4f72)](https://oter.shubhh.xyz)
[![Documentation](https://img.shields.io/badge/Docs-oceanembed--sih26.vercel.app-2a78d6)](https://oceanembed-sih26.vercel.app)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Reconstructs ocean temperature at **15 standard depth levels (0–1000 m)** from **7 satellite surface fields** over the Arabian Sea and Bay of Bengal, at **0.25° daily** resolution.

</div>

---

## 📌 Problem Statement

Subsurface ocean temperature is critical for ocean circulation, heat content, stratification, marine heatwaves, fisheries, and data assimilation. However, direct measurements from Argo floats, buoys and gliders are spatially and temporally sparse — insufficient for continuous basin-scale fields.

Satellite observations, in contrast, provide continuous surface monitoring. Surface variables (SST, SSS, SSH/SLA, currents, winds) carry indirect signatures of subsurface structure through thermocline displacement, mesoscale eddies, vertical mixing and air–sea coupling.

**Goal:** Reconstruct 3D subsurface temperature using *only* surface satellite observations.

---

## 💡 Solution

We built **OTER**, an end-to-end deep learning framework that:

1. **Harmonizes** 7 surface satellite variables from 5 public products onto a common 0.25° daily grid.
2. **Learns compact satellite embeddings** via a CNN encoder → ConvLSTM (7-day window) → U-Net decoder.
3. **Reconstructs** temperature at 15 standard depths (0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000 m).
4. **Corrects residual bias** using depth-wise offsets fitted on held-out Argo observations.
5. **Validates** against independent Argo profiles from 2023–24.

```
[7 vars × 7-day window, 96×176]
        │
        ▼
CNN encoder — spatial features per day
7 → 32 → 64 → 128 → 256 ch · 96×176 → 48×88 → 24×44 → 12×22
        │
        ▼
ConvLSTM — evolution across the 7-day window
7 recurrent steps → 256×12×22 latent
        │
        ▼
U-Net decoder — skip-connected to encoder
256 → … → 15 ch · 12×22 → 96×176
        │
        ▼
15 depth-wise temperature maps [15, 96, 176]
        │
        ▼
6-member ensemble — 3× M4-MSE + 3× M4-DW seeds, averaged
        │
        ▼
Depth-wise bias correction — offset fit on 2022 Argo, frozen
        │
        ▼
Final prediction (2023–24)  ──▶  Argo evaluation (RMSE · MAE · Bias · R²)
```
---

## 🏆 Results

Validated against **6,056 held-out Argo casts** (2023–24):

| Method | Blended RMSE (↓) |
|---|---|
| M0 Monthly Climatology (baseline) | 1.160 °C |
| **OTER (ours)** | **0.786 °C** |
| GLORYS12V1 reanalysis (training target) | 0.728 °C |

- **32% improvement over climatology**
- **Better than climatology at every one of the 15 depths** — including below 500 m where a single U-Net fails
- **Outperforms the reanalysis it was trained on at 700 m and 1000 m**, by correcting GLORYS's known cold bias at depth

**Second validation track** — against INCOIS LAS Gridded Argo (PS-named reference):

| Method | RMSE (↓) |
|---|---|
| Climatology | 1.278 °C |
| GLORYS12V1 | 1.437 °C |
| **OTER** | **1.232 °C** |

---

## ✨ Key Features

- **Satellite-only inference** — no in-situ data required at prediction time
- **Basin-scale 3D reconstruction** in ~0.5 s on a 4-core laptop CPU; no GPU needed
- **6.65 M parameters / 26.6 MB** fp32 weights
- **Full test split coverage** — 725 days (2023-01-07 → 2024-12-31) precomputed for instant demo lookup
- **Interactive Streamlit demo** — pick any date/depth, click the map for the 0–1000 m profile with nearest Argo cast overlaid
- **Reproducible & audited** — leakage assertions, train/val/test splits, float-blocked bootstrap for error bars

---

## 🛠️ Tech Stack

| Layer | Tools |
|---|---|
| Deep Learning | PyTorch, ConvLSTM, U-Net, CNN encoder |
| Data | Xarray, Zarr, NetCDF4, Dask |
| Downloads | Copernicus Marine (`copernicusmarine`), NASA PO.DAAC, NOAA THREDDS, Ifremer ERDDAP |
| Evaluation | NumPy, Pandas, SciPy (float-blocked bootstrap) |
| Demo | Streamlit, nginx, systemd, AWS EC2 (t3.medium, ap-south-1) |
| Docs | Python static-site generator → Vercel |
| CI | GitHub Actions (self-checks on synthetic data) |

---

## 📊 Datasets

**Inputs (7 surface variables, 5 products):**

| Variable | Product | Resolution · Cadence | Source |
|---|---|---|---|
| SST | NOAA OISST v2.1 | 0.25° · daily | NOAA NCEI |
| SSS | SMAP RSS L3 V6 | 0.25° · 8-day | NASA PO.DAAC |
| SSH / SLA | Copernicus DUACS L4 | 0.125° · daily | Copernicus Marine |
| Currents (U, V) | NASA OSCAR v2.0 | 0.25° · daily | NASA PO.DAAC |
| Winds (U, V) | Copernicus ASCAT L3 | 0.25° · daily | Copernicus Marine |

**Training target:** GLORYS12V1 reanalysis — `doi:10.48670/moi-00021` (1/12°, daily, 50 levels)

**Independent validation:**
- Argo profiles via Ifremer ERDDAP (raw casts — track B2, reported)
- INCOIS LAS Gridded Argo (PS-named — track B1)

---

## 🚀 Quick Start

### Try the live demo
👉 **<https://oter.shubhh.xyz>**

### Run locally (no GPU, no network, no credentials)
```bash
pip install -r app/requirements.txt
streamlit run app/streamlit_app.py
```

### Reproduce results end-to-end
```bash
pip install -r requirements.txt

# 1. Self-check individual modules (synthetic data)
for f in metrics datasets baselines argo_eval bias_correct ablation models/unet; do
    python src/$f.py
done

# 2. Download data (resumable; needs free CMEMS + NASA Earthdata accounts)
cp .env.example .env
python src/download/oisst.py
python src/download/argo.py
python src/download/podaac.py
python src/download/cmems.py

# 3. Build processed store (~3.1 GB Zarr)
python src/preprocess/build_store.py
python src/datasets.py --clim

# 4. Train / evaluate
python src/train.py configs/m4_convlstm.yaml --seed 1
python src/predict_cube.py --split test --run ens_mix6 \
    --ensemble results/m4_convlstm_s{1,2,3}_best_test_cube.nc \
               results/m4_dw_s{1,2,3}_best_test_cube.nc \
    --offset results/ens_mix6_offset.json
python src/ablation.py --split test

# 5. Methodology audit
python src/audit_leakage.py
```

---

## 📁 Repository Structure

```
configs/        YAML configs (one per experiment)
docs/           Detailed methodology, experiments, handover notes
app/            Streamlit demo + precomputed 478 MB offline bundle
scripts/        build_demo_bundle.py
src/
  download/     one script per data source
  preprocess/   QC, regrid, align → Zarr store
  models/       U-Net, attention, ConvLSTM
  train.py      config-driven training entrypoint
  predict_cube.py, bias_correct.py, argo_eval.py, incois_eval.py
  ablation.py, audit_leakage.py
results/        Metric CSVs and ablation tables
deploy/         demo_ec2.sh (hosting), setup.sh (training)
```

---

## 🔬 Evaluation Methodology

- **Splits:** train 2015–21 / val 2022 / test 2023–24 — ordered, non-overlapping, no 7-day window crosses a boundary
- **Normalization:** statistics fitted on train split only
- **Argo:** never a model input or training target; used only for validation and bias-offset fitting (2022)
- **Metrics:** RMSE, correlation, bias per depth, blended across depths
- **Error bars:** float-blocked paired bootstrap (effective N = 147 floats, not 6,448 profiles)
- **Leakage audit:** `python src/audit_leakage.py` — 8/8 assertions pass

**Caveat stated honestly:** GLORYS assimilates Argo, so Argo is not statistically independent in general. Our defensible claim is narrower: the model trained only on 2015–21 GLORYS, so no 2023–24 cast (or the GLORYS state it informed) was ever seen in training.

---

## 📚 Documentation

Full methodology, experiments, and design decisions:

- [Problem Statement Interpretation](docs/01-problem-statement.md)
- [Architecture](docs/03-architecture.md)
- [Data Pipeline](docs/04-data.md)
- [Training & Evaluation](docs/05-training-evaluation.md)
- [Results & Final Number](docs/11-day3-handover.md)
- [Experiment Programme (incl. failures)](docs/10-experiment-programme.md)

Published docs site: **<https://oceanembed-sih26.vercel.app>**

---

## 🙏 Acknowledgements

- **Ministry of Earth Sciences (MoES)** and **INCOIS** for the problem statement
- **Copernicus Marine Service**, **NASA PO.DAAC**, **NOAA NCEI**, **Ifremer Argo** for open data
- **Smart India Hackathon 2026** organisers

---

## 📄 License

[MIT](LICENSE) for the code. Datasets retain their original terms (Copernicus Marine, NASA PO.DAAC, NOAA, Argo programme).