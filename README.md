<div align="center">

# OTER — Ocean Thermal Embedding Reconstruction

**Satellite Embedding-Based Deep Learning Framework for Reconstruction of Subsurface Ocean Temperature**

Smart India Hackathon 2026 · PS 26066 · Ministry of Earth Sciences · INCOIS · Space Technology

[![live demo](https://img.shields.io/badge/live_demo-oter.shubhh.xyz-1b4f72)](https://oter.shubhh.xyz)
[![docs](https://img.shields.io/badge/docs-oceanembed--sih26.vercel.app-2a78d6)](https://oceanembed-sih26.vercel.app)
[![self-checks](https://github.com/swaraj-shubh/OceanEmbed/actions/workflows/selfchecks.yml/badge.svg)](https://github.com/swaraj-shubh/OceanEmbed/actions/workflows/selfchecks.yml)
[![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**Reconstructs ocean temperature at 15 depth levels (0–1000 m) from 7 satellite surface fields**<br>
Arabian Sea + Bay of Bengal · 0.25° daily · validated against held-out Argo floats

</div>

---

## Try it

| | |
|---|---|
| 🌊 **Live demo** | **<https://oter.shubhh.xyz>** · mirror <https://oter.swarajz.xyz> · <https://65.2.207.204> |
| 📖 **Documentation site** | **<https://oceanembed-sih26.vercel.app>** |
| 💻 **Run it offline** | `streamlit run app/streamlit_app.py` — no GPU, no network, no credentials |

The demo covers the **full test split — 725 days, 2023-01-07 → 2024-12-31** — every day the
model can predict and never a day it trained on. Pick a date and depth, get the
reconstructed field; click the map, get the 0–1000 m profile with the nearest Argo cast
overlaid and the local error. Predictions are precomputed, so a click is an array lookup.

> **90-second path:** 5 Dec 2023 (Cyclone Michaung, Bay of Bengal) → tab ② at 100 m →
> *Difference* → tab ③, click into the Bay → tab ④ for the depth curve.

---

## Result

**0.786 °C** blended RMSE against **6,056 held-out Argo casts** in 2023–24.

| | Blended RMSE ↓ |
|---|---|
| M0 monthly climatology — baseline | 1.160 |
| **OTER (final)** | **0.786** |
| GLORYS12V1 reanalysis — *the training target itself* | 0.728 |

**32% better than climatology, and better at every one of the 15 depths.** That last part is
not free: a single U-Net *loses* to a monthly mean below 500 m, and a single ConvLSTM loses
below 700 m. Clearing climatology everywhere is a property of the final corrected ensemble.

**At 700 m and 1000 m the system beats the reanalysis it was trained on:**

| Depth | OTER | GLORYS12V1 | Climatology |
|---|---|---|---|
| 700 m | **0.224** | 0.246 | 0.263 |
| 1000 m | **0.217** | 0.243 | 0.237 |

The mechanism is explicit rather than mysterious: GLORYS carries a cold bias at depth
(−0.091 °C at 700 m), and the depth-wise correction removes it. Correlation is unchanged
(0.981 both) — the model does not out-*resolve* the reanalysis, it out-*calibrates* it
against observations.

**Second reference product, track B1** ([doc 13](docs/13-track-b1.md)): scored against
INCOIS LAS gridded Argo, **1.232** °C versus 1.278 for climatology and 1.437 for GLORYS.
B1 ranks systems differently from B2 because a 1°/10-day objective analysis penalises any
field sharper than itself — B2 is the reported track, and B1's value is that the correction
fitted on raw Argo **transfers** to an independent product.

---

## The actual research finding

Seven model-side interventions were tried — attention alone, attention fused with the
ConvLSTM, gradient loss, anomaly targets, climatology-as-input, auxiliary channels, depth
weighting. **None improved the observational score, and attention made it worse.**

The two things that worked act on the *output*:

| Step | Blended RMSE | Gain |
|---|---|---|
| Single M4 ConvLSTM (mean of 3 seeds) | 0.890 | — |
| 6-member ensemble | 0.859 | 0.031 |
| **+ 15-constant depth-wise correction** | **0.786** | **0.073** |

**Fifteen numbers deliver 2.4× the gain of six neural networks.** A plain U-Net with the
correction (0.844) beats a six-model ConvLSTM ensemble without it (0.859).

Why: the bias lives in the *labels*. GLORYS is the training target, so its offset is
exactly what the network is rewarded for reproducing — no amount of capacity recovers
information the target does not contain. Only a new source of information does, and that
is Argo. Full ablation, including every failure:
[doc 10](docs/10-experiment-programme.md).

> **Do not add model capacity or input channels.** Seven interventions now say the same thing.

---

## Method

```
7 surface fields [7,96,176]  ──▶  CNN encoder  ──▶  ConvLSTM (7-day window)
                                                          │  ← the OceanEmbed latent
                                  U-Net decoder  ◀────────┘
                                        │
                              15 depths [15,96,176]
                                        │
                     6-member ensemble ─┴─ depth-wise Argo bias correction
```

No attention: tested alone (M3) and fused with the ConvLSTM (M4+attn), and it helped
neither time — in the second case it improved validation RMSE while making the Argo score
*worse*, which is why every claim in this repo is reported against Argo and never against
validation loss.

**Footprint.** 6.65 M parameters, **26.6 MB** of fp32 weights. A full basin-wide 3D
reconstruction (96×176 × 15 depths) takes **~0.5 s on a 4-core laptop CPU**; the shipped
6-model ensemble takes ~3 s, and its members are independent so they parallelise trivially.
**No GPU is required at inference.** The whole training programme cost ~3.5 GPU-hours on
one T4.

---

## Data

Seven variables, five products, three agencies — all public, all operational, all subset
**server-side** to the basin, which is why a decade of inputs is tens of MB per year.

| Variable | Product | Res · cadence | Access |
|---|---|---|---|
| SST | [NOAA OISST v2.1](https://www.ncei.noaa.gov/products/optimum-interpolation-sst) | 0.25° · daily | THREDDS OPeNDAP |
| SSS | [SMAP RSS L3 V6](https://podaac.jpl.nasa.gov/dataset/SMAP_RSS_L3_SSS_SMI_8DAY-RUNNINGMEAN_V6) | 0.25° · 8-day composite | CMR REST → OPeNDAP |
| SSH / SLA | [Copernicus DUACS L4](https://data.marine.copernicus.eu/product/SEALEVEL_GLO_PHY_L4_MY_008_047) | 0.125° · daily | `copernicusmarine` |
| Currents U, V | [NASA OSCAR v2.0](https://podaac.jpl.nasa.gov/dataset/OSCAR_L4_OC_FINAL_V2.0) | 0.25° · daily, 0–30 m | CMR REST → OPeNDAP |
| Winds U, V | [Copernicus ASCAT L3](https://data.marine.copernicus.eu/product/WIND_GLO_PHY_L3_MY_012_005) | 0.25° · daily, 4 swaths merged | `copernicusmarine` |
| **Target** | [**GLORYS12V1**](https://data.marine.copernicus.eu/product/GLOBAL_MULTIYEAR_PHY_001_030) `doi:10.48670/moi-00021` | 1/12° · daily · 50 levels | *named by the PS* |
| **Validation B2** | [Argo profiles](https://erddap.ifremer.fr/erddap/tabledap/ArgoFloats.html) via Ifremer ERDDAP | point casts | *never a model input* |
| **Validation B1** | [INCOIS LAS Gridded ARGO](https://incois.gov.in/site/dataholdings.jsp) | 1° · 10-day | *named by the PS* |

SSS is the only non-daily input. GLORYS is pulled to 1100 m, not 1000, because its native
levels are 902.3 and 1062.4 m — a 1000 m ceiling would force the headline depth to be
extrapolated. Full detail: [doc 04](docs/04-data.md).

---

## Methodology, and how to check it

```bash
python src/audit_leakage.py          # 8/8 assertions, against the real store
```

Splits are **train 2015–21 / val 2022 / test 2023–24**, ordered and non-overlapping, with
no 7-day sample window crossing a boundary and normalisation statistics fitted on the train
split alone. Argo never appears in the training store in any form.

Two things the audit states precisely, because they are what a reviewer will probe:

- **GLORYS12V1 assimilates Argo**, so Argo is not statistically independent of the target in
  general. The defensible claim is narrower and true: the model trains on GLORYS 2015–21
  only, so no 2023–24 cast — nor the GLORYS state it informed — was ever seen in training.
  We say **held-out**, not *independent*.
- **Effective sample size is 147 floats, not 6,448 profiles.** A profile-level bootstrap
  reads ~6.6× too narrow, so model comparisons use `argo_eval.paired_bootstrap`, blocked by
  float.

The correction constants are fitted on **2022** Argo and applied unchanged to 2023–24. Argo
is never a model input or a training target, in either validation track.

---

## Documentation

| Doc | Contents |
|---|---|
| [01 Problem Statement](docs/01-problem-statement.md) | Official PS, interpretation, deliverables |
| [02 Research Review](docs/02-research-review.md) | Literature, design justification, novelty pitch |
| [03 Architecture](docs/03-architecture.md) | CNN → ConvLSTM → U-Net spec |
| [04 Data](docs/04-data.md) | Sources, access, preprocessing pipeline |
| [05 Training & Evaluation](docs/05-training-evaluation.md) | Loss, protocol, metrics, ablations |
| [06 Demo & Roadmap](docs/06-demo-and-roadmap.md) | Demo spec, milestones, risks, hosting |
| [07 Results & Handover](docs/07-results-and-handover.md) | Day 1: first results, infra, limitations |
| [08 Challenges Faced](docs/08-challenges.md) | Every bug that cost real time, and its actual cause |
| [09 Day 2 Results](docs/09-day2-handover.md) | Multi-seed error bars, the GLORYS-bias ceiling finding |
| [10 Experiment Programme](docs/10-experiment-programme.md) | Every ablation, including the failures |
| [**11 Day 3 Handover**](docs/11-day3-handover.md) | **The result.** 0.786 °C |
| [12 Day 4 Audit](docs/12-day4-audit.md) | Three directions closed; where the remaining error lives |
| [**13 Track B1**](docs/13-track-b1.md) | **INCOIS gridded Argo — the second PS-named reference** |

---

## Running the code

```bash
pip install -r requirements.txt
for f in metrics datasets baselines argo_eval bias_correct ablation models/unet; do
    python src/$f.py                 # each module self-checks on synthetic data
done
```

### Data

Downloads are resumable and safe to re-run. SST and Argo need no account; the rest need a
free Copernicus Marine and NASA Earthdata login. **Training, evaluation and the demo need
no credentials at all** — the processed store is self-contained.

```bash
cp .env.example .env                 # fill in four values; .env is gitignored

python src/download/oisst.py         # SST
python src/download/argo.py          # raw Argo profiles (track B2)
python src/download/podaac.py        # SMAP SSS, OSCAR currents
python src/download/cmems.py         # SLA, winds, GLORYS12V1 target

python src/preprocess/build_store.py # -> data/processed/nio_daily.zarr (3.1 GB)
python src/datasets.py --clim        # climatology + bathymetry caches
```

### Train and evaluate

```bash
python src/train.py configs/m4_convlstm.yaml --seed 1     # resumable; logs to results/
python src/predict_cube.py --ckpt checkpoints/m4_convlstm_s1_best.pt --split val
python src/ablation.py --split test                       # -> results/ablation_test.md
```

### Reproduce the headline number

```bash
python src/predict_cube.py --split test --run ens_mix6 \
    --ensemble results/m4_convlstm_s{1,2,3}_best_test_cube.nc \
               results/m4_dw_s{1,2,3}_best_test_cube.nc \
    --offset results/ens_mix6_offset.json
```

Manifest of exactly what "final" means: [`results/FROZEN.md`](results/FROZEN.md).

### GPU

Training runs on any 16 GB GPU. Everything else — scoring, ensembling, bias correction,
bootstrapping — runs on CPU.

```bash
BUCKET=<your-bucket> bash deploy/setup.sh    # bootstraps the box AND starts checkpoint sync
```

**The checkpoint sync is not optional.** `setup.sh` starts it; a day's worth of checkpoints
once existed only on one instance's root volume because it was a separate manual command
nobody ran.

---

## Deploying the demo

The hosted demo runs on a single **t3.medium** in `ap-south-1` behind nginx, with Let's
Encrypt certificates for both domains and for the bare IP. One idempotent script does the
whole thing:

```bash
scp -i key.pem deploy/demo_ec2.sh ubuntu@<ip>:
ssh -i key.pem ubuntu@<ip> 'sudo bash demo_ec2.sh'
```

Shallow clone, venv from `app/requirements.txt` (**not** the root one — that pulls torch and
would time the build out), a systemd unit so the app survives a crash or reboot, a 2 GB
swapfile, and an nginx front end that emits one TLS server block per certificate found under
`/etc/letsencrypt/live/`. Adding a name is `certbot certonly` plus a re-run — there is
nothing in the script to edit. Re-running redeploys the tip of `main`.

Streamlit binds `127.0.0.1`, so it is never publicly exposed; open 80 and 443 only. The full
recipe, including HTTPS on a bare IP (possible since Let's Encrypt began issuing IP
certificates in January 2026, under the 6-day `shortlived` profile), is in
[`app/README.md`](app/README.md).

**Memory.** The demo bundle is 478 MB across 8 quarterly chunks; the loader keeps at most
two resident, so memory plateaus at **~1.1 GB** however long a judge browses. An unbounded
cache would reach 3.2 GB and exceed every free host's ceiling.

---

## Layout

```
.env.example    template for the two download accounts
.gitattributes  line-ending normalisation (Windows dev box, Linux GPU box)
docs/           01-13, published to the docs site by build_site.py
configs/        one YAML per experiment
app/            the Streamlit demo + its committed 478 MB offline bundle
scripts/        build_demo_bundle.py -- regenerates that bundle
src/
  download/     one script per data source
  preprocess/   QC, regrid, align, build the Zarr store
  models/       unet.py -- M2 U-Net, M3 attention, M4 ConvLSTM
  train.py      single config-driven entrypoint, auto-resumes
  predict_cube.py   inference -> NetCDF cube; --ensemble and --offset
  bias_correct.py   fit/apply the depth-wise Argo offset
  argo_eval.py      profile matching, depth-wise metrics, float-blocked bootstrap
  incois_eval.py    track B1 against INCOIS gridded Argo
  ablation.py       builds the ablation table from results/*.csv
  audit_leakage.py  runnable methodology assertions
results/        metric CSVs and ablation tables (tracked); cubes are gitignored
deploy/         demo_ec2.sh (hosting) + setup.sh, sync_checkpoints.sh (training)
```

`data/`, `checkpoints/`, `results/*.nc` and `.env` are gitignored. Checkpoints live in
`s3://oceanembed-sih26-data/oceanembed/checkpoints/`.

CI (`.github/workflows/selfchecks.yml`) runs the self-checks that build their own synthetic
data, on every push. `audit_leakage.py` is not in CI because it asserts against the real
3.1 GB store — **run it locally before publishing any number.**

---

## Status

Modelling is **complete and frozen**. Both PS-named validation tracks are closed: B2 raw
Argo (the reported number) and B1 INCOIS gridded Argo ([doc 13](docs/13-track-b1.md)). The
demo is built, hosted and reachable at three names over HTTPS.

Open, and honestly small: refitting the bias offset annually if this ever runs
operationally, and the untested question of whether monthly or spatially-varying offsets
beat the current one flat constant per depth.

---

## License

[MIT](LICENSE) for the code. The datasets carry their own terms — Copernicus Marine, NASA
PO.DAAC, NOAA and the Argo programme — see [doc 04](docs/04-data.md) for products and DOIs.
