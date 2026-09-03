# OceanEmbed — SIH 2026 (PS 26066)

Satellite Embedding-Based Deep Learning Framework for Reconstruction of Subsurface Ocean Temperature
*Ministry of Earth Sciences · INCOIS · Space Technology*

Reconstructs ocean temperature at **15 depth levels (0–1000 m)** from **7 satellite surface fields**
(SST, SSS, SSH/SLA, currents U/V, winds U/V) over the **Arabian Sea + Bay of Bengal**,
at **0.25° daily** resolution — validated against independent Argo profiles.

[![self-checks](https://github.com/swaraj-shubh/OceanEmbed/actions/workflows/selfchecks.yml/badge.svg)](https://github.com/swaraj-shubh/OceanEmbed/actions/workflows/selfchecks.yml)
[![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**📖 Documentation site: https://oceanembed-sih26.vercel.app**

---

## Result

**0.786 °C** blended RMSE against **6,056 independent Argo casts** in 2023–24, years the
model never saw.

| | Blended RMSE |
|---|---|
| M0 monthly climatology (baseline) | 1.160 |
| **OceanEmbed (final)** | **0.786** |
| GLORYS12V1 reanalysis — the training target itself | 0.728 |

**32% better than climatology, and better than climatology at every one of the 15 depths.**
The error the model adds on top of its own training target is 0.058 °C.

The final system is a **six-member ensemble** (3 seeds of the M4 ConvLSTM + 3 seeds of the
same network with a depth-weighted loss) followed by a **depth-wise bias correction** fitted
on validation-year Argo. Both extra stages are post-processing; Argo is never a model input
or a training target. Manifest: [`results/FROZEN.md`](results/FROZEN.md).

The honest part, and the actual research finding: **seven model-side interventions were
tried — attention, ConvLSTM, gradient loss, anomaly formulation, climatology-as-input,
auxiliary channels, depth weighting — and none improved the observational score on its own.**
The two things that worked act on the output, not the model. Full ablation including every
failure: [doc 10](docs/10-experiment-programme.md).

---

## Documentation

| Doc | Contents |
|---|---|
| [01 Problem Statement](docs/01-problem-statement.md) | Official PS, interpretation, deliverables |
| [02 Research Review](docs/02-research-review.md) | Literature, design justification, novelty pitch |
| [03 Architecture](docs/03-architecture.md) | CNN → ConvLSTM → Attention → U-Net spec |
| [04 Data](docs/04-data.md) | Sources, access, preprocessing pipeline |
| [05 Training & Evaluation](docs/05-training-evaluation.md) | Loss, protocol, metrics, ablations |
| [06 Demo & Roadmap](docs/06-demo-and-roadmap.md) | Streamlit spec, milestones, risks |
| [07 Results & Handover](docs/07-results-and-handover.md) | Day 1: first results, infra, limitations |
| [08 Challenges Faced](docs/08-challenges.md) | Every bug that cost real time, and its actual cause |
| [09 Day 2 Results](docs/09-day2-handover.md) | Multi-seed error bars, 4 interventions, the GLORYS-bias ceiling finding. Headline superseded by 10 |
| [**10 Day 3 Results**](docs/10-day3-ensemble-and-bias-correction.md) | **Current numbers.** Best model 0.786 °C (ensemble + bias correction), full leakage audit, attention question closed |

## Running the code

```bash
pip install -r requirements.txt
```

**Status.** Full pipeline built, all 7 input channels + GLORYS12V1 target downloaded and
regridded (2015-04 → 2024-12), M0–M4 trained and validated against 5,980–6,093 independent
Argo profiles. Current best: **0.786 °C** blended RMSE (ensemble of 6 ConvLSTM models +
bias correction), vs 1.160 for climatology and 0.728 for the GLORYS target's own error
against Argo. Attention was tested twice (alone, and combined with the ConvLSTM) and
helped neither time — the final architecture is CNN encoder → ConvLSTM → U-Net decoder,
no attention. Full detail: [doc 09](docs/09-day2-handover.md) and
[doc 10](docs/10-day3-ensemble-and-bias-correction.md). Streamlit demo written
(`app/streamlit_app.py`), INCOIS LAS gridded Argo track (B1) blocked on an upstream
service outage — see doc 10 §8 for both.

```bash
for f in metrics datasets baselines argo_eval bias_correct ablation models/unet; do
    python src/$f.py
done
python src/audit_leakage.py          # 8/8 methodology assertions
```

### Data

Downloads are resumable and safe to re-run. SST and Argo need no account; the rest need a
free Copernicus Marine and NASA Earthdata account:

```bash
cp .env.example .env     # then fill in the four values; .env is gitignored
```

Credentials are only ever read by `src/download/`. **Training, evaluation and the demo need
no credentials at all** — the processed store is self-contained.

```bash
python src/download/oisst.py                 # SST
python src/download/argo.py                  # raw Argo profiles (validation track B2)
python src/download/podaac.py                # SMAP SSS, OSCAR currents
python src/download/cmems.py                 # SLA, winds, GLORYS12V1 target

python src/preprocess/build_store.py         # -> data/processed/nio_daily.zarr (3.1 GB)
python src/datasets.py --clim                # climatology + bathymetry caches
```

### Train and evaluate

```bash
python src/train.py configs/m4_convlstm.yaml --seed 1     # resumable; logs to results/<run>.csv
python src/predict_cube.py --ckpt checkpoints/m4_convlstm_s1_best.pt --split val
python src/ablation.py --split test                        # -> results/ablation_test.md
```

### Reproduce the final result

```bash
# average the six members, then subtract the val-fitted offset
python src/predict_cube.py --split test --run ens_mix6 \
    --ensemble results/m4_convlstm_s{1,2,3}_best_test_cube.nc \
               results/m4_dw_s{1,2,3}_best_test_cube.nc \
    --offset results/ens_mix6_offset.json
```

### GPU

Training runs on any 16 GB GPU; the final programme cost about 3.5 GPU-hours on one T4.
Everything else — scoring, ensembling, bias correction, bootstrapping — runs on CPU.

```bash
BUCKET=<your-bucket> bash deploy/setup.sh    # bootstraps the box AND starts the checkpoint sync
```

**The checkpoint sync is not optional.** `setup.sh` starts it for you; a day's worth of
checkpoints once existed only on one instance's root volume because it was a separate manual
command nobody ran.

---

## Layout

```
.env.example  template for the two download accounts; copy to .env
.gitattributes line-ending normalisation (Windows dev box, Linux GPU box)
docs/         01-11, published to the docs site by build_site.py
configs/      one YAML per experiment
app/          the Streamlit demo + its committed 60 MB offline data bundle
scripts/      build_demo_bundle.py -- regenerates that bundle
src/
  download/   one script per data source
  preprocess/ QC, regrid, align, build the Zarr store
  models/     unet.py -- M2 U-Net, M3 attention, M4 ConvLSTM
  train.py    single config-driven entrypoint, auto-resumes
  predict_cube.py  inference -> NetCDF cube; --ensemble and --offset
  bias_correct.py  fit/apply the depth-wise Argo offset
  argo_eval.py     profile matching, depth-wise metrics, float-blocked bootstrap
  ablation.py      builds the ablation table from results/*.csv
  audit_leakage.py runnable methodology assertions
results/      metric CSVs and ablation tables (tracked); cubes are gitignored
deploy/       EC2 bootstrap + S3 checkpoint sync
```

`data/`, `checkpoints/`, `results/*.nc` and `.env` are gitignored. Checkpoints live in
`s3://oceanembed-sih26-data/oceanembed/checkpoints/`.

CI (`.github/workflows/selfchecks.yml`) runs the seven self-checks that build their own
synthetic data, on every push. `audit_leakage.py` is not in CI because it asserts against
the real 3.1 GB store — **run it locally before publishing any number.**

---

## License

[MIT](LICENSE) for the code. The datasets carry their own terms — Copernicus Marine, NASA
PO.DAAC, NOAA and the Argo programme — see [doc 04](docs/04-data.md) for products and DOIs.

---

## Status

Modelling is complete and frozen, and **the demo is built** — `streamlit run app/streamlit_app.py`
(see [app/README.md](app/README.md)). Outstanding: validation track B1 against INCOIS
gridded Argo, and refitting the bias offset annually if this ever runs operationally.

Do not add model capacity or input channels — seven interventions now say the same thing.
