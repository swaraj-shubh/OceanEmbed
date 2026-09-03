# OceanEmbed — SIH 2026 (PS 26066)

Satellite Embedding-Based Deep Learning Framework for Reconstruction of Subsurface Ocean Temperature
*Ministry of Earth Sciences · INCOIS · Space Technology*

Reconstructs ocean temperature at **15 depth levels (0–1000 m)** from **7 satellite surface fields**
(SST, SSS, SSH/SLA, currents U/V, winds U/V) over the **Arabian Sea + Bay of Bengal**,
at **0.25° daily** resolution — validated against independent Argo profiles.

**📖 Full documentation site: https://REPLACE_ME.github.io/oceanembed-sih26/**

| Doc | Contents |
|---|---|
| [01 Problem Statement](docs/01-problem-statement.md) | Official PS, interpretation, deliverables |
| [02 Research Review](docs/02-research-review.md) | Literature, design justification, novelty |
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

# every module carries its own self-check -- these should all print "OK"
for f in metrics datasets baselines argo_eval preprocess/regrid models/unet; do python src/$f.py; done

# downloads that need no account
python src/download/oisst.py                 # SST, 2832 daily files, resumable
python src/download/argo.py                  # raw Argo profiles for validation track B2
python src/preprocess/build_store.py --check-sst   # verify the SST channel

# once the CMEMS / Earthdata sources are in place
python src/preprocess/build_store.py         # -> data/processed/nio_daily.zarr
python src/train.py configs/m2_unet.yaml     # resumable; logs to results/<run>.csv
python src/evaluate.py --model climatology --split test   # M0 reference
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

Team working agreement: [`CLAUDE.md`](CLAUDE.md)
