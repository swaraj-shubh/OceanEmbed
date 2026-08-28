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

**Status.** SST (2015-04 -> 2022-12) downloaded and verified. The other six input channels
and the GLORYS12V1 target need a free Copernicus Marine account and a NASA Earthdata
account; `build_store.py` names the exact product for each missing channel.

Team working agreement: [`CLAUDE.md`](CLAUDE.md)
