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

Team working agreement: [`CLAUDE.md`](CLAUDE.md)
