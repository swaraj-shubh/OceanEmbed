---
title: "Home"
nav_order: 1
---

# OceanEmbed — Documentation Index (SIH PS 26066)

Reconstruction of subsurface ocean temperature (15 depths, 0–1000 m) from 7 surface satellite fields over the Arabian Sea + Bay of Bengal. Team constitution: [`../CLAUDE.md`](../CLAUDE.md).

| Doc | Contents |
|---|---|
| [01 — Problem Statement](01-problem-statement.md) | Official PS 26066 text (verified), interpretation, judging criteria, deliverables checklist |
| [02 — Research Review](02-research-review.md) | Literature landscape, the 3 findings that shaped the design, novelty pitch, citations |
| [03 — Architecture](03-architecture.md) | End-to-end system + full model spec (CNN → ConvLSTM → Attention → U-Net), tensor shapes, M0–M4 stages |
| [04 — Data](04-data.md) | Exact products, access/download commands, region/period/splits, preprocessing pipeline, BoB gotchas |
| [05 — Training & Evaluation](05-training-evaluation.md) | Loss, recipe, GLORYS + independent-Argo protocol, metrics, ablations, compute plan |
| [06 — Demo & Roadmap](06-demo-and-roadmap.md) | Streamlit demo spec, weekly milestones, risk register, Q&A ammunition |

**The pitch in one sentence:** an attention-fused, temporally-aware CNN embedding of 7 satellite surface fields, trained on GLORYS12 and validated depth-wise against independent Argo observations — the first such multivariate system focused on the North Indian Ocean.
