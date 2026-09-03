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
| [**07 — Results & Handover**](07-results-and-handover.md) | **What was actually built and measured**: final depth-wise accuracy vs independent Argo, the three findings, the bugs that would have corrupted results, infrastructure, limitations, next steps |
| [**08 — Challenges Faced**](08-challenges.md) | Every problem that cost real time: symptom, actual cause, fix. Download failures, silent data corruption, the OOM misdiagnosed as a scheduler kill, and the result that turned out to be noise |
| [09 — Day 2 · Results](09-day2-handover.md) | Multi-seed error bars, four interventions measured (attention, gradient loss, anomaly, ConvLSTM), and the finding that the training target itself carries a +0.72 °C thermocline bias — a measured ceiling of 0.728 °C. Headline superseded by doc 11. |
| [10 — Experiment Programme](10-experiment-programme.md) | The 10-step programme that took M4 from 0.890 to 0.786: six interventions under one selection discipline, selection on val Argo, test touched exactly once |
| [**11 — Day 3 · Handover**](11-day3-handover.md) | **The result.** 0.786 °C from a 6-member ensemble plus a depth-wise Argo bias correction — and why almost none of the gain came from training anything |
| [**12 — Day 4 · Audit**](12-day4-audit.md) | Three directions closed: six alternative correction forms (all null), per-depth ensemble weights (null), and an error budget showing 62% of what remains sits in 75–150 m as *variance, not bias* — beyond the reach of post-hoc correction |
| [**13 — Day 5 · Track B1**](13-track-b1.md) | **Current page.** The PS-named INCOIS gridded Argo validation, now closed: 1.232 °C — and why B1 ranks the same four systems *backwards* from B2, which is a fact about the reference product's smoothing, not about the models |

**The pitch in one sentence:** a CNN embedding of 7 satellite surface fields, fed through a ConvLSTM and ensembled with a post-hoc bias correction, reconstructs temperature at 15 depths to **0.786 °C — 32% better than climatology** — validated depth-wise against 6,056 *independent* Argo casts in years the model never saw, with the training reanalysis' own +0.72 °C thermocline bias measured against those same observations.

Attention and temporal context were both built and measured, alone and combined. Temporal context (ConvLSTM) is in the final architecture; attention is not — it never improved the independent-observation score, and once combined with the ConvLSTM it measurably hurt it. That result, not a hidden one, is what closes the architecture question ([doc 11](11-day3-handover.md)).
