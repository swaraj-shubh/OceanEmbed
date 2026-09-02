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
| [**09 — Day 2 · Latest Results**](09-day2-handover.md) | **Current results.** Multi-seed error bars, four interventions measured (attention, gradient loss, anomaly, ConvLSTM), and the finding that the training target itself carries a +0.72 °C thermocline bias — a measured ceiling of 0.728 °C |
| [**10 — Experiment Programme & Results**](10-experiment-programme.md) | **Current best results.** Eleven tasks with a fixed selection discipline (screen on *val*, read test once). A seed ensemble plus an Argo bias correction take **0.890 → 0.786 °C, almost all of it with no training at all**, beating every architecture change in the project; **all 15 depths now beat climatology**; and a float-blocked bootstrap settles which differences are real |

**The pitch in one sentence:** a CNN embedding of 7 satellite surface fields reconstructs temperature at 15 depths to **0.786 °C — 32% better than climatology and better at every single depth** — validated depth-wise against ~6,000 *independent* Argo profiles in years the model never saw, closing to within **0.058 °C of the training reanalysis' own accuracy** after correcting the +0.72 °C thermocline bias we measured in it.

Attention and temporal context were both built and measured; neither moved the observational score by more than one standard deviation, and the attention result's bootstrap interval contains zero. That is reported rather than hidden ([doc 09](09-day2-handover.md), [doc 10](10-experiment-programme.md)) — and it is what pointed at the data-side fixes that did work.
