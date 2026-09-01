---
title: "Challenges Faced"
nav_order: 9
---

# 08 — Challenges Faced

Every problem that cost real time, with the symptom, the actual cause, and the fix. Kept
because most of these did not announce themselves — several produced plausible-looking
output rather than an error, and the ones that did throw usually blamed the wrong thing.

Grouped by where they bit, roughly in the order they were hit.

---

## 1. Environment and tooling

**`argopy` would not install.** Python 3.14 on Windows; `aiohttp` has no cp314 wheel and
building it needs MSVC. Rather than downgrade Python for one library, the Argo fetch was
rewritten as direct Ifremer ERDDAP `tabledap` CSV queries. Fewer dependencies, and the
query is explicit about what it asks for.

**xESMF needs conda/Linux.** Every source here is ≥0.125° native, so `xarray.interp`
bilinear is adequate and portable. The method used is recorded in the Zarr attributes
rather than left implicit.

**GPG signing timed out repeatedly**, blocking commits with
`gpg: signing failed: Timeout` — pinentry launches a GUI passphrase dialog that expires.
Retrying usually worked; it cost several minutes each time and blocked one commit for an
hour.

**Bash mangled backticks in commit messages** (a message containing `` `latitude` ``
executed it as a command). Switched to `git commit -F -` with quoted heredocs.

**`python -m copernicusmarine` fails** — "is a package and cannot be directly executed".
The console script at the full path works.

---

## 2. Data access and download

**OISST would have taken ~10 hours.** Measured 0.13 MB/s pulling whole-globe daily files.
Switched to NCEI's OPeNDAP endpoint so the region is subset *server-side* — ~50 KB/day
instead of 1.6 MB.

**Threads made it slower, not faster.** netCDF4's DAP client holds the GIL, so a thread
pool measured worse than sequential. `ProcessPoolExecutor` fixed it.

**NCEI THREDDS went down for days.** 503s and read timeouts through 2023-06 while the
plain-HTTPS archive kept serving 200s. Added an archive fallback — 30× more bytes, but it
completes.

**ERDDAP: `CERTIFICATE_VERIFY_FAILED`.** Windows Python has no usable CA bundle by
default; pointed the SSL context at `certifi`.

**HTTP 400, "Invalid character found in the request target."** Tomcat rejects raw `<`,
`>`, `[`, `]` in query strings. Percent-encoded them in both the ERDDAP queries and the
DAP4 constraint expressions.

**SMAP granules were stamped with the wrong date.** The 8-day running mean is labelled
with its window *start* in CMR. Using that would have shifted the whole salinity channel
by four days. Now takes the `RangeDateTime` midpoint — the centre date.

**OSCAR returned HTTP 500 on a sliced `/lat`.** Hyrax fails when the variable's dimension
is named `latitude`. Coordinates are read once up front and carried locally; the grids are
static anyway.

**OSCAR came back shape (3, 180, 100).** A DAP4 constraint expression is *positional*, and
OSCAR's variables are ordered `(time, lon, lat)` — not a typo, genuinely lon-before-lat.
Added an explicit `dims` entry per product instead of assuming.

**Only 481 SMAP granules were found instead of ~3,500.** The OPeNDAP link is typed
"USE SERVICE API" on some granules and "GET DATA" on others. Matching on link *type*
silently dropped most of the record; matching on hostname finds all of them.

**The Argo fetch was killed twice, losing hours** — it accumulated everything in memory
and only wrote at the end. Added a per-month parquet cache so a kill costs one month.

**The SSS product had been retired.** SMAP RSS V4 stops at 2022-07-11, which would have
left the validation year half empty and nobody would have noticed until the metrics looked
odd. Switched to V6, which runs to the present.

**Both wind fallbacks were unusable.** The PS names L3 scatterometer with L4 as fallback.
The 0.25° L4 record *ends in 2009*; the 0.125° L4 is hourly, ~50 GB over this box for a
field we only want a daily mean of. Went with L3 ASCAT swaths: MetOp-A (2015–2021) spliced
with MetOp-B (2019–2024), ascending and descending, merged by nanmean.

**ASCAT swath gaps.** Raw daily coverage of cells ASCAT can see is ~55% in single-satellite
years and ~86% when both MetOps flew. A centred 3-day mean lifts every year to ~97%;
±2 days added only 0.3 points, so that is where it stops. Gaps among usable ocean cells:
39% → 2.9%.

**GLORYS "kept getting killed" — it was running out of memory.** Repeated background-job
deaths looked like the scheduler. `glorys.err` actually said
`OpenBLAS error: Memory allocation still failed after 10 retries`, and later a
`MemoryError` on a **2.73 MiB** allocation: system commit charge was 15.16 GB against a
16.15 GB limit. A monthly chunk decodes to ~1.5 GB in float64. Weekly still OOMed; 3-day
chunks (~37 MB) run clean. Several restart cycles were wasted blaming the harness.

**The GLORYS depth ceiling was too shallow.** `maximum_depth=1000` returns levels topping
out at **902.3 m** — the next level is 1062.4 m. Interpolating the 1000 m target onto that
is extrapolation, and 1000 m is a headline metric depth. Caught after one month of
downloading rather than nineteen hours.

**42 chunks re-downloaded days already on disk.** When the chunk size changed mid-download
(monthly → 3-day after the OOM), the new filenames stopped matching the old ones, so the
`dst.exists()` skip missed every time. The success counter climbed while real coverage
stood still at 648 days. Skipping is now by *date coverage* parsed from filenames, not by
filename equality.

---

## 3. Preprocessing and the store

**Reading a channel took 19 minutes.** `xr.open_dataset` costs ~0.4 s per file and there
were ~2,800 per-day files per product. Added a consolidation step to per-year NetCDFs; the
per-day files stay as the resumable download cache.

**OISST timestamps are 12:00Z**, so `.sel(time=days)` raised `KeyError` on every day. The
time coordinate is floored to the day on read.

**`ffill(limit=)` requires `bottleneck`**, which was not installed. Used
`reindex(method="ffill", tolerance=...)` instead — and the tolerance is the honest control
anyway, since it refuses to carry a value further than the product's own compositing
window.

**`xarray.interp` upcasts to float64.** Six years of 0.125° SLA asked for a 1.6 GiB
allocation and died. GLORYS at 1/12° over 36 levels would have been ~80 GB. Reads are now
dask-chunked and cast to float32.

**GLORYS decodes to float64 *before* the cast can help.** Packed variables are unpacked by
scale/offset first, so 16 days × 36 levels × 313 × 553 was a 713 MiB allocation that an
8 GB box refused. GLORYS reads use a time chunk of 2.

**Zarr v3 raced between dask writer threads** creating nested chunk directories
(`FileExistsError` on `X/c/0`). The store is written as Zarr v2 — flat chunk keys, and
better compatibility with Kaggle and older zarr installs.

**Ragged time chunks.** Concatenating sources that were chunked differently produced
uneven chunk sizes, which Zarr rejects. Rechunked to `time=1` before writing — which is
also the right layout, since the DataLoader reads random single days.

**The target silently vanished.** This was the worst one. Spatial dims were named
`y`/`x`, so the data variable `Y` and the dimension coordinate `y` are *the same directory*
in a Zarr store on a case-insensitive filesystem. The written store contained `X` and `y`
and **no target at all** — no error, no warning. It would have worked on Linux and failed
only on this machine. Dims are `lat`/`lon` now.

**The surface level was 100% NaN.** GLORYS's shallowest level is 0.494 m, so interpolating
to exactly 0 m fell outside the source range. Same class of bug as the 1000 m ceiling, at
the opposite end — and it was reintroduced *after* already having been burned by it once.
Depths are clamped into the source range rather than extrapolated, and **both ends are now
guarded by an assert**, because an all-NaN level is invisible until the metrics table comes
back empty.

**The SSS QC floor would have deleted a real signal.** The floor was first written as
25 PSU. The Ganga–Brahmaputra plume genuinely pushes northern Bay of Bengal salinity into
the teens — 15.56 PSU observed on 2018-05-28. That freshwater cap is exactly the
barrier-layer signal SSS is an input for; a 25 floor would have masked it away as bad data.
Floor is 5.0.

---

## 4. Evaluation

**`df.corr` resolved to the DataFrame *method*, not the column.** Silent wrong behaviour
rather than an error. Used `df["corr"]`.

**The U-Net overfit self-check failed on a white-noise target** (1.157 → 0.560). A
convolutional net cannot memorise white noise, and real temperature fields are smooth, so
the test was wrong, not the model. Switched to a bilinear-upsampled smooth target: 0.600 →
0.0096.

**Argo metrics came back all-NaN.** A `n >= 2` row guard in the stats table dropped rows
that had valid RMSE/MAE/bias but undefined correlation.

**Every Argo profile was rejected at 0 m.** The acceptance rule refused to interpolate
outside the observed depth range — but Argo floats surface at 2–5 dbar and never at
exactly 0, and 0 m is one of the six headline depths. The gap test is the correct gate;
refusing at the ends is not. Acceptance went 0% → 87%.

**Timezone mismatch.** ERDDAP returns tz-aware UTC; the split bounds and the cube's time
axis are naive. Pandas raises rather than guessing an offset.

**The prediction cube had to carry the *cropped* coordinates.** `crop_to_model` trims two
cells off each edge; labelling the cube with the uncropped grid would have matched every
Argo profile to a cell two rows from where it actually is — a subtle, entirely silent
error in the headline validation.

---

## 5. Modelling and method

**Attention did nothing.** M3 adds multi-head self-attention over the bottleneck
(+331k parameters). Against raw Argo it scores **0.907**, against M2's **0.901 ± 0.013**
over three seeds — 0.5 σ away. Early stopping did not change it, so it is not overfitting.
Reported as a null result, and it holds under a real error bar.

**Three further interventions also failed to beat M2 significantly.** A vertical-gradient
loss (0.918 ± 0.004) was worse and made the 100 m bias worse. An anomaly formulation
(0.975 ± 0.020) was worse overall, though it was the first thing to beat climatology below
500 m. M4 ConvLSTM (0.890 ± 0.008) was better but only by 0.80 pooled σ. Four
interventions, no significant movement — see [doc 09](09-day2-handover.html).

**The 100 m bias turned out not to be ours.** GLORYS12V1, the training target, carries a
+0.723 °C warm bias at 100 m against the same Argo profiles; the model's +0.848 is largely
that, reproduced faithfully. It also puts a measured ceiling of **0.728 °C** on anything
trained on this target. Three attempts to fix the bias by modelling were, in hindsight,
attempts to fix the data with a loss function.

**A result was reported that turned out to be noise.** M3 initially looked like an 11%
improvement in GLORYS validation RMSE (0.739 → 0.660) and was reported as real. Rerunning
the *same config with the same seed* produced 0.729 — a ~10% spread from nondeterministic
cuDNN kernel selection and dataloader worker ordering alone, which is **larger than the
effect being claimed**. Any architecture claim on this setup needs multiple seeds and a
reported spread.

**The surface result inverted depending on what it was scored against.** Against GLORYS,
M2 looked **38% worse** than climatology at 0 m — a serious-looking regression that was
flagged as a real weakness. Against raw Argo, the same model is **32% better**. The model
was disagreeing with the reanalysis and the observations sided with the model. This is the
strongest evidence in the project for validating against independent data, and it was very
nearly written up as a defect instead.

---

## 6. Infrastructure

**The stored AWS credentials were dead** — `InvalidClientTokenId` on every call.

**GPU Spot quota was zero.** "All G and VT Spot Instance Requests" = 0, so the recommended
~$0.16/hr Spot g4dn could not launch at all. On-demand G quota was 4 vCPUs, exactly one
g4dn.xlarge, so the plan moved to on-demand at ~$0.53/hr. At 12 minutes per training run
the difference is a few dollars across the project.

**`python` does not exist on the Deep Learning AMI.** PyTorch lives in a virtualenv at
`/opt/pytorch`; the system `python3` has no torch. Also worth knowing:
`pip install torch` on that AMI *replaces the CUDA build with a CPU one* — a very quiet way
to rent a GPU and train on the processor.

**The instance directory was not a git clone**, so `git pull` failed and the code running
there was not version-tracked. Files were copied in with `scp` as a workaround; it should
be a proper clone.

**Argo profiles were scored against land.** A network emits *some* number on land, and
Argo matching takes the nearest grid cell, so 42 of 6093 coastal profiles (0.7%) were
scored against output the loss never constrained. Nearly invisible for M2 (0.891 → 0.890)
and catastrophic for the anomaly model, whose climatology base is zeroed on land: it
emitted ~0 °C where the truth is ~10, and those 0.7% of profiles alone moved its 500 m
RMSE from 0.30 to 0.94. Found by chasing a result that was implausible rather than merely
disappointing. Cubes now carry NaN on unsupervised cells.

**A fork-after-threads deadlock.** Training sat at zero epochs for twelve minutes with four
DataLoader workers asleep and load average 0.00. Fitting the climatology runs a dask
reduction that leaves a live thread pool in the parent; the DataLoader then forks workers
onto that broken lock state. The cache is built in its own process now.

**A global cache path that the self-check poisoned.** The climatology cache was first
written to a fixed location, so the self-check — which builds a tiny *fake* store — wrote
its climatology where real runs would silently have picked it up. Caught before use.

**The Vercel CLI crashed Node with a V8 out-of-memory.** It uploads the whole project
directory, and `data/` is 48 GB. A `.vercelignore` cuts the payload to 172 KB.

**Pushing to GitHub did not deploy the site.** The Vercel project is CLI-linked, not
connected to the repository, so `vercel --prod` is required (or connect the repo under
Project → Settings → Git).

---

## What the pattern says

Of the problems above, the expensive ones were not the crashes. A crash tells you where it
happened. The costly failures were the **silent** ones: a target that disappeared without
an error, a depth level quietly full of NaN, every Argo profile rejected at the surface, a
download counter climbing while coverage stood still, and a 10% noise band reported as an
11% improvement.

That is why the pipeline now asserts things that "obviously" hold — that no report depth is
empty, that normalisation statistics differ from the whole-record mean, that attention
actually moves information between positions, that chunk edges tile the record exactly. Each
of those guards exists because the corresponding assumption was, at some point, false.
