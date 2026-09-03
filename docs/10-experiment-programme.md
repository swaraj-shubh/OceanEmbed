---
title: "10 · Experiment Programme"
nav_order: 11
---

# 10 — The 10-Step Experiment Programme

> **For agentic workers:** REQUIRED SUB-SKILL: use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Take M4 from 0.890 °C to the best defensible score against independent Argo by testing six interventions under one selection discipline, and produce a single ablation table — including the failures — plus one frozen, once-evaluated final number.

**Architecture:** Every intervention is a config flag on the existing M4 ConvLSTM. Three are data-side (Argo bias correction, climatology-as-input, auxiliary channels), two are loss-side (depth-weighted, vertical-gradient), one is formulation-side (anomaly residual). Nothing adds capacity — doc 09 §4 measured that capacity is not the binding constraint. Selection happens on **val (2022) Argo**; **test (2023–24) Argo is touched exactly once**, in Task 10.

**Tech Stack:** PyTorch, xarray/Zarr, NumPy, pandas. Kaggle T4 for training (~8 GPU-hours total), local CPU for all evaluation and post-processing.

**Spec:** the 10-item list below, plus [doc 09 §8](09-day2-handover.html) ("What is left").

> **This page is the programme.** For the narrative of what happened and the current results in handover form, read [doc 11 (Day 3)](11-day3-handover.html) instead.

---

## The spec, verbatim

1. **Freeze & reproduce M4 baseline** — verify current metrics and training pipeline.
2. **Audit data leakage** — especially GLORYS ↔ Argo matching, train/val/test split, normalization.
3. **Test Argo post-hoc bias correction** — depth-only, then depth+month; validation Argo only.
4. **Test anomaly prediction** — predict `temperature - climatology`, then add climatology back.
5. **Test climatology as input** — add the 15 monthly climatology channels.
6. **Test auxiliary features** — bathymetry + day-of-year sin/cos + lat/lon.
7. **Test losses** — depth-weighted loss and vertical-gradient loss.
8. **Run 3 seeds + ensemble** — check whether improvements are stable.
9. **Create one ablation table** — M4 → each modification → metrics, including failures.
10. **Freeze the best model and evaluate ONCE on untouched test Argo** — report overall + depth-wise RMSE/MAE/bias/correlation.

Out of scope for this programme (deferred): the Streamlit demo, track B1 INCOIS gridded Argo, any increase in model capacity, ViT/GNN/foundation models.

---

## Global Constraints

Every task's requirements implicitly include these. They come from `CLAUDE.md` and are not negotiable.

- **Argo is never a model input or a training target.** The network never sees Argo in any task. Task 3 fits a 15-number offset on val-split Argo — that is post-processing, is reported as its own table row, and never touches the network.
- **Selection is on val (2022) Argo. Test (2023–24) Argo is opened once, in Task 10.** This is a change from Day 2, where every intervention was scored on test. Say so in the write-up.
- **Normalization stats come from the train split only** (`data/processed/norm_stats.json`, already frozen). Do not recompute them with new channels present unless the task says to.
- **Time-based splits, frozen:** train `2015-04-01 … 2021-12-31`, val `2022`, test `2023-01-01 … 2024-12-31`. Never random mixing.
- **Land and unsupervised cells are masked out of the loss and out of every metric.** Prediction cubes carry NaN there.
- **Every experiment has a YAML config and a run name; every result lands in `results/` as CSV.** No untracked runs.
- **Checkpoints must reach S3 before the instance dies** — see Task 0. Day 2's checkpoints did not, and no longer exist.
- **Any architecture or intervention claim needs ≥3 seeds and a reported spread.** Measured seed σ on the Argo score is ≈0.010 °C; on GLORYS val RMSE it is ≈0.03 (8%), which is why the Argo score is the benchmark.
- Python 3.10+, `pathlib`, no hardcoded Windows paths in `src/`. Tests are `if __name__ == "__main__"` assert blocks in the module, not pytest (CLAUDE.md §13).

### Baseline numbers this programme starts from

Measured, three seeds, vs test-split Argo (doc 09):

| | Argo blended RMSE |
|---|---|
| GLORYS12V1 target itself | **0.728** (the ceiling) |
| M4 ConvLSTM | **0.890 ± 0.008** |
| M2 U-Net | 0.901 ± 0.013 |
| M0 climatology | 1.160 |

### Two facts established before this plan was written

**A. There is no M4 checkpoint anywhere.** `s3://oceanembed-sih26-data/oceanembed/checkpoints/` holds only `m2_unet{,_best}.pt` and `m3_oceanembed{,_best}.pt`, dated 2026-08-31. Every Day 2 checkpoint (all seeds of M2/M3/M4/anomaly/grad) was lost with the instance. The result CSVs survived in git; the weights did not. **Task 1 is a retrain, not a reload**, and Task 0 exists so it does not happen again.

**B. The Argo bias correction works and transfers.** Measured on this machine using GLORYS itself as the probe (no checkpoint needed), 2023–24 test Argo, 6,389 profiles:

| Correction | Blended RMSE | vs raw |
|---|---|---|
| GLORYS raw | 0.734 | — |
| depth offset fitted on 2021 (train) | 0.671 | +8.5% |
| **depth offset fitted on 2022 (val)** | **0.665** | **+9.3%** |
| depth × month offset fitted on 2021 | 0.675 | +7.9% (worse — overfits) |
| oracle offset fitted on test itself | 0.663 | +9.7% (ceiling) |

A 15-number lookup table recovers **96% of the achievable gain**. The bias also *drifts*: +0.475 °C at 100 m in 2021 versus +0.716 in 2023–24, which is why the 2022 fit beats the 2021 fit. Applying the same correction to M4 (whose perfect-debias bound is 0.890 → 0.810) should land near **0.82**. For scale, every architectural change tried on Day 2 moved the score by 0.011 in total.

---

## File Structure

| File | Status | Responsibility |
|---|---|---|
| `deploy/setup.sh` | modify | Start the S3 checkpoint sync automatically so it cannot be forgotten |
| `src/argo_eval.py` | modify | Add `match_profiles()` — the matched (pred, obs, float-id) arrays that Tasks 2, 3 and 9 all need; `evaluate_argo()` becomes a thin wrapper. Add `paired_bootstrap()` |
| `src/audit_leakage.py` | create | Task 2's runnable audit: eight assertions over splits, stats, climatology, windows and Argo matching |
| `src/bias_correct.py` | create | Fit a depth (or depth×month) offset from a val cube + val Argo; apply it to any cube. Refuses to fit on test |
| `src/config.py` | modify | `AUX_CHANNELS`, `CLIM_CHANNELS`, `n_channels(extra)`, `bathy_path()` |
| `src/datasets.py` | modify | `extra=` parameter adding climatology (15 ch) and auxiliary (5 ch) input channels; `build_bathymetry()` cache |
| `src/models/unet.py` | modify | `depth_weight` argument on `masked_mse` |
| `src/train.py` | modify | Thread `extra` through; derive `in_ch`; build depth weights from norm stats |
| `src/predict_cube.py` | modify | `--offset` to apply a bias correction; `--ensemble` to average several cubes; extract `score_cube()` |
| `src/ablation.py` | create | Read every `results/*_argo.csv` into one markdown table, failures included |
| `configs/m4_*.yaml` | create | One per intervention |
| `docs/10-experiment-programme.md` | this file | Filled in with results as tasks complete |

---

## Task 0: Stop losing checkpoints

Day 2 produced 18 checkpoints and kept none. `deploy/sync_checkpoints.sh` is correct but has to be started by hand in a second shell, and on Day 2 it was not. This programme will produce ~20 more. Fix the mechanism before generating anything worth keeping.

**Files:**
- Modify: `deploy/setup.sh` (append, after the `aws s3 sync` of the store)

**Interfaces:**
- Produces: a running background sync loop on every bootstrapped GPU box; `checkpoints/` mirrored to S3 within 5 minutes of any `torch.save`.

- [ ] **Step 1: Append the auto-start to `deploy/setup.sh`**

```bash
# The sync loop is not optional. Day 2 lost 18 checkpoints because it was a separate
# manual command in a second shell that nobody ran. Starting it here means the only way
# to train without a sync is to not use this script.
pkill -f sync_checkpoints.sh || true
BUCKET="$BUCKET" PREFIX="$PREFIX" nohup bash deploy/sync_checkpoints.sh > sync.log 2>&1 &
echo "checkpoint sync -> s3://$BUCKET/$PREFIX/checkpoints (pid $!)"
```

- [ ] **Step 2: Verify the loop starts**

Run on the GPU box: `bash deploy/setup.sh`, then after ~30 s: `pgrep -f sync_checkpoints.sh && cat sync.log`
Expected: one PID printed, `sync.log` empty (the script runs `--quiet`).

- [ ] **Step 3: Verify a file actually lands in S3**

```bash
mkdir -p checkpoints && echo probe > checkpoints/_probe.txt
sleep 310
aws s3 ls "s3://$BUCKET/$PREFIX/checkpoints/_probe.txt"
rm checkpoints/_probe.txt
```
Expected: the probe file is listed. If it is not, the sync is broken — stop and fix it. Every later task depends on this.

- [ ] **Step 4: Commit**

```bash
git add deploy/setup.sh
git commit -m "Start the checkpoint sync from setup.sh: Day 2 lost every checkpoint"
```

---

## Task 1: Freeze and reproduce the M4 baseline

Spec item 1. The point is not a new number — it is proving that the pipeline on a fresh box reproduces the committed one, and establishing the **val-split** baseline that Tasks 3–8 are selected against. Day 2 selected on test; this programme does not.

**Files:**
- Modify: none (uses `configs/m4_convlstm.yaml` exactly as committed)
- Produces: `checkpoints/m4_convlstm_s{1,2,3}_best.pt`, `results/m4_convlstm_s{1,2,3}_best_val_argo.csv`

**Interfaces:**
- Consumes: Task 0's working sync.
- Produces: **`VAL_BASELINE`** — the 3-seed mean and σ of M4's blended RMSE against **2022** Argo. Every later task compares against this, not against 0.890.

- [ ] **Step 1: Bootstrap the GPU box and build the climatology cache**

```bash
BUCKET=oceanembed-sih26-data bash deploy/setup.sh
python src/datasets.py --clim        # must be its own process (doc 09 §7 fork deadlock)
```
Expected: `cuda: True <GPU name>` and `climatology cached (12, 15, 96, 176) -> nio_daily.clim.npy`.

- [ ] **Step 2: Run the three seeds**

```bash
for s in 1 2 3; do python src/train.py configs/m4_convlstm.yaml --seed $s; done
```
Expected: ~66 s/epoch × 20 epochs × 3 seeds ≈ 66 min. Each run prints `*best` on improving epochs and writes `checkpoints/m4_convlstm_s{N}_best.pt`.

- [ ] **Step 3: Score all three on TEST and confirm reproduction**

```bash
for s in 1 2 3; do python src/predict_cube.py --ckpt checkpoints/m4_convlstm_s${s}_best.pt --split test; done
```
Expected: 3-seed mean within ±0.02 of the committed 0.890 ± 0.008. This is the only sanctioned test read before Task 10 — it is a reproduction check against numbers already published in doc 09, not a selection decision. **If the mean lands outside 0.870–0.910, stop**: something in the environment differs, and every later comparison would be built on sand.

- [ ] **Step 4: Score all three on VAL — this is the working baseline**

```bash
for s in 1 2 3; do python src/predict_cube.py --ckpt checkpoints/m4_convlstm_s${s}_best.pt --split val; done
```
Expected: three `results/m4_convlstm_s{N}_best_val_argo.csv`, ~3,440 profiles matched each. Record the mean and σ as `M4 baseline (val)` in the results table at the bottom of this document.

- [ ] **Step 5: Confirm the checkpoints reached S3**

Run: `aws s3 ls s3://oceanembed-sih26-data/oceanembed/checkpoints/ | grep m4_convlstm`
Expected: six files. If missing, Task 0 failed — fix before continuing.

- [ ] **Step 6: Commit**

```bash
git add results/m4_convlstm_s*_val_argo.csv
git commit -m "M4 baseline reproduced; val-split Argo baseline recorded for selection"
```

---

## Task 2: Audit data leakage

Spec item 2. Turn every methodology rule into a runnable assertion, so the answer to "how do you know there's no leakage?" is a command rather than a claim. This task also does the `match_profiles` refactor, because Argo matching is one of the things being audited and three later tasks need the same matched arrays.

**Files:**
- Modify: `src/argo_eval.py` (extract `match_profiles`; `evaluate_argo` becomes a wrapper)
- Create: `src/audit_leakage.py`

**Interfaces:**
- Produces: `match_profiles(cube, profiles, max_days=1) -> (pred[15,N], obs[15,N], floats[N], times[N])`. `floats` is the Argo platform number parsed from the `profile` id; `times` is each matched cast's timestamp. Consumed by `bias_correct.fit_offset` (Task 3) and `paired_bootstrap` (Task 9).

- [ ] **Step 1: Extract `match_profiles` in `src/argo_eval.py`**

```python
def match_profiles(cube, profiles, max_days=1):
    """cube: xr.DataArray (time, depth, lat, lon). profiles: DataFrame (see module doc).

    Returns (pred[15, N], obs[15, N], floats[N], times[N]) for the N matched casts.
    `floats` is the Argo platform number, so callers can block-resample by float: 6,448
    test profiles come from only 147 floats, and two casts from one float ten days apart
    in the same water mass are not independent samples. `times` is returned so callers
    that bin by month (bias_correct) never have to re-derive the match order.
    """
    lat, lon = cube.lat.values, cube.lon.values
    times = cube.time.values.astype("datetime64[D]")
    P, O, F, T = [], [], [], []
    for pid, g in profiles.groupby("profile"):
        la, lo = float(g.lat.iloc[0]), float(g.lon.iloc[0])
        if not (lat.min() <= la <= lat.max() and lon.min() <= lo <= lon.max()):
            continue
        day = np.datetime64(pd.Timestamp(g.time.iloc[0]).date())
        k = int(np.abs(times - day).argmin())
        if abs((times[k] - day).astype(int)) > max_days:
            continue
        obs = interp_profile(g.pres.values, g.temp.values)
        if not np.isfinite(obs).any():
            continue
        i, j = int(np.abs(lat - la).argmin()), int(np.abs(lon - lo).argmin())
        pred = cube.isel(time=k, y=i, x=j).values if "y" in cube.dims else \
            cube.isel(time=k, lat=i, lon=j).values
        P.append(pred); O.append(obs)
        F.append(str(pid).split("_")[0]); T.append(pd.Timestamp(g.time.iloc[0]))
    if not P:
        z = np.zeros((len(TARGET), 0))
        return z, z, np.array([]), pd.DatetimeIndex([])
    return np.array(P).T, np.array(O).T, np.array(F), pd.DatetimeIndex(T)


def evaluate_argo(cube, profiles, max_days=1):
    """Returns (depth-wise table, n_profiles_matched)."""
    from metrics import depthwise
    pred, obs, _, _ = match_profiles(cube, profiles, max_days)
    return depthwise(pred, obs), pred.shape[1]
```

- [ ] **Step 2: Run the existing self-check to prove the refactor changed nothing**

Run: `python src/argo_eval.py`
Expected: `argo_eval self-check OK -- matched 1 profile, RMSE ...`. Every existing assertion must still pass — they are the regression test for this refactor. Do not weaken any of them.

- [ ] **Step 3: Write `src/audit_leakage.py`**

```python
"""Every methodology rule from CLAUDE.md sec.6, as a runnable assertion.

    python src/audit_leakage.py

The answer to "how do you know there is no leakage?" should be a command, not a claim.
Raises on the first violation.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.append(str(Path(__file__).resolve().parent))
from config import CHANNELS, INTERIM, SPLITS, ZARR, crop_to_model
from datasets import NIODataset, STATS_PATH, clim_path


def ok(msg):
    print(f"  PASS  {msg}")


print("1. splits are ordered and non-overlapping")
bounds = [(pd.Timestamp(a), pd.Timestamp(b))
          for a, b in (SPLITS["train"], SPLITS["val"], SPLITS["test"])]
for (a0, a1), (b0, b1) in zip(bounds, bounds[1:]):
    assert a1 < b0, f"{a1} >= {b0}: splits overlap"
ok("train < val < test, no overlap")

print("2. no sample window crosses a split boundary")
for w in (1, 7):
    prev = None
    for s in ("train", "val", "test"):
        d = NIODataset(s, window=w)
        assert len(d) == d.ds.sizes["time"] - w + 1
        if prev is not None:
            assert prev.max() < d.time.min(), f"{s}: window {w} leaks backwards"
        prev = d.time
ok("window=1 and window=7 both stay inside their split")

print("3. normalisation stats come from the train split only")
s = json.loads(STATS_PATH.read_text())
ds = xr.open_zarr(ZARR)
tr_mu = ds.X.sel(time=slice(*SPLITS["train"])).mean(
    dim=("time", "lat", "lon"), skipna=True).values
all_mu = ds.X.mean(dim=("time", "lat", "lon"), skipna=True).values
assert np.allclose(tr_mu, s["X"]["mean"], atol=1e-3), "stats are not a train-only fit"
assert not np.allclose(all_mu, s["X"]["mean"], atol=1e-4), "stats look fitted on all years"
ok("X mean reproduces the train-only fit and differs from the all-years fit")

print("4. the climatology cache is a train-split fit")
clim = np.load(clim_path(ZARR))
y = ds.Y.sel(time=slice(*SPLITS["train"])).groupby("time.month").mean("time", skipna=True)
assert np.allclose(np.nan_to_num(clim), np.nan_to_num(crop_to_model(y.values)), atol=1e-3)
ok("cached climatology reproduces a train-only monthly mean")

print("5. Argo never appears in the store")
assert set(ds.data_vars) == {"X", "Y"}, sorted(ds.data_vars)
assert list(ds.channel.values) == CHANNELS
ok("store holds only X (7 satellite channels) and Y (GLORYS); no Argo variable exists")

print("6. Argo used for scoring falls inside its split")
prof = pd.read_parquet(INTERIM / "argo_nio.parquet")
prof["time"] = pd.to_datetime(prof["time"]).dt.tz_localize(None)
prof["float"] = prof.profile.astype(str).str.split("_").str[0]
for split in ("val", "test"):
    lo, hi = SPLITS[split]
    sub = prof[(prof.time >= lo) & (prof.time <= pd.Timestamp(hi) + pd.Timedelta(days=1))]
    assert sub.time.min() >= pd.Timestamp(lo)
    assert sub.time.max() <= pd.Timestamp(hi) + pd.Timedelta(days=1)
    print(f"        {split}: {sub.profile.nunique()} profiles from {sub['float'].nunique()} floats")
ok("val and test Argo windows do not spill across their bounds")

print("7. GLORYS <-> Argo circularity, stated precisely")
print("        GLORYS12V1 assimilates Argo, so Argo is not statistically independent of")
print("        the target in general. The defensible claim is narrower, and is true: the")
print("        model trains on GLORYS 2015-2021 only, so no 2022-2024 Argo cast -- nor")
print("        the GLORYS state it informed -- was ever seen in training. Say it that way,")
print("        not 'Argo is independent of GLORYS'.")
ok("circularity is bounded by the time split, not by product independence")

print("8. effective sample size is floats, not profiles")
te = prof[prof.time >= SPLITS["test"][0]]
n_p, n_f = te.profile.nunique(), te["float"].nunique()
assert n_f < n_p / 10, "unexpectedly many floats -- recheck the profile-id parsing"
print(f"        {n_p} test profiles come from {n_f} floats; error bars must block by float")
ok("recorded; Task 9 uses a float-blocked bootstrap")

print("\nleakage audit: all checks passed")
```

- [ ] **Step 4: Run the audit**

Run: `python src/audit_leakage.py`
Expected: eight `PASS` lines, then `leakage audit: all checks passed`. Checks 6 and 8 should print `val: 3440 profiles from 86 floats` and `test: 6448 profiles from 147 floats`.

**If check 3 or 4 fails, stop the whole programme** — every published number is affected and must be recomputed, not just this task.

- [ ] **Step 5: Commit**

```bash
git add src/argo_eval.py src/audit_leakage.py
git commit -m "Runnable leakage audit; extract match_profiles for reuse"
```

---

## Task 3: Argo post-hoc bias correction

Spec item 3. Already validated on GLORYS as a probe (+9.3% from a val-fitted depth offset, 96% of the oracle). Now fit it on M4's own residuals: M4's 100 m bias is +0.850 against GLORYS's +0.716, so ~0.13 °C is the model's own and only the model's residuals capture both parts.

**Files:**
- Create: `src/bias_correct.py`
- Modify: `src/predict_cube.py` (add `--offset`, extract `score_cube()`)

**Interfaces:**
- Consumes: `match_profiles()` from Task 2.
- Produces: `fit_offset(cube, profiles, by_month=False) -> np.ndarray` of shape `[15]` or `[15, 12]`; `apply_offset(cube, off) -> xr.DataArray`; JSON artifacts `results/<run>_offset.json` and `results/<run>_offset_month.json`.

- [ ] **Step 1: Write `src/bias_correct.py` with its self-check**

```python
"""Post-hoc Argo bias correction -- a 15-number lookup table, fitted on VAL Argo.

    python src/bias_correct.py --cube results/m4_convlstm_s1_best_val_cube.nc --split val
    python src/predict_cube.py --ckpt ... --split test --offset results/m4_..._offset.json

The network never sees Argo. This stage does, so it is reported as its own row in the
ablation table and never folded into the model's number. Fitting on the test split is
refused outright: that is the one way this stage could become a lie.

Measured with GLORYS as the probe (2023-24 test Argo, 6389 casts): raw 0.734, corrected
0.665 with a val-fitted depth offset, against an oracle ceiling of 0.663. The bias drifts
(+0.475 degC at 100 m in 2021 vs +0.716 in 2023-24), so fit on the split nearest the
target period -- val, not train.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.append(str(Path(__file__).resolve().parent))
from argo_eval import match_profiles
from config import DEPTHS, INTERIM, SPLITS


def fit_offset(cube, profiles, by_month=False, max_days=1, min_n=20):
    """Mean (prediction - observation) per depth over the matched casts.

    by_month=True fits [15, 12] instead. Measured WORSE than depth-only on 4,345 casts
    (0.675 vs 0.671): 180 bins over ~3,400 profiles overfits. Kept because the spec asks
    for the comparison, and because more Argo years would change the answer. Months come
    from match_profiles' own `times`, so the bins can never fall out of step with the
    matched casts.
    """
    pred, obs, _, times = match_profiles(cube, profiles, max_days)
    assert pred.shape[1] >= min_n, f"only {pred.shape[1]} casts matched; refusing to fit"
    d = pred - obs
    flat = np.nanmean(d, axis=1)
    if not by_month:
        return flat
    off = np.tile(flat[:, None], (1, 12))
    for m in range(1, 13):
        sel = times.month == m
        if sel.sum() >= min_n:
            off[:, m - 1] = np.nanmean(d[:, sel], axis=1)
    return off


def apply_offset(cube, off):
    """cube minus the offset. `off` is [15] or [15, 12] (depth x calendar month)."""
    off = np.asarray(off, float)
    if off.ndim == 1:
        sub = xr.DataArray(off, dims=("depth",), coords={"depth": cube.depth})
    else:
        sub = xr.DataArray(off[:, cube.time.dt.month.values - 1], dims=("depth", "time"),
                           coords={"depth": cube.depth, "time": cube.time})
    return cube - sub


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cube", required=True)
    p.add_argument("--split", default="val")
    p.add_argument("--argo", default=str(INTERIM / "argo_nio.parquet"))
    p.add_argument("--by-month", action="store_true")
    p.add_argument("--max-days", type=int, default=1)
    a = p.parse_args()
    assert a.split != "test", (
        "refusing to fit an offset on the test split -- fit on val (CLAUDE.md rule 3)")

    cube = xr.open_dataarray(a.cube)
    prof = pd.read_parquet(a.argo)
    prof["time"] = pd.to_datetime(prof["time"]).dt.tz_localize(None)
    lo, hi = SPLITS[a.split]
    prof = prof[(prof.time >= lo) & (prof.time <= pd.Timestamp(hi) + pd.Timedelta(days=1))]

    off = fit_offset(cube, prof, by_month=a.by_month, max_days=a.max_days)
    stem = Path(a.cube).stem.replace(f"_{a.split}_cube", "")
    out = Path(a.cube).with_name(
        stem + ("_offset_month.json" if a.by_month else "_offset.json"))
    out.write_text(json.dumps({"split_fitted_on": a.split, "by_month": a.by_month,
                               "depths": DEPTHS, "offset": off.tolist()}, indent=2))
    print(f"fitted on {a.split} -> {out.name}")
    for z, v in zip(DEPTHS, off if off.ndim == 1 else off.mean(1)):
        print(f"  {z:5d} m  {v:+.3f}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main()
        raise SystemExit

    # A cube carrying a known constant per-depth bias must have exactly that bias
    # recovered, and applying the fit must drive the residual to zero.
    from metrics import depthwise
    rng = np.random.default_rng(0)
    lat, lon = np.arange(0.125, 5, 0.25), np.arange(55.125, 60, 0.25)
    t = pd.date_range("2022-01-01", periods=60)
    truth = np.linspace(29.0, 4.0, len(DEPTHS))
    bias = np.linspace(0.0, 0.9, len(DEPTHS))
    cube = xr.DataArray(
        np.broadcast_to((truth + bias)[None, :, None, None],
                        (len(t), len(DEPTHS), len(lat), len(lon))).copy(),
        coords={"time": t, "depth": DEPTHS, "lat": lat, "lon": lon},
        dims=("time", "depth", "lat", "lon"))
    z = np.arange(0.0, 1010.0, 10.0)
    prof = pd.concat([
        pd.DataFrame({"profile": f"29{k:05d}_1", "time": t[k],
                      "lat": float(rng.choice(lat)), "lon": float(rng.choice(lon)),
                      "pres": z, "temp": np.interp(z, DEPTHS, truth)})
        for k in range(40)], ignore_index=True)

    off = fit_offset(cube, prof)
    assert np.allclose(off, bias, atol=1e-6), f"offset not recovered: {off - bias}"
    pred, obs, floats, times = match_profiles(apply_offset(cube, off), prof)
    assert np.allclose(depthwise(pred, obs)["rmse"], 0.0, atol=1e-6), \
        "correction did not null the bias"
    assert len(np.unique(floats)) == 40, "float ids not parsed from the profile id"
    assert len(times) == 40, "one timestamp per matched cast, in match order"

    offm = fit_offset(cube, prof, by_month=True)
    assert offm.shape == (len(DEPTHS), 12)
    assert np.allclose(apply_offset(cube, offm).isel(time=0).values[:, 0, 0], truth, atol=1e-6)

    try:
        fit_offset(cube, prof.head(50), min_n=20)
        raise AssertionError("should have refused: far too few casts matched")
    except AssertionError as e:
        assert "refusing to fit" in str(e), e
    print("bias_correct self-check OK -- offset recovered to 1e-6, month form applies")
```

- [ ] **Step 2: Run the self-check**

Run: `python src/bias_correct.py`
Expected: `bias_correct self-check OK -- offset recovered to 1e-6, month form applies`.

- [ ] **Step 3: Add `--offset` and `score_cube()` to `src/predict_cube.py`**

Extract the scoring block from `main()` into a reusable function (Task 8's ensemble needs it too):

```python
def score_cube(cube, split, argo=str(INTERIM / "argo_nio.parquet"), max_days=1):
    """Depth-wise table for a cube against the raw Argo of one split."""
    prof = pd.read_parquet(argo)
    # ERDDAP hands back tz-aware UTC; the split bounds are naive and pandas refuses to
    # compare the two rather than silently guessing an offset.
    prof["time"] = pd.to_datetime(prof["time"]).dt.tz_localize(None)
    lo, hi = SPLITS[split]
    prof = prof[(prof.time >= lo) & (prof.time <= pd.Timestamp(hi) + pd.Timedelta(days=1))]
    return evaluate_argo(cube, prof, max_days=max_days)
```

Add the flag (`p.add_argument("--offset", default=None)`) and, in `main()` right after the cube is built:

```python
    if a.offset:
        from bias_correct import apply_offset
        meta = json.loads(Path(a.offset).read_text())
        assert meta["split_fitted_on"] != "test", "offset fitted on test -- circular"
        cube = apply_offset(cube, np.asarray(meta["offset"]))
        run = f"{run}_bc" + ("m" if meta["by_month"] else "")
```

- [ ] **Step 4: Fit on val and sanity-check the sign**

```bash
python src/predict_cube.py --ckpt checkpoints/m4_convlstm_s1_best.pt --split val
python src/bias_correct.py --cube results/m4_convlstm_s1_best_val_cube.nc --split val
python src/bias_correct.py --cube results/m4_convlstm_s1_best_val_cube.nc --split val --by-month
```
Expected: offsets **positive**, peaking near +0.7 to +0.9 at 100 m, near zero at 0 m and 500 m, slightly negative at 700–1000 m. A negative 100 m offset means the sign convention is inverted — `fit_offset` returns *prediction minus observation* and `apply_offset` **subtracts** it.

- [ ] **Step 5: Choose depth-only vs depth×month on VAL**

```bash
python src/predict_cube.py --ckpt checkpoints/m4_convlstm_s1_best.pt --split val \
    --offset results/m4_convlstm_s1_best_offset.json
python src/predict_cube.py --ckpt checkpoints/m4_convlstm_s1_best.pt --split val \
    --offset results/m4_convlstm_s1_best_offset_month.json
```
Note: fitting and scoring on the same split flatters both, so this comparison is only usable to pick the *form*, and the honest number comes from Step 6. Prediction from the GLORYS probe: depth-only wins. Record both val numbers, then carry the winner forward.

- [ ] **Step 6: One pre-registered confirmation read on test (seed 1 only)**

Write the prediction down *before* running, in this document: **0.82 ± 0.01 blended**, derived from M4's perfect-debias bound of 0.810 and the 96%-of-oracle recovery measured on the GLORYS probe.

```bash
python src/predict_cube.py --ckpt checkpoints/m4_convlstm_s1_best.pt --split test \
    --offset results/m4_convlstm_s1_best_offset.json
```

One seed, one pre-registered number, no choice made afterwards — that is what keeps this defensible. The full 3-seed test evaluation happens once, in Task 10. If the result lands far from 0.82, do **not** start tuning the correction against test; record the miss and diagnose on val.

The offset is fitted once, on seed 1's val cube, and later applied to every seed — refitting per seed would let three fits chase three sets of noise.

- [ ] **Step 7: Commit**

```bash
git add src/bias_correct.py src/predict_cube.py results/*_offset*.json results/*_bc*_argo.csv
git commit -m "Post-hoc Argo bias correction: depth offset fitted on val, ~9% expected"
```

---

## Interlude: the screening protocol for Tasks 4–7

Tasks 4–7 are five candidate interventions. Running all of them at three seeds costs ~5.5 GPU-hours; screening at one seed first costs ~1.8 and rejects the obvious losers. Day 2's measured seed spread on the Argo score is **σ ≈ 0.010 °C**.

**Decision rule, fixed now so it cannot be bent later:**

| Single-seed val score vs `VAL_BASELINE` | Action |
|---|---|
| better, or worse by ≤ 0.010 (1σ) | **promote** to 3 seeds in Task 8 |
| worse by 0.010 – 0.020 | promote only if a depth-wise table shows a clear win somewhere useful (e.g. below 300 m) |
| worse by > 0.020 (2σ) | **reject**; record the number in the ablation table as a failure and move on |

A one-seed screen cannot resolve a 0.01 effect — it is not meant to. It is meant to reject things that are clearly worse, the way the gradient loss (+0.017) and the anomaly formulation (+0.074) were on Day 2.

Every screening run is `--seed 1`, scored on **val** only. No test reads in Tasks 4–8.

---

## Task 4: Anomaly prediction on M4

Spec item 4. Already implemented (`anomaly: true` — the model predicts a departure from the train-split monthly climatology, and `base` is added back inside the model path so loss and metrics stay in absolute °C). Day 2 measured it on M2: **0.975 ± 0.020, clearly worse overall — but the first thing to beat climatology at 500 m and 1000 m.** This task asks the same question of M4, where the deep is already stronger.

**Files:**
- Create: `configs/m4_anomaly.yaml`

**Interfaces:**
- Consumes: `VAL_BASELINE` from Task 1; the climatology cache from `python src/datasets.py --clim`.
- Produces: `results/m4_anomaly_s1_best_val_argo.csv`.

- [ ] **Step 1: Write `configs/m4_anomaly.yaml`**

```yaml
run: m4_anomaly
model:
  kind: temporal
  out_ch: 15
  base: 32
  depth: 3
anomaly: true          # predict the departure from train-split monthly climatology
window: 7
batch_size: 16
num_workers: 4
lr: 0.001
epochs: 20
seed: 0
```

`in_ch` is deliberately absent — Task 5 makes `train.py` derive it. Until then, add `in_ch: 7`.

- [ ] **Step 2: Train and score on val**

```bash
python src/train.py configs/m4_anomaly.yaml --seed 1
python src/predict_cube.py --ckpt checkpoints/m4_anomaly_s1_best.pt --split val
```
Expected: ~22 min. **Prediction: worse than baseline overall (Day 2's M2 anomaly was +0.074), better at 500–1000 m.** If it is not worse overall, that is a genuine surprise worth a paragraph in the write-up.

- [ ] **Step 3: Apply the decision rule and record**

Compare the blended val RMSE to `VAL_BASELINE`, and compare the 500/700/1000 m rows to the baseline's. Record both in the results table at the bottom of this document, including the promote/reject decision and its reason.

- [ ] **Step 4: Commit**

```bash
git add configs/m4_anomaly.yaml results/m4_anomaly_s1_best_val_argo.csv
git commit -m "M4 anomaly formulation, screened on val"
```

---

## Task 5: Climatology as input channels

Spec item 5. Doc 09 §5.1 concluded the right formulation is **depth-dependent** — absolute up top, climatology-anchored in the deep — and proposed a hand-picked crossover. Feeding climatology as 15 input channels instead lets the network learn that crossover per depth, per location, and per season. [TS-Cast (Ocean Science, 2026)](https://os.copernicus.org/articles/22/2161/2026/) feeds monthly climatological profiles to a U-Net for exactly this reason.

This is the task that adds the `extra` channel machinery; Task 6 reuses it.

**Files:**
- Modify: `src/config.py`, `src/datasets.py`, `src/train.py`, `src/predict_cube.py`
- Create: `configs/m4_clim.yaml`

**Interfaces:**
- Produces: `config.n_channels(extra) -> int`; `NIODataset(..., extra=("clim",))` returning `x` with 22 channels in the frozen order `[7 surface] + [15 climatology]`.

- [ ] **Step 1: Add the channel bookkeeping to `src/config.py`**

```python
CLIM_CHANNELS = [f"clim_{d}m" for d in DEPTHS]                    # 15
AUX_CHANNELS = ["doy_sin", "doy_cos", "lat", "lon", "bathy"]      # 5

def n_channels(extra=()):
    """Input channel count for a channel set. The ORDER is frozen: surface, then
    climatology, then auxiliary. A checkpoint stores its `extra` list, so anything that
    rebuilds a dataset for inference must pass the same one -- doc 09 sec.7 records what
    happens when predict_cube guesses (M4 got [B,C,H,W] where it wanted [B,T,C,H,W])."""
    n = len(CHANNELS)
    if "clim" in extra:
        n += len(CLIM_CHANNELS)
    if "aux" in extra:
        n += len(AUX_CHANNELS)
    return n
```

- [ ] **Step 2: Add the `extra` parameter to `NIODataset`**

In `src/datasets.py`, extend `__init__`:

```python
    def __init__(self, split, zarr_path=ZARR, window=1, stats=STATS_PATH, crop=True,
                 anomaly=False, extra=()):
        ...
        self.extra = tuple(extra)
        # climatology is needed as a residual base (anomaly) or as input channels, or both
        self.clim = (self.climatology(zarr_path)
                     if (anomaly or "clim" in self.extra) else None)
        self.anomaly = anomaly
        self.ymu = np.asarray(s["Y"]["mean"], np.float32)[:, None, None]
        self.ysd = np.asarray(s["Y"]["std"], np.float32)[:, None, None]
```

and extend `__getitem__`, after the existing normalisation of `x`:

```python
        parts = [x]                                  # x is [window, 7, H, W]
        if "clim" in self.extra:
            # The climatology for each FRAME's own month -- a 7-day window can straddle a
            # month boundary. Normalised with the Y stats because it is in degC on the
            # target's scale, not the inputs'.
            c = self.clim[self.months[i:t + 1] - 1]
            parts.append((np.nan_to_num(c, nan=0.0) - self.ymu) / self.ysd)
        x = np.concatenate(parts, axis=1).astype(np.float32)
```

Note `self.clim` is `[12, 15, H, W]` and already cropped by `build_climatology(crop=True)`, so it needs no further crop. `self.months` is already the per-index month array for this split.

- [ ] **Step 3: Extend the `datasets.py` self-check**

Add to the `__main__` block, after the existing `window=7` assertions:

```python
    ce = NIODataset("train", store, stats=tmp / "stats.json", extra=("clim",))
    xc, yc, mc, bc = ce[0]
    from config import n_channels
    assert xc.shape == (n_channels(("clim",)), 96, 176) == (22, 96, 176), xc.shape
    assert np.allclose(xc[:7], x), "surface channels must be unchanged and come first"
    assert np.isfinite(xc).all(), "NaN leaked in via the climatology channels"
    assert not np.allclose(xc[7:], 0.0), "climatology channels are all zero"
    assert not bc.any(), "extra=('clim',) must NOT switch on the anomaly residual base"

    cw = NIODataset("train", store, stats=tmp / "stats.json", window=7, extra=("clim",))
    assert cw[0][0].shape == (7, 22, 96, 176), cw[0][0].shape
```

Run: `python src/datasets.py`
Expected: `datasets self-check OK`. The `bc.any()` assertion is the important one — climatology-as-input and the anomaly residual are two different mechanisms and must stay independently switchable, or Task 4 and Task 5 stop being separable experiments.

- [ ] **Step 4: Thread `extra` through `train.py` and `predict_cube.py`**

In `src/train.py`, inside `main()` before `build(cfg)`:

```python
    extra = tuple(cfg.get("extra", ()))
    # Derived, never hand-typed: a YAML that says in_ch: 7 alongside extra: [clim] is a
    # silent shape bug at best. The checkpoint carries cfg, so inference reads it back.
    cfg["model"]["in_ch"] = n_channels(extra)
```

and add `"extra": extra` to the `kw` dict passed to both `NIODataset` calls. Import `n_channels` from `config`.

In `src/predict_cube.py`, in `predict_cube()`:

```python
    ds = NIODataset(split, zarr, window=st["cfg"].get("window", 1),
                    anomaly=st["cfg"].get("anomaly", False),
                    extra=tuple(st["cfg"].get("extra", ())))
```

- [ ] **Step 5: Verify the existing configs are unaffected**

Run: `python src/train.py configs/m2_unet.yaml --seed 9` and stop it after epoch 0 (Ctrl-C).
Expected: it starts and the first epoch's loss is in the same range as `results/m2_unet.csv` epoch 0 (≈485). `n_channels(())` is 7, so nothing about the committed configs changes. Then `rm checkpoints/m2_unet_s9* results/m2_unet_s9*`.

- [ ] **Step 6: Write `configs/m4_clim.yaml` and screen it**

```yaml
run: m4_clim
model:
  kind: temporal
  out_ch: 15
  base: 32
  depth: 3
extra: [clim]          # 7 surface + 15 monthly climatology channels = 22
window: 7
batch_size: 16
num_workers: 4
lr: 0.001
epochs: 20
seed: 0
```

```bash
python src/train.py configs/m4_clim.yaml --seed 1
python src/predict_cube.py --ckpt checkpoints/m4_clim_s1_best.pt --split val
```
Expected: ~25 min (22 input channels instead of 7 costs a little in the first conv only). **This is the intervention most likely to beat baseline** — it hands the model the seasonal cycle and the deep-ocean anchor without forcing the residual form that failed on Day 2.

- [ ] **Step 7: Apply the decision rule, record, commit**

```bash
git add src/config.py src/datasets.py src/train.py src/predict_cube.py \
        configs/m4_clim.yaml results/m4_clim_s1_best_val_argo.csv
git commit -m "Climatology as 15 input channels; derive in_ch from the channel set"
```

---

## Task 6: Auxiliary features — bathymetry, day-of-year, lat/lon

Spec item 6. The model currently cannot tell January from July except through the fields themselves, in a basin whose entire dynamics are monsoonal; and it cannot tell the 200 m shelf off Gujarat from 4 km of open Arabian Sea. Longitude, latitude, day-of-year and bathymetry are the standard auxiliary set in this literature.

**Bathymetry comes free from the store.** `Y` is NaN below the sea floor, so the count of valid depth levels per cell *is* a 16-value bathymetry on exactly the right grid: measured over the train split, 4,781 land cells, 11,199 full-1000 m cells, and ~2,020 shelf cells spread across 2–14 levels. No GEBCO download, no regrid, no account.

**Files:**
- Modify: `src/config.py` (`bathy_path`), `src/datasets.py` (`build_bathymetry`, aux channels)
- Create: `configs/m4_aux.yaml`

**Interfaces:**
- Consumes: `n_channels` and the `extra` plumbing from Task 5.
- Produces: `build_bathymetry(zarr_path)` writing `nio_daily.bathy.npy` `[96,176]` float32 in [0,1]; `NIODataset(..., extra=("aux",))` returning 12 channels, or 27 with `("clim","aux")`.

- [ ] **Step 1: Add `bathy_path` to `src/config.py`**

```python
def bathy_path(zarr_path):
    """Cache beside the store it was fitted from -- same rule as the climatology cache."""
    return Path(zarr_path).with_suffix(".bathy.npy")
```

(`from pathlib import Path` is already imported there.)

- [ ] **Step 2: Add `build_bathymetry` to `src/datasets.py`**

```python
def build_bathymetry(zarr_path=ZARR, crop=True):
    """Fraction of the 15 depth levels GLORYS resolves at each cell, from the TRAIN split.

    0 on land, 1 where the full 1000 m column exists. This is free bathymetry: Y is
    already NaN below the sea floor, on exactly the model grid, with no download. Fitted
    on train only so no one can argue the shelf mask carries test-period information --
    it is static, but the cost of proving that is one slice.

    ponytail: 15 quantised levels, not metres. If the shelf channel earns its place,
    GEBCO/ETOPO regridded to 0.25 deg is the continuous upgrade.

    Run in its own process, like the climatology: the dask reduction leaves a thread pool
    behind that deadlocks DataLoader workers forked afterwards (doc 09 sec.7).
    """
    y = xr.open_zarr(zarr_path).Y.sel(time=slice(*SPLITS["train"]))
    valid = np.isfinite(y.isel(time=slice(0, 30))).all("time").values     # [15, H, W]
    b = (valid.sum(0) / y.sizes["depth"]).astype(np.float32)
    b = crop_to_model(b) if crop else b
    np.save(bathy_path(zarr_path), b)
    return b
```

Import `bathy_path` and `DEPTHS`/`LAT`/`LON` from `config` at the top of the module.

Extend the `--clim` entrypoint to build both caches (one flag, no new command to forget):

```python
    if "--clim" in _sys.argv:                   # build both caches, then exit
        c = build_climatology()
        b = build_bathymetry()
        print(f"climatology cached {c.shape} -> {clim_path(ZARR).name}")
        print(f"bathymetry cached {b.shape}, {(b == 0).mean():.1%} land -> {bathy_path(ZARR).name}")
        raise SystemExit
```

- [ ] **Step 3: Add the aux channels to `NIODataset`**

In `__init__`:

```python
        if "aux" in self.extra:
            self.bathy = np.load(bathy_path(zarr_path))[None]              # [1, H, W]
            self.doy = pd.DatetimeIndex(self.ds.time.values).dayofyear.to_numpy()
            # Absolute position is real signal here, not a nuisance: this is a regional
            # model on a frozen grid, and the Arabian Sea and the Bay of Bengal behave
            # differently. Scaled to [-1, 1] to sit on the same scale as the normalised
            # surface channels.
            la = np.broadcast_to(LAT[:, None], GRID_SHAPE).astype(np.float32)
            lo = np.broadcast_to(LON[None, :], GRID_SHAPE).astype(np.float32)
            rescale = lambda v: (2 * (v - v.min()) / (v.max() - v.min()) - 1).astype(np.float32)
            self.latlon = np.stack([rescale(crop_to_model(la)), rescale(crop_to_model(lo))])
```

In `__getitem__`, after the climatology block:

```python
        if "aux" in self.extra:
            # sin/cos so 31 Dec and 1 Jan are adjacent; per frame, because a 7-day window
            # spans seven different days.
            d = self.doy[i:t + 1].astype(np.float32)
            ang = 2 * np.pi * d / 365.25
            hw = self.bathy.shape[-2:]
            season = np.stack([np.sin(ang), np.cos(ang)], 1)[:, :, None, None]
            season = np.broadcast_to(season, (len(d), 2, *hw))
            static = np.broadcast_to(np.concatenate([self.latlon, self.bathy])[None],
                                     (len(d), 3, *hw))
            parts.append(np.concatenate([season, static], 1).astype(np.float32))
```

- [ ] **Step 4: Extend the self-check**

Add to `__main__`, after the Task 5 assertions:

```python
    build_bathymetry(store)
    ae = NIODataset("train", store, stats=tmp / "stats.json", extra=("aux",))
    xa2 = ae[0][0]
    assert xa2.shape == (n_channels(("aux",)), 96, 176) == (12, 96, 176), xa2.shape
    assert np.allclose(xa2[:7], x), "surface channels must come first and be unchanged"
    assert np.isfinite(xa2).all()
    # doy sin/cos must be constant in space and must MOVE between two different days
    assert xa2[7].std() < 1e-6 and xa2[8].std() < 1e-6, "day-of-year is not spatially flat"
    assert abs(float(xa2[7, 0, 0]) - float(ae[10][0][7, 0, 0])) > 1e-3, \
        "day-of-year identical ten days apart -- the channel is inert"
    assert xa2[9].min() >= -1.001 and xa2[9].max() <= 1.001, "lat channel out of [-1,1]"
    assert xa2[10].min() >= -1.001 and xa2[10].max() <= 1.001, "lon channel out of [-1,1]"
    assert xa2[9, 0, 0] < xa2[9, -1, 0], "lat channel is upside down"
    assert xa2[11].min() == 0.0 and xa2[11].max() == 1.0, "bathymetry is not in [0,1]"

    both = NIODataset("train", store, stats=tmp / "stats.json", window=7,
                      extra=("clim", "aux"))
    assert both[0][0].shape == (7, 27, 96, 176), both[0][0].shape
```

Run: `python src/datasets.py`
Expected: `datasets self-check OK`. The "identical ten days apart" assertion is the one that catches a day-of-year channel accidentally computed from the split's first date instead of each sample's.

- [ ] **Step 5: Write `configs/m4_aux.yaml` and screen it**

```yaml
run: m4_aux
model:
  kind: temporal
  out_ch: 15
  base: 32
  depth: 3
extra: [aux]           # 7 surface + doy sin/cos + lat + lon + bathymetry = 12
window: 7
batch_size: 16
num_workers: 4
lr: 0.001
epochs: 20
seed: 0
```

```bash
python src/datasets.py --clim          # rebuild both caches, including bathymetry
python src/train.py configs/m4_aux.yaml --seed 1
python src/predict_cube.py --ckpt checkpoints/m4_aux_s1_best.pt --split val
```

- [ ] **Step 6: If Tasks 5 and 6 both pass the rule, screen the combination**

```yaml
# configs/m4_clim_aux.yaml -- as m4_clim.yaml but:
run: m4_clim_aux
extra: [clim, aux]     # 27 channels
```

```bash
python src/train.py configs/m4_clim_aux.yaml --seed 1
python src/predict_cube.py --ckpt checkpoints/m4_clim_aux_s1_best.pt --split val
```
The two are not obviously additive — climatology already carries most of the seasonal cycle that day-of-year encodes — so the combination is a real question, not a formality.

- [ ] **Step 7: Record and commit**

```bash
git add src/config.py src/datasets.py configs/m4_aux.yaml configs/m4_clim_aux.yaml \
        results/m4_aux_s1_best_val_argo.csv results/m4_clim_aux_s1_best_val_argo.csv
git commit -m "Auxiliary channels: bathymetry from the store's own mask, day-of-year, lat/lon"
```

---

## Task 7: Losses — depth-weighted and vertical-gradient

Spec item 7. The vertical-gradient loss is already implemented and was measured **negative on M2 (0.918 ± 0.004 vs 0.901)**; it is re-run on M4 because the spec asks and it costs 22 minutes. The depth-weighted loss is new.

**State the prediction before running, because it follows from the metric definition.** The blended score is an n-weighted RMS across depths with roughly equal n per depth, so it is very nearly the mean per-depth MSE in °C² — which is exactly what plain masked MSE already minimises. **Plain MSE is already the loss matched to the reported metric.** Inverse-variance depth weighting (from the frozen train stats: 0.38 at 100 m rising to 2.53 at 1000 m, a 6.6× swing) deliberately trades thermocline accuracy for deep accuracy. Expect the blended number to get *worse* and 500–1000 m to get *better*. That is a shape change, not a free win — and it is worth measuring precisely because 500/700/1000 m are the three depths where climatology still beats M4.

**Files:**
- Modify: `src/models/unet.py` (`depth_weight` on `masked_mse`), `src/train.py`
- Create: `configs/m4_dw.yaml`, `configs/m4_grad.yaml`

**Interfaces:**
- Produces: `masked_mse(pred, true, mask, grad_weight=0.0, depth_weight=None)` where `depth_weight` is a `[15]` tensor on the same device.

- [ ] **Step 1: Add `depth_weight` to `masked_mse`**

```python
def masked_mse(pred, true, mask, grad_weight=0.0, depth_weight=None):
    """...existing docstring...

    `depth_weight` is a [15] tensor re-weighting the per-level contribution. Plain MSE in
    degC already matches the reported blended metric almost exactly, so this is a
    deliberate trade: inverse-variance weights (1/sd^2 from the frozen train stats) push
    effort from the thermocline into the deep, where climatology still beats us.
    """
    n = mask.sum()
    assert n > 0, "batch has no valid target cells"
    if depth_weight is None:
        loss = ((pred - true) ** 2 * mask).sum() / n
    else:
        w = depth_weight.view(1, -1, 1, 1).to(pred.device)
        wm = mask * w
        loss = ((pred - true) ** 2 * wm).sum() / wm.sum()
    if grad_weight:
        ...unchanged...
```

- [ ] **Step 2: Extend the `unet.py` self-check**

Add after the existing `masked_mse` assertions:

```python
    w_flat = torch.ones(15)
    assert torch.isclose(masked_mse(y, y + 1, m, depth_weight=w_flat),
                         masked_mse(y, y + 1, m)), "uniform weights must be a no-op"
    w_deep = torch.zeros(15); w_deep[-1] = 1.0
    err = y.detach().clone(); err[:, :-1] += 5.0        # error only at levels 0..13
    assert masked_mse(err, y.detach(), m, depth_weight=w_deep) < 1e-6, \
        "zero-weighted levels still contributed"
    assert masked_mse(err, y.detach(), m) > 1.0, "control: unweighted must see the error"
```

Run: `python src/models/unet.py`
Expected: all self-checks pass, including the existing overfit-one-sample checks.

- [ ] **Step 3: Build the weights in `train.py`**

```python
    dw = cfg.get("depth_weight")
    if dw == "inv_var":
        # 1/sd^2 from the frozen TRAIN stats, normalised to mean 1 so the loss stays on
        # the same scale as every other run and the printed numbers remain comparable.
        sd = np.asarray(json.loads(Path(stats or STATS_PATH).read_text())["Y"]["std"])
        w = 1.0 / sd ** 2
        depth_weight = torch.tensor(w / w.mean(), dtype=torch.float32, device=dev)
    elif dw == "inv_std":
        sd = np.asarray(json.loads(Path(stats or STATS_PATH).read_text())["Y"]["std"])
        w = 1.0 / sd
        depth_weight = torch.tensor(w / w.mean(), dtype=torch.float32, device=dev)
    else:
        assert dw is None, f"unknown depth_weight: {dw}"
        depth_weight = None
```

and pass `depth_weight=depth_weight` in the `masked_mse` call. Import `json` and `STATS_PATH`.

- [ ] **Step 4: Write both configs**

```yaml
# configs/m4_dw.yaml -- as configs/m4_convlstm.yaml but:
run: m4_dw
depth_weight: inv_var
```

```yaml
# configs/m4_grad.yaml -- as configs/m4_convlstm.yaml but:
run: m4_grad
grad_weight: 1.0
```

- [ ] **Step 5: Screen both**

```bash
python src/train.py configs/m4_dw.yaml   --seed 1
python src/train.py configs/m4_grad.yaml --seed 1
for r in m4_dw m4_grad; do python src/predict_cube.py --ckpt checkpoints/${r}_s1_best.pt --split val; done
```
Expected: ~45 min total. **Predictions: `m4_grad` worse (Day 2's M2 grad was +0.017); `m4_dw` worse on the blended number but better at 500/700/1000 m.** If `m4_dw` improves the deep without costing more than 1σ overall, promote it — that is the one combination that would make every depth beat climatology.

- [ ] **Step 6: Record and commit**

```bash
git add src/models/unet.py src/train.py configs/m4_dw.yaml configs/m4_grad.yaml \
        results/m4_dw_s1_best_val_argo.csv results/m4_grad_s1_best_val_argo.csv
git commit -m "Depth-weighted loss; re-screen the gradient loss on M4"
```

---

## Task 8: Three seeds for the survivors, then ensemble

Spec item 8. Everything promoted by the decision rule gets two more seeds, so its claim carries a spread. Then average the seed cubes — a same-config seed ensemble typically buys 2–4% for nothing, and it needs no training at all.

**Files:**
- Modify: `src/predict_cube.py` (add `--ensemble`)

**Interfaces:**
- Consumes: `score_cube()` from Task 3.
- Produces: `results/<name>_ens_<split>_cube.nc` and its Argo CSV.

- [ ] **Step 1: Run seeds 2 and 3 for every promoted config**

```bash
for cfg in <promoted configs>; do
  for s in 2 3; do python src/train.py configs/${cfg}.yaml --seed $s; done
done
for cfg in <promoted configs>; do
  for s in 1 2 3; do python src/predict_cube.py --ckpt checkpoints/${cfg}_s${s}_best.pt --split val; done
done
```
Budget: ~45 min per promoted config. If more than three configs are promoted, run them in order of their single-seed val score and stop when the GPU budget runs out — record which ones were not completed rather than reporting a partial spread as a full one.

- [ ] **Step 2: Add `--ensemble` to `predict_cube.py`**

```python
    p.add_argument("--ensemble", nargs="+", default=None,
                   help="average these cubes instead of running a checkpoint")
```

and at the top of `main()`:

```python
    if a.ensemble:
        cubes = [xr.open_dataarray(c) for c in a.ensemble]
        t0 = cubes[0]
        for c in cubes[1:]:
            assert c.shape == t0.shape, f"cube shapes differ: {c.shape} vs {t0.shape}"
            assert (c.time.values == t0.time.values).all(), "cubes cover different days"
        # NaN on land in every member, so a plain mean keeps land NaN -- which is what the
        # metrics need. Do NOT use nanmean here: it would invent values on land.
        cube = sum(cubes) / len(cubes)
        run = a.run or "ensemble"
    else:
        cube = predict_cube(a.ckpt, a.split)
        run = Path(a.ckpt).stem
```

with `p.add_argument("--run", default=None)` for the output name.

- [ ] **Step 3: Score the seed ensemble of the best config on val**

```bash
python src/predict_cube.py --ensemble results/<best>_s{1,2,3}_best_val_cube.nc \
    --split val --run <best>_ens
```
Expected: better than the mean of the individual seeds, by ~2–4%. If it is not, check that the three cubes are genuinely different — identical cubes mean the `--seed` override is not reaching `torch.manual_seed`.

- [ ] **Step 4: Score a cross-config ensemble on val**

```bash
python src/predict_cube.py --ensemble results/m4_convlstm_s1_best_val_cube.nc \
    results/<second best>_s1_best_val_cube.nc results/<third>_s1_best_val_cube.nc \
    --split val --run m4_mix_ens
```
Averaging *different* configurations usually beats averaging seeds of one, because the members' errors are less correlated. Whichever of Step 3 and Step 4 wins on val is the ensemble that goes into Task 10.

- [ ] **Step 5: Record and commit**

```bash
git add src/predict_cube.py results/*_val_argo.csv
git commit -m "Three seeds for promoted configs; seed and cross-config ensembles"
```

---

## Task 9: One ablation table, failures included

Spec item 9. One command that reads every result CSV and emits the table that goes in the deck — including every intervention that failed, because the failures are the evidence that the ceiling finding in doc 09 §4 is real.

This task also adds the **float-blocked bootstrap**, which is what finally answers "is M4 genuinely better than M2?". The 6,448 test profiles come from **147 floats**; two casts from one float ten days apart in the same water mass are not independent, so the honest error bar resamples floats, not profiles.

**Files:**
- Modify: `src/argo_eval.py` (add `paired_bootstrap`)
- Create: `src/ablation.py`

**Interfaces:**
- Consumes: `match_profiles()` from Task 2, `summary()` from `metrics`.
- Produces: `paired_bootstrap(cube_a, cube_b, profiles, n=1000, seed=0) -> (delta, lo, hi)` — the blended-RMSE difference `a - b` and its 95% float-blocked interval; `results/ablation.md`.

- [ ] **Step 1: Add `paired_bootstrap` to `src/argo_eval.py`**

```python
def paired_bootstrap(cube_a, cube_b, profiles, n=1000, seed=0, max_days=1):
    """95% CI on (blended RMSE of a) - (blended RMSE of b), resampling FLOATS.

    Paired: both models are scored on the same resampled casts, so the shared difficulty
    of a hard water mass cancels. Blocked by float because profiles are not independent --
    147 floats produced the 6,448 test casts, so a profile-level bootstrap would report an
    interval roughly sqrt(6448/147) ~ 6.6x too narrow and turn noise into a result.
    """
    from metrics import depthwise, summary
    pa, oa, fa, _ = match_profiles(cube_a, profiles, max_days)
    pb, ob, fb, _ = match_profiles(cube_b, profiles, max_days)
    assert fa.shape == fb.shape and (fa == fb).all(), \
        "cubes matched different casts -- compare cubes on the same grid and days"
    assert np.allclose(np.nan_to_num(oa), np.nan_to_num(ob)), "observations differ"

    floats = np.unique(fa)
    idx = {f: np.flatnonzero(fa == f) for f in floats}
    rng = np.random.default_rng(seed)
    d = np.empty(n)
    for k in range(n):
        take = np.concatenate([idx[f] for f in rng.choice(floats, floats.size, replace=True)])
        d[k] = (summary(depthwise(pa[:, take], oa[:, take]))
                - summary(depthwise(pb[:, take], ob[:, take])))
    point = summary(depthwise(pa, oa)) - summary(depthwise(pb, ob))
    return point, float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))
```

- [ ] **Step 2: Add a self-check for it**

Append to the `__main__` block in `argo_eval.py`:

```python
    # A cube that is uniformly 0.5 degC worse must show a strictly positive interval;
    # the same cube against itself must bracket zero.
    lat2, lon2 = np.arange(0.125, 5, 0.25), np.arange(55.125, 60, 0.25)
    t2 = pd.date_range("2023-01-01", periods=40)
    truth2 = np.linspace(29.0, 4.0, len(DEPTHS))
    base = xr.DataArray(
        np.broadcast_to(truth2[None, :, None, None],
                        (len(t2), len(DEPTHS), len(lat2), len(lon2))).copy(),
        coords={"time": t2, "depth": DEPTHS, "lat": lat2, "lon": lon2},
        dims=("time", "depth", "lat", "lon"))
    zz = np.arange(0.0, 1010.0, 10.0)
    pr = pd.concat([pd.DataFrame({"profile": f"59{k:05d}_1", "time": t2[k],
                                  "lat": float(lat2[k % len(lat2)]),
                                  "lon": float(lon2[k % len(lon2)]),
                                  "pres": zz, "temp": np.interp(zz, DEPTHS, truth2)})
                    for k in range(40)], ignore_index=True)
    pt, lo_, hi_ = paired_bootstrap(base + 0.5, base, pr, n=200)
    assert lo_ > 0 and abs(pt - 0.5) < 1e-6, (pt, lo_, hi_)
    pt0, lo0, hi0 = paired_bootstrap(base, base, pr, n=200)
    assert abs(pt0) < 1e-9 and lo0 <= 0 <= hi0
    print(f"paired_bootstrap self-check OK -- +0.5 degC gives {pt:.3f} [{lo_:.3f}, {hi_:.3f}]")
```

Run: `python src/argo_eval.py`
Expected: both the original assertions and the new bootstrap line pass.

- [ ] **Step 3: Write `src/ablation.py`**

```python
"""One ablation table from every result CSV in results/. Failures included on purpose.

    python src/ablation.py [--split test] [--out results/ablation.md]

The table is the core evidence of the whole project (CLAUDE.md sec.12): same held-out
Argo, same depths, every stage that was tried -- including the four interventions that
did not work, because those are what make the ceiling finding credible.
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
from config import REPORT_DEPTHS, ROOT
from metrics import summary

RESULTS = ROOT / "results"
SEED = re.compile(r"_s\d+_")


def rows(split):
    """Group every <run>_<split>_argo.csv by run, averaging across seeds."""
    out = {}
    for f in sorted(RESULTS.glob(f"*_{split}_argo.csv")):
        name = SEED.sub("_", f.name).replace(f"_{split}_argo.csv", "")
        out.setdefault(name, []).append(pd.read_csv(f))
    table = []
    for name, dfs in out.items():
        blends = [summary(d) for d in dfs]
        mean = pd.concat(dfs).groupby("depth_m").mean(numeric_only=True)
        table.append({
            "run": name, "seeds": len(dfs),
            "blended": float(np.mean(blends)),
            "sd": float(np.std(blends, ddof=1)) if len(blends) > 1 else np.nan,
            **{f"rmse_{z}m": float(mean.loc[z, "rmse"]) for z in REPORT_DEPTHS},
            "bias_100m": float(mean.loc[100, "bias"]),
        })
    return pd.DataFrame(table).sort_values("blended").reset_index(drop=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="test")
    p.add_argument("--out", default=None)
    a = p.parse_args()
    df = rows(a.split)
    assert len(df) > 0, f"no results/*_{a.split}_argo.csv found"
    df["blended"] = df.apply(
        lambda r: f"{r.blended:.3f}" + ("" if np.isnan(r.sd) else f" ± {r.sd:.3f}"), axis=1)
    md = df.drop(columns=["sd"]).to_markdown(index=False, floatfmt=".3f")
    out = Path(a.out) if a.out else RESULTS / f"ablation_{a.split}.md"
    out.write_text(f"# Ablation vs Argo ({a.split} split)\n\n{md}\n")
    print(md)
    print(f"\n-> {out}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main()
        raise SystemExit
    # self-check: two fake runs, one clearly worse, must come out in the right order
    import tempfile
    from metrics import depthwise
    tmp = Path(tempfile.mkdtemp())
    globals()["RESULTS"] = tmp
    rng = np.random.default_rng(0)
    true = rng.normal(size=(15, 400))
    for name, err in (("good", 0.4), ("bad", 1.2)):
        for s in (1, 2):
            depthwise(true + err * rng.normal(size=true.shape), true).to_csv(
                tmp / f"{name}_s{s}_val_argo.csv", index=False)
    df = rows("val")
    assert list(df.run) == ["good", "bad"], list(df.run)
    assert (df.seeds == 2).all() and df.sd.notna().all()
    assert df.blended.iloc[0] < df.blended.iloc[1]
    print("ablation self-check OK\n", df[["run", "seeds", "blended"]].to_string(index=False))
```

- [ ] **Step 4: Run the self-check**

Run: `python src/ablation.py`
Expected: `ablation self-check OK` and a two-row frame with `good` first.

- [ ] **Step 5: Settle M4 vs M2 with the float-blocked bootstrap**

```bash
python - <<'EOF'
import sys, pandas as pd, xarray as xr
sys.path.append("src")
from argo_eval import paired_bootstrap
from config import SPLITS, INTERIM
prof = pd.read_parquet(INTERIM / "argo_nio.parquet")
prof["time"] = pd.to_datetime(prof["time"]).dt.tz_localize(None)
lo, hi = SPLITS["test"]
prof = prof[(prof.time >= lo) & (prof.time <= pd.Timestamp(hi) + pd.Timedelta(days=1))]
a = xr.open_dataarray("results/m4_convlstm_s1_best_test_cube.nc")
b = xr.open_dataarray("results/m2_unet_s1_best_test_cube.nc")
print("M4 - M2: %.4f  95%% CI [%.4f, %.4f]" % paired_bootstrap(a, b, prof))
EOF
```
Expected: a negative point estimate (M4 better) with an interval that probably **straddles zero** — which is the same conclusion doc 09 reached from three seeds, now on a proper footing. Report the interval either way; an interval that excludes zero would be the first statistically defensible architecture claim in the project.

This needs `results/m2_unet_s1_best_test_cube.nc`, which requires the M2 checkpoint — it is one of the four that survived on S3. Fetch it: `aws s3 cp s3://oceanembed-sih26-data/oceanembed/checkpoints/m2_unet_best.pt checkpoints/`.

- [ ] **Step 6: Generate the val table and commit**

```bash
python src/ablation.py --split val
git add src/argo_eval.py src/ablation.py results/ablation_val.md
git commit -m "Ablation table generator; float-blocked paired bootstrap for model comparison"
```

---

## Task 10: Freeze, evaluate once, write it up

Spec item 10. One model, chosen on val, evaluated on test exactly once, reported depth-wise.

**Files:**
- Modify: `docs/10-experiment-programme.md` (this file — the results section), `docs/index.md`, `CLAUDE.md` §2b
- Create: `results/ablation_test.md`, `checkpoints/FROZEN.md`

- [ ] **Step 1: Choose the winner on val, and write the choice down before touching test**

Record in the results table below: the winning configuration, its val blended RMSE and spread, the ensemble decision from Task 8, whether the bias correction is applied, and **why** — all before running Step 2. Once test is opened, this choice is not revisited.

- [ ] **Step 2: Generate the final test cubes**

```bash
for s in 1 2 3; do python src/predict_cube.py --ckpt checkpoints/<winner>_s${s}_best.pt --split test; done
python src/predict_cube.py --ensemble results/<winner>_s{1,2,3}_best_test_cube.nc \
    --split test --run <winner>_ens
python src/predict_cube.py --ensemble results/<winner>_s{1,2,3}_best_test_cube.nc \
    --split test --run <winner>_ens_bc --offset results/<winner>_s1_best_offset.json
```

- [ ] **Step 3: Produce the final table**

```bash
python src/ablation.py --split test --out results/ablation_test.md
```
Report **RMSE, MAE, bias and correlation** at 0, 50, 100, 200, 500 and 1000 m (`REPORT_DEPTHS`), plus the blended number with its 3-seed spread — and keep the `M0 climatology` and `GLORYS12V1 target` rows in the table, because the model's number is only interpretable between that floor and that ceiling.

- [ ] **Step 4: Freeze the checkpoint**

`checkpoints/` is gitignored and `results/*.md` is tracked, so the manifest lives in `results/`.

```bash
aws s3 sync checkpoints s3://oceanembed-sih26-data/oceanembed/checkpoints
cat > results/FROZEN.md <<'EOF'
<winner>_s1_best.pt  -- the demo and every reported number use this file.
config: configs/<winner>.yaml      norm stats: data/processed/norm_stats.json
climatology cache: data/processed/nio_daily.clim.npy  (train split, monthly)
bathymetry cache:  data/processed/nio_daily.bathy.npy (train split)
bias offset:       results/<winner>_s1_best_offset.json  (fitted on 2022 val Argo)
git commit: <sha>
EOF
git add results/FROZEN.md && git commit -m "Freeze the final checkpoint"
```

- [ ] **Step 5: Fill in the results section of this document**

Every task above says "record". Consolidate: one table of all interventions with val and test numbers, the promote/reject decision for each, the bootstrap interval for the headline comparison, and a paragraph on what the programme taught. **Include the failures.** Doc 09's credibility comes from reporting four null results; this document should read the same way.

- [ ] **Step 6: Update `CLAUDE.md` §2b and `docs/index.md`**

§2b is the frozen-results table the next session reads first. Update the numbers, keep the benchmark rule, and replace "the remaining levers are bias-correcting the target, adding input channels, and ensembling" with what this programme actually found. Update the one-sentence pitch in `docs/index.md` and add row 10 to its table.

- [ ] **Step 7: Rebuild the site and commit**

```bash
python build_site.py
git add docs results CLAUDE.md site
git commit -m "Programme complete: <winner> at <score> vs independent Argo"
```

---

## Compute budget

| Task | Runs | GPU time |
|---|---|---|
| 1 — M4 baseline, 3 seeds | 3 × 20 ep × 66 s | ~66 min |
| 4 — anomaly screen | 1 | ~22 min |
| 5 — climatology-as-input screen | 1 | ~25 min |
| 6 — aux screen (+ combined) | 2 | ~46 min |
| 7 — depth-weighted + gradient screens | 2 | ~45 min |
| 8 — 2 extra seeds × up to 3 promoted | ≤6 | ≤~2.5 h |
| 2, 3, 9, 10 | CPU only | — |
| **Total** | | **≈6–7 GPU-hours** |

Fits inside one week of Kaggle's free T4 quota (30 h). No paid compute needed. Everything outside training runs on the Windows box.

## Risks

| Risk | Mitigation |
|---|---|
| Task 1 fails to reproduce 0.890 | Stop. Compare torch/CUDA versions against `results/m4_convlstm_s*.csv` epoch times (66 s/epoch on the Day 2 T4); a different GPU changes timing but should not change the score by >2σ. |
| Instance dies mid-programme | Task 0. Also `train.py` auto-resumes from `checkpoints/<run>.pt`. |
| `extra` channels break the existing configs | Task 5 Step 5 explicitly re-runs `m2_unet` epoch 0 and compares the loss. `n_channels(()) == 7`. |
| An inference path forgets `extra` and silently gets the wrong channel count | The first conv would raise a shape error, not fail silently — but `predict_cube` must read `extra` from the checkpoint's cfg (Task 5 Step 4). This is the exact bug class of doc 09 §7's window bug. |
| Bias correction quietly becomes selection-on-test | `bias_correct.py` refuses `--split test`; `predict_cube --offset` asserts the offset was not fitted on test. Two guards, both in code. |
| Every intervention is null | That *is* a result, and doc 09 §4 predicts it. The bias correction (Task 3) does not depend on any of them, so the programme still ends ~9% better than it started. |
| GPU budget runs out during Task 8 | Promote in val-score order and stop; record which configs did not get their extra seeds rather than reporting a partial spread as a full one. |

## Results — programme complete

All eleven tasks executed. Six training experiments screened, one promoted, the winner
selected on val and read on test exactly once.

### Headline: 0.890 → 0.786 °C

Blended RMSE against 6,056 independent Argo casts, test split 2023–24:

| Run | Blended | vs M4 | Note |
|---|---|---|---|
| **FINAL — 6-model ensemble + Argo bias correction** | **0.786** | **−11.7%** | selected on val, read on test once |
| GLORYS12V1 target itself | 0.728 | — | ceiling for an *uncorrected* model |
| M4 3-seed ensemble + correction | 0.792 | −11.0% | |
| M2+M3 ensemble + correction | 0.818 | −8.1% | from the two Aug-31 checkpoints alone |
| M2 + correction (depth) | 0.844 | −5.2% | fifteen numbers |
| M2 + correction (depth × month) | 0.850 | −4.5% | worse, as predicted out-of-sample |
| 6-model ensemble, uncorrected | 0.859 | −3.5% | |
| M4 3-seed ensemble | 0.862 | −3.1% | |
| M4 ConvLSTM, 3 seeds | 0.890 ± 0.010 | — | doc 09's headline |
| M2 U-Net, 4 seeds | 0.902 ± 0.014 | +1.3% | |
| M3 attention | 0.907 | +1.9% | null |
| M2 + gradient loss, 3 seeds | 0.918 ± 0.005 | +3.1% | negative |
| M2 anomaly, 3 seeds | 0.975 ± 0.024 | +9.6% | negative |
| M0 climatology | 1.160 | +30.3% | floor |

**The error the model adds on top of its own training target falls from 0.162 °C to
0.058 °C — a 64% reduction.** Neither of the two steps that got us there trained a new
model: one averages checkpoints that already existed, the other is a fifteen-number table.

The final model is a six-member ensemble — three seeds of M4 ConvLSTM plus three seeds of
M4 with the inverse-variance depth-weighted loss — with a depth-wise offset fitted on 2022
val Argo subtracted from its output.

### Full depth-wise table (the primary result)

| Depth (m) | M0 clim. | M4 | **FINAL** | GLORYS | MAE | Bias | Corr | R² |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.745 | 0.468 | **0.426** | 0.357 | 0.292 | −0.108 | 0.965 | 0.925 |
| 5 | 0.746 | 0.457 | **0.405** | 0.326 | 0.287 | −0.119 | 0.968 | 0.931 |
| 10 | 0.751 | 0.455 | **0.425** | 0.367 | 0.290 | −0.122 | 0.964 | 0.924 |
| 20 | 0.844 | 0.682 | **0.656** | 0.505 | 0.408 | −0.092 | 0.918 | 0.839 |
| 30 | 0.949 | 0.846 | **0.810** | 0.614 | 0.560 | −0.151 | 0.897 | 0.797 |
| 50 | 1.308 | 1.031 | **1.011** | 0.856 | 0.772 | −0.253 | 0.889 | 0.774 |
| 75 | 1.946 | 1.411 | **1.274** | 1.206 | 0.973 | +0.051 | 0.850 | 0.717 |
| 100 | 2.163 | 1.637 | **1.372** | 1.343 | 1.056 | +0.299 | 0.819 | 0.653 |
| 125 | 1.821 | 1.423 | **1.158** | 1.178 | 0.883 | +0.241 | 0.863 | 0.732 |
| 150 | 1.385 | 1.086 | **0.923** | 0.895 | 0.713 | +0.201 | 0.913 | 0.824 |
| 200 | 0.881 | 0.730 | **0.680** | 0.603 | 0.523 | +0.176 | 0.953 | 0.902 |
| 300 | 0.558 | 0.511 | **0.482** | 0.408 | 0.339 | +0.097 | 0.965 | 0.928 |
| 500 | 0.273 | 0.279 | **0.245** | 0.226 | 0.185 | −0.018 | 0.982 | 0.963 |
| 700 | 0.263 | 0.280 | **0.224** | 0.246 | 0.172 | −0.022 | 0.981 | 0.963 |
| 1000 | 0.237 | 0.254 | **0.216** | 0.243 | 0.165 | −0.014 | 0.973 | 0.945 |
| **blended** | **1.160** | **0.890** | **0.786** | **0.728** | | | | |

**15 of 15 depths beat climatology.** Doc 09's standing weakness — losing to a monthly
climatology at 500, 700 and 1000 m — is gone. At 100 m, R² goes 0.502 → **0.653** and the
bias +0.850 → **+0.299**.

**At 125 m, 700 m and 1000 m the corrected model beats GLORYS itself.** That does not break
the 0.728 ceiling, it defines what the ceiling means: GLORYS bounds any model that only
ever sees GLORYS. Once a correction fitted on *independent observations* is applied, the
target's bias is no longer inherited, and at depths where that bias dominates GLORYS' own
error we pass it. Quote the ceiling as **"the bound for an uncorrected model"**.

### Screening: six interventions, one survivor

Screened at one seed against a val baseline of **0.860 ± 0.004** (M4, three seeds, 2022
Argo). Promote at ≤ 0.870, reject above 0.880.

| Config | Val blended | vs baseline | Decision |
|---|---|---|---|
| M4 baseline (3 seeds) | 0.860 ± 0.004 | — | — |
| **M4 depth-weighted loss** | **0.865** → 0.854 ± 0.010 at 3 seeds | +0.005 | **promoted** |
| M4 gradient loss | 0.875 | +0.015 | rejected — borderline, no depth-wise win |
| M4 + aux channels | 0.889 | +0.029 | rejected |
| M4 + clim + aux | 0.889 | +0.029 | rejected |
| M4 + climatology channels | 0.899 | +0.039 | rejected |
| M4 anomaly | 0.928 | +0.068 | rejected |

**The depth-weighted loss did exactly what this document predicted before it was run.** The
prediction, written down in Task 7: plain MSE already matches the reported metric, so
inverse-variance weighting is a trade, not a win — expect the blended number to worsen and
500–1000 m to improve. Measured at one seed: 50 m 1.142 vs the baseline's 1.040 (worse),
500 m 0.292 vs 0.347 and 1000 m 0.234 vs 0.257 (better). It is not a better model; it is a
*complementary* one, which is why it earns its place in the ensemble rather than replacing
anything.

**The clearest single lesson came from `m4_aux`**, which had the best GLORYS validation RMSE
of the three channel experiments (0.659, better than the baseline typically manages) and was
*worse* against Argo (0.889 vs 0.860). Fitting the reanalysis better made agreement with
observations worse. That is doc 09's benchmark rule demonstrated live, and it is why
climatology-as-input fails too: handing the model the climatology lets it lean harder on
GLORYS' climatological bias structure — it learns the target's errors more faithfully.

**Running total: seven model-side interventions tried (attention, ConvLSTM, gradient loss,
anomaly, climatology channels, auxiliary channels, depth weighting), none of which improved
the observational score on its own. Two output-side steps, both of which did.**

### Ensemble composition, chosen on val

Selected uncorrected, so the bias offset was never fitted and chosen on the same data:

| Composition | Val blended |
|---|---|
| **6 members: M4 ×3 + depth-weighted ×3** | **0.823** |
| depth-weighted ×3 | 0.829 |
| M4 baseline ×3 | 0.831 |

The mixed six wins, and the depth-wise tables say why: the two families are strong in
different layers. Test was opened only after this choice was locked.

### The float-blocked bootstrap: which differences are real

1,000 paired resamples over the 147 floats behind the test casts. Resampling profiles
instead of floats would report intervals ~6.6× too narrow.

| Comparison | Δ blended | 95% CI | Verdict |
|---|---|---|---|
| M3 attention − M2 | −0.0009 | [−0.0110, +0.0105] | **not significant** |
| 6-model ensemble − M4 single seed | −0.0354 | [−0.0427, −0.0278] | significant |
| correction, on top of the ensemble | −0.0730 | [−0.0863, −0.0601] | significant |
| **FINAL − M4 single seed** | **−0.1084** | **[−0.1208, −0.0972]** | **significant** |

Doc 09 called attention a null result from three seeds and an eyeball. It now has a proper
interval and that interval contains zero. Every post-processing gain is unambiguous.

### The bias drifts, and it costs us

Fitted on 2022 val, the final ensemble's offset at 100 m is **+0.590 °C**; its actual
2023–24 bias is **+0.893**. The correction under-shoots, which is why the model gains ~8%
where the GLORYS probe gained 9.3%. A real limitation: an operational version must refit
the offset annually against the most recent Argo.

Depth × month scored 0.850 against depth-only's 0.844 — worse, exactly as the out-of-sample
GLORYS probe predicted (0.675 vs 0.671). 180 bins over ~3,400 casts overfits. The form was
chosen on that prior out-of-sample evidence, not on an in-sample val comparison that would
have flattered the more flexible model.

### Reproduction and leakage

The current code reproduces every published number against the recovered checkpoints:
**M2 = 0.908, M3 = 0.907, M4 = 0.890 (0.895 / 0.897 / 0.879)** — across two machines and
both devices (CPU on the Windows box, T4 on the instance).

`python src/audit_leakage.py` passes **8 / 8**, including the control that catches
normalisation stats fitted on all years rather than train only.

### On the checkpoints Day 2 "lost"

They were never lost. All 28 were sitting on the running g4dn.xlarge, unsynced — the sync
script existed but had to be started by hand, and was not. Task 0's change to
`deploy/setup.sh` now starts it automatically, and everything is mirrored to
`s3://oceanembed-sih26-data/oceanembed/checkpoints/`. Recovering them turned Task 1 from a
retrain into a three-minute scoring pass.

### Three bugs caught by the new self-checks

- **The anomaly residual base leaked into climatology-as-input mode.** `__getitem__` keyed
  the residual base off `self.clim is not None`, which is now also true for
  `extra=("clim",)`. Tasks 4 and 5 would have been the same experiment. Fixed by gating on
  `self.anomaly`.
- **`--ensemble` tried to overwrite a cube it had open for reading**, because the input path
  was relative and the output path absolute. Fixed by resolving both.
- **Ensembling a window=7 model with a window=1 model.** A 7-day model cannot predict a
  split's first six days, so its cube is shorter. `--ensemble` now aligns members on the
  intersection of their time axes and reports how many days it dropped, rather than
  silently averaging misaligned days.

### Compute actually used

About 3.5 GPU-hours on one g4dn.xlarge (T4): six screening runs at 20 epochs plus two extra
seeds of the promoted config, at 65–90 s/epoch. Everything else — all scoring, ensembling,
bias correction, bootstrapping — ran on CPU.

### Files added or changed

`src/audit_leakage.py`, `src/bias_correct.py`, `src/ablation.py` (new); `src/argo_eval.py`
(`match_profiles`, `paired_bootstrap`), `src/config.py` (`n_channels`, `bathy_path`,
channel names), `src/datasets.py` (`extra=` channel sets, `build_bathymetry`),
`src/models/unet.py` (`depth_weight`), `src/train.py` (derive `in_ch`, thread `extra`, build
depth weights), `src/predict_cube.py` (`score_cube`, `--offset`, `--ensemble`, time
alignment), `deploy/setup.sh` (auto-start the checkpoint sync), and six `configs/m4_*.yaml`.

Full tables: `results/ablation_test.md`, `results/ablation_val.md`.

### What is left

The Streamlit demo (doc 06's spec, an explicit PS requirement) and track B1 INCOIS gridded
Argo. Nothing else in this programme is outstanding. Do not add model capacity — seven
interventions now say the same thing.
