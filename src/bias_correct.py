"""Post-hoc Argo bias correction -- a 15-number lookup table, fitted on VAL Argo.

    python src/bias_correct.py --cube results/m4_convlstm_s1_best_val_cube.nc --split val
    python src/predict_cube.py --ckpt ... --split test --offset results/m4_..._offset.json

The network never sees Argo. This stage does, so it is reported as its own row in the
ablation table and never folded into the model's number. Fitting on the test split is
refused outright: that is the one way this stage could become a lie.

Measured with GLORYS itself as the probe (2023-24 test Argo, 6389 casts): raw 0.734,
corrected 0.665 with a val-fitted depth offset, against an oracle ceiling of 0.663 -- 96%
of the achievable gain from fifteen numbers. depth x month scored 0.675: 180 bins over
~3,400 casts overfits.

The bias also DRIFTS (+0.475 degC at 100 m in 2021 vs +0.716 in 2023-24), which is why
fitting on val beats fitting on train, and why an operational version would refit annually.
"""
import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.append(str(Path(__file__).resolve().parent))
from argo_eval import match_profiles
from config import DEPTHS, INTERIM, SPLITS


def float_se(d, floats):
    """Standard error of each depth's mean residual, BLOCKING BY FLOAT.

    Averaging within a float first is the whole point: 3,107 val casts come from 83 floats,
    and casts from one float days apart in the same water mass are not independent. A
    cast-level standard error is roughly sqrt(3107/83) ~ 6x too small, which would make
    every depth look significant.

    Depths seen by fewer than two floats get an infinite standard error, so shrinkage takes
    them to zero rather than trusting one float's water mass.
    """
    uf = np.unique(floats)
    means = np.full((d.shape[0], uf.size), np.nan)
    for j, f in enumerate(uf):
        col = d[:, floats == f]
        ok = np.isfinite(col).any(axis=1)
        if ok.any():
            means[ok, j] = np.nanmean(col[ok], axis=1)
    n = np.isfinite(means).sum(axis=1)
    with warnings.catch_warnings():                 # n <= 1 -> ddof warning; handled below
        warnings.simplefilter("ignore", RuntimeWarning)
        with np.errstate(invalid="ignore", divide="ignore"):
            sd = np.nanstd(means, axis=1, ddof=1)
    return np.where(n > 1, sd / np.sqrt(np.maximum(n, 1)), np.inf)


def shrink_offset(flat, se):
    """Empirical-Bayes weight w = t^2/(1+t^2), i.e. offset^2 / (offset^2 + SE^2).

    Six of fifteen val depths have |t| < 3 -- 0, 5, 10, 200, 300 and 500 m carry offsets
    indistinguishable from zero, so shrinking them looks obviously right.

    MEASURED, AND IT IS NOT: 0.7921 raw vs 0.7928 shrunk. Two reasons, both visible in the
    per-depth table. The noise-dominated offsets are also *tiny* (0.003 to 0.051 degC), so
    zeroing them changes almost nothing; and at 200 m the val offset of +0.051 was already
    an UNDER-correction for the test period (residual bias +0.171), so shrinking it further
    made that depth worse. The real damage -- bias flipping from +0.198 to -0.253 at 50 m --
    is at |t| = 9.0, a well-measured offset whose problem is that the thermocline moved
    between 2022 and 2023-24. That is drift, and shrinkage cannot touch it.

    Kept, off by default, because the standard errors it computes are what make the
    correction auditable.
    """
    flat = np.asarray(flat, float)
    se = np.asarray(se, float)
    with np.errstate(invalid="ignore", divide="ignore"):
        w = flat ** 2 / (flat ** 2 + se ** 2)
    return np.nan_to_num(flat * w, nan=0.0), np.nan_to_num(w, nan=0.0)


def fit_offset(cube, profiles, by_month=False, max_days=1, min_n=20, shrink=False):
    """Mean (prediction - observation) per depth over the matched casts.

    shrink=True scales each depth's offset by how well measured it is (see shrink_offset).
    It defaults to FALSE because it was tried and it does not help: on the M4 ensemble,
    0.7921 raw vs 0.7928 shrunk against test Argo. That gap is far below the seed noise of
    +/-0.010, so the two are a statistical tie and raw is kept only because it is what the
    frozen result used. The diagnostics shrink_offset needs are still written to the offset
    file either way, which is the part that earned its place.

    by_month=True fits [15, 12] instead; sparse months fall back to the depth-only fit.

    Months come from match_profiles' own `times`, so the bins can never fall out of step
    with the matched columns -- deriving them again from `profiles` would silently
    mis-align the moment a cast failed to match.
    """
    pred, obs, floats, times = match_profiles(cube, profiles, max_days)
    assert pred.shape[1] >= min_n, f"only {pred.shape[1]} casts matched; refusing to fit"
    d = pred - obs
    flat = np.nanmean(d, axis=1)
    if shrink:
        flat, _ = shrink_offset(flat, float_se(d, floats))
    if not by_month:
        return flat
    off = np.tile(flat[:, None], (1, 12))     # months with too few casts keep the flat fit
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
    p.add_argument("--shrink", action="store_true",
                   help="scale each offset by how well measured it is. Measured as a null "
                        "(0.7921 raw vs 0.7928 shrunk); off by default")
    a = p.parse_args()
    assert a.split != "test", (
        "refusing to fit an offset on the test split -- fit on val (CLAUDE.md rule 3)")

    cube = xr.open_dataarray(a.cube)
    prof = pd.read_parquet(a.argo)
    prof["time"] = pd.to_datetime(prof["time"]).dt.tz_localize(None)
    lo, hi = SPLITS[a.split]
    prof = prof[(prof.time >= lo) & (prof.time <= pd.Timestamp(hi) + pd.Timedelta(days=1))]

    shrink = a.shrink
    off = fit_offset(cube, prof, by_month=a.by_month, max_days=a.max_days, shrink=shrink)

    # Diagnostics alongside the offset so the shrinkage is auditable, not a black box.
    pred, obs, floats, _ = match_profiles(cube, prof, a.max_days)
    raw = np.nanmean(pred - obs, axis=1)
    se = float_se(pred - obs, floats)
    stem = Path(a.cube).stem.replace(f"_{a.split}_cube", "")
    out = Path(a.cube).with_name(
        stem + ("_offset_month.json" if a.by_month else "_offset.json"))
    out.write_text(json.dumps(
        {"split_fitted_on": a.split, "by_month": a.by_month, "shrunk": shrink,
         "n_casts": int(pred.shape[1]), "n_floats": int(np.unique(floats).size),
         "depths": DEPTHS, "offset": off.tolist(),
         "offset_raw": np.nan_to_num(raw).tolist(),
         "standard_error": np.where(np.isfinite(se), se, -1).tolist()}, indent=2))
    print(f"fitted on {a.split} ({pred.shape[1]} casts / {np.unique(floats).size} floats)"
          f"{' with shrinkage' if shrink else ' WITHOUT shrinkage'} -> {out.name}")
    applied = off if off.ndim == 1 else off.mean(1)
    print(f"  {'depth':>6} {'raw':>8} {'SE':>7} {'|t|':>6} {'applied':>9}")
    for z, r, s, v in zip(DEPTHS, raw, se, applied):
        t = abs(r / s) if np.isfinite(s) and s > 0 else 0.0
        print(f"  {z:6d} {r:+8.3f} {s:7.3f} {t:6.1f} {v:+9.3f}"
              + ("   <- shrunk away" if abs(v) < abs(r) / 2 else ""))


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main()
        raise SystemExit

    # A cube carrying a known constant per-depth bias must have exactly that bias recovered,
    # and applying the fit must drive the residual to zero.
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

    # shrink=False: a clean constant bias must come back exactly.
    off = fit_offset(cube, prof, shrink=False)
    assert np.allclose(off, bias, atol=1e-6), f"offset not recovered: {off - bias}"

    # --- shrinkage: real bias survives, pure noise does not ---------------------------
    # 40 identical casts means zero between-float spread, so SE is 0 and shrinkage is a
    # no-op. That is the correct behaviour and worth pinning.
    off_s = fit_offset(cube, prof, shrink=True)
    assert np.allclose(off_s, bias, atol=1e-6), "a perfectly measured bias must not shrink"

    # Now the case that matters: no true bias, only float-to-float scatter. Every depth
    # must shrink to ~0, because this is exactly the situation at 0/5/10/200/300/500 m.
    rng2 = np.random.default_rng(7)
    noisy = []
    for k in range(60):
        jitter = rng2.normal(0.0, 0.6, size=len(DEPTHS))     # this float's water mass
        noisy.append(pd.DataFrame({
            "profile": f"39{k:05d}_1", "time": t[k % len(t)],
            "lat": float(rng2.choice(lat)), "lon": float(rng2.choice(lon)),
            "pres": np.asarray(DEPTHS, float),
            "temp": truth + jitter}))
    noisy = pd.concat(noisy, ignore_index=True)
    clean = xr.DataArray(
        np.broadcast_to(truth[None, :, None, None],
                        (len(t), len(DEPTHS), len(lat), len(lon))).copy(),
        coords={"time": t, "depth": DEPTHS, "lat": lat, "lon": lon},
        dims=("time", "depth", "lat", "lon"))
    raw_n = fit_offset(clean, noisy, shrink=False)
    shr_n = fit_offset(clean, noisy, shrink=True)
    p_n, o_n, f_n, _ = match_profiles(clean, noisy)
    t_n = np.abs(raw_n) / float_se(p_n - o_n, f_n)

    # The contract, stated as it actually is rather than as a magic threshold. w =
    # t^2/(1+t^2) is a Bayesian compromise, not a hypothesis test: at t = 2 it still keeps
    # 80%, because at t = 2 there IS some evidence. What it guarantees is the two things
    # below -- it never inflates an offset, and it guts the ones with no evidence at all.
    assert np.all(np.abs(shr_n) <= np.abs(raw_n) + 1e-12), "shrinkage inflated an offset"
    weak = t_n < 1.0
    assert weak.any(), "test setup produced no weak depths; cannot check the shrinkage"
    assert np.all(np.abs(shr_n[weak]) < 0.5 * np.abs(raw_n[weak]) + 1e-12), \
        "depths with no evidence (|t| < 1) kept more than half their offset"

    # A depth seen by a single float cannot have its spread estimated, so it must shrink
    # to exactly zero rather than trust that one water mass.
    one = noisy[noisy.profile == "3900000_1"]
    assert np.allclose(fit_offset(clean, one, min_n=1, shrink=True), 0.0), \
        "a single-float fit must shrink to zero"
    pred, obs, floats, times = match_profiles(apply_offset(cube, off), prof)
    assert np.allclose(depthwise(pred, obs)["rmse"], 0.0, atol=1e-6), \
        "correction did not null the bias"
    assert len(np.unique(floats)) == 40, "float ids not parsed from the profile id"
    assert len(times) == 40, "one timestamp per matched cast, in match order"

    offm = fit_offset(cube, prof, by_month=True)
    assert offm.shape == (len(DEPTHS), 12)
    assert np.allclose(apply_offset(cube, offm).isel(time=0).values[:, 0, 0], truth,
                       atol=1e-6), "month form did not null the bias"
    # apply_offset must not reorder dims -- the cube is written straight to NetCDF after
    assert apply_offset(cube, offm).dims == cube.dims

    try:
        fit_offset(cube, prof[prof.profile < "2900005_1"], min_n=20)
        raise SystemExit("FAIL: should have refused, too few casts matched")
    except AssertionError as e:
        assert "refusing to fit" in str(e), e
    print("bias_correct self-check OK -- offset recovered to 1e-6, month form applies")
