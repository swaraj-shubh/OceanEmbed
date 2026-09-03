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

SIX alternative forms were then screened under float-blocked cross-validation and ALL are
null -- basin, season, latitude band, basin x season, and a per-depth linear a + b*pred all
land within +/-0.6% of this one against +/-1.9% fold noise. `src/correction_forms.py` is
the instrument; docs/12 sec.2 is the table. Do not stratify this further: after the offset
the residual is variance, not bias, and no lookup table reaches it.
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

    by_month=True fits [15, 12] instead. Months come from match_profiles' own `times`, so
    the bins can never fall out of step with the matched columns -- deriving them again
    from `profiles` would silently mis-align the moment a cast failed to match.
    """
    pred, obs, _, times = match_profiles(cube, profiles, max_days)
    assert pred.shape[1] >= min_n, f"only {pred.shape[1]} casts matched; refusing to fit"
    d = pred - obs
    flat = np.nanmean(d, axis=1)
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
    print(f"fitted on {a.split} ({prof.profile.nunique()} casts available) -> {out.name}")
    for z, v in zip(DEPTHS, off if off.ndim == 1 else off.mean(1)):
        print(f"  {z:5d} m  {v:+.3f}")


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

    off = fit_offset(cube, prof)
    assert np.allclose(off, bias, atol=1e-6), f"offset not recovered: {off - bias}"
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
