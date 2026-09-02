"""Validation track B2 -- score a prediction cube against raw Argo profiles.

Argo is never an input or a target (CLAUDE.md rule 3). Matching is:
  profile -> nearest 0.25 deg grid cell -> nearest available day (within `max_days`)
  -> profile interpolated onto the 15 SIH depths, with the OceanDepths acceptance rule:
     reject a level if the nearest observed depth is farther than max(0.1*z, 10 m)
     (no extrapolation past the ends of the profile either).

Profiles come in as a DataFrame with columns: time, lat, lon, pres (depth m), temp, and a
`profile` id grouping the levels of one cast.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
from config import DEPTHS

TARGET = np.asarray(DEPTHS, float)
TOL = np.maximum(0.1 * TARGET, 10.0)


def interp_profile(z, t, depths=TARGET, tol=TOL):
    """Observed (z, t) -> [len(depths)] with NaN where the acceptance rule rejects."""
    z, t = np.asarray(z, float), np.asarray(t, float)
    ok = np.isfinite(z) & np.isfinite(t)
    z, t = z[ok], t[ok]
    if z.size < 2:
        return np.full(len(depths), np.nan)
    o = np.argsort(z)
    z, t = z[o], t[o]
    # np.interp clamps outside [z[0], z[-1]] rather than refusing; the gap test below is
    # what enforces the acceptance rule, and it is the correct gate. Refusing at the ends
    # instead would reject EVERY profile at 0 m -- Argo floats surface at ~2-5 dbar, never
    # at exactly 0 -- and 0 m is one of the six depths in the headline metrics table.
    out = np.interp(depths, z, t)
    gap = np.abs(z[np.searchsorted(z, depths).clip(0, len(z) - 1)] - depths)
    gap = np.minimum(gap, np.abs(z[(np.searchsorted(z, depths) - 1).clip(0, len(z) - 1)] - depths))
    return np.where(gap <= tol, out, np.nan)


def match_profiles(cube, profiles, max_days=1):
    """cube: xr.DataArray (time, depth, lat, lon) in degC. profiles: DataFrame (module doc).

    Returns (pred[15, N], obs[15, N], floats[N], times[N]) for the N matched casts.

    `floats` is the Argo platform number, so callers can block-resample by float: the 6,448
    test-split casts come from only 147 floats, and two casts from one float ten days apart
    in the same water mass are not independent samples -- a profile-level bootstrap would
    report an interval about sqrt(6448/147) ~ 6.6x too narrow. `times` is returned so
    callers that bin by calendar month (bias_correct) never have to re-derive the match
    order and risk falling out of step with the returned columns.
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
        P.append(pred)
        O.append(obs)
        F.append(str(pid).split("_")[0])
        T.append(pd.Timestamp(g.time.iloc[0]))
    if not P:
        z = np.zeros((len(TARGET), 0))
        return z, z, np.array([], dtype=object), pd.DatetimeIndex([])
    return np.array(P).T, np.array(O).T, np.array(F, dtype=object), pd.DatetimeIndex(T)


def evaluate_argo(cube, profiles, max_days=1):
    """Returns (depth-wise table, n_profiles_matched)."""
    from metrics import depthwise
    pred, obs, _, _ = match_profiles(cube, profiles, max_days)
    return depthwise(pred, obs), pred.shape[1]


def paired_bootstrap(cube_a, cube_b, profiles, n=1000, seed=0, max_days=1):
    """95% CI on (blended RMSE of a) - (blended RMSE of b), resampling FLOATS.

    Paired: both models are scored on the same resampled casts, so the shared difficulty of
    a hard water mass cancels and only the difference between the models is left. Blocked by
    float for the reason in match_profiles' docstring -- resampling profiles would turn seed
    noise into a publishable result.
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


if __name__ == "__main__":
    import xarray as xr

    # a full-resolution cast is accepted everywhere; the deep levels of a shallow cast are not
    z = np.arange(0, 1010, 10.0)
    prof = 20 - 15 * (1 - np.exp(-z / 300))
    got = interp_profile(z, prof)
    assert np.isfinite(got).all()
    assert np.allclose(got, np.interp(TARGET, z, prof))

    shallow = interp_profile(z[z <= 200], prof[z <= 200])
    assert np.isfinite(shallow[TARGET <= 200]).all()
    assert np.isnan(shallow[TARGET > 200]).all(), "extrapolated past the profile"

    # a real cast starts a few dbar down, never at 0; 0 m must still be accepted there
    deep_start = z[(z >= 2.8) & (z <= 400)]
    surf = interp_profile(deep_start, np.interp(deep_start, z, prof))
    assert np.isfinite(surf[TARGET == 0][0]), "0 m rejected -- Argo never samples exactly 0"
    assert np.isnan(interp_profile(z[z >= 40], prof[z >= 40])[TARGET == 0][0]), \
        "shallowest obs 40 m away from 0 m -- must reject"

    sparse = interp_profile([0.0, 400.0, 1000.0], [30.0, 10.0, 4.0])
    assert np.isnan(sparse[TARGET == 200][0]), "200 m is 200 m from the nearest obs -- reject"
    assert np.isfinite(sparse[TARGET == 0][0]) and np.isfinite(sparse[TARGET == 1000][0])

    # a cube that equals the profile everywhere must score exactly zero
    lat, lon = np.arange(0.125, 3, 0.25), np.arange(55.125, 58, 0.25)
    t = pd.date_range("2022-01-01", periods=3)
    cube = xr.DataArray(np.broadcast_to(np.interp(TARGET, z, prof)[None, :, None, None],
                                        (3, 15, len(lat), len(lon))).copy(),
                        coords={"time": t, "depth": DEPTHS, "lat": lat, "lon": lon},
                        dims=("time", "depth", "lat", "lon"))
    df = pd.DataFrame({"profile": 1, "time": t[1], "lat": 1.1, "lon": 56.2,
                       "pres": z, "temp": prof})
    tab, n = evaluate_argo(cube, df)
    assert n == 1 and np.allclose(tab["rmse"], 0.0, atol=1e-6)

    far = df.assign(profile=2, lat=40.0)                    # outside the box -> no match
    assert evaluate_argo(cube, far)[1] == 0
    old = df.assign(profile=3, time=pd.Timestamp("2021-06-01"))
    assert evaluate_argo(cube, old)[1] == 0, "stale profile must not match"
    print(f"argo_eval self-check OK -- matched {n} profile, RMSE {tab['rmse'].max():.2e}")

    # match_profiles must hand back one column, one float id and one timestamp per cast
    P, O, F, T = match_profiles(cube, df)
    assert P.shape == O.shape == (len(DEPTHS), 1) and len(F) == 1 and len(T) == 1
    assert F[0] == "1", "float id must be the part of the profile id before the underscore"

    # A cube uniformly 0.5 degC warm must give a strictly positive interval; the same cube
    # against itself must bracket zero exactly. Without the second check the first one
    # would pass for a bootstrap that ignored its inputs.
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
    assert abs(pt - 0.5) < 1e-6 and lo_ > 0, (pt, lo_, hi_)
    pt0, lo0, hi0 = paired_bootstrap(base, base, pr, n=200)
    assert abs(pt0) < 1e-9 and lo0 <= 0 <= hi0, (pt0, lo0, hi0)
    print(f"paired_bootstrap self-check OK -- +0.5 degC reads {pt:.3f} [{lo_:.3f}, {hi_:.3f}]")
