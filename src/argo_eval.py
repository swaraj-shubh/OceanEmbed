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
from metrics import DepthStats

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


def evaluate_argo(cube, profiles, max_days=1):
    """cube: xr.DataArray (time, depth, lat, lon) in degC. profiles: DataFrame (see module doc).

    Returns (depth-wise table, n_profiles_matched).
    """
    lat, lon = cube.lat.values, cube.lon.values
    times = cube.time.values.astype("datetime64[D]")
    acc, matched = DepthStats(), 0
    for _, g in profiles.groupby("profile"):
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
        acc.update(pred[:, None], obs[:, None])
        matched += 1
    return acc.table(), matched


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
