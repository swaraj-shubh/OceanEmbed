"""Data access for the Streamlit demo. Everything the app reads goes through here.

    python app/loader.py        # self-check against the shipped bundle

The app is deliberately dumb about where data comes from: it loads one committed bundle
(app/demo_data/, built by scripts/build_demo_bundle.py) and never touches the 3.1 GB store,
a checkpoint, or torch. A click is an array lookup.

Caching is Streamlit's, but every function here works without Streamlit so the self-check
and any script can call them.
"""
import json
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "app" / "demo_data"

# Reuse the project's own profile interpolation rather than writing a second one: the
# acceptance rule (reject a level if the nearest observation is farther than max(0.1z, 10 m))
# is what every reported number in docs/09-11 was computed with, so the metrics a judge sees
# on screen must come from the same code or they are a different measurement.
sys.path.append(str(ROOT / "src"))
from argo_eval import interp_profile  # noqa: E402

# Streamlit's cache when running in the app, a plain lru_cache otherwise, so this module
# stays importable (and testable) outside Streamlit.
try:
    import streamlit as st
    cache = st.cache_data(show_spinner=False)
except ModuleNotFoundError:                                   # pragma: no cover
    def cache(fn):
        return lru_cache(maxsize=None)(fn)

CHANNEL_LABEL = {
    "sst": ("Sea surface temperature", "°C"),
    "sss": ("Sea surface salinity", "PSU"),
    "sla": ("Sea level anomaly", "m"),
    "cur_u": ("Current — eastward", "m/s"),
    "cur_v": ("Current — northward", "m/s"),
    "wind_u": ("Wind — eastward", "m/s"),
    "wind_v": ("Wind — northward", "m/s"),
}

REGIONS = {                       # lon bounds; both basins share the full latitude range
    "Both basins": (55.0, 100.0),
    "Arabian Sea": (55.0, 78.0),
    "Bay of Bengal": (78.0, 100.0),
}


class BundleMissing(RuntimeError):
    """Raised with an actionable message rather than a stack trace in the UI."""


def _need(path):
    if not path.exists():
        raise BundleMissing(
            f"{path.relative_to(DATA.parent)} is missing.\n\n"
            "Build the demo bundle first:\n    python scripts/build_demo_bundle.py")
    return path


@cache
def manifest():
    return json.loads(_need(DATA / "manifest.json").read_text())


@cache
def inputs():
    """7 surface fields, (time, lat, lon) each. int16 on disk, float32 after decoding."""
    return xr.open_dataset(_need(DATA / "inputs.nc")).load()


@cache
def prediction():
    """The frozen bias-corrected ensemble: (time, depth, lat, lon) in degC."""
    return xr.open_dataset(_need(DATA / "pred.nc")).thetao.load()


@cache
def truth():
    """GLORYS12V1, the training target, for the side-by-side toggle. NOT ground truth --
    it carries a +0.72 degC warm bias at 100 m, which is the whole point of docs/09 sec.4."""
    return xr.open_dataset(_need(DATA / "truth.nc")).thetao.load()


@cache
def argo():
    """Independent Argo casts inside the window. Never a model input (CLAUDE.md rule 3)."""
    df = pd.read_parquet(_need(DATA / "argo.parquet"))
    df["time"] = pd.to_datetime(df["time"])
    return df


@cache
def metrics(name):
    """One metric CSV from the bundle, e.g. 'ens_mix6_bc_test_argo.csv'."""
    return pd.read_csv(_need(DATA / "metrics" / name))


def dates():
    return pd.DatetimeIndex(prediction().time.values)


def depths():
    return [int(d) for d in prediction().depth.values]


def field(date, depth, source="prediction"):
    """One 2-D map: (lat, lon) at `depth` on `date`. source: prediction | truth | error."""
    p = prediction().sel(time=date, depth=depth)
    if source == "prediction":
        return p
    t = truth().sel(time=date, depth=depth)
    return t if source == "truth" else (p - t)


def profile(date, lat, lon, source="prediction"):
    """Full 0-1000 m column at the nearest grid cell. Returns (depths, values)."""
    da = (prediction() if source == "prediction" else truth())
    col = da.sel(time=date).sel(lat=lat, lon=lon, method="nearest")
    return np.asarray(col.depth.values, float), np.asarray(col.values, float)


def nearest_argo(date, lat, lon, max_days=3, max_deg=1.5):
    """Closest independent Argo cast to a click, or None.

    Ranked by great-circle-ish distance in degrees, not by time: within a few days the
    ocean has not moved much at these depths, but 1.5 deg is already ~165 km and the
    profile comparison stops being meaningful much beyond that.
    """
    df = argo()
    win = df[(df.time >= pd.Timestamp(date) - pd.Timedelta(days=max_days))
             & (df.time <= pd.Timestamp(date) + pd.Timedelta(days=max_days))]
    if win.empty:
        return None
    first = win.groupby("profile")[["lat", "lon", "time"]].first()
    d = np.hypot(first.lat - lat, (first.lon - lon) * np.cos(np.radians(lat)))
    if d.min() > max_deg:
        return None
    pid = d.idxmin()
    cast = win[win.profile == pid].sort_values("pres")
    return {"profile": str(pid), "lat": float(first.lat[pid]), "lon": float(first.lon[pid]),
            "time": pd.Timestamp(first.time[pid]), "distance_deg": float(d.min()),
            "pres": cast.pres.to_numpy(float), "temp": cast.temp.to_numpy(float)}


def argo_comparison(date, lat, lon, max_days=3, max_deg=1.5):
    """Match a click to the nearest Argo cast and score our profile against it.

    Returns None when no cast is near enough. Otherwise adds `depths`, `pred`, `obs`
    (obs on the project's 15 depths, NaN where the acceptance rule rejects a level) and
    the same RMSE / bias / correlation the reported tables use -- computed here over one
    cast, so the numbers are honest but noisy, and the UI must say so.
    """
    cast = nearest_argo(date, lat, lon, max_days, max_deg)
    if cast is None:
        return None
    obs = interp_profile(cast["pres"], cast["temp"])
    zz, pred = profile(date, cast["lat"], cast["lon"])
    ok = np.isfinite(obs) & np.isfinite(pred)
    d = pred[ok] - obs[ok]
    cast.update({
        "depths": zz, "pred": pred, "obs": obs, "n_levels": int(ok.sum()),
        "rmse": float(np.sqrt((d ** 2).mean())) if ok.sum() else float("nan"),
        "bias": float(d.mean()) if ok.sum() else float("nan"),
        "corr": float(np.corrcoef(pred[ok], obs[ok])[0, 1]) if ok.sum() > 2 else float("nan"),
    })
    return cast


def land_mask():
    """True where the model was never supervised. Prediction cubes carry NaN there, which
    is what docs/09 sec.2 fixed -- scoring against unconstrained output moved 500 m RMSE
    from 0.30 to 0.94 for the anomaly model."""
    return np.isnan(prediction().isel(time=0, depth=0).values)


if __name__ == "__main__":
    m = manifest()
    print(f"bundle: {m['window']['days']} days {m['window']['start']}..{m['window']['end']}"
          f"  |  {m['argo_profiles']} Argo casts  |  git {m['git_sha']}")

    p, t, x = prediction(), truth(), inputs()
    assert p.dims == ("time", "depth", "lat", "lon"), p.dims
    assert p.shape == t.shape, (p.shape, t.shape)
    assert depths() == m["depths_m"], "depth axis does not match the manifest"
    assert list(x.data_vars) == m["channels"], "input channels do not match the manifest"
    assert x.sizes["time"] == p.sizes["time"], "inputs and prediction cover different days"
    assert len(dates()) == m["window"]["days"]

    # The demo must only ever show the held-out period. If this fires, the bundle is
    # showing a judge dates the model trained on.
    assert str(dates().min().date()) >= "2023-01-01", "bundle escapes the test split"

    # int16 packing must be transparent: physical values, not counts.
    v = p.sel(depth=0).values
    assert np.nanmin(v) > -5 and np.nanmax(v) < 40, f"0 m looks unscaled: {np.nanmin(v)}..{np.nanmax(v)}"
    assert np.isnan(v).any(), "no land in the prediction -- masking was lost"
    assert land_mask().mean() > 0.1, "land mask covers implausibly little of the grid"

    d0 = dates()[len(dates()) // 2]
    zz, vv = profile(d0, 15.0, 88.0)          # Bay of Bengal, open water
    assert len(zz) == 15 and np.isfinite(vv).all(), "profile has gaps in open water"
    assert vv[0] > vv[-1], "profile is not warmer at the surface than at 1000 m"

    err = field(d0, 100, "error")
    assert err.shape == field(d0, 100).shape

    hit = None
    for d in dates():                          # find any day with a cast to prove matching
        hit = nearest_argo(d, 15.0, 88.0)
        if hit:
            break
    assert hit is not None, "no Argo cast matched anywhere in the window"
    assert hit["pres"].min() < 50 and len(hit["pres"]) > 5
    print(f"nearest Argo demo: {hit['profile']} at {hit['distance_deg']:.2f} deg, "
          f"{len(hit['pres'])} levels")

    tab = metrics("ens_mix6_bc_test_argo.csv")
    assert {"depth_m", "rmse", "bias", "corr"} <= set(tab.columns)
    print(f"loader self-check OK -- {p.sizes['lat']}x{p.sizes['lon']} grid, "
          f"{len(depths())} depths, headline 100 m RMSE "
          f"{float(tab[tab.depth_m == 100].rmse.iloc[0]):.3f} degC")
