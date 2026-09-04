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
from metrics import blend_all  # noqa: E402

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


def _quarter(date):
    """'2023Q1' etc -- which chunk a date lives in. One calendar day is always in exactly
    one quarter, so callers never need more than one chunk for a single-date lookup."""
    return str(pd.Period(pd.Timestamp(date), freq="Q"))


@cache
def _inputs_q(q):
    """7 surface fields for one quarter, (time, lat, lon) each. int16 on disk."""
    return xr.open_dataset(_need(DATA / f"inputs_{q}.nc")).load()


@cache
def _prediction_q(q):
    """The frozen bias-corrected ensemble for one quarter: (time, depth, lat, lon) degC."""
    return xr.open_dataset(_need(DATA / f"pred_{q}.nc")).thetao.load()


@cache
def _truth_q(q):
    """GLORYS12V1 for one quarter, for the side-by-side toggle. NOT ground truth -- it
    carries a +0.72 degC warm bias at 100 m, which is the whole point of docs/09 sec.4."""
    return xr.open_dataset(_need(DATA / f"truth_{q}.nc")).thetao.load()


def inputs(date):
    """7 surface fields at one date, (lat, lon) each. Loads (and caches) only the quarter
    that date falls in -- the app never has more than one quarter's worth in memory."""
    return _inputs_q(_quarter(date)).sel(time=date)


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


@cache
def final_vs_glorys():
    """The frozen model's blended skill vs the training target it was measured against,
    every standard metric -- not just RMSE. Source: the same two CSVs the Skill tab's
    RMSE-vs-depth curve already reads. Returns (final, glorys), each a blend_all() dict."""
    fin = blend_all(metrics("ens_mix6_bc_test_argo.csv"))
    glo = blend_all(metrics("GLORYS_target_test_argo.csv"))
    return fin, glo


def dates():
    """Full test-split date list, straight from the manifest -- reading it back off the
    NetCDFs would mean opening all 8 quarter files just to populate the slider."""
    return pd.DatetimeIndex(manifest()["dates"])


def depths():
    return [int(d) for d in manifest()["depths_m"]]


def field(date, depth, source="prediction"):
    """One 2-D map: (lat, lon) at `depth` on `date`. source: prediction | truth | error."""
    q = _quarter(date)
    p = _prediction_q(q).sel(time=date, depth=depth)
    if source == "prediction":
        return p
    t = _truth_q(q).sel(time=date, depth=depth)
    return t if source == "truth" else (p - t)


def profile(date, lat, lon, source="prediction"):
    """Full 0-1000 m column at the nearest grid cell. Returns (depths, values)."""
    q = _quarter(date)
    da = (_prediction_q(q) if source == "prediction" else _truth_q(q))
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
    from 0.30 to 0.94 for the anomaly model. Land is static, so any single day answers this
    -- the first day of the first quarter costs one file load, not all eight."""
    q0 = manifest()["quarters"][0]
    return np.isnan(_prediction_q(q0).isel(time=0, depth=0).values)


if __name__ == "__main__":
    m = manifest()
    all_dates = dates()
    print(f"bundle: {len(all_dates)} days {str(all_dates.min().date())}.."
          f"{str(all_dates.max().date())}  |  {len(m['quarters'])} quarters "
          f"{m['quarters']}  |  {m['argo_profiles']} Argo casts  |  git {m['git_sha']}")

    assert depths() == m["depths_m"], "depth axis does not match the manifest"
    # The demo must only ever show the held-out period. If this fires, the bundle is
    # showing a judge dates the model trained on.
    assert str(all_dates.min().date()) >= "2023-01-01", "bundle escapes the test split"

    # Every quarter file must exist, cover the days the manifest claims, and share one grid
    # -- checked for ALL of them, not just one, since a chunked bundle can go stale one file
    # at a time (e.g. a rebuild that only reran the last quarter).
    expected_by_q = {}
    for d in all_dates:
        expected_by_q.setdefault(_quarter(d), 0)
        expected_by_q[_quarter(d)] += 1
    assert sorted(expected_by_q) == sorted(m["quarters"]), \
        "manifest date list and quarter list disagree"

    grid_shape = None
    for q in m["quarters"]:
        p, t, x = _prediction_q(q), _truth_q(q), _inputs_q(q)
        assert p.dims == ("time", "depth", "lat", "lon"), (q, p.dims)
        assert p.shape == t.shape, (q, p.shape, t.shape)
        assert list(x.data_vars) == m["channels"], (q, "input channels do not match")
        assert x.sizes["time"] == p.sizes["time"] == expected_by_q[q], \
            (q, "day count does not match the manifest")
        shape = (p.sizes["lat"], p.sizes["lon"])
        grid_shape = grid_shape or shape
        assert shape == grid_shape, (q, "grid shape differs from an earlier quarter")
        # int16 packing must be transparent: physical values, not counts.
        v = p.sel(depth=0).values
        assert np.nanmin(v) > -5 and np.nanmax(v) < 40, \
            (q, f"0 m looks unscaled: {np.nanmin(v)}..{np.nanmax(v)}")
        assert np.isnan(v).any(), (q, "no land in the prediction -- masking was lost")
    print(f"all {len(m['quarters'])} quarters verified individually "
          f"({sum(expected_by_q.values())} days total)")

    assert land_mask().mean() > 0.1, "land mask covers implausibly little of the grid"

    # Exercise the quarter boundary explicitly: the first and last day of the bundle, plus
    # the middle, must each resolve through _quarter() to a file that actually has that day
    # -- a boundary off-by-one would only show up at the edges, never in the middle.
    for d0 in (all_dates[0], all_dates[len(all_dates) // 2], all_dates[-1]):
        zz, vv = profile(d0, 15.0, 88.0)      # Bay of Bengal, open water
        assert len(zz) == 15 and np.isfinite(vv).all(), (d0, "profile has gaps in open water")
        assert vv[0] > vv[-1], (d0, "profile is not warmer at the surface than at 1000 m")
        err = field(d0, 100, "error")
        assert err.shape == field(d0, 100).shape

    hit = None
    for d in all_dates:                        # find any day with a cast to prove matching
        hit = nearest_argo(d, 15.0, 88.0)
        if hit:
            break
    assert hit is not None, "no Argo cast matched anywhere in the full test split"
    assert hit["pres"].min() < 50 and len(hit["pres"]) > 5
    print(f"nearest Argo demo: {hit['profile']} at {hit['distance_deg']:.2f} deg, "
          f"{len(hit['pres'])} levels")

    tab = metrics("ens_mix6_bc_test_argo.csv")
    assert {"depth_m", "rmse", "bias", "corr"} <= set(tab.columns)

    # The blended FINAL-vs-GLORYS comparison: known values, checked to 3dp so a future
    # change to either source CSV is caught here before it silently drifts in the UI.
    fin, glo = final_vs_glorys()
    for k, v in {"rmse": 0.786, "mae": 0.511, "bias": 0.012, "corr": 0.926, "r2": 0.853}.items():
        assert abs(fin[k] - v) < 5e-3, f"final_vs_glorys()[FINAL][{k}] drifted: {fin[k]:.4f}"
    for k, v in {"rmse": 0.728, "mae": 0.442, "bias": 0.193, "corr": 0.947, "r2": 0.877}.items():
        assert abs(glo[k] - v) < 5e-3, f"final_vs_glorys()[GLORYS][{k}] drifted: {glo[k]:.4f}"

    print(f"loader self-check OK -- {grid_shape[0]}x{grid_shape[1]} grid, "
          f"{len(depths())} depths, headline 100 m RMSE "
          f"{float(tab[tab.depth_m == 100].rmse.iloc[0]):.3f} degC, "
          f"blended FINAL {fin['rmse']:.3f} vs GLORYS {glo['rmse']:.3f}")
