"""Validation track B1 -- score a prediction cube against INCOIS LAS Gridded ARGO.

    python src/incois_eval.py --cube results/ens_mix6_test_cube.nc \
                              --incois data/interim/argo_10d.nc --split test

The PS names this product, and it also permits the regridding this needs (requirement 7).
Our output is 0.25 deg daily; the product is 1 deg / 10-day objective analysis. So we
**aggregate our prediction UP to the coarser reference** -- never interpolate the
reference down. Downscaling an objectively-analysed field invents structure it does not
have and flatters the score, which is the opposite of a validation.

B1 is the compliance track. B2 (raw profiles, argo_eval.py) stays the stricter test: at
1 deg this product cannot resolve what a 0.25 deg model produces, and it has already been
smoothed. Where the two disagree, that gap measures what objective analysis removes.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.append(str(Path(__file__).resolve().parent))
from config import DEPTHS, REPORT_DEPTHS, ROOT, SPLITS
from metrics import DepthStats

RESULTS = ROOT / "results"


def _edges(centres):
    """Cell edges from centres, assuming a regular grid. Asserts regularity rather than
    trusting it -- an irregular reference axis would silently mis-bin every cell."""
    c = np.asarray(centres, float)
    d = np.diff(c)
    assert c.size > 1 and np.allclose(d, d[0], rtol=1e-6), f"reference axis not regular: {d[:5]}"
    step = float(d[0])
    return np.concatenate([[c[0] - step / 2], c + step / 2]), step


def aggregate_to(cube, ref_lat, ref_lon, ref_time, window_days=10):
    """Average our fine cube onto the reference's (lat, lon, time) cells.

    Spatial: a plain binned mean over the reference cell footprint. Temporal: the mean of
    the `window_days` ending at each reference stamp -- 10-day products are labelled by the
    end of their window, so a centred window would pull in days the reference never saw.
    Returns [time, depth, lat, lon] aligned to the reference, NaN where we have no data.
    """
    lat_e, _ = _edges(ref_lat)
    lon_e, _ = _edges(ref_lon)
    # Our cells fall wholly inside one reference cell (0.25 divides 1.0), so a groupby_bins
    # mean is exact -- no area weighting needed at these latitudes for a 1 deg box.
    sp = (cube.groupby_bins("lat", lat_e, labels=np.asarray(ref_lat))
              .mean("lat", skipna=True)
              .groupby_bins("lon", lon_e, labels=np.asarray(ref_lon))
              .mean("lon", skipna=True)
              .rename({"lat_bins": "lat", "lon_bins": "lon"}))

    out = []
    for t in pd.DatetimeIndex(ref_time):
        lo = t - pd.Timedelta(days=window_days - 1)
        sel = sp.sel(time=slice(lo, t))
        # A reference window we cannot cover at all must stay NaN, not silently average
        # whatever few days happen to overlap the split edge.
        out.append(sel.mean("time", skipna=True) if sel.sizes.get("time", 0) else
                   sp.isel(time=0) * np.nan)
    agg = xr.concat(out, dim=pd.Index(pd.DatetimeIndex(ref_time), name="time"))
    return agg.transpose("time", "depth", "lat", "lon")


def align_depths(ref, our_depths=DEPTHS):
    """Reference levels -> the 15 SIH depths, on the INTERSECTION only.

    Deliberately no interpolation of the reference onto levels it does not carry: the PS
    permits regridding, but inventing a 125 m level from 100 and 150 m would be scoring
    ourselves against a number nobody measured. Report which levels were used.
    """
    rd = np.asarray(ref.depth.values, float)
    keep = [d for d in our_depths if np.min(np.abs(rd - d)) < 1e-6]
    assert keep, f"no shared depths: reference has {rd[:8]}..., we need {our_depths}"
    return keep


def evaluate_incois(cube, ref, window_days=10):
    """cube: our [time, depth, lat, lon] in degC. ref: the INCOIS field, same dim names.
    Returns (depth-wise table, n reference cells scored, depths used)."""
    depths = align_depths(ref)
    ref = ref.sel(depth=depths)
    cube = cube.sel(depth=depths)
    agg = aggregate_to(cube, ref.lat.values, ref.lon.values, ref.time.values, window_days)
    assert agg.shape == ref.shape, f"{agg.shape} vs {ref.shape} after aggregation"

    acc = DepthStats(depths)
    p = agg.values.transpose(1, 0, 2, 3)          # [depth, time, lat, lon]
    o = ref.values.transpose(1, 0, 2, 3)
    acc.update(p, o)
    return acc.table(), int(np.isfinite(p * o).sum()), depths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cube", required=True, help="our prediction cube (NetCDF)")
    ap.add_argument("--incois", required=True, help="INCOIS gridded Argo NetCDF")
    ap.add_argument("--split", default="test")
    ap.add_argument("--var", default=None, help="temperature variable name in the INCOIS file")
    ap.add_argument("--window-days", type=int, default=10)
    a = ap.parse_args()

    cube = xr.open_dataarray(a.cube)
    ds = xr.open_dataset(a.incois)
    if a.var is None:
        cand = [v for v in ds.data_vars if ds[v].ndim == 4]
        assert len(cand) == 1, f"pass --var; 4-D candidates are {cand}"
        a.var = cand[0]
    ref = ds[a.var]
    ren = {n: t for n, t in (("latitude", "lat"), ("longitude", "lon"),
                             ("LATITUDE", "lat"), ("LONGITUDE", "lon"),
                             ("lev", "depth"), ("level", "depth"), ("z", "depth"), ("ZAX", "depth"),
                             ("TIME", "time"), ("DEPTH", "depth")) if n in ref.dims}
    ref = ref.rename(ren) if ren else ref

    lo, hi = SPLITS[a.split]
    ref = ref.sel(time=slice(lo, hi))
    assert ref.sizes["time"], f"no INCOIS windows inside {a.split} ({lo}..{hi})"

    df, n, depths = evaluate_incois(cube, ref, a.window_days)
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"{Path(a.cube).stem.replace('_cube', '')}_incois_b1.csv"
    df.to_csv(out, index=False)
    print(f"track B1: {Path(a.cube).name} vs INCOIS gridded Argo, {a.split}")
    print(f"  {ref.sizes['time']} reference windows, {n:,} cell-levels, "
          f"depths used: {depths}")
    print(df[df.depth_m.isin(REPORT_DEPTHS)].to_string(
        index=False, float_format=lambda v: f"{v:8.3f}"))
    w = df["n"].to_numpy(float)
    print(f"  blended RMSE {np.sqrt(np.nansum(w * df.rmse ** 2) / w.sum()):.3f} degC -> {out.name}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main()
        raise SystemExit

    # --- self-check on synthetic data; no network, no INCOIS file needed ---
    rng = np.random.default_rng(0)
    our_lat = np.arange(0.625, 24.5, 0.25)
    our_lon = np.arange(55.625, 99.5, 0.25)
    t = pd.date_range("2023-01-01", periods=40)
    ref_lat = np.arange(1.5, 24.0, 1.0)
    ref_lon = np.arange(56.5, 99.0, 1.0)
    ref_t = pd.to_datetime(["2023-01-10", "2023-01-20", "2023-01-30"])

    def cube_of(vals):
        return xr.DataArray(vals, dims=("time", "depth", "lat", "lon"),
                            coords={"time": t, "depth": DEPTHS, "lat": our_lat, "lon": our_lon})

    # 1. a constant field must aggregate to exactly that constant
    const = cube_of(np.full((len(t), len(DEPTHS), len(our_lat), len(our_lon)), 7.0, "f4"))
    agg = aggregate_to(const, ref_lat, ref_lon, ref_t)
    assert agg.shape == (3, 15, len(ref_lat), len(ref_lon)), agg.shape
    assert np.allclose(agg.values, 7.0), "constant field did not survive aggregation"

    # 2. aggregation must be a MEAN over the footprint, not a sample of one cell
    ramp = np.broadcast_to(our_lon.astype("f4"), (len(t), len(DEPTHS), len(our_lat), len(our_lon)))
    agg = aggregate_to(cube_of(ramp.copy()), ref_lat, ref_lon, ref_t)
    # each 1 deg cell holds our 4 cells at -0.375,-0.125,+0.125,+0.375 -> mean = the centre
    assert np.allclose(agg.isel(time=0, depth=0).values[0], ref_lon, atol=1e-4), \
        "spatial aggregation is not the footprint mean"

    # 3. the time window must END at the reference stamp, never look ahead
    ramp_t = np.broadcast_to(np.arange(len(t), dtype="f4")[:, None, None, None],
                             (len(t), len(DEPTHS), len(our_lat), len(our_lon)))
    agg = aggregate_to(cube_of(ramp_t.copy()), ref_lat, ref_lon, ref_t, window_days=10)
    # window ending 2023-01-10 covers days 0..9 -> mean 4.5
    assert np.isclose(float(agg.isel(time=0, depth=0)[0, 0]), 4.5), \
        f"window is not the 10 days ending at the stamp: {float(agg.isel(time=0,depth=0)[0,0])}"
    assert np.isclose(float(agg.isel(time=1, depth=0)[0, 0]), 14.5), "second window wrong"

    # 4. a known constant offset must be recovered exactly as bias
    truth = np.full((3, 15, len(ref_lat), len(ref_lon)), 12.0, "f4")
    ref = xr.DataArray(truth, dims=("time", "depth", "lat", "lon"),
                       coords={"time": ref_t, "depth": DEPTHS, "lat": ref_lat, "lon": ref_lon})
    off = cube_of(np.full((len(t), len(DEPTHS), len(our_lat), len(our_lon)), 12.5, "f4"))
    df, n, used = evaluate_incois(off, ref)
    assert np.allclose(df.bias, 0.5) and np.allclose(df.rmse, 0.5), df[["bias", "rmse"]]
    assert used == DEPTHS and n > 0

    # 5. land/NaN in our cube must not poison a reference cell that is partly ocean
    holed = off.copy()
    holed[:, :, :2, :2] = np.nan
    df2, _, _ = evaluate_incois(holed, ref)
    assert np.allclose(df2.bias, 0.5), "NaN in our cube leaked into the aggregate"

    # 6. depth intersection: a reference carrying only 5 of our levels must use only those
    sub = ref.sel(depth=[0, 50, 100, 500, 1000])
    df3, _, used3 = evaluate_incois(off, sub)
    assert used3 == [0, 50, 100, 500, 1000] and len(df3) == 5, used3

    # 7. an irregular reference axis must be refused, not silently mis-binned
    try:
        _edges([0.0, 1.0, 3.0])
        raise SystemExit("FAIL: irregular axis accepted")
    except AssertionError as e:
        assert "not regular" in str(e)

    print("incois_eval self-check OK -- aggregation is a footprint mean over the "
          "10 days ending at each stamp; bias recovered to 1e-6")
