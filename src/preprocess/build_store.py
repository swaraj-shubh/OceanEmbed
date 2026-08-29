"""Assemble the per-source interim files into the model-ready Zarr store.

    python src/preprocess/build_store.py [--start ...] [--end ...]

One loader per channel, each returning a daily (time, lat, lon) DataArray already on the
frozen 0.25 deg grid, NaN where missing. NaN is the mask -- see docs/04 sec.4.
Download the sources first (src/download/*.py); this step never touches provider files.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import (CHANNELS, DEPTHS, END, INTERIM, LAT, LON, QC_RANGE,
                    REPORT_DEPTHS, START, ZARR)
from consolidate import year_file
from regrid import qc, to_grid


def _read(product, var, days):
    """Read one variable from the consolidated per-year files covering `days`."""
    files = [year_file(product, y) for y in sorted({d.year for d in days})]
    missing = [f.name for f in files if not f.exists()]
    assert not missing, (f"{missing} absent -- run "
                         f"python src/preprocess/consolidate.py {product}")
    da = xr.concat([xr.open_dataset(f)[var] for f in files], "time").sortby("time")
    # OISST stamps its daily field at 12:00Z; floor everything so the daily index lines up
    return da.assign_coords(time=da.time.dt.floor("D"))


def load_sst(days):
    """NOAA OISST v2.1 -- already on the target grid, so no interpolation is applied."""
    da = _read("oisst", "sst", days).sel(time=days)
    assert np.allclose(da.lat, LAT) and np.allclose(da.lon, LON), "OISST off the frozen grid"
    return qc(da.astype("float32"), "sst")


def _daily(product, var, days, ffill=0):
    """Regrid a consolidated product onto the frozen grid and align it onto `days`.

    Days with no granule stay NaN (the missing mask carries them). `ffill` allows a
    short carry-forward for SMAP, whose 8-day composite genuinely covers the gap --
    but only a few days, so the 2019 and 2022 outages are never papered over.
    """
    da = _read(product, var, days)
    # OSCAR's variables are (time, lon, lat); pin the axis order rather than inherit it
    da = to_grid(da).transpose("time", "lat", "lon")
    tol = {"method": "ffill", "tolerance": pd.Timedelta(days=ffill)} if ffill else {}
    return da.reindex(time=days, **tol)


def load_sss(days):
    """SMAP RSS L3 SSS v4 -- an 8-day running mean stamped at its centre date."""
    return qc(_daily("sss", "sss_smap", days, ffill=4).astype("float32"), "sss")


def _oscar(var, channel):
    def loader(days):
        return qc(_daily("oscar", var, days).astype("float32"), channel)
    return loader


CMEMS = INTERIM / "cmems"


def _lazy(f, var, tchunk=16):
    """Open one CMEMS file dask-chunked and float32.

    Both halves matter. xarray's `interp` upcasts to float64, so six years of 0.125 deg
    SLA asked for a 1.6 GiB allocation and died; GLORYS at 1/12 deg over 36 levels would
    be ~80 GB. Chunked + float32 keeps the regrid lazy and the peak bounded.

    `tchunk` is small for GLORYS because packed variables are decoded to float64 by
    scale/offset *before* the cast here can apply: 16 days x 36 levels x 313 x 553 is a
    713 MiB allocation, which an 8 GB box refuses while a download is also running.
    """
    return xr.open_dataset(f, chunks={"time": tchunk})[var].astype("float32")


def _cmems(product, var, days):
    """Read a CMEMS product from data/interim/cmems/<product>/ onto the frozen grid.

    Several datasets under one product (the four ASCAT satellite/pass combinations) are
    merged with a nanmean: they are swaths, so on any given day each covers part of the
    box. Where none of them saw a cell it stays NaN -- swath gaps are carried by the
    missing mask, never filled in, because pretending a scatterometer saw the whole basin
    would be inventing wind.
    """
    files = sorted((CMEMS / product).glob("*.nc"))
    assert files, f"{product}: run python src/download/cmems.py {product}"
    by_ds = {}
    for f in files:                       # <dataset_id>_<start>_<end>.nc
        by_ds.setdefault(f.stem.rsplit("_", 2)[0], []).append(f)
    das = []
    for _, fs in sorted(by_ds.items()):
        da = xr.concat([_lazy(f, var) for f in sorted(fs)], "time").sortby("time")
        da = da.assign_coords(time=da.time.dt.floor("D"))
        da = da.drop_duplicates("time")   # asc/des files can overlap a day at a year edge
        das.append(to_grid(da, "latitude", "longitude").reindex(time=days))
    out = das[0] if len(das) == 1 else xr.concat(das, "src").mean("src", skipna=True)
    return out.transpose("time", "lat", "lon").astype("float32")


def load_sla(days):
    """DUACS L4 sea level anomaly, 0.125 deg -> 0.25 deg."""
    return qc(_cmems("sla", "sla", days), "sla")


def _wind(var, channel):
    """ASCAT winds, gap-filled with a centred 3-day mean.

    Scatterometers are swaths: raw daily coverage of the cells ASCAT can see is ~55% in
    the years one MetOp was flying (2015-18, 2022-24) and ~86% when both were (2019-21).
    A centred 3-day mean lifts every year to ~97%; +/-2 days adds only 0.3 points, so one
    day either side is where it stops paying.

    That window looks one day ahead, which is deliberate and worth stating: this is a
    reconstruction task, not a forecast, and the SSS channel is already an 8-day running
    mean centred on its date -- so this is strictly less lookahead than a channel the PS
    itself specifies. Cells still empty after the fill (~3%) stay NaN.
    """
    def loader(days):
        da = _cmems("wind", var, days)
        return qc(da.rolling(time=3, center=True, min_periods=1).mean(), channel)
    return loader


LOADERS = {
    "sst": load_sst,
    "sss": load_sss,
    "sla": load_sla,
    "cur_u": _oscar("u", "cur_u"),        # total current, not geostrophic; 0-30 m mean
    "cur_v": _oscar("v", "cur_v"),
    "wind_u": _wind("eastward_wind", "wind_u"),
    "wind_v": _wind("northward_wind", "wind_v"),
}


def load_target(days):
    """GLORYS12V1 thetao, regridded to 0.25 deg and interpolated onto the 15 SIH depths."""
    da = _read_glorys(days)
    return qc(da.astype("float32"), "thetao")


def _read_glorys(days):
    files = sorted((CMEMS / "glorys").glob("*.nc"))
    assert files, "glorys: run python src/download/cmems.py glorys"
    da = xr.concat([_lazy(f, "thetao", tchunk=2) for f in files], "time").sortby("time")
    da = da.assign_coords(time=da.time.dt.floor("D")).drop_duplicates("time")
    assert float(da.depth.max()) >= max(DEPTHS), (
        f"deepest GLORYS level is {float(da.depth.max()):.1f} m < {max(DEPTHS)} m -- "
        "the download's maximum_depth is too shallow and 1000 m would be extrapolated")
    # Clamp the requested depths into the source range instead of extrapolating. GLORYS's
    # shallowest level is 0.494 m, so interpolating to exactly 0 m fell outside it and the
    # whole surface level came back NaN -- the same failure as the 1000 m ceiling, at the
    # other end, and 0 m is both a headline metric depth and the map the demo shows first.
    # Clamping means "0 m" is GLORYS's 0.494 m level, which is a sub-0.01 degC difference,
    # and nothing anywhere is extrapolated.
    src = np.clip(DEPTHS, float(da.depth.min()), float(da.depth.max()))
    da = to_grid(da, "latitude", "longitude").interp(depth=src).assign_coords(depth=DEPTHS)
    return da.reindex(time=days).transpose("time", "depth", "lat", "lon")


def build(start=START, end=END, out=ZARR):
    days = pd.date_range(start, end, freq="D")
    X = xr.concat([LOADERS[c](days) for c in CHANNELS], pd.Index(CHANNELS, name="channel"))
    Y = load_target(days)
    # Spatial dims stay named lat/lon. They were briefly y/x, which silently DESTROYED the
    # target on Windows: NTFS is case-insensitive, so the data variable "Y" and the
    # dimension coordinate "y" are the same directory in a Zarr store, and the store came
    # back holding X and y with no Y at all. It would have worked on Linux, so it would
    # only ever have failed here.
    ds = xr.Dataset({"X": X.transpose("time", "channel", ...),
                     "Y": Y.transpose("time", "depth", ...)})
    ds.attrs["sss_note"] = "SMAP is an 8-day running mean assigned to its centre date"
    ds.attrs["regrid"] = "bilinear xarray.interp; OISST used as-is (native 0.25 deg)"
    ds.attrs["wind_note"] = ("ASCAT L3 swaths (MetOp-A 2015-2021 + MetOp-B 2019-2024, "
                             "ascending+descending) merged by nanmean, then a centred "
                             "3-day mean: ~55-86% raw daily coverage -> ~97%")
    # Concatenating sources that were chunked differently leaves ragged time chunks, which
    # Zarr rejects. time=1 because the DataLoader reads random single days -- a bigger time
    # chunk would pull a fortnight off disk to serve one sample.
    # Both depth-end bugs (0 m outside the source range, 1000 m past a too-shallow
    # download) produced an entirely NaN level that nothing else would have complained
    # about until the metrics table came back empty. One time slice is cheap to check.
    probe = ds.Y.isel(time=0).sel(depth=REPORT_DEPTHS).load()
    dead = [int(z) for z in REPORT_DEPTHS if not bool(np.isfinite(probe.sel(depth=z)).any())]
    assert not dead, f"report depths with no data: {dead} m"
    ds = ds.chunk({"time": 1, "channel": -1, "depth": -1, "lat": -1, "lon": -1})
    out.parent.mkdir(parents=True, exist_ok=True)
    # zarr_format=2: v3's nested chunk directories race between dask writer threads
    # (FileExistsError on X/c/0), and v2's flat keys are the portable choice for Kaggle
    # and older zarr installs anyway.
    ds.to_zarr(out, mode="w", zarr_format=2)
    return ds


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--start", default=START)
    p.add_argument("--end", default=END)
    p.add_argument("--check", metavar="CHANNEL", help="verify one channel instead of building")
    a = p.parse_args()
    if a.check:
        days = pd.date_range(a.start, a.end, freq="D")
        da = (load_target if a.check == "target" else LOADERS[a.check])(days).load()
        want = (len(days), len(LAT), len(LON))
        assert da.shape == want if a.check != "target" else da.shape[0] == len(days), da.shape
        lo, hi = QC_RANGE["thetao" if a.check == "target" else a.check]
        assert bool(((da >= lo) & (da <= hi)).any()), f"no valid {a.check}"
        always_nan = np.isnan(da).all("time")
        print(f"{a.check} ok: {da.shape}, "
              f"{float(always_nan.mean()) * 100:.1f}% never-observed cells, "
              f"mean {float(da.mean()):.3f}, range {float(da.min()):.2f}..{float(da.max()):.2f}, "
              f"{float(np.isnan(da).mean()) * 100:.1f}% NaN overall")
    else:
        build(a.start, a.end)
