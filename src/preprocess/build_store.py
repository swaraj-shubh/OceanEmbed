"""Assemble the per-source interim files into the model-ready Zarr store.

    python src/preprocess/build_store.py [--start ...] [--end ...]

One loader per channel, each returning a daily (time, lat, lon) DataArray already on the
frozen 0.25 deg grid, NaN where missing. NaN is the mask -- see docs/04 sec.4. Only the
loaders below are implemented; the rest need CMEMS / Earthdata credentials.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import CHANNELS, DEPTHS, END, INTERIM, LAT, LON, QC_RANGE, START, ZARR
from regrid import qc, to_grid


def load_sst(days):
    """NOAA OISST v2.1 -- already on the target grid, so no interpolation is applied."""
    root = INTERIM / "oisst"
    files = [root / f"{d:%Y}" / f"oisst_{d:%Y%m%d}.nc" for d in days]
    missing = [f.name for f in files if not f.exists()]
    assert not missing, f"{len(missing)} OISST days missing, first {missing[0]}"
    da = xr.concat([xr.open_dataset(f).sst.squeeze(drop=True) for f in files], "time")
    da = da.assign_coords(time=days)
    assert np.allclose(da.lat, LAT) and np.allclose(da.lon, LON), "OISST off the frozen grid"
    return qc(da.astype("float32"), "sst")


def _daily(product, var, days, ffill=0):
    """Read one PO.DAAC-style per-day directory, regrid, and align onto `days`.

    Days with no granule stay NaN (the missing mask carries them). `ffill` allows a
    short carry-forward for SMAP, whose 8-day composite genuinely covers the gap --
    but only a few days, so the 2019 and 2022 outages are never papered over.
    """
    root = INTERIM / product
    want = {f"{d:%Y%m%d}" for d in days}
    files = [f for f in sorted(root.rglob(f"{product}_*.nc")) if f.stem.split("_")[-1] in want]
    assert files, f"no {product} files under {root} -- run src/download/podaac.py {product}"
    da = xr.concat([xr.open_dataset(f)[var] for f in files], "time").sortby("time")
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


def _todo(product):
    def loader(days):
        raise NotImplementedError(f"{product}: download it first (needs an account)")
    return loader


LOADERS = {
    "sst": load_sst,
    "sss": load_sss,
    "sla": _todo("CMEMS SEALEVEL_GLO_PHY_L4_MY_008_047"),
    "cur_u": _oscar("u", "cur_u"),        # total current, not geostrophic; 0-30 m mean
    "cur_v": _oscar("v", "cur_v"),
    "wind_u": _todo("CMEMS WIND_GLO_PHY_L3_MY_012_005 eastward"),
    "wind_v": _todo("CMEMS WIND_GLO_PHY_L3_MY_012_005 northward"),
}


def load_target(days):
    """GLORYS12V1 thetao, regridded to 0.25 deg and interpolated onto the 15 SIH depths."""
    src = INTERIM / "glorys"
    if not src.exists():
        raise NotImplementedError("GLORYS12V1: run the CMEMS download first")
    da = xr.open_mfdataset(sorted(src.glob("*.nc"))).thetao
    da = to_grid(da, "latitude", "longitude").interp(depth=DEPTHS).sel(time=days)
    return qc(da.astype("float32"), "thetao")


def build(start=START, end=END, out=ZARR):
    days = pd.date_range(start, end, freq="D")
    X = xr.concat([LOADERS[c](days) for c in CHANNELS], pd.Index(CHANNELS, name="channel"))
    Y = load_target(days)
    ds = xr.Dataset({"X": X.transpose("time", "channel", ...).rename({"lat": "y", "lon": "x"}),
                     "Y": Y.transpose("time", "depth", ...).rename({"lat": "y", "lon": "x"})})
    ds = ds.assign_coords(lat=("y", LAT), lon=("x", LON))
    ds.attrs["sss_note"] = "SMAP is an 8-day running mean assigned to its centre date"
    ds.attrs["regrid"] = "bilinear xarray.interp; OISST used as-is (native 0.25 deg)"
    out.parent.mkdir(parents=True, exist_ok=True)
    ds.to_zarr(out, mode="w")
    return ds


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--start", default=START)
    p.add_argument("--end", default=END)
    p.add_argument("--check", metavar="CHANNEL", help="verify one channel instead of building")
    a = p.parse_args()
    if a.check:
        days = pd.date_range(a.start, a.end, freq="D")
        da = LOADERS[a.check](days).load()
        assert da.shape == (len(days), len(LAT), len(LON)), da.shape
        lo, hi = QC_RANGE[a.check]
        assert bool(((da >= lo) & (da <= hi)).any()), f"no valid {a.check}"
        always_nan = np.isnan(da).all("time")
        print(f"{a.check} ok: {da.shape}, "
              f"{float(always_nan.mean()) * 100:.1f}% never-observed cells, "
              f"mean {float(da.mean()):.3f}, range {float(da.min()):.2f}..{float(da.max()):.2f}, "
              f"{float(np.isnan(da).mean()) * 100:.1f}% NaN overall")
    else:
        build(a.start, a.end)
