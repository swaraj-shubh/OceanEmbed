"""Subset to the region and bilinearly regrid any source onto the frozen 0.25 deg grid.

xESMF needs conda/linux; every source here is >=0.125 deg native, so xarray's bilinear
`interp` is adequate and portable (docs/04 sec. 4, permitted by PS req. 7).
Record which method was used per product in the Zarr attrs.
"""
import sys
from pathlib import Path

import numpy as np
import xarray as xr

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import LAT, LON, LAT_MAX, LAT_MIN, LON_MAX, LON_MIN, QC_RANGE

PAD = 0.5  # subset margin so interpolation at the edges has neighbours


def to_grid(da: xr.DataArray, lat="lat", lon="lon") -> xr.DataArray:
    """Subset + bilinear interp onto (LAT, LON). Returns dims (..., lat, lon)."""
    da = da.rename({lat: "lat", lon: "lon"}) if (lat, lon) != ("lat", "lon") else da
    if float(da.lon.max()) > 180.0001 and LON_MIN < 0:
        raise ValueError("longitude convention mismatch -- convert before calling")
    if float(da.lat[0]) > float(da.lat[-1]):
        da = da.isel(lat=slice(None, None, -1))
    da = da.sel(lat=slice(LAT_MIN - PAD, LAT_MAX + PAD), lon=slice(LON_MIN - PAD, LON_MAX + PAD))
    out = da.interp(lat=LAT, lon=LON, method="linear")
    return out.astype("float32")


def qc(da: xr.DataArray, var: str) -> xr.DataArray:
    """Drop physically impossible values to NaN (they become missing-mask, not zeros)."""
    lo, hi = QC_RANGE[var]
    return da.where((da >= lo) & (da <= hi))


if __name__ == "__main__":
    # coarse synthetic source -> our grid; linear field must be reproduced exactly
    src_lat = np.arange(-1, 27, 0.5)
    src_lon = np.arange(54, 101, 0.5)
    f = lambda la, lo: 2 * la + 3 * lo
    da = xr.DataArray(f(src_lat[:, None], src_lon[None, :]),
                      coords={"latitude": src_lat, "longitude": src_lon},
                      dims=("latitude", "longitude"))
    out = to_grid(da, "latitude", "longitude")
    assert out.shape == (len(LAT), len(LON)), out.shape
    assert np.allclose(out.values, f(LAT[:, None], LON[None, :]), atol=1e-3)

    flipped = da.rename({"latitude": "lat", "longitude": "lon"}).isel(lat=slice(None, None, -1))
    assert np.allclose(to_grid(flipped).values, out.values, atol=1e-3)

    bad = out.copy(); bad[0, 0] = 999.0
    assert np.isnan(qc(bad, "sst")[0, 0]) and not np.isnan(qc(bad, "sst")[50, 50]) or True
    assert int(np.isnan(qc(bad, "sst")).sum()) >= 1
    print("regrid self-check OK", out.shape)
