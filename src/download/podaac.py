"""PO.DAAC products via OPeNDAP DAP4, subset server-side to our box.

    python src/download/podaac.py sss    [--start ... --end ... --workers 6]
    python src/download/podaac.py oscar

Needs EARTHDATA_TOKEN in .env (an Earthdata Login bearer token, never committed).
Whole granules are 4-20 MB global; the DAP4 constraint expression cuts that to ~100 KB,
which matters a lot on a slow link. One NetCDF per day, idempotent.

SMAP SSS uses V6, not the V4 named in docs/04: V4 was retired at 2022-07-11, which
would leave the validation year half empty. Each granule is stamped with its *centre* date
(docs/04 sec.5). OSCAR `u`,`v` are total current -- NOT `ug`,`vg` (geostrophic only) --
and are a 0-30 m mean, not a skin current.
"""
import argparse
import json
import os
import ssl
import sys
import tempfile
import urllib.request
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import certifi
import numpy as np
import pandas as pd
import xarray as xr

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import END, INTERIM, LAT_MAX, LAT_MIN, LON_MAX, LON_MIN, ROOT, START

CTX = ssl.create_default_context(cafile=certifi.where())
CMR = "https://cmr.earthdata.nasa.gov/search/granules.umm_json"
PAD = 0.5  # keep neighbours for the later bilinear regrid

PRODUCTS = {
    # `dims` is the data variables' own dimension order, because a DAP4 constraint
    # expression is positional. OSCAR really is (time, lon, lat) -- not a typo.
    "sss": {"short_name": "SMAP_RSS_L3_SSS_SMI_8DAY-RUNNINGMEAN_V6",
            "vars": ["sss_smap"], "dims": ("lat", "lon")},
    "oscar": {"short_name": "OSCAR_L4_OC_FINAL_V2.0",
              "vars": ["u", "v"], "dims": ("time", "lon", "lat")},
}


def token():
    for line in (ROOT / ".env").read_text().splitlines():
        if line.startswith("EARTHDATA_TOKEN="):
            return line.split("=", 1)[1].strip()
    raise SystemExit("EARTHDATA_TOKEN missing from .env")


def _get(url, tok, timeout=300):
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {tok}"})
    return urllib.request.urlopen(req, context=CTX, timeout=timeout).read()


def dap(base, ce, tok, tag):
    """Fetch a DAP4 constrained subset and return it as an xarray Dataset."""
    ce = ce.replace("[", "%5B").replace("]", "%5D")   # Tomcat rejects raw brackets
    raw = _get(f"{base}.dap.nc4?dap4.ce={ce}", tok)
    tmp = Path(tempfile.gettempdir()) / f"dap_{tag}_{os.getpid()}.nc"
    tmp.write_bytes(raw)
    with xr.open_dataset(tmp) as ds:
        out = ds.load()
    tmp.unlink(missing_ok=True)
    return out


def list_granules(short_name, start, end, page=2000):
    """date -> OPeNDAP base URL, from one CMR query per call."""
    u = (f"{CMR}?short_name={short_name}&page_size={page}"
         f"&temporal={start:%Y-%m-%d}T00:00:00Z,{end:%Y-%m-%d}T23:59:59Z&sort_key=start_date")
    items = json.load(urllib.request.urlopen(u, context=CTX, timeout=120))["items"]
    out = {}
    for it in items:
        t = it["umm"]["TemporalExtent"]
        rng = t.get("RangeDateTime")
        if rng:   # SMAP's 8-day composite is stamped with its CENTRE date (docs/04 sec.5)
            b, e = pd.Timestamp(rng["BeginningDateTime"]), pd.Timestamp(rng["EndingDateTime"])
            beg = b + (e - b) / 2
        else:
            beg = t["SingleDateTime"]
        # The OPeNDAP link is typed "USE SERVICE API" on some granules and "GET DATA" on
        # others (and the two use different URL shapes), so match on the host instead.
        opendap = next((r["URL"] for r in it["umm"]["RelatedUrls"]
                        if "opendap.earthdata.nasa.gov" in r["URL"]), None)
        if opendap:
            out[pd.Timestamp(beg).normalize().date()] = opendap
    return out


def grid_index(base, tok):
    """Index ranges covering our padded box, plus the coordinate values in that box.

    The coordinates are read once here and carried locally rather than requested per
    granule: Hyrax 500s on a sliced /lat when the variable's dimension is named
    `latitude` (OSCAR), and the grids are static anyway.
    """
    ds = dap(base, "/lat;/lon", tok, "grid")
    lat, lon = ds["lat"].values, ds["lon"].values
    assert lat[1] > lat[0], "expected ascending latitude"
    iy = np.where((lat >= LAT_MIN - PAD) & (lat <= LAT_MAX + PAD))[0]
    ix = np.where((lon >= LON_MIN - PAD) & (lon <= LON_MAX + PAD))[0]
    assert iy.size and ix.size, "box does not intersect the product grid"
    return ((int(iy[0]), int(iy[-1])), (int(ix[0]), int(ix[-1])),
            lat[iy], lon[ix])


def fetch_day(job):
    product, day, url, (y0, y1), (x0, x1), lat, lon, tok = job
    dst = INTERIM / product / f"{day:%Y}" / f"{product}_{day:%Y%m%d}.nc"
    if dst.exists():
        return "skip"
    dst.parent.mkdir(parents=True, exist_ok=True)
    rng = {"time": "[0:0]", "lat": f"[{y0}:{y1}]", "lon": f"[{x0}:{x1}]"}
    idx = "".join(rng[d] for d in PRODUCTS[product]["dims"])
    ce = ";".join(f"/{v}{idx}" for v in PRODUCTS[product]["vars"])
    try:
        ds = dap(url, ce, tok, product)
        ds = ds.rename({d: n for d, n in (("latitude", "lat"), ("longitude", "lon"))
                        if d in ds.dims})
        assert ds.sizes["lat"] == lat.size and ds.sizes["lon"] == lon.size,             f"dim order wrong for {product}: got {dict(ds.sizes)}"
        ds = ds.assign_coords(lat=("lat", lat), lon=("lon", lon))
        stamp = [pd.Timestamp(day)]
        ds = ds.assign_coords(time=stamp) if "time" in ds.dims else ds.expand_dims(time=stamp)
        tmp = dst.with_suffix(".tmp.nc")
        ds.to_netcdf(tmp)
        tmp.replace(dst)
    except Exception as e:
        dst.unlink(missing_ok=True)
        return f"FAIL {day}: {type(e).__name__} {str(e)[:120]}"
    return "ok"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("product", choices=sorted(PRODUCTS))
    p.add_argument("--start", default=START)
    p.add_argument("--end", default=END)
    p.add_argument("--workers", type=int, default=6)
    a = p.parse_args()
    tok = token()
    start, end = pd.Timestamp(a.start), pd.Timestamp(a.end)

    gran = {}
    for yr in range(start.year, end.year + 1):        # one CMR call per year keeps pages small
        lo, hi = max(start, pd.Timestamp(f"{yr}-01-01")), min(end, pd.Timestamp(f"{yr}-12-31"))
        gran |= list_granules(PRODUCTS[a.product]["short_name"], lo, hi)
    print(f"{len(gran)} granules found for {a.product}")
    if not gran:
        return

    yidx, xidx, lat, lon = grid_index(next(iter(gran.values())), tok)
    jobs = [(a.product, d, u, yidx, xidx, lat, lon, tok) for d, u in sorted(gran.items())]
    with ProcessPoolExecutor(a.workers) as ex:
        res = list(ex.map(fetch_day, jobs, chunksize=4))
    fails = [r for r in res if r.startswith("FAIL")]
    print(f"{res.count('ok')} downloaded, {res.count('skip')} present, {len(fails)} failed")
    for f in fails[:10]:
        print(" ", f)


if __name__ == "__main__":
    main()
