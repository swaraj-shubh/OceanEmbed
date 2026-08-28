"""NOAA OISST v2.1 daily SST -> region subset, one tiny NetCDF per day.

No account needed. Idempotent: existing days are skipped.
    python src/download/oisst.py [--start 2015-04-01] [--end 2022-12-31] [--workers 8]
"""
import argparse
import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import xarray as xr

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import INTERIM, LAT_MAX, LAT_MIN, LON_MAX, LON_MIN, START, END, GRID_SHAPE

BASE = ("https://www.ncei.noaa.gov/data/sea-surface-temperature-optimum-interpolation"
        "/v2.1/access/avhrr/{ym}/oisst-avhrr-v02r01.{ymd}.nc")
OUT = INTERIM / "oisst"


def fetch_day(day: pd.Timestamp) -> str:
    dst = OUT / f"{day:%Y}" / f"oisst_{day:%Y%m%d}.nc"
    if dst.exists():
        return "skip"
    dst.parent.mkdir(parents=True, exist_ok=True)
    url = BASE.format(ym=f"{day:%Y%m}", ymd=f"{day:%Y%m%d}")
    tmp = dst.with_suffix(".tmp.nc")
    try:
        urllib.request.urlretrieve(url, tmp)
        with xr.open_dataset(tmp) as ds:
            sub = (ds[["sst"]]
                   .sel(lat=slice(LAT_MIN, LAT_MAX), lon=slice(LON_MIN, LON_MAX))
                   .squeeze("zlev", drop=True))
            assert sub.sizes["lat"] == GRID_SHAPE[0] and sub.sizes["lon"] == GRID_SHAPE[1], \
                f"{day:%Y-%m-%d}: got {sub.sizes['lat']}x{sub.sizes['lon']}, want {GRID_SHAPE}"
            sub.to_netcdf(dst)
    except (urllib.error.HTTPError, urllib.error.URLError, OSError) as e:
        dst.unlink(missing_ok=True)
        return f"FAIL {day:%Y-%m-%d}: {e}"
    finally:
        tmp.unlink(missing_ok=True)
    return "ok"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--start", default=START)
    p.add_argument("--end", default=END)
    p.add_argument("--workers", type=int, default=8)
    a = p.parse_args()
    days = pd.date_range(a.start, a.end, freq="D")
    with ThreadPoolExecutor(a.workers) as ex:
        results = list(ex.map(fetch_day, days))
    fails = [r for r in results if r.startswith("FAIL")]
    print(f"{results.count('ok')} downloaded, {results.count('skip')} already present, "
          f"{len(fails)} failed -> {OUT}")
    for f in fails[:20]:
        print(" ", f)


if __name__ == "__main__":
    main()
