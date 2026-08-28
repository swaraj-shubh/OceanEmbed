"""Fold the per-day download cache into one NetCDF per product-year.

    python src/preprocess/consolidate.py [oisst sss oscar] [--force]

Opening a small NetCDF costs ~0.4 s on this box, so reading a channel straight from the
2832 per-day files takes ~19 minutes -- every time the store is built. Eight year files
per product turn that into seconds. The per-day files stay as the resumable download
cache; these are what the loaders read.
"""
import argparse
import sys
from pathlib import Path

import xarray as xr

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import INTERIM

OUT = INTERIM / "consolidated"
PRODUCTS = ["oisst", "sss", "oscar"]


def year_file(product, year):
    return OUT / f"{product}_{year}.nc"


def consolidate(product, force=False):
    src = INTERIM / product
    files = sorted(src.rglob(f"{product}_*.nc"))
    assert files, f"no per-day files under {src}"
    by_year = {}
    for f in files:
        by_year.setdefault(f.stem.split("_")[-1][:4], []).append(f)
    OUT.mkdir(parents=True, exist_ok=True)
    for year, fs in sorted(by_year.items()):
        dst = year_file(product, year)
        if dst.exists() and not force:
            print(f"{dst.name}: present ({len(fs)} days)")
            continue
        ds = xr.concat([xr.open_dataset(f) for f in fs], "time").sortby("time")
        tmp = dst.with_suffix(".tmp.nc")
        ds.to_netcdf(tmp)
        tmp.replace(dst)
        print(f"{dst.name}: {ds.sizes['time']} days, {dst.stat().st_size / 1e6:.0f} MB")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("products", nargs="*", default=PRODUCTS, choices=PRODUCTS + [])
    p.add_argument("--force", action="store_true")
    a = p.parse_args()
    for prod in (a.products or PRODUCTS):
        consolidate(prod, a.force)
