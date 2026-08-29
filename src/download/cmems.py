"""Copernicus Marine products via the `copernicusmarine` client, subset to our box.

    python src/download/cmems.py sla|wind|glorys [--start ...] [--end ...]
    python src/download/cmems.py glorys --probe          # one day, report bytes + seconds

Needs a free CMEMS account and a one-time `copernicusmarine login` (credentials land in
~/.copernicusmarine/, never in this repo). One NetCDF per dataset per year, idempotent.

Product choices, with the reasons, because the obvious picks are wrong:
  sla    DUACS L4 is 0.125 deg now (was 0.25); regridded down in the loader.
  wind   docs/04 names L3 scatterometer. The L4 fallback is unusable here: the 0.25 deg
         L4 record ENDS 2009-10, and the 0.125 deg L4 is hourly -- ~50 GB over this box
         and record for a field we only want a daily mean of. So: L3 ASCAT, 0.25 deg,
         ascending + descending, MetOp-A (-> 2021-11) spliced with MetOp-B (2019-01 ->).
         These are swaths, so days have gaps; the loader merges the four and the gap
         fraction is reported rather than hidden.
  glorys thetao only, 0-1000 m, native 50-level grid -- the loader interpolates to the 15
         SIH depths, which is better than letting the server snap to nearest levels.
"""
import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import END, INTERIM, LAT_MAX, LAT_MIN, LON_MAX, LON_MIN, START

PAD = 0.5  # neighbours for the later bilinear regrid
OUT = INTERIM / "cmems"

_ASCAT = [f"cmems_obs-wind_glo_phy_my_l3-metop{s}-ascat-{d}-0.25deg_P1D-i"
          for s in "ab" for d in ("asc", "des")]

PRODUCTS = {
    "sla": {"datasets": ["cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D"],
            "vars": ["sla"]},
    "wind": {"datasets": _ASCAT, "vars": ["eastward_wind", "northward_wind"]},
    "glorys": {"datasets": ["cmems_mod_glo_phy_my_0.083deg_P1D-m"],
               "vars": ["thetao"], "depth": (0.0, 1000.0)},
}


def fetch(product, dataset, start, end, out_dir=OUT):
    import copernicusmarine

    spec = PRODUCTS[product]
    dst = out_dir / product / f"{dataset}_{start:%Y%m%d}_{end:%Y%m%d}.nc"
    if dst.exists():
        return "skip", dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    lo, hi = spec.get("depth", (None, None))
    copernicusmarine.subset(
        dataset_id=dataset, variables=spec["vars"],
        minimum_longitude=LON_MIN - PAD, maximum_longitude=LON_MAX + PAD,
        minimum_latitude=LAT_MIN - PAD, maximum_latitude=LAT_MAX + PAD,
        start_datetime=str(start), end_datetime=str(end),
        minimum_depth=lo, maximum_depth=hi,
        output_directory=str(dst.parent), output_filename=dst.name,
        overwrite=True, disable_progress_bar=True)
    return "ok", dst


def main():
    p = argparse.ArgumentParser()
    p.add_argument("product", choices=sorted(PRODUCTS))
    p.add_argument("--start", default=START)
    p.add_argument("--end", default=END)
    p.add_argument("--probe", action="store_true",
                   help="fetch a single day into a scratch file and report size/time")
    a = p.parse_args()
    start, end = pd.Timestamp(a.start), pd.Timestamp(a.end)

    if a.probe:
        # Sizing the download before committing to it: GLORYS at 1/12 deg over 50 levels is
        # ~20 MB/day raw, but it is packed as int16 and compressed, so measure, don't guess.
        for ds in PRODUCTS[a.product]["datasets"][:1]:
            t0 = time.time()
            _, f = fetch(a.product, ds, start, start, OUT / "probe")
            mb, dt = f.stat().st_size / 1e6, time.time() - t0
            days = len(pd.date_range(a.start, a.end))
            print(f"{ds}: {mb:.2f} MB in {dt:.1f}s -> {mb * days / 1000:.1f} GB, "
                  f"{dt * days / 3600:.1f} h for {days} days")
        return

    for ds in PRODUCTS[a.product]["datasets"]:
        for yr in range(start.year, end.year + 1):
            lo = max(start, pd.Timestamp(f"{yr}-01-01"))
            hi = min(end, pd.Timestamp(f"{yr}-12-31"))
            try:
                what, f = fetch(a.product, ds, lo, hi)
                print(f"{what:4} {f.name} "
                      f"{f.stat().st_size / 1e6:.1f} MB" if what == "ok" else f"skip {f.name}")
            except Exception as e:
                # A satellite that simply was not flying that year is not an error.
                print(f"FAIL {ds} {yr}: {type(e).__name__} {str(e)[:140]}")


if __name__ == "__main__":
    main()
