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
  glorys thetao only, native level grid, interpolated to the 15 SIH depths by the loader
         rather than snapped to nearest levels by the server. The depth ceiling is 1100 m,
         not 1000: GLORYS levels near the bottom are 902.3 and 1062.4 m, so asking for
         maximum_depth=1000 returns 902.3 as the deepest level and the 1000 m target --
         one of the six depths in the headline metrics table -- would be extrapolated.
"""
import argparse
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor
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
    # Weekly chunks. Not for resumability -- for memory. A month is 30 x 36 x 313 x 553,
    # which copernicusmarine decodes to ~1.5 GB in float64 before copies, and this box has
    # 1.4 GB free of 8 GB: the downloader was dying with "OpenBLAS error: Memory allocation
    # still failed after 10 retries", which reads like a harness kill but is a plain OOM.
    # A week still OOMs; 3 days (~37 MB per file) survives. The others are fine per year.
    "glorys": {"datasets": ["cmems_mod_glo_phy_my_0.083deg_P1D-m"],
               "vars": ["thetao"], "depth": (0.0, 1100.0), "chunk": "3D"},
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


def _one(job):
    product, ds, lo, hi = job
    try:
        what, f = fetch(product, ds, lo, hi)
        return (f"ok   {f.name} {f.stat().st_size / 1e6:.1f} MB" if what == "ok"
                else f"skip {f.name}")
    except Exception as e:
        # A satellite that simply was not flying in that window is not an error.
        return f"FAIL {ds.split('_l3-')[-1][:24]} {lo:%Y-%m-%d}: {type(e).__name__} {str(e)[:110]}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("product", choices=sorted(PRODUCTS))
    p.add_argument("--start", default=START)
    p.add_argument("--end", default=END)
    p.add_argument("--workers", type=int, default=1,
                   help="parallel chunk downloads; each holds a decoded chunk in RAM")
    p.add_argument("--probe", action="store_true",
                   help="fetch a single day into a scratch file and report size/time")
    a = p.parse_args()
    start, end = pd.Timestamp(a.start), pd.Timestamp(a.end)

    if a.probe:
        # Sizing the download before committing to it: GLORYS at 1/12 deg over 50 levels is
        # ~20 MB/day raw, but it is packed as int16 and compressed, so measure, don't guess.
        # Probe the range actually asked for -- a single day is dominated by ~30 s of
        # catalogue setup and badly overestimates the total. Use a month.
        got = len(pd.date_range(start, end))
        full = len(pd.date_range(START, END))
        for ds in PRODUCTS[a.product]["datasets"][:1]:
            t0 = time.time()
            _, f = fetch(a.product, ds, start, end, OUT / "probe")
            mb, dt = f.stat().st_size / 1e6, time.time() - t0
            print(f"{ds}: {mb:.1f} MB / {got} days in {dt:.1f}s -> "
                  f"{mb * full / got / 1000:.1f} GB, {dt * full / got / 3600:.1f} h "
                  f"for the full {full}-day record")
        return

    freq = PRODUCTS[a.product].get("chunk", "YS")
    edges = pd.date_range(start, end, freq=freq)
    edges = pd.DatetimeIndex([start]).union(edges)
    # Skip by DATE COVERAGE, not by filename. The chunk size changed mid-download (monthly
    # -> 3-day after the OOM), so the new filenames never matched the old ones and 42 chunks
    # were spent re-fetching days already on disk. Coverage is also not contiguous, so a
    # manual --start cannot express it.
    have = set()
    for f in (OUT / a.product).glob("*.nc"):
        m = re.search(r"(\d{8})_(\d{8})", f.name)
        if m:
            have |= set(pd.date_range(*m.groups()))

    jobs = []
    for ds in PRODUCTS[a.product]["datasets"]:
        for i, lo in enumerate(edges):
            hi = min(end, edges[i + 1] - pd.Timedelta(days=1) if i + 1 < len(edges) else end)
            if not set(pd.date_range(lo, hi)) <= have:
                jobs.append((a.product, ds, lo, hi))
    print(f"{len(have)} days already on disk, {len(jobs)} chunks to fetch")
    if a.workers > 1:
        # The transfer measured 0.62 MB/s on one stream, which is the link and not the
        # server, so parallel chunks multiply throughput. Keep `workers` low: each holds a
        # decoded chunk in RAM and this box OOMed at one monthly chunk.
        with ProcessPoolExecutor(a.workers) as ex:
            for r in ex.map(_one, jobs):
                print(r)
    else:
        for j in jobs:
            print(_one(j))


if __name__ == "__main__":
    main()
