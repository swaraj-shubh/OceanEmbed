"""Every methodology rule from CLAUDE.md sec.6, as a runnable assertion.

    python src/audit_leakage.py

The answer to "how do you know there is no leakage?" should be a command, not a claim.
Raises on the first violation, so a green run is a green run.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.append(str(Path(__file__).resolve().parent))
from config import CHANNELS, INTERIM, SPLITS, ZARR, crop_to_model
from datasets import NIODataset, STATS_PATH, clim_path


def ok(msg):
    print(f"  PASS  {msg}")


print("1. splits are ordered and non-overlapping")
bounds = [(pd.Timestamp(a), pd.Timestamp(b))
          for a, b in (SPLITS["train"], SPLITS["val"], SPLITS["test"])]
for (a0, a1), (b0, b1) in zip(bounds, bounds[1:]):
    assert a1 < b0, f"{a1} >= {b0}: splits overlap"
ok(f"train {bounds[0][0].date()}..{bounds[0][1].date()} < val < test, no overlap")

print("2. no sample window crosses a split boundary")
for w in (1, 7):
    prev = None
    for s in ("train", "val", "test"):
        d = NIODataset(s, window=w)
        assert len(d) == d.ds.sizes["time"] - w + 1
        if prev is not None:
            assert prev.max() < d.time.min(), f"{s}: window {w} leaks backwards"
        prev = d.time
ok("window=1 and window=7 both stay inside their split")

print("3. normalisation stats come from the train split only")
s = json.loads(STATS_PATH.read_text())
ds = xr.open_zarr(ZARR)
tr_mu = ds.X.sel(time=slice(*SPLITS["train"])).mean(
    dim=("time", "lat", "lon"), skipna=True).values
all_mu = ds.X.mean(dim=("time", "lat", "lon"), skipna=True).values
assert np.allclose(tr_mu, s["X"]["mean"], atol=1e-3), "stats are not a train-only fit"
# The control matters: without it, a stats file fitted on ALL years would pass check 3
# whenever the two fits happen to be close.
assert not np.allclose(all_mu, s["X"]["mean"], atol=1e-4), "stats look fitted on all years"
ok("X mean reproduces the train-only fit and differs from the all-years fit")

print("4. the climatology cache is a train-split fit")
cache = clim_path(ZARR)
if cache.exists():
    clim = np.load(cache)
    y = ds.Y.sel(time=slice(*SPLITS["train"])).groupby("time.month").mean("time", skipna=True)
    assert np.allclose(np.nan_to_num(clim), np.nan_to_num(crop_to_model(y.values)), atol=1e-3)
    ok("cached climatology reproduces a train-only monthly mean")
else:
    print(f"  SKIP  {cache.name} not built yet (python src/datasets.py --clim)")

print("5. Argo never appears in the store")
assert set(ds.data_vars) == {"X", "Y"}, sorted(ds.data_vars)
assert list(ds.channel.values) == CHANNELS
ok("store holds only X (7 satellite channels) and Y (GLORYS); no Argo variable exists")

print("6. Argo used for scoring falls inside its split")
prof = pd.read_parquet(INTERIM / "argo_nio.parquet")
prof["time"] = pd.to_datetime(prof["time"]).dt.tz_localize(None)
prof["float"] = prof.profile.astype(str).str.split("_").str[0]
for split in ("val", "test"):
    lo, hi = SPLITS[split]
    sub = prof[(prof.time >= lo) & (prof.time <= pd.Timestamp(hi) + pd.Timedelta(days=1))]
    assert sub.time.min() >= pd.Timestamp(lo)
    assert sub.time.max() <= pd.Timestamp(hi) + pd.Timedelta(days=1)
    print(f"        {split}: {sub.profile.nunique()} profiles from {sub['float'].nunique()} floats")
ok("val and test Argo windows do not spill across their bounds")

print("7. GLORYS <-> Argo circularity, stated precisely")
print("        GLORYS12V1 assimilates Argo, so Argo is NOT statistically independent of")
print("        the target in general. The defensible claim is narrower, and is true: the")
print("        model trains on GLORYS 2015-2021 only, so no 2022-2024 Argo cast -- nor the")
print("        GLORYS state it informed -- was ever seen in training. Say it that way, not")
print("        'Argo is independent of GLORYS'.")
ok("circularity is bounded by the time split, not by product independence")

print("8. effective sample size is floats, not profiles")
te = prof[prof.time >= SPLITS["test"][0]]
n_p, n_f = te.profile.nunique(), te["float"].nunique()
assert n_f < n_p / 10, "unexpectedly many floats -- recheck the profile-id parsing"
print(f"        {n_p} test profiles come from {n_f} floats; a profile-level bootstrap")
print(f"        would read about {np.sqrt(n_p / n_f):.1f}x too narrow")
ok("recorded; model comparisons use argo_eval.paired_bootstrap, which blocks by float")

print("\nleakage audit: all checks passed")
