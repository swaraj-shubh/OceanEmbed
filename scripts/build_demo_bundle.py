"""Build the offline demo bundle from the full artifacts. Run once; commit the output.

    python scripts/build_demo_bundle.py

Writes app/demo_data/ (~490 MB total, chunked) holding everything the Streamlit app needs
and nothing else: the 7 surface inputs, the GLORYS truth, the frozen bias-corrected
prediction, the Argo casts inside the test split, the metric tables, and a manifest.

Why chunked by quarter rather than one file each: the full test split is 725 days (2023-01-07
.. 2024-12-31 -- the model's 7-day window can't start before the split does). One file per
variable at that length is ~94/181/197 MB for inputs/pred/truth -- the last two exceed
GitHub's 100 MB per-file limit outright, so a plain `git push` would be rejected. Chunking by
calendar quarter keeps every tracked file in the same size class as the original 92-day demo
(each quarter is ~90 days), needs no Git LFS, and lets the app load only the quarter a click
actually touches.

Why a bundle at all rather than the live store: the store is 3.1 GB and the six prediction
cubes are 0.68 GB each, so the app would need 4+ GB and six checkpoints to answer a click.
Slicing per quarter and shipping the answers makes a click a small array lookup, needs
neither torch nor a GPU, and is small enough to commit -- which is what lets the same
artifact serve both the offline demo laptop and Streamlit Cloud.

Coverage: the full test split, 2023-01-07 .. 2024-12-31 (docs/09-12) -- every day the frozen
model can predict, not a representative slice.

Values are stored int16 with per-variable scale/offset. Round-trip error is ~0.0002 degC,
0.03% of the model's 0.786 degC RMSE -- irrelevant for anything the demo shows, and it takes
each quarter from ~55 MB to ~24 MB.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))
from config import CHANNELS, DEPTHS, INTERIM, SPLITS, ZARR, crop_to_model  # noqa: E402

OUT = ROOT / "app" / "demo_data"
RESULTS = ROOT / "results"
START, END = SPLITS["test"]                    # "2023-01-01", "2024-12-31" -- the full split

# The frozen model (results/FROZEN.md): six members, averaged, minus a val-fitted offset.
MEMBERS = [f"m4_convlstm_s{s}_best" for s in (1, 2, 3)] + [f"m4_dw_s{s}_best" for s in (1, 2, 3)]
OFFSET = RESULTS / "ens_mix6_offset.json"

# Metric tables the Skill tab reads. Keep the failures: they are the evidence (CLAUDE.md 12).
METRIC_FILES = ["ens_mix6_bc_test_argo.csv", "ens_mix6_test_argo.csv",
                "m4_convlstm_s1_best_test_argo.csv", "m4_convlstm_s2_best_test_argo.csv",
                "m4_convlstm_s3_best_test_argo.csv", "m2_unet_best_test_argo.csv",
                "m3_oceanembed_best_test_argo.csv", "M0_climatology_test_argo.csv",
                "GLORYS_target_test_argo.csv", "ablation_test.md"]


def packing(values):
    """int16 encoding with the offset at the midpoint, so int16 0 sits mid-range.

    One scale per variable: the seven surface channels have wildly different ranges (winds
    +/-40 m/s, SLA +/-2 m), and a shared scale would quantise SLA at the wind scale.
    """
    lo, hi = float(np.nanmin(values)), float(np.nanmax(values))
    sf = (hi - lo) / 65532.0 or 1e-6
    return {"dtype": "int16", "scale_factor": sf, "add_offset": (lo + hi) / 2.0,
            "_FillValue": -32767, "zlib": True, "complevel": 5}


def build_prediction():
    """Average the six members over the full test split, then subtract the val-fitted
    offset. Full split, not a window: the demo bundle now covers every predictable day."""
    from bias_correct import apply_offset
    cubes = []
    for m in MEMBERS:
        p = RESULTS / f"{m}_test_cube.nc"
        assert p.exists(), f"missing {p.name} -- run predict_cube.py --split test for {m}"
        cubes.append(xr.open_dataarray(p))
    aligned = list(xr.align(*cubes, join="inner"))
    assert aligned[0].sizes["time"] > 0, "members do not overlap"
    mean = sum(aligned) / len(aligned)
    meta = json.loads(OFFSET.read_text())
    assert meta["split_fitted_on"] == "val", (
        f"offset was fitted on {meta['split_fitted_on']}; the demo must ship the val-fitted "
        "one or every number on screen is circular")
    return apply_offset(mean, np.asarray(meta["offset"])), meta


def quarter_labels(dates):
    """'2023Q1' etc, in the order they first appear -- the chunk boundaries."""
    periods = pd.PeriodIndex(pd.DatetimeIndex(dates), freq="Q")
    seen = []
    for p in periods:
        s = str(p)
        if not seen or seen[-1] != s:
            seen.append(s)
    return periods, sorted(set(seen))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    assert pd.Timestamp(START) >= pd.Timestamp(SPLITS["test"][0]), \
        "the demo must stay inside the test split -- the only period the model never saw"

    pred, off_meta = build_prediction()
    days = pred.sizes["time"]
    print(f"full test split: {days} days, {str(pred.time.values[0])[:10]} -> "
          f"{str(pred.time.values[-1])[:10]}, prediction from {len(MEMBERS)} members")

    # Inputs and truth, cropped to the model grid so every array on screen shares one grid,
    # aligned to exactly the days we can predict (the store runs longer than the cubes do).
    ds = xr.open_zarr(ZARR).sel(time=pred.time)
    inputs = xr.Dataset({c: crop_to_model(ds.X.sel(channel=c).drop_vars("channel"))
                         for c in CHANNELS})
    truth = crop_to_model(ds.Y).to_dataset(name="thetao")

    periods, quarters = quarter_labels(pred.time.values)
    qidx = xr.DataArray(np.asarray([str(p) for p in periods]), dims="time",
                        coords={"time": pred.time})

    print(f"{len(quarters)} quarters: {quarters}")
    for q in quarters:
        sel = (qidx == q).values
        q_inputs, q_truth, q_pred = inputs.isel(time=sel), truth.isel(time=sel), pred.isel(time=sel)
        q_inputs.to_netcdf(OUT / f"inputs_{q}.nc",
                           encoding={c: packing(q_inputs[c].values) for c in CHANNELS})
        q_truth.to_netcdf(OUT / f"truth_{q}.nc",
                          encoding={"thetao": packing(q_truth.thetao.values)})
        q_pred.to_dataset(name="thetao").to_netcdf(
            OUT / f"pred_{q}.nc", encoding={"thetao": packing(q_pred.values)})
        print(f"  {q}: {int(sel.sum())} days")

    # Argo casts across the WHOLE test split, one file -- small enough (~11 MB) that
    # chunking it would only add lookup complexity for no size benefit.
    argo = pd.read_parquet(INTERIM / "argo_nio.parquet")
    argo["time"] = pd.to_datetime(argo["time"]).dt.tz_localize(None)
    argo = argo[(argo.time >= START) & (argo.time <= pd.Timestamp(END) + pd.Timedelta(days=1))]
    argo.to_parquet(OUT / "argo.parquet", index=False)

    (OUT / "metrics").mkdir(exist_ok=True)
    for f in METRIC_FILES:
        src = RESULTS / f
        if src.exists():
            (OUT / "metrics" / f).write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
        else:
            print(f"  note: {f} absent, skipped")

    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                             capture_output=True, text=True).stdout.strip() or "unknown"
    except Exception:
        sha = "unknown"
    manifest = {
        "quarters": quarters,
        # Full date list, not re-derived from NetCDF at app startup: populating the date
        # slider must not require opening 8 files just to read a time coordinate.
        "dates": [str(pd.Timestamp(t).date()) for t in pred.time.values],
        "split": f"test ({START}..{END}) -- never seen in training",
        "model": {"members": MEMBERS, "offset": OFFSET.name,
                  "offset_fitted_on": off_meta["split_fitted_on"],
                  "blended_rmse_full_test": 0.786},
        "grid": {"lat": [float(pred.lat.min()), float(pred.lat.max())],
                 "lon": [float(pred.lon.min()), float(pred.lon.max())],
                 "shape": [pred.sizes["lat"], pred.sizes["lon"]]},
        "depths_m": DEPTHS,
        "channels": CHANNELS,
        "argo_profiles": int(argo.profile.nunique()),
        "git_sha": sha,
        "note": "int16-packed, ~0.0002 degC round-trip error. Chunked by calendar quarter "
                "(quarters_ + inputs_/pred_/truth_<quarter>.nc) so no single tracked file "
                "exceeds GitHub's 100 MB limit. Built by scripts/build_demo_bundle.py; do "
                "not hand-edit.",
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))

    total = sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file())
    biggest = max((f.stat().st_size for f in OUT.rglob("*") if f.is_file()), default=0)
    print(f"\n{'file':<26}{'MB':>8}")
    for f in sorted(OUT.rglob("*")):
        if f.is_file():
            print(f"{f.relative_to(OUT).as_posix():<26}{f.stat().st_size / 1e6:8.1f}")
    print(f"{'TOTAL':<26}{total / 1e6:8.1f} MB   ({argo.profile.nunique()} Argo casts, "
          f"{days} days, {len(quarters)} quarters)")
    assert biggest < 95e6, (
        f"a single tracked file is {biggest/1e6:.0f} MB -- over GitHub's 100 MB limit; "
        "quarterly chunking should have prevented this, so something is wrong upstream")


if __name__ == "__main__":
    main()
