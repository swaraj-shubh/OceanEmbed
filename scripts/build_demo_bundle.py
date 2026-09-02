"""Build the offline demo bundle from the full artifacts. Run once; commit the output.

    python scripts/build_demo_bundle.py

Writes app/demo_data/ (~70 MB) holding everything the Streamlit app needs and nothing else:
the 7 surface inputs, the GLORYS truth, the frozen bias-corrected prediction, the Argo casts
that fall in the window, the metric tables, and a manifest recording provenance.

Why a bundle rather than the live store: the store is 3.1 GB and the prediction cubes are
0.68 GB each, so the app would need 4+ GB and six checkpoints to answer a click. Slicing a
window and shipping the answers makes a click an array lookup, needs neither torch nor a
GPU, and is small enough to commit -- which is what lets the same artifact serve both the
offline demo laptop and Streamlit Cloud.

Window: 2023-10-01 .. 2023-12-31. Inside the test split (the model never trained on it) and
it contains Cyclone Tej in the Arabian Sea and Cyclone Michaung in the Bay of Bengal, so the
scripted demo path has a real event to point at.

Values are stored int16 with per-variable scale/offset. Round-trip error is ~0.0002 degC,
0.03% of the model's 0.786 degC RMSE -- irrelevant for anything the demo shows, and it takes
the bundle from ~166 MB to ~70 MB.
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
START, END = "2023-10-01", "2023-12-31"

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
    """Average the six members over the window, then subtract the val-fitted offset."""
    from bias_correct import apply_offset
    cubes = []
    for m in MEMBERS:
        p = RESULTS / f"{m}_test_cube.nc"
        assert p.exists(), f"missing {p.name} -- run predict_cube.py --split test for {m}"
        cubes.append(xr.open_dataarray(p).sel(time=slice(START, END)))
    aligned = list(xr.align(*cubes, join="inner"))
    assert aligned[0].sizes["time"] > 0, "members do not overlap the window"
    mean = sum(aligned) / len(aligned)
    meta = json.loads(OFFSET.read_text())
    assert meta["split_fitted_on"] == "val", (
        f"offset was fitted on {meta['split_fitted_on']}; the demo must ship the val-fitted "
        "one or every number on screen is circular")
    return apply_offset(mean, np.asarray(meta["offset"])), meta


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    assert pd.Timestamp(START) >= pd.Timestamp(SPLITS["test"][0]), \
        "the demo window must sit inside the test split -- it is the only period the model " \
        "never saw, and the whole demo claim rests on that"

    pred, off_meta = build_prediction()
    days = pred.sizes["time"]
    print(f"window {START}..{END}: {days} days, prediction from {len(MEMBERS)} members")

    # Inputs and truth, cropped to the model grid so every array on screen shares one grid.
    ds = xr.open_zarr(ZARR).sel(time=slice(START, END))
    ds = ds.sel(time=pred.time)                       # exactly the days we can predict
    inputs = xr.Dataset({c: crop_to_model(ds.X.sel(channel=c).drop_vars("channel"))
                         for c in CHANNELS})
    truth = crop_to_model(ds.Y).to_dataset(name="thetao")

    inputs.to_netcdf(OUT / "inputs.nc",
                     encoding={c: packing(inputs[c].values) for c in CHANNELS})
    truth.to_netcdf(OUT / "truth.nc", encoding={"thetao": packing(truth.thetao.values)})
    pred.to_dataset(name="thetao").to_netcdf(
        OUT / "pred.nc", encoding={"thetao": packing(pred.values)})

    # Argo casts inside the window. These are the independent observations the profile tab
    # overlays; they are never a model input (CLAUDE.md rule 3).
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
        "window": {"start": START, "end": str(pred.time.values[-1])[:10], "days": int(days)},
        "split": "test (2023-01-01..2024-12-31) -- never seen in training",
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
        "note": "int16-packed, ~0.0002 degC round-trip error. Built by "
                "scripts/build_demo_bundle.py; do not hand-edit.",
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))

    total = sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file())
    print(f"\n{'file':<22}{'MB':>8}")
    for f in sorted(OUT.rglob("*")):
        if f.is_file():
            print(f"{f.relative_to(OUT).as_posix():<22}{f.stat().st_size / 1e6:8.1f}")
    print(f"{'TOTAL':<22}{total / 1e6:8.1f} MB   ({argo.profile.nunique()} Argo casts)")
    assert total < 95e6, f"bundle is {total/1e6:.0f} MB -- too large to commit comfortably"


if __name__ == "__main__":
    main()
