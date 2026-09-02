"""Generate a prediction cube from a checkpoint, then score it against raw Argo (track B2).

    python src/predict_cube.py --ckpt checkpoints/m2_unet.pt --split test

Writes results/<run>_<split>_cube.nc and prints the depth-wise Argo table. Argo is only
ever a scoring target here -- never an input, never a training target (CLAUDE.md rule 3).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xarray as xr

sys.path.append(str(Path(__file__).resolve().parent))
from argo_eval import evaluate_argo
from config import DEPTHS, INTERIM, LAT, LON, MODEL_SHAPE, GRID_SHAPE, REPORT_DEPTHS, ROOT, SPLITS, ZARR
from datasets import NIODataset
from evaluate import report
from metrics import summary
from train import MODELS

RESULTS = ROOT / "results"


def predict_cube(ckpt, split, zarr=ZARR, batch=32, dev=None):
    dev = dev or ("cuda" if torch.cuda.is_available() else "cpu")
    st = torch.load(ckpt, map_location=dev, weights_only=False)
    cfg = dict(st["cfg"]["model"])
    net = MODELS[cfg.pop("kind")](**cfg).to(dev).eval()   # M2 and M3 share this path
    net.load_state_dict(st["model"])

    # window must come from the checkpoint, not default to 1: M4 wants [B, T, C, H, W] and
    # a 1-day sample silently arrives as [B, C, H, W], which is a shape error at best.
    # `extra` must come from the checkpoint for the same reason `window` does: a model
    # trained on 22 channels handed 7 is a shape error, and one handed the WRONG 22 would
    # be worse -- silently wrong numbers.
    ds = NIODataset(split, zarr, window=st["cfg"].get("window", 1),
                    anomaly=st["cfg"].get("anomaly", False),
                    extra=tuple(st["cfg"].get("extra", ())))
    out = np.empty((len(ds), len(DEPTHS), *MODEL_SHAPE), np.float32)
    with torch.no_grad():
        for i in range(0, len(ds), batch):
            j0, j1 = i, min(i + batch, len(ds))
            xb = torch.from_numpy(np.stack([ds[j][0] for j in range(j0, j1)]))
            bb = torch.from_numpy(np.stack([ds[j][3] for j in range(j0, j1)]))
            out[j0:j1] = (net(xb.to(dev)) + bb.to(dev)).cpu().numpy()

    # crop_to_model trims 2 cells off each edge; the cube's coords must say so, or every
    # Argo profile would be matched to a cell two rows away from where it actually is.
    dy = (GRID_SHAPE[0] - MODEL_SHAPE[0]) // 2
    dx = (GRID_SHAPE[1] - MODEL_SHAPE[1]) // 2
    cube = xr.DataArray(out, dims=("time", "depth", "lat", "lon"),
                        coords={"time": ds.time, "depth": DEPTHS,
                                "lat": LAT[dy:dy + MODEL_SHAPE[0]],
                                "lon": LON[dx:dx + MODEL_SHAPE[1]]})

    # Blank the cells the model was never supervised on. A network still emits *some*
    # number on land, and Argo matching takes the nearest grid cell, so 42 of 6093 coastal
    # profiles (0.7%) were being scored against unconstrained output. That was survivable
    # while the garbage happened to look ocean-like, and catastrophic for the anomaly model
    # -- its base is zeroed on land, so it emitted ~0 degC where the truth is ~10, and 0.7%
    # of profiles moved 500 m RMSE from 0.30 to 0.94. Metrics already drop non-finite
    # predictions, so NaN here is all that is needed.
    land = np.isnan(ds.ds.Y).all("time").compute().values
    land = land[:, dy:dy + MODEL_SHAPE[0], dx:dx + MODEL_SHAPE[1]]
    return cube.where(~xr.DataArray(land, dims=("depth", "lat", "lon"),
                                    coords={"depth": DEPTHS, "lat": cube.lat, "lon": cube.lon}))


def split_profiles(split, argo=str(INTERIM / "argo_nio.parquet")):
    """Raw Argo casts restricted to one split's date window."""
    prof = pd.read_parquet(argo)
    # ERDDAP hands back tz-aware UTC; the split bounds and the cube's time axis are naive,
    # and pandas refuses to compare the two rather than silently guessing an offset.
    prof["time"] = pd.to_datetime(prof["time"]).dt.tz_localize(None)
    lo, hi = SPLITS[split]
    return prof[(prof.time >= lo) & (prof.time <= pd.Timestamp(hi) + pd.Timedelta(days=1))]


def score_cube(cube, split, argo=str(INTERIM / "argo_nio.parquet"), max_days=1):
    """Depth-wise table for a cube against the raw Argo of one split (track B2)."""
    return evaluate_argo(cube, split_profiles(split, argo), max_days=max_days)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/m2_unet.pt")
    p.add_argument("--split", default="test")
    p.add_argument("--argo", default=str(INTERIM / "argo_nio.parquet"))
    p.add_argument("--max-days", type=int, default=1)
    p.add_argument("--offset", default=None,
                   help="JSON from bias_correct.py; subtracted from the cube before scoring")
    p.add_argument("--ensemble", nargs="+", default=None,
                   help="average these cubes instead of running a checkpoint")
    p.add_argument("--run", default=None, help="output name (required with --ensemble)")
    a = p.parse_args()
    RESULTS.mkdir(exist_ok=True)

    if a.ensemble:
        cubes = [xr.open_dataarray(c) for c in a.ensemble]
        # Members can legitimately span different days: a window=7 model (M4) cannot predict
        # the first six days of a split, so its cube is 6 shorter than a window=1 model's.
        # Align on the intersection rather than refusing, but say how much was dropped --
        # silently averaging misaligned days would be far worse than a shape error.
        n_before = [c.sizes["time"] for c in cubes]
        cubes = list(xr.align(*cubes, join="inner")) if len(cubes) > 1 else cubes
        if cubes[0].sizes["time"] != max(n_before):
            print(f"time axes differ {n_before} -> intersecting on "
                  f"{cubes[0].sizes['time']} days")
        t0 = cubes[0]
        for c in cubes[1:]:
            assert c.shape == t0.shape, f"cube shapes differ after align: {c.shape} vs {t0.shape}"
            assert (c.time.values == t0.time.values).all(), "alignment failed"
        # Every member is NaN on land, so a plain mean keeps land NaN -- which is what the
        # metrics need. Do NOT use nanmean: it would invent values on cells no member was
        # supervised on, which is exactly the bug docs/09 sec.2 describes.
        cube = sum(cubes) / len(cubes)
        run = a.run or "ensemble"
        print(f"ensemble of {len(cubes)} cubes -> {run}")
    else:
        cube = predict_cube(a.ckpt, a.split)
        run = a.run or Path(a.ckpt).stem

    # Save BEFORE any offset is applied, under the pre-correction name: the artifact on disk
    # is always the raw cube, so an offset can be fitted on it later without a corrected
    # cube ever masquerading as an uncorrected one.
    out_cube = RESULTS / f"{run}_{a.split}_cube.nc"
    # Resolve both sides: `--ensemble results/x.nc` is relative and RESULTS is absolute, so
    # a naive compare misses and xarray then tries to write the file it has open for read.
    srcs = {Path(c).resolve() for c in (a.ensemble or [])}
    if out_cube.resolve() not in srcs:
        cube.to_netcdf(out_cube)

    if a.offset:
        from bias_correct import apply_offset
        meta = json.loads(Path(a.offset).read_text())
        assert meta["split_fitted_on"] != "test", \
            "that offset was fitted on the test split -- circular, refusing"
        cube = apply_offset(cube, np.asarray(meta["offset"]))
        run = f"{run}_bc" + ("m" if meta["by_month"] else "")
        print(f"applied offset fitted on {meta['split_fitted_on']} -> {run}")

    print(f"cube {cube.shape}  {str(cube.time.values[0])[:10]} -> {str(cube.time.values[-1])[:10]}")
    prof = split_profiles(a.split, a.argo)
    print(f"{prof.profile.nunique()} Argo profiles in {a.split} ({SPLITS[a.split][0]}..{SPLITS[a.split][1]})")

    df, n = evaluate_argo(cube, prof, max_days=a.max_days)
    df.to_csv(RESULTS / f"{run}_{a.split}_argo.csv", index=False)
    print(f"\n--- track B2: {run} vs raw Argo, {a.split} split, {n} profiles matched")
    print(report(df))
    print(f"blended RMSE {summary(df):.3f} degC")


if __name__ == "__main__":
    main()
