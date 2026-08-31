"""Generate a prediction cube from a checkpoint, then score it against raw Argo (track B2).

    python src/predict_cube.py --ckpt checkpoints/m2_unet.pt --split test

Writes results/<run>_<split>_cube.nc and prints the depth-wise Argo table. Argo is only
ever a scoring target here -- never an input, never a training target (CLAUDE.md rule 3).
"""
import argparse
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

    ds = NIODataset(split, zarr)
    out = np.empty((len(ds), len(DEPTHS), *MODEL_SHAPE), np.float32)
    with torch.no_grad():
        for i in range(0, len(ds), batch):
            xb = torch.from_numpy(np.stack([ds[j][0] for j in range(i, min(i + batch, len(ds)))]))
            out[i:i + xb.shape[0]] = net(xb.to(dev)).cpu().numpy()

    # crop_to_model trims 2 cells off each edge; the cube's coords must say so, or every
    # Argo profile would be matched to a cell two rows away from where it actually is.
    dy = (GRID_SHAPE[0] - MODEL_SHAPE[0]) // 2
    dx = (GRID_SHAPE[1] - MODEL_SHAPE[1]) // 2
    return xr.DataArray(out, dims=("time", "depth", "lat", "lon"),
                        coords={"time": ds.time, "depth": DEPTHS,
                                "lat": LAT[dy:dy + MODEL_SHAPE[0]],
                                "lon": LON[dx:dx + MODEL_SHAPE[1]]})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/m2_unet.pt")
    p.add_argument("--split", default="test")
    p.add_argument("--argo", default=str(INTERIM / "argo_nio.parquet"))
    p.add_argument("--max-days", type=int, default=1)
    a = p.parse_args()

    cube = predict_cube(a.ckpt, a.split)
    run = Path(a.ckpt).stem
    RESULTS.mkdir(exist_ok=True)
    cube.to_netcdf(RESULTS / f"{run}_{a.split}_cube.nc")
    print(f"cube {cube.shape}  {str(cube.time.values[0])[:10]} -> {str(cube.time.values[-1])[:10]}")

    prof = pd.read_parquet(a.argo)
    # ERDDAP hands back tz-aware UTC; the split bounds and the cube's time axis are naive,
    # and pandas refuses to compare the two rather than silently guessing an offset.
    prof["time"] = pd.to_datetime(prof["time"]).dt.tz_localize(None)
    lo, hi = SPLITS[a.split]
    prof = prof[(prof.time >= lo) & (prof.time <= pd.Timestamp(hi) + pd.Timedelta(days=1))]
    print(f"{prof.profile.nunique()} Argo profiles in {a.split} ({lo}..{hi})")

    df, n = evaluate_argo(cube, prof, max_days=a.max_days)
    df.to_csv(RESULTS / f"{run}_{a.split}_argo.csv", index=False)
    print(f"\n--- track B2: {run} vs raw Argo, {a.split} split, {n} profiles matched")
    print(report(df))
    print(f"blended RMSE {summary(df):.3f} degC")


if __name__ == "__main__":
    main()
