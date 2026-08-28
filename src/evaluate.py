"""Depth-wise evaluation of any predictor against GLORYS on a time split.

    python src/evaluate.py --model climatology --split test

`predict(x, t)` takes one sample's inputs and its timestamp and returns [15,H,W] in degC.
Metrics are computed in physical units on unmasked (ocean, observed) cells only.
"""
import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))
from config import REPORT_DEPTHS, ROOT, ZARR
from datasets import NIODataset
from metrics import DepthStats, summary

RESULTS = ROOT / "results"


def evaluate(predict, split="test", window=1, zarr=ZARR, stats=None, name=None):
    ds = NIODataset(split, zarr, window=window, **({} if stats is None else {"stats": stats}))
    acc = DepthStats()
    times = ds.time
    for i in range(len(ds)):
        x, y, m = ds[i]
        acc.update(predict(x, times[i]), y, m)
    df = acc.table()
    if name:
        RESULTS.mkdir(exist_ok=True)
        df.to_csv(RESULTS / f"{name}_{split}.csv", index=False)
    return df


def report(df):
    sub = df[df["depth_m"].isin(REPORT_DEPTHS)]
    return sub.to_string(index=False, float_format=lambda v: f"{v:8.3f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="climatology")
    p.add_argument("--split", default="test")
    p.add_argument("--zarr", default=ZARR)
    a = p.parse_args()
    if a.model != "climatology":
        raise SystemExit(f"unknown model {a.model!r} -- only M0 exists so far")
    from baselines import Climatology
    m0 = Climatology.fit(a.zarr)
    df = evaluate(m0, a.split, zarr=a.zarr, name="M0_climatology")
    print(report(df), f"\nblended RMSE {summary(df):.3f} degC")
