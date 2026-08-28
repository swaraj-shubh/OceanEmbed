"""M0 -- monthly climatology of GLORYS temperature, fitted on the TRAIN split only.

Non-AI reference every later stage must beat (CLAUDE.md sec.3). In a monsoon-dominated
region this is a strong baseline, which is exactly why it has to be reported.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.append(str(Path(__file__).resolve().parent))
from config import SPLITS, ZARR, crop_to_model


class Climatology:
    """predict(x, t) -> [15,H,W]: the train-split mean field for t's calendar month."""

    def __init__(self, months):        # [12, 15, H, W]
        self.months = months

    @classmethod
    def fit(cls, zarr_path=ZARR, split="train", crop=True):
        y = xr.open_zarr(zarr_path).Y.sel(time=slice(*SPLITS[split]))
        clim = y.groupby("time.month").mean("time", skipna=True)
        clim = clim.reindex(month=range(1, 13)).values.astype("float32")
        assert np.isfinite(clim).any(), "empty climatology"
        return cls(crop_to_model(clim) if crop else clim)

    def __call__(self, x, t):
        m = pd.Timestamp(t).month
        c = self.months[m - 1]
        if not np.isfinite(c).any():   # month unseen in train (only happens on toy stores)
            c = np.nanmean(self.months, axis=0)
        return c


if __name__ == "__main__":
    import shutil
    import tempfile
    from datasets import _fake_store, compute_stats
    from evaluate import evaluate
    from metrics import summary

    tmp = Path(tempfile.mkdtemp())
    store, sj = tmp / "fake.zarr", tmp / "stats.json"
    ds = _fake_store(store, days=40)
    compute_stats(store, sj)

    m0 = Climatology.fit(store)
    assert m0.months.shape[1:] == (15, 96, 176)
    assert np.isfinite(m0.months[11]).any(), "December seen in train, must be fitted"
    assert not np.isfinite(m0.months[5]).any(), "June absent from train, must stay NaN"

    df = evaluate(m0, "val", zarr=store, stats=sj)
    assert (df["n"] > 0).all() and np.isfinite(df["rmse"]).all()
    # land corner is masked everywhere, so it can never enter the count
    assert df["n"].iloc[0] == 20 * (96 * 176 - 1 * 1), df["n"].iloc[0]
    print(f"baseline self-check OK -- toy blended RMSE {summary(df):.3f}")
