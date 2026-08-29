"""Model-ready samples from the processed Zarr store.

Store contract (frozen, docs/04 sec.4):
    data/processed/nio_daily.zarr
      X (time, channel, y, x) float32   7 surface inputs, NaN where missing
      Y (time, depth,   y, x) float32   15 depth levels, NaN on land / missing
    coords: time, channel=CHANNELS, depth=DEPTHS, lat, lon

NaN *is* the mask -- no separate mask arrays to keep in sync. __getitem__ returns
inputs with NaN replaced by 0 (== the train mean, post-normalisation) and a boolean
target mask so the loss and the metrics never score land.

Plain sequence, not a torch.utils.data.Dataset: DataLoader accepts any object with
__len__/__getitem__, so this module stays importable without torch.
"""
import json
import sys
from pathlib import Path

import numpy as np
import xarray as xr

sys.path.append(str(Path(__file__).resolve().parent))
from config import CHANNELS, DEPTHS, PROCESSED, SPLITS, ZARR, crop_to_model

STATS_PATH = PROCESSED / "norm_stats.json"
SPLIT_EDGE = SPLITS["val"][0]   # first day of the validation split; used by the self-check


def compute_stats(zarr_path=ZARR, out=STATS_PATH):
    """Per-channel mean/std over the TRAIN split only (CLAUDE.md rule 2). Run once."""
    ds = xr.open_zarr(zarr_path)
    tr = ds.sel(time=slice(*SPLITS["train"]))
    stats = {}
    for name, dim in (("X", "channel"), ("Y", "depth")):
        mu = tr[name].mean(dim=("time", "lat", "lon"), skipna=True).values
        sd = tr[name].std(dim=("time", "lat", "lon"), skipna=True).values
        assert np.all(sd > 0), f"{name}: zero-variance level"
        stats[name] = {"mean": mu.tolist(), "std": sd.tolist(),
                       "coord": tr[dim].values.tolist()}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(stats, indent=2))
    return stats


class NIODataset:
    """(X, Y, mask) samples for one time split.

    window=1 -> X is [7, H, W]; window=7 (M4) -> X is [7days, 7ch, H, W] ending at t.
    Targets are returned in physical units (degC); metrics are computed there.
    """

    def __init__(self, split, zarr_path=ZARR, window=1, stats=STATS_PATH, crop=True):
        assert split in SPLITS, split
        self.ds = xr.open_zarr(zarr_path).sel(time=slice(*SPLITS[split]))
        self.window, self.crop = window, crop
        s = json.loads(Path(stats).read_text()) if not isinstance(stats, dict) else stats
        self.xmu = np.asarray(s["X"]["mean"], np.float32)[:, None, None]
        self.xsd = np.asarray(s["X"]["std"], np.float32)[:, None, None]
        assert list(self.ds.channel.values) == CHANNELS
        assert list(self.ds.depth.values) == DEPTHS
        assert len(self) > 0, f"{split}: window {window} longer than the split"

    def __len__(self):
        return self.ds.sizes["time"] - self.window + 1  # windows never cross a split edge

    def __getitem__(self, i):
        t = i + self.window - 1
        x = self.ds.X.isel(time=slice(i, t + 1)).values.astype(np.float32)
        y = self.ds.Y.isel(time=t).values.astype(np.float32)
        if self.crop:
            x, y = crop_to_model(x), crop_to_model(y)
        x = (x - self.xmu) / self.xsd
        x = np.nan_to_num(x, nan=0.0)          # missing -> train mean, post-normalisation
        mask = np.isfinite(y)
        return (x[0] if self.window == 1 else x), np.nan_to_num(y, nan=0.0), mask

    @property
    def time(self):
        return self.ds.time.values[self.window - 1:]


def _fake_store(path, days=40, seed=0):
    """Tiny synthetic store matching the contract -- lets the pipeline run before real data."""
    from config import GRID_SHAPE, LAT, LON
    rng = np.random.default_rng(seed)
    # straddle the train/val boundary so the split logic is actually exercised
    t = np.arange(np.datetime64(SPLIT_EDGE) - days // 2, np.datetime64(SPLIT_EDGE) + days // 2)
    X = rng.normal(size=(days, len(CHANNELS), *GRID_SHAPE)).astype("float32")
    Y = rng.normal(size=(days, len(DEPTHS), *GRID_SHAPE)).astype("float32")
    Y[:, :, :3, :3] = np.nan                    # land corner
    X[:, 1, :5, :] = np.nan                     # SSS swath gap
    ds = xr.Dataset(
        {"X": (("time", "channel", "lat", "lon"), X), "Y": (("time", "depth", "lat", "lon"), Y)},
        coords={"time": t, "channel": CHANNELS, "depth": DEPTHS,
                "lat": LAT, "lon": LON})
    ds.to_zarr(path, mode="w", zarr_format=2)
    return ds


if __name__ == "__main__":
    import shutil
    import tempfile
    tmp = Path(tempfile.mkdtemp())
    store = tmp / "fake.zarr"
    _fake_store(store)                          # spans the train/val edge
    stats = compute_stats(store, tmp / "stats.json")
    assert len(stats["X"]["mean"]) == len(CHANNELS) and stats["X"]["coord"] == CHANNELS

    tr = NIODataset("train", store, stats=tmp / "stats.json")
    va = NIODataset("val", store, stats=tmp / "stats.json")
    assert len(tr) == 20 and len(va) == 20, (len(tr), len(va))          # split boundary respected
    assert tr.time.max() < va.time.min(), "train/val leakage"

    x, y, m = tr[0]
    assert x.shape == (7, 96, 176) and y.shape == (15, 96, 176) and m.shape == y.shape
    assert np.isfinite(x).all(), "NaN leaked into inputs"
    assert not m[:, 0, 0].any() and m[:, 50, 50].all(), "land mask wrong"
    assert abs(float(x[0].mean())) < 0.5, "normalisation looks unapplied"

    w = NIODataset("train", store, window=7, stats=tmp / "stats.json")
    assert len(w) == 14 and w[0][0].shape == (7, 7, 96, 176)
    assert w.time[0] == tr.time[6], "window must end at t, not start at it"
    shutil.rmtree(tmp, ignore_errors=True)
    print("datasets self-check OK")
