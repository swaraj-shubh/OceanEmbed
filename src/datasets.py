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
import pandas as pd
import xarray as xr

sys.path.append(str(Path(__file__).resolve().parent))
from config import (CHANNELS, DEPTHS, GRID_SHAPE, LAT, LON, PROCESSED, SPLITS, ZARR,
                    bathy_path, crop_to_model, n_channels)

STATS_PATH = PROCESSED / "norm_stats.json"
def clim_path(zarr_path):
    """Cache file beside the store it was fitted from. NOT a fixed global path: the
    self-check builds a tiny fake store, and a global cache meant its climatology was
    written to the real one's location and would have been silently reused by real runs."""
    return Path(zarr_path).with_suffix(".clim.npy")
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


def build_climatology(zarr_path=ZARR, crop=True):
    """Fit and cache the train-split monthly climatology. Run as its own process, because
    the dask reduction it performs leaves a thread pool behind that deadlocks DataLoader
    workers forked afterwards."""
    from baselines import Climatology
    c = Climatology.fit(zarr_path, crop=crop).months
    np.save(clim_path(zarr_path), c)
    return c


def build_bathymetry(zarr_path=ZARR, crop=True):
    """Fraction of the 15 depth levels GLORYS resolves at each cell, from the TRAIN split.

    0 on land, 1 where the full 1000 m column exists. This is free bathymetry: Y is already
    NaN below the sea floor, on exactly the model grid, needing no download and no regrid.
    Over the real store it is not degenerate -- 4,781 land cells, 11,199 full-depth cells
    and ~2,020 shelf cells spread across 2-14 levels.

    Fitted on train only. It is static, so no split could leak through it, but proving that
    costs one slice and removes the argument.

    ponytail: 15 quantised levels, not metres. If the channel earns its place, GEBCO or
    ETOPO regridded to 0.25 deg is the continuous upgrade.

    Run in its own process, like the climatology, for the same fork-after-threads reason.
    """
    y = xr.open_zarr(zarr_path).Y.sel(time=slice(*SPLITS["train"]))
    # .all() over a month of days, not one day, so a single bad day cannot carve a hole in
    # the sea floor.
    valid = np.isfinite(y.isel(time=slice(0, 30))).all("time").values     # [15, H, W]
    b = (valid.sum(0) / y.sizes["depth"]).astype(np.float32)
    b = crop_to_model(b) if crop else b
    np.save(bathy_path(zarr_path), b)
    return b


class NIODataset:
    """(X, Y, mask) samples for one time split.

    window=1 -> X is [7, H, W]; window=7 (M4) -> X is [7days, 7ch, H, W] ending at t.
    Targets are returned in physical units (degC); metrics are computed there.
    """

    def climatology(self, zarr_path):
        """Load the cached train-split monthly mean. Deliberately load-only.

        Fitting it here instead deadlocked training: the fit runs a dask reduction over
        the whole train split, which leaves a live thread pool in the parent, and the
        DataLoader then forks workers on top of that broken lock state -- four workers
        asleep, load average 0.00, no epoch ever starting. Build the cache in its own
        process (`python src/datasets.py --clim`) and this constructor only ever np.loads.
        """
        cache = clim_path(zarr_path)
        assert cache.exists(), (
            f"{cache.name} missing -- build it first with: python src/datasets.py --clim")
        return np.load(cache)

    def __init__(self, split, zarr_path=ZARR, window=1, stats=STATS_PATH, crop=True,
                 anomaly=False, extra=()):
        assert split in SPLITS, split
        self.ds = xr.open_zarr(zarr_path).sel(time=slice(*SPLITS[split]))
        self.window, self.crop = window, crop
        self.extra = tuple(extra)
        assert set(self.extra) <= {"clim", "aux"}, self.extra
        # Anomaly mode: the sample carries the climatology for its month and the model
        # predicts the departure from it, so climatology becomes the floor rather than
        # something to relearn. Zeros in absolute mode keeps one code path.
        # extra=("clim",) needs the same array for a different purpose -- as INPUT channels
        # the model may lean on per depth rather than a residual base it must lean on
        # everywhere. The two are independently switchable on purpose: docs/09 sec.5.1
        # measured the forced residual as worse above 200 m and better below 500 m.
        self.clim = (self.climatology(zarr_path)
                     if (anomaly or "clim" in self.extra) else None)
        self.anomaly = anomaly
        self.months = pd.DatetimeIndex(self.ds.time.values).month.to_numpy()
        s = json.loads(Path(stats).read_text()) if not isinstance(stats, dict) else stats
        self.xmu = np.asarray(s["X"]["mean"], np.float32)[:, None, None]
        self.xsd = np.asarray(s["X"]["std"], np.float32)[:, None, None]
        self.ymu = np.asarray(s["Y"]["mean"], np.float32)[:, None, None]
        self.ysd = np.asarray(s["Y"]["std"], np.float32)[:, None, None]
        if "aux" in self.extra:
            self.bathy = np.load(bathy_path(zarr_path))[None]              # [1, H, W]
            self.doy = pd.DatetimeIndex(self.ds.time.values).dayofyear.to_numpy()
            # Absolute position is real signal here, not a nuisance to be made invariant:
            # this is a regional model on a frozen grid, and the Arabian Sea and the Bay of
            # Bengal behave differently. Scaled to [-1, 1] to sit on the same scale as the
            # normalised surface channels.
            def _plane(v):
                v = crop_to_model(v) if crop else v
                return (2 * (v - v.min()) / (v.max() - v.min()) - 1).astype(np.float32)
            la = np.broadcast_to(LAT[:, None], GRID_SHAPE).astype(np.float32)
            lo = np.broadcast_to(LON[None, :], GRID_SHAPE).astype(np.float32)
            self.latlon = np.stack([_plane(la), _plane(lo)])
            assert self.latlon.shape[-2:] == self.bathy.shape[-2:], (
                "bathymetry cache and lat/lon planes disagree on the grid -- rebuild with "
                "python src/datasets.py --clim")
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
        if self.extra:
            parts = [x]                        # x is [window, 7, H, W]; order is frozen
            if "clim" in self.extra:
                # Each FRAME's own month -- a 7-day window can straddle a month boundary.
                # Normalised with the Y stats: it is in degC on the target's scale, not the
                # inputs'.
                c = self.clim[self.months[i:t + 1] - 1]
                parts.append((np.nan_to_num(c, nan=0.0) - self.ymu) / self.ysd)
            if "aux" in self.extra:
                # sin/cos so 31 Dec and 1 Jan are adjacent rather than 364 apart; per frame,
                # because a 7-day window spans seven different days.
                ang = 2 * np.pi * self.doy[i:t + 1].astype(np.float32) / 365.25
                hw = self.bathy.shape[-2:]
                season = np.broadcast_to(
                    np.stack([np.sin(ang), np.cos(ang)], 1)[:, :, None, None],
                    (t - i + 1, 2, *hw))
                static = np.broadcast_to(
                    np.concatenate([self.latlon, self.bathy])[None], (t - i + 1, 3, *hw))
                parts.append(np.concatenate([season, static], 1))
            x = np.concatenate(parts, axis=1).astype(np.float32)
        mask = np.isfinite(y)
        # Gate on `anomaly`, NOT on `self.clim is not None`: the climatology array is also
        # loaded for extra=("clim",), where it is an input channel and must NOT become a
        # residual base as well.
        base = (np.nan_to_num(self.clim[self.months[t] - 1], nan=0.0).astype(np.float32)
                if self.anomaly else np.zeros_like(y))
        return (x[0] if self.window == 1 else x), np.nan_to_num(y, nan=0.0), mask, base

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
    import sys as _sys
    import tempfile

    if "--clim" in _sys.argv:                   # build both caches, then exit
        c = build_climatology()
        b = build_bathymetry()
        print(f"climatology cached {c.shape} -> {clim_path(ZARR).name}")
        print(f"bathymetry cached {b.shape}, {(b == 0).mean():.1%} land "
              f"-> {bathy_path(ZARR).name}")
        raise SystemExit
    tmp = Path(tempfile.mkdtemp())
    store = tmp / "fake.zarr"
    _fake_store(store)                          # spans the train/val edge
    stats = compute_stats(store, tmp / "stats.json")
    assert len(stats["X"]["mean"]) == len(CHANNELS) and stats["X"]["coord"] == CHANNELS

    tr = NIODataset("train", store, stats=tmp / "stats.json")
    va = NIODataset("val", store, stats=tmp / "stats.json")
    assert len(tr) == 20 and len(va) == 20, (len(tr), len(va))          # split boundary respected
    assert tr.time.max() < va.time.min(), "train/val leakage"

    x, y, m, b = tr[0]
    assert x.shape == (7, 96, 176) and y.shape == (15, 96, 176) and m.shape == y.shape
    assert np.isfinite(x).all(), "NaN leaked into inputs"
    assert not m[:, 0, 0].any() and m[:, 50, 50].all(), "land mask wrong"
    assert abs(float(x[0].mean())) < 0.5, "normalisation looks unapplied"
    # absolute mode: the climatology base must be exactly zero, so pred + base == pred
    assert b.shape == y.shape and not b.any(), "base must be zeros unless anomaly=True"

    build_climatology(store)                    # loader is load-only; build the cache first
    an = NIODataset("train", store, stats=tmp / "stats.json", anomaly=True)
    xa, ya, ma, ba = an[0]
    assert np.allclose(ya, y) and np.allclose(xa, x), "anomaly mode must not alter x or y"
    assert ba.shape == y.shape and np.isfinite(ba).all()
    assert ba.any(), "anomaly mode returned an all-zero climatology"

    w = NIODataset("train", store, window=7, stats=tmp / "stats.json")
    assert len(w) == 14 and w[0][0].shape == (7, 7, 96, 176)
    assert w.time[0] == tr.time[6], "window must end at t, not start at it"

    # --- extra input channels (docs/10 tasks 5 and 6) ---
    ce = NIODataset("train", store, stats=tmp / "stats.json", extra=("clim",))
    xc, yc, mc, bc = ce[0]
    assert xc.shape == (n_channels(("clim",)), 96, 176) == (22, 96, 176), xc.shape
    assert np.allclose(xc[:7], x), "surface channels must come first and be unchanged"
    assert np.isfinite(xc).all(), "NaN leaked in through the climatology channels"
    assert np.abs(xc[7:]).max() > 0, "climatology channels are all zero"
    # The two mechanisms must stay independently switchable, or tasks 4 and 5 stop being
    # separable experiments: extra=("clim",) is INPUT, anomaly=True is a residual BASE.
    assert not bc.any(), "extra=('clim',) must not switch on the anomaly residual base"

    cw = NIODataset("train", store, window=7, stats=tmp / "stats.json", extra=("clim",))
    assert cw[0][0].shape == (7, 22, 96, 176), cw[0][0].shape

    build_bathymetry(store)
    ae = NIODataset("train", store, stats=tmp / "stats.json", extra=("aux",))
    xa2 = ae[0][0]
    assert xa2.shape == (n_channels(("aux",)), 96, 176) == (12, 96, 176), xa2.shape
    assert np.allclose(xa2[:7], x), "surface channels must come first and be unchanged"
    assert np.isfinite(xa2).all()
    assert xa2[7].std() < 1e-6 and xa2[8].std() < 1e-6, "day-of-year is not spatially flat"
    # Catches a day-of-year computed once from the split's first date instead of per sample.
    assert abs(float(xa2[7, 0, 0]) - float(ae[10][0][7, 0, 0])) > 1e-3, \
        "day-of-year identical ten days apart -- the channel is inert"
    assert xa2[9].min() >= -1.001 and xa2[9].max() <= 1.001, "lat channel out of [-1,1]"
    assert xa2[10].min() >= -1.001 and xa2[10].max() <= 1.001, "lon channel out of [-1,1]"
    assert xa2[9, 0, 0] < xa2[9, -1, 0], "lat channel is upside down"
    assert xa2[11].min() == 0.0 and xa2[11].max() == 1.0, "bathymetry is not in [0,1]"

    both = NIODataset("train", store, window=7, stats=tmp / "stats.json",
                      extra=("clim", "aux"))
    assert both[0][0].shape == (7, n_channels(("clim", "aux")), 96, 176) == (7, 27, 96, 176)

    shutil.rmtree(tmp, ignore_errors=True)
    print("datasets self-check OK -- 7 / 22 / 12 / 27 channel sets all verified")
