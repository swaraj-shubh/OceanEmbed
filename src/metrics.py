"""Depth-wise masked metrics -- the primary result table (CLAUDE.md rule 4).

Streaming accumulator so a whole test year never has to fit in memory; `depthwise()`
is the one-shot wrapper for small arrays (e.g. Argo profile matching).

pred/true: [D, ...]; mask: same shape or broadcastable, True = valid.
"""
import numpy as np
import pandas as pd

from config import DEPTHS


class DepthStats:
    """Accumulates per-depth sums; .table() -> RMSE/MAE/bias/corr DataFrame."""

    def __init__(self, depths=DEPTHS):
        self.depths = list(depths)
        self.s = np.zeros((len(self.depths), 7))  # n, sum_d, sum_d2, sum|d|, sum_p, sum_t, sum_pt
        self.sq = np.zeros((len(self.depths), 2))  # sum_p2, sum_t2

    def update(self, pred, true, mask=None):
        pred, true = np.asarray(pred, float), np.asarray(true, float)
        assert pred.shape == true.shape, (pred.shape, true.shape)
        assert pred.shape[0] == len(self.depths), "first axis must be depth"
        valid = np.isfinite(pred) & np.isfinite(true)
        if mask is not None:
            valid &= np.broadcast_to(np.asarray(mask, bool), pred.shape)
        for i in range(len(self.depths)):
            v = valid[i]
            if not v.any():
                continue
            p, t = pred[i][v], true[i][v]
            d = p - t
            self.s[i] += [v.sum(), d.sum(), (d ** 2).sum(), np.abs(d).sum(),
                          p.sum(), t.sum(), (p * t).sum()]
            self.sq[i] += [(p ** 2).sum(), (t ** 2).sum()]
        return self

    def table(self):
        n, sd, sd2, sad, sp, st, spt = self.s.T
        sp2, st2 = self.sq.T
        with np.errstate(invalid="ignore", divide="ignore"):
            cov = spt / n - (sp / n) * (st / n)
            vp, vt = sp2 / n - (sp / n) ** 2, st2 / n - (st / n) ** 2
            corr = np.where((vp > 1e-12) & (vt > 1e-12), cov / np.sqrt(vp * vt), np.nan)
            # R2 against the mean of the observations: 1 - SSE/SST, both already in the
            # accumulators, so no extra state. It is NOT corr^2 -- R2 punishes bias and can
            # go negative, which is the honest read at depths where we lose to climatology.
            sst = st2 - st ** 2 / n
            r2 = np.where(sst > 1e-12, 1.0 - sd2 / sst, np.nan)
            df = pd.DataFrame({"depth_m": self.depths, "n": n.astype(int),
                               "rmse": np.sqrt(sd2 / n), "mae": sad / n,
                               "bias": sd / n, "corr": corr, "r2": r2})
        # n == 0 -> 0/0 -> NaN; n == 1 -> zero variance -> corr NaN but RMSE/MAE/bias valid
        return df


def depthwise(pred, true, mask=None, depths=DEPTHS):
    return DepthStats(depths).update(pred, true, mask).table()


def summary(df):
    """One blended number, for logging only -- never the headline result."""
    w = df["n"].to_numpy(float)
    return float(np.sqrt(np.nansum(w * df["rmse"] ** 2) / w.sum()))


def blend_all(df):
    """Blend every column of a depth-wise table the same way summary() blends RMSE:
    n-weighted, so a depth with more matched casts counts for more. RMSE stays a weighted
    RMS (it is built from squared error, same as summary()); MAE/bias/corr/r2 are plain
    weighted means. Returns a dict -- this is one row, not a table."""
    w = df["n"].to_numpy(float)
    out = {"rmse": summary(df)}
    for col in ("mae", "bias", "corr", "r2"):
        if col in df:
            out[col] = float(np.nansum(w * df[col]) / w.sum())
    return out


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    true = rng.normal(size=(len(DEPTHS), 20, 20))
    df = depthwise(true + 1.0, true)                       # constant +1 offset
    assert np.allclose(df["rmse"], 1.0) and np.allclose(df["bias"], 1.0)
    assert np.allclose(df["mae"], 1.0) and np.allclose(df["corr"], 1.0)
    assert np.isclose(summary(df), 1.0)

    m = np.zeros((20, 20), bool); m[:5] = True             # 100 valid cells per level
    bad = true.copy(); bad[:, 5:] = 99.0                   # garbage only outside the mask
    assert np.allclose(depthwise(bad, true, m)["rmse"], 0.0)
    assert (depthwise(bad, true, m)["n"] == 100).all()

    nan = true.copy(); nan[0, 0, 0] = np.nan
    assert depthwise(nan, true)["n"][0] == 399

    acc = DepthStats()                                     # streaming == one-shot
    for k in range(4):
        acc.update(true[:, k * 5:(k + 1) * 5] + 1.0, true[:, k * 5:(k + 1) * 5])
    assert np.allclose(acc.table()["rmse"], df["rmse"]) and (acc.table()["n"] == df["n"]).all()

    off = rng.normal(size=true.shape)                      # corr against an independent field
    assert abs(float(depthwise(off, true)["corr"].mean())) < 0.2

    # blend_all must agree with summary() on rmse, and give the right weighted mean for
    # the rest -- built from two depths with DIFFERENT n and DIFFERENT values, so an
    # unweighted mean would give a different (wrong) answer than the n-weighted one.
    two = pd.DataFrame({"depth_m": [0, 100], "n": [100.0, 300.0],
                        "rmse": [1.0, 2.0], "mae": [1.0, 2.0],
                        "bias": [1.0, -1.0], "corr": [1.0, 0.5], "r2": [1.0, 0.0]})
    ba = blend_all(two)
    assert np.isclose(ba["rmse"], summary(two)), "blend_all rmse must match summary()"
    assert np.isclose(ba["mae"], (100 * 1.0 + 300 * 2.0) / 400), "mae not n-weighted"
    assert np.isclose(ba["bias"], (100 * 1.0 + 300 * -1.0) / 400), "bias not n-weighted"
    unweighted_mae = (1.0 + 2.0) / 2
    assert not np.isclose(ba["mae"], unweighted_mae), \
        "blend_all matches a plain average -- the n-weighting is not doing anything"
    print("metrics self-check OK\n", df.head(3).to_string(index=False))
