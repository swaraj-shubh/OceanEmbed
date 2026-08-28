"""Depth-wise masked metrics. The primary result table (CLAUDE.md rule 4).

pred/true: [D, ...] arrays; mask: same shape (or broadcastable), True = valid.
Land/missing cells must be masked out here and in the loss.
"""
import numpy as np
import pandas as pd

from config import DEPTHS


def depthwise(pred, true, mask=None, depths=DEPTHS):
    pred, true = np.asarray(pred, float), np.asarray(true, float)
    assert pred.shape == true.shape, (pred.shape, true.shape)
    assert pred.shape[0] == len(depths), "first axis must be depth"
    valid = np.isfinite(pred) & np.isfinite(true)
    if mask is not None:
        valid &= np.broadcast_to(np.asarray(mask, bool), pred.shape)

    rows = []
    for i, z in enumerate(depths):
        v = valid[i]
        n = int(v.sum())
        if n < 2:
            rows.append(dict(depth_m=z, n=n, rmse=np.nan, mae=np.nan, bias=np.nan, corr=np.nan))
            continue
        p, t = pred[i][v], true[i][v]
        d = p - t
        # corr is undefined when either field is constant (e.g. a single-value slab)
        sp, st = p.std(), t.std()
        corr = float(np.corrcoef(p, t)[0, 1]) if sp > 0 and st > 0 else np.nan
        rows.append(dict(depth_m=z, n=n, rmse=float(np.sqrt((d ** 2).mean())),
                         mae=float(np.abs(d).mean()), bias=float(d.mean()), corr=corr))
    return pd.DataFrame(rows)


def summary(df):
    """One blended number, for logging only -- never the headline result."""
    w = df["n"].to_numpy(float)
    return float(np.sqrt(np.nansum(w * df["rmse"] ** 2) / w.sum()))


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    true = rng.normal(size=(len(DEPTHS), 20, 20))
    df = depthwise(true + 1.0, true)                      # constant +1 offset
    assert np.allclose(df["rmse"], 1.0) and np.allclose(df["bias"], 1.0)
    assert np.allclose(df["corr"], 1.0) and np.allclose(df["mae"], 1.0)
    assert np.isclose(summary(df), 1.0)

    m = np.zeros((20, 20), bool); m[:5] = True            # mask keeps 100 cells/level
    bad = true.copy(); bad[:, 5:] = 99.0                  # garbage only outside the mask
    assert np.allclose(depthwise(bad, true, m).rmse, 0.0)
    assert (depthwise(bad, true, m)["n"] == 100).all()

    nan = true.copy(); nan[0, 0, 0] = np.nan
    assert depthwise(nan, true)["n"][0] == 399
    print("metrics self-check OK\n", df.head(3).to_string(index=False))
