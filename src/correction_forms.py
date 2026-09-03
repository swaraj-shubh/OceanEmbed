"""Compare post-hoc correction FORMS honestly, on val, without touching test.

    python src/correction_forms.py --cube results/ens_mix6_val_cube.nc --split val

`bias_correct.py` fits the correction the project ships: fifteen numbers, one mean residual
per depth. The obvious next move is to condition it on something -- basin, season, latitude,
or the prediction itself. This module is the instrument that answers whether any of that
helps, and the answer it returned (docs/12 sec.2) is NO for all six alternatives tried.

The measurement has to be cross-validated or it is worthless: fitted in-sample, a form with
more bins ALWAYS scores better on the data it was fitted to, so an in-sample val comparison
selects the most flexible form every time regardless of truth. Here the split is a repeated
half-split over val FLOATS -- fit on half the floats, score the other half -- which is both
out-of-sample and blocked the way `argo_eval.paired_bootstrap` is blocked, for the same
reason: 3,107 val casts come from 83 floats, so casts are not independent samples.

Measured on the six-member ensemble, 40 x 2 folds, held-out blended RMSE in degC:

    depth x latitude band     0.7657 +/- 0.0180     -0.6%
    depth only  (SHIPPED)     0.7704 +/- 0.0191        --
    depth x basin             0.7705 +/- 0.0188     +0.0%
    depth, linear a + b*pred  0.7711 +/- 0.0202     +0.1%
    depth x season            0.7715 +/- 0.0193     +0.1%
    depth x basin x season    0.7719 +/- 0.0187     +0.2%
    no correction             0.8230 +/- 0.0156     +6.8%

Every alternative sits within +/-0.6% of the shipped form against +/-1.9% fold-to-fold
noise. Fifteen numbers is the right answer. Do not stratify the correction further; the
remaining thermocline error is variance, not bias, and no lookup table reaches it.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
from config import DEPTHS, INTERIM, SPLITS
from metrics import depthwise, summary

ND = len(DEPTHS)
BASIN_LON = 78.0        # Arabian Sea / Bay of Bengal, at the tip of India
MONSOON = (6, 7, 8, 9)  # SW monsoon
LAT_EDGES = (8.0, 16.0)


def _group_offsets(res, g, ngrp, min_n=40):
    """Mean residual per (depth, group). Groups too thin to fit keep the pooled offset."""
    pooled = np.nanmean(res, axis=1)
    off = np.tile(pooled[:, None], (1, ngrp))
    for k in range(ngrp):
        s = g == k
        if s.sum() >= min_n:
            m = np.nanmean(res[:, s], axis=1)
            off[:, k] = np.where(np.isfinite(m), m, pooled)
    return off


def _offset_form(key):
    """Build (fit, apply) for a grouped mean-offset form. `key.n` is the group count."""
    def fit(pred, obs, ctx):
        return _group_offsets(pred - obs, key(ctx), key.n)

    def apply(pred, ctx, off):
        return pred - off[:, key(ctx)]
    return fit, apply


def fit_linear(pred, obs, ctx, min_n=60):
    """Per depth: obs ~ a + b*pred. Corrects conditional bias, not just the mean.

    Falls back to a plain offset (a = mean residual, b = 1) at depths with too few casts
    or no spread in the prediction, so the form degrades to the shipped one rather than
    to a wild extrapolation.
    """
    ab = np.zeros((ND, 2))
    for d in range(ND):
        ok = np.isfinite(pred[d]) & np.isfinite(obs[d])
        if ok.sum() >= min_n and np.std(pred[d][ok]) > 1e-6:
            b, a = np.polyfit(pred[d][ok], obs[d][ok], 1)
            ab[d] = (a, b)
        else:
            ab[d] = (float(np.nanmean(obs[d] - pred[d])), 1.0)
    return ab


def apply_linear(pred, ctx, ab):
    return ab[:, 0:1] + ab[:, 1:2] * pred


def key_all(ctx):
    return np.zeros(len(ctx["lon"]), int)
key_all.n = 1


def key_basin(ctx):
    return (ctx["lon"] >= BASIN_LON).astype(int)
key_basin.n = 2


def key_season(ctx):
    return np.isin(ctx["month"], MONSOON).astype(int)
key_season.n = 2


def key_basin_season(ctx):
    return key_basin(ctx) * 2 + key_season(ctx)
key_basin_season.n = 4


def key_lat(ctx):
    return np.digitize(ctx["lat"], LAT_EDGES)
key_lat.n = len(LAT_EDGES) + 1


FORMS = {
    "none": (lambda p, o, c: None, lambda p, c, q: p),
    "depth (shipped)": _offset_form(key_all),
    "depth x basin": _offset_form(key_basin),
    "depth x season": _offset_form(key_season),
    "depth x basin x season": _offset_form(key_basin_season),
    "depth x lat band": _offset_form(key_lat),
    "depth linear a+b*p": (fit_linear, apply_linear),
}


def cross_validate(pred, obs, ctx, floats, forms=FORMS, reps=40, seed=0):
    """Repeated float-blocked half-split. -> {form name: (mean, std) of held-out blended}.

    Blocking by float rather than by cast is what makes this honest: a random cast-level
    split puts casts from the SAME float on both sides, so a form that has memorised that
    float's water mass scores well out-of-sample and looks like it generalises.
    """
    uf = np.unique(floats)
    assert uf.size >= 4, f"need at least 4 floats to half-split, got {uf.size}"
    rng = np.random.default_rng(seed)
    out = {k: [] for k in forms}
    for _ in range(reps):
        sh = rng.permutation(uf)
        h0 = np.isin(floats, sh[:uf.size // 2])
        h1 = ~h0
        for fit_m, sc_m in ((h0, h1), (h1, h0)):
            cf = {k: v[fit_m] for k, v in ctx.items()}
            cs = {k: v[sc_m] for k, v in ctx.items()}
            for name, (fit, apply) in forms.items():
                q = fit(pred[:, fit_m], obs[:, fit_m], cf)
                out[name].append(summary(depthwise(apply(pred[:, sc_m], cs, q),
                                                   obs[:, sc_m])))
    return {k: (float(np.mean(v)), float(np.std(v))) for k, v in out.items()}


def main():
    import xarray as xr
    from argo_eval import match_profiles

    p = argparse.ArgumentParser()
    p.add_argument("--cube", required=True)
    p.add_argument("--split", default="val")
    p.add_argument("--argo", default=str(INTERIM / "argo_nio.parquet"))
    p.add_argument("--reps", type=int, default=40)
    a = p.parse_args()
    assert a.split != "test", (
        "refusing to select a correction form on the test split -- screen on val "
        "(CLAUDE.md rule 3, docs/11 sec.2)")

    prof = pd.read_parquet(a.argo)
    prof["time"] = pd.to_datetime(prof["time"]).dt.tz_localize(None)
    lo, hi = SPLITS[a.split]
    prof = prof[(prof.time >= lo) & (prof.time <= pd.Timestamp(hi) + pd.Timedelta(days=1))]

    pred, obs, floats, times = match_profiles(xr.open_dataarray(a.cube), prof)
    # Position comes back through (float, cast time), which identifies a cast uniquely --
    # rather than re-deriving the match order from `prof` and risking falling out of step.
    first = prof.groupby("profile").first()
    loc = {(str(pid).split("_")[0], pd.Timestamp(r.time)): (r.lat, r.lon)
           for pid, r in first.iterrows()}
    ll = np.array([loc.get((f, t), (np.nan, np.nan)) for f, t in zip(floats, times)])
    ctx = {"lat": ll[:, 0], "lon": ll[:, 1], "month": times.month.to_numpy()}

    print(f"{a.split}: {pred.shape[1]} casts / {np.unique(floats).size} floats "
          f"(AS {int((ll[:, 1] < BASIN_LON).sum())} / BoB {int((ll[:, 1] >= BASIN_LON).sum())})")
    res = cross_validate(pred, obs, ctx, floats, reps=a.reps)
    base = res["depth (shipped)"][0]
    print(f"\nheld-out blended RMSE, {a.reps} x 2 float-blocked folds")
    for k, (m, s) in sorted(res.items(), key=lambda kv: kv[1][0]):
        print(f"  {k:24s} {m:.4f} +/- {s:.4f}   {100 * (m - base) / base:+5.1f}% vs shipped")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main()
        raise SystemExit

    # --- self-check: the comparison must detect stratification when it is real, and must
    # NOT prefer the extra bins when it is not. The second assertion is the one that
    # matters; the first is the control that stops it being vacuous.
    rng = np.random.default_rng(0)
    NF, NC = 60, 20                                   # floats, casts each
    floats = np.repeat([f"{5900000 + i}" for i in range(NF)], NC)
    lon = rng.uniform(55, 100, NF * NC)
    lat = rng.uniform(0, 25, NF * NC)
    month = rng.integers(1, 13, NF * NC)
    ctx = {"lat": lat, "lon": lon, "month": month}

    truth = np.linspace(29.0, 7.0, ND)[:, None]
    water = np.repeat(rng.normal(0, 0.5, (ND, NF)), NC, axis=1)   # per-float water mass
    obs = truth + water + rng.normal(0, 0.3, (ND, NF * NC))

    depth_bias = np.linspace(-0.1, 0.9, ND)[:, None]
    # Irreducible per-cast model error. Without it pred - obs is EXACTLY the bias, every
    # form corrects perfectly, every score is 0.0000 and the comparison below is vacuous.
    # This is also the real situation: docs/12 sec.3 measured the thermocline residual as
    # variance, not bias, so no form of lookup table can drive it to zero.
    noise = rng.normal(0, 0.6, (ND, NF * NC))

    # (a) truth is depth-only. More bins can only fit noise, so they must not win.
    flat = cross_validate(obs + depth_bias + noise, obs, ctx, floats, reps=12)
    shipped = flat["depth (shipped)"][0]
    assert flat["none"][0] > shipped, "the shipped correction did not beat doing nothing"
    for name in ("depth x basin", "depth x season", "depth x basin x season"):
        assert flat[name][0] >= shipped - 1e-3, \
            f"{name} beat the shipped form on a purely depth-wise bias: {flat[name][0]:.4f}"

    # (b) control: a genuinely basin-dependent bias MUST be found. Without this, (a) would
    # pass for a cross_validate that ignored its groups entirely.
    extra = np.where(lon >= BASIN_LON, 0.8, -0.8)[None, :]
    strat = cross_validate(obs + depth_bias + extra + noise, obs, ctx, floats, reps=12)
    assert strat["depth x basin"][0] < strat["depth (shipped)"][0] - 0.05, \
        ("a real basin-dependent bias was not detected: "
         f"{strat['depth x basin'][0]:.4f} vs {strat['depth (shipped)'][0]:.4f}")
    # and the season split, which is orthogonal to the injected bias, must NOT be fooled
    assert strat["depth x season"][0] > strat["depth x basin"][0], \
        "an irrelevant grouping matched a relevant one"

    # the linear form must recover a pure multiplicative distortion the offset cannot
    lin = cross_validate(obs * 1.15 + noise, obs, ctx, floats, reps=12)
    assert lin["depth linear a+b*p"][0] < lin["depth (shipped)"][0], \
        "linear form failed on a scale error, which is exactly what it is for"

    print(f"correction_forms self-check OK -- shipped {shipped:.4f}, "
          f"no-correction {flat['none'][0]:.4f}; basin bias detected "
          f"({strat['depth x basin'][0]:.4f} < {strat['depth (shipped)'][0]:.4f})")
