"""Build the methodology page's real-data assets from the committed demo bundle and results.

    app/venv/bin/python methodology/make_assets.py

Writes methodology/img/*.webp (the 7 satellite inputs, the 7-day SST window and the
reconstruction at all 15 depths, for one showcase day) and methodology/data.js (every
number the page quotes, read from results/ -- nothing on the page is typed in by hand).
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "app"), str(ROOT / "src")]
import loader as L  # noqa: E402
from metrics import blend_all  # noqa: E402

OUT = Path(__file__).parent
IMG = OUT / "img"
DAY = "2023-12-04"                      # Cyclone Michaung over the Bay of Bengal, test split

LAND = np.array([38, 49, 61])           # slate, reads as land on the dark page
GAP = np.array([20, 28, 38])            # ocean with no observation that day
SEQ_TEMP = ["#fff5eb", "#fdd0a2", "#fd8d3c", "#d94801", "#7f2704"]
SEQ_BLUE = ["#f7fbff", "#c6dbef", "#6baed6", "#2171b5", "#08306b"]
DIVERGING = ["#2a78d6", "#f0efec", "#e34948"]
SIGNED = {"sla", "cur_u", "cur_v", "wind_u", "wind_v"}


def _rgb(h):
    return np.array([int(h[i:i + 2], 16) for i in (1, 3, 5)], float)


def colorize(a, lo, hi, stops, land):
    """Map a 2-D field to RGB on `stops`, land slate, other NaN a dark gap; north up."""
    cols = np.stack([_rgb(s) for s in stops])
    t = np.clip((a - lo) / (hi - lo), 0, 1)
    pos = np.nan_to_num(t) * (len(stops) - 1)
    i = np.minimum(pos.astype(int), len(stops) - 2)
    f = (pos - i)[..., None]
    rgb = cols[i] * (1 - f) + cols[i + 1] * f
    rgb[np.isnan(a)] = GAP
    rgb[land] = LAND
    return np.flipud(rgb).astype(np.uint8)


def save(rgb, name):
    im = Image.fromarray(rgb).resize((rgb.shape[1] * 3, rgb.shape[0] * 3), Image.LANCZOS)
    im.save(IMG / f"{name}.webp", quality=88)


def main():
    IMG.mkdir(exist_ok=True)
    m = L.manifest()
    cr = m["colour_ranges"]
    land = L.land_mask()

    x = L.inputs(DAY)
    for ch in m["channels"]:
        lo, hi = cr["inputs"][ch]
        stops = DIVERGING if ch in SIGNED else SEQ_TEMP if ch == "sst" else SEQ_BLUE
        save(colorize(x[ch].values, lo, hi, stops, land), f"in_{ch}")

    window = pd.date_range(end=DAY, periods=7)
    lo, hi = cr["inputs"]["sst"]
    for k, d in enumerate(window):
        save(colorize(L.inputs(d)["sst"].values, lo, hi, SEQ_TEMP, land), f"win_{k}")

    for d in m["depths_m"]:
        lo, hi = cr["temp"][str(d)]
        save(colorize(L.field(DAY, d).values, lo, hi, SEQ_TEMP, land), f"out_{d}")

    # Alpha mask of the shelf at 100 m (ocean, but no water that deep): the page hatches it
    # to show what the masked loss leaves out.
    shelf = np.isnan(L.field(DAY, 100).values) & ~land
    a8 = np.flipud(np.where(shelf, 255, 0)).astype(np.uint8)
    rgba = np.dstack([np.full_like(a8, 255)] * 3 + [a8])
    Image.fromarray(rgba, "RGBA").resize((176 * 3, 96 * 3), Image.NEAREST).save(IMG / "shelf_100.png")

    # One real column: the Bay of Bengal Argo cast nearest the showcase day with a full
    # 15-level match, and our reconstruction + GLORYS at that exact cell.
    a = L.argo()
    near = a[(a.time - pd.Timestamp(DAY)).abs() <= pd.Timedelta(days=1)]   # eval matches ±1 day
    first = near.groupby("profile")[["lat", "lon"]].first()
    first = first[(first.lon > 80) & (first.lon < 95) & (first.lat > 5)]
    sample = None
    for pid, r in first.iterrows():
        c = L.argo_comparison(DAY, r.lat, r.lon, max_days=1)
        if c and c["n_levels"] == 15:
            _, glo = L.profile(DAY, c["lat"], c["lon"], source="truth")
            keep = c["pres"] <= 1000
            sample = {"float": c["profile"], "lat": c["lat"], "lon": c["lon"],
                      "date": str(pd.Timestamp(c["time"]).date()),
                      "pred": np.round(c["pred"], 3).tolist(),
                      "glorys": np.round(glo, 3).tolist(),
                      "argo15": np.round(c["obs"], 3).tolist(),
                      "argo_raw": {"p": np.round(c["pres"][keep], 1).tolist(),
                                   "t": np.round(c["temp"][keep], 3).tolist()},
                      "rmse": round(c["rmse"], 3)}
            break
    assert sample, "no Bay of Bengal cast with a full 15-level match near the showcase day"

    # Depth-weighted MSE uses 1/variance per depth. Shape shown from GLORYS variability in
    # the bundle (the real weights come from train-split stats, which are not shipped).
    t = L._truth_q(L._quarter(DAY))
    sd = np.array([float(np.nanstd(t.sel(depth=d).values)) for d in m["depths_m"]])
    w = 1 / sd ** 2
    w = w / w.mean()

    def col(name, key="rmse"):
        return np.round(pd.read_csv(ROOT / "results" / name)[key].to_numpy(), 3).tolist()

    def blended(name):
        return round(blend_all(pd.read_csv(ROOT / "results" / name))["rmse"], 3)

    abl = (ROOT / "results" / "ablation_test.md").read_text()
    single = re.search(r"\|\s*m4_convlstm_best\s*\|\s*3\s*\|\s*([\d.]+) \+/- ([\d.]+)", abl)
    casts = re.search(r"against ([\d,]+)\s+independent\s+Argo\s+casts",
                      (ROOT / "results" / "FROZEN.md").read_text()).group(1)
    offset = json.loads((ROOT / "results" / "ens_mix6_offset.json").read_text())
    assert offset["split_fitted_on"] == "val"

    data = {
        "day": DAY,
        "depths": m["depths_m"],
        "grid": {"lat": [0.625, 24.375], "lon": [55.625, 99.375], "shape": [96, 176]},
        "window": [str(d.date()) for d in window],
        "colour_ranges": {"temp": cr["temp"], "inputs": cr["inputs"]},
        "sample": sample,
        "dw_weights": np.round(w, 3).tolist(),
        "offset": np.round(offset["offset"], 3).tolist(),
        "rmse": {"before_bc": col("ens_mix6_test_argo.csv"),
                 "final": col("ens_mix6_bc_test_argo.csv"),
                 "climatology": col("M0_climatology_test_argo.csv"),
                 "glorys": col("GLORYS_target_test_argo.csv")},
        "blended": {"final": blended("ens_mix6_bc_test_argo.csv"),
                    "ensemble": blended("ens_mix6_test_argo.csv"),
                    "single": float(single.group(1)), "single_sd": float(single.group(2)),
                    "climatology": blended("M0_climatology_test_argo.csv"),
                    "glorys": blended("GLORYS_target_test_argo.csv")},
        "argo_casts": casts,
        # Every cast within 5 days of the showcase day: how sparse the deep view really is.
        "argo_dots": np.round(a[(a.time - pd.Timestamp(DAY)).abs() <= pd.Timedelta(days=5)]
                              .groupby("profile")[["lat", "lon"]].first().to_numpy(), 2).tolist(),
    }
    # A script, not JSON, so the page reads it with a plain <script> tag -- no fetch.
    (OUT / "data.js").write_text("window.DATA = " + json.dumps(data) + ";\n")
    print(f"assets: {len(list(IMG.glob('*.webp')))} images, "
          f"{sum(f.stat().st_size for f in IMG.glob('*.webp')) / 1e3:.0f} kB; "
          f"sample float {sample['float']} at {sample['lat']:.2f}N {sample['lon']:.2f}E "
          f"(RMSE {sample['rmse']}); blended {data['blended']}")


if __name__ == "__main__":
    main()
