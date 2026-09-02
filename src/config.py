"""Frozen project constants (docs/04 §2-3). Import these; never re-type them."""
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW, INTERIM, PROCESSED = DATA / "raw", DATA / "interim", DATA / "processed"
ZARR = PROCESSED / "nio_daily.zarr"

# Region: Arabian Sea + Bay of Bengal. Cell centres on the OISST 0.25 deg grid.
LAT_MIN, LAT_MAX, LON_MIN, LON_MAX = 0.0, 25.0, 55.0, 100.0
RES = 0.25
LAT = np.arange(LAT_MIN + RES / 2, LAT_MAX, RES)   # 100
LON = np.arange(LON_MIN + RES / 2, LON_MAX, RES)   # 180
GRID_SHAPE = (len(LAT), len(LON))                  # (100, 180)
MODEL_SHAPE = (96, 176)                            # divisible by 16 for U-Net; centre crop of GRID_SHAPE

CHANNELS = ["sst", "sss", "sla", "cur_u", "cur_v", "wind_u", "wind_v"]
DEPTHS = [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]
REPORT_DEPTHS = [0, 50, 100, 200, 500, 1000]  # depth-wise metrics table

# Optional extra input channels (docs/10 tasks 5 and 6). Off by default: every committed
# result up to doc 09 was measured with the seven surface channels alone.
CLIM_CHANNELS = [f"clim_{d}m" for d in DEPTHS]                    # 15
AUX_CHANNELS = ["doy_sin", "doy_cos", "lat", "lon", "bathy"]      # 5


def n_channels(extra=()):
    """Input channel count for a channel set.

    The ORDER is frozen: surface, then climatology, then auxiliary. A checkpoint stores its
    `extra` list, so anything that rebuilds a dataset for inference must pass the same one.
    docs/09 sec.7 records what happens when predict_cube guesses instead: M4 was handed
    [B, C, H, W] where it wanted [B, T, C, H, W].
    """
    n = len(CHANNELS)
    if "clim" in extra:
        n += len(CLIM_CHANNELS)
    if "aux" in extra:
        n += len(AUX_CHANNELS)
    return n


def bathy_path(zarr_path):
    """Cache beside the store it was fitted from -- same rule as the climatology cache."""
    return Path(zarr_path).with_suffix(".bathy.npy")

# Start is a hard limit: SMAP SSS begins 2015-03-27 and nothing earlier exists.
# End was originally 2022 because SMAP SSS *V4* stops 2022-07-11 -- but V4 is simply a
# retired version. V6 runs to the present, OSCAR to 2026-01, GLORYS12V1 to 2026-06, so
# 2024 costs nothing but download time and buys two more training years.
START, END = "2015-04-01", "2024-12-31"
SPLITS = {"train": ("2015-04-01", "2021-12-31"),
          "val":   ("2022-01-01", "2022-12-31"),
          "test":  ("2023-01-01", "2024-12-31")}

# Physical range QC (docs/04 §4)
# sss floor is 5, not the 25 first written down: the Ganga-Brahmaputra plume really does
# push northern Bay of Bengal salinity into the teens, and that freshwater cap is exactly
# the barrier-layer signal SSS is an input for (docs/04 sec.5). Observed 15.6 PSU on
# 2018-05-28. A 25 floor would have masked the signal away as bad data.
QC_RANGE = {"sst": (-2.0, 36.0), "sss": (5.0, 41.0), "sla": (-2.0, 2.0),
            "cur_u": (-3.0, 3.0), "cur_v": (-3.0, 3.0),
            "wind_u": (-40.0, 40.0), "wind_v": (-40.0, 40.0), "thetao": (-2.0, 36.0)}


def crop_to_model(da):
    """Centre-crop (100,180) -> (96,176) on the last two dims."""
    dy = (GRID_SHAPE[0] - MODEL_SHAPE[0]) // 2   # 2
    dx = (GRID_SHAPE[1] - MODEL_SHAPE[1]) // 2   # 2
    return da[..., dy:dy + MODEL_SHAPE[0], dx:dx + MODEL_SHAPE[1]]
