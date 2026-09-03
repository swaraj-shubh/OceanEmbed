"""Build a tiny self-contained artifact set (fake zarr + stats + checkpoints + argo + metric
CSVs) so the API can run and be tested with no real data / no GPU. Reuses src._fake_store,
the same helper the research self-checks use.

    python server/scripts/make_fake_artifacts.py /tmp/oe_fake
    # then export the printed OCEANEMBED_* vars and: uvicorn app.main:app
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from config import DEPTHS                                    # noqa: E402
from datasets import _fake_store, compute_stats             # noqa: E402
from models.unet import OceanEmbedTemporal, UNet            # noqa: E402

SPLIT = "val"           # _fake_store straddles the 2022 val edge, so 'val' has ~20 days
ARGO_DATE = "2022-01-05"


def _metric_csv(path, r2=True):
    n = len(DEPTHS)
    df = pd.DataFrame({"depth_m": DEPTHS, "n": [100] * n,
                       "rmse": np.linspace(0.5, 1.2, n), "mae": np.linspace(0.4, 1.0, n),
                       "bias": np.linspace(-0.2, 0.8, n), "corr": np.linspace(0.98, 0.8, n)})
    if r2:
        df["r2"] = np.linspace(0.95, 0.6, n)
    df.to_csv(path, index=False)


def build(out: Path):
    out = Path(out)
    (out / "ckpts").mkdir(parents=True, exist_ok=True)
    (out / "results").mkdir(parents=True, exist_ok=True)

    zarr = out / "nio.zarr"
    _fake_store(zarr)
    stats = out / "stats.json"
    compute_stats(zarr, stats)

    # two checkpoints exercising both code paths: window=1 (UNet) and window=7 (temporal)
    for run, net, kind, window in (("unet", UNet(), "unet", 1),
                                   ("temporal", OceanEmbedTemporal(), "temporal", 7)):
        torch.save({"cfg": {"model": {"kind": kind}, "window": window, "anomaly": False},
                    "model": net.state_dict()}, out / "ckpts" / f"{run}_s1_best.pt")

    # a couple of Argo casts inside the region on a val date
    z = np.array([0, 10, 30, 50, 75, 100, 150, 200, 300, 500, 700, 1000], float)
    rows = []
    for pid, (la, lo) in enumerate([(10.1, 80.2), (15.0, 88.4)]):
        temp = 29 - 24 * (1 - np.exp(-z / 300))
        rows += [{"profile": f"p{pid}", "time": ARGO_DATE, "lat": la, "lon": lo,
                  "pres": float(zz), "temp": float(tt)} for zz, tt in zip(z, temp)]
    pd.DataFrame(rows).to_parquet(out / "argo.parquet")

    for name in ("unet", "m2_unet", "m3_oceanembed", "m4_convlstm_s1_best",
                 "M0_climatology", "GLORYS_target"):
        _metric_csv(out / "results" / f"{name}_test_argo.csv")

    env = {"OCEANEMBED_ZARR": str(zarr), "OCEANEMBED_STATS": str(stats),
           "OCEANEMBED_CHECKPOINTS": str(out / "ckpts"),
           "OCEANEMBED_ARGO": str(out / "argo.parquet"),
           "OCEANEMBED_RESULTS": str(out / "results"),
           "OCEANEMBED_SERVED_MODELS": "unet,temporal",
           "OCEANEMBED_DEFAULT_MODEL": "unet", "OCEANEMBED_SPLIT": SPLIT}
    (out / "env.json").write_text(json.dumps(env, indent=2))
    return env


if __name__ == "__main__":
    dest = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/oe_fake")
    env = build(dest)
    print(f"built fake artifacts in {dest}\n")
    for k, v in env.items():
        print(f"export {k}={v}")
