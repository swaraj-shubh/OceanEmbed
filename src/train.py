"""Config-driven training entrypoint. One YAML per experiment (CLAUDE.md rule 7).

    python src/train.py configs/m2_unet.yaml

Resumes from the latest checkpoint automatically -- required for Spot/preemptible GPUs
(rule 6). Per-epoch depth-wise val metrics are appended to results/<run>.csv.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parent))
from config import ROOT, ZARR
from datasets import NIODataset
from metrics import DepthStats, summary
from models.unet import OceanEmbed, UNet, masked_mse

CKPT = ROOT / "checkpoints"
RESULTS = ROOT / "results"
MODELS = {"unet": UNet, "oceanembed": OceanEmbed}


def build(cfg):
    m = dict(cfg["model"])
    return MODELS[m.pop("kind")](**m)


def run_val(net, loader, dev):
    net.eval()
    acc = DepthStats()
    with torch.no_grad():
        for x, y, m, b in loader:
            # `b` is the climatology in anomaly mode and zeros otherwise, so the loss and
            # every metric stay in absolute degC and compare directly against M0/M2.
            p = (net(x.to(dev)) + b.to(dev)).cpu().numpy()
            acc.update(p.transpose(1, 0, 2, 3), y.numpy().transpose(1, 0, 2, 3),
                       m.numpy().transpose(1, 0, 2, 3))
    net.train()
    return acc.table()


def main(cfg_path, seed=None):
    cfg = yaml.safe_load(Path(cfg_path).read_text())
    run = cfg.get("run") or Path(cfg_path).stem
    if seed is not None:
        # One config, several seeds. The run name carries the seed so checkpoints and
        # result CSVs never collide -- a fixed-seed rerun of this setup moved val RMSE by
        # ~10%, so single-run comparisons mean nothing and every claim needs a spread.
        cfg["seed"], run = seed, f"{run}_s{seed}"
    zarr = cfg.get("zarr", ZARR)
    stats = cfg.get("stats")
    torch.manual_seed(cfg.get("seed", 0))
    dev = cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")

    kw = {"window": cfg.get("window", 1), "anomaly": cfg.get("anomaly", False),
          **({} if stats is None else {"stats": stats})}
    # Measured on a T4: 142 ms/step of compute against 286 ms/batch of Zarr reads, so the
    # GPU idles unless loading runs in worker processes. The store is chunked time=1, which
    # is right for random access but means one small read per sample.
    nw = cfg.get("num_workers", 4)
    tr = DataLoader(NIODataset("train", zarr, **kw), batch_size=cfg.get("batch_size", 8),
                    shuffle=True, drop_last=True, num_workers=nw, persistent_workers=nw > 0)
    va = DataLoader(NIODataset("val", zarr, **kw), batch_size=cfg.get("batch_size", 8),
                    num_workers=nw, persistent_workers=nw > 0)

    net = build(cfg).to(dev)
    opt = torch.optim.Adam(net.parameters(), cfg.get("lr", 1e-3))
    CKPT.mkdir(exist_ok=True); RESULTS.mkdir(exist_ok=True)
    ckpt_path, start = CKPT / f"{run}.pt", 0
    if ckpt_path.exists():                                   # auto-resume after preemption
        st = torch.load(ckpt_path, map_location=dev, weights_only=False)
        net.load_state_dict(st["model"]); opt.load_state_dict(st["opt"])
        start = st["epoch"] + 1
        print(f"resumed {run} from epoch {start}")

    best = float(st["best"]) if ckpt_path.exists() and "best" in st else float("inf")
    log = RESULTS / f"{run}.csv"
    if not log.exists():
        log.write_text("epoch,train_loss,val_rmse,secs\n")
    for ep in range(start, cfg.get("epochs", 30)):
        t0, losses = time.time(), []
        for x, y, m, b in tr:
            opt.zero_grad()
            loss = masked_mse(net(x.to(dev)) + b.to(dev), y.to(dev), m.to(dev))
            loss.backward(); opt.step()
            losses.append(float(loss.detach()))
        df = run_val(net, va, dev)
        vr = summary(df)
        print(f"ep {ep:3d}  train {np.mean(losses):.4f}  val RMSE {vr:.4f} degC "
              f"({time.time() - t0:.0f}s){'  *best' if vr < best else ''}")
        with log.open("a") as f:
            f.write(f"{ep},{np.mean(losses):.5f},{vr:.5f},{time.time() - t0:.0f}\n")
        state = {"model": net.state_dict(), "opt": opt.state_dict(), "epoch": ep,
                 "cfg": cfg, "stats": str(stats or ""), "best": best}
        torch.save(state, ckpt_path)
        df.to_csv(RESULTS / f"{run}_val_depthwise.csv", index=False)
        # Keep the best-val epoch separately. The last epoch is not the best one: both M2
        # and M3 flattened around epoch 20 while train loss kept falling, so scoring the
        # final weights conflates "does attention help" with "did it overfit".
        if vr < best:
            best = vr
            state["best"] = best
            torch.save(state, CKPT / f"{run}_best.pt")
            df.to_csv(RESULTS / f"{run}_best_val_depthwise.csv", index=False)
    return net


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("config")
    p.add_argument("--seed", type=int, default=None)
    a = p.parse_args()
    main(a.config, a.seed)
