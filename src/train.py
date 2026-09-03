"""Config-driven training entrypoint. One YAML per experiment (CLAUDE.md rule 7).

    python src/train.py configs/m2_unet.yaml

Resumes from the latest checkpoint automatically -- required for Spot/preemptible GPUs
(rule 6). Per-epoch depth-wise val metrics are appended to results/<run>.csv.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parent))
from config import ROOT, ZARR, n_channels
from datasets import NIODataset, STATS_PATH
from metrics import DepthStats, summary
from models.unet import OceanEmbed, OceanEmbedTemporal, UNet, masked_mse

CKPT = ROOT / "checkpoints"
RESULTS = ROOT / "results"
MODELS = {"unet": UNet, "oceanembed": OceanEmbed, "temporal": OceanEmbedTemporal}


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

    extra = tuple(cfg.get("extra", ()))
    # Derived, never hand-typed: a YAML saying in_ch: 7 next to extra: [clim] is a silent
    # shape bug waiting to happen. The checkpoint carries cfg, so inference reads it back.
    cfg["model"]["in_ch"] = n_channels(extra)

    dw = cfg.get("depth_weight")
    if dw in ("inv_var", "inv_std"):
        # From the frozen TRAIN stats, normalised to mean 1 so the printed loss stays on the
        # same scale as every other run and remains comparable across the ablation table.
        sd = np.asarray(json.loads(Path(stats or STATS_PATH).read_text())["Y"]["std"])
        w = 1.0 / (sd ** 2 if dw == "inv_var" else sd)
        depth_weight = torch.tensor(w / w.mean(), dtype=torch.float32, device=dev)
        print(f"depth_weight={dw}: {np.round(w / w.mean(), 2).tolist()}")
    else:
        assert dw is None, f"unknown depth_weight: {dw}"
        depth_weight = None

    kw = {"window": cfg.get("window", 1), "anomaly": cfg.get("anomaly", False),
          "extra": extra, **({} if stats is None else {"stats": stats})}
    # Measured on a T4: 142 ms/step of compute against 286 ms/batch of Zarr reads, so the
    # GPU idles unless loading runs in worker processes. The store is chunked time=1, which
    # is right for random access but means one small read per sample.
    nw = cfg.get("num_workers", 4)
    tr = DataLoader(NIODataset("train", zarr, **kw), batch_size=cfg.get("batch_size", 8),
                    shuffle=True, drop_last=True, num_workers=nw, persistent_workers=nw > 0)
    va = DataLoader(NIODataset("val", zarr, **kw), batch_size=cfg.get("batch_size", 8),
                    num_workers=nw, persistent_workers=nw > 0)

    net = build(cfg).to(dev)
    if cfg.get("init_head_bias"):
        # Targets are raw degC (basin mean 15-25), and the 1x1 head starts at ~0, so the
        # first five epochs are spent learning a constant: m4_convlstm_s1 opens at train
        # loss 434 degC^2, i.e. RMSE 20.8, which is exactly "predict zero". Seeding the
        # bias with the train-split per-depth mean hands the model that constant for free.
        ym = json.loads(Path(stats or STATS_PATH).read_text())["Y"]["mean"]
        with torch.no_grad():
            net.head.bias.copy_(torch.tensor(ym, dtype=torch.float32))
        print(f"head bias initialised to train means {[round(v, 1) for v in ym]}")
    opt = torch.optim.Adam(net.parameters(), cfg.get("lr", 1e-3))
    # Constant 1e-3 leaves val RMSE bouncing +/-0.02 at the last epoch, so "best" is a
    # draw from that noise rather than a converged number. Decay to ~0 instead.
    sched = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.get("epochs", 30))
             if cfg.get("sched") == "cosine" else None)
    CKPT.mkdir(exist_ok=True); RESULTS.mkdir(exist_ok=True)
    ckpt_path, start = CKPT / f"{run}.pt", 0
    if ckpt_path.exists():                                   # auto-resume after preemption
        st = torch.load(ckpt_path, map_location=dev, weights_only=False)
        net.load_state_dict(st["model"]); opt.load_state_dict(st["opt"])
        if sched is not None and st.get("sched"):
            sched.load_state_dict(st["sched"])       # resume must not restart the decay
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
            loss = masked_mse(net(x.to(dev)) + b.to(dev), y.to(dev), m.to(dev),
                              grad_weight=cfg.get("grad_weight", 0.0),
                              depth_weight=depth_weight)
            loss.backward(); opt.step()
            losses.append(float(loss.detach()))
        if sched is not None:
            sched.step()
        df = run_val(net, va, dev)
        vr = summary(df)
        print(f"ep {ep:3d}  train {np.mean(losses):.4f}  val RMSE {vr:.4f} degC "
              f"({time.time() - t0:.0f}s){'  *best' if vr < best else ''}")
        with log.open("a") as f:
            f.write(f"{ep},{np.mean(losses):.5f},{vr:.5f},{time.time() - t0:.0f}\n")
        state = {"model": net.state_dict(), "opt": opt.state_dict(), "epoch": ep,
                 "cfg": cfg, "stats": str(stats or ""), "best": best,
                 "sched": sched.state_dict() if sched is not None else None}
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
