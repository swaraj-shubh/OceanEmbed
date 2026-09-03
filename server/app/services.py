"""All numeric logic: model loading, single-date inference, surface/target reads, Argo
matching, metrics, embedding PCA. Reuses src/ (models, dataset, config, argo_eval, metrics)
— never re-implements the physics/normalisation (README §1).

The scientific stack (numpy/torch/xarray + src) is imported behind a guard so the app still
imports where it is absent (dev boxes, CLAUDE.md §13); build_state then reports not-ready and
every data endpoint returns 503 instead of the process failing to boot.
"""
import math
import sys
from pathlib import Path

from app.state import AppState, BadInput, Bundle, NotFound, NotReady

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))   # src modules import each other by bare name

try:
    import numpy as np
    import pandas as pd
    import torch
    import xarray as xr

    from config import (CHANNELS, DEPTHS, GRID_SHAPE, LAT, LAT_MAX, LAT_MIN, LON, LON_MAX,
                        LON_MIN, MODEL_SHAPE, REPORT_DEPTHS, RES, SPLITS, crop_to_model)
    from datasets import NIODataset
    from argo_eval import interp_profile
    from metrics import depthwise, summary
    from models.unet import OceanEmbed, OceanEmbedTemporal, UNet

    # the canonical registry (train.MODELS), rebuilt here to avoid dragging train.py's
    # argparse/yaml/DataLoader into the serving import path.
    MODELS = {"unet": UNet, "oceanembed": OceanEmbed, "temporal": OceanEmbedTemporal}
    HEAVY_OK, HEAVY_ERR = True, None
except Exception as e:                     # noqa: BLE001 — any import failure -> degrade
    HEAVY_OK, HEAVY_ERR = False, repr(e)

# metadata is pure python — safe outside the guard so /meta channel info needs no stack
CHANNEL_META = {
    "sst": ("Sea surface temperature", "degC", "thermal"),
    "sss": ("Sea surface salinity", "PSU", "haline"),
    "sla": ("Sea level anomaly", "m", "balance"),
    "cur_u": ("Zonal current (0-30 m)", "m/s", "balance"),
    "cur_v": ("Meridional current (0-30 m)", "m/s", "balance"),
    "wind_u": ("Zonal wind", "m/s", "balance"),
    "wind_v": ("Meridional wind", "m/s", "balance"),
}


# ------------------------------------------------------------------ helpers
def _model_grid():
    dy = (GRID_SHAPE[0] - MODEL_SHAPE[0]) // 2
    dx = (GRID_SHAPE[1] - MODEL_SHAPE[1]) // 2
    return LAT[dy:dy + MODEL_SHAPE[0]], LON[dx:dx + MODEL_SHAPE[1]]


def _nanlist(a):
    return [float(v) if v is not None and math.isfinite(v) else None for v in a]


def _grid_json(a):
    return [[float(v) if math.isfinite(v) else None for v in row] for row in a]


def _field2d(arr, lat, lon, units, colormap):
    a = np.asarray(arr, float)
    fin = np.isfinite(a)
    vmin = float(np.nanpercentile(a, 2)) if fin.any() else 0.0
    vmax = float(np.nanpercentile(a, 98)) if fin.any() else 1.0
    return {"values": _grid_json(a), "lat": [float(v) for v in lat],
            "lon": [float(v) for v in lon], "units": units,
            "vmin": vmin, "vmax": vmax, "colormap": colormap}


def _ts(state, date):
    if date not in state.date_index:
        raise NotFound(f"date {date} not in {state.errors.get('split','')} range "
                       f"({state.dates[0]}..{state.dates[-1]})" if state.dates
                       else f"date {date} not available")
    return state.date_index[date]


def _bundle(state, model):
    model = model or state.default_model
    if model not in state.models:
        raise NotFound(f"unknown model '{model}'; served: {list(state.models)}")
    return model, state.models[model]


def _need_store(state):
    if state.store is None:
        raise NotReady(f"data store not loaded: {state.errors.get('store', HEAVY_ERR)}")


# ------------------------------------------------------------------ loading (startup)
def build_state(settings):
    st = AppState(device=settings.device, default_model=settings.default_model)
    st.errors["split"] = settings.split
    if not HEAVY_OK:
        st.errors["stack"] = HEAVY_ERR
        return st

    try:
        store = xr.open_zarr(settings.zarr).sel(time=slice(*SPLITS[settings.split]))
        st.store = store
        days = np.asarray(store.time.values, "datetime64[D]")
        st.dates = [str(d) for d in days]
        st.date_index = {str(d): t for d, t in zip(days, store.time.values)}
    except Exception as e:                 # noqa: BLE001
        st.errors["store"] = repr(e)

    for run in settings.served_list():
        try:
            st.models[run] = _load_model(run, settings)
        except Exception as e:             # noqa: BLE001
            st.errors[f"model:{run}"] = repr(e)

    try:
        df = pd.read_parquet(settings.argo)
        df["time"] = pd.to_datetime(df["time"]).dt.tz_localize(None)
        df = df[(df.lat >= LAT_MIN) & (df.lat <= LAT_MAX)
                & (df.lon >= LON_MIN) & (df.lon <= LON_MAX)]
        st.argo = df.reset_index(drop=True)
    except Exception as e:                 # noqa: BLE001
        st.errors["argo"] = repr(e)

    try:
        st.metrics = _load_metrics(settings.results)
    except Exception as e:                 # noqa: BLE001
        st.errors["metrics"] = repr(e)

    return st


def _find_ckpt(ckpt_dir, run):
    d = Path(ckpt_dir)
    for name in (f"{run}_s1_best.pt", f"{run}.pt", f"{run}_best.pt"):
        if (d / name).exists():
            return d / name
    hits = sorted(d.glob(f"{run}*.pt"))
    if hits:
        return hits[0]
    raise FileNotFoundError(f"no checkpoint for '{run}' in {d}")


def _load_model(run, settings):
    ckpt = _find_ckpt(settings.checkpoints, run)
    sd = torch.load(ckpt, map_location=settings.device, weights_only=False)
    cfg = dict(sd["cfg"]["model"])
    kind = cfg.pop("kind")
    net = MODELS[kind](**cfg).to(settings.device).eval()
    net.load_state_dict(sd["model"])
    window = sd["cfg"].get("window", 1)     # must come from ckpt: M4 needs 7 (predict_cube)
    anomaly = sd["cfg"].get("anomaly", False)
    ds = NIODataset(settings.split, str(settings.zarr), window=window,
                    stats=str(settings.stats), anomaly=anomaly)
    dy = (GRID_SHAPE[0] - MODEL_SHAPE[0]) // 2
    dx = (GRID_SHAPE[1] - MODEL_SHAPE[1]) // 2
    land = np.isnan(ds.ds.Y).all("time").compute().values
    land = land[:, dy:dy + MODEL_SHAPE[0], dx:dx + MODEL_SHAPE[1]]
    dates = [str(d) for d in np.asarray(ds.time, "datetime64[D]")]
    return Bundle(net=net, ds=ds, kind=kind, window=window, anomaly=anomaly,
                  land=land, dates=dates)


def _load_metrics(results_dir):
    out = {}
    for f in sorted(Path(results_dir).glob("*_argo.csv")):
        out[f.stem] = pd.read_csv(f)
    return out


# ------------------------------------------------------------------ inference
def _reconstruct(state, run, date):
    """[15,96,176] degC, land = NaN. Whole cube cached per (model, date)."""
    _need_store(state)
    name, b = _bundle(state, run)
    if date in b.cache:
        return name, b, b.cache[date]
    if date not in b.dates:
        raise NotFound(f"model '{name}' cannot predict {date}; earliest is {b.dates[0]} "
                       f"(needs a {b.window}-day input window)")
    i = b.dates.index(date)
    x, _, _, base = b.ds[i]
    xb = torch.from_numpy(np.stack([x])).to(state.device)
    bb = torch.from_numpy(np.stack([base])).to(state.device)
    with torch.no_grad():
        out = (b.net(xb) + bb).cpu().numpy()[0].astype("float32")
    out[b.land] = np.nan
    b.cache[date] = out
    return name, b, out


# ------------------------------------------------------------------ endpoint services
def meta(state, settings):
    _need_store(state)
    mlat, mlon = _model_grid()
    models = []
    for run, b in state.models.items():
        key = _resolve_metric(state, run)
        models.append({"key": run, "label": run, "kind": b.kind, "window": b.window,
                       "is_default": run == settings.default_model,
                       "argo_rmse": (round(float(summary(state.metrics[key])), 3)
                                     if key else None),
                       "n_dates": len(b.dates)})
    return {
        "region": {"name": "Arabian Sea + Bay of Bengal",
                   "bbox": {"lat_min": LAT_MIN, "lat_max": LAT_MAX,
                            "lon_min": LON_MIN, "lon_max": LON_MAX}},
        "grid": {"model_shape": list(MODEL_SHAPE), "res_deg": RES,
                 "lat": [float(v) for v in mlat], "lon": [float(v) for v in mlon]},
        "dates": state.dates,
        "depths_m": DEPTHS,
        "report_depths_m": REPORT_DEPTHS,
        "channels": [{"key": c, "long_name": CHANNEL_META[c][0],
                      "units": CHANNEL_META[c][1], "colormap": CHANNEL_META[c][2]}
                     for c in CHANNELS],
        "models": models,
    }


def surface(state, date, channel, grid="model"):
    _need_store(state)
    if channel not in CHANNELS:
        raise BadInput(f"unknown channel '{channel}'; valid: {CHANNELS}")
    ts = _ts(state, date)
    arr = state.store.X.sel(time=ts, channel=channel).values
    if grid == "model":
        arr = crop_to_model(arr)
        lat, lon = _model_grid()
    else:
        lat, lon = LAT, LON
    name, units, cmap = CHANNEL_META[channel]
    f = _field2d(arr, lat, lon, units, cmap)
    f.update(channel=channel, long_name=name)
    return f


def surface_all(state, date, grid="model"):
    _need_store(state)          # before touching CHANNELS: undefined if the stack is absent
    return {c: surface(state, date, c, grid) for c in CHANNELS}


def target(state, date, depth, grid="model"):
    _need_store(state)
    if depth not in DEPTHS:
        raise BadInput(f"depth {depth} not in {DEPTHS}")
    ts = _ts(state, date)
    arr = state.store.Y.sel(time=ts, depth=depth).values
    if grid == "model":
        arr = crop_to_model(arr)
        lat, lon = _model_grid()
    else:
        lat, lon = LAT, LON
    f = _field2d(arr, lat, lon, "degC", "thermal")
    f.update(date=date, depth_m=depth)
    return f


def reconstruction(state, date, depth, model=None):
    _need_store(state)          # before touching DEPTHS: it's undefined if the stack is absent
    if depth not in DEPTHS:
        raise BadInput(f"depth {depth} not in {DEPTHS}")
    name, b, cube = _reconstruct(state, model, date)
    mlat, mlon = _model_grid()
    f = _field2d(cube[DEPTHS.index(depth)], mlat, mlon, "degC", "thermal")
    f.update(date=date, depth_m=depth, model=name)
    return f


def profile(state, date, lat, lon, model=None):
    name, b, cube = _reconstruct(state, model, date)
    mlat, mlon = _model_grid()
    i = int(np.argmin(np.abs(mlat - lat)))
    j = int(np.argmin(np.abs(mlon - lon)))
    pred = cube[:, i, j]
    tgt = crop_to_model(state.store.Y.sel(time=_ts(state, date)).values)[:, i, j]
    return {
        "cell": {"lat": float(mlat[i]), "lon": float(mlon[j])},
        "date": date, "model": name, "depths_m": DEPTHS,
        "predicted": _nanlist(pred), "target": _nanlist(tgt),
        "argo": _match_argo(state, date, float(mlat[i]), float(mlon[j]), pred),
    }


# ------------------------------------------------------------------ argo
def _days_off(times, day):
    return np.abs((times.astype("datetime64[D]") - day) / np.timedelta64(1, "D"))


def _match_argo(state, date, clat, clon, pred, max_days=3, max_deg=1.0):
    """Nearest held-out Argo cast to the clicked cell; overlay + point metrics.
    Argo is scored against, never fed in (CLAUDE.md rule 3)."""
    if state.argo is None or len(state.argo) == 0:
        return None
    df = state.argo
    day = np.datetime64(date)
    near = df[_days_off(df.time.values, day) <= max_days]
    if len(near) == 0:
        return None
    dlat = near.lat.values - clat
    dlon = (near.lon.values - clon) * math.cos(math.radians(clat))
    d2 = dlat ** 2 + dlon ** 2
    k = int(np.argmin(d2))
    if d2[k] > max_deg ** 2:
        return None
    pid = near.profile.iloc[k]
    g = near[near.profile == pid]
    obs = interp_profile(g.pres.values, g.temp.values)
    tab = depthwise(pred[:, None], obs[:, None])
    pm = [{"depth_m": int(r.depth_m),
           "rmse": _f(r.rmse), "mae": _f(r.mae), "bias": _f(r.bias), "corr": _f(r.corr)}
          for r in tab.itertuples() if r.n > 0]
    return {
        "profile_id": str(pid),
        "lat": float(g.lat.iloc[0]), "lon": float(g.lon.iloc[0]),
        "distance_km": float(math.sqrt(d2[k]) * 111.0),
        "days_off": int(_days_off(np.array([g.time.iloc[0]], "datetime64[ns]"), day)[0]),
        "obs_on_depths": _nanlist(obs),
        "point_metrics": pm,
    }


def argo_nearby(state, date, lat, lon, radius_deg=1.5, max_days=3):
    if state.argo is None:      # not loaded -> 503; loaded-but-none-near -> 200 empty below
        raise NotReady(f"argo not loaded: {state.errors.get('argo', HEAVY_ERR)}")
    if len(state.argo) == 0:
        return {"date": date, "count": 0, "profiles": []}
    df = state.argo
    day = np.datetime64(date)
    near = df[_days_off(df.time.values, day) <= max_days]
    # one row per profile (first level carries the location)
    first = near.groupby("profile", as_index=False).first()
    dlat = first.lat.values - lat
    dlon = (first.lon.values - lon) * math.cos(math.radians(lat))
    dist = np.sqrt(dlat ** 2 + dlon ** 2)
    keep = dist <= radius_deg
    out = [{"profile_id": str(p), "lat": float(la), "lon": float(lo),
            "time": str(t)[:10], "distance_km": float(dd * 111.0)}
           for p, la, lo, t, dd in zip(first.profile[keep], first.lat[keep],
                                       first.lon[keep], first.time[keep], dist[keep])]
    out.sort(key=lambda r: r["distance_km"])
    return {"date": date, "count": len(out), "profiles": out}


# ------------------------------------------------------------------ metrics
def _f(v):
    return float(v) if v is not None and math.isfinite(v) else None


def _resolve_metric(state, key):
    if key in state.metrics:
        return key
    cands = [s for s in state.metrics if key in s]
    cands.sort(key=lambda s: ("test" not in s, len(s)))
    return cands[0] if cands else None


def metrics(state, model):
    if not state.metrics:
        raise NotReady(f"no metric tables loaded: {state.errors.get('metrics','')}")
    key = _resolve_metric(state, model)
    if key is None:
        raise NotFound(f"no metric table matching '{model}'; have: {list(state.metrics)}")
    df = state.metrics[key]
    rows = []
    for r in df.to_dict("records"):
        rows.append({k: (_f(v) if k not in ("depth_m", "n") else
                         (int(v) if math.isfinite(v) else None)) for k, v in r.items()})
    return {"model": model, "source": key, "rows": rows}


ABLATION = [("M0 climatology", "M0_climatology"), ("M2 U-Net", "m2_unet"),
            ("M3 attention", "m3_oceanembed"), ("M4 ConvLSTM", "m4_convlstm"),
            ("GLORYS target (ceiling)", "GLORYS_target")]


def ablation(state):
    if not state.metrics:
        raise NotReady(f"no metric tables loaded: {state.errors.get('metrics','')}")
    series = []
    for label, pref in ABLATION:
        key = _resolve_metric(state, pref)
        if key is None:
            continue
        df = state.metrics[key]
        series.append({"label": label, "source": key,
                       "depths_m": [int(d) for d in df.depth_m],
                       "rmse": _nanlist(df.rmse.values)})
    return {"series": series}


# ------------------------------------------------------------------ embedding
def embedding(state, date, model=None):
    name, b, _ = _reconstruct(state, model, date)   # validates date + warms cache
    i = b.dates.index(date)
    x, _, _, _ = b.ds[i]
    xb = torch.from_numpy(np.stack([x])).to(state.device)
    with torch.no_grad():
        latent, _ = b.net.embed(xb)                 # [1, C, h, w]
    z = latent[0].cpu().numpy()
    c, h, w = z.shape
    M = z.reshape(c, -1).T                           # [h*w, C]
    M = M - M.mean(0)
    _, s, vt = np.linalg.svd(M, full_matrices=False)
    comp = M @ vt[:3].T                              # [h*w, 3]
    lo, hi = comp.min(0), comp.max(0)
    rng = np.where(hi > lo, hi - lo, 1.0)
    rgb = ((comp - lo) / rng).reshape(h, w, 3)
    ev = (s[:3] ** 2 / (s ** 2).sum()).tolist()
    return {"date": date, "model": name, "shape": [h, w],
            "rgb": rgb.astype(float).tolist(), "explained_variance": ev}


def readiness(state):
    return {
        "ready": state.ready(),
        "components": {"store": state.store is not None,
                       "models": len(state.models) > 0,
                       "argo": state.argo is not None,
                       "metrics": bool(state.metrics)},
        "models": list(state.models),
        "errors": state.errors,
    }
