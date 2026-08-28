"""Raw Argo temperature profiles for the region (validation track B2). No account needed.

    python src/download/argo.py [--start 2021-01-01] [--end 2022-12-31]

Queries the Ifremer ERDDAP tabledap endpoint directly -- argopy pins an aiohttp that has
no wheel for this Python and wants a C compiler, and all we need is one CSV GET per month.
Writes data/interim/argo_nio.parquet with the columns src/argo_eval.py expects:
profile, time, lat, lon, pres, temp. Argo is never a model input or target (rule 3).
"""
import argparse
import io
import ssl
import sys
import urllib.error
import urllib.request
from pathlib import Path

import certifi
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import INTERIM, LAT_MAX, LAT_MIN, LON_MAX, LON_MIN

URL = "https://erddap.ifremer.fr/erddap/tabledap/ArgoFloats.csv?"
COLS = "platform_number,cycle_number,time,latitude,longitude,pres,temp,temp_qc"
GOOD_QC = (1, 2)          # 1 = good, 2 = probably good; 3/4 are bad or probably bad
OUT = INTERIM / "argo_nio.parquet"
CTX = ssl.create_default_context(cafile=certifi.where())


def query(t0, t1, dmax=1100.0, timeout=600):
    """One ERDDAP CSV GET. Returns an empty frame when the window holds no profiles."""
    q = (f"{COLS}"
         f"&latitude>={LAT_MIN}&latitude<={LAT_MAX}"
         f"&longitude>={LON_MIN}&longitude<={LON_MAX}"
         f"&time>={t0:%Y-%m-%d}T00:00:00Z&time<={t1:%Y-%m-%d}T00:00:00Z"
         f"&pres<={dmax}").replace(">", "%3E").replace("<", "%3C")  # Tomcat rejects raw < >
    try:
        raw = urllib.request.urlopen(URL + q, context=CTX, timeout=timeout).read().decode()
    except urllib.error.HTTPError as e:
        if e.code == 404:                       # ERDDAP's "no matching results"
            return pd.DataFrame(columns=COLS.split(","))
        raise
    return pd.read_csv(io.StringIO(raw), skiprows=[1])  # row 2 is the units header


def tidy(df):
    df = df[df.temp_qc.isin(GOOD_QC)]
    out = pd.DataFrame({
        "profile": df.platform_number.astype(str) + "_" + df.cycle_number.astype(str),
        "time": pd.to_datetime(df.time), "lat": df.latitude, "lon": df.longitude,
        "pres": df.pres, "temp": df.temp})
    return out.dropna(subset=["pres", "temp"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--start", default="2021-01-01")
    p.add_argument("--end", default="2022-12-31")
    p.add_argument("--out", default=OUT)
    a = p.parse_args()
    out = Path(a.out)
    if out.exists():
        print(f"{out} exists -- delete it to refetch")
        return
    edges = pd.date_range(a.start, a.end, freq="MS").union(
        pd.DatetimeIndex([a.start, a.end]))      # month at a time: one GET each, ~1-2 min
    parts = []
    for t0, t1 in zip(edges[:-1], edges[1:]):
        df = tidy(query(t0, t1))
        parts.append(df)
        print(f"{t0:%Y-%m}: {df['profile'].nunique():4d} profiles, {len(df):6d} good levels")
    df = pd.concat(parts).drop_duplicates(["profile", "pres"]).reset_index(drop=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out)
    print(f"TOTAL {df['profile'].nunique()} profiles, {len(df)} levels -> {out}")


if __name__ == "__main__":
    main()
