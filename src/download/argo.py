"""Raw Argo temperature profiles for the region (validation track B2). No account needed.

    python src/download/argo.py --start 2021-01-01 --end 2022-12-31

Writes data/interim/argo_nio.parquet with the columns src/argo_eval.py expects:
profile, time, lat, lon, pres, temp. Delayed-mode + real-time, QC-good levels only.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import INTERIM, LAT_MAX, LAT_MIN, LON_MAX, LON_MIN

OUT = INTERIM / "argo_nio.parquet"


def fetch(start, end, dmax=1100):
    from argopy import DataFetcher
    f = DataFetcher(mode="standard", src="gdac").region(
        [LON_MIN, LON_MAX, LAT_MIN, LAT_MAX, 0, dmax, start, end])
    df = f.to_dataframe()
    df = df[df["TEMP_QC"].isin([1, 2])] if "TEMP_QC" in df else df
    out = pd.DataFrame({
        "profile": df["PLATFORM_NUMBER"].astype(str) + "_" + df["CYCLE_NUMBER"].astype(str),
        "time": df["TIME"], "lat": df["LATITUDE"], "lon": df["LONGITUDE"],
        "pres": df["PRES"], "temp": df["TEMP"]})
    return out.dropna(subset=["pres", "temp"]).reset_index(drop=True)


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
    df = fetch(a.start, a.end)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out)
    print(f"{df['profile'].nunique()} profiles, {len(df)} levels -> {out}")


if __name__ == "__main__":
    main()
