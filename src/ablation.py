"""One ablation table from every result CSV in results/. Failures included on purpose.

    python src/ablation.py [--split test] [--out results/ablation_test.md]

The table is the core evidence of the whole project (CLAUDE.md sec.12): same held-out Argo,
same depths, every stage that was tried -- including the interventions that did not work,
because those are what make the ceiling finding in docs/09 sec.4 credible.

Seeds of one config are averaged and their spread reported; a run with one seed shows no
spread, which is itself information (docs/09 sec.2: a single seed is unfalsifiable).
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
from config import REPORT_DEPTHS, ROOT
from metrics import summary

RESULTS = ROOT / "results"
SEED = re.compile(r"_s\d+_")


def rows(split, results=None):
    """Group every <run>_<split>_argo.csv by run, averaging across seeds."""
    base = Path(results) if results is not None else RESULTS
    got = {}
    for f in sorted(base.glob(f"*_{split}_argo.csv")):
        name = SEED.sub("_", f.name).replace(f"_{split}_argo.csv", "")
        got.setdefault(name, []).append(pd.read_csv(f))
    table = []
    for name, dfs in got.items():
        blends = [summary(d) for d in dfs]
        mean = pd.concat(dfs).groupby("depth_m").mean(numeric_only=True)
        table.append({
            "run": name,
            "seeds": len(dfs),
            "blended": float(np.mean(blends)),
            "sd": float(np.std(blends, ddof=1)) if len(blends) > 1 else np.nan,
            **{f"rmse_{z}m": float(mean.loc[z, "rmse"]) for z in REPORT_DEPTHS},
            "bias_100m": float(mean.loc[100, "bias"]),
            "corr_100m": float(mean.loc[100, "corr"]),
        })
    return pd.DataFrame(table).sort_values("blended").reset_index(drop=True)


def to_markdown(df):
    """Minimal markdown table. Not worth a `tabulate` dependency for six lines."""
    cols = list(df.columns)
    cells = [[f"{v:.3f}" if isinstance(v, float) and not isinstance(v, bool) else str(v)
              for v in row] for row in df.itertuples(index=False)]
    w = [max(len(c), *(len(r[i]) for r in cells)) if cells else len(c)
         for i, c in enumerate(cols)]
    out = ["| " + " | ".join(c.ljust(w[i]) for i, c in enumerate(cols)) + " |",
           "|" + "|".join("-" * (w[i] + 2) for i in range(len(cols))) + "|"]
    out += ["| " + " | ".join(r[i].ljust(w[i]) for i in range(len(cols))) + " |"
            for r in cells]
    return "\n".join(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="test")
    p.add_argument("--out", default=None)
    a = p.parse_args()
    df = rows(a.split)
    assert len(df) > 0, f"no results/*_{a.split}_argo.csv found"
    df["blended"] = [f"{b:.3f}" + ("" if np.isnan(s) else f" +/- {s:.3f}")
                     for b, s in zip(df["blended"], df["sd"])]
    md = to_markdown(df.drop(columns=["sd"]))
    out = Path(a.out) if a.out else RESULTS / f"ablation_{a.split}.md"
    out.write_text(f"# Ablation vs independent Argo ({a.split} split)\n\n"
                   f"Blended RMSE in degC, sorted best first. Seeds averaged; spread shown "
                   f"where more than one seed exists.\n\n{md}\n")
    print(md)
    print(f"\n-> {out}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main()
        raise SystemExit

    # self-check: two fake runs, one clearly worse, must come out in the right order with
    # the right seed counts and a real spread.
    import tempfile
    from config import DEPTHS
    from metrics import depthwise
    tmp = Path(tempfile.mkdtemp())
    rng = np.random.default_rng(0)
    true = rng.normal(size=(len(DEPTHS), 400))
    for name, err in (("good", 0.4), ("bad", 1.2)):
        for s in (1, 2):
            depthwise(true + err * rng.normal(size=true.shape), true).to_csv(
                tmp / f"{name}_s{s}_val_argo.csv", index=False)
    df = rows("val", results=tmp)
    assert list(df.run) == ["good", "bad"], list(df.run)
    assert (df.seeds == 2).all() and df.sd.notna().all()
    assert df.blended.iloc[0] < df.blended.iloc[1]
    assert abs(df.blended.iloc[0] - 0.4) < 0.05 and abs(df.blended.iloc[1] - 1.2) < 0.1
    md = to_markdown(df.drop(columns=["sd"]))
    assert md.count("\n") == len(df) + 1 and md.startswith("| run")
    import shutil
    shutil.rmtree(tmp, ignore_errors=True)
    print("ablation self-check OK\n" + md.split("\n")[0])
    print(df[["run", "seeds", "blended"]].to_string(index=False))
