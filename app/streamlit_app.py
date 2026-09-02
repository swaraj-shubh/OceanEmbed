"""OceanEmbed — subsurface ocean temperature from satellite surface fields.

    streamlit run app/streamlit_app.py

Runs fully offline from app/demo_data/ (see scripts/build_demo_bundle.py). No torch, no
GPU, no network: predictions are precomputed, so a click is an array lookup.

Colour follows docs' dataviz rules: a single warm hue for temperature (magnitude), a
blue-grey-red diverging scale centred on zero for differences (polarity), and a fixed
categorical order for model comparisons -- validated for colour-vision deficiency rather
than chosen by eye.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent))
import loader as L

st.set_page_config(page_title="OceanEmbed — subsurface temperature",
                   page_icon="🌊", layout="wide")

# --- palette (validated with the dataviz validator; do not substitute by eye) ----------
# Categorical slots in FIXED order, never cycled. Light / dark pairs.
SERIES = {"light": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"],
          "dark":  ["#3987e5", "#d95926", "#199e70", "#c98500"]}
# Temperature is magnitude -> ONE hue, light to dark. Warm, because the quantity is heat.
SEQ_TEMP = "Oranges"
# A difference is polarity -> two hues either side of a neutral grey midpoint. Never a
# rainbow, and never a hue at the midpoint.
DIVERGING = [[0.0, "#2a78d6"], [0.5, "#f0efec"], [1.0, "#e34948"]]
GRID = "rgba(128,128,128,0.22)"

# Which surface channels carry POLARITY rather than magnitude. A sea level *anomaly* and
# the signed components of a vector are meaningless without a zero: on a light-to-dark ramp
# zero lands at an arbitrary shade and "westward" reads as "less eastward". These get the
# diverging scale, centred on zero. Temperature and salinity are magnitudes and do not.
SIGNED = {"sla", "cur_u", "cur_v", "wind_u", "wind_v"}


def theme():
    return "dark" if st.get_option("theme.base") == "dark" else "light"


def series(i):
    return SERIES[theme()][i % 4]


def base_layout(fig, height=420, **kw):
    fig.update_layout(
        height=height, margin=dict(l=8, r=8, t=34, b=8),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(size=12), hoverlabel=dict(font_size=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0,
                    bgcolor="rgba(0,0,0,0)"),
        **kw)
    fig.update_xaxes(showgrid=True, gridcolor=GRID, zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor=GRID, zeroline=False)
    return fig


def heatmap(da, title, colorscale=SEQ_TEMP, unit="°C", diverging=False, height=430,
            key=None):
    """One field on the model grid. NaN (land, unsupervised cells) renders as a gap --
    the model is never scored there and must never be shown as if it were."""
    z = np.asarray(da.values, float)
    kw = {}
    if diverging:                      # symmetric about zero or the midpoint lies
        m = float(np.nanmax(np.abs(z))) or 1.0
        kw = dict(zmin=-m, zmax=m, zmid=0)
    fig = go.Figure(go.Heatmap(
        z=z, x=np.asarray(da.lon.values, float), y=np.asarray(da.lat.values, float),
        colorscale=colorscale, hoverongaps=False,
        colorbar=dict(title=dict(text=unit, side="right"), thickness=12, outlinewidth=0),
        hovertemplate="%{y:.2f}°N  %{x:.2f}°E<br><b>%{z:.2f} " + unit + "</b><extra></extra>",
        **kw))
    base_layout(fig, height=height, title=dict(text=title, x=0, font=dict(size=13)))
    # Pin both axes to the data. An equal-aspect lock (scaleanchor) is geographically
    # purer, but the region is 45 deg wide and 25 deg tall inside a panel that is roughly
    # square, so plotly satisfies the lock by padding latitude out to -5..42 and the map
    # collapses into a strip. Filling the panel costs a little aspect fidelity; every axis
    # is labelled in degrees and the coastline is still unmistakably India.
    fig.update_yaxes(title=None, range=[float(da.lat.min()), float(da.lat.max())])
    fig.update_xaxes(title=None, range=[float(da.lon.min()), float(da.lon.max())])
    return fig


@st.cache_data(show_spinner=False)
def skill_table():
    """Depth-wise numbers for the frozen model plus the floor and the ceiling."""
    fin = L.metrics("ens_mix6_bc_test_argo.csv").set_index("depth_m")
    out = pd.DataFrame({
        "Depth (m)": fin.index,
        "RMSE (°C)": fin["rmse"].values, "MAE (°C)": fin["mae"].values,
        "Bias (°C)": fin["bias"].values, "Corr": fin["corr"].values,
        "R²": fin["r2"].values if "r2" in fin else np.nan,
    })
    for label, f in (("Climatology RMSE", "M0_climatology_test_argo.csv"),
                     ("GLORYS RMSE", "GLORYS_target_test_argo.csv")):
        try:
            out[label] = L.metrics(f).set_index("depth_m")["rmse"].reindex(fin.index).values
        except Exception:
            pass
    return out


# ======================================================================================
man = L.manifest()
st.title("OceanEmbed — subsurface ocean temperature from space")
st.caption(
    f"Seven satellite surface fields → temperature at 15 depths, 0–1000 m, over the "
    f"Arabian Sea and Bay of Bengal. Showing **{man['window']['start']} to "
    f"{man['window']['end']}** — inside the held-out test period the model never trained on."
)

with st.sidebar:
    st.header("Controls")
    dates = L.dates()
    date = st.select_slider("Date", options=list(dates),
                            value=dates[len(dates) // 2],
                            format_func=lambda d: pd.Timestamp(d).strftime("%d %b %Y"))
    depth = st.select_slider("Depth (m)", options=L.depths(), value=100)
    region = st.radio("Region", list(L.REGIONS), horizontal=False)
    st.divider()
    st.metric("Overall error vs Argo", "0.786 °C", "-11.7% vs single model",
              delta_color="inverse")
    st.caption(
        "Measured against ~6,000 **independent** Argo float profiles — observations the "
        "model never saw, in years it never trained on. Climatology scores 1.160 °C; the "
        "reanalysis we learn from scores 0.728 °C."
    )
    st.divider()
    st.caption(f"Model: 6-member ensemble + Argo bias correction · build `{man['git_sha']}`")

lon0, lon1 = L.REGIONS[region]
sub = lambda da: da.sel(lon=slice(lon0, lon1))

t_inputs, t_map, t_profile, t_skill = st.tabs(
    ["① Surface inputs", "② Reconstruction", "③ Profile", "④ Skill"])

# --- ① the seven things a satellite can see -------------------------------------------
with t_inputs:
    st.subheader("What the satellite sees")
    st.caption("These seven surface fields are the model's only input. Everything in the "
               "next tabs is inferred from them.")
    x = L.inputs().sel(time=date)
    cols = st.columns(2)
    for i, ch in enumerate(man["channels"]):
        label, unit = L.CHANNEL_LABEL[ch]
        signed = ch in SIGNED
        with cols[i % 2]:
            st.plotly_chart(
                heatmap(sub(x[ch]), f"{label}  ({unit})",
                        colorscale=(DIVERGING if signed else
                                    SEQ_TEMP if ch == "sst" else "Blues"),
                        diverging=signed, unit=unit, height=300),
                use_container_width=True, key=f"in_{ch}")

# --- ② the reconstruction -------------------------------------------------------------
with t_map:
    c1, c2 = st.columns([3, 1])
    with c2:
        view = st.radio("Show", ["Our reconstruction", "GLORYS reanalysis",
                                 "Difference (ours − GLORYS)"])
        st.caption(
            "GLORYS is the reanalysis the model was **trained on**, not ground truth: it "
            "runs about **+0.72 °C too warm at 100 m** against Argo floats in this basin. "
            "Measuring that is what let us correct it."
        )
    src = {"Our reconstruction": "prediction", "GLORYS reanalysis": "truth",
           "Difference (ours − GLORYS)": "error"}[view]
    da = sub(L.field(date, depth, src))
    with c1:
        st.plotly_chart(
            heatmap(da, f"{view} — {depth} m, {pd.Timestamp(date):%d %b %Y}",
                    colorscale=DIVERGING if src == "error" else SEQ_TEMP,
                    diverging=(src == "error"), height=560),
            use_container_width=True, key="recon")

# --- ③ click a point, get the column --------------------------------------------------
with t_profile:
    st.caption("**Click anywhere on the map** to pull the full 0–1000 m column at that "
               "point, with the nearest independent Argo float profile overlaid.")
    c1, c2 = st.columns([1, 1])
    with c1:
        ev = st.plotly_chart(
            heatmap(sub(L.field(date, depth, "prediction")),
                    f"Reconstruction — {depth} m (click to sample)", height=470),
            use_container_width=True, on_select="rerun", selection_mode="points",
            key="clickmap")
        pts = (ev.get("selection", {}) or {}).get("points", []) if ev else []
        if pts:
            st.session_state["pick"] = (float(pts[0]["y"]), float(pts[0]["x"]))

    lat_s, lon_s = st.session_state.get("pick", (15.0, 88.0))
    with c2:
        zz, pred = L.profile(date, lat_s, lon_s)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=pred, y=zz, name="OceanEmbed", mode="lines+markers",
            line=dict(color=series(0), width=2), marker=dict(size=8),
            hovertemplate="%{y:.0f} m<br><b>%{x:.2f} °C</b><extra>OceanEmbed</extra>"))

        cmp = L.argo_comparison(date, lat_s, lon_s)
        if cmp:
            fig.add_trace(go.Scatter(
                x=cmp["temp"], y=cmp["pres"], name="Argo float (independent)",
                mode="lines", line=dict(color=series(1), width=2),
                hovertemplate="%{y:.0f} m<br><b>%{x:.2f} °C</b><extra>Argo</extra>"))
        fig.update_yaxes(autorange="reversed", title="Depth (m)")
        fig.update_xaxes(title="Temperature (°C)")
        # No chart title: the legend sits along the top and the two would collide. The
        # caption below carries the coordinates instead.
        base_layout(fig, height=470, hovermode="y unified")
        st.markdown(f"**Column at {lat_s:.2f}°N, {lon_s:.2f}°E**")
        st.plotly_chart(fig, use_container_width=True, key="prof")

    if cmp:
        st.success(
            f"Matched Argo float **{cmp['profile']}** — {cmp['distance_deg']:.2f}° away "
            f"({cmp['distance_deg'] * 111:.0f} km), {pd.Timestamp(cmp['time']):%d %b %Y}."
        )
        m = st.columns(4)
        m[0].metric("RMSE here", f"{cmp['rmse']:.3f} °C")
        m[1].metric("Bias here", f"{cmp['bias']:+.3f} °C")
        m[2].metric("Correlation", f"{cmp['corr']:.3f}")
        m[3].metric("Levels compared", cmp["n_levels"])
        st.caption("One profile is a noisy sample — these numbers will bounce around as "
                   "you click. The headline 0.786 °C is over ~6,000 of them.")
    else:
        st.info("No Argo float within 1.5° and 3 days of this point. Try another date or "
                "click elsewhere — coverage is sparse, which is the entire reason this "
                "project exists.")

# --- ④ does it actually work ----------------------------------------------------------
with t_skill:
    st.subheader("Accuracy against independent Argo floats")
    k = st.columns(4)
    k[0].metric("OceanEmbed", "0.786 °C")
    k[1].metric("Climatology baseline", "1.160 °C", "-32% error", delta_color="inverse")
    k[2].metric("GLORYS reanalysis", "0.728 °C", help="The product the model learns from.")
    k[3].metric("Depths beating baseline", "15 / 15")

    tab = skill_table()
    fig = go.Figure()
    for i, (col, name) in enumerate([("Climatology RMSE", "Climatology"),
                                     ("RMSE (°C)", "OceanEmbed"),
                                     ("GLORYS RMSE", "GLORYS reanalysis")]):
        if col in tab:
            fig.add_trace(go.Scatter(
                x=tab[col], y=tab["Depth (m)"], name=name, mode="lines+markers",
                line=dict(color=series([2, 0, 3][i]), width=2), marker=dict(size=8),
                hovertemplate="%{y:.0f} m<br><b>%{x:.3f} °C</b><extra>" + name + "</extra>"))
    fig.update_yaxes(autorange="reversed", title="Depth (m)")
    fig.update_xaxes(title="RMSE (°C) — lower is better")
    # Title lives in the markdown above, not in the figure: a top-anchored horizontal
    # legend and a top-left title occupy the same strip and overlap.
    base_layout(fig, height=460, hovermode="y unified")
    st.markdown("**Error against Argo, by depth**")
    st.plotly_chart(fig, use_container_width=True, key="skill")
    st.caption("OceanEmbed sits below climatology at **every** depth, and below GLORYS "
               "itself at 125, 700 and 1000 m — where the reanalysis' own bias dominates "
               "its error.")

    st.dataframe(tab, use_container_width=True, hide_index=True,
                 column_config={c: st.column_config.NumberColumn(format="%.3f")
                                for c in tab.columns if c != "Depth (m)"})
    st.caption(f"Test split, {man['argo_profiles']} Argo casts inside this window "
               f"(~6,000 over the full 2023–24 test period).")
