"""
OceanEmbed — subsurface ocean temperature from satellite surface fields.

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

from PIL import Image
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent))
import loader as L

logo_path = Path(__file__).parent / "logo.png"
if logo_path.exists():
    page_icon = Image.open(logo_path)
else:
    page_icon = "🌊"

st.set_page_config(page_title="OceanEmbed — subsurface temperature",
                   page_icon=page_icon, layout="wide")

# ============================================================
# GLASSMORPHISM - Grayscale glass
# ============================================================
# Monochrome, with #6b7a8a (neutral grey-blue) used sparingly for active states and hover.
# The page is light/dark GREY, never pure white or black -- glass panels need a tone
# behind them or the blur has nothing to work with.
#
# Two accent variables, not one. #6b7a8a is only 3.9:1 as text on the light glass and
# 4.4:1 under white, both short of WCAG 4.5 -- so the spec colour drives borders, glows
# and hover tints, while --accent-text carries a darkened/lightened variant (5.9:1) for
# anything that is actually read.
#
# Dark column lives under prefers-color-scheme -- Streamlit emits no data-theme attribute
# anywhere in its static bundle, so that selector would never have matched.
st.markdown("""
<style>
/* ----- import font ----- */
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Poppins', system-ui, sans-serif;
}

:root {
    --app-bg: linear-gradient(145deg, #000000 0%, #0a0a0a 100%);
    --glass-bg: rgba(20, 20, 20, 0.7);
    --glass-border: transparent;
    --glass-border-hover: rgba(255, 255, 255, 0.08);
    --glass-shadow: 0 8px 32px rgba(0, 0, 0, 0.5);
    --heading-color: #ffffff;
    --body-color: #ffffff;
    --caption-color: #f0f0f0;
    --accent: #f0f0f0;             /* active states */
    --accent-text: #ffffff;        
    --accent-tint: rgba(255, 255, 255, 0.1);
    --control-hover: rgba(255, 255, 255, 0.1);
    --metric-bg: rgba(20, 20, 20, 0.7);
    --metric-text: #ffffff;
    --alert-bg: rgba(20, 20, 20, 0.7);
    --alert-border: rgba(255, 255, 255, 0.08);
    --sidebar-bg: rgba(10, 10, 10, 0.8);
    --sidebar-border: rgba(255, 255, 255, 0.05);
}

/* ----- base ----- */
.stApp {
    background: var(--app-bg);
    background-attachment: fixed;   /* one gradient over the page, not one per scroll */
    color: var(--body-color);
}
p, label, span {
    color: var(--body-color) !important;
}
.block-container { padding-top: 2.4rem; }

h1, h2, h3, h4 {
    color: var(--heading-color) !important;
    font-weight: 700;
    letter-spacing: 0.2px;
}

[data-testid="stCaptionContainer"],
[data-testid="stCaptionContainer"] * {
    color: var(--caption-color) !important;
}

code {
    background: rgba(255, 255, 255, 0.8) !important;
    color: #000000 !important;
    padding: 2px 6px !important;
    border-radius: 4px !important;
    font-weight: 600 !important;
}

/* ----- glass panels ----- */
[data-testid="stMetric"],
[data-testid="stAlert"],
.stPlotlyChart,
[data-testid="stDataFrame"],
[data-testid="stTable"] {
    background: var(--glass-bg) !important;
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    border: 1px solid var(--glass-border) !important;
    border-radius: 24px !important;
    box-shadow: var(--glass-shadow);
    padding: 16px 18px;
    transition: all 0.2s ease;
}
[data-testid="stMetric"]:hover,
[data-testid="stAlert"]:hover,
.stPlotlyChart:hover,
[data-testid="stDataFrame"]:hover,
[data-testid="stTable"]:hover {
    border: 1px solid var(--glass-border-hover) !important;
}

/* right sidebar / bordered containers */
[data-testid="stVerticalBlockBorderWrapper"] {
    background: var(--glass-bg) !important;
    backdrop-filter: blur(14px) !important;
    -webkit-backdrop-filter: blur(14px) !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: 22px !important;
    box-shadow: var(--glass-shadow) !important;
    padding: 16px !important;
    transition: all 0.2s ease;
}
[data-testid="stVerticalBlockBorderWrapper"]:hover {
    border: 1px solid var(--glass-border-hover) !important;
}

[data-testid="stMetric"] {
    background: var(--metric-bg) !important;
    padding: 16px 18px;
}
[data-testid="stMetricValue"] {
    color: var(--metric-text);
    font-weight: 600;
}
[data-testid="stMetricLabel"], [data-testid="stMetricLabel"] * {
    color: var(--body-color);
}
[data-testid="stAlert"] {
    background: var(--alert-bg) !important;
    border-color: var(--alert-border);
    padding: 14px 16px;
}
.stPlotlyChart { padding: 12px !important; overflow: hidden; }
[data-testid="stDataFrame"] { padding: 6px; overflow: hidden; }

/* ----- sidebar glass ----- */
[data-testid="stSidebar"] {
    background: var(--sidebar-bg) !important;
    backdrop-filter: blur(18px);
    -webkit-backdrop-filter: blur(18px);
    border-right: 1px solid var(--sidebar-border);
}
[data-testid="stSidebarHeader"], [data-testid="stHeader"] {
    padding: 0 !important;
    display: none !important;
}
[data-testid="stSidebarUserContent"] {
    padding-top: 0 !important;
    padding-bottom: 0 !important;
}
/* ----- fixed heading ----- */
div[data-testid="stVerticalBlock"] > div:has(#fixed-header) {
    position: sticky;
    top: 12px;
    z-index: 1000;
    background: var(--glass-bg);
    backdrop-filter: blur(18px);
    -webkit-backdrop-filter: blur(18px);
    border: 1px solid var(--glass-border);
    border-radius: 20px;
    padding: 12px 20px 24px 20px;
    box-shadow: var(--glass-shadow);
    margin-bottom: 1rem;
}

[data-testid="stSidebar"] > div:first-child {
    margin: 12px !important;
    padding: 16px !important;
    border-radius: 22px !important;
    background: var(--glass-bg) !important;
    backdrop-filter: blur(14px) !important;
    -webkit-backdrop-filter: blur(14px) !important;
    border: 1px solid var(--glass-border) !important;
    box-shadow: var(--glass-shadow) !important;
    height: calc(100vh - 24px) !important;
    max-height: calc(100vh - 24px) !important;
    overflow-y: auto !important;
}

/* ----- tabs ----- */
.stTabs [data-baseweb="tab-list"] {
    gap: 10px;
    padding: 14px 20px;
    margin-bottom: 18px;
    border-radius: 999px;
    background: var(--glass-bg);
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    border: 1px solid var(--glass-border);
    box-shadow: var(--glass-shadow);
}
.stTabs [data-baseweb="tab"] {
    border-radius: 999px;
    padding: 10px 26px;
    background: transparent;
    color: var(--body-color);
    font-weight: 500;
    /* transparent border matches the selected tab's 1px, so selecting doesn't nudge
       the row by a pixel */
    border: 1px solid transparent;
    transition: all 0.15s ease;
}
.stTabs [data-baseweb="tab"]:hover {
    background: var(--control-hover);
}
/* active: a touch more glass + the accent, sparingly */
.stTabs [aria-selected="true"] {
    background: var(--accent-tint) !important;
    border: 1px solid transparent !important;
    color: var(--accent-text) !important;
    font-weight: 600;
    outline: none !important;
}
.stTabs [data-baseweb="tab-highlight"],
.stTabs [data-baseweb="tab-border"] {
    display: none;
}

/* ----- controls ----- */
[data-baseweb="select"] > div,
.stSlider [data-baseweb="slider"] > div:first-child {
    background: var(--glass-bg) !important;
    backdrop-filter: blur(6px);
    -webkit-backdrop-filter: blur(6px);
    border: 1px solid var(--glass-border) !important;
    border-radius: 999px !important;
    box-shadow: var(--glass-shadow);
}
.stSlider [role="slider"] {
    background: var(--accent);
    border: 1px solid var(--accent);
    box-shadow: var(--glass-shadow);
}

[data-testid="stRadio"] label[data-baseweb="radio"] {
    background: var(--glass-bg);
    backdrop-filter: blur(6px);
    -webkit-backdrop-filter: blur(6px);
    border: 1px solid var(--glass-border);
    border-radius: 999px;
    padding: 8px 16px;
    margin: 4px 4px; /* adjusted for horizontal */
    color: var(--body-color);
    box-shadow: var(--glass-shadow);
    transition: all 0.15s ease;
    cursor: pointer;
}
/* Hide the default radio circle to make it look like a pure pill/button */
[data-testid="stRadio"] [data-baseweb="radio"] > div:first-child {
    display: none !important;
}
[data-testid="stRadio"] label[data-baseweb="radio"]:hover {
    background: var(--control-hover);
}
/* the picked region / view is the only "active" control on the page -- without this the
   accent never shows up outside the tab bar */
[data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked) {
    background: var(--accent-tint);
    border-color: var(--accent);
    color: var(--accent-text);
    font-weight: 600;
}

/* ----- buttons ----- */
.stButton > button,
.stDownloadButton > button {
    background: var(--glass-bg);
    backdrop-filter: blur(8px);
    -webkit-backdrop-filter: blur(8px);
    border: 1px solid var(--glass-border);
    border-radius: 999px;
    padding: 0.55rem 1.8rem;
    color: var(--accent-text);
    font-weight: 600;
    box-shadow: var(--glass-shadow);
    transition: all 0.15s ease;
}
.stButton > button:hover,
.stDownloadButton > button:hover {
    background: var(--control-hover);
    border-color: var(--accent);
    transform: scale(0.98);
}
.stButton > button:active,
.stDownloadButton > button:active {
    transform: scale(0.95);
}

/* ----- divider ----- */
hr {
    border: none;
    height: 1px;
    background: var(--glass-border);
    margin: 1.5rem 0;
}
</style>
""", unsafe_allow_html=True)

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
        template="plotly_dark",
        height=height, margin=dict(l=12, r=12, t=36, b=36),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(size=13, family="Poppins, system-ui, sans-serif", color="#ffffff"),
        hoverlabel=dict(font_size=13, font_family="Poppins, system-ui, sans-serif", font_color="#ffffff"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0,
                    bgcolor="rgba(0,0,0,0)", font=dict(size=14, color="#ffffff")),
        **kw)
    fig.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.1)", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.1)", zeroline=False)
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
    base_layout(fig, height=height, title=dict(text=title, x=0, font=dict(size=14, color="#ffffff")))
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
logo_path = Path(__file__).parent / "logo.png"
if logo_path.exists():
    import base64
    logo_b64 = base64.b64encode(logo_path.read_bytes()).decode()
    st.markdown(
        f"""
        <div id="fixed-header" style="display: flex; align-items: center; gap: 16px;">
            <img src="data:image/png;base64,{logo_b64}" width="48" style="background-color: white; border-radius: 50%; padding: 2px; box-shadow: var(--glass-shadow);" />
            <h1 style="margin: 0; padding: 0; padding-bottom: 4px;">OTER - Ocean Thermal Embedding Reconstruction</h1>
        </div>
        """,
        unsafe_allow_html=True
    )
else:
    st.title("OTER - Ocean Thermal Embedding Reconstruction")
_d0, _d1 = L.dates().min(), L.dates().max()
st.caption(
    f"Seven satellite surface fields → temperature at 15 depths, 0–1000 m, over the "
    f"Arabian Sea and Bay of Bengal. Showing **every day from {_d0:%d %b %Y} to "
    f"{_d1:%d %b %Y}** — the full held-out test period the model never trained on."
)

with st.sidebar:
    st.header("Controls")
    dates = L.dates()
    date = st.select_slider("Date", options=list(dates),
                            value=dates[len(dates) // 2],
                            format_func=lambda d: pd.Timestamp(d).strftime("%d %b %Y"))
    depth = st.select_slider("Depth (m)", options=L.depths(), value=100)
    region = st.radio("Region", list(L.REGIONS), horizontal=True)
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
    ["Surface inputs", "Reconstruction", "Profile", "Skill"])

# --- ① the seven things a satellite can see -------------------------------------------
with t_inputs:
    st.subheader("What the satellite sees")
    st.caption("These seven surface fields are the model's only input. Everything in the "
               "next tabs is inferred from them.")
    x = L.inputs(date)
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
        with st.container(border=True):
            st.markdown("#### Display Mode")
            view = st.radio("Show", ["Our reconstruction", "GLORYS reanalysis",
                                     "Difference (ours − GLORYS)"],
                            label_visibility="collapsed")
            st.divider()
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
        # st.markdown(f"**Column at {lat_s:.2f}°N, {lon_s:.2f}°E**")
        st.plotly_chart(fig, use_container_width=True, key="prof")
        st.markdown(f"**Column at {lat_s:.2f}°N, {lon_s:.2f}°E**")

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
    st.caption(f"Test split, all {man['argo_profiles']:,} independent Argo casts across "
               f"the full 2023–24 test period.")
               