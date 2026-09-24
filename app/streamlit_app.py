"""
OTER (Ocean Thermal Embedding Reconstruction) -- subsurface ocean temperature from satellite surface fields.

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
import plotly.figure_factory as ff
import plotly.graph_objects as go
import streamlit as st
from streamlit import config as st_config

sys.path.append(str(Path(__file__).resolve().parent))
import loader as L

logo_path = Path(__file__).parent / "logo.png"
if logo_path.exists():
    page_icon = Image.open(logo_path)
else:
    page_icon = "🌊"

st.set_page_config(page_title="OTER — subsurface temperature",
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
    --metric-label: #e6e6e6;       /* was the same dim grey as caption text -- too low
                                       contrast for a label sitting right above a big
                                       number; lightened and bolded below */
    --metric-delta: #5be08a;       /* every delta shown here is delta_color="inverse" on
                                       a negative number, i.e. always "good news" -- one
                                       bright colour, not Streamlit's default muted pair
                                       that reads poorly on a near-black card */
    --alert-bg: rgba(20, 20, 20, 0.7);
    --alert-border: rgba(255, 255, 255, 0.08);
    --sidebar-bg: rgba(10, 10, 10, 0.8);
    --table-head: rgba(255, 255, 255, 0.08);
    --table-rule: rgba(255, 255, 255, 0.06);
    --table-hover: rgba(255, 255, 255, 0.04);
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
    color: var(--metric-label) !important;
    font-size: 15px !important;
    font-weight: 500 !important;
    opacity: 1 !important;   /* Streamlit's own label style ships a reduced opacity that
                                 a plain color override doesn't cancel */
}
[data-testid="stMetricDelta"], [data-testid="stMetricDelta"] * {
    color: var(--metric-delta) !important;   /* the arrow SVG is fill="currentColor", so
                                                 this colours it too. No fill: override --
                                                 it also filled the icon's invisible 24x24
                                                 bounding-box path, drawing a solid square */
    font-weight: 600 !important;
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
.st-key-topbar {
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

/* the theme toggle sits in a narrow column; the global pill padding would truncate it */
.st-key-mode_btn button { padding: 0.5rem 0.9rem; white-space: nowrap; }

/* ----- divider ----- */
hr {
    border: none;
    height: 1px;
    background: var(--glass-border);
    margin: 1.5rem 0;
}

/* ----- plain HTML benchmark tables -----
   st.dataframe's grid is canvas-drawn (glide-data-grid): no CSS selector reaches into it,
   which is why header background/alignment can't be set through it -- only cell content
   alignment is exposed, via column_config. A real <table> is CSS-reachable end to end. */
.oter-table { width: 100%; border-collapse: collapse; margin: 6px 0 4px; font-size: 14px; }
.oter-table th {
    background: var(--table-head);
    color: var(--body-color);
    text-align: center;
    padding: 10px 12px;
    font-weight: 600;
    border-bottom: 1px solid var(--glass-border-hover);
}
.oter-table td {
    text-align: center;
    padding: 8px 12px;
    color: var(--body-color);
    border-bottom: 1px solid var(--table-rule);
}
.oter-table tr:last-child td { border-bottom: none; }
.oter-table tbody tr:hover td { background: var(--table-hover); }
</style>
""", unsafe_allow_html=True)

# --- light / dark ----------------------------------------------------------------------
# One switch drives three layers: the CSS variables above (page), Streamlit's own chrome
# (dataframe grid, popovers -- reachable only through theme config) and ui() (charts).
THEMES = {
    "dark": {"base": "dark", "backgroundColor": "#0a0a0a",
             "secondaryBackgroundColor": "#141414", "textColor": "#ffffff"},
    "light": {"base": "light", "backgroundColor": "#e8eaec",
              "secondaryBackgroundColor": "#dfe2e6", "textColor": "#1a1a1a"},
}
st.session_state.setdefault("mode", "dark")
mode = st.session_state["mode"]
if st.get_option("theme.base") != mode:
    # The theme ships to the browser at the start of each run, so set it and rerun once.
    # ponytail: theme config is process-wide -- every open session follows the last toggle.
    # Fine for a one-laptop demo; per-session native theming needs Streamlit support.
    for k, v in THEMES[mode].items():
        st_config.set_option(f"theme.{k}", v)
    st.rerun()

if mode == "light":
    st.markdown("""
<style>
:root {
    --app-bg: linear-gradient(145deg, #eef0f2 0%, #dde1e5 100%);
    --glass-bg: rgba(255, 255, 255, 0.72);
    --glass-border-hover: rgba(0, 0, 0, 0.08);
    --glass-shadow: 0 8px 32px rgba(0, 0, 0, 0.08);
    --heading-color: #111418;
    --body-color: #1f2328;
    --caption-color: #3d444b;
    --accent: #4a5563;
    --accent-text: #111418;
    --accent-tint: rgba(0, 0, 0, 0.06);
    --control-hover: rgba(0, 0, 0, 0.05);
    --metric-bg: rgba(255, 255, 255, 0.72);
    --metric-text: #111418;
    --metric-label: #2b3137;
    --metric-delta: #16803c;
    --table-head: rgba(0, 0, 0, 0.05);
    --table-rule: rgba(0, 0, 0, 0.06);
    --table-hover: rgba(0, 0, 0, 0.03);
    --alert-bg: rgba(255, 255, 255, 0.72);
    --alert-border: rgba(0, 0, 0, 0.08);
    --sidebar-bg: rgba(245, 246, 247, 0.85);
    --sidebar-border: rgba(0, 0, 0, 0.06);
}
code { background: rgba(0, 0, 0, 0.08) !important; color: #111418 !important; }
</style>
""", unsafe_allow_html=True)


def flip_mode():
    st.session_state["mode"] = "light" if st.session_state["mode"] == "dark" else "dark"

# --- palette (validated with the dataviz validator; do not substitute by eye) ----------
# Categorical slots in FIXED order, never cycled. Light / dark pairs.
SERIES = {"light": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"],
          "dark":  ["#3987e5", "#d95926", "#199e70", "#c98500"]}
# Temperature is magnitude -> ONE hue, light to dark. Warm, because the quantity is heat.
SEQ_TEMP = "Oranges"
# A difference is polarity -> two hues either side of a neutral grey midpoint. Never a
# rainbow, and never a hue at the midpoint.
DIVERGING = [[0.0, "#2a78d6"], [0.5, "#f0efec"], [1.0, "#e34948"]]
# Speed is magnitude, but arrows are drawn on top: the ramp stops at mid-blue so dark
# arrows stay readable over the fastest water.
SPEED = [[0.0, "#e3eef8"], [1.0, "#4a90c9"]]
ARROW = "#1a1a1a"

# Which surface channels carry POLARITY rather than magnitude. A sea level *anomaly* and
# the signed components of a vector are meaningless without a zero: on a light-to-dark ramp
# zero lands at an arbitrary shade and "westward" reads as "less eastward". These get the
# diverging scale, centred on zero. Temperature and salinity are magnitudes and do not.
SIGNED = {"sla", "cur_u", "cur_v", "wind_u", "wind_v"}

# Depth axes are log(z + DZ): 0-200 m, where the thermocline and 62% of the error live
# (docs/12 sec.3), gets about two thirds of the axis instead of a linear fifth.
DZ = 10
HOVER = dict(bgcolor="rgba(15,15,15,0.95)", bordercolor="rgba(255,255,255,0.35)",
             font=dict(color="#ffffff", size=14))
DEPTH_TICKS = [0, 10, 20, 50, 100, 200, 500, 1000]


def theme():
    return st.session_state.get("mode", "dark")


def ui():
    """Chart colours for the active theme, so charts and page chrome never disagree."""
    if theme() == "dark":
        return dict(template="plotly_dark", fg="#ffffff", grid="rgba(255,255,255,0.1)",
                    land="#4a4f56", shelf="#2c3036", edge="#000000")
    return dict(template="plotly_white", fg="#1a1a1a", grid="rgba(0,0,0,0.08)",
                land="#c8c8c8", shelf="#e6e6e6", edge="#ffffff")


def series(i):
    return SERIES[theme()][i % 4]


def base_layout(fig, height=420, **kw):
    c = ui()
    fig.update_layout(
        template=c["template"],
        height=height, margin=dict(l=12, r=12, t=36, b=36),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(size=13, family="Poppins, system-ui, sans-serif", color=c["fg"]),
        # Plotly heatmaps default the hover box's background to the hovered cell's OWN
        # colour when bgcolor is left unset -- white text on a light cell (the pale end of
        # "Oranges", or the midpoint of the diverging scale) then reads as white-on-white.
        # A fixed dark background makes every hover legible regardless of what's under it,
        # in either page theme.
        hoverlabel=dict(font_size=14, font_family="Poppins, system-ui, sans-serif",
                        font_color="#ffffff", bgcolor="rgba(15,15,15,0.95)",
                        bordercolor="rgba(255,255,255,0.35)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0,
                    bgcolor="rgba(0,0,0,0)", font=dict(size=14, color=c["fg"])),
        **kw)
    fig.update_xaxes(showgrid=True, gridcolor=c["grid"], zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor=c["grid"], zeroline=False)
    return fig


def depth_axis(fig):
    """Reversed log depth axis; traces must be plotted at y = depth + DZ."""
    fig.update_yaxes(type="log", range=[np.log10(1000 + DZ), np.log10(DZ)],
                     tickvals=[d + DZ for d in DEPTH_TICKS],
                     ticktext=[str(d) for d in DEPTH_TICKS], title="Depth (m)")
    return fig


# --- fixed colour ranges: one per depth / channel across the whole test period, so moving the
# date slider changes colour only where the ocean changed ------------------------------
# Ranges span the whole test period. They are computed once, exactly, when the bundle is
# built (scripts/build_demo_bundle.py::colour_ranges) and read from the manifest here --
# deriving them at app time means decoding every quarter file.
def temp_range(depth):
    """Shared by our map and GLORYS so the two views are directly comparable."""
    return tuple(L.manifest()["colour_ranges"]["temp"][str(depth)])


def diff_range(depth):
    """Symmetric, clipped at the 98th percentile: one bad cell must not grey out the map."""
    m = L.manifest()["colour_ranges"]["diff"][str(depth)]
    return -m, m


def input_range(ch):
    return tuple(L.manifest()["colour_ranges"]["inputs"][ch])


def speed_top(prefix):
    return L.manifest()["colour_ranges"]["speed"][prefix]


@st.cache_resource(show_spinner=False)
def _dry():
    """(depth, lat, lon) True where there is no water, read once: land and seafloor are
    static, so the first day of the first quarter stands for every day."""
    ds = next(L.each_quarter("pred"))
    return ds.thetao.isel(time=0).load().isnull()


def masks(depth):
    """(land, shelf) as lat/lon DataArrays. Land is dry at the surface; shelf is ocean whose
    floor is shallower than `depth`, so there is no water there to predict."""
    dry = _dry()
    land = dry.sel(depth=0)
    return land, dry.sel(depth=depth) & ~land


@st.cache_data(show_spinner=False)
def floats_near(date, days=3):
    """One row per Argo cast within +/-days of `date` -- the same window argo_comparison
    matches in, so every marker on the map is one a click can actually compare against."""
    a = L.argo()
    a = a[(a.time - pd.Timestamp(date)).abs() <= pd.Timedelta(days=days)]
    return a.groupby("profile")[["lat", "lon"]].first()


def ground(da, depth):
    """Grey land and a lighter 'seafloor above this depth' layer, drawn under the data so
    a gap is never mistaken for a missing value."""
    c = ui()
    land, shelf = (m.sel(lat=da.lat, lon=da.lon, method="nearest").values
                   for m in masks(depth))
    g = np.where(land, 0.0, np.where(shelf, 1.0, np.nan))
    text = np.where(land, "Land", np.where(shelf, f"Seafloor shallower than {depth} m", ""))
    return go.Heatmap(z=g, x=np.asarray(da.lon.values, float),
                      y=np.asarray(da.lat.values, float), zmin=0, zmax=1,
                      colorscale=[[0, c["land"]], [1, c["shelf"]]], showscale=False,
                      text=text, hovertemplate="%{text}<extra></extra>", hoverongaps=False,
                      hoverlabel=HOVER)


def isotherms(z, lon, lat, zrange):
    """Contour lines at a step giving ~6-12 lines over the depth's fixed range, so levels
    stay put as the date changes."""
    lo, hi = zrange
    step = next((s for s in (0.25, 0.5, 1, 2) if (hi - lo) / s <= 12), 5)
    return go.Contour(
        z=z, x=lon, y=lat, showscale=False, hoverinfo="skip",
        contours=dict(coloring="none", start=np.ceil(lo / step) * step, end=hi, size=step,
                      showlabels=True, labelfont=dict(size=10, color="rgba(0,0,0,0.75)")),
        line=dict(width=0.8, color="rgba(0,0,0,0.45)"))


def heatmap(da, title, colorscale=SEQ_TEMP, unit="°C", zrange=None, depth=0,
            contours=False, height=430):
    """One field on the model grid, over grey land. NaN over ocean renders as a gap --
    the model is never scored there and must never be shown as if it were."""
    z = np.asarray(da.values, float)
    lon, lat = np.asarray(da.lon.values, float), np.asarray(da.lat.values, float)
    kw = {} if zrange is None else dict(zmin=zrange[0], zmax=zrange[1])
    fig = go.Figure([ground(da, depth), go.Heatmap(
        z=z, x=lon, y=lat, colorscale=colorscale, hoverongaps=False,
        colorbar=dict(title=dict(text=unit, side="right"), thickness=12, outlinewidth=0),
        hovertemplate="%{y:.2f}°N  %{x:.2f}°E<br><b>%{z:.2f} " + unit + "</b><extra></extra>",
        # Set again at the trace level, not just in base_layout: heatmap hover boxes can
        # ignore the layout-level default and fall back to colouring themselves from the
        # cell underneath, so the fix has to hold here too.
        hoverlabel=HOVER, **kw)])
    if contours and zrange is not None:
        fig.add_trace(isotherms(z, lon, lat, zrange))
    base_layout(fig, height=height, title=dict(text=title, x=0, font=dict(size=14)))
    # Pin both axes to the data. An equal-aspect lock (scaleanchor) is geographically
    # purer, but the region is 45 deg wide and 25 deg tall inside a panel that is roughly
    # square, so plotly satisfies the lock by padding latitude out to -5..42 and the map
    # collapses into a strip. Filling the panel costs a little aspect fidelity; every axis
    # is labelled in degrees and the coastline is still unmistakably India.
    fig.update_yaxes(title=None, range=[float(lat.min()), float(lat.max())])
    fig.update_xaxes(title=None, range=[float(lon.min()), float(lon.max())])
    return fig


def vector_map(x, prefix, label, height=300):
    """Speed as colour, direction as arrows: one readable panel instead of two signed U/V
    maps a judge has to combine in their head."""
    u, v = sub(x[f"{prefix}_u"]), sub(x[f"{prefix}_v"])
    top = speed_top(prefix)
    fig = heatmap(np.hypot(u, v), f"{label} — speed and direction (m/s)", colorscale=SPEED,
                  unit="m/s", zrange=(0, top), height=height)
    s = 6                                            # every 6th cell (1.5 deg) or it's a hairball
    uu, vv = u.values[::s, ::s], v.values[::s, ::s]
    X, Y = np.meshgrid(u.lon.values[::s], u.lat.values[::s])
    ok = np.isfinite(uu) & np.isfinite(vv)
    if ok.any():
        q = ff.create_quiver(X[ok], Y[ok], uu[ok], vv[ok], scale=1.2 / top, arrow_scale=0.3,
                             line=dict(color=ARROW, width=1), hoverinfo="skip",
                             showlegend=False)
        fig.add_traces(q.data)
    return fig


def timelapse(date, src, depth, view):
    """Every day of `date`'s quarter at one depth, as plotly frames -- the chunk already in
    memory, so it costs no extra load. Uses the same fixed range as the still map, so the
    animation shows the ocean changing, not the colour scale."""
    field = sub(L.quarter_field(date, depth, src))
    diverging = src == "error"
    days = [pd.Timestamp(t) for t in field.time.values]
    title = lambda d: f"{view} — {depth} m, {d:%d %b %Y}"  # noqa: E731
    fig = heatmap(field.isel(time=0), title(days[0]),
                  colorscale=DIVERGING if diverging else SEQ_TEMP,
                  zrange=diff_range(depth) if diverging else temp_range(depth),
                  depth=depth, height=560)
    fig.frames = [go.Frame(data=[go.Heatmap(z=field.isel(time=i).values.astype("float32"))],
                           traces=[1], name=f"{d:%Y-%m-%d}",
                           layout=dict(title=dict(text=title(d))))
                  for i, d in enumerate(days)]
    play = dict(frame=dict(duration=150, redraw=True), fromcurrent=True, transition=dict(duration=0))
    stop = dict(frame=dict(duration=0, redraw=False), mode="immediate")
    fig.update_layout(
        margin=dict(b=90),
        updatemenus=[dict(type="buttons", direction="left", x=0, y=-0.08, xanchor="left",
                          yanchor="top", showactive=False,
                          buttons=[dict(label="▶ Play", method="animate", args=[None, play]),
                                   dict(label="❚❚ Pause", method="animate",
                                        args=[[None], stop])])],
        sliders=[dict(x=0.2, y=-0.08, len=0.8, yanchor="top", currentvalue=dict(visible=False),
                      steps=[dict(method="animate", label=f"{d:%d %b}",
                                  args=[[f"{d:%Y-%m-%d}"], stop]) for d in days])])
    return fig


def html_table(df, formats=None):
    """A benchmark table as plain HTML (see .oter-table CSS): centred, grey header, glass
    body. st.dataframe's grid can't take a header colour or reliable header alignment from
    CSS at all -- it's canvas-drawn -- so for a table this heavily styled, real HTML is the
    native fit, not a workaround."""
    formats = formats or {}
    head = "".join(f"<th>{c}</th>" for c in df.columns)
    rows_html = []
    for _, row in df.iterrows():
        cells = "".join(f"<td>{formats.get(c, '{}').format(row[c])}</td>" for c in df.columns)
        rows_html.append(f"<tr>{cells}</tr>")
    st.markdown(
        f'<table class="oter-table"><thead><tr>{head}</tr></thead>'
        f'<tbody>{"".join(rows_html)}</tbody></table>',
        unsafe_allow_html=True)


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
with st.container(key="topbar"):
    h1, h2 = st.columns([5, 1], vertical_alignment="center")
    with h1:
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
    with h2:
        dark = mode == "dark"
        st.button("Light" if dark else "Dark", on_click=flip_mode, width="content",
                  help="Switch to light mode" if dark else "Switch to dark mode",
                  icon=":material/light_mode:" if dark else ":material/dark_mode:",
                  key="mode_btn")
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
sub = lambda da: da.sel(lon=slice(lon0, lon1))  # noqa: E731

t_inputs, t_map, t_profile, t_skill = st.tabs(
    ["Surface inputs", "Reconstruction", "Profile", "Benchmarks"])

# --- ① the seven things a satellite can see -------------------------------------------
with t_inputs:
    st.subheader("What the satellite sees")
    st.caption("These seven surface fields are the model's only input. Everything in the "
               "next tabs is inferred from them. Currents and winds are drawn as one "
               "speed-and-direction map each; their U/V channels are in the panel below.")
    x = L.inputs(date)
    panels = [("sst", SEQ_TEMP), ("sss", "Blues"), ("sla", DIVERGING), ("cur", None),
              ("wind", None)]
    cols = st.columns(2)
    for i, (ch, cs) in enumerate(panels):
        with cols[i % 2]:
            if cs is None:
                fig = vector_map(x, ch, "Surface current" if ch == "cur" else "Surface wind")
            else:
                label, unit = L.CHANNEL_LABEL[ch]
                fig = heatmap(sub(x[ch]), f"{label}  ({unit})", colorscale=cs, unit=unit,
                              zrange=input_range(ch), height=300)
            st.plotly_chart(fig, width="stretch", key=f"in_{ch}")

    with st.expander("Raw U / V components — the four vector channels as the model sees them"):
        cols = st.columns(2)
        for i, ch in enumerate(["cur_u", "cur_v", "wind_u", "wind_v"]):
            label, unit = L.CHANNEL_LABEL[ch]
            with cols[i % 2]:
                st.plotly_chart(
                    heatmap(sub(x[ch]), f"{label}  ({unit})", colorscale=DIVERGING,
                            unit=unit, zrange=input_range(ch), height=300),
                    width="stretch", key=f"raw_{ch}")

# --- ② the reconstruction -------------------------------------------------------------
with t_map:
    c1, c2 = st.columns([3, 1])
    with c2:
        with st.container(border=True):
            st.markdown("#### Display Mode")
            view = st.radio("Show", ["OTER vs GLORYS", "Difference (OTER − GLORYS)"],
                            label_visibility="collapsed")
            st.divider()
            st.caption(
                "GLORYS is the reanalysis the model was **trained on**, not ground truth: it "
                "runs about **+0.72 °C too warm at 100 m** against Argo floats in this basin. "
                "Measuring that is what let us correct it."
            )
            st.caption("Lines are isotherms. Colours are fixed per depth across the whole "
                       "test period, so changing the date only recolours what actually changed.")
    diff = view.startswith("Difference")
    when = f"{depth} m, {pd.Timestamp(date):%d %b %Y}"
    with c1:
        if diff:
            st.plotly_chart(
                heatmap(sub(L.field(date, depth, "error")), f"{view} — {when}",
                        colorscale=DIVERGING, zrange=diff_range(depth), depth=depth,
                        height=560),
                width="stretch", key="recon")
        else:
            # Stacked on one fixed colour scale (temp_range is shared), so the same colour
            # means the same temperature in both maps.
            for src, name, key in (("prediction", "OTER", "recon"),
                                   ("truth", "GLORYS reanalysis", "recon_glorys")):
                st.plotly_chart(
                    heatmap(sub(L.field(date, depth, src)), f"{name} — {when}",
                            zrange=temp_range(depth), depth=depth, contours=True,
                            height=430),
                    width="stretch", key=key)

    if st.toggle("▶ Timelapse — every day of this quarter at this depth",
                 help="Early December 2023 shows Cyclone Michaung crossing the Bay of Bengal."):
        with st.spinner("Building frames…"):
            st.plotly_chart(timelapse(date, "error" if diff else "prediction", depth,
                                      view if diff else "OTER"),
                            width="stretch", key="timelapse")

# --- ③ click a point, get the column --------------------------------------------------
with t_profile:
    st.caption("**Click anywhere on the map** to pull the full 0–1000 m column at that "
               "point. Diamonds are Argo floats within 3 days — click one for a direct "
               "comparison against an independent observation.")
    # The clickable map must not depend on the click: Streamlit derives the chart's widget
    # id from the figure spec, so a marker drawn from `pick` would make every click a new
    # widget. The ring on the picked cell is plotly's client-side selection style instead.
    prev = st.session_state.get("clickmap") or {}
    pts = (prev.get("selection") or {}).get("points", [])
    if pts:
        st.session_state["pick"] = (float(pts[0]["y"]), float(pts[0]["x"]))
    lat_s, lon_s = st.session_state.get("pick", (15.0, 88.0))

    c = ui()
    c1, c2 = st.columns([1, 1])
    with c1:
        dap = sub(L.field(date, depth, "prediction"))
        fig = heatmap(dap, f"Reconstruction — {depth} m (click to sample)",
                      zrange=temp_range(depth), depth=depth, height=470)
        # Plotly heatmaps are not selectable, so a click on one never reaches Streamlit.
        # One invisible selectable point per ocean cell makes the whole map clickable, and
        # plotly's own selected-style draws the ring on the picked cell client-side.
        LA, LO = np.meshgrid(dap.lat.values, dap.lon.values, indexing="ij")
        v = dap.values
        wet = np.isfinite(v)
        fig.add_trace(go.Scattergl(
            x=LO[wet], y=LA[wet], customdata=v[wet], mode="markers", showlegend=False,
            marker=dict(symbol="circle-open", size=16, color=c["fg"], opacity=0,
                        line=dict(width=3)),
            selected=dict(marker=dict(opacity=1)), unselected=dict(marker=dict(opacity=0)),
            hovertemplate="%{y:.2f}°N  %{x:.2f}°E<br><b>%{customdata:.2f} °C</b><extra></extra>"))
        fl = floats_near(date)
        fl = fl[(fl.lon >= lon0) & (fl.lon <= lon1)]
        fig.add_trace(go.Scatter(
            x=fl.lon, y=fl.lat, mode="markers", name="Argo float (±3 days)", text=fl.index,
            marker=dict(symbol="diamond", size=10, color=series(2),
                        line=dict(width=1.5, color=c["edge"])),
            selected=dict(marker=dict(color=c["fg"], size=14)),
            unselected=dict(marker=dict(opacity=1)),
            hovertemplate="Argo %{text}<br>%{y:.2f}°N  %{x:.2f}°E<extra>click to compare</extra>"))
        fig.update_layout(legend=dict(yanchor="top", y=-0.06), margin=dict(b=70))
        st.plotly_chart(fig, width="stretch", on_select="rerun",
                        selection_mode="points", key="clickmap")

    zz, pred = L.profile(date, lat_s, lon_s)
    cmp = None
    with c2:
        if not np.isfinite(pred).any():
            st.info("That point is land — click on the ocean.")
        else:
            cmp = L.argo_comparison(date, lat_s, lon_s)
            _, glo = L.profile(date, lat_s, lon_s, source="truth")
            rmse = (skill_table().set_index("Depth (m)")["RMSE (°C)"]
                    .reindex(zz.astype(int)).values)
            ok = np.isfinite(pred) & np.isfinite(rmse)
            y = zz + DZ
            fig = go.Figure()
            # A basin-wide error, not a per-point interval -- labelled as such.
            fig.add_trace(go.Scatter(
                x=np.r_[pred[ok] - rmse[ok], (pred[ok] + rmse[ok])[::-1]],
                y=np.r_[y[ok], y[ok][::-1]], fill="toself", line=dict(width=0),
                fillcolor="rgba(57,135,229,0.32)", name="Typical error (±RMSE vs Argo)",
                hoverinfo="skip"))
            fig.add_trace(go.Scatter(
                x=glo, y=y, customdata=zz, name="GLORYS (training target)", mode="lines",
                line=dict(color=series(3), width=2, dash="dash"),
                hovertemplate="%{customdata:.0f} m<br><b>%{x:.2f} °C</b><extra>GLORYS</extra>"))
            fig.add_trace(go.Scatter(
                x=pred, y=y, customdata=zz, name="OTER", mode="lines+markers",
                line=dict(color=series(0), width=2), marker=dict(size=8),
                hovertemplate="%{customdata:.0f} m<br><b>%{x:.2f} °C</b><extra>OTER</extra>"))
            if cmp:
                keep = cmp["pres"] <= 1000
                fig.add_trace(go.Scatter(
                    x=cmp["temp"][keep], y=cmp["pres"][keep] + DZ,
                    customdata=cmp["pres"][keep], name="Argo float (independent)",
                    mode="lines", line=dict(color=series(1), width=2),
                    hovertemplate="%{customdata:.0f} m<br><b>%{x:.2f} °C</b><extra>Argo</extra>"))
            fig.update_xaxes(title="Temperature (°C)")
            # No chart title: the legend sits along the top and the two would collide. The
            # caption below carries the coordinates instead.
            base_layout(fig, height=470, hovermode="closest")
            depth_axis(fig)
            st.plotly_chart(fig, width="stretch", key="prof")
            st.markdown(f"**Column at {lat_s:.2f}°N, {lon_s:.2f}°E** · depth axis is "
                        f"stretched near the surface, where the thermocline is")

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
    elif np.isfinite(pred).any():
        st.info("No Argo float within 1.5° and 3 days of this point. Click one of the "
                "diamonds on the map — coverage is sparse, which is the entire reason this "
                "project exists.")

# --- ④ does it actually work ----------------------------------------------------------
with t_skill:
    st.subheader("Accuracy against independent Argo floats")
    k = st.columns(4)
    k[0].metric("OTER", "0.786 °C")
    k[1].metric("Climatology baseline", "1.160 °C", "-32% error", delta_color="inverse")
    k[2].metric("GLORYS reanalysis", "0.728 °C", help="The product the model learns from.")
    k[3].metric("Depths beating baseline", "15 / 15")

    tab = skill_table()
    fig = go.Figure()
    for i, (col, name) in enumerate([("Climatology RMSE", "Climatology"),
                                     ("RMSE (°C)", "OTER"),
                                     ("GLORYS RMSE", "GLORYS reanalysis")]):
        if col in tab:
            fig.add_trace(go.Scatter(
                x=tab[col], y=tab["Depth (m)"] + DZ, customdata=tab["Depth (m)"], name=name,
                mode="lines+markers", line=dict(color=series([2, 0, 3][i]), width=2),
                marker=dict(size=8),
                hovertemplate="%{customdata:.0f} m<br><b>%{x:.3f} °C</b><extra>" + name + "</extra>"))
    fig.update_xaxes(title="RMSE (°C) — lower is better")
    # Title lives in the markdown above, not in the figure: a top-anchored horizontal
    # legend and a top-left title occupy the same strip and overlap.
    base_layout(fig, height=460, hovermode="y")
    depth_axis(fig)
    st.markdown("**Error against Argo, by depth**")
    st.plotly_chart(fig, width="stretch", key="skill")

    html_table(tab, formats={"Depth (m)": "{:.0f}",
                             **{c: "{:.3f}" for c in tab.columns if c != "Depth (m)"}})
    st.caption(f"Test split, all {man['argo_profiles']:,} independent Argo casts across "
               f"the full 2023–24 test period.")

    with st.expander("Full metric comparison — OTER vs the GLORYS reanalysis it learns from"):
        fin, glo = L.final_vs_glorys()
        LABEL = {"rmse": "RMSE", "mae": "MAE", "bias": "Bias", "corr": "Corr", "r2": "R²"}
        rows = pd.DataFrame([
            {"Metric": LABEL[k], "GLORYS": glo[k], "OTER": fin[k]}
            for k in ("rmse", "mae", "bias", "corr", "r2")])
        html_table(rows, formats={"GLORYS": "{:.4f}", "OTER": "{:.4f}"})
