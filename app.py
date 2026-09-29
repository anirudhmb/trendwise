"""India Stocks — STL Decomposition Dashboard.

Streamlit app: pick a NIFTY 500 stock, fetch daily prices via yfinance,
run STL decomposition, and render an annotated dashboard.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots
from statsmodels.tsa.seasonal import STL

from forecasting import ForecastResult, forecast_stl_classical, forecast_timesfm, load_timesfm_model

APP_DIR = Path(__file__).parent
UNIVERSE_CSV = APP_DIR / "data" / "nifty500.csv"
HISTORY_START = "2018-01-01"
SEASONAL_PERIOD = 21  # ~1 trading month
MIN_HISTORY_FOR_RELIABLE_FORECAST = 500  # ~2 years of trading days
FORECAST_TIMESFM_COLOR = "#f472b6"
FORECAST_STL_COLOR = "#ffb454"

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="TrendWise · NIFTY 500 STL",
    page_icon="🇮🇳",
    layout="wide",
    initial_sidebar_state="collapsed",
)

CSS = """
<style>
    #MainMenu, footer, header {visibility: hidden;}
    .block-container {padding-top: 0.2rem; padding-bottom: 1rem; max-width: 1500px;}

    .brand {
        display: flex; align-items: baseline; gap: 12px;
        margin: 0 0 2px 0;
    }
    .brand h1 {color: #f8fafc; font-size: 18px; margin: 0; letter-spacing: 0.4px;}
    .brand p  {color: #94a3b8; margin: 0; font-size: 12px;}

    .card {
        background: #111827;
        border: 1px solid #1f2937;
        border-radius: 10px;
        padding: 8px 12px;
        text-align: left;
        height: 100%;
    }
    .card .lbl {color: #94a3b8; font-size: 10px; text-transform: uppercase; letter-spacing: 0.5px;}
    .card .val {color: #f8fafc; font-size: 17px; font-weight: 600; margin-top: 2px;}
    .card .sub {font-size: 11px; margin-top: 2px; color: #cbd5e1;}
    .up   {color: #22c55e;}
    .down {color: #ef4444;}
    .flat {color: #94a3b8;}

    .insight {
        background: #0b1220;
        border-left: 3px solid #38bdf8;
        border-radius: 6px;
        padding: 8px 12px;
        color: #e2e8f0;
        font-size: 12.5px;
        line-height: 1.5;
        margin: 6px 0;
    }
    .insight b {color: #f8fafc;}
    div[data-testid="stCaptionContainer"] {margin-top: -6px;}

    /* Section demarcation: wrap Streamlit columns in bordered panels */
    div[data-testid="stHorizontalBlock"] > div[data-testid="column"] {
        background: #0b1220;
        border: 1px solid #1f2937;
        border-radius: 12px;
        padding: 10px 12px;
    }
    /* Don't panel the top control row (Stock / Lookback / STL input) */
    div[data-testid="stHorizontalBlock"]:first-of-type > div[data-testid="column"] {
        background: transparent;
        border: none;
        padding: 0;
    }
    /* Don't panel the nested 2x2 metric-card columns inside the right panel */
    div[data-testid="column"] div[data-testid="stHorizontalBlock"] > div[data-testid="column"] {
        background: transparent;
        border: none;
        padding: 0;
    }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# Preload TimesFM once per container so the first stock pick doesn't stall on
# a ~800MB download / model compile. Fails soft — the app still runs on the
# STL+Holt baseline if TimesFM can't load.
if "_timesfm_boot_attempted" not in st.session_state:
    st.session_state["_timesfm_boot_attempted"] = True
    with st.spinner(""):
        load_timesfm_model()


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


@st.cache_data
def load_universe() -> pd.DataFrame:
    df = pd.read_csv(UNIVERSE_CSV)
    df = df.rename(columns={"Company Name": "name", "Industry": "industry", "Symbol": "symbol"})
    df["ticker"] = df["symbol"].str.strip() + ".NS"
    df["label"] = df["symbol"] + " — " + df["name"]
    return df[["ticker", "symbol", "name", "industry", "label"]].sort_values("symbol").reset_index(drop=True)


@st.cache_data(ttl=60 * 30, show_spinner=False)
def fetch_prices(ticker: str) -> pd.DataFrame:
    end = (date.today() + timedelta(days=1)).isoformat()
    df = yf.download(ticker, start=HISTORY_START, end=end, auto_adjust=True, progress=False)
    if df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df[["Open", "High", "Low", "Close", "Volume"]].dropna()


def decompose(series: pd.Series) -> pd.DataFrame:
    stl = STL(series, period=SEASONAL_PERIOD, robust=True).fit()
    return pd.DataFrame(
        {
            "observed": stl.observed,
            "trend": stl.trend,
            "seasonal": stl.seasonal,
            "resid": stl.resid,
        }
    )


def transform_for_stl(close: pd.Series, mode: str) -> pd.Series:
    """Prep the close series for STL based on the chosen mode.

    - price:   raw close (multiplicative variance, non-stationary trend)
    - log:     log(close)  → additive variance, still non-stationary
    - returns: log(close).diff() → ~stationary daily log returns
    """
    if mode == "Log price":
        return np.log(close).dropna()
    if mode == "Returns":
        return np.log(close).diff().dropna()
    return close


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------


def arrow(x: float) -> tuple[str, str]:
    if x > 0:
        return "▲", "up"
    if x < 0:
        return "▼", "down"
    return "■", "flat"


def card(col, label: str, value: str, sub_html: str = "") -> None:
    col.markdown(
        f'<div class="card"><div class="lbl">{label}</div>'
        f'<div class="val">{value}</div>'
        f'<div class="sub">{sub_html}</div></div>',
        unsafe_allow_html=True,
    )


def insight(text: str) -> None:
    st.markdown(f'<div class="insight">{text}</div>', unsafe_allow_html=True)


def _add_forecast_traces(
    fig: go.Figure,
    forecasts: list[ForecastResult],
    show_band: bool,
    anchor_date: pd.Timestamp | None = None,
    anchor_value: float | None = None,
    low_confidence: bool = False,
) -> None:
    """Overlay forecast lines (+ optional TimesFM band) on the price figure.

    If anchor_date/anchor_value are given, the last-actual point is prepended to
    each forecast line so there's no visual gap between actuals and forecast.
    """
    if not forecasts:
        return

    sep_date = anchor_date if anchor_date is not None else min(f.dates[0] for f in forecasts) - pd.Timedelta(days=1)
    fig.add_vline(
        x=sep_date, line=dict(color="#64748b", width=1, dash="dot"),
        annotation_text="forecast →", annotation_position="top right",
        annotation=dict(font=dict(color="#94a3b8", size=10),
                        bgcolor="rgba(11,18,32,0.7)"),
    )

    def _prepend(x_seq, y_seq):
        if anchor_date is None or anchor_value is None:
            return list(x_seq), list(y_seq)
        return [anchor_date] + list(x_seq), [anchor_value] + list(y_seq)

    for fc in forecasts:
        if fc.method.startswith("TimesFM"):
            color = FORECAST_TIMESFM_COLOR
            display_name = "Model Foreseer"
            if show_band and fc.lower is not None and fc.upper is not None:
                lx, ly = _prepend(fc.dates, fc.lower)
                ux, uy = _prepend(fc.dates, fc.upper)
                fig.add_trace(
                    go.Scatter(
                        x=list(ux) + list(lx[::-1]),
                        y=list(uy) + list(ly[::-1]),
                        fill="toself",
                        fillcolor="rgba(244,114,182,0.18)",
                        line=dict(color="rgba(0,0,0,0)"),
                        name="Model Foreseer · p10–p90",
                        hoverinfo="skip",
                        showlegend=True,
                    )
                )
        else:
            color = FORECAST_STL_COLOR
            display_name = "Model Pulsecast"

        x_line, y_line = _prepend(fc.dates, fc.point)
        line_kwargs = dict(color=color, width=2)
        if low_confidence:
            line_kwargs["dash"] = "dash"
        fig.add_trace(
            go.Scatter(
                x=x_line, y=y_line, mode="lines",
                name=f"{display_name} forecast",
                line=line_kwargs,
                hovertemplate="%{x|%d %b %Y}<br><b>₹%{y:,.2f}</b> · " + display_name + "<extra></extra>",
            )
        )


def price_chart(
    prices: pd.DataFrame,
    window: pd.DataFrame,
    forecasts: list[ForecastResult] | None = None,
    show_band: bool = True,
    low_confidence: bool = False,
) -> go.Figure:
    fig = go.Figure()

    hi_52w = prices["Close"].iloc[-252:].max() if len(prices) >= 20 else prices["Close"].max()
    lo_52w = prices["Close"].iloc[-252:].min() if len(prices) >= 20 else prices["Close"].min()
    ma50 = window["Close"].rolling(50).mean()
    ma200 = window["Close"].rolling(200).mean()

    fig.add_trace(
        go.Scatter(
            x=window.index, y=window["Close"], mode="lines", name="Close",
            line=dict(color="#38bdf8", width=2),
            hovertemplate="%{x|%d %b %Y}<br><b>₹%{y:,.2f}</b><extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(x=window.index, y=ma50, mode="lines", name="50d MA",
                   line=dict(color="#f59e0b", width=1, dash="dot"))
    )
    fig.add_trace(
        go.Scatter(x=window.index, y=ma200, mode="lines", name="200d MA",
                   line=dict(color="#a78bfa", width=1, dash="dot"))
    )
    fig.add_hline(
        y=hi_52w, line=dict(color="#22c55e", width=1, dash="dash"),
        annotation_text=f"52w High ₹{hi_52w:,.0f}",
        annotation_position="top left",
        annotation=dict(
            yshift=10, font=dict(color="#22c55e", size=11),
            bgcolor="rgba(11,18,32,0.7)",
        ),
    )
    fig.add_hline(
        y=lo_52w, line=dict(color="#ef4444", width=1, dash="dash"),
        annotation_text=f"52w Low ₹{lo_52w:,.0f}",
        annotation_position="bottom left",
        annotation=dict(
            yshift=-10, font=dict(color="#ef4444", size=11),
            bgcolor="rgba(11,18,32,0.7)",
        ),
    )

    anchor_date = window.index[-1] if len(window) else None
    anchor_value = float(window["Close"].iloc[-1]) if len(window) else None
    _add_forecast_traces(fig, forecasts or [], show_band, anchor_date, anchor_value, low_confidence)

    y_candidates_min = [float(window["Close"].min()), lo_52w]
    y_candidates_max = [float(window["Close"].max()), hi_52w]
    for fc in forecasts or []:
        y_candidates_min.append(float(np.min(fc.lower if fc.lower is not None else fc.point)))
        y_candidates_max.append(float(np.max(fc.upper if fc.upper is not None else fc.point)))
    y_min = min(y_candidates_min)
    y_max = max(y_candidates_max)
    pad = (y_max - y_min) * 0.06
    fig.update_yaxes(range=[y_min - pad, y_max + pad])

    fig.update_layout(
        template="plotly_dark",
        height=360,
        margin=dict(t=60, b=25, l=10, r=10),
        title=dict(
            text="<b>Price</b> · 50 / 200-day MA · 52-week band",
            x=0.01, y=0.97, font=dict(size=13, color="#f8fafc"),
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.02,
            xanchor="right", x=1.0,
            bgcolor="rgba(17,24,39,0.85)",
            bordercolor="#334155",
            borderwidth=1,
            font=dict(color="#e2e8f0", size=11),
            itemsizing="constant",
        ),
        hovermode="x unified",
        paper_bgcolor="#0b1220",
        plot_bgcolor="#0b1220",
    )
    return fig


def _mode_labels(mode: str) -> tuple[str, str, str]:
    """Return (trend_title, trend_hover, unit_hover) for a given STL mode."""
    if mode == "Price":
        return (
            "STL Trend (long-term price direction)",
            "%{x|%d %b %Y}<br>Trend ₹%{y:,.2f}<extra></extra>",
            "%{x|%d %b %Y}<br>%{y:+.2f}<extra></extra>",
        )
    if mode == "Log price":
        return (
            "STL Trend of log(price)",
            "%{x|%d %b %Y}<br>log-Trend %{y:.3f}<extra></extra>",
            "%{x|%d %b %Y}<br>%{y:+.4f}<extra></extra>",
        )
    return (
        "STL Trend of daily log-returns (drift)",
        "%{x|%d %b %Y}<br>Drift %{y:+.4f}<extra></extra>",
        "%{x|%d %b %Y}<br>%{y:+.4f}<extra></extra>",
    )


def trend_chart(decomp: pd.DataFrame, mode: str) -> go.Figure:
    trend_title, trend_hover, _ = _mode_labels(mode)
    fig = go.Figure(
        go.Scatter(
            x=decomp.index, y=decomp["trend"], mode="lines",
            line=dict(color="#22c55e", width=2), name="trend",
            hovertemplate=trend_hover,
        )
    )
    fig.update_layout(
        template="plotly_dark",
        height=260,
        margin=dict(t=40, b=25, l=10, r=10),
        title=dict(text=f"<b>{trend_title}</b>", x=0.01, y=0.97,
                   font=dict(size=13, color="#f8fafc")),
        showlegend=False,
        paper_bgcolor="#0b1220",
        plot_bgcolor="#0b1220",
        hovermode="x unified",
    )
    return fig


def seasonal_resid_chart(decomp: pd.DataFrame, mode: str) -> go.Figure:
    _, _, unit_hover = _mode_labels(mode)
    resid_std = decomp["resid"].std()
    anomalies = decomp[decomp["resid"].abs() > 2 * resid_std]

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=False,
        subplot_titles=(f"Seasonal ({SEASONAL_PERIOD}-day cycle)",
                        "Residual · red = |z| > 2"),
        vertical_spacing=0.18,
        row_heights=[0.5, 0.5],
    )
    fig.add_trace(
        go.Scatter(x=decomp.index, y=decomp["seasonal"], mode="lines",
                   line=dict(color="#f59e0b", width=1.3), name="seasonal",
                   hovertemplate=unit_hover),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=decomp.index, y=decomp["resid"], mode="lines",
                   line=dict(color="#94a3b8", width=1), name="resid",
                   hovertemplate=unit_hover),
        row=2, col=1,
    )
    fig.add_hrect(y0=-2 * resid_std, y1=2 * resid_std, fillcolor="#38bdf8",
                  opacity=0.08, line_width=0, row=2, col=1)
    if not anomalies.empty:
        fig.add_trace(
            go.Scatter(
                x=anomalies.index, y=anomalies["resid"], mode="markers",
                marker=dict(color="#ef4444", size=6, symbol="circle-open", line=dict(width=2)),
                name="anomaly",
                hovertemplate="%{x|%d %b %Y}<br><b>Anomaly</b> %{y:+.2f}<extra></extra>",
            ),
            row=2, col=1,
        )
    fig.update_layout(
        template="plotly_dark",
        height=360,
        margin=dict(t=30, b=20, l=10, r=10),
        showlegend=False,
        paper_bgcolor="#0b1220",
        plot_bgcolor="#0b1220",
        hovermode="x unified",
    )
    fig.update_annotations(font_size=11)
    return fig


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

st.markdown('<div class="brand"><h1>TrendWise</h1></div>', unsafe_allow_html=True)

universe = load_universe()

top = st.columns([6, 1], vertical_alignment="bottom")
choice = top[0].selectbox(
    "Stock",
    options=universe["label"],
    index=None,
    placeholder="Search a NIFTY 500 stock (e.g. RELIANCE, TCS, INFY)…",
)
lookback = top[1].selectbox("Lookback", ["6M", "1Y", "2Y", "3Y", "Max"], index=2)

# Hardcoded defaults — dropdowns hidden from UI, all forecasting/decomposition
# machinery below still honours the same variables so nothing else has to change.
stl_mode = "Log price"

# Kept for reference; the mode picker is no longer rendered but the code paths
# in forecasting.py still support Raw / Trend / Trend+Resid.
timesfm_mode_labels = {
    "Raw": "Raw — TimesFM on close prices directly (foundation-model default).",
    "Trend": "Trend — TimesFM on STL trend only; seasonal-naive repeat. Fairest vs STL+Holt.",
    "Trend+Resid": "Trend + Resid — TimesFM on trend and residual; seasonal-naive repeat.",
    "Components": "Components — TimesFM on all three STL components (paper-style).",
}

period_map = {"6M": 126, "1Y": 252, "2Y": 504, "3Y": 756, "Max": None}

if choice is None:
    st.markdown(
        '<div class="insight" style="margin-top:24px;">'
        '👋 <b>Welcome to TrendWise.</b> Pick any NIFTY 500 stock from the search box above '
        'to see its price broken down into <b>Trend</b>, <b>Seasonal cycle</b>, and <b>Residual</b> '
        '(unusual moves) — with anomaly markers, moving averages, and 52-week context.'
        '</div>',
        unsafe_allow_html=True,
    )
    with st.expander("How to read this dashboard", expanded=True):
        st.markdown(
            """
- **Trend** — smoothed long-term direction. Rising trend + rising price = healthy uptrend.
- **Seasonal** — repeating pattern over a fixed period (here 21 trading days ≈ 1 month).
- **Residual** — what's left after removing trend and seasonal. Days where |residual| > 2σ
  (red circles) are statistically unusual moves — often earnings, news, or macro shocks.
- **52-week band** — dashed green/red lines on price chart show yearly high/low context.
- **50/200-day MA** — classic momentum reference; 50 > 200 is a "golden cross" regime.

Data: Yahoo Finance via `yfinance` (`.NS` = NSE). Cached 30 min per session.
            """
        )
    st.stop()

selected = universe[universe["label"] == choice].iloc[0]

with st.spinner(f"Fetching {selected['ticker']}…"):
    prices = fetch_prices(selected["ticker"])

if prices.empty or len(prices) < SEASONAL_PERIOD * 3:
    st.error(f"Not enough data for {selected['ticker']}. Try a different stock.")
    st.stop()

n = period_map[lookback]
window = prices if n is None else prices.iloc[-n:]

close = window["Close"]
latest = float(close.iloc[-1])
prev = float(close.iloc[-2])
chg = latest - prev
pct = chg / prev * 100
hi_52w = float(prices["Close"].iloc[-252:].max())
lo_52w = float(prices["Close"].iloc[-252:].min())
pos_in_range = (latest - lo_52w) / (hi_52w - lo_52w) * 100 if hi_52w > lo_52w else 50.0
vol_today = int(window["Volume"].iloc[-1])
vol_avg30 = float(window["Volume"].iloc[-30:].mean())
vol_ratio = vol_today / vol_avg30 if vol_avg30 > 0 else 1.0

with st.spinner(f"Running STL on {stl_mode.lower()} ({lookback})…"):
    stl_input = transform_for_stl(close, stl_mode)
    decomp = decompose(stl_input)

# ---- Forecasts: only computed if the user turns the forecast unit on.
forecasts: list[ForecastResult] = []
timesfm_fc: ForecastResult | None = None
stl_fc: ForecastResult | None = None
timesfm_error: str | None = None

timesfm_mode = "Components"  # dropdown hidden; forecasting.py still supports the other modes
horizon = 0  # set only when forecast is enabled below

# Trend delta over the last 21 sessions — read from the actual STL trend series
# so the metric card matches whatever the trend chart is showing (Price / Log price / Returns).
trend_series = decomp["trend"]
trend_end = float(trend_series.iloc[-1])
trend_start = float(trend_series.iloc[-min(len(trend_series), SEASONAL_PERIOD + 1)])

if stl_mode == "Price":
    # trend is in rupees
    trend_pct = (trend_end - trend_start) / trend_start * 100 if trend_start else 0.0
    trend_headline = f"{trend_pct:+.2f}%"
    trend_sub = f"₹{trend_start:,.0f} → ₹{trend_end:,.0f}"
    trend_dir = trend_pct
elif stl_mode == "Log price":
    # trend is log-price; % change ≈ exp(Δlog) - 1
    trend_pct = (np.exp(trend_end - trend_start) - 1) * 100
    trend_headline = f"{trend_pct:+.2f}%"
    trend_sub = f"₹{np.exp(trend_start):,.0f} → ₹{np.exp(trend_end):,.0f} (implied)"
    trend_dir = trend_pct
else:  # Returns mode — trend IS the drift; report drift level, not % change of a rate
    # Daily log-return → bps/day
    drift_start_bps = trend_start * 10_000
    drift_end_bps = trend_end * 10_000
    trend_headline = f"{drift_end_bps:+.1f} bps/d"
    trend_sub = f"drift {drift_start_bps:+.1f} → {drift_end_bps:+.1f} bps/day"
    trend_dir = drift_end_bps  # sign of current drift drives the arrow
    trend_pct = drift_end_bps  # used later for the insight line

resid_std = float(decomp["resid"].std())
latest_resid = float(decomp["resid"].iloc[-1])
resid_z = latest_resid / resid_std if resid_std > 0 else 0.0
seasonal_today = float(decomp["seasonal"].iloc[-1])
seasonal_amp = float(decomp["seasonal"].abs().max())

# --- header meta (single line above the grid)
st.caption(f"**{selected['name']}** · {selected['industry']} · `{selected['ticker']}` · "
           f"data through {prices.index.max().date().isoformat()}")

# --- 2-column grid: charts on the left, metrics + secondary charts on the right
left, right = st.columns([3, 2], gap="small")

with left:
    # Forecast unit: Off by default. When On, Model + Horizon appear together
    # and drive computation. Nothing runs while Off.
    show_timesfm = show_stl = False
    show_band = False
    fcols = st.columns([1.2, 2, 2, 3])
    forecast_on = fcols[0].toggle("Forecast", value=False, key="forecast_on")
    if forecast_on:
        model_pick = fcols[1].selectbox(
            "Model", options=["Model Foreseer", "Model Pulsecast"],
            index=0, key="forecast_model", label_visibility="collapsed",
        )
        horizon = int(fcols[2].number_input(
            "Horizon (days)", min_value=5, max_value=63, value=21, step=1,
            key="forecast_horizon", label_visibility="collapsed",
        ))
        if model_pick == "Model Foreseer" and len(close) >= 100:
            with st.spinner("Forecasting with Model Foreseer…"):
                timesfm_fc = forecast_timesfm(
                    close, decomp, horizon, SEASONAL_PERIOD, stl_mode, timesfm_mode,
                )
            if timesfm_fc is None:
                timesfm_error = st.session_state.get("_timesfm_error", "Model Foreseer unavailable")
                fcols[3].caption(f"⚠️ {timesfm_error[:80]}")
            show_timesfm = timesfm_fc is not None
            show_band = show_timesfm
        elif model_pick == "Model Pulsecast":
            stl_fc = forecast_stl_classical(decomp, close, horizon, SEASONAL_PERIOD, stl_mode)
            show_stl = stl_fc is not None

    forecasts = []
    if timesfm_fc is not None and show_timesfm:
        forecasts.append(timesfm_fc)
    if stl_fc is not None and show_stl:
        forecasts.append(stl_fc)

    low_confidence = len(prices) < MIN_HISTORY_FOR_RELIABLE_FORECAST
    if forecast_on and low_confidence and forecasts:
        st.markdown(
            f'<div style="background:#3f1d1d;border-left:3px solid #ef4444;'
            f'border-radius:6px;padding:6px 10px;margin:2px 0 6px 0;'
            f'color:#fecaca;font-size:12px;">'
            f'⚠️ Only <b>{len(prices)}</b> trading days of history — forecast is illustrative, '
            f'treat with caution. Reliable forecasts need ≥ {MIN_HISTORY_FOR_RELIABLE_FORECAST} days.'
            f'</div>',
            unsafe_allow_html=True,
        )
    with st.spinner("Rendering price chart…"):
        st.plotly_chart(
            price_chart(prices, window, forecasts=forecasts, show_band=show_band,
                        low_confidence=low_confidence),
            use_container_width=True,
        )
    with st.spinner("Rendering trend…"):
        st.plotly_chart(trend_chart(decomp, stl_mode), use_container_width=True)

with right:
    # metric cards, 2x2
    r1 = st.columns(2)
    arr_sym, arr_cls = arrow(chg)
    card(r1[0], "Last close", f"₹{latest:,.2f}",
         f'<span class="{arr_cls}">{arr_sym} {chg:+,.2f} ({pct:+.2f}%)</span>')

    t_sym, t_cls = arrow(trend_dir)
    card(r1[1], "Trend (21d)", f'<span class="{t_cls}">{t_sym} {trend_headline}</span>',
         trend_sub)

    r2 = st.columns(2)
    r_cls = "down" if abs(resid_z) > 2 else ("up" if abs(resid_z) > 1 else "flat")
    card(r2[0], "Residual today", f'<span class="{r_cls}">{latest_resid:+.2f}</span>',
         f"z-score {resid_z:+.2f}σ")
    card(r2[1], "52-week range", f"₹{lo_52w:,.0f}–{hi_52w:,.0f}",
         f"{pos_in_range:.0f}% of range · Vol {vol_ratio:.1f}×")

    # plain-english insight
    trend_word = "rising" if trend_dir > 0 else ("falling" if trend_dir < 0 else "flat")
    resid_word = ("an unusually large move" if abs(resid_z) > 2
                  else "a noticeable move" if abs(resid_z) > 1
                  else "within normal noise")
    season_word = ("near a cyclical peak" if seasonal_today > 0.5 * seasonal_amp
                   else "near a cyclical trough" if seasonal_today < -0.5 * seasonal_amp
                   else "mid-cycle")
    insight(
        f"<b>Trend</b> {trend_word} {trend_headline} (21d) · "
        f"<b>Seasonal</b> {season_word} · "
        f"<b>Residual</b> {resid_word} (z {resid_z:+.2f})."
    )

    with st.spinner("Rendering seasonal & residual…"):
        st.plotly_chart(seasonal_resid_chart(decomp, stl_mode), use_container_width=True)

st.caption(
    f"STL input: **{stl_mode}** — "
    + {
        "Price": "raw close; trend/residual carry rupee units.",
        "Log price": "log(close); additive variance, residual is a log deviation.",
        "Returns": "log(close).diff(); stationary daily log-returns.",
    }[stl_mode]
)

if horizon > 0 and (timesfm_fc is not None or stl_fc is not None):
    with st.expander("How to read the forecast"):
        st.markdown(
            """
- **Model Foreseer** is a pretrained time-series foundation model applied to the
  STL components (trend, seasonal, residual) and recombined. The shaded band is
  its own **p10–p90 uncertainty** estimate.
- **Model Pulsecast** extrapolates the STL trend (Holt's damped linear method) and
  repeats the last seasonal cycle. It's a transparent, classical baseline — useful
  for sanity-checking the neural forecast.
- Neither is investment advice. Stock forecasting is genuinely hard; treat these as
  pattern extrapolations, not predictions. **Divergence between the two lines is itself
  informative** — when they disagree, uncertainty is high.
            """
        )

with st.expander("How to read this dashboard"):
    st.markdown(
        """
- **Trend** — smoothed long-term direction.
- **Seasonal** — repeating ~monthly (21-day) pattern.
- **Residual** — noise after removing the two above. Red circles = |z| > 2σ (unusual days).
- **52-week band** — dashed lines mark the yearly high/low.
- **50/200-day MA** — golden cross when 50 crosses above 200.

Data: Yahoo Finance via `yfinance` (`.NS` = NSE). Cached 30 min per session.
        """
    )
