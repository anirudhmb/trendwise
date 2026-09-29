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

APP_DIR = Path(__file__).parent
UNIVERSE_CSV = APP_DIR / "data" / "nifty500.csv"
HISTORY_START = "2018-01-01"
SEASONAL_PERIOD = 21  # ~1 trading month

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
    .block-container {padding-top: 0.6rem; padding-bottom: 1rem; max-width: 1500px;}

    .brand {
        display: flex; align-items: baseline; gap: 12px;
        margin: 0 0 6px 0;
    }
    .brand h1 {color: #f8fafc; font-size: 20px; margin: 0; letter-spacing: 0.4px;}
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


def price_chart(prices: pd.DataFrame, window: pd.DataFrame) -> go.Figure:
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

    y_min = min(float(window["Close"].min()), lo_52w)
    y_max = max(float(window["Close"].max()), hi_52w)
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
            xanchor="left", x=0,
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

st.markdown(
    '<div class="brand"><h1>TrendWise</h1>'
    '<p>NIFTY 500 · STL decomposition · yfinance</p></div>',
    unsafe_allow_html=True,
)

universe = load_universe()

top = st.columns([4, 1, 1], vertical_alignment="bottom")
choice = top[0].selectbox(
    "Stock",
    options=universe["label"],
    index=None,
    placeholder="Search a NIFTY 500 stock (e.g. RELIANCE, TCS, INFY)…",
)
lookback = top[1].selectbox("Lookback", ["6M", "1Y", "2Y", "3Y", "Max"], index=2)
stl_mode = top[2].selectbox(
    "STL input",
    ["Price", "Log price", "Returns"],
    index=1,
    help=(
        "Price: raw close (multiplicative variance).  "
        "Log price: additive variance — recommended.  "
        "Returns: ~stationary daily log-returns, trend becomes drift."
    ),
)
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

# Trend % over the last 21 sessions — computed on raw price regardless of mode,
# so the metric card stays interpretable ("trend rose 3.2%") in every mode.
price_21_ago = float(close.iloc[-22])
price_now = float(close.iloc[-1])
trend_pct = (price_now - price_21_ago) / price_21_ago * 100
trend_from = price_21_ago
trend_to = price_now

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
    with st.spinner("Rendering price chart…"):
        st.plotly_chart(price_chart(prices, window), use_container_width=True)
    with st.spinner("Rendering trend…"):
        st.plotly_chart(trend_chart(decomp, stl_mode), use_container_width=True)

with right:
    # metric cards, 2x2
    r1 = st.columns(2)
    arr_sym, arr_cls = arrow(chg)
    card(r1[0], "Last close", f"₹{latest:,.2f}",
         f'<span class="{arr_cls}">{arr_sym} {chg:+,.2f} ({pct:+.2f}%)</span>')

    t_sym, t_cls = arrow(trend_pct)
    card(r1[1], "Trend (21d)", f'<span class="{t_cls}">{t_sym} {trend_pct:+.2f}%</span>',
         f"₹{trend_from:,.0f} → ₹{trend_to:,.0f}")

    r2 = st.columns(2)
    r_cls = "down" if abs(resid_z) > 2 else ("up" if abs(resid_z) > 1 else "flat")
    card(r2[0], "Residual today", f'<span class="{r_cls}">{latest_resid:+.2f}</span>',
         f"z-score {resid_z:+.2f}σ")
    card(r2[1], "52-week range", f"₹{lo_52w:,.0f}–{hi_52w:,.0f}",
         f"{pos_in_range:.0f}% of range · Vol {vol_ratio:.1f}×")

    # plain-english insight
    trend_word = "rising" if trend_pct > 0 else ("falling" if trend_pct < 0 else "flat")
    resid_word = ("an unusually large move" if abs(resid_z) > 2
                  else "a noticeable move" if abs(resid_z) > 1
                  else "within normal noise")
    season_word = ("near a cyclical peak" if seasonal_today > 0.5 * seasonal_amp
                   else "near a cyclical trough" if seasonal_today < -0.5 * seasonal_amp
                   else "mid-cycle")
    insight(
        f"<b>Trend</b> {trend_word} {trend_pct:+.2f}% (21d) · "
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
