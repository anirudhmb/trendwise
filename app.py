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
    #MainMenu, footer {visibility: hidden;}
    .block-container {padding-top: 1.5rem; padding-bottom: 2rem; max-width: 1400px;}

    .hero {
        background: linear-gradient(120deg, #0f172a 0%, #1e293b 40%, #334155 100%);
        border: 1px solid #334155;
        border-radius: 14px;
        padding: 22px 28px;
        margin-bottom: 18px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.25);
    }
    .hero h1 {color: #f8fafc; font-size: 26px; margin: 0 0 4px 0; letter-spacing: 0.5px;}
    .hero p  {color: #94a3b8; margin: 0; font-size: 14px;}

    .card {
        background: #111827;
        border: 1px solid #1f2937;
        border-radius: 12px;
        padding: 14px 18px;
        text-align: left;
        height: 100%;
    }
    .card .lbl {color: #94a3b8; font-size: 12px; text-transform: uppercase; letter-spacing: 0.6px;}
    .card .val {color: #f8fafc; font-size: 22px; font-weight: 600; margin-top: 4px;}
    .card .sub {font-size: 12px; margin-top: 4px;}
    .up   {color: #22c55e;}
    .down {color: #ef4444;}
    .flat {color: #94a3b8;}

    .insight {
        background: #0b1220;
        border-left: 3px solid #38bdf8;
        border-radius: 8px;
        padding: 10px 14px;
        color: #e2e8f0;
        font-size: 14px;
        margin: 8px 0;
    }
    .insight b {color: #f8fafc;}
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
        height=420,
        margin=dict(t=90, b=40, l=10, r=10),
        title=dict(
            text="<b>Price</b> · 50 / 200-day MA · 52-week band",
            x=0.01, y=0.97, font=dict(size=15, color="#f8fafc"),
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.02,
            xanchor="left", x=0,
            bgcolor="rgba(17,24,39,0.85)",
            bordercolor="#334155",
            borderwidth=1,
            font=dict(color="#e2e8f0", size=12),
            itemsizing="constant",
        ),
        hovermode="x unified",
        paper_bgcolor="#0b1220",
        plot_bgcolor="#0b1220",
    )
    return fig


def stl_chart(decomp: pd.DataFrame) -> go.Figure:
    resid_std = decomp["resid"].std()
    anomalies = decomp[decomp["resid"].abs() > 2 * resid_std]

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        subplot_titles=("Trend (long-term direction)",
                        f"Seasonal (repeating {SEASONAL_PERIOD}-day cycle)",
                        "Residual (unusual moves — markers = |z| > 2)"),
        vertical_spacing=0.09,
        row_heights=[0.4, 0.3, 0.3],
    )

    fig.add_trace(
        go.Scatter(x=decomp.index, y=decomp["trend"], mode="lines",
                   line=dict(color="#22c55e", width=2), name="trend",
                   hovertemplate="%{x|%d %b %Y}<br>Trend ₹%{y:,.2f}<extra></extra>"),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=decomp.index, y=decomp["seasonal"], mode="lines",
                   line=dict(color="#f59e0b", width=1.5), name="seasonal",
                   hovertemplate="%{x|%d %b %Y}<br>Seasonal %{y:+.2f}<extra></extra>"),
        row=2, col=1,
    )
    fig.add_trace(
        go.Scatter(x=decomp.index, y=decomp["resid"], mode="lines",
                   line=dict(color="#94a3b8", width=1), name="resid",
                   hovertemplate="%{x|%d %b %Y}<br>Resid %{y:+.2f}<extra></extra>"),
        row=3, col=1,
    )
    fig.add_hrect(y0=-2 * resid_std, y1=2 * resid_std, fillcolor="#38bdf8",
                  opacity=0.08, line_width=0, row=3, col=1)
    if not anomalies.empty:
        fig.add_trace(
            go.Scatter(
                x=anomalies.index, y=anomalies["resid"], mode="markers",
                marker=dict(color="#ef4444", size=7, symbol="circle-open", line=dict(width=2)),
                name="anomaly",
                hovertemplate="%{x|%d %b %Y}<br><b>Anomaly</b> %{y:+.2f}<extra></extra>",
            ),
            row=3, col=1,
        )

    fig.update_layout(
        template="plotly_dark",
        height=620,
        margin=dict(t=40, b=20, l=10, r=10),
        showlegend=False,
        paper_bgcolor="#0b1220",
        plot_bgcolor="#0b1220",
        hovermode="x unified",
    )
    return fig


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

st.markdown(
    '<div class="hero"><h1>TrendWise</h1>'
    '<p>NIFTY 500 · Daily · Trend / Seasonal / Residual decomposition powered by statsmodels STL</p></div>',
    unsafe_allow_html=True,
)

universe = load_universe()

top = st.columns([5, 1], vertical_alignment="bottom")
choice = top[0].selectbox(
    "Stock",
    options=universe["label"],
    index=None,
    placeholder="Search a NIFTY 500 stock (e.g. RELIANCE, TCS, INFY)…",
)
lookback = top[1].selectbox("Lookback", ["6M", "1Y", "2Y", "3Y", "Max"], index=2)
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

with st.spinner("Running STL…"):
    decomp = decompose(close)

trend_slope = float(decomp["trend"].iloc[-1] - decomp["trend"].iloc[-22])
trend_pct = trend_slope / float(decomp["trend"].iloc[-22]) * 100
resid_std = float(decomp["resid"].std())
latest_resid = float(decomp["resid"].iloc[-1])
resid_z = latest_resid / resid_std if resid_std > 0 else 0.0
seasonal_today = float(decomp["seasonal"].iloc[-1])
seasonal_amp = float(decomp["seasonal"].abs().max())

# --- header meta
st.caption(f"**{selected['name']}** · {selected['industry']} · `{selected['ticker']}` · "
           f"data through {prices.index.max().date().isoformat()}")

# --- metric cards
c = st.columns(4)
arr_sym, arr_cls = arrow(chg)
card(c[0], "Last close", f"₹{latest:,.2f}",
     f'<span class="{arr_cls}">{arr_sym} {chg:+,.2f} ({pct:+.2f}%)</span>')

t_sym, t_cls = arrow(trend_slope)
card(c[1], "Trend (21d)", f'<span class="{t_cls}">{t_sym} {trend_pct:+.2f}%</span>',
     f"₹{decomp['trend'].iloc[-22]:,.2f} → ₹{decomp['trend'].iloc[-1]:,.2f}")

r_cls = "down" if abs(resid_z) > 2 else ("up" if abs(resid_z) > 1 else "flat")
card(c[2], "Residual today", f'<span class="{r_cls}">{latest_resid:+.2f}</span>',
     f"z-score {resid_z:+.2f}σ")

card(c[3], "52-week range", f"₹{lo_52w:,.0f} – ₹{hi_52w:,.0f}",
     f"Now at <b>{pos_in_range:.0f}%</b> of range · Vol {vol_ratio:.1f}× 30d avg")

# --- plain english interpretation
trend_word = "rising" if trend_slope > 0 else ("falling" if trend_slope < 0 else "flat")
resid_word = ("an unusually large move" if abs(resid_z) > 2
              else "a noticeable move" if abs(resid_z) > 1
              else "within its normal noise band")
season_word = ("near a cyclical peak" if seasonal_today > 0.5 * seasonal_amp
               else "near a cyclical trough" if seasonal_today < -0.5 * seasonal_amp
               else "mid-cycle")

insight(
    f"<b>Trend:</b> {trend_word} at {trend_pct:+.2f}% over the last 21 trading days. "
    f"<b>Seasonal:</b> currently {season_word} in the {SEASONAL_PERIOD}-day cycle. "
    f"<b>Residual:</b> today's move is {resid_word} (z = {resid_z:+.2f})."
)

# --- charts
st.plotly_chart(price_chart(prices, window), use_container_width=True)
st.plotly_chart(stl_chart(decomp), use_container_width=True)

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
