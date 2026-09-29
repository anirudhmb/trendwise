# TrendWise · NIFTY 500 STL Dashboard

Streamlit dashboard that runs [STL decomposition](https://otexts.com/fpp3/stl.html)
(Trend / Seasonal / Residual) on any NIFTY 500 stock, using free Yahoo Finance data.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open http://localhost:8501.

## Deploy free on Streamlit Community Cloud

1. Push this repo to GitHub (public).
2. Go to https://share.streamlit.io → **New app**.
3. Select repo, branch `main`, main file `app.py`.
4. Deploy. First build takes ~2 minutes.

No secrets, no API keys required — yfinance works out of the box.

## Data

- **Source:** Yahoo Finance via `yfinance` (`.NS` suffix = NSE).
- **Universe:** NIFTY 500 constituent list in `data/nifty500.csv` (from NSE archives).
- **Caching:** `@st.cache_data(ttl=30min)` keeps repeat requests fast within a session.
  On cold start after redeploy, data is re-fetched (fast for one stock at a time).

## What the dashboard shows

- Hero header + 4 metric cards: last close, 21-day trend %, today's residual (z-score),
  52-week range with position + volume ratio.
- Plain-English interpretation of trend / seasonal / residual state.
- Price chart with 50 & 200-day moving averages and 52-week high/low band.
- STL decomposition (trend, seasonal, residual) with anomaly markers where |z| > 2.
- "How to read" expander for viewers new to STL.

## Layout

```
app.py                 # Streamlit app (single file)
requirements.txt
.streamlit/config.toml # dark theme
data/nifty500.csv      # NIFTY 500 constituents
CLAUDE.md              # Notes for Claude Code sessions
README.md
```
