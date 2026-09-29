# CLAUDE.md

Guidance for Claude Code when working in this repo.

## What this is

Single-file Streamlit dashboard (`app.py`) that:
1. Loads NIFTY 500 constituent list from `data/nifty500.csv`.
2. Fetches daily OHLCV for the selected ticker via `yfinance` (Yahoo, `.NS` suffix).
3. Runs `statsmodels.tsa.seasonal.STL` (period=21, robust=True) on Close.
4. Renders a dark-themed dashboard: hero header, metric cards, price chart with
   MAs + 52-week band, STL panels with anomaly markers, plain-English captions.

Designed to deploy free on **Streamlit Community Cloud** — no secrets, no keys.

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Architecture decisions (do not silently reverse)

- **yfinance, not NSEpy/nsepython** — NSE blocks cloud IPs, so scraping libs break
  on Streamlit Cloud. yfinance is unofficial but works from cloud hosts.
- **No persistent cache** — Streamlit Cloud filesystem is ephemeral. We rely on
  `@st.cache_data(ttl=30min)` for in-session caching. For one stock at a time
  the fetch is <2s so this is acceptable.
- **Single file (with one carveout)** — `app.py` holds the UI. Forecasting lives
  in `forecasting.py` because it owns a heavyweight model lifecycle (TimesFM,
  ~200M params) that doesn't belong mixed into Streamlit render code. Any further
  splitting should still clear the ~500 LOC bar per file.
- **Forecasting: TimesFM (headline) + STL+Holt (baseline)** — TimesFM is loaded
  once per session via `@st.cache_resource` and gracefully falls back (warning
  banner, TimesFM controls hidden) if the model can't load — the app must keep
  working on the baseline alone. The STL baseline extrapolates trend with Holt +
  seasonal-naive repeat, and must **invert whichever STL transform** was chosen
  (Price / Log price / Returns) to end up on the price chart in rupees.
- **No LSTM/Time-LLM in the app** — training loops and 7B checkpoints don't fit
  the free-tier deploy story. STL+LSTM lives in `notebooks/` as research only.
- **Custom CSS in `app.py`** — the user picked "custom CSS + hero + cards" over
  plain defaults. Keep the visual polish; don't strip it back to `st.metric`.
- **STL period = 21 trading days** (~1 month). If changing, update the caption
  text and the "How to read" expander accordingly.

## Files

- `app.py` — the whole app.
- `data/nifty500.csv` — NIFTY 500 constituents from NSE archives (columns:
  Company Name, Industry, Symbol, Series, ISIN Code). Symbols are bare (no
  `.NS`); the app appends the suffix.
- `.streamlit/config.toml` — dark theme + disable telemetry.
- `requirements.txt` — pinned lower bounds only.

## Common tasks

- **Add a new ticker universe (e.g. NIFTY MIDCAP):** drop CSV in `data/`, add a
  radio to switch, extend `load_universe`.
- **Change STL period:** edit `SEASONAL_PERIOD` at the top of `app.py`; the
  chart subtitle and captions read from that constant.
- **Add persistence (Cloudflare R2 etc.):** would replace the fetch cache. See
  README for why we don't do this yet.

## Things NOT to do

- Don't add API-key-gated data sources — breaks free hosting story.
- Don't add a database — keeps deploy trivial. Ephemeral cache is fine.
- Don't break the `.NS` suffix convention; NSE symbols only work with it on
  Yahoo.
