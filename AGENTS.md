# AGENTS.md

Guidance for any AI coding assistant working in this repo (Claude Code, Codex,
Cursor, Aider, etc.). Claude-specific notes live in `CLAUDE.md`; this file
holds the tool-agnostic shared context.

## What this project is

**TrendWise** — a single-file Streamlit dashboard that runs STL
(Seasonal-Trend-Loess) decomposition on any NIFTY 500 stock and renders a
dark-themed dashboard with trend / seasonal / residual charts, anomaly
markers, and plain-English interpretation.

Deployed on **Streamlit Community Cloud** (free tier). Zero secrets, zero API
keys — designed so anyone can fork and deploy.

## Tech stack

- **Python 3.10+**
- **Streamlit** — UI framework
- **yfinance** — Yahoo Finance price data (`.NS` suffix = NSE tickers)
- **statsmodels** — `STL` decomposition (period=21, robust=True)
- **pandas / numpy** — data wrangling
- **plotly** — interactive charts (dark theme, custom styling)

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

App runs at http://localhost:8501.

## File layout

```
app.py                    # Whole app (single file, ~350 lines)
requirements.txt          # Pinned lower bounds
.streamlit/config.toml    # Dark theme + telemetry off
data/nifty500.csv         # NIFTY 500 constituents (from NSE archives)
CLAUDE.md                 # Claude Code-specific guidance
AGENTS.md                 # This file (tool-agnostic AI guide)
README.md                 # User-facing docs + deploy steps
.gitignore
```

## Architecture decisions (do not silently reverse)

1. **yfinance, not NSEpy/nsepython.** NSE actively blocks cloud IPs, so
   scraping libs work locally but break on Streamlit Cloud. yfinance is
   unofficial but works from cloud hosts.
2. **No persistent cache.** Streamlit Cloud filesystem is ephemeral. We use
   `@st.cache_data(ttl=30min)` for in-session caching only. Fetch for one
   stock is <2s; acceptable on cold start.
3. **Single file.** The app is small; splitting into modules is premature.
   Refactor only if it grows past ~500 lines or gains a real backend.
4. **Custom CSS in `app.py`.** Hero header + card metrics were an explicit
   design choice. Keep the visual polish; don't strip back to plain
   `st.metric`.
5. **STL period = 21 trading days** (≈ 1 month). If changed, update the
   subtitle text and the "How to read" expander accordingly.
6. **Empty landing state.** On first load nothing is selected (`index=None`),
   welcome card is shown, `st.stop()` prevents any fetch/render. Preserve
   this — it keeps cold-start fast and avoids picking an arbitrary default.

## Data model

- NIFTY 500 CSV: `Company Name, Industry, Symbol, Series, ISIN Code`.
- App appends `.NS` to `Symbol` to make yfinance ticker.
- Fetched frame: `Open, High, Low, Close, Volume` indexed by date.
- STL runs on `Close` only.

## Common tasks

- **Add a new universe (e.g. NIFTY MIDCAP):** drop CSV in `data/`, add a
  radio to switch universes, extend `load_universe()`.
- **Change STL period:** edit `SEASONAL_PERIOD` at the top of `app.py`; chart
  subtitles and captions read from that constant.
- **Add another metric card:** append to the `st.columns(4)` block, add the
  computation above it. Keep the card CSS class consistent.
- **Add persistence (Cloudflare R2 / S3):** would replace the `@st.cache_data`
  layer. See README for rationale on why we don't do this yet.

## Things NOT to do

- Don't add API-key-gated data sources — breaks the free-hosting story.
- Don't add a database — keeps deploy trivial.
- Don't break the `.NS` suffix convention.
- Don't commit `data/*.parquet` — regenerated at runtime, in `.gitignore`.
- Don't commit `.streamlit/secrets.toml` — reserved for deploy-time secrets.

## Deploy

Push to a public GitHub repo → https://share.streamlit.io → New app →
select repo, branch, `app.py`. First build ~2 min. No secrets needed.

## Code style

- Prefer plain functions over classes; the app has no state beyond
  Streamlit's session cache.
- Type hints on public helpers where they clarify intent.
- Keep comments sparse — code + docstrings should read cleanly.
