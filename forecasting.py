"""Forecasting for TrendWise.

Two families of forecasters, one return shape (ForecastResult, always in rupees):

- STL + Holt classical baseline (transparent, cheap).
- TimesFM (Google's pretrained foundation model) in four modes:
    * Raw         — TimesFM on raw close prices directly.
    * Trend       — TimesFM on the STL trend component only; seasonal-naive repeat.
    * Trend+Resid — TimesFM on trend and residual; seasonal-naive repeat.
    * Components  — TimesFM on all three STL components (trend, seasonal, residual).

The decomposed modes reuse the STL decomposition already computed for the panels
(no re-fit), and batch multiple component forecasts into a single model.forecast()
call so 1-3 components cost only one forward pass through TimesFM.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import streamlit as st
from statsmodels.tsa.holtwinters import Holt

TIMESFM_CONTEXT = 512
TIMESFM_MAX_HORIZON = 64
TIMESFM_REPO = "google/timesfm-2.5-200m-pytorch"

TimesFmMode = Literal["Raw", "Trend", "Trend+Resid", "Components"]


@dataclass
class ForecastResult:
    dates: pd.DatetimeIndex
    point: np.ndarray                # rupees
    lower: np.ndarray | None = None  # rupees (p10)
    upper: np.ndarray | None = None  # rupees (p90)
    method: str = ""


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _future_business_days(last_date: pd.Timestamp, horizon: int) -> pd.DatetimeIndex:
    return pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=horizon)


def _extend_seasonal(seasonal: pd.Series, horizon: int, period: int) -> np.ndarray:
    """Repeat the last full seasonal cycle forward (seasonal-naive)."""
    last_cycle = seasonal.iloc[-period:].to_numpy()
    reps = int(np.ceil(horizon / period))
    return np.tile(last_cycle, reps)[:horizon]


def _reconstruct_price(components_sum: np.ndarray, close: pd.Series, mode: str) -> np.ndarray:
    """Invert whichever STL transform was used, back to rupee prices.

    - Price:     price = components_sum
    - Log price: price = exp(components_sum)
    - Returns:   price = last_close * exp(cumsum(components_sum))
    """
    if mode == "Price":
        return components_sum
    if mode == "Log price":
        return np.exp(components_sum)
    last_close = float(close.iloc[-1])
    return last_close * np.exp(np.cumsum(components_sum))


# ---------------------------------------------------------------------------
# TimesFM loader (shared)
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner=False)
def load_timesfm_model():
    """Load TimesFM 2.5 once per session. Returns None if unavailable."""
    try:
        import timesfm  # noqa: WPS433

        model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(TIMESFM_REPO)
        model.compile(
            timesfm.ForecastConfig(
                max_context=TIMESFM_CONTEXT,
                max_horizon=TIMESFM_MAX_HORIZON,
                normalize_inputs=True,
                use_continuous_quantile_head=True,
                fix_quantile_crossing=True,
            )
        )
        return model
    except Exception as exc:  # noqa: BLE001 — graceful degrade is the point
        st.session_state["_timesfm_error"] = f"{type(exc).__name__}: {exc}"
        return None


def _timesfm_batch_forecast(
    model,
    series_list: list[np.ndarray],
    horizon: int,
) -> tuple[list[np.ndarray], list[np.ndarray | None]] | None:
    """One TimesFM forward pass for a list of series. Returns (points, quantile_arrays)."""
    inputs = [np.asarray(s, dtype=np.float32)[-TIMESFM_CONTEXT:] for s in series_list]
    try:
        point_batch, quantile_batch = model.forecast(horizon=int(horizon), inputs=inputs)
    except Exception as exc:  # noqa: BLE001
        st.session_state["_timesfm_error"] = f"forecast() failed: {type(exc).__name__}: {exc}"
        return None
    points = [np.asarray(p)[:horizon] for p in point_batch]
    quantiles: list[np.ndarray | None] = []
    for q in quantile_batch:
        q_arr = np.asarray(q)
        # TimesFM 2.5 returns shape (horizon, 10): [mean, p10, p20, ..., p90].
        if q_arr.ndim == 2 and q_arr.shape[1] >= 10:
            quantiles.append(q_arr[:horizon])
        else:
            quantiles.append(None)
    return points, quantiles


# ---------------------------------------------------------------------------
# STL + Holt baseline
# ---------------------------------------------------------------------------


def _extrapolate_trend_holt(trend: pd.Series, horizon: int) -> np.ndarray:
    fit = Holt(trend.astype(float).to_numpy(), damped_trend=True, initialization_method="estimated").fit(
        optimized=True,
    )
    return np.asarray(fit.forecast(horizon))


def forecast_stl_classical(
    decomp: pd.DataFrame,
    close: pd.Series,
    horizon: int,
    period: int,
    mode: str,
) -> ForecastResult | None:
    """STL trend → Holt, seasonal → naive repeat, residual → zero. Inverted to price."""
    if len(decomp) < period * 2:
        return None
    try:
        trend_future = _extrapolate_trend_holt(decomp["trend"], horizon)
        season_future = _extend_seasonal(decomp["seasonal"], horizon, period)
    except Exception:  # noqa: BLE001
        return None
    price_future = _reconstruct_price(trend_future + season_future, close, mode)
    return ForecastResult(
        dates=_future_business_days(close.index[-1], horizon),
        point=price_future,
        method="STL + Holt",
    )


# ---------------------------------------------------------------------------
# TimesFM (all modes)
# ---------------------------------------------------------------------------


@st.cache_data(ttl=60 * 30, show_spinner=False)
def forecast_timesfm(
    close: pd.Series,
    decomp: pd.DataFrame | None,
    horizon: int,
    period: int,
    stl_mode: str,
    timesfm_mode: TimesFmMode,
) -> ForecastResult | None:
    """Dispatch TimesFM across the four modes. Cached per (ticker, horizon, mode)."""
    model = load_timesfm_model()
    if model is None:
        return None

    if timesfm_mode == "Raw":
        result = _timesfm_batch_forecast(model, [close.astype(float).to_numpy()], horizon)
        if result is None:
            return None
        points, quantiles = result
        return ForecastResult(
            dates=_future_business_days(close.index[-1], horizon),
            point=points[0],
            lower=quantiles[0][:, 1] if quantiles[0] is not None else None,   # p10
            upper=quantiles[0][:, 9] if quantiles[0] is not None else None,   # p90
            method="TimesFM (raw)",
        )

    if decomp is None or len(decomp) < period * 2:
        return None

    # Decide which components go through TimesFM vs seasonal-naive vs zero.
    trend = decomp["trend"].astype(float).to_numpy()
    seasonal = decomp["seasonal"]
    residual = decomp["resid"].astype(float).to_numpy()

    series_to_forecast: list[np.ndarray] = [trend]
    slots: dict[str, int] = {"trend": 0}

    if timesfm_mode in ("Trend+Resid", "Components"):
        slots["resid"] = len(series_to_forecast)
        series_to_forecast.append(residual)
    if timesfm_mode == "Components":
        slots["seasonal"] = len(series_to_forecast)
        series_to_forecast.append(seasonal.astype(float).to_numpy())

    batch = _timesfm_batch_forecast(model, series_to_forecast, horizon)
    if batch is None:
        return None
    points, quantiles = batch

    trend_future = points[slots["trend"]]
    trend_q = quantiles[slots["trend"]]

    if "seasonal" in slots:
        seasonal_future = points[slots["seasonal"]]
    else:
        seasonal_future = _extend_seasonal(seasonal, horizon, period)

    if "resid" in slots:
        residual_future = points[slots["resid"]]
    else:
        residual_future = np.zeros(horizon)

    components_sum = trend_future + seasonal_future + residual_future
    price_future = _reconstruct_price(components_sum, close, stl_mode)

    # Uncertainty band: only trend's quantiles are meaningful (dominant component).
    # Shift trend's quantile spread by the deterministic (seasonal + residual) part,
    # then invert. Approximate but honest.
    lower = upper = None
    if trend_q is not None:
        trend_lower = trend_q[:, 1]   # p10
        trend_upper = trend_q[:, 9]   # p90
        lower = _reconstruct_price(trend_lower + seasonal_future + residual_future, close, stl_mode)
        upper = _reconstruct_price(trend_upper + seasonal_future + residual_future, close, stl_mode)

    return ForecastResult(
        dates=_future_business_days(close.index[-1], horizon),
        point=price_future,
        lower=lower,
        upper=upper,
        method=f"TimesFM ({timesfm_mode})",
    )
