"""
Point-in-time macro data via ALFRED (Archival FRED).

The problem this module solves:
    Federal Reserve macro data (payrolls, industrial production, GDP, CPI)
    is revised - often materially - after first release. A signal built on
    today's FRED value for, say, January 2024 uses a number that was
    revised twice since Feb 2024 when the market actually got the first
    print. Backtests using revised data therefore contain look-ahead:
    the model implicitly "knows" what got revised in.

The solution:
    ALFRED gives point-in-time snapshots. get_first_release(series_id)
    returns each observation as it was FIRST PUBLISHED, which is what a
    live trader would have had. Backtests using first-release data are
    honest.

We keep both series available for direct side-by-side comparison, so
the revision-alpha gap can be quantified.
"""
from pathlib import Path
import sys

import pandas as pd
from fredapi import Fred

sys.path.insert(0, str(Path(__file__).parent.parent))
import config


# =============================================================================
# 1. FRED CLIENT (reuses same API key as data.py)
# =============================================================================

def _get_fred_client() -> Fred:
    """
    Instantiate the fredapi client with the API key from environment.
    Same key we use for FRED - ALFRED endpoints are gated by the same auth.
    """
    from dotenv import load_dotenv
    import os
    load_dotenv(config.PROJECT_ROOT / ".env")
    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        raise RuntimeError("FRED_API_KEY missing from .env - see data.py setup")
    return Fred(api_key=api_key)


# =============================================================================
# 2. FETCH FIRST-RELEASE SERIES (POINT-IN-TIME, "HONEST")
# =============================================================================

def get_first_release(series_id: str, start: str = None) -> pd.Series:
    """
    Return the FIRST-RELEASE value of each observation. This is what a live
    trader/model would have seen at the release date; no later revisions
    are included.

    Parameters
    ----------
    series_id : str
        FRED series ID (e.g. "PAYEMS", "INDPRO").
    start : str
        ISO date string. Defaults to config.START_DATE.

    Returns
    -------
    pd.Series
        Indexed by the observation date (not the release date - keeps
        alignment consistent with revised-series comparison).
    """
    if start is None:
        start = config.START_DATE
    fred = _get_fred_client()
    print(f"[get_first_release] fetching {series_id} first-release values from {start}")
    s = fred.get_series_first_release(series_id)
    s = s[s.index >= pd.Timestamp(start)]
    s.name = f"{series_id}_first"
    print(f"[get_first_release] {series_id}: {len(s)} obs, {s.index.min().date()} to {s.index.max().date()}")
    return s


# =============================================================================
# 3. FETCH REVISED SERIES (CURRENT VALUES, "CHEATING")
# =============================================================================

def get_revised(series_id: str, start: str = None) -> pd.Series:
    """
    Return the current (fully revised) values of the series. This is what
    a naive researcher would use, and it contains look-ahead: revisions
    happened AFTER each observation date but are baked into the series.
    """
    if start is None:
        start = config.START_DATE
    fred = _get_fred_client()
    print(f"[get_revised] fetching {series_id} revised values from {start}")
    s = fred.get_series(series_id, observation_start=start)
    s.name = f"{series_id}_revised"
    print(f"[get_revised] {series_id}: {len(s)} obs, {s.index.min().date()} to {s.index.max().date()}")
    return s


# =============================================================================
# 4. QUANTIFY REVISION MAGNITUDE
# =============================================================================

def revision_summary(series_id: str, start: str = None) -> dict:
    """
    Quantify how much a series gets revised by. Useful for justifying which
    series are worth studying vs which are essentially unrevised.

    Returns dict with:
      - n_obs                : shared sample size
      - mean_abs_revision    : mean |revised - first_release|
      - median_abs_revision  : median |revised - first_release|
      - max_abs_revision     : largest single revision
      - corr                 : correlation of the two series (should be ~1)
    """
    first   = get_first_release(series_id, start=start)
    revised = get_revised(series_id, start=start)
    common = first.index.intersection(revised.index)
    if len(common) == 0:
        return {"error": "no overlapping dates"}
    a = first.loc[common]
    b = revised.loc[common]
    diff = (b - a).abs()
    return {
        "series":              series_id,
        "n_obs":               len(common),
        "sample":              f"{common.min().date()} to {common.max().date()}",
        "mean_abs_revision":   float(diff.mean()),
        "median_abs_revision": float(diff.median()),
        "max_abs_revision":    float(diff.max()),
        "corr":                float(a.corr(b)),
    }

# =============================================================================
# 5. MACRO MOMENTUM SIGNAL: 3M vs 12M PAYEMS
# =============================================================================

def payems_momentum_signal(payems: pd.Series) -> pd.Series:
    """
    Classic macro-momentum signal on payrolls level.

    +1  when 3-month moving average of level > 12-month moving average
        (economy accelerating)
    -1  when 3M MA <= 12M MA (economy decelerating)

    Signal is generated at each MONTHLY observation date. Downstream
    resampling to daily/weekly is a forward-fill.
    """
    ma_3  = payems.rolling(window=3,  min_periods=3 ).mean()
    ma_12 = payems.rolling(window=12, min_periods=12).mean()
    signal = pd.Series(0, index=payems.index, dtype=float)
    signal[(ma_3 >  ma_12)] = 1.0
    signal[(ma_3 <= ma_12)] = -1.0
    # Drop the initial NaN window
    signal = signal.iloc[12:]
    # Align to month-end so index matches SPX month-end resample downstream.
    # PAYEMS observations are dated at month-start (labor-stats convention);
    # month-end representation is the standard finance convention.
    signal.index = signal.index + pd.offsets.MonthEnd(0)
    return signal


# =============================================================================
# 6. LONG-CASH SPX BACKTEST GIVEN A SIGNAL
# =============================================================================

def long_cash_backtest(signal_monthly: pd.Series,
                       spx_prices: pd.Series,
                       rf_annual_pct: float = 4.0) -> dict:
    """
    Simple long/cash SPX backtest.
    - When signal +1: hold 100% SPX for the next month.
    - When signal -1: hold 100% cash, earning rf_annual_pct annualized.
    Signal is LAGGED one month to prevent look-ahead (signal at month-end t
    determines position for t -> t+1).

    Returns: annualized return, vol, Sharpe.
    """
    import numpy as np

    # Align SPX to month-end frequency, compute monthly log returns
    spx_monthly = spx_prices.resample("ME").last()
    spx_ret_monthly = np.log(spx_monthly / spx_monthly.shift(1))
    # Monthly cash return from annualized rate
    rf_monthly = np.log(1 + rf_annual_pct / 100) / 12

    # Lag signal by one month (weight-decision-t applied to return t+1)
    signal_lagged = signal_monthly.shift(1)

    # Align on common index
    common = spx_ret_monthly.index.intersection(signal_lagged.index)
    ret_spx = spx_ret_monthly.loc[common]
    sig     = signal_lagged.loc[common]

    # Portfolio return: SPX when signal +1, cash when signal -1
    port_ret = pd.Series(index=common, dtype=float)
    port_ret[sig == 1.0]  = ret_spx[sig == 1.0]
    port_ret[sig == -1.0] = rf_monthly
    port_ret = port_ret.dropna()

    # Stats (log returns, annualized)
    ann_ret = port_ret.mean() * 12
    ann_vol = port_ret.std() * (12 ** 0.5)
    sharpe  = ann_ret / ann_vol if ann_vol > 0 else float("nan")

    return {
        "ann_return": float(ann_ret),
        "ann_vol":    float(ann_vol),
        "sharpe":     float(sharpe),
        "n_obs":      len(port_ret),
        "pct_long":   float((sig == 1.0).mean() * 100),
        "returns":    port_ret,
    }