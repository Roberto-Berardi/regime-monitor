"""
Data layer for the Cross-Asset Regime Monitor.

Handles: download from Yahoo Finance + FRED, merge, cache with parquet,
validate, and fall back gracefully on failure.

Built one function at a time, tested at each step.
"""
from pathlib import Path
import sys

import pandas as pd
import numpy as np
import yfinance as yf

# --- Import our project config -----------------------------------------------
# src/data.py needs to import from config.py at the project root.
# The line below adds the project root to Python's search path.
sys.path.insert(0, str(Path(__file__).parent.parent))
import config


# =============================================================================
# 1. YAHOO FINANCE DOWNLOAD (one ticker at a time)
# =============================================================================

def fetch_one_yahoo(ticker: str, start: str = None) -> pd.Series:
    """
    Download adjusted-close daily prices for ONE Yahoo Finance ticker.

    Parameters
    ----------
    ticker : str
        Yahoo Finance ticker symbol (e.g., "^GSPC" for S&P 500).
    start : str, optional
        Start date "YYYY-MM-DD". Defaults to config.START_DATE.

    Returns
    -------
    pd.Series
        Daily adjusted close prices, indexed by date.
        Empty Series if download fails.
    """
    if start is None:
        start = config.START_DATE

    print(f"[fetch_one_yahoo] downloading {ticker} from {start}...")

    try:
        df = yf.download(
            ticker,
            start=start,
            progress=False,       # no progress bar (cleaner for scripts)
            auto_adjust=True,     # use adjusted close by default
        )
    except Exception as e:
        print(f"[fetch_one_yahoo] ERROR downloading {ticker}: {e}")
        return pd.Series(dtype=float)

    if df.empty:
        print(f"[fetch_one_yahoo] WARNING: empty result for {ticker}")
        return pd.Series(dtype=float)

    # yfinance returns a DataFrame; we want the "Close" column as a named Series
    prices = df["Close"].squeeze()   # squeeze in case of a 1-col multiindex
    prices.name = ticker

    print(f"[fetch_one_yahoo] {ticker}: {len(prices)} rows, "
          f"{prices.index.min().date()} to {prices.index.max().date()}, "
          f"{prices.isna().sum()} NaN")

    return prices

# =============================================================================
# 2. YAHOO FINANCE - ALL ASSETS TOGETHER
# =============================================================================

def fetch_all_yahoo(start: str = None) -> pd.DataFrame:
    """
    Download all Yahoo Finance assets defined in config.ASSETS.

    Loops through the config.ASSETS dict, downloads each ticker one at a time
    (safer than yf.download's multi-ticker mode, which handles failures poorly),
    and combines them into one DataFrame.

    Parameters
    ----------
    start : str, optional
        Start date "YYYY-MM-DD". Defaults to config.START_DATE.

    Returns
    -------
    pd.DataFrame
        Wide-format daily prices: rows = dates, columns = asset names
        (using config.ASSETS keys, e.g., "SP500", "Gold", not tickers).
    """
    if start is None:
        start = config.START_DATE

    print(f"\n[fetch_all_yahoo] fetching {len(config.ASSETS)} assets\n")

    series_list = []
    failed = []

    for name, ticker in config.ASSETS.items():
        s = fetch_one_yahoo(ticker, start=start)
        if s.empty:
            failed.append(name)
            continue
        s.name = name   # rename from ticker (e.g., "^GSPC") to friendly ("SP500")
        series_list.append(s)

    if not series_list:
        print("[fetch_all_yahoo] ERROR: nothing downloaded")
        return pd.DataFrame()

    # Combine into a single wide DataFrame; outer join keeps all dates
    df = pd.concat(series_list, axis=1)

    print(f"\n[fetch_all_yahoo] SUMMARY")
    print(f"  shape:    {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"  dates:    {df.index.min().date()} to {df.index.max().date()}")
    print(f"  columns:  {list(df.columns)}")
    print(f"  NaN per column:")
    for col in df.columns:
        n = df[col].isna().sum()
        print(f"    {col:15s} {n:5d} NaN")
    if failed:
        print(f"  FAILED:   {failed}")

    return df

# =============================================================================
# 3. FRED DOWNLOAD (macro/rates series)
# =============================================================================

# Import the extras we need for FRED. Kept here for grouping;
# in a bigger project they'd all live at the top of the file.
import os
from functools import lru_cache
from dotenv import load_dotenv
from fredapi import Fred

# Load environment variables from .env into the process (once, at import).
# In production (Streamlit Cloud), .env doesn't exist and the key comes from
# Streamlit's secrets manager instead — we handle that in Phase 10.
load_dotenv()


@lru_cache(maxsize=1)
def _get_fred_client() -> Fred:
    """
    Create (or reuse) a FRED API client.

    lru_cache means this runs ONCE per Python session. Every subsequent
    call returns the same client — no re-authentication overhead.
    """
    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        raise RuntimeError(
            "FRED_API_KEY not found. Check that .env exists in the project "
            "root and contains 'FRED_API_KEY=your_key_here'."
        )
    return Fred(api_key=api_key)


def fetch_one_fred(series_id: str, start: str = None) -> pd.Series:
    """
    Download ONE FRED series (e.g., "DGS10" for 10-year Treasury yield).

    Parameters
    ----------
    series_id : str
        FRED series code (see https://fred.stlouisfed.org).
    start : str, optional
        Start date "YYYY-MM-DD". Defaults to config.START_DATE.

    Returns
    -------
    pd.Series
        Daily series indexed by date; name = series_id.
        Empty Series if download fails.
    """
    if start is None:
        start = config.START_DATE

    print(f"[fetch_one_fred] downloading {series_id} from {start}...")

    try:
        client = _get_fred_client()
        s = client.get_series(series_id, observation_start=start)
    except Exception as e:
        print(f"[fetch_one_fred] ERROR downloading {series_id}: {e}")
        return pd.Series(dtype=float)

    if s.empty:
        print(f"[fetch_one_fred] WARNING: empty result for {series_id}")
        return pd.Series(dtype=float)

    s.name = series_id
    print(f"[fetch_one_fred] {series_id}: {len(s)} rows, "
          f"{s.index.min().date()} to {s.index.max().date()}, "
          f"{s.isna().sum()} NaN")

    return s


def fetch_all_fred(start: str = None) -> pd.DataFrame:
    """
    Download all FRED series defined in config.FRED_SERIES.
    """
    if start is None:
        start = config.START_DATE

    print(f"\n[fetch_all_fred] fetching {len(config.FRED_SERIES)} series\n")

    series_list = []
    failed = []

    for name, sid in config.FRED_SERIES.items():
        s = fetch_one_fred(sid, start=start)
        if s.empty:
            failed.append(name)
            continue
        s.name = name   # rename from FRED code to friendly key
        series_list.append(s)

    if not series_list:
        print("[fetch_all_fred] ERROR: nothing downloaded")
        return pd.DataFrame()

    df = pd.concat(series_list, axis=1)

    print(f"\n[fetch_all_fred] SUMMARY")
    print(f"  shape:    {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"  dates:    {df.index.min().date()} to {df.index.max().date()}")
    print(f"  columns:  {list(df.columns)}")
    print(f"  NaN per column:")
    for col in df.columns:
        n = df[col].isna().sum()
        print(f"    {col:15s} {n:5d} NaN")
    if failed:
        print(f"  FAILED:   {failed}")
        core = set(config.DURATIONS) | {"RF_RATE"}
        missing_core = sorted(core.intersection(failed))
        if missing_core:
            raise RuntimeError(
                f"core FRED series failed to download: {missing_core}. "
                "Refusing to build a panel with a missing asset - the "
                "portfolio would silently run short one leg."
            )

    return df

# =============================================================================
# 4. MERGE YAHOO + FRED INTO ONE PANEL
# =============================================================================

def fetch_all(start: str = None) -> pd.DataFrame:
    """
    Download all Yahoo and FRED data and merge into one wide DataFrame.

    - Yahoo columns: raw adjusted-close prices.
    - FRED columns: yields (%) and credit spreads (%).

    Returns
    -------
    pd.DataFrame
        Rows = business days (union of Yahoo + FRED indices).
        Columns = 7 Yahoo assets + 6 FRED series = 13 total.
        NaN patterns are expected: different exchanges have different holidays,
        Yahoo doesn't publish yields, FRED doesn't publish equity prices.
    """
    print("\n" + "="*70)
    print("[fetch_all] downloading full data panel")
    print("="*70)

    yahoo_df = fetch_all_yahoo(start=start)
    fred_df  = fetch_all_fred(start=start)

    if yahoo_df.empty and fred_df.empty:
        print("[fetch_all] ERROR: both Yahoo and FRED failed")
        return pd.DataFrame()

    if yahoo_df.empty:
        print("[fetch_all] WARNING: Yahoo empty, using FRED only")
        return fred_df
    if fred_df.empty:
        print("[fetch_all] WARNING: FRED empty, using Yahoo only")
        return yahoo_df

    # Outer join preserves every date in either source; NaN pattern is expected.
    df = yahoo_df.join(fred_df, how="outer")

    print(f"\n[fetch_all] MERGED PANEL")
    print(f"  shape:    {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"  dates:    {df.index.min().date()} to {df.index.max().date()}")
    print(f"  columns:  {list(df.columns)}")

    return df

# =============================================================================
# 5. CACHE, VALIDATION, GRACEFUL FALLBACK
# =============================================================================
# The rule: get_data() NEVER raises. If fresh download fails validation,
# we return the last-good cache. If nothing works, we return an empty
# DataFrame and log why - the app renders "data unavailable" but stays alive.

from datetime import datetime, timezone
import json


def save_cache(df: pd.DataFrame) -> None:
    """
    Save the panel to a parquet file with a JSON sidecar holding the timestamp.
    """
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)

    df.to_parquet(config.CACHE_FILE)

    meta = {
        "saved_utc":   datetime.now(timezone.utc).isoformat(),
        "rows":        int(df.shape[0]),
        "cols":        int(df.shape[1]),
        "min_date":    str(df.index.min().date()),
        "max_date":    str(df.index.max().date()),
        "columns":     list(df.columns),
    }
    meta_file = config.CACHE_FILE.with_suffix(".json")
    meta_file.write_text(json.dumps(meta, indent=2))

    print(f"[save_cache] wrote {config.CACHE_FILE.name}: "
          f"{df.shape[0]} rows, latest {df.index.max().date()}")


def load_cache() -> tuple[pd.DataFrame, dict]:
    """
    Load the cached panel and its metadata. Returns (empty df, empty dict)
    if no cache exists.
    """
    if not config.CACHE_FILE.exists():
        print("[load_cache] no cache file found")
        return pd.DataFrame(), {}

    df = pd.read_parquet(config.CACHE_FILE)

    meta_file = config.CACHE_FILE.with_suffix(".json")
    meta = json.loads(meta_file.read_text()) if meta_file.exists() else {}

    print(f"[load_cache] loaded {df.shape[0]} rows, "
          f"saved {meta.get('saved_utc', 'unknown')}")
    return df, meta


def validate(df: pd.DataFrame) -> tuple[bool, list[str]]:
    """
    Sanity checks. All must pass or we reject the panel and keep the cache.
    Returns (passed, list_of_issues).
    """
    issues = []

    if df.empty:
        issues.append("panel is empty")
        return False, issues

    # (a) at least 8 of 9 core columns present
    #     (SP500, EuroStoxx50, MSCI_EM, Gold, Oil_WTI, US_IG, US_HY, US_2Y, US_10Y)
    core_cols = list(config.ASSETS.keys()) + list(config.DURATIONS.keys())
    absent = [c for c in core_cols if c not in df.columns]
    if absent:
        issues.append(f"core columns missing: {absent}")

    # (b) no core column entirely NaN
    for col in core_cols:
        if col in df.columns and df[col].isna().all():
            issues.append(f"{col} is entirely NaN")

    # (c) most recent date within 5 business days of today
    latest = df.index.max()
    days_stale = pd.bdate_range(latest, pd.Timestamp.today()).size - 1
    if days_stale > 5:
        issues.append(f"latest data is {days_stale} business days old ({latest.date()})")

    # (d) ticker-glitch detector: check ONLY the most recent 30 business days.
    #     Rationale: historical extreme events (e.g., WTI going negative on
    #     2020-04-20 during COVID) are real and should not invalidate the
    #     panel. A live ticker glitch, by contrast, would appear in the
    #     most recent data. Also skip through zero-crossings (pct_change is
    #     mathematically meaningless when a series crosses zero).
    RECENT_WINDOW_DAYS = 30
    GLITCH_THRESHOLD   = 0.25

    for col in config.ASSETS:
        if col not in df.columns:
            continue
        s = df[col].dropna().tail(RECENT_WINDOW_DAYS)
        if len(s) < 2:
            continue
        # skip pairs where the series changes sign (log-return ill-defined)
        prev = s.shift(1)
        same_sign = (s * prev) > 0
        if not same_sign.any():
            continue
        # only compute returns on same-sign consecutive pairs
        rets = (s[same_sign] / prev[same_sign] - 1).abs()
        if len(rets) == 0:
            continue
        max_move = rets.max()
        if max_move > GLITCH_THRESHOLD:
            worst_date = rets.idxmax().date()
            issues.append(
                f"{col} moved {max_move:.1%} on {worst_date} "
                f"(within last {RECENT_WINDOW_DAYS} days) - possible ticker glitch"
            )

    passed = len(issues) == 0
    if passed:
        print("[validate] all checks passed")
    else:
        print("[validate] FAILED:")
        for i in issues:
            print(f"    - {i}")

    return passed, issues


def get_data(force_refresh: bool = False) -> tuple[pd.DataFrame, dict]:
    """
    Master entry point. Try fresh download, validate, then EITHER save fresh
    OR fall back to cache. This function NEVER raises.

    Parameters
    ----------
    force_refresh : bool
        If True, skip the cache-load-on-failure step (used in tests).

    Returns
    -------
    (df, meta) : DataFrame + metadata dict.
        meta contains 'source' key: "fresh" or "cache" or "empty".
    """
    print("\n" + "="*70)
    print("[get_data] attempting fresh download")
    print("="*70)

    try:
        fresh = fetch_all()
    except Exception as e:
        print(f"[get_data] fresh download raised: {e}")
        fresh = pd.DataFrame()

    passed, issues = validate(fresh)

    if passed:
        save_cache(fresh)
        meta = {
            "source":    "fresh",
            "saved_utc": datetime.now(timezone.utc).isoformat(),
            "issues":    [],
        }
        return fresh, meta

    print("[get_data] fresh data rejected, attempting cache fallback")

    if force_refresh:
        print("[get_data] force_refresh=True, not loading cache")
        return pd.DataFrame(), {"source": "empty", "issues": issues}

    cached, cache_meta = load_cache()
    if cached.empty:
        print("[get_data] no cache available either - returning empty")
        return pd.DataFrame(), {"source": "empty", "issues": issues}

    print(f"[get_data] serving cache from {cache_meta.get('saved_utc', 'unknown')}")
    cache_meta["source"] = "cache"
    cache_meta["issues"] = issues
    return cached, cache_meta

# =============================================================================
# 6. SF FED DAILY NEWS SENTIMENT INDEX (reference indicator, not a signal)
# =============================================================================
# The DNSI is a daily time series of US economic-news sentiment computed by
# lexical analysis of 24 major US newspapers. History back to 1980. Updated
# weekly by the SF Fed. Reference: Buckman, Shapiro, Sudhof, Wilson (2020).
#
# We do NOT use this as a portfolio input. It's a dashboard context indicator:
# "the news environment right now vs 45 years of history."

DNSI_URL       = "https://www.frbsf.org/wp-content/uploads/news-sentiment-chart-1.csv"
DNSI_CACHE     = config.DATA_DIR / "dnsi_cache.parquet"


def fetch_dnsi() -> pd.Series:
    """
    Download the SF Fed Daily News Sentiment Index. Cache to parquet so
    subsequent calls don't re-hit the SF Fed server.

    Returns
    -------
    pd.Series indexed by date, values are the smoothed daily sentiment
    score (typically between -0.7 and +0.4).
    Falls back to cached copy on network failure.
    """
    import requests

    try:
        print(f"[fetch_dnsi] downloading from SF Fed...")
        r = requests.get(DNSI_URL, timeout=15)
        r.raise_for_status()
        # SF Fed CSV: columns are 'date' and one value column with a
        # dynamic name (e.g. 'News Sentiment' or similar - varies).
        from io import StringIO
        df = pd.read_csv(StringIO(r.text))
        # Robust column detection: first col is date, second is value.
        date_col = df.columns[0]
        val_col  = df.columns[1]
        df[date_col] = pd.to_datetime(df[date_col])
        s = df.set_index(date_col)[val_col]
        s.name = "DNSI"
        s = s.sort_index()

        # Cache
        config.DATA_DIR.mkdir(parents=True, exist_ok=True)
        s.to_frame().to_parquet(DNSI_CACHE)
        print(f"[fetch_dnsi] downloaded {len(s)} obs, {s.index.min().date()} to {s.index.max().date()}")
        print(f"[fetch_dnsi] cached to {DNSI_CACHE.name}")
        return s

    except Exception as e:
        print(f"[fetch_dnsi] download failed: {e}")
        if DNSI_CACHE.exists():
            print(f"[fetch_dnsi] falling back to cache")
            s = pd.read_parquet(DNSI_CACHE).squeeze()
            s.name = "DNSI"
            return s
        raise RuntimeError("DNSI download failed and no cache available") from e


def dnsi_summary(dnsi: pd.Series = None) -> dict:
    """
    Compact summary for the dashboard.
    - current value + date
    - percentile within full history
    - recent 1-month vs 12-month change
    - all-time min/max reference points
    """
    if dnsi is None:
        dnsi = fetch_dnsi()

    latest_date  = dnsi.index[-1]
    latest_value = float(dnsi.iloc[-1])
    pct_rank     = float((dnsi <= latest_value).mean() * 100)

    # Recent momentum: last 21 days (1 month) vs 252 days (1 year)
    recent_1m  = float(dnsi.tail(21).mean())
    recent_12m = float(dnsi.tail(252).mean())

    hist_min = float(dnsi.min())
    hist_max = float(dnsi.max())
    hist_min_date = dnsi.idxmin()
    hist_max_date = dnsi.idxmax()
    hist_mean = float(dnsi.mean())

    return {
        "latest_date":   latest_date.date().isoformat(),
        "latest_value":  latest_value,
        "percentile":    pct_rank,
        "recent_1m":     recent_1m,
        "recent_12m":    recent_12m,
        "delta_1m_12m":  recent_1m - recent_12m,
        "hist_min":      hist_min,
        "hist_min_date": hist_min_date.date().isoformat(),
        "hist_max":      hist_max,
        "hist_max_date": hist_max_date.date().isoformat(),
        "hist_mean":     hist_mean,
    }


