"""
Returns computation for the Cross-Asset Regime Monitor.

Return computation: log returns for prices, modified-duration proxies
Log returns with bond carry and unified
winsorization.

Conventions:
- Price assets  -> log returns: ln(P_t / P_{t-1})
- Yield series  -> first difference converted to bond return proxy via
                   ret = -D * dY / 100  +  y_prev / (100 * 252)
                   (price term + one day of coupon; y_prev prevents look-ahead)
- Panel forward-filled to business-day frequency before differencing.

Winsorization (unified for estimation AND backtest):
- Daily returns are capped at +/- config.RETURN_CAP (25%).
- Serves two purposes simultaneously:
  1. Stabilises GARCH/DCC parameter estimation against extreme days.
  2. Provides realistic P&L for liquid ETF/index instruments. No asset in
     the universe (SPX, EuroStoxx50, LQD, HYG, EEM, gold/oil futures)
     actually moves +/-25% in a single day. When log-return math implies
     otherwise (WTI going negative on 2020-04-20 producing a fake +230%
     bounce the next day), the math is what is unrealistic, not the market.
- Documented in the app's Limitations section.

Edge cases documented:
- WTI < $0 on 2020-04-20: prices floored at $1 before log() (log is
  undefined for x <= 0). Winsorization then caps the resulting extreme
  log-returns at +/-25%, matching the ~30% cumulative loss a USO/DBO
  holder actually experienced.
- Credit instruments (LQD, HYG) use ETF proxies; vols run 2-6pp higher
  than Project 2's bond-index equivalents.
- MSCI_EM uses EEM ETF (USD-denominated); vol runs higher than the pure
  index due to intraday ETF pricing.
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
import config


# =============================================================================
# 1. FORWARD-FILL TO BUSINESS-DAY FREQUENCY
# =============================================================================

def align_business_days(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Resample to business-day frequency and forward-fill.
    Matches Project 2 alignment: df.resample("B").last().ffill().

    Different exchanges have different holidays. Forward-filling aligns
    everything on the union business-day index so returns can be compared
    like-for-like. The cost is that a market holiday shows as a zero return,
    which slightly depresses vol.
    """
    aligned = panel.resample("B").last().ffill()
    print(f"[align_business_days] {panel.shape} -> {aligned.shape} after B-day ffill")
    return aligned


# =============================================================================
# 2. RETURNS COMPUTATION
# =============================================================================

def compute_returns(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Convert the raw panel into a 9-column daily returns DataFrame.

    Steps:
    1. Align to business-day frequency (ffill).
    2. Price columns -> log returns (with WTI negative-price handling).
    3. Yield columns -> first difference, then bond proxy = price + carry.
    4. Combine, winsorize at +/- config.RETURN_CAP.
    """
    # Step 1: forward-fill alignment
    df = align_business_days(panel)

    # Which columns are prices vs yields?
    price_cols = [c for c in config.ASSETS if c in df.columns]
    yield_cols = [c for c in config.DURATIONS if c in df.columns]

    print(f"[compute_returns] price columns:  {price_cols}")
    print(f"[compute_returns] yield columns:  {yield_cols}")

    # Step 2a: WTI negative-price handling (log is undefined for x <= 0)
    prices = df[price_cols].copy()
    if "Oil_WTI" in prices.columns:
        below = (prices["Oil_WTI"] < 1).sum()
        if below > 0:
            print(f"[compute_returns] Oil_WTI: {below} days below $1, floored")
        prices["Oil_WTI"] = prices["Oil_WTI"].clip(lower=1.0)

    # Step 2b: log returns for price columns
    log_returns = np.log(prices / prices.shift(1))

    # Step 3: yield first differences (in percentage points)
    yield_diff = df[yield_cols].diff()

    # Step 4: bond return proxies via modified duration + daily carry
    #   ret = -D * dY / 100  +  y_prev / (100 * 252)
    # The first term is price change from yield move.
    # The second term is one day of coupon income (yesterday's yield / 252 trading days).
    # Using y_prev (not y_t) avoids look-ahead: on day t we earn the coupon
    # that was known at yesterday's close.
    YIELD_SCALE = 100.0
    TRADING_DAYS = 252
    bond_proxies = pd.DataFrame(index=df.index)
    for yc in yield_cols:
        D = config.DURATIONS[yc]
        price_return = -D * yield_diff[yc] / YIELD_SCALE
        carry_return = df[yc].shift(1) / (YIELD_SCALE * TRADING_DAYS)
        bond_proxies[f"{yc}_proxy"] = price_return + carry_return
        print(f"[compute_returns] bond proxy: {yc}_proxy = -{D} * d{yc}/{YIELD_SCALE:.0f} + {yc}_prev/{YIELD_SCALE*TRADING_DAYS:.0f}")

    # Step 5: combine and drop the initial NaN row
    returns = pd.concat([log_returns, bond_proxies], axis=1).iloc[1:]

    # Step 6: winsorize at +/- config.RETURN_CAP
    # Unified treatment for estimation (GARCH/DCC/regime/ERC) AND backtest P&L.
    # Justified because no asset in the universe realistically moves +/-25%
    # in a day; the raw log-return math briefly implies otherwise on the
    # 2020-04-20 WTI negative-price event, but no ETF holder experienced
    # a -95% loss or +896% gain on those days.
    cap = config.RETURN_CAP
    n_before = (returns.abs() > cap).sum()
    returns = returns.clip(lower=-cap, upper=cap)
    total_capped = int(n_before.sum())
    if total_capped > 0:
        print(f"[compute_returns] winsorized {total_capped} extreme returns at +/-{cap:.0%}:")
        for col, n in n_before.items():
            if n > 0:
                print(f"    {col:15s} {int(n):3d} days")

    print(f"[compute_returns] final shape: {returns.shape}")
    return returns


# =============================================================================
# 3. RECONCILIATION vs PROJECT 2
# =============================================================================

# reference annualised daily vols, for the reconciliation check
# Sample: 2005-01-04 to 2026-04-24, 5559 obs, ffill business days.
# NB: reconciliation targets assume the PRE-CARRY convention Project 2 used.
# With carry added, our bond proxies will show slightly higher return but
# effectively unchanged vol - the reconciliation gate is on vol.
PROJECT2_ANN_VOLS = {
    "SP500":        0.1878,
    "EuroStoxx50":  0.2069,
    "MSCI_EM":      0.1871,
    "Gold":         0.1756,
    "Oil_WTI":      0.4118,
    # US_IG and US_HY intentionally omitted - Project 2 used bond
    # total-return indices; we use LQD/HYG ETFs. Expected to differ.
    "US_10Y_proxy": 0.0768,   # derived: 8.5 * 0.9034 / 100
}

RECON_TOLERANCE = 0.05


def reconcile_vs_project2(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Compare our annualized daily vols against Project 2 targets.
    """
    ann_factor = np.sqrt(252)
    rows = []
    for col in returns.columns:
        s = returns[col].dropna()
        if len(s) < 2:
            continue
        our_vol = s.std() * ann_factor
        target = PROJECT2_ANN_VOLS.get(col, None)
        if target is None:
            rows.append({
                "asset":       col,
                "our_ann_vol": f"{our_vol:.4f}",
                "target":      "  n/a",
                "rel_dev":     "  n/a",
                "verdict":     "no target",
            })
            continue
        rel_dev = (our_vol - target) / target
        verdict = "PASS" if abs(rel_dev) <= RECON_TOLERANCE else "FAIL"
        rows.append({
            "asset":       col,
            "our_ann_vol": f"{our_vol:.4f}",
            "target":      f"{target:.4f}",
            "rel_dev":     f"{rel_dev:+.1%}",
            "verdict":     verdict,
        })
    return pd.DataFrame(rows)