"""
Time-series momentum signals for the Cross-Asset Regime Monitor.

Pre-committed specification (see config.py):
- Primary signal: 12-1 month time-series momentum
    return from (t - 252 trading days) to (t - 21)
    Moskowitz, Ooi & Pedersen (2012) - the academic benchmark.
- Confirmation: price above/below 200-day moving average
    Faber (2007) - the practitioner benchmark.

Combined signal:
    +1  if both agree positive (trend confirmed up)
    -1  if both agree negative (trend confirmed down)
     0  otherwise (uncertain / conflicting)

NO parameter search anywhere in this module. The lookbacks are pre-committed
in config.py and defended as academic/industry standards. Any robustness
checks live in a clearly-labeled appendix.
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
import config


# =============================================================================
# 1. INDIVIDUAL SIGNALS
# =============================================================================

def momentum_12_1(prices: pd.DataFrame) -> pd.DataFrame:
    """
    12-1 month time-series momentum: return from t-lookback to t-skip.
    Skip the most recent month to avoid short-term reversal contamination.

    Signal value: sign of the return (+1, -1, or 0).
    """
    lookback = config.MOM_LOOKBACK_DAYS   # 252
    skip     = config.MOM_SKIP_DAYS       # 21

    # Return from (t - lookback) to (t - skip); shift by skip to align to t
    trailing_return = (prices.shift(skip) / prices.shift(lookback)) - 1
    signal = np.sign(trailing_return)  # +1, -1, 0
    signal.name = "mom_12_1"
    return signal


def ma_200_signal(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Sign of (price - 200-day moving average).
    +1 = price above MA (trend up)
    -1 = price below MA (trend down)
    """
    window = config.MA_WINDOW_DAYS  # 200
    ma = prices.rolling(window=window, min_periods=window).mean()
    signal = np.sign(prices - ma)
    signal.name = "ma_200"
    return signal


# =============================================================================
# 2. COMBINED SIGNAL (agreement gate)
# =============================================================================

def combined_signal(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Combine 12-1M and 200-day MA signals.
    +1 only when both agree positive.
    -1 only when both agree negative.
     0 otherwise.

    Rationale: reduces false signals in choppy markets; both indicators
    must confirm before we tilt. Trade-off: fewer trades, but higher
    conviction per trade.
    """
    mom = momentum_12_1(prices)
    ma  = ma_200_signal(prices)

    combined = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    combined[(mom == 1)  & (ma == 1)]  = 1.0
    combined[(mom == -1) & (ma == -1)] = -1.0
    return combined


# =============================================================================
# 3. WRAPPER: BUILD SIGNAL PANEL FOR CONFIG.ASSETS
# =============================================================================

def build_signal_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Compute combined momentum signals for all price-based assets in the
    raw data panel. Bonds (yield-based) get zero signal - momentum on
    yield changes is not a meaningful concept.

    Parameters
    ----------
    panel : pd.DataFrame
        Raw wide panel from data.get_data() (contains prices AND yields).

    Returns
    -------
    pd.DataFrame
        Rows = business days, columns = ALL asset labels used downstream
        (7 price assets + 2 bond proxies).
        Bond-proxy columns are always 0 (no momentum on rates).
    """
    price_cols = [c for c in config.ASSETS if c in panel.columns]
    prices = panel[price_cols].ffill()

    signals = combined_signal(prices)

    # Add zero-signal columns for bond proxies so the shape matches ERC
    for yc in config.DURATIONS:
        signals[f"{yc}_proxy"] = 0.0

    return signals

# =============================================================================
# 4. SIGNAL PANEL WITH BOND RETURN MOMENTUM
# =============================================================================
# Used by cross-asset trend strategies (Strategy B) which need momentum
# signals on ALL 9 assets, not just the 7 price-based ones. Bonds are
# handled via momentum on their cumulative-return synthetic price series
# (standard practice at CTAs like AQR Managed Futures, Man AHL).

def build_signal_panel_full(panel: pd.DataFrame, bond_returns: pd.DataFrame) -> pd.DataFrame:
    """
    Full-universe momentum signal panel, 12-1M + 200DMA on all 9 assets.
    - Price assets: momentum on price levels (as in build_signal_panel).
    - Bond assets:  momentum on cumulative-return synthetic prices, so
                    a trend in bond total returns produces a signal even
                    though the underlying yields are stationary.

    Both signals use the same combined_signal() logic - agreement gate.
    """
    price_cols = [c for c in config.ASSETS if c in panel.columns]
    prices = panel[price_cols].ffill()
    signals_price = combined_signal(prices)

    # Synthetic bond prices: (1 + r).cumprod() - so momentum on these =
    # momentum on cumulative bond returns.
    bond_cols = [c for c in bond_returns.columns if c.endswith("_proxy")]
    synth_prices = (1.0 + bond_returns[bond_cols].fillna(0.0)).cumprod()
    signals_bond = combined_signal(synth_prices)

    signals = pd.concat([signals_price, signals_bond], axis=1)
    return signals