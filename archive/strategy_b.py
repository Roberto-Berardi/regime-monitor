"""
Strategy B — Cross-Asset Trend with Aggregate Vol Targeting.

Pre-committed spec (see config.py):
- Universe: all 9 assets (equities, rates, credit, commodities)
- Per-asset sizing: raw_w_i = per_asset_vol_budget / sigma_i_63d when signal=+1
                    (long-only; signal <=0 -> weight=0)
                    capped at max_per_asset (25%)
- Aggregate vol targeting: L = clip(vol_target / port_realized_vol, 0.5, 1.25)
- Regime gate: leverage cap = 1.0x when P(high-corr) > 0.70
- Financing cost: 50bp/yr on borrowed portion above 1.0x

Design philosophy:
    A genuine cross-asset trend-follower in the AQR Managed Futures /
    Man AHL / Winton mold. Long-only for defensibility (shorting duration
    proxies via ETF costs would need extra infrastructure). Sizing is
    per-asset-inverse-vol so high-vol assets (Oil) get small weights and
    low-vol assets (US_2Y) get larger ones, then aggregate leverage
    scales to hit portfolio vol target.

This is genuinely different from Tilted:
    - Tilted: ERC anchor + small momentum tilts, 9 assets, weekly. Conservative.
    - Strategy B: signal-driven positions on 9 assets, aggressively vol-scaled,
                  regime-gated leverage. Higher octane.
Different clients. Different pitch.
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
import config


# =============================================================================
# 1. PER-ASSET REALIZED VOL (63-day rolling)
# =============================================================================

def per_asset_realized_vol(daily_returns: pd.DataFrame, lookback: int = None) -> pd.DataFrame:
    """
    63-day rolling annualized vol for each asset column.
    Uses trailing std * sqrt(252). NaN in first `lookback` rows.
    """
    if lookback is None:
        lookback = config.STRAT_B_VOL_LOOKBACK_DAYS
    return daily_returns.rolling(window=lookback, min_periods=lookback).std() * np.sqrt(252)


# =============================================================================
# 2. RAW PER-ASSET WEIGHTS FROM SIGNAL + VOL BUDGET
# =============================================================================

def signal_scaled_raw_weights(signals_weekly: pd.DataFrame,
                              vols_weekly: pd.DataFrame) -> pd.DataFrame:
    """
    Long-only per-asset weights from signal-driven vol targeting.

    For each (t, asset i):
      - signal = +1: w_i = per_asset_vol_budget / sigma_i_t, capped at max_per_asset
      - signal <= 0: w_i = 0

    Inverse-vol scaling ensures each active asset contributes ~equal vol
    to the raw portfolio; the aggregate vol scaler then adjusts overall
    leverage.
    """
    budget    = config.STRAT_B_PER_ASSET_VOL_BUDGET
    max_w     = config.STRAT_B_MAX_PER_ASSET_WEIGHT

    W = pd.DataFrame(0.0, index=signals_weekly.index, columns=signals_weekly.columns)
    long_mask = (signals_weekly == 1)
    # Where long: weight = vol budget / asset vol (in decimal, both annualized)
    # Guard against divide-by-zero on assets with tiny recent vol (US_2Y in calm times)
    safe_vols = vols_weekly.clip(lower=0.001)  # min 0.1% ann vol to avoid explosive weights
    W[long_mask] = (budget / safe_vols)[long_mask]
    W = W.clip(upper=max_w)
    return W


# =============================================================================
# 3. PORTFOLIO REALIZED VOL UNDER CURRENT WEIGHTS
# =============================================================================

def portfolio_vol_under_weights(daily_returns: pd.DataFrame,
                                weights_weekly: pd.DataFrame,
                                lookback: int = None) -> pd.Series:
    """
    Estimate the portfolio realized vol if the current weights had been
    held constant over the trailing lookback period.

    At each Friday t:
      port_ret_daily = sum_i(w_i(t) * r_i(day)) over past `lookback` days
      port_vol_t     = std(port_ret_daily) * sqrt(252)

    This is a common approximation used by real vol-targeted trend funds -
    it treats current weights as if they'd been held historically to
    estimate what "our current portfolio's vol would be."
    """
    if lookback is None:
        lookback = config.STRAT_B_VOL_LOOKBACK_DAYS

    port_vols = pd.Series(index=weights_weekly.index, dtype=float, name="port_vol_agg")
    for t in weights_weekly.index:
        w_t = weights_weekly.loc[t]
        # Slice past `lookback` daily returns up to (and including) t
        window = daily_returns.loc[:t].tail(lookback)
        if len(window) < lookback:
            continue
        # Portfolio daily returns under these weights
        port_daily = (window * w_t).sum(axis=1)
        port_vols.loc[t] = port_daily.std() * np.sqrt(252)
    return port_vols


# =============================================================================
# 4. AGGREGATE LEVERAGE WITH REGIME GATE
# =============================================================================

def aggregate_leverage_series(port_vol: pd.Series, regime_prob: pd.Series) -> pd.Series:
    """
    Aggregate leverage multiplier at each Friday:
      raw_L = clip(vol_target / port_vol, LEV_MIN, LEV_MAX)
      gated_L = min(raw_L, LEV_CAP_HIGH_REGIME) if P(high-corr) > threshold else raw_L
    """
    vol_target = config.STRAT_B_VOL_TARGET
    lev_min    = config.STRAT_B_LEV_MIN
    lev_max    = config.STRAT_B_LEV_MAX
    lev_gated  = config.STRAT_B_LEV_CAP_HIGH_REGIME
    threshold  = config.REGIME_THRESHOLD

    raw = (vol_target / port_vol).clip(lev_min, lev_max)
    gated = raw.copy()
    high_mask = regime_prob > threshold
    gated[high_mask] = np.minimum(raw[high_mask], lev_gated)
    gated.name = "aggregate_leverage"
    return gated


# =============================================================================
# 5. BUILD STRATEGY B WEIGHTS (FULL CROSS-ASSET)
# =============================================================================

def build_strategy_b_weights(daily_returns: pd.DataFrame,
                             signals_full: pd.DataFrame,
                             regime_prob: pd.Series,
                             asset_columns: list,
                             verbose: bool = True) -> dict:
    """
    Full cross-asset trend book construction.

    Parameters
    ----------
    daily_returns : pd.DataFrame
        Daily returns for all 9 assets (from compute_returns).
    signals_full : pd.DataFrame
        Momentum signals for all 9 assets (from build_signal_panel_full),
        so bonds have real signals from cumulative-return momentum.
    regime_prob : pd.Series
        Weekly filtered P(high-corr) from Phase 5.
    asset_columns : list
        Column order for the output.
    """
    # Compute per-asset vols daily, then sample to Friday weekly
    per_asset_vols_daily = per_asset_realized_vol(daily_returns)
    signals_wk = signals_full[asset_columns].resample("W-FRI").last()
    vols_wk    = per_asset_vols_daily[asset_columns].resample("W-FRI").last()
    regime_wk  = regime_prob.resample("W-FRI").last()

    # Common valid index
    valid_idx = (
        signals_wk.dropna(how="any").index
        .intersection(vols_wk.dropna(how="any").index)
        .intersection(regime_wk.dropna().index)
    )
    signals_wk = signals_wk.loc[valid_idx]
    vols_wk    = vols_wk.loc[valid_idx]
    regime_wk  = regime_wk.loc[valid_idx]

    if verbose:
        print(f"[strategy_b] {len(valid_idx)} weekly decisions")
        print(f"[strategy_b] sample: {valid_idx[0].date()} to {valid_idx[-1].date()}")

    # Step 1: raw per-asset weights from signal + vol budget
    W_raw = signal_scaled_raw_weights(signals_wk, vols_wk)

    # Production spec (post-ablation): trend filter + Markov regime gate.
    # Vol targeting removed - see config.py and notebooks/ablation_2026-08-05.md.
    gate = pd.Series(1.0, index=valid_idx)
    gate[regime_wk > config.REGIME_THRESHOLD] = config.STRAT_B_REGIME_GATE_SCALE
    W_final = W_raw.mul(gate, axis=0)
    lev = gate                      # kept for API compatibility with the runner
    port_vol = pd.Series(np.nan, index=valid_idx)   # no longer used in production

    # Cash = 1 - sum(weights). Positive = uninvested. Negative = borrowing.
    cash = 1.0 - W_final.sum(axis=1)

    # Unit tests
    test_details = {}
    test_details["weights_non_negative"] = {
        "min_weight": float(W_final.min().min()),
        "passed":     bool((W_final >= 0).all().all()),
    }
    test_details["max_gross_within_cap"] = {
        "max_gross": float(W_final.sum(axis=1).max()),
        "passed":    bool(W_final.sum(axis=1).max() <= config.STRAT_B_LEV_MAX + 1e-9),
    }
    n_borrow = int((cash < 0).sum())
    test_details["borrowing_frequency"] = {
        "n_weeks":   n_borrow,
        "pct":       float(n_borrow / len(valid_idx) * 100),
    }
    n_gated = int((regime_wk > config.REGIME_THRESHOLD).sum())
    test_details["regime_gate_active"] = {
        "n_gated_weeks": n_gated,
        "pct":           float(n_gated / len(valid_idx) * 100),
    }
    passed = all(v.get("passed", True) for v in test_details.values())

    if verbose:
        print(f"[strategy_b] UNIT TESTS:")
        for name, d in test_details.items():
            print(f"    {name}: {d}")

    return {
        "weights":       W_final,
        "raw_weights":   W_raw,
        "leverage":      lev,
        "port_vol":      port_vol,
        "cash_weight":   cash,
        "signals_used":  signals_wk,
        "passed_tests":  passed,
        "test_details":  test_details,
    }


# =============================================================================
# 6. BACKTEST RUNNER (handles cash + financing spread)
# =============================================================================

def run_strategy_b(weights: pd.DataFrame,
                   cash: pd.Series,
                   weekly_returns: pd.DataFrame,
                   rf: pd.Series,
                   cost_bps: float = None,
                   financing_bps: float = None,
                   label: str = "StrategyB",
                   verbose: bool = True) -> dict:
    """
    Backtest a levered/cash portfolio.
    - Positive cash earns RF weekly.
    - Negative cash (borrowing) pays RF + financing_spread on the borrowed amount.
    - Weight lag = 1 week (same convention as run_strategy).
    """
    if cost_bps is None:
        cost_bps = config.TX_COST_BPS
    if financing_bps is None:
        financing_bps = config.STRAT_B_FINANCING_BPS

    w_lagged    = weights.shift(1).dropna(how="all")
    cash_lagged = cash.shift(1).dropna()

    common_cols  = [c for c in w_lagged.columns if c in weekly_returns.columns]
    common_dates = (w_lagged.index
                    .intersection(weekly_returns.index)
                    .intersection(cash_lagged.index)
                    .intersection(rf.index))

    W  = w_lagged.loc[common_dates, common_cols]
    R  = weekly_returns.loc[common_dates, common_cols]
    C  = cash_lagged.loc[common_dates]
    RF = rf.loc[common_dates]

    asset_gross = (W * R).sum(axis=1)

    fin_weekly = (financing_bps / 10000.0) / 52.0
    cash_pos = C.clip(lower=0) * RF
    cash_neg = C.clip(upper=0) * (RF + fin_weekly)
    cash_contrib = cash_pos + cash_neg

    gross = asset_gross + cash_contrib

    delta_w = W.diff()
    delta_w.iloc[0] = W.iloc[0]
    turnover = 0.5 * delta_w.abs().sum(axis=1)
    cost = turnover * (cost_bps / 10000.0)

    net = gross - cost

    if verbose:
        n_years = (common_dates[-1] - common_dates[0]).days / 365.25
        ann_ret = net.mean() * 52
        ann_vol = net.std() * np.sqrt(52)
        sharpe  = ann_ret / ann_vol if ann_vol > 0 else np.nan
        ann_turn = turnover.mean() * 52
        cost_drag = cost.mean() * 52 * 10000
        pct_borrow = (C < 0).mean() * 100

        print(f"[run_strategy_b] {label}")
        print(f"    period:            {common_dates[0].date()} to {common_dates[-1].date()} ({n_years:.1f}y)")
        print(f"    cost model:        {cost_bps} bps one-way + {financing_bps} bps/yr financing")
        print(f"    ann return net:    {ann_ret:+.2%}")
        print(f"    ann vol:           {ann_vol:.2%}")
        print(f"    raw Sharpe:        {sharpe:.2f}")
        print(f"    ann turnover:      {ann_turn:.0%}")
        print(f"    cost drag:         {cost_drag:.0f} bps/yr")
        print(f"    % weeks borrowing: {pct_borrow:.1f}%")

    return {
        "gross":    gross,
        "net":      net,
        "turnover": turnover,
        "cost":     cost,
        "weights":  W,
        "label":    label,
    }

# =============================================================================
# 7. LAYER ABLATION — WHICH MECHANISM DRIVES PERFORMANCE?
# =============================================================================

def build_ablation_variants(daily_returns: pd.DataFrame,
                            signals_full: pd.DataFrame,
                            regime_prob: pd.Series,
                            asset_columns: list,
                            verbose: bool = True) -> dict:
    """
    Build 5 variants of Strategy B, each adding one mechanism, to attribute
    performance to layers rather than reporting one blended number.

    V0 Base        : all assets long at vol-budget size, leverage fixed 1.0
    V1 +Trend      : trend filter on, leverage fixed 1.0
    V2 +VolTarget  : all assets long, aggregate vol targeting on
    V3 +Both       : trend filter AND vol targeting, no regime gate
    V4 Full        : trend + vol targeting + regime gate (= production Strategy B)

    Returns dict of {variant_name: {"weights": df, "cash": series}}
    """
    per_asset_vols_daily = per_asset_realized_vol(daily_returns)
    signals_wk = signals_full[asset_columns].resample("W-FRI").last()
    vols_wk    = per_asset_vols_daily[asset_columns].resample("W-FRI").last()
    regime_wk  = regime_prob.resample("W-FRI").last()

    valid_idx = (signals_wk.dropna(how="any").index
                 .intersection(vols_wk.dropna(how="any").index)
                 .intersection(regime_wk.dropna().index))
    signals_wk = signals_wk.loc[valid_idx]
    vols_wk    = vols_wk.loc[valid_idx]
    regime_wk  = regime_wk.loc[valid_idx]

    budget = config.STRAT_B_PER_ASSET_VOL_BUDGET
    max_w  = config.STRAT_B_MAX_PER_ASSET_WEIGHT
    safe_vols = vols_wk.clip(lower=0.001)

    # --- Raw weights: WITH trend filter (long only when signal == +1) ---
    W_trend = pd.DataFrame(0.0, index=valid_idx, columns=asset_columns)
    long_mask = (signals_wk == 1)
    W_trend[long_mask] = (budget / safe_vols)[long_mask]
    W_trend = W_trend.clip(upper=max_w)

    # --- Raw weights: NO trend filter (always long every asset) ---
    W_notrend = (budget / safe_vols).clip(upper=max_w)

    if verbose:
        print(f"[ablation] {len(valid_idx)} weeks, {valid_idx[0].date()} to {valid_idx[-1].date()}")

    variants = {}

    # V0: base, no trend, no vol targeting (leverage = 1.0)
    variants["V0_Base"] = {"weights": W_notrend, "cash": 1.0 - W_notrend.sum(axis=1)}

    # V1: trend only, leverage = 1.0
    variants["V1_Trend"] = {"weights": W_trend, "cash": 1.0 - W_trend.sum(axis=1)}

    # V2: vol targeting only (no trend filter)
    pv_notrend = portfolio_vol_under_weights(daily_returns[asset_columns], W_notrend)
    lev_notrend_ungated = (config.STRAT_B_VOL_TARGET / pv_notrend).clip(
        config.STRAT_B_LEV_MIN, config.STRAT_B_LEV_MAX).fillna(1.0)
    W_v2 = W_notrend.mul(lev_notrend_ungated, axis=0)
    variants["V2_VolTarget"] = {"weights": W_v2, "cash": 1.0 - W_v2.sum(axis=1)}

    # V3: trend + vol targeting, NO regime gate
    pv_trend = portfolio_vol_under_weights(daily_returns[asset_columns], W_trend)
    lev_trend_ungated = (config.STRAT_B_VOL_TARGET / pv_trend).clip(
        config.STRAT_B_LEV_MIN, config.STRAT_B_LEV_MAX).fillna(1.0)
    W_v3 = W_trend.mul(lev_trend_ungated, axis=0)
    variants["V3_Trend_Vol"] = {"weights": W_v3, "cash": 1.0 - W_v3.sum(axis=1)}

    # V4: full stack (trend + vol targeting + regime gate) = production Strategy B
    lev_gated = aggregate_leverage_series(pv_trend, regime_wk).fillna(1.0)
    W_v4 = W_trend.mul(lev_gated, axis=0)
    variants["V4_Full"] = {"weights": W_v4, "cash": 1.0 - W_v4.sum(axis=1)}

    if verbose:
        for name, v in variants.items():
            gross = v["weights"].sum(axis=1)
            print(f"[ablation]   {name:<15s} mean gross={gross.mean():.3f}  "
                  f"min={gross.min():.3f}  max={gross.max():.3f}")

    return variants