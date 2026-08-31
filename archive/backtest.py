"""
Backtest engine for the Cross-Asset Regime Monitor.

Design discipline:
- Weights decided at Friday t are applied to returns from t to t+1
  (weight-lag prevents same-week look-ahead).
- Log returns aggregated to weekly, then converted to simple returns
  for portfolio-linear combination.
- Transaction cost: config.TX_COST_BPS one-way on turnover
  (turnover = 0.5 * L1 norm of weight change).
- Both gross and net series stored; the cost stress case is a config toggle.
- No dividends, no borrow, no financing — the asset universe is a
  demonstration set. Explicitly disclosed in Limitations.
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
import config


# =============================================================================
# 1. RETURNS TRANSFORMATION
# =============================================================================

def daily_log_to_weekly_simple(daily_returns: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate daily log returns to weekly, then convert to simple returns
    for use in portfolio calculations (which are LINEAR in simple returns,
    not log returns).

    Weekly log return  = sum of daily log returns within Fri-Fri week
    Weekly simple ret  = exp(weekly log) - 1
    """
    # Fri-Fri weekly aggregation (log returns are additive over time)
    weekly_log    = daily_returns.resample("W-FRI").sum()
    weekly_simple = np.exp(weekly_log) - 1
    return weekly_simple
def get_rf_weekly(panel: pd.DataFrame, weekly_index: pd.DatetimeIndex) -> pd.Series:
    """
    Convert the DGS3MO annualized yield (percent) to weekly simple returns.

    DGS3MO gives annual yield in percent (e.g. 4.5 means 4.5%). We convert to
    a weekly simple return via compounding: rf_weekly = (1 + y/100)^(1/52) - 1.
    Forward-filled to the strategy's weekly Friday grid.
    """
    if "RF_RATE" not in panel.columns:
        raise KeyError("RF_RATE column missing from panel. Refresh cache after "
                       "adding DGS3MO to config.FRED_SERIES.")
    y_annual = panel["RF_RATE"].ffill() / 100.0
    y_weekly_daily = (1.0 + y_annual) ** (1.0 / 52.0) - 1.0
    rf = y_weekly_daily.reindex(weekly_index, method="ffill").fillna(0.0)
    rf.name = "rf_weekly"
    return rf


# =============================================================================
# 2. STRATEGY EXECUTION
# =============================================================================

def run_strategy(
    weights: pd.DataFrame,
    weekly_returns: pd.DataFrame,
    cost_bps: float = None,
    label: str = "strategy",
    verbose: bool = True,
) -> dict:
    """
    Apply a weight matrix to weekly returns with a one-week lag.
    Compute gross return, cost drag, net return, and turnover per week.

    Parameters
    ----------
    weights : pd.DataFrame
        Weekly weights, one row per Friday, columns = assets.
        Weights at date t are treated as decided at close of t and applied
        to returns of the following week (t -> t+1).
    weekly_returns : pd.DataFrame
        Weekly simple returns from daily_log_to_weekly_simple().
    cost_bps : float
        One-way transaction cost in basis points. Defaults to
        config.TX_COST_BPS.

    Returns
    -------
    dict with:
        gross      : pd.Series of weekly portfolio gross returns
        net        : pd.Series of weekly portfolio net returns
        turnover   : pd.Series of weekly turnover
        cost       : pd.Series of weekly cost drag
        weights    : the lagged weight matrix actually used
    """
    if cost_bps is None:
        cost_bps = config.TX_COST_BPS

    # ---- Weight lag: crucial ----
    # w decided at time t applies to returns realised t -> t+1.
    # Shift(1) means: on Fri t+1, we hold weights from Fri t. This is the
    # standard convention for "weights decided at close of previous period".
    w_lagged = weights.shift(1).dropna(how="all")

    # Align columns and dates
    common_cols  = [c for c in w_lagged.columns if c in weekly_returns.columns]
    common_dates = w_lagged.index.intersection(weekly_returns.index)
    W = w_lagged.loc[common_dates, common_cols]
    R = weekly_returns.loc[common_dates, common_cols]

    # ---- Portfolio gross return per week ----
    # Portfolios are linear in SIMPLE returns: r_p = sum(w_i * r_i)
    gross = (W * R).sum(axis=1)

    # ---- Turnover: L1 norm of weight change divided by 2 ----
    #   turnover_t = 0.5 * sum_i |w_i(t) - w_i(t-1)|
    # First observation: we assume weights were set from cash so
    # initial "turnover" is 0.5 * sum(|w|) = 0.5 (weights sum to 1),
    # which correctly reflects the setup cost.
    delta_w = W.diff()
    delta_w.iloc[0] = W.iloc[0]  # initial position from cash
    turnover = 0.5 * delta_w.abs().sum(axis=1)

    # ---- Cost drag: turnover * cost_bps (both sides included via /2 above) ----
    cost = turnover * (cost_bps / 10000.0)

    # ---- Net return ----
    net = gross - cost

    if verbose:
        # Compact stats print
        n_years = (common_dates[-1] - common_dates[0]).days / 365.25
        ann_ret_net = net.mean() * 52
        ann_vol_net = net.std() * np.sqrt(52)
        sharpe_net  = ann_ret_net / ann_vol_net if ann_vol_net > 0 else np.nan
        ann_turn    = turnover.mean() * 52
        cost_drag   = cost.mean() * 52 * 10000  # bps/yr

        print(f"[run_strategy] {label}")
        print(f"    period:         {common_dates[0].date()} to {common_dates[-1].date()} "
              f"({n_years:.1f} years, {len(common_dates)} weeks)")
        print(f"    cost model:     {cost_bps} bps one-way")
        print(f"    ann return net: {ann_ret_net:+.2%}")
        print(f"    ann vol:        {ann_vol_net:.2%}")
        print(f"    Sharpe (net):   {sharpe_net:.2f}")
        print(f"    ann turnover:   {ann_turn:.0%}")
        print(f"    cost drag:      {cost_drag:.0f} bps/yr")

    return {
        "gross":    gross,
        "net":      net,
        "turnover": turnover,
        "cost":     cost,
        "weights":  W,
        "label":    label,
    }


# =============================================================================
# 3. HELPER: CUMULATIVE EQUITY CURVE
# =============================================================================

def cumulative_equity(weekly_returns: pd.Series) -> pd.Series:
    """
    Cumulative growth of $1 given weekly simple returns.
    Uses (1 + r) product convention.
    """
    return (1.0 + weekly_returns).cumprod()


# =============================================================================
# 4. HELPER: MAX DRAWDOWN
# =============================================================================

def max_drawdown(equity_curve: pd.Series) -> float:
    """
    Peak-to-trough max drawdown as a NEGATIVE number.
    """
    peak = equity_curve.cummax()
    dd   = (equity_curve / peak) - 1.0
    return float(dd.min())

# =============================================================================
# 5. BENCHMARK WEIGHT BUILDERS
# =============================================================================

def build_60_40_weights(returns_index: pd.DatetimeIndex,
                       assets: list) -> pd.DataFrame:
    """
    Classic 60/40: 60% SP500, 40% US_10Y_proxy.
    Rebalanced weekly (implicit continuous rebalancing).
    """
    W = pd.DataFrame(0.0, index=returns_index, columns=assets)
    if "SP500" in assets:         W["SP500"] = 0.60
    if "US_10Y_proxy" in assets:  W["US_10Y_proxy"] = 0.40
    # Handle missing columns by renormalising
    row_sums = W.sum(axis=1)
    W = W.div(row_sums, axis=0)
    return W


def build_equal_weight(returns_index: pd.DatetimeIndex,
                      assets: list) -> pd.DataFrame:
    """
    1/N equally weighted across all assets, weekly rebal.
    """
    n = len(assets)
    W = pd.DataFrame(1.0 / n, index=returns_index, columns=assets)
    return W


def build_pure_erc_weekly(erc_hist: pd.DataFrame,
                          weekly_index: pd.DatetimeIndex,
                          asset_cols: list) -> pd.DataFrame:
    """
    Convert monthly ERC weights to the same weekly Friday grid used by
    the tilted strategy (forward-fill within month). No tilt, no gate.
    """
    W = erc_hist[asset_cols].reindex(weekly_index, method="ffill").dropna(how="all")
    return W


# =============================================================================
# 6. PERFORMANCE SUMMARY TABLE
# =============================================================================

def performance_stats(net: pd.Series, rf: pd.Series = None,
                      turnover: pd.Series = None, cost: pd.Series = None) -> dict:
    """
    Stats for a strategy: raw AND excess Sharpe, ann return, ann vol,
    max DD, Calmar, turnover, cost drag.

    Raw Sharpe:    mean(net) / std(net) * sqrt(52)
    Excess Sharpe: mean(net - rf) / std(net - rf) * sqrt(52)  [textbook]
    """
    if len(net) < 2:
        return {}
    ann_ret = net.mean() * 52
    ann_vol = net.std() * np.sqrt(52)
    sharpe_raw = ann_ret / ann_vol if ann_vol > 0 else np.nan
    eq  = (1 + net).cumprod()
    mdd = max_drawdown(eq)
    calmar = ann_ret / abs(mdd) if mdd < 0 else np.nan

    stats = {
        "ann_return":    ann_ret,
        "ann_vol":       ann_vol,
        "sharpe_raw":    sharpe_raw,
        "max_dd":        mdd,
        "calmar":        calmar,
        "final_eq":      float(eq.iloc[-1]),
    }

    if rf is not None:
        rf_aligned = rf.reindex(net.index).fillna(0.0)
        excess = net - rf_aligned
        ann_excess = excess.mean() * 52
        vol_excess = excess.std() * np.sqrt(52)
        stats["ann_rf"]        = float(rf_aligned.mean() * 52)
        stats["ann_excess"]    = ann_excess
        stats["sharpe_excess"] = ann_excess / vol_excess if vol_excess > 0 else np.nan

    if turnover is not None:
        stats["ann_turnover"] = float(turnover.mean() * 52)
    if cost is not None:
        stats["cost_bps"] = float(cost.mean() * 52 * 10000)
    return stats


def compare_strategies(results: dict, rf: pd.Series = None) -> pd.DataFrame:
    """
    Take a dict of {label: run_strategy_output} and return a comparison
    DataFrame with all key stats side by side. If rf is provided, excess
    Sharpe columns are added.
    """
    rows = {}
    for label, r in results.items():
        rows[label] = performance_stats(r["net"], rf=rf,
                                        turnover=r["turnover"], cost=r["cost"])
    df = pd.DataFrame(rows).T
    # Reorder columns to put the interview-facing ones first
    preferred = ["ann_return", "ann_rf", "ann_excess", "ann_vol",
                 "sharpe_raw", "sharpe_excess", "max_dd", "calmar",
                 "final_eq", "ann_turnover", "cost_bps"]
    ordered = [c for c in preferred if c in df.columns]
    return df[ordered]


# =============================================================================
# 7. LOOK-AHEAD SAFETY TEST
# =============================================================================

def lookahead_test(weights: pd.DataFrame,
                   weekly_returns: pd.DataFrame,
                   extra_lag_weeks: int = 1,
                   cost_bps: float = None) -> dict:
    """
    Lag the weights by an EXTRA week and re-run. If performance collapses
    (Sharpe halves or turns negative), the original backtest was peeking.
    If performance degrades only mildly, the original is safe.

    Returns
    -------
    dict:
        base_sharpe   : Sharpe of the strategy as-is
        lagged_sharpe : Sharpe with extra_lag_weeks additional lag
        degradation   : (base - lagged) / base
        verdict       : "SAFE" or "SUSPECT LOOK-AHEAD"
    """
    if cost_bps is None:
        cost_bps = config.TX_COST_BPS

    base   = run_strategy(weights, weekly_returns, cost_bps=cost_bps,
                          label="base", verbose=False)
    lagged = run_strategy(weights.shift(extra_lag_weeks),
                          weekly_returns, cost_bps=cost_bps,
                          label=f"lagged_{extra_lag_weeks}w", verbose=False)

    base_sharpe   = performance_stats(base["net"])["sharpe_raw"]
    lagged_sharpe = performance_stats(lagged["net"])["sharpe_raw"]
    degradation   = (base_sharpe - lagged_sharpe) / base_sharpe if base_sharpe != 0 else np.nan

    # Rule of thumb: if the extra lag halves or reverses Sharpe, look-ahead
    # was in play. If it degrades by 10-30%, that's normal signal decay.
    if abs(degradation) < 0.30:
        verdict = "SAFE"
    elif abs(degradation) < 0.60:
        verdict = "BORDERLINE"
    else:
        verdict = "SUSPECT LOOK-AHEAD"

    return {
        "base_sharpe":   base_sharpe,
        "lagged_sharpe": lagged_sharpe,
        "degradation":   degradation,
        "verdict":       verdict,
    }

# =============================================================================
# 8. BLOCK BOOTSTRAP FOR SHARPE CIs
# =============================================================================

def bootstrap_sharpe_ci(net_series_dict: dict, rf: pd.Series,
                       n_boots: int = 5000, block_size: int = 52,
                       baseline: str = "Tilted", seed: int = 42) -> dict:
    """
    Block bootstrap for excess-Sharpe confidence intervals AND pairwise
    difference tests across multiple strategies.

    Method:
    1. Sample n_blocks = len(series) // block_size contiguous 52-week blocks
       WITH REPLACEMENT.
    2. Concatenate blocks to form a resampled series of the same length.
    3. Use the SAME block indices for every strategy (preserves pairing).
    4. Compute excess Sharpe on each resample.
    5. 5000 iterations -> distribution of Sharpes per strategy, and of
       pairwise DIFFERENCES vs baseline.

    Why blocks: financial returns have serial dependence (vol clustering).
    IID bootstrap breaks that structure and produces tight-but-wrong CIs.
    52-week blocks preserve annual cycles and vol regimes.

    Returns
    -------
    dict with:
        strategy_cis    : per-strategy mean Sharpe + 95% CI
        pairwise_diffs  : baseline-vs-others: mean diff, 95% CI, p-value,
                          verdict (distinguishable / indistinguishable)
    """
    rng = np.random.default_rng(seed)
    labels = list(net_series_dict.keys())

    # Common date index across strategies
    common_idx = net_series_dict[labels[0]].index
    n = len(common_idx)
    n_blocks = n // block_size

    print(f"[bootstrap_sharpe_ci] {n_boots} resamples, {n_blocks} blocks of "
          f"{block_size} weeks each, baseline = {baseline}")

    # RF aligned once
    rf_arr = rf.reindex(common_idx).fillna(0.0).values

    # Pre-extract net arrays per strategy (aligned)
    net_arrays = {
        label: net_series_dict[label].reindex(common_idx).fillna(0.0).values
        for label in labels
    }

    # Storage
    sharpes = {label: np.zeros(n_boots) for label in labels}

    for b in range(n_boots):
        # Sample block START positions
        block_starts = rng.integers(0, n - block_size + 1, size=n_blocks)
        # Build resampled positional index by unrolling each block
        idx_positions = np.concatenate([
            np.arange(start, start + block_size) for start in block_starts
        ])
        rf_resamp = rf_arr[idx_positions]

        for label in labels:
            net_resamp = net_arrays[label][idx_positions]
            excess = net_resamp - rf_resamp
            mean_e = excess.mean() * 52
            std_e  = excess.std()  * np.sqrt(52)
            sharpes[label][b] = mean_e / std_e if std_e > 0 else np.nan

        if (b + 1) % 1000 == 0:
            print(f"[bootstrap_sharpe_ci]   {b+1}/{n_boots} resamples done")

    # Per-strategy CIs
    strategy_cis = {}
    for label in labels:
        s = sharpes[label]
        s = s[np.isfinite(s)]
        strategy_cis[label] = {
            "mean":    float(np.mean(s)),
            "ci_low":  float(np.percentile(s,  2.5)),
            "ci_high": float(np.percentile(s, 97.5)),
        }

    # Pairwise diffs vs baseline
    pairwise_diffs = {}
    if baseline not in labels:
        print(f"[bootstrap_sharpe_ci] baseline '{baseline}' not in labels, skipping pairwise")
    else:
        for label in labels:
            if label == baseline:
                continue
            d = sharpes[baseline] - sharpes[label]
            d = d[np.isfinite(d)]
            # Two-sided p-value: min tail mass, doubled
            p_val = 2 * min((d > 0).mean(), (d < 0).mean())
            ci_low  = float(np.percentile(d,  2.5))
            ci_high = float(np.percentile(d, 97.5))
            distinguishable = (ci_low > 0) or (ci_high < 0)
            pairwise_diffs[f"{baseline} vs {label}"] = {
                "mean_diff":       float(np.mean(d)),
                "ci_low":          ci_low,
                "ci_high":         ci_high,
                "p_value":         float(p_val),
                "distinguishable": bool(distinguishable),
            }

    return {
        "strategy_cis":   strategy_cis,
        "pairwise_diffs": pairwise_diffs,
        "n_boots":        n_boots,
        "block_size":     block_size,
    }