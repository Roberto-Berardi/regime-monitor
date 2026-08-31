"""
Ex-ante risk analytics for the current book.

Everything the backtest reports is realised - what the portfolio DID. This
module answers what risk it is RUNNING right now, which is the question a
risk committee actually asks: what is our volatility, our duration, our
equity beta, and what does a bad week look like.

All figures are ex-ante: derived from the current GARCH/DCC covariance, not
from historical returns.
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

TRADING_DAYS = 252
WEEKS = 52


def latest_covariance(fits: dict, dcc: dict, assets: list) -> pd.DataFrame:
    """
    Rebuild the covariance matrix at the most recent date: Sigma = D R D,
    where D is the diagonal matrix of conditional volatilities and R is the
    DCC correlation matrix. Daily units, same scale as the return series.
    """
    R = pd.DataFrame(dcc["R"][-1], index=dcc["assets"], columns=dcc["assets"])
    R = R.loc[assets, assets]
    sigma = np.array([float(fits[a]["sigma"].iloc[-1]) for a in assets])
    D = np.diag(sigma)
    cov = D @ R.values @ D
    return pd.DataFrame(cov, index=assets, columns=assets)


def risk_contributions(weights: pd.Series, cov: pd.DataFrame) -> pd.DataFrame:
    """
    Decompose portfolio variance by asset.

        marginal_i = (Sigma w)_i          sensitivity of portfolio vol to w_i
        RC_i       = w_i * marginal_i     absolute contribution to variance
        pct_i      = RC_i / (w' Sigma w)  share of total risk, sums to 1

    Under equal risk contribution every pct_i equals 1/N. Comparing pct_i
    against the capital weight is the clearest way to show the difference
    between allocating capital and allocating risk.
    """
    assets = [a for a in weights.index if a in cov.columns]
    w = weights.loc[assets].values
    S = cov.loc[assets, assets].values

    port_var = float(w @ S @ w)
    marginal = S @ w
    rc_abs = w * marginal
    rc_pct = rc_abs / port_var if port_var > 0 else np.zeros_like(rc_abs)

    return pd.DataFrame({
        "weight":      w,
        "rc_pct":      rc_pct,
        "vol_ann":     np.sqrt(np.diag(S)) * np.sqrt(TRADING_DAYS),
        "mctr_ann":    marginal / np.sqrt(port_var) * np.sqrt(TRADING_DAYS),
    }, index=assets)


def portfolio_duration(weights: pd.Series) -> float:
    """
    Weighted modified duration in years. Only the rate proxies carry duration;
    credit ETFs carry spread duration we do not model, so this understates
    true rate sensitivity of the book.
    """
    total = 0.0
    for asset, w in weights.items():
        base = asset.replace("_proxy", "")
        if base in config.DURATIONS:
            total += float(w) * config.DURATIONS[base]
    return total


def equity_beta(weights: pd.Series, cov: pd.DataFrame,
                equity_asset: str = "SP500") -> float:
    """
    Ex-ante beta of the book to the equity market, from the current
    covariance: beta = cov(portfolio, equity) / var(equity).
    """
    if equity_asset not in cov.columns:
        return float("nan")
    assets = [a for a in weights.index if a in cov.columns]
    w = weights.loc[assets].values
    cov_pe = float(w @ cov.loc[assets, equity_asset].values)
    var_e = float(cov.loc[equity_asset, equity_asset])
    return cov_pe / var_e if var_e > 0 else float("nan")


def risk_summary(weights: pd.Series, cov: pd.DataFrame) -> dict:
    """
    The risk strip: what book are we running right now?

    Expected shortfall uses a normal approximation (ES_95 = 2.063 * sigma).
    The fitted GARCH innovations are Student-t with 4-10 degrees of freedom,
    so the true tail is fatter - treat this as a floor, not a worst case.
    That caveat belongs on the page.
    """
    assets = [a for a in weights.index if a in cov.columns]
    w = weights.loc[assets].values
    S = cov.loc[assets, assets].values

    var_daily = float(w @ S @ w)
    vol_ann = np.sqrt(var_daily * TRADING_DAYS)
    vol_weekly = np.sqrt(var_daily * 5)

    ES_MULT_95 = 2.063   # E[X | X < -1.645s] for a standard normal

    return {
        "vol_ann":          vol_ann,
        "vol_weekly":       vol_weekly,
        "duration_years":   portfolio_duration(weights.loc[assets]),
        "equity_beta":      equity_beta(weights, cov),
        "var95_weekly":     1.645 * vol_weekly,
        "es95_weekly":      ES_MULT_95 * vol_weekly,
        "n_assets_held":    int((weights.loc[assets] > 0.001).sum()),
        "largest_weight":   float(weights.loc[assets].max()),
        "largest_asset":    str(weights.loc[assets].idxmax()),
    }


def trailing_performance(net_weekly: pd.Series, rf_weekly: pd.Series = None,
                         weeks: int = 52) -> dict:
    """Trailing-window performance. A PM asks 'how has it done this year'."""
    s = net_weekly.dropna().tail(weeks)
    if len(s) < 10:
        return {}
    eq = (1 + s).cumprod()
    out = {
        "period_weeks": len(s),
        "ret":          float(eq.iloc[-1] - 1),
        "vol_ann":      float(s.std() * np.sqrt(WEEKS)),
        "best_week":    float(s.max()),
        "worst_week":   float(s.min()),
        "max_dd":       float((eq / eq.cummax() - 1).min()),
        "pct_up_weeks": float((s > 0).mean()),
    }
    if rf_weekly is not None:
        r = rf_weekly.reindex(s.index).fillna(0.0)
        ex = s - r
        v = ex.std() * np.sqrt(WEEKS)
        out["sharpe_excess"] = float(ex.mean() * WEEKS / v) if v > 0 else float("nan")
    return out
