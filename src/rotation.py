"""
Rate-regime factor rotation backtest.

Structure, fixed in advance (see src/rate_regime.py for the signal itself):

    Strategic split     70% equity / 30% defensive
    Within equity       ERC across Growth and Value
    Regime tilt         +/-10pp between Growth and Value, from the score
    Defensive sleeve    IEF when rates falling or neutral, SHY when rising
    Extreme state       at score -3, shift 10pp from equity to defensive

Rebalanced monthly on the last business day, with a 2pp no-trade band.
Signals are lagged one period: the weights applied to month t+1 are decided
using data available at the close of month t.

Equal risk contribution across ALL FOUR sleeves was rejected on structural
grounds: with SHY at 1.5% annualised volatility against 20% for equity, an
inverse-volatility construction assigns roughly 87% to Treasury bills in any
rising-rate period. The strategic split is a policy choice, not an optimiser
output, and is stated as such.
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

# ------------------------------------------------------------------- policy --
EQUITY_BUDGET = 0.70
DEFENSIVE_BUDGET = 0.30
CRISIS_SHIFT = 0.10          # equity -> defensive at score -3
NO_TRADE_BAND = 0.02         # 2pp
COST_BPS = 5.0               # one-way, in basis points

EQUITY = ["Growth", "Value"]
BONDS = {"IEF": "Bond_Long", "SHY": "Bond_Short"}
ALL_SLEEVES = ["Growth", "Value", "Bond_Long", "Bond_Short"]


# ------------------------------------------------------------------ weights --
def target_weights(score: int, defensive_etf: str,
                   erc_split: tuple[float, float]) -> pd.Series:
    """
    Translate one regime reading into a full weight vector.

    erc_split : (growth, value) shares of the equity budget, summing to 1.
    """
    equity_budget = EQUITY_BUDGET
    defensive_budget = DEFENSIVE_BUDGET
    if score <= -3:
        equity_budget -= CRISIS_SHIFT
        defensive_budget += CRISIS_SHIFT

    g_share, v_share = erc_split
    w_growth = equity_budget * g_share
    w_value = equity_budget * v_share

    # Regime tilt, expressed in portfolio percentage points.
    tilt = (10.0 * score / 3.0) / 100.0
    w_growth += tilt
    w_value -= tilt

    # Long-only: if a tilt would push a sleeve negative, cap it there and
    # give the remainder to the other sleeve.
    if w_growth < 0:
        w_value += w_growth
        w_growth = 0.0
    if w_value < 0:
        w_growth += w_value
        w_value = 0.0

    w = pd.Series(0.0, index=ALL_SLEEVES)
    w["Growth"] = w_growth
    w["Value"] = w_value
    w[BONDS[defensive_etf]] = defensive_budget
    return w


def apply_no_trade_band(target: pd.Series, current: pd.Series,
                        band: float = NO_TRADE_BAND) -> pd.Series:
    """Keep the current weight where the target is within `band` of it."""
    if current is None:
        return target
    # If any sleeve breaches the band, rebalance the whole book to target.
    # Partial updates followed by renormalisation silently rescale the
    # sleeves that were deliberately left alone, and the error compounds.
    if ((target - current).abs() > band).any():
        return target.copy()
    return current.copy()


def erc_two_asset(cov: np.ndarray) -> tuple[float, float]:
    """
    Closed-form ERC for two assets: weights are inversely proportional to
    volatility. Exact, no optimiser needed.
    """
    s1, s2 = np.sqrt(cov[0, 0]), np.sqrt(cov[1, 1])
    if s1 <= 0 or s2 <= 0 or not np.isfinite(s1 + s2):
        return 0.5, 0.5
    w1 = (1 / s1) / (1 / s1 + 1 / s2)
    return float(w1), float(1 - w1)


# ------------------------------------------------------------ trend overlay --
TREND_WINDOW = 200


def trend_filter(prices: pd.DataFrame, date: pd.Timestamp,
                 window: int = TREND_WINDOW) -> dict:
    """
    True where the sleeve's close is at or above its own moving average.
    Uses only prices up to and including `date`.
    """
    hist = prices.loc[:date]
    if len(hist) < window:
        return {c: True for c in EQUITY}
    ma = hist.tail(window).mean()
    last = hist.iloc[-1]
    return {c: bool(last[c] >= ma[c]) for c in EQUITY}


def apply_trend(weights: pd.Series, above: dict, defensive_etf: str) -> pd.Series:
    """Move a sleeve's weight to the defensive bond when it is below its MA."""
    w = weights.copy()
    parked = 0.0
    for c in EQUITY:
        if not above.get(c, True):
            parked += w[c]
            w[c] = 0.0
    if parked > 0:
        w[BONDS[defensive_etf]] += parked
    return w


# ----------------------------------------------------------------- backtest --
def run_backtest(returns: pd.DataFrame,
                 regime: pd.DataFrame,
                 cost_bps: float = COST_BPS,
                 vol_window: int = 63,
                 prices: pd.DataFrame | None = None,
                 use_trend: bool = False) -> dict:
    """
    Monthly rebalanced backtest of the rotation strategy plus benchmarks.

    returns : daily simple or log returns for the four sleeves plus Market.
    regime  : output of rate_regime.build_regime, daily.
    """
    cols = [c for c in ALL_SLEEVES if c in returns.columns]
    if len(cols) < 4:
        raise KeyError(f"missing sleeves: {set(ALL_SLEEVES) - set(cols)}")

    rets = returns[cols].dropna()
    reg = regime.reindex(rets.index).ffill().dropna(subset=["score"])
    rets = rets.loc[reg.index]

    month_ends = rets.resample("ME").last().index
    month_ends = [d for d in month_ends if d in rets.index or True]

    # rolling covariance for the equity ERC split
    eq_cov = rets[EQUITY].rolling(vol_window).cov()

    w_hist, current = {}, None
    for i, d in enumerate(rets.resample("ME").last().index[:-1]):
        window = rets.loc[:d]
        if len(window) < vol_window:
            continue
        cov = window[EQUITY].tail(vol_window).cov().values * 252
        split = erc_two_asset(cov)

        row = reg.loc[:d]
        if row.empty:
            continue
        row = row.iloc[-1]

        tgt = target_weights(int(row["score"]), row["defensive"], split)
        if use_trend:
            if prices is None:
                raise ValueError("use_trend requires the price panel")
            above = trend_filter(prices, d)
            tgt = apply_trend(tgt, above, row["defensive"])
        current = apply_no_trade_band(tgt, current)
        w_hist[d] = current.copy()

    weights = pd.DataFrame(w_hist).T
    if weights.empty:
        raise RuntimeError("no rebalance dates produced weights")

    # daily weights, held between rebalances, applied with a one-period lag
    daily_w = weights.reindex(rets.index, method="ffill").shift(1).dropna()
    aligned = rets.loc[daily_w.index]

    gross = (daily_w * aligned).sum(axis=1)
    turnover = daily_w.diff().abs().sum(axis=1).fillna(0.0)
    cost = turnover * cost_bps / 10000.0
    net = gross - cost

    # benchmarks
    eq_w = pd.Series(0.25, index=ALL_SLEEVES)
    bench_ew = (aligned * eq_w).sum(axis=1)

    b6040 = pd.Series(0.0, index=ALL_SLEEVES)
    b6040["Growth"] = 0.30
    b6040["Value"] = 0.30
    b6040["Bond_Long"] = 0.40
    bench_6040 = (aligned * b6040).sum(axis=1)

    # Same policy split, no regime view. This is the benchmark that
    # isolates what the tilt actually contributes.
    b_policy = pd.Series(0.0, index=ALL_SLEEVES)
    b_policy["Growth"] = EQUITY_BUDGET / 2
    b_policy["Value"] = EQUITY_BUDGET / 2
    b_policy["Bond_Long"] = DEFENSIVE_BUDGET
    bench_policy = (aligned * b_policy).sum(axis=1)

    out = {
        "weights": weights,
        "daily_weights": daily_w,
        "gross": gross,
        "net": net,
        "cost": cost,
        "turnover": turnover,
        "bench_ew": bench_ew,
        "bench_6040": bench_6040,
        "bench_policy": bench_policy,
        "regime": reg.loc[daily_w.index],
    }
    if "Market" in returns.columns:
        out["spy"] = returns["Market"].loc[daily_w.index]
    return out


# ------------------------------------------------------------------ metrics --
def summarise(series: pd.Series, rf: pd.Series | None = None,
              name: str = "strategy") -> dict:
    """Annualised statistics from a daily log-return series."""
    s = series.dropna()
    n = len(s)
    if n < 50:
        return {}
    years = n / 252.0
    total = float(np.exp(s.sum()))
    ann_ret = total ** (1 / years) - 1
    ann_vol = float(s.std() * np.sqrt(252))

    excess = s - (rf.reindex(s.index).fillna(0.0) if rf is not None else 0.0)
    sharpe = float(excess.mean() / excess.std() * np.sqrt(252)) if excess.std() > 0 else np.nan

    eq = np.exp(s.cumsum())
    dd = eq / eq.cummax() - 1
    max_dd = float(dd.min())

    return {
        "name": name,
        "ann_return": ann_ret,
        "ann_vol": ann_vol,
        "sharpe_excess": sharpe,
        "max_dd": max_dd,
        "calmar": ann_ret / abs(max_dd) if max_dd < 0 else np.nan,
        "best_year": float(np.exp(s.resample("YE").sum()).max() - 1),
        "worst_year": float(np.exp(s.resample("YE").sum()).min() - 1),
        "pct_up_months": float((s.resample("ME").sum() > 0).mean()),
    }


def compare(res: dict, rf: pd.Series | None = None) -> pd.DataFrame:
    rows = [
        summarise(res["net"], rf, "Rotation (net)"),
        summarise(res["gross"], rf, "Rotation (gross)"),
        summarise(res["bench_policy"], rf, "Same 70/30, no tilt"),
        summarise(res["bench_ew"], rf, "Equal weight (50/50)"),
        summarise(res["bench_6040"], rf, "60/40"),
    ]
    if "spy" in res:
        rows.append(summarise(res["spy"], rf, "SPY (reference)"))
    return pd.DataFrame([r for r in rows if r]).set_index("name")


def by_regime(res: dict) -> pd.DataFrame:
    """Strategy and benchmark returns within each regime score."""
    df = pd.DataFrame({
        "net": res["net"],
        "ew": res["bench_policy"],
        "score": res["regime"]["score"],
    }).dropna()
    rows = []
    for sc, g in df.groupby("score"):
        yrs = len(g) / 252.0
        rows.append({
            "score": int(sc),
            "days": len(g),
            "years": round(yrs, 1),
            "rotation_ann": float(np.exp(g["net"].sum()) ** (1 / yrs) - 1) if yrs > 0.2 else np.nan,
            "ew_ann": float(np.exp(g["ew"].sum()) ** (1 / yrs) - 1) if yrs > 0.2 else np.nan,
        })
    out = pd.DataFrame(rows)
    out["difference"] = out["rotation_ann"] - out["ew_ann"]
    return out.sort_values("score", ascending=False)


if __name__ == "__main__":
    from src.data import get_data
    from src.returns import compute_returns
    from src.rate_regime import build_regime

    panel, _ = get_data()
    rets = compute_returns(panel)
    reg = build_regime(panel)

    res = run_backtest(rets, reg)
    rf = rets["RF_RATE"] if "RF_RATE" in rets.columns else None

    print(f"\nbacktest {res['net'].index[0].date()} to {res['net'].index[-1].date()}"
          f"  ({len(res['net']):,} days, {len(res['weights'])} rebalances)\n")

    print("HEADLINE")
    c = compare(res, rf)
    print((c * 100).round(2).to_string())

    print("\n\nBY REGIME SCORE")
    print(by_regime(res).round(4).to_string(index=False))

    print("\n\nTURNOVER")
    print(f"  annualised {res['turnover'].sum() / (len(res['turnover'])/252):.1%}")
    print(f"  cost drag  {res['cost'].sum() / (len(res['cost'])/252)*100:.2f}% per year")

    print("\n\nLATEST WEIGHTS")
    print((res["weights"].iloc[-1] * 100).round(2).to_string())
