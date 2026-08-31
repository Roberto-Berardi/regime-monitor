"""
The tests that decide whether the rotation result is worth publishing.

Run after the backtest. Every one of these can kill the claim, which is the
point: a result that survives them is worth showing, and one that does not
is worth showing as a negative finding.

    1. Momentum control   does the regime signal beat simply holding
                          whichever sleeve won over the past 12 months?
    2. Cost ladder        at what transaction cost does the edge vanish?
    3. Lag test           delay every signal one extra month
    4. Dwell sensitivity  0 / 4 / 8 weeks
    5. Liquidity proxy    WALCL vs WRESBAL
    6. Bootstrap          is the Sharpe difference distinguishable from zero?
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import get_data                       # noqa: E402
from src.returns import compute_returns             # noqa: E402
from src.rate_regime import build_regime            # noqa: E402
from src.rotation import (                          # noqa: E402
    run_backtest, summarise, EQUITY_BUDGET, DEFENSIVE_BUDGET, ALL_SLEEVES,
)

SEED = 20260820


def _ann(s: pd.Series) -> float:
    yrs = len(s) / 252.0
    return float(np.exp(s.sum()) ** (1 / yrs) - 1)


def _sharpe(s: pd.Series, rf: pd.Series | None) -> float:
    e = s - (rf.reindex(s.index).fillna(0.0) if rf is not None else 0.0)
    return float(e.mean() / e.std() * np.sqrt(252)) if e.std() > 0 else np.nan


# ---------------------------------------------------------------- 1. momentum
def momentum_control(rets: pd.DataFrame, res: dict, rf) -> pd.DataFrame:
    """
    Same 70/30 policy, but the equity tilt follows 12-month relative
    performance instead of the monetary regime. If this matches or beats
    the regime version, the regime adds nothing over price momentum.
    """
    daily_w = res["daily_weights"]
    aligned = rets[ALL_SLEEVES].loc[daily_w.index]

    mom = (rets["Growth"].rolling(252).sum()
           - rets["Value"].rolling(252).sum())
    mom_m = mom.resample("ME").last()

    rows = {}
    for d in daily_w.resample("ME").last().index:
        prior = mom_m.loc[:d]
        if prior.empty or pd.isna(prior.iloc[-1]):
            continue
        tilt = np.clip(prior.iloc[-1] * 2.0, -0.10, 0.10)   # scaled, capped at 10pp
        w = pd.Series(0.0, index=ALL_SLEEVES)
        w["Growth"] = EQUITY_BUDGET / 2 + tilt
        w["Value"] = EQUITY_BUDGET / 2 - tilt
        w["Bond_Long"] = DEFENSIVE_BUDGET
        rows[d] = w

    mw = pd.DataFrame(rows).T.reindex(daily_w.index, method="ffill").shift(1).dropna()
    common = mw.index.intersection(aligned.index)
    mom_ret = (mw.loc[common] * aligned.loc[common]).sum(axis=1)
    cost = mw.diff().abs().sum(axis=1).fillna(0.0) * 5.0 / 10000.0
    mom_net = mom_ret - cost.loc[common]

    return pd.DataFrame([
        summarise(res["net"].loc[common], rf, "Regime tilt"),
        summarise(mom_net, rf, "12M momentum tilt"),
        summarise(res["bench_policy"].loc[common], rf, "No tilt"),
    ]).set_index("name")


# ------------------------------------------------------------- 2. cost ladder
def cost_ladder(rets: pd.DataFrame, reg: pd.DataFrame, rf) -> pd.DataFrame:
    rows = []
    for bps in (0, 5, 10, 20, 40):
        r = run_backtest(rets, reg, cost_bps=bps)
        s_rot = _sharpe(r["net"], rf)
        s_pol = _sharpe(r["bench_policy"], rf)
        rows.append({
            "cost_bps": bps,
            "rotation_ann": _ann(r["net"]),
            "rotation_sharpe": s_rot,
            "policy_sharpe": s_pol,
            "edge": s_rot - s_pol,
            "rotation_maxdd": summarise(r["net"], rf)["max_dd"],
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------- 3. lag test
def lag_test(rets: pd.DataFrame, reg: pd.DataFrame, rf) -> pd.DataFrame:
    """Delay the regime signal by one extra month and re-run."""
    base = run_backtest(rets, reg)
    lagged_reg = reg.copy()
    lagged_reg[["score", "tilt_pp", "favours", "defensive", "rate", "liq"]] = (
        lagged_reg[["score", "tilt_pp", "favours", "defensive", "rate", "liq"]].shift(21)
    )
    lagged_reg = lagged_reg.dropna(subset=["score"])
    lag = run_backtest(rets, lagged_reg)

    b, l = _sharpe(base["net"], rf), _sharpe(lag["net"], rf)
    return pd.DataFrame([{
        "base_sharpe": b,
        "lagged_sharpe": l,
        "degradation_pct": 100 * (b - l) / b if b else np.nan,
        "base_ann": _ann(base["net"]),
        "lagged_ann": _ann(lag["net"]),
    }])


# --------------------------------------------------------- 4. dwell sensitivity
def dwell_sensitivity(panel: pd.DataFrame, rets: pd.DataFrame, rf) -> pd.DataFrame:
    rows = []
    for dwell in (0, 20, 40):
        reg = build_regime(panel, dwell=dwell)
        r = run_backtest(rets, reg)
        rows.append({
            "dwell_days": dwell,
            "dwell_weeks": dwell // 5,
            "ann_return": _ann(r["net"]),
            "sharpe": _sharpe(r["net"], rf),
            "max_dd": summarise(r["net"], rf)["max_dd"],
            "turnover": float(r["turnover"].sum() / (len(r["turnover"]) / 252)),
        })
    return pd.DataFrame(rows)


# ------------------------------------------------------- 5. liquidity proxy
def liquidity_proxy(panel: pd.DataFrame, rets: pd.DataFrame, rf) -> pd.DataFrame:
    rows = []
    for col in ("WALCL", "WRESBAL"):
        if col not in panel.columns:
            continue
        reg = build_regime(panel, liq_col=col)
        r = run_backtest(rets, reg)
        rows.append({
            "liquidity_series": col,
            "ann_return": _ann(r["net"]),
            "sharpe": _sharpe(r["net"], rf),
            "max_dd": summarise(r["net"], rf)["max_dd"],
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------- 6. bootstrap
def bootstrap_edge(res: dict, rf, n: int = 5000, block: int = 252) -> dict:
    """
    Block bootstrap on the Sharpe difference between the rotation and the
    same-policy benchmark. Blocks preserve volatility clustering.
    """
    rng = np.random.default_rng(SEED)
    a = res["net"].dropna()
    b = res["bench_policy"].reindex(a.index).dropna()
    a = a.reindex(b.index)
    T = len(a)
    n_blocks = int(np.ceil(T / block))

    obs = _sharpe(a, rf) - _sharpe(b, rf)
    diffs = np.empty(n)
    for i in range(n):
        starts = rng.integers(0, T - block, size=n_blocks)
        idx = np.concatenate([np.arange(s, s + block) for s in starts])[:T]
        diffs[i] = _sharpe(a.iloc[idx], None) - _sharpe(b.iloc[idx], None)

    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {
        "observed_edge": obs,
        "ci_low": float(lo),
        "ci_high": float(hi),
        "p_two_sided": float(2 * min((diffs <= 0).mean(), (diffs >= 0).mean())),
        "significant": bool(lo > 0 or hi < 0),
    }


# ---------------------------------------------------------------------- main
if __name__ == "__main__":
    panel, _ = get_data()
    rets = compute_returns(panel)
    reg = build_regime(panel)
    res = run_backtest(rets, reg)
    rf = rets["RF_RATE"] if "RF_RATE" in rets.columns else None

    print("\n" + "=" * 70)
    print("1. MOMENTUM CONTROL  - does the regime beat price momentum?")
    print("=" * 70)
    print((momentum_control(rets, res, rf) * 100).round(2).to_string())

    print("\n" + "=" * 70)
    print("2. COST LADDER")
    print("=" * 70)
    print(cost_ladder(rets, reg, rf).round(4).to_string(index=False))

    print("\n" + "=" * 70)
    print("3. LAG TEST  - one extra month of delay")
    print("=" * 70)
    print(lag_test(rets, reg, rf).round(4).to_string(index=False))

    print("\n" + "=" * 70)
    print("4. DWELL SENSITIVITY")
    print("=" * 70)
    print(dwell_sensitivity(panel, rets, rf).round(4).to_string(index=False))

    print("\n" + "=" * 70)
    print("5. LIQUIDITY PROXY")
    print("=" * 70)
    print(liquidity_proxy(panel, rets, rf).round(4).to_string(index=False))

    print("\n" + "=" * 70)
    print("6. BOOTSTRAP  - is the Sharpe edge distinguishable from zero?")
    print("=" * 70)
    bs = bootstrap_edge(res, rf)
    print(f"  observed edge   {bs['observed_edge']:+.4f}")
    print(f"  95% interval    [{bs['ci_low']:+.4f}, {bs['ci_high']:+.4f}]")
    print(f"  p (two-sided)   {bs['p_two_sided']:.3f}")
    print(f"  significant     {bs['significant']}")
    print()
