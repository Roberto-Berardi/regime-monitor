"""
Credit-stress override.

The rate-regime framework has a structural blind spot: it scores falling
rates as growth-supportive, but rates also fall hard in a crisis. In 2008 and
2020 the model tilted toward growth precisely when equity was collapsing.

Credit spreads separate the two cases. In an easing cycle spreads are stable
or tightening; in a crisis they blow out. The override:

    When the Baa credit spread exceeds its 95th percentile over the trailing
    five years, suspend the regime tilt and move the book to its defensive
    allocation, regardless of what rates are doing.

The threshold is a rolling percentile, not a fitted level, and uses only past
observations. Gilchrist & Zakrajsek (2012) on credit spreads and real activity.
"""
from __future__ import annotations

from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import src.rotation as R                         # noqa: E402
from src.data import get_data                    # noqa: E402
from src.returns import compute_returns          # noqa: E402
from src.rate_regime import build_regime         # noqa: E402

SPREAD_COL = "BAA_SPREAD"
STRESS_PCTILE = 95
LOOKBACK_DAYS = 1260          # five years
CRISIS_EQUITY = 0.30          # equity budget while stressed
EXIT_PCTILE = 80              # de-escalate below this, avoids flapping


def stress_flag(panel: pd.DataFrame,
                col: str = SPREAD_COL,
                enter: int = STRESS_PCTILE,
                exit_: int = EXIT_PCTILE,
                window: int = LOOKBACK_DAYS) -> pd.DataFrame:
    """
    Daily credit-stress state with hysteresis: enter above the 95th
    percentile of the trailing five years, exit below the 80th.
    """
    s = panel[col].ffill().dropna()
    hi = s.rolling(window, min_periods=252).quantile(enter / 100.0)
    lo = s.rolling(window, min_periods=252).quantile(exit_ / 100.0)

    state, out = False, []
    for v, h, l in zip(s.values, hi.values, lo.values):
        if np.isnan(h):
            out.append(False)
            continue
        if not state and v > h:
            state = True
        elif state and v < l:
            state = False
        out.append(state)

    return pd.DataFrame({"spread": s, "enter_at": hi, "exit_at": lo,
                         "stressed": out}, index=s.index)


def run_with_override(rets: pd.DataFrame, reg: pd.DataFrame,
                      stress: pd.DataFrame, cost_bps: float = 5.0,
                      vol_window: int = 63) -> dict:
    cols = R.ALL_SLEEVES
    r = rets[cols].dropna()
    reg = reg.reindex(r.index).ffill().dropna(subset=["score"])
    r = r.loc[reg.index]
    st = stress["stressed"].reindex(r.index).ffill().fillna(False)

    w_hist, current, flags = {}, None, {}
    for d in r.resample("ME").last().index[:-1]:
        hist = r.loc[:d]
        if len(hist) < vol_window:
            continue
        cov = hist[R.EQUITY].tail(vol_window).cov().values * 252
        split = R.erc_two_asset(cov)
        row = reg.loc[:d].iloc[-1]
        stressed = bool(st.loc[:d].iloc[-1])

        if stressed:
            # Suspend the tilt, cut equity, hold the short-duration sleeve.
            g, v = split
            w = pd.Series(0.0, index=cols)
            w["Growth"] = CRISIS_EQUITY * g
            w["Value"] = CRISIS_EQUITY * v
            w["Bond_Short"] = 1.0 - CRISIS_EQUITY
        else:
            w = R.target_weights(int(row["score"]), row["defensive"], split)

        current = R.apply_no_trade_band(w, current)
        w_hist[d] = current.copy()
        flags[d] = stressed

    weights = pd.DataFrame(w_hist).T
    daily_w = weights.reindex(r.index, method="ffill").shift(1).dropna()
    aligned = r.loc[daily_w.index]

    gross = (daily_w * aligned).sum(axis=1)
    turnover = daily_w.diff().abs().sum(axis=1).fillna(0.0)
    net = gross - turnover * cost_bps / 10000.0

    return {"weights": weights, "net": net, "gross": gross,
            "turnover": turnover,
            "stressed": pd.Series(flags).reindex(weights.index)}


if __name__ == "__main__":
    panel, _ = get_data()
    rets = compute_returns(panel)
    reg = build_regime(panel)
    rf = rets["RF_RATE"] if "RF_RATE" in rets.columns else None
    prices = panel[R.EQUITY]

    st = stress_flag(panel)
    print(f"credit-stress flag: {st.index[0].date()} to {st.index[-1].date()}")
    print(f"  stressed on {st['stressed'].mean():.1%} of days\n")

    grp = (st["stressed"] != st["stressed"].shift()).cumsum()
    print("STRESS EPISODES")
    for _, g in st.groupby(grp):
        if g["stressed"].iloc[0] and len(g) > 10:
            print(f"  {g.index[0].date()} -> {g.index[-1].date()}  "
                  f"({len(g):>4} days, peak spread {g['spread'].max():.2f})")

    base = R.run_backtest(rets, reg)
    trend = R.run_backtest(rets, reg, prices=prices, use_trend=True)
    ovr = run_with_override(rets, reg, st)

    rows = [
        R.summarise(base["net"], rf, "Rotation (base)"),
        R.summarise(ovr["net"], rf, "+ credit override"),
        R.summarise(trend["net"], rf, "+ 200d trend"),
        R.summarise(base["bench_policy"], rf, "No tilt"),
        R.summarise(base["spy"], rf, "SPY (reference)"),
    ]
    print("\n" + "=" * 74)
    print("HEADLINE")
    print("=" * 74)
    print((pd.DataFrame(rows).set_index("name") * 100).round(2).to_string())

    print("\nTURNOVER")
    for nm, x in [("base", base), ("override", ovr), ("trend", trend)]:
        yrs = len(x["turnover"]) / 252
        print(f"  {nm:<10} {x['turnover'].sum()/yrs:>7.1%}")

    print("\n" + "=" * 74)
    print("CALENDAR YEARS (%)")
    print("=" * 74)
    yr = pd.DataFrame({
        "base": np.exp(base["net"].resample("YE").sum()) - 1,
        "override": np.exp(ovr["net"].resample("YE").sum()) - 1,
        "trend": np.exp(trend["net"].resample("YE").sum()) - 1,
        "no_tilt": np.exp(base["bench_policy"].resample("YE").sum()) - 1,
    })
    yr.index = yr.index.year
    print((yr * 100).round(1).to_string())

    def sh(s):
        e = s - (rf.reindex(s.index).fillna(0) if rf is not None else 0)
        return float(e.mean() / e.std() * np.sqrt(252))

    print("\n" + "=" * 74)
    print("EX-2008 CHECK  - is the benefit broad or one episode?")
    print("=" * 74)
    for lbl in ("full", "ex-2008", "ex-2008 & ex-2020"):
        b, o, t = base["net"], ovr["net"], trend["net"]
        if lbl != "full":
            b, o, t = [s[s.index.year != 2008] for s in (b, o, t)]
        if "2020" in lbl:
            b, o, t = [s[s.index.year != 2020] for s in (b, o, t)]
        print(f"  {lbl:<20} base {sh(b):.3f}   override {sh(o):.3f}   trend {sh(t):.3f}")

    print("\n" + "=" * 74)
    print("LAG TEST on the override variant")
    print("=" * 74)
    lr = reg.copy()
    c = ["score", "tilt_pp", "favours", "defensive", "rate", "liq"]
    lr[c] = lr[c].shift(21)
    lr = lr.dropna(subset=["score"])
    ls = st.copy()
    ls["stressed"] = ls["stressed"].shift(21).fillna(False)
    ol = run_with_override(rets, lr, ls)
    a, b_ = sh(ovr["net"]), sh(ol["net"])
    print(f"  {a:.4f} -> {b_:.4f}   ({100*(a-b_)/a:+.1f}% decay)")
