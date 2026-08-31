"""
Extended universe test: add gold and energy to the duration axis.

Positioning rule, fixed BEFORE running:

    Long duration bucket   IWF (growth equity) + GLD (gold)
    Short duration bucket  IWD (value equity)  + XLE (energy)

Gold has no cash flow at all, so its value is a pure discounted claim on
scarcity - the longest duration asset available. Energy generates cash now
and its revenues are linked to the inflation that usually accompanies rising
rates - the shortest duration equity exposure available.

The 70/30 policy split, the scoring rule, the tilt size and the defensive
sleeve are all unchanged. Weights within each bucket are set by equal risk
contribution; the tilt moves between buckets rather than between single names.

Nothing is written to config until the result justifies it.
"""
from __future__ import annotations

from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import src.rotation as R                        # noqa: E402
from src.data import get_data                   # noqa: E402
from src.returns import compute_returns         # noqa: E402
from src.rate_regime import build_regime        # noqa: E402

LONG_BUCKET = ["Growth", "Gold"]
SHORT_BUCKET = ["Value", "Energy"]
EXTRA = {"Gold": "GLD", "Energy": "XLE"}

EQUITY_BUDGET = 0.70
DEFENSIVE_BUDGET = 0.30
CRISIS_SHIFT = 0.10
TILT_PP = 10.0
NO_TRADE = 0.02
COST_BPS = 5.0


def inverse_vol(cov: pd.DataFrame, cols: list) -> pd.Series:
    sd = np.sqrt(np.diag(cov.loc[cols, cols].values))
    if (sd <= 0).any():
        return pd.Series(1.0 / len(cols), index=cols)
    inv = 1.0 / sd
    return pd.Series(inv / inv.sum(), index=cols)


def extended_targets(score: int, defensive_etf: str,
                     long_w: pd.Series, short_w: pd.Series,
                     sleeves: list) -> pd.Series:
    eq, dfn = EQUITY_BUDGET, DEFENSIVE_BUDGET
    if score <= -3:
        eq -= CRISIS_SHIFT
        dfn += CRISIS_SHIFT

    tilt = (TILT_PP * score / 3.0) / 100.0
    long_budget = eq / 2 + tilt
    short_budget = eq / 2 - tilt
    long_budget = max(long_budget, 0.0)
    short_budget = max(short_budget, 0.0)
    tot = long_budget + short_budget
    if tot > 0:
        long_budget, short_budget = long_budget * eq / tot, short_budget * eq / tot

    w = pd.Series(0.0, index=sleeves)
    for c in LONG_BUCKET:
        w[c] = long_budget * long_w[c]
    for c in SHORT_BUCKET:
        w[c] = short_budget * short_w[c]
    w[R.BONDS[defensive_etf]] = dfn
    return w


def run_extended(rets: pd.DataFrame, reg: pd.DataFrame,
                 cost_bps: float = COST_BPS, vol_window: int = 63) -> dict:
    sleeves = LONG_BUCKET + SHORT_BUCKET + ["Bond_Long", "Bond_Short"]
    r = rets[sleeves].dropna()
    reg = reg.reindex(r.index).ffill().dropna(subset=["score"])
    r = r.loc[reg.index]

    w_hist, current = {}, None
    for d in r.resample("ME").last().index[:-1]:
        hist = r.loc[:d]
        if len(hist) < vol_window:
            continue
        cov = hist.tail(vol_window).cov() * 252
        lw = inverse_vol(cov, LONG_BUCKET)
        sw = inverse_vol(cov, SHORT_BUCKET)
        row = reg.loc[:d].iloc[-1]
        tgt = extended_targets(int(row["score"]), row["defensive"], lw, sw, sleeves)
        if current is not None and not ((tgt - current).abs() > NO_TRADE).any():
            tgt = current.copy()
        current = tgt
        w_hist[d] = current.copy()

    weights = pd.DataFrame(w_hist).T
    daily_w = weights.reindex(r.index, method="ffill").shift(1).dropna()
    aligned = r.loc[daily_w.index]

    gross = (daily_w * aligned).sum(axis=1)
    turnover = daily_w.diff().abs().sum(axis=1).fillna(0.0)
    net = gross - turnover * cost_bps / 10000.0

    # same policy, no tilt
    flat = pd.Series(0.0, index=sleeves)
    for c in LONG_BUCKET + SHORT_BUCKET:
        flat[c] = EQUITY_BUDGET / 4
    flat["Bond_Long"] = DEFENSIVE_BUDGET
    no_tilt = (aligned * flat).sum(axis=1)

    return {"weights": weights, "net": net, "turnover": turnover,
            "no_tilt": no_tilt}


if __name__ == "__main__":
    print("checking inception dates...")
    for name, t in EXTRA.items():
        h = yf.download(t, period="max", progress=False, auto_adjust=False)
        print(f"  {name:8} {t:5} {h.index[0].date()}  ({len(h):,} rows)")

    panel, _ = get_data()
    px = yf.download(list(EXTRA.values()), start="2002-07-30",
                     progress=False, auto_adjust=False)["Adj Close"]
    px.columns = [k for k, v in EXTRA.items() for c in px.columns if v == c] \
        if len(px.columns) == 2 else px.columns
    px = px.rename(columns={v: k for k, v in EXTRA.items()})

    ext = np.log(px / px.shift(1))
    rets = compute_returns(panel).join(ext, how="inner")
    reg = build_regime(panel)
    rf = rets["RF_RATE"] if "RF_RATE" in rets.columns else None

    print(f"\nextended sample starts {rets.dropna().index[0].date()}\n")

    print("=" * 70)
    print("CORRELATIONS  (this is the point of the exercise)")
    print("=" * 70)
    cols = ["Growth", "Gold", "Value", "Energy", "Bond_Long", "Bond_Short"]
    print(rets[cols].dropna().corr().round(3).to_string())

    print("\n" + "=" * 70)
    print("ANNUALISED VOL (%)")
    print("=" * 70)
    print((rets[cols].dropna().std() * np.sqrt(252) * 100).round(1).to_string())

    old = R.run_backtest(rets, reg)
    new = run_extended(rets, reg)

    common = old["net"].index.intersection(new["net"].index)
    rows = [
        R.summarise(new["net"].loc[common], rf, "6 sleeves (extended)"),
        R.summarise(new["no_tilt"].loc[common], rf, "6 sleeves, no tilt"),
        R.summarise(old["net"].loc[common], rf, "4 sleeves (current)"),
        R.summarise(old["bench_policy"].loc[common], rf, "4 sleeves, no tilt"),
        R.summarise(old["spy"].loc[common], rf, "SPY (reference)"),
    ]
    print("\n" + "=" * 70)
    print("HEADLINE  (common sample)")
    print("=" * 70)
    print((pd.DataFrame(rows).set_index("name") * 100).round(2).to_string())

    def sh(s):
        e = s - (rf.reindex(s.index).fillna(0) if rf is not None else 0)
        return float(e.mean() / e.std() * np.sqrt(252))

    print("\n" + "=" * 70)
    print("PERIOD ROBUSTNESS  (Sharpe)")
    print("=" * 70)
    for lbl, start in [("full", None), ("post-2009", "2009-07-01"),
                       ("last 10y", "2016-08-01")]:
        seg = common if start is None else common[common >= start]
        print(f"  {lbl:<11} ext {sh(new['net'].loc[seg]):.3f}  "
              f"ext-no-tilt {sh(new['no_tilt'].loc[seg]):.3f}  "
              f"4slv {sh(old['net'].loc[seg]):.3f}  "
              f"4slv-no-tilt {sh(old['bench_policy'].loc[seg]):.3f}")

    print("\nTURNOVER")
    for nm, x in [("extended", new), ("current", old)]:
        print(f"  {nm:<10} {x['turnover'].sum()/(len(x['turnover'])/252):>7.1%}")

    print("\nLATEST WEIGHTS (%)")
    print((new["weights"].iloc[-1] * 100).round(2).to_string())
