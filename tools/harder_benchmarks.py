"""
Harder benchmarks.

Equal weight is the correct scientific control - same assets, no decisions -
but it is a strawman nobody would actually run. A PM's first question is what
happens against the alternatives a real allocator would consider.

Five benchmarks, all at the same 55/45 risk budget where applicable, all with
the same monthly rebalance, no-trade band and 5bp costs:

    E1  equal weight, both bonds        the scientific control
    RP  risk parity, inverse vol        same assets, standard construction, no view
    RPc risk parity capped at 25%       because uncapped RP concentrates in SHY
    6040  60% SPY / 40% IEF             what a normal investor holds
    6040m 60% SPY / 40% AGG-proxy       the textbook version, if AGG is available

The interesting one is RP. Same six assets, a construction any allocator would
recognise, and no view at all. If the strategy cannot beat that, the macro
tilt is not earning its place.
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

from src.data import get_data                       # noqa: E402
from src.returns import compute_returns             # noqa: E402
from src.rate_regime import build_regime            # noqa: E402
import src.rotation as R                            # noqa: E402

LONG, SHORT = ["Growth", "Gold"], ["Value", "Energy"]
SLEEVES = LONG + SHORT + ["Bond_Long", "Bond_Short"]
BUDGET = 0.55
NO_TRADE, COST_BPS, CRISIS_SHIFT, TILT_PP, VOL_WINDOW = 0.02, 5.0, 0.10, 10.0, 63
RF = None


def inv_vol(cov, cols):
    sd = np.sqrt(np.diag(cov.loc[cols, cols].values))
    if (sd <= 0).any() or not np.isfinite(sd).all():
        return pd.Series(1.0 / len(cols), index=cols)
    inv = 1.0 / sd
    return pd.Series(inv / inv.sum(), index=cols)


def capped_inv_vol(cov, cols, cap):
    w = inv_vol(cov, cols)
    for _ in range(50):
        over = w > cap
        if not over.any():
            break
        excess = (w[over] - cap).sum()
        w[over] = cap
        under = ~over
        if not under.any() or w[under].sum() == 0:
            break
        w[under] += excess * w[under] / w[under].sum()
    return w / w.sum()


def strategy(rets, reg):
    r = rets[SLEEVES].dropna()
    rg = reg.reindex(r.index).ffill().dropna(subset=["score"])
    r = r.loc[rg.index]
    hw, cur = {}, None
    for d in r.resample("ME").last().index[:-1]:
        hist = r.loc[:d]
        if len(hist) < 300:
            continue
        cov = hist.tail(VOL_WINDOW).cov() * 252
        lw, sw = inv_vol(cov, LONG), inv_vol(cov, SHORT)
        row = rg.loc[:d].iloc[-1]
        s = int(row["score"])
        eq, dfn = BUDGET, 1 - BUDGET
        if s <= -3:
            eq -= CRISIS_SHIFT
            dfn += CRISIS_SHIFT
        t = (TILT_PP * s / 3.0) / 100.0
        lb, sb = max(eq / 2 + t, 0), max(eq / 2 - t, 0)
        tot = lb + sb
        if tot > 0:
            lb, sb = lb * eq / tot, sb * eq / tot
        w = pd.Series(0.0, index=SLEEVES)
        for c in LONG:
            w[c] = lb * lw[c]
        for c in SHORT:
            w[c] = sb * sw[c]
        w[R.BONDS[row["defensive"]]] = dfn
        if cur is not None and not ((w - cur).abs() > NO_TRADE).any():
            w = cur.copy()
        cur = w
        hw[d] = cur.copy()
    return net_of(pd.DataFrame(hw).T, r)


def risk_parity(rets, cap=None):
    r = rets[SLEEVES].dropna()
    hw, cur = {}, None
    for d in r.resample("ME").last().index[:-1]:
        hist = r.loc[:d]
        if len(hist) < 300:
            continue
        cov = hist.tail(VOL_WINDOW).cov() * 252
        w = capped_inv_vol(cov, SLEEVES, cap) if cap else inv_vol(cov, SLEEVES)
        if cur is not None and not ((w - cur).abs() > NO_TRADE).any():
            w = cur.copy()
        cur = w
        hw[d] = cur.copy()
    return net_of(pd.DataFrame(hw).T, r)


def static(rets, weights):
    cols = list(weights)
    r = rets[cols].dropna()
    w = pd.Series(weights)
    return (r * w).sum(axis=1), pd.Series(0.0, index=r.index)


def net_of(weights, r):
    dw = weights.reindex(r.index, method="ffill").shift(1).dropna()
    al = r.loc[dw.index]
    to = dw.diff().abs().sum(axis=1).fillna(0)
    return (dw * al).sum(axis=1) - to * COST_BPS / 10000.0, to


def sharpe(x):
    yrs = len(x) / 252
    ann = float(np.exp(x.sum())) ** (1 / yrs) - 1
    vol = float(x.std() * np.sqrt(252))
    return (ann - RF) / vol


def stat(x):
    yrs = len(x) / 252
    ann = float(np.exp(x.sum())) ** (1 / yrs) - 1
    vol = float(x.std() * np.sqrt(252))
    eq = np.exp(x.cumsum())
    return ann, vol, (ann - RF) / vol, float((eq / eq.cummax() - 1).min())


if __name__ == "__main__":
    panel, _ = get_data()
    rets = compute_returns(panel)
    reg = build_regime(panel)
    RF = float((panel["RF_RATE"].dropna() / 100.0).mean())
    print(f"risk-free {RF*100:.2f}%\n")

    runs = {}
    runs["STRATEGY"] = strategy(rets, reg)
    runs["E1  equal weight"] = static(
        rets, {c: BUDGET / 4 for c in LONG + SHORT} |
              {"Bond_Long": (1 - BUDGET) / 2, "Bond_Short": (1 - BUDGET) / 2})
    runs["RP  risk parity"] = risk_parity(rets)
    runs["RPc risk parity, 25% cap"] = risk_parity(rets, cap=0.25)
    runs["60/40  SPY + IEF"] = static(
        rets, {"Market": 0.60, "Bond_Long": 0.40})

    common = None
    for net, _ in runs.values():
        common = net.index if common is None else common.intersection(net.index)

    print("=" * 88)
    print(f"AGAINST HARDER BENCHMARKS  ({common[0].date()} to {common[-1].date()})")
    print("=" * 88)
    print(f"{'':<28}{'ann ret':>9}{'vol':>8}{'sharpe':>9}{'maxDD':>9}"
          f"{'turnover':>10}{'vs strat':>11}")
    print("-" * 88)
    s0 = sharpe(runs["STRATEGY"][0].loc[common])
    for k, (net, to) in runs.items():
        s = net.loc[common]
        a, v, sh, dd = stat(s)
        t = to.loc[common].sum() / (len(common) / 252) if to.sum() else 0.0
        d = "" if k == "STRATEGY" else f"{s0 - sh:>+11.3f}"
        print(f"{k:<28}{a*100:>8.2f}%{v*100:>7.2f}%{sh:>9.3f}{dd*100:>8.1f}%"
              f"{t:>9.0%}{d}")

    print("\n" + "=" * 88)
    print("STRATEGY EDGE BY PERIOD  (positive = strategy ahead)")
    print("=" * 88)
    print(f"{'benchmark':<28}{'full':>12}{'post-09':>12}{'last 10y':>12}")
    print("-" * 88)
    for k, (net, _) in runs.items():
        if k == "STRATEGY":
            continue
        line = f"{k:<28}"
        for start in (None, "2009-07-01", "2016-08-01"):
            seg = common if start is None else common[common >= start]
            line += f"{sharpe(runs['STRATEGY'][0].loc[seg]) - sharpe(net.loc[seg]):>+12.3f}"
        print(line)

    print("\n" + "=" * 88)
    print("BOOTSTRAP  (4,000 resamples, 252-day blocks)")
    print("=" * 88)
    rng = np.random.default_rng(20260906)
    A = runs["STRATEGY"][0].loc[common].dropna()
    for k, (net, _) in runs.items():
        if k == "STRATEGY":
            continue
        B = net.loc[common].reindex(A.index).dropna()
        a = A.reindex(B.index)
        T = len(a)
        nb = int(np.ceil(T / 252))
        obs = sharpe(a) - sharpe(B)
        d = np.empty(4000)
        for i in range(4000):
            st = rng.integers(0, T - 252, size=nb)
            ii = np.concatenate([np.arange(s, s + 252) for s in st])[:T]
            ea, eb = a.iloc[ii], B.iloc[ii]
            d[i] = (ea.mean() / ea.std() - eb.mean() / eb.std()) * np.sqrt(252)
        lo, hi = np.percentile(d, [2.5, 97.5])
        p = 2 * min((d <= 0).mean(), (d >= 0).mean())
        flag = "  SIGNIFICANT" if (lo > 0 or hi < 0) else ""
        print(f"  {k:<28}{obs:>+8.3f}  CI [{lo:+.3f}, {hi:+.3f}]  p {p:.3f}{flag}")

    print("\n" + "=" * 88)
    print("WHERE UNCAPPED RISK PARITY PUTS THE MONEY")
    print("=" * 88)
    r = rets[SLEEVES].dropna()
    cov = r.tail(VOL_WINDOW).cov() * 252
    w = inv_vol(cov, SLEEVES)
    for c in SLEEVES:
        print(f"  {c:<12}{w[c]:>7.1%}   vol {np.sqrt(cov.loc[c, c])*100:>5.1f}%")
