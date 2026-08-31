"""
Volatility targeting via recursive GARCH.

Replaces the binary trend filter with continuous scaling of equity exposure.
At each monthly rebalance the portfolio's forward volatility is forecast from
GARCH(1,1)-t conditional variances and a trailing correlation matrix; equity
exposure is then scaled so forecast volatility sits at the target. Anything
scaled out of equity goes to the regime-selected bond.

    scale = target_vol / forecast_vol,  capped at [0.25, 1.00]

The cap at 1.00 means no leverage: this can de-risk but never gear up.

GARCH parameters are refitted every January on an expanding window and the
conditional variance is then filtered FORWARD using only past returns, so no
future information enters the forecast. The lag test will confirm this.

Moreira & Muir (2017), Volatility-Managed Portfolios, Journal of Finance.
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

from arch import arch_model                          # noqa: E402
import src.rotation as R                             # noqa: E402
from src.data import get_data                        # noqa: E402
from src.returns import compute_returns              # noqa: E402
from src.rate_regime import build_regime             # noqa: E402

TARGET_VOL = 0.10          # annualised
SCALE_MIN, SCALE_MAX = 0.25, 1.00
CORR_WINDOW = 252
MIN_OBS = 750


def recursive_garch_vol(r: pd.Series, min_obs: int = MIN_OBS) -> pd.Series:
    """
    One-day-ahead conditional volatility, refitted each January on an
    expanding window and filtered forward with past returns only.
    """
    r = r.dropna()
    out = pd.Series(np.nan, index=r.index)
    years = sorted(r.index.year.unique())

    for y in years:
        train = r[r.index < f"{y}-01-01"]
        if len(train) < min_obs:
            continue
        sd = float(train.std())
        scale = 10.0 ** round(np.log10(10.0 / sd)) if sd > 0 else 100.0
        try:
            res = arch_model(train * scale, mean="constant", vol="GARCH",
                             p=1, q=1, dist="t", rescale=False
                             ).fit(disp="off", show_warning=False)
            p = res.params
            omega, alpha, beta = (float(p["omega"]), float(p["alpha[1]"]),
                                  float(p["beta[1]"]))
            mu = float(p["mu"])
        except Exception:
            continue

        # filter forward through year y using only realised past returns
        hist = r[r.index < f"{y}-01-01"] * scale
        h = float(hist.var())
        for t in hist.index[-250:]:                       # warm up the recursion
            e = float(hist.loc[t]) - mu
            h = omega + alpha * e * e + beta * h

        yr = r[(r.index >= f"{y}-01-01") & (r.index < f"{y+1}-01-01")] * scale
        for t, v in yr.items():
            out.loc[t] = np.sqrt(h) / scale               # forecast made BEFORE seeing v
            e = float(v) - mu
            h = omega + alpha * e * e + beta * h

    return out * np.sqrt(252)


def forecast_portfolio_vol(w: pd.Series, sig: pd.Series,
                           corr: pd.DataFrame) -> float:
    """Annualised forward vol of a weight vector given sleeve vols + correlation."""
    cols = [c for c in w.index if c in sig.index and c in corr.columns]
    ww = w[cols].values
    ss = sig[cols].values
    cc = corr.loc[cols, cols].values
    cov = np.outer(ss, ss) * cc
    var = float(ww @ cov @ ww)
    return float(np.sqrt(max(var, 0.0)))


def run_voltarget(rets: pd.DataFrame, reg: pd.DataFrame,
                  garch_vol: pd.DataFrame,
                  target: float = TARGET_VOL,
                  cost_bps: float = 5.0) -> dict:
    cols = R.ALL_SLEEVES
    r = rets[cols].dropna()
    reg = reg.reindex(r.index).ffill().dropna(subset=["score"])
    r = r.loc[reg.index]
    gv = garch_vol.reindex(r.index).ffill()

    w_hist, current, diag = {}, None, {}
    for d in r.resample("ME").last().index[:-1]:
        hist = r.loc[:d]
        if len(hist) < 300 or d not in gv.index:
            continue
        sig = gv.loc[d].dropna()
        if len(sig) < 4:
            continue

        cov = hist[R.EQUITY].tail(63).cov().values * 252
        split = R.erc_two_asset(cov)
        row = reg.loc[:d].iloc[-1]
        tgt = R.target_weights(int(row["score"]), row["defensive"], split)

        corr = hist.tail(CORR_WINDOW).corr()
        fv = forecast_portfolio_vol(tgt, sig, corr)
        scale = np.clip(target / fv, SCALE_MIN, SCALE_MAX) if fv > 0 else 1.0

        scaled = tgt.copy()
        moved = 0.0
        for c in R.EQUITY:
            keep = scaled[c] * scale
            moved += scaled[c] - keep
            scaled[c] = keep
        scaled[R.BONDS[row["defensive"]]] += moved

        current = R.apply_no_trade_band(scaled, current)
        w_hist[d] = current.copy()
        diag[d] = {"forecast_vol": fv, "scale": scale}

    weights = pd.DataFrame(w_hist).T
    daily_w = weights.reindex(r.index, method="ffill").shift(1).dropna()
    aligned = r.loc[daily_w.index]

    gross = (daily_w * aligned).sum(axis=1)
    turnover = daily_w.diff().abs().sum(axis=1).fillna(0.0)
    net = gross - turnover * cost_bps / 10000.0

    return {"weights": weights, "net": net, "gross": gross,
            "turnover": turnover, "diag": pd.DataFrame(diag).T}


if __name__ == "__main__":
    panel, _ = get_data()
    rets = compute_returns(panel)
    reg = build_regime(panel)
    rf = rets["RF_RATE"] if "RF_RATE" in rets.columns else None
    prices = panel[R.EQUITY]

    print("fitting recursive GARCH per sleeve (refit each January)...")
    gv = pd.DataFrame({c: recursive_garch_vol(rets[c]) for c in R.ALL_SLEEVES})
    print(f"  forecasts available from {gv.dropna(how='all').index[0].date()}\n")

    base = R.run_backtest(rets, reg)
    trend = R.run_backtest(rets, reg, prices=prices, use_trend=True)
    vt = run_voltarget(rets, reg, gv)

    rows = [
        R.summarise(base["net"], rf, "Rotation (base)"),
        R.summarise(trend["net"], rf, "+ 200d trend"),
        R.summarise(vt["net"], rf, "+ vol target 10%"),
        R.summarise(base["bench_policy"], rf, "No tilt"),
        R.summarise(base["spy"], rf, "SPY (reference)"),
    ]
    print("=" * 74)
    print("HEADLINE")
    print("=" * 74)
    print((pd.DataFrame(rows).set_index("name") * 100).round(2).to_string())

    print("\nTURNOVER")
    for nm, x in [("base", base), ("trend", trend), ("voltarget", vt)]:
        yrs = len(x["turnover"]) / 252
        print(f"  {nm:<10} {x['turnover'].sum()/yrs:>7.1%}")

    print("\n" + "=" * 74)
    print("CALENDAR YEARS (%)")
    print("=" * 74)
    yr = pd.DataFrame({
        "base": np.exp(base["net"].resample("YE").sum()) - 1,
        "trend": np.exp(trend["net"].resample("YE").sum()) - 1,
        "voltgt": np.exp(vt["net"].resample("YE").sum()) - 1,
        "no_tilt": np.exp(base["bench_policy"].resample("YE").sum()) - 1,
    })
    yr.index = yr.index.year
    print((yr * 100).round(1).to_string())

    print("\n" + "=" * 74)
    print("EX-2008 CHECK")
    print("=" * 74)
    def sh(s):
        e = s - (rf.reindex(s.index).fillna(0) if rf is not None else 0)
        return float(e.mean() / e.std() * np.sqrt(252))
    for lbl in ("full", "ex-2008"):
        b, t, v = base["net"], trend["net"], vt["net"]
        if lbl == "ex-2008":
            b, t, v = [s[s.index.year != 2008] for s in (b, t, v)]
        print(f"  {lbl:<9} base {sh(b):.3f}   trend {sh(t):.3f}   voltgt {sh(v):.3f}")

    print("\n" + "=" * 74)
    print("EQUITY SCALING OVER TIME (annual mean)")
    print("=" * 74)
    d = vt["diag"]
    d.index = pd.to_datetime(d.index)
    ann = d.resample("YE").mean()
    ann.index = ann.index.year
    print(ann.round(3).to_string())
