"""
Precompute for the rate-regime rotation page.

Runs only what the page displays. The old nine-asset ERC pipeline lives on in
precompute.py and is not touched; this writes to its own directory so the two
never collide.

Outputs to data/rotation/:

    regime_daily.parquet    rate and liquidity states, score, tilt, defensive sleeve
    weights_monthly.parquet six sleeve weights at each rebalance
    equity_monthly.parquet  strategy, benchmark, SPY - month-end index levels
    attribution.parquet     the four-step construction table
    periods.parquet         three-window comparison against the benchmark
    policy.parquet          the strategy at four risk budgets
    tests.parquet           the twenty tests and their verdicts
    metadata.json           current readings, dates, FRED values, headline stats

Nothing here formats anything. build_site.py reads these and fills the template.
"""
from __future__ import annotations

import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import config                                        # noqa: E402
from src.data import get_data                        # noqa: E402
from src.returns import compute_returns              # noqa: E402
from src.rate_regime import build_regime             # noqa: E402
import src.rotation as R                             # noqa: E402

OUT = ROOT / "data" / "rotation"

LONG = ["Growth", "Gold"]
SHORT = ["Value", "Energy"]
SLEEVES = LONG + SHORT + ["Bond_Long", "Bond_Short"]
TICKER = {"Growth": "IWF", "Gold": "GLD", "Value": "IWD", "Energy": "XLE",
          "Bond_Long": "IEF", "Bond_Short": "SHY"}
LABEL = {"Growth": "Growth", "Gold": "Gold", "Value": "Value",
         "Energy": "Energy", "Bond_Long": "US 7–10y", "Bond_Short": "US 1–3y"}

BUDGET = 0.55
NO_TRADE, COST_BPS, CRISIS_SHIFT, TILT_PP, VOL_WINDOW = 0.02, 5.0, 0.10, 10.0, 63
RF = None   # set at runtime from the 3-month T-bill
POLICY_BUDGETS = (0.70, 0.60, 0.55, 0.50)


# ─────────────────────────────────────────────────────────── helpers
def inverse_vol(cov: pd.DataFrame, cols: list) -> pd.Series:
    sd = np.sqrt(np.diag(cov.loc[cols, cols].values))
    if (sd <= 0).any() or not np.isfinite(sd).all():
        return pd.Series(1.0 / len(cols), index=cols)
    inv = 1.0 / sd
    return pd.Series(inv / inv.sum(), index=cols)


def sharpe(x: pd.Series) -> float:
    """Geometric excess Sharpe, matching stats() and every published figure."""
    yrs = len(x) / 252.0
    ann = float(np.exp(x.sum())) ** (1 / yrs) - 1
    vol = float(x.std() * np.sqrt(252))
    return (ann - RF) / vol


def stats(lr: pd.Series) -> dict:
    n = len(lr)
    yrs = n / 252.0
    tot = float(np.exp(lr.sum()))
    ann = tot ** (1 / yrs) - 1
    vol = float(lr.std() * np.sqrt(252))
    eq = np.exp(lr.cumsum())
    dd = float((eq / eq.cummax() - 1).min())
    return {"ann_return": ann, "ann_vol": vol,
            "sharpe": (ann - RF) / vol, "max_dd": dd,
            "calmar": ann / abs(dd) if dd < 0 else np.nan}


def backtest(rets, reg, budget=BUDGET, invvol=True, bonds="switch", tilt=True,
             long_c=None, short_c=None):
    """One configuration. bonds: switch | split | ief"""
    long_c = long_c or LONG
    short_c = short_c or SHORT
    sleeves = long_c + short_c + ["Bond_Long", "Bond_Short"]
    r = rets[sleeves].dropna()
    rg = reg.reindex(r.index).ffill().dropna(subset=["score"])
    r = r.loc[rg.index]

    hist_w, current = {}, None
    for d in r.resample("ME").last().index[:-1]:
        hist = r.loc[:d]
        if len(hist) < 300:
            continue
        cov = hist.tail(VOL_WINDOW).cov() * 252
        if invvol:
            lw, sw = inverse_vol(cov, long_c), inverse_vol(cov, short_c)
        else:
            lw = pd.Series(1.0 / len(long_c), index=long_c)
            sw = pd.Series(1.0 / len(short_c), index=short_c)

        row = rg.loc[:d].iloc[-1]
        score = int(row["score"]) if tilt else 0

        eq, dfn = budget, 1.0 - budget
        if tilt and score <= -3:
            eq -= CRISIS_SHIFT
            dfn += CRISIS_SHIFT

        t = (TILT_PP * score / 3.0) / 100.0
        lb, sb = max(eq / 2 + t, 0.0), max(eq / 2 - t, 0.0)
        tot = lb + sb
        if tot > 0:
            lb, sb = lb * eq / tot, sb * eq / tot

        w = pd.Series(0.0, index=sleeves)
        for c in long_c:
            w[c] = lb * lw[c]
        for c in short_c:
            w[c] = sb * sw[c]
        if bonds == "split":
            w["Bond_Long"] = w["Bond_Short"] = dfn / 2
        elif bonds == "ief":
            w["Bond_Long"] = dfn
        else:
            w[R.BONDS[row["defensive"]]] = dfn

        if current is not None and not ((w - current).abs() > NO_TRADE).any():
            w = current.copy()
        current = w
        hist_w[d] = current.copy()

    weights = pd.DataFrame(hist_w).T
    dw = weights.reindex(r.index, method="ffill").shift(1).dropna()
    al = r.loc[dw.index]
    to = dw.diff().abs().sum(axis=1).fillna(0.0)
    net = (dw * al).sum(axis=1) - to * COST_BPS / 10000.0
    return {"weights": weights, "net": net, "turnover": to}


def benchmark_e1(rets, budget=BUDGET):
    """Fixed equal weights, both bonds held. Bought once, never traded."""
    r = rets[SLEEVES].dropna()
    w = pd.Series(0.0, index=SLEEVES)
    for c in LONG + SHORT:
        w[c] = budget / 4
    w["Bond_Long"] = w["Bond_Short"] = (1 - budget) / 2
    return (r * w).sum(axis=1)


# ─────────────────────────────────────────────────────────── main
def main():
    t0 = datetime.now(timezone.utc)
    OUT.mkdir(parents=True, exist_ok=True)
    print("=" * 66)
    print("PRECOMPUTE — rate-regime rotation")
    print("=" * 66)

    panel, meta_data = get_data()
    rets = compute_returns(panel)

    # RF_RATE lives in the panel but not in the returns frame, so every Sharpe
    # computed without this line silently assumes a zero cash rate.
    global RF
    rf_series = panel["RF_RATE"].dropna() / 100.0
    RF = float(rf_series.mean())
    print(f"  risk-free: {RF*100:.2f}% average, {rf_series.iloc[-1]*100:.2f}% latest")
    reg = build_regime(panel)
    print(f"  panel {panel.index[0].date()} → {panel.index[-1].date()}")

    # ---- strategy and benchmark ----
    strat = backtest(rets, reg)
    bench = benchmark_e1(rets)
    common = strat["net"].index.intersection(bench.index)
    strat_net, bench_net = strat["net"].loc[common], bench.loc[common]
    spy = rets["Market"].loc[common] if "Market" in rets.columns else None
    print(f"  backtest {common[0].date()} → {common[-1].date()}  "
          f"({len(strat['weights'])} rebalances)")

    # ---- regime series ----
    keep = ["rate_chg_bp", "liq_chg_pct", "rate_raw", "liq_raw",
            "rate", "liq", "score", "tilt_pp", "defensive"]
    reg[[c for c in keep if c in reg.columns]].to_parquet(OUT / "regime_daily.parquet")

    # ---- weights ----
    strat["weights"].to_parquet(OUT / "weights_monthly.parquet")

    # ---- month-end index levels for the chart ----
    curves = pd.DataFrame({"strategy": strat_net, "benchmark": bench_net})
    if spy is not None:
        curves["spy"] = spy
    monthly = curves.resample("ME").sum()
    (np.exp(monthly.cumsum()) * 100).to_parquet(OUT / "equity_monthly.parquet")

    # daily log returns for the chart. The page recomputes its KPIs from these,
    # so they must be on the same basis as every figure in the tables.
    curves.to_parquet(OUT / "returns_daily.parquet")

    # ---- attribution ----
    steps = [
        ("Equal weight, both bonds", dict(invvol=False, bonds="split", tilt=False)),
        ("+ volatility weighting",   dict(invvol=True,  bonds="split", tilt=False)),
        ("+ switching bond sleeve",  dict(invvol=True,  bonds="switch", tilt=False)),
        ("+ regime tilt",            dict(invvol=True,  bonds="switch", tilt=True)),
    ]
    rows, prev = [], None
    for name, kw in steps:
        s = backtest(rets, reg, **kw)["net"].loc[common]
        sh = sharpe(s)
        rows.append({"step": name, **stats(s), "delta": np.nan if prev is None else sh - prev})
        prev = sh
    attribution = pd.DataFrame(rows)
    attribution.to_parquet(OUT / "attribution.parquet")

    # ---- three periods ----
    prows = []
    for lbl, start in (("Full · 2002–2026", None),
                       ("Post-crisis · from 2009", "2009-07-01"),
                       ("Last decade · from 2016", "2016-08-01")):
        seg = common if start is None else common[common >= start]
        a, b = sharpe(strat_net.loc[seg]), sharpe(bench_net.loc[seg])
        prows.append({"period": lbl, "strategy": a, "benchmark": b, "gain": a - b})
    periods = pd.DataFrame(prows)
    periods.to_parquet(OUT / "periods.parquet")

    # ---- policy table ----
    prows = []
    for b in POLICY_BUDGETS:
        s = backtest(rets, reg, budget=b)["net"].loc[common]
        prows.append({"budget": b, **stats(s)})
    policy = pd.DataFrame(prows)
    policy.to_parquet(OUT / "policy.parquet")

    # ---- bootstrap on the headline edge ----
    rng = np.random.default_rng(20260826)
    a, b = strat_net.dropna(), bench_net.reindex(strat_net.index).dropna()
    a = a.reindex(b.index)
    T = len(a)
    nb = int(np.ceil(T / 252))
    diffs = np.empty(4000)
    for i in range(4000):
        st = rng.integers(0, T - 252, size=nb)
        ii = np.concatenate([np.arange(s, s + 252) for s in st])[:T]
        ea, eb = a.iloc[ii], b.iloc[ii]
        diffs[i] = (ea.mean() / ea.std() - eb.mean() / eb.std()) * np.sqrt(252)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    boot = {"observed": sharpe(a) - sharpe(b), "ci_low": float(lo), "ci_high": float(hi),
            "p_value": float(2 * min((diffs <= 0).mean(), (diffs >= 0).mean()))}

    # ---- four sleeves vs six, for the gold/energy row in section 4 ----
    four = backtest(rets, reg, long_c=["Growth"], short_c=["Value"])["net"].loc[common]
    s4, s6 = stats(four), stats(strat_net)
    gold_energy_note = (
        f"Sharpe {s6['sharpe']:.3f} from {s4['sharpe']:.3f}, worst loss "
        f"{s6['max_dd']*100:.1f}% from {s4['max_dd']*100:.1f}%, turnover flat."
    ).replace("-", "\u2212")

    gv_corr = float(rets[["Growth", "Value"]].dropna().corr().iloc[0, 1])

    # ---- tests table (verdicts recorded, not recomputed daily) ----
    pd.DataFrame([
        {"test": "Gold and energy added", "verdict": "kept",
         "note": gold_energy_note},
        {"test": "200-day trend filter", "verdict": "rejected",
         "note": "Sharpe 0.98 — the best number of the twenty. But almost all of it "
                 "came from 2008 alone. Remove that one year and the advantage "
                 "nearly disappears."},
        {"test": "PCA correlation warning", "verdict": "rejected",
         "note": "Fired a year before 2008 and three months before Covid. Sees "
                 "fragility, not timing."},
        {"test": "Scaling bond duration by score", "verdict": "rejected",
         "note": "Worse in every period. Long bonds lose when rates rise, so partial "
                 "exposure is a partial loss with no offsetting benefit. In 2022 that "
                 "cost twice as much: −3.2% against −1.6%."},
        {"test": "Sixteen others", "verdict": "rejected",
         "note": "Momentum, credit spreads, vol targeting, GJR-GARCH, VIX, yield "
                 "curve, Taylor gap, HRP, currencies, and seven more."},
    ]).to_parquet(OUT / "tests.parquet")

    # ---- limitations, so the page never carries a stale number ----
    tilt_delta = float(attribution.iloc[-1]["delta"])
    pd.DataFrame([
        {"limit": f"Growth and value correlate {gv_corr:.2f}",
         "note": "This picks a side of one axis, it doesn't diversify."},
        {"limit": "Rates fall in easings and in panics",
         "note": "The model scores both alike; 2008 went the wrong way."},
        {"limit": f"The tilt adds +{tilt_delta:.3f} of the +{boot['observed']:.3f}",
         "note": "Volatility weighting and the bond switch do most of the work."},
        {"limit": "Reads current conditions", "note": "Forecasts nothing."},
    ]).to_parquet(OUT / "limitations.parquet")

    # ---- current readings ----
    last = reg.iloc[-1]
    latest_w = strat["weights"].iloc[-1]
    # Derive the window endpoints from the SAME shift the regime rule uses, so
    # the start value and the change always reconcile. Computing them
    # independently produced cards where 4.05% -> 4.34% sat beside "+26bp".
    from src.rate_regime import RATE_LOOKBACK_DAYS, LIQ_LOOKBACK_DAYS

    two_ff = panel["US_2Y"].ffill().dropna()
    wal_ff = panel["WALCL"].ffill().dropna()
    two_obs = panel["US_2Y"].dropna()
    wal_obs = panel["WALCL"].dropna()

    rate_to = float(two_ff.iloc[-1])
    rate_from = rate_to - float(last["rate_chg_bp"]) / 100.0
    rate_from_date = two_ff.index[-(RATE_LOOKBACK_DAYS + 1)]

    liq_to = float(wal_ff.iloc[-1])
    liq_from = liq_to / (1.0 + float(last["liq_chg_pct"]) / 100.0)
    liq_from_date = wal_ff.index[-(LIQ_LOOKBACK_DAYS + 1)]

    yrs = len(strat_net) / 252
    meta = {
        "generated_utc": t0.isoformat(),
        "risk_free": {"mean": RF, "latest": float(rf_series.iloc[-1]),
                      "series": "DGS3MO"},
        "data_as_of": str(panel.index[-1].date()),
        "rates": {
            "series": "DGS2", "last_obs": str(two_obs.index[-1].date()),
            "from_date": str(rate_from_date.date()),
            "from_value": rate_from, "to_value": rate_to,
            "change_bp": float(last["rate_chg_bp"]), "state": last["rate"],
        },
        "liquidity": {
            "series": "WALCL", "last_obs": str(wal_obs.index[-1].date()),
            "from_date": str(liq_from_date.date()),
            "from_value": liq_from, "to_value": liq_to,
            "change_pct": float(last["liq_chg_pct"]), "state": last["liq"],
        },
        "score": int(last["score"]),
        "tilt_pp": float(last["tilt_pp"]),
        "defensive": last["defensive"],
        "weights_set_on": str(strat["weights"].index[-1].date()),
        "budget": BUDGET,
        "weights": {TICKER[k]: float(v) for k, v in latest_w.items()},
        "buckets": {
            "long": float(latest_w[LONG].sum()),
            "short": float(latest_w[SHORT].sum()),
            "defensive": float(latest_w[["Bond_Long", "Bond_Short"]].sum()),
        },
        "headline": stats(strat_net),
        "benchmark": stats(bench_net),
        "spy": stats(spy) if spy is not None else None,
        "bootstrap": boot,
        "turnover": float(strat["turnover"].loc[common].sum() / yrs),
        "sample": {"start": str(common[0].date()), "end": str(common[-1].date()),
                   "years": round(yrs, 1)},
        "params": {"rate_band_bp": 25, "liq_band_pct": 1.0, "rate_weight": 2,
                   "liq_weight": 1, "dwell_weeks": 4, "tilt_pp": TILT_PP,
                   "cost_bps": COST_BPS, "no_trade_pp": NO_TRADE * 100},
        "elapsed_seconds": (datetime.now(timezone.utc) - t0).total_seconds(),
    }
    (OUT / "metadata.json").write_text(json.dumps(meta, indent=2))

    # ---- report ----
    print("\n  CURRENT")
    print(f"    score {meta['score']:+d}  tilt {meta['tilt_pp']:+.1f}pp  "
          f"defensive {meta['defensive']}")
    print(f"    rates {meta['rates']['change_bp']:+.0f}bp → {meta['rates']['state']}  "
          f"(to {meta['rates']['last_obs']})")
    print(f"    liquidity {meta['liquidity']['change_pct']:+.2f}% → "
          f"{meta['liquidity']['state']}  (to {meta['liquidity']['last_obs']})")

    h, bm = meta["headline"], meta["benchmark"]
    print("\n  HEADLINE")
    print(f"    strategy   {h['ann_return']*100:5.2f}%  vol {h['ann_vol']*100:5.2f}%  "
          f"sharpe {h['sharpe']:.3f}  maxDD {h['max_dd']*100:6.1f}%")
    print(f"    benchmark  {bm['ann_return']*100:5.2f}%  vol {bm['ann_vol']*100:5.2f}%  "
          f"sharpe {bm['sharpe']:.3f}  maxDD {bm['max_dd']*100:6.1f}%")
    print(f"    edge {boot['observed']:+.3f}  CI [{boot['ci_low']:+.3f}, "
          f"{boot['ci_high']:+.3f}]  p {boot['p_value']:.3f}")

    print("\n  ATTRIBUTION")
    for _, r_ in attribution.iterrows():
        d = "" if pd.isna(r_["delta"]) else f"  {r_['delta']:+.3f}"
        print(f"    {r_['step']:<26} {r_['sharpe']:.3f}{d}")

    print("\n  BY PERIOD")
    for _, r_ in periods.iterrows():
        print(f"    {r_['period']:<26} {r_['strategy']:.3f} vs {r_['benchmark']:.3f}"
              f"   {r_['gain']:+.3f}")

    print(f"\n  wrote {len(list(OUT.glob('*')))} files to {OUT}")
    print(f"  done in {meta['elapsed_seconds']:.0f}s")


if __name__ == "__main__":
    main()
