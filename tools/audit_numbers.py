"""
Audit: every number on docs/index.html against data/rotation/.

Prints what the page says, what the artifacts say, and flags disagreements.
Also recomputes the two figures section 4 needs but the pipeline does not yet
produce: the four-sleeve comparison for the gold/energy row, and the
growth/value correlation.
"""
from __future__ import annotations

import json
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

ART = ROOT / "data" / "rotation"
PAGE = ROOT / "docs" / "index.html"

m = json.loads((ART / "metadata.json").read_text())
attr = pd.read_parquet(ART / "attribution.parquet")
per = pd.read_parquet(ART / "periods.parquet")
pol = pd.read_parquet(ART / "policy.parquet")
html = PAGE.read_text()

h, bm, boot = m["headline"], m["benchmark"], m["bootstrap"]
RF = m["risk_free"]["mean"]

ok, bad = [], []


def chk(label, expected, present=None):
    """expected: the string the page should contain."""
    found = expected in html
    (ok if found else bad).append((label, expected, found))
    print(f"  {'ok  ' if found else 'MISS'}  {label:<38} {expected}")


print("=" * 78)
print("NUMBERS THE PIPELINE PRODUCES  (should all be on the page)")
print("=" * 78)
chk("headline Sharpe", f"{h['sharpe']:.3f}")
chk("benchmark Sharpe", f"{bm['sharpe']:.3f}")
chk("edge", f"{boot['observed']:+.3f}".replace("-", "−"))
chk("bootstrap p", f"{boot['p_value']:.3f}")
chk("ann return", f"{h['ann_return']*100:.2f}%")
chk("ann vol", f"{h['ann_vol']*100:.2f}%")
chk("max drawdown", f"{h['max_dd']*100:.1f}%".replace("-", "−"))
for _, p in per.iterrows():
    chk(f"period {p['period'][:18]}", f"{p['gain']:+.3f}".replace("-", "−"))

print()
print("=" * 78)
print("SECTION 4 — HARDCODED, NOT DRIVEN BY THE ARTIFACTS")
print("=" * 78)

tilt_delta = float(attr.iloc[-1]["delta"])
total_edge = float(boot["observed"])
stale = [
    ("limitations: tilt contribution", "+0.036 of the +0.139",
     f"+{tilt_delta:.3f} of the +{total_edge:.3f}"),
    ("tests: gold/energy sharpe", "Sharpe 0.812 from 0.744", "recompute below"),
    ("tests: gold/energy drawdown", "−24.0% from −29.9%", "recompute below"),
]
for label, on_page, should_be in stale:
    present = on_page in html
    print(f"  {'FOUND' if present else '  -  '}  {label}")
    print(f"           page says : {on_page}")
    print(f"           should be : {should_be}")

print()
print("=" * 78)
print("RECOMPUTING THE SECTION 4 FIGURES AT THE CORRECT BASIS")
print("=" * 78)

from src.data import get_data                       # noqa: E402
from src.returns import compute_returns             # noqa: E402
from src.rate_regime import build_regime            # noqa: E402
import src.rotation as R                            # noqa: E402

panel, _ = get_data()
rets = compute_returns(panel)
reg = build_regime(panel)

LONG4, SHORT4 = ["Growth"], ["Value"]
LONG6, SHORT6 = ["Growth", "Gold"], ["Value", "Energy"]
BUDGET, VOL_WINDOW = 0.55, 63
NO_TRADE, COST_BPS, CRISIS_SHIFT, TILT_PP = 0.02, 5.0, 0.10, 10.0


def inv_vol(cov, cols):
    if len(cols) == 1:
        return pd.Series(1.0, index=cols)
    sd = np.sqrt(np.diag(cov.loc[cols, cols].values))
    inv = 1.0 / sd
    return pd.Series(inv / inv.sum(), index=cols)


def run(long_c, short_c):
    sl = long_c + short_c + ["Bond_Long", "Bond_Short"]
    r = rets[sl].dropna()
    rg = reg.reindex(r.index).ffill().dropna(subset=["score"])
    r = r.loc[rg.index]
    hw, cur = {}, None
    for d in r.resample("ME").last().index[:-1]:
        hist = r.loc[:d]
        if len(hist) < 300:
            continue
        cov = hist.tail(VOL_WINDOW).cov() * 252
        lw, sw = inv_vol(cov, long_c), inv_vol(cov, short_c)
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
        w = pd.Series(0.0, index=sl)
        for c in long_c:
            w[c] = lb * lw[c]
        for c in short_c:
            w[c] = sb * sw[c]
        w[R.BONDS[row["defensive"]]] = dfn
        if cur is not None and not ((w - cur).abs() > NO_TRADE).any():
            w = cur.copy()
        cur = w
        hw[d] = cur.copy()
    wt = pd.DataFrame(hw).T
    dw = wt.reindex(r.index, method="ffill").shift(1).dropna()
    al = r.loc[dw.index]
    to = dw.diff().abs().sum(axis=1).fillna(0)
    return (dw * al).sum(axis=1) - to * COST_BPS / 10000.0, to


def stat(lr):
    yrs = len(lr) / 252
    ann = float(np.exp(lr.sum())) ** (1 / yrs) - 1
    vol = float(lr.std() * np.sqrt(252))
    eq = np.exp(lr.cumsum())
    return ann, vol, (ann - RF) / vol, float((eq / eq.cummax() - 1).min())


n4, t4 = run(LONG4, SHORT4)
n6, t6 = run(LONG6, SHORT6)
c = n4.index.intersection(n6.index)
a4, v4, s4, d4 = stat(n4.loc[c])
a6, v6, s6, d6 = stat(n6.loc[c])
yrs = len(c) / 252

print(f"  four sleeves : sharpe {s4:.3f}  maxDD {d4*100:.1f}%  "
      f"turnover {t4.loc[c].sum()/yrs:.0%}")
print(f"  six sleeves  : sharpe {s6:.3f}  maxDD {d6*100:.1f}%  "
      f"turnover {t6.loc[c].sum()/yrs:.0%}")
print()
print(f"  → gold/energy row should read:")
print(f"    Sharpe {s6:.3f} from {s4:.3f}, worst loss "
      f"{d6*100:.1f}% from {d4*100:.1f}%, turnover flat.".replace("-", "−"))

corr = rets[["Growth", "Value"]].dropna().corr().iloc[0, 1]
print(f"\n  growth/value correlation: {corr:.3f}   (page says 0.86)")

print("\n" + "=" * 78)
print(f"SUMMARY: {len(ok)} pipeline numbers on the page, {len(bad)} missing")
print("=" * 78)
if bad:
    for label, exp, _ in bad:
        print(f"  MISSING  {label}: {exp}")
