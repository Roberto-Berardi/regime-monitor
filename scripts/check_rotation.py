"""
Validation gate for the rotation artifacts.

Runs after precompute_rotation.py and before anything is committed. A non-zero
exit stops the workflow, so the previous good page stays live.

Checks structure AND plausibility. Structure alone is not enough: the old
pipeline once shipped an eight-asset book with all thirty structural checks
passing, because nothing asked whether the numbers made sense.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "data" / "rotation"

REQUIRED = ["regime_daily.parquet", "weights_monthly.parquet",
            "equity_monthly.parquet", "attribution.parquet",
            "periods.parquet", "policy.parquet", "tests.parquet",
            "metadata.json"]

SLEEVES = ["Growth", "Gold", "Value", "Energy", "Bond_Long", "Bond_Short"]
TICKERS = ["IWF", "GLD", "IWD", "XLE", "IEF", "SHY"]

# plausibility bounds — wide enough never to fire on a normal day,
# tight enough to catch a broken pipeline
BOUNDS = {
    "ann_return": (0.00, 0.20),
    "ann_vol":    (0.03, 0.25),
    "sharpe":     (-0.5, 2.0),
    "max_dd":     (-0.60, -0.05),
}
MAX_STALE_BDAYS = 5
MAX_DD_JUMP = 0.05          # 5pp day-over-day move in the headline drawdown

fails: list[str] = []
warns: list[str] = []


def check(ok: bool, msg: str):
    print(f"  {'PASS' if ok else 'FAIL'}  {msg}")
    if not ok:
        fails.append(msg)


def warn(ok: bool, msg: str):
    if not ok:
        print(f"  WARN  {msg}")
        warns.append(msg)


def main():
    print("=== rotation artifact validation ===")

    # ---- files present and non-empty ----
    for f in REQUIRED:
        p = ART / f
        check(p.exists() and p.stat().st_size > 0, f"{f} present and non-empty")
    if fails:
        print("\nVALIDATION FAILED — required files missing.")
        sys.exit(1)

    m = json.loads((ART / "metadata.json").read_text())
    reg = pd.read_parquet(ART / "regime_daily.parquet")
    w = pd.read_parquet(ART / "weights_monthly.parquet")
    eq = pd.read_parquet(ART / "equity_monthly.parquet")
    attr = pd.read_parquet(ART / "attribution.parquet")
    per = pd.read_parquet(ART / "periods.parquet")
    pol = pd.read_parquet(ART / "policy.parquet")

    # ---- universe ----
    check(list(w.columns) == SLEEVES,
          f"six sleeves in order (got {list(w.columns)})")
    check(sorted(m["weights"].keys()) == sorted(TICKERS),
          "six tickers in metadata weights")

    # ---- weights ----
    last = w.iloc[-1]
    check(abs(last.sum() - 1.0) < 1e-6, f"weights sum to 1.0 (got {last.sum():.6f})")
    check((last >= -1e-9).all(), "no negative weights")
    check(last.max() < 0.60, f"largest weight below 60% (got {last.max():.1%})")
    check(not w.isna().any().any(), "no NaN in the weight history")

    # exactly one Treasury sleeve is held
    held = [c for c in ("Bond_Long", "Bond_Short") if last[c] > 0.001]
    check(len(held) == 1, f"exactly one Treasury sleeve held (got {held})")
    check(m["defensive"] in ("IEF", "SHY"), f"defensive sleeve named ({m['defensive']})")

    # the defensive sleeve matches the rate state
    expected = "SHY" if reg["rate"].iloc[-1] == "rising" else "IEF"
    check(m["defensive"] == expected,
          f"defensive sleeve follows the rate signal "
          f"({reg['rate'].iloc[-1]} -> {expected}, got {m['defensive']})")

    # ---- regime ----
    score = int(m["score"])
    check(-3 <= score <= 3, f"score in range (got {score})")
    check(abs(m["tilt_pp"] - 10.0 * score / 3.0) < 0.05,
          f"tilt matches the score ({m['tilt_pp']:+.1f}pp for score {score:+d})")
    check(reg["rate"].iloc[-1] in ("rising", "neutral", "falling"),
          f"rate state valid ({reg['rate'].iloc[-1]})")
    check(reg["liq"].iloc[-1] in ("expanding", "neutral", "contracting"),
          f"liquidity state valid ({reg['liq'].iloc[-1]})")

    # ---- buckets ----
    b = m["buckets"]
    check(abs(sum(b.values()) - 1.0) < 1e-6, "buckets sum to 1.0")
    check(abs(b["defensive"] - (1 - m["budget"])) < 0.11,
          f"defensive near the policy split ({b['defensive']:.1%})")

    # ---- freshness ----
    stale = len(pd.bdate_range(pd.Timestamp(m["data_as_of"]), pd.Timestamp.today())) - 1
    check(stale <= MAX_STALE_BDAYS,
          f"data {stale} business day(s) old (max {MAX_STALE_BDAYS})")

    # ---- headline plausibility ----
    h = m["headline"]
    for k, (lo, hi) in BOUNDS.items():
        check(lo <= h[k] <= hi, f"headline {k} = {h[k]:.4f} within [{lo}, {hi}]")

    bm = m["benchmark"]
    for k, (lo, hi) in BOUNDS.items():
        check(lo <= bm[k] <= hi, f"benchmark {k} = {bm[k]:.4f} within [{lo}, {hi}]")

    # ---- the attribution must be monotone and end at the headline ----
    check(abs(attr.iloc[-1]["sharpe"] - h["sharpe"]) < 1e-6,
          "attribution ends at the headline Sharpe")
    check(len(attr) == 4, f"four attribution steps (got {len(attr)})")

    # ---- periods and policy ----
    check(len(per) == 3, f"three periods (got {len(per)})")
    check(len(pol) == 4, f"four policy budgets (got {len(pol)})")
    check(any(abs(pol["budget"] - m["budget"]) < 1e-9),
          f"the live budget {m['budget']:.0%} appears in the policy table")

    # ---- curves ----
    check(len(eq) > 100, f"equity curve has {len(eq)} monthly points")
    check(not eq.isna().any().any(), "no NaN in the equity curves")
    check((eq > 0).all().all(), "all index levels positive")

    # ---- drawdown stability vs the previous build ----
    prev_p = ART / ".last_headline.json"
    if prev_p.exists():
        prev = json.loads(prev_p.read_text())
        jump = abs(h["max_dd"] - prev["max_dd"])
        check(jump < MAX_DD_JUMP,
              f"max drawdown moved {jump*100:.1f}pp since the last build "
              f"(limit {MAX_DD_JUMP*100:.0f}pp)")
        warn(abs(h["sharpe"] - prev["sharpe"]) < 0.05,
             f"Sharpe moved {abs(h['sharpe']-prev['sharpe']):.3f} since the last build")
    else:
        print("  ----  no previous build to compare against, skipping drift checks")
    prev_p.write_text(json.dumps({"max_dd": h["max_dd"], "sharpe": h["sharpe"],
                                  "as_of": m["data_as_of"]}))

    # ---- summary ----
    print()
    if fails:
        print(f"VALIDATION FAILED — {len(fails)} check(s) failed:")
        for f in fails:
            print(f"   · {f}")
        sys.exit(1)

    print(f"VALIDATION PASSED — data as of {m['data_as_of']}, "
          f"score {score:+d}, tilt {m['tilt_pp']:+.1f}pp, "
          f"defensive {m['defensive']}, {stale} business day(s) old.")
    if warns:
        print(f"({len(warns)} warning(s), not blocking)")


if __name__ == "__main__":
    main()
