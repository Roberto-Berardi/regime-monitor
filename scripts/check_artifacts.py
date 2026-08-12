"""
Validation gate. Runs after precompute.py, before anything is committed
or published. Exits non-zero on any failure, which stops the workflow and
leaves the previous good page live.
"""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PRE = ROOT / "data" / "precomputed"

EXPECTED_ASSETS = {
    "SP500", "EuroStoxx50", "MSCI_EM", "Gold", "Oil_WTI",
    "US_IG", "US_HY", "US_2Y_proxy", "US_10Y_proxy",
}
REQUIRED_FILES = [
    "metadata.json", "panel.parquet", "returns_daily.parquet",
    "tilted_weights_weekly.parquet", "erc_weights_monthly.parquet",
    "equity_curves.parquet", "drawdowns.parquet", "comparison.parquet",
    "crisis_episodes.parquet", "cost_sensitivity.parquet",
    "risk_contrib_erc_at_rebalance.parquet", "vol_percentiles.parquet",
    "weekly_moves.parquet", "signals_weekly.parquet", "regime.parquet",
]
MAX_STALE_BDAYS = -1

failures = []
def check(ok, msg):
    print(f"  {'PASS' if ok else 'FAIL'}  {msg}")
    if not ok:
        failures.append(msg)

print("\n=== artifact validation ===\n")

# 1. every required file exists and is non-empty
for f in REQUIRED_FILES:
    p = PRE / f
    check(p.exists() and p.stat().st_size > 0, f"{f} present and non-empty")

if failures:
    print(f"\n{len(failures)} failure(s) - aborting early.\n")
    sys.exit(1)

meta = json.loads((PRE / "metadata.json").read_text())

# 2. the full asset universe survived
w = pd.read_parquet(PRE / "tilted_weights_weekly.parquet")
got = set(w.columns)
check(got == EXPECTED_ASSETS,
      f"9 assets in weights (missing: {sorted(EXPECTED_ASSETS - got) or 'none'})")

r = pd.read_parquet(PRE / "returns_daily.parquet")
check(set(r.columns) == EXPECTED_ASSETS, "9 assets in returns")

# 3. weights are a valid long-only book
last = w.iloc[-1]
check(abs(last.sum() - 1.0) < 1e-6, f"weights sum to 1.0 (got {last.sum():.6f})")
check((last >= -1e-9).all(), "no negative weights")
check(last.max() < 0.60, f"largest weight below 60% (got {last.max():.1%})")

# 4. regime probability is a probability
p = meta.get("p_high_latest")
check(p is not None and 0.0 <= p <= 1.0, f"regime probability in [0,1] (got {p})")

# 5. data is recent
as_of = pd.Timestamp(meta["data_as_of"])
stale = len(pd.bdate_range(as_of, pd.Timestamp.today())) - 1
check(stale <= MAX_STALE_BDAYS, f"data {stale} business days old (max {MAX_STALE_BDAYS})")

# 6. no NaNs where the page reads
check(not last.isna().any(), "latest weights contain no NaN")
eq = pd.read_parquet(PRE / "equity_curves.parquet")
check(not eq.iloc[-1].isna().any(), "latest equity curve values are not NaN")

# 7. narrative sections all generated
narr = meta.get("narrative", {})
for sec in ["regime", "moves", "signals", "positioning", "macro", "sentiment"]:
    check(bool(narr.get(sec)), f"narrative section '{sec}' is non-empty")

print()
if failures:
    print(f"VALIDATION FAILED - {len(failures)} issue(s):")
    for f in failures:
        print(f"  - {f}")
    print("\nRefusing to publish. The previous good page stays live.\n")
    sys.exit(1)

print(f"VALIDATION PASSED - data as of {meta['data_as_of']}, "
      f"{len(got)} assets, {stale} business day(s) old.\n")
