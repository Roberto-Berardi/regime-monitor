"""Adaptive GARCH scaling + plausibility guard.

The 2Y proxy's percent-scale returns have std ~0.009, far below the 1-1000
band arch's optimiser needs, so the fit diverged (mu = -897) while still
reporting converged=True. This scales each series to a workable magnitude
and rejects any fit whose implied volatility is nowhere near the realised one.
"""
from pathlib import Path

p = Path("src/garch.py")
s = p.read_text()

old = """    # Scale to percent, matching Project 2's convention for numerical stability
    r_pct = r * 100.0

    # Fit GARCH(1,1) with Student-t
    am = arch_model(r_pct, mean="constant", vol="GARCH", p=1, q=1, dist="t", rescale=False)
    res = am.fit(disp="off", show_warning=False)"""

new = """    # Scale so the series sits inside arch's workable band (std ~10).
    # For equities this lands at ~100, matching Project 2's convention; for
    # very low-volatility series such as the 2Y duration proxy it lands
    # higher. The multiplier cancels out when converting sigma back, so it
    # changes nothing except the optimiser's numerical footing.
    raw_sd = float(r.std())
    if raw_sd <= 0 or not np.isfinite(raw_sd):
        raise ValueError(f"[{name}] degenerate return series (std={raw_sd})")
    SCALE = 10.0 ** round(np.log10(10.0 / raw_sd))
    r_pct = r * SCALE

    # Fit GARCH(1,1) with Student-t
    am = arch_model(r_pct, mean="constant", vol="GARCH", p=1, q=1, dist="t", rescale=False)
    res = am.fit(disp="off", show_warning=False)"""

assert old in s, "scale/fit block not found"
s = s.replace(old, new, 1)

old = """    # Conditional vol is on percent scale; convert back to decimal
    sigma_pct     = res.conditional_volatility   # daily vol in percent
    sigma_decimal = sigma_pct / 100.0"""

new = """    # Convert conditional vol back to decimal units
    sigma_pct     = res.conditional_volatility
    sigma_decimal = sigma_pct / SCALE

    # Plausibility guard. `converged` alone is not trustworthy: a diverged
    # fit can report success while implying a volatility hundreds of times
    # the realised one. Reject anything that far from reality so the caller
    # falls back to EWMA.
    ann_fit      = float(sigma_decimal.iloc[-1]) * np.sqrt(252)
    ann_realised = raw_sd * np.sqrt(252)
    ratio = ann_fit / ann_realised if ann_realised > 0 else np.inf
    if not (1.0 / 3.0) <= ratio <= 3.0:
        raise RuntimeError(
            f"[{name}] GARCH fit implausible: implied annualised vol "
            f"{ann_fit:.2%} vs realised {ann_realised:.2%} "
            f"(ratio {ratio:.1f}x, mu={float(params['mu']):.4f}). "
            "Rejecting so the caller can fall back to EWMA."
        )"""

assert old in s, "sigma conversion block not found"
s = s.replace(old, new, 1)

if "import numpy as np" not in s:
    s = s.replace("from arch import arch_model",
                  "import numpy as np\nfrom arch import arch_model", 1)

p.write_text(s)
import ast; ast.parse(s)
print("src/garch.py patched\n")

# ── verify against every asset ──────────────────────────────────────────
import sys; sys.path.insert(0, ".")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import importlib, src.garch as G
importlib.reload(G)

rets = pd.read_parquet("data/precomputed/returns_daily.parquet")
print(f"{'ASSET':<14}{'REALISED':>10}{'FITTED':>10}{'RATIO':>8}  STATUS")
print("-" * 54)
for a in rets.columns:
    r = rets[a].dropna()
    realised = r.std() * np.sqrt(252)
    try:
        f = G.fit_garch_t(r, name=a)
        fitted = float(f["sigma"].iloc[-1]) * np.sqrt(252)
        print(f"{a:<14}{realised*100:>9.2f}%{fitted*100:>9.2f}%"
              f"{fitted/realised:>7.2f}x  ok")
    except Exception as ex:
        print(f"{a:<14}{realised*100:>9.2f}%{'-':>10}{'-':>8}  "
              f"{type(ex).__name__} -> EWMA fallback")
