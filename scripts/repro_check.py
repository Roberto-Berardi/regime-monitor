"""Verify the runner reproduces the local GARCH baseline."""
import platform
import numpy as np, pandas as pd, yfinance as yf, scipy, statsmodels, arch
from arch import arch_model

LOCAL = {"mu": 0.1004764624, "omega": 0.0222704214,
         "alpha[1]": 0.1740864938, "beta[1]": 0.8207911716,
         "nu": 5.4483248347}
LOCAL_LL, LOCAL_OBS = -3462.77505795, 2765

px = yf.download("SPY", start="2015-01-01", end="2026-01-01",
                 progress=False, auto_adjust=False)["Adj Close"]
r = (100 * np.log(px / px.shift(1))).dropna()
if isinstance(r, pd.DataFrame):
    r = r.iloc[:, 0]

res = arch_model(r, vol="GARCH", p=1, q=1, dist="t").fit(disp="off")
p = res.params

print(f"python     {platform.python_version()}")
print(f"arch       {arch.__version__}")
print(f"numpy      {np.__version__}   scipy {scipy.__version__}")
print(f"pandas     {pd.__version__}   statsmodels {statsmodels.__version__}")
print(f"obs        {len(r)}  (local {LOCAL_OBS})  "
      f"{'MATCH' if len(r) == LOCAL_OBS else '*** MISMATCH ***'}")
print("-" * 66)
print(f"{'PARAM':<10} {'RUNNER':>16} {'LOCAL':>16} {'DIFF':>12}")
print("-" * 66)

worst = 0.0
for k, loc in LOCAL.items():
    run = float(p[k]); d = abs(run - loc)
    worst = max(worst, d)
    print(f"{k:<10} {run:>16.10f} {loc:>16.10f} {d:>12.2e}")

dll = abs(res.loglikelihood - LOCAL_LL)
print(f"{'loglik':<10} {res.loglikelihood:>16.8f} {LOCAL_LL:>16.8f} {dll:>12.2e}")
print("-" * 66)
print(f"\nWorst parameter difference: {worst:.2e}")

if worst < 1e-6 and dll < 1e-4:
    print("REPRODUCIBLE - runner matches local within tolerance.")
else:
    print("*** DRIFT DETECTED - runner does not match local. ***")
    raise SystemExit(1)
