"""
Equal Risk Contribution (ERC) portfolio construction.

Equal risk contribution solver.
Extensions for production:
- Handles the case where one asset uses EWMA-fallback vol (US_HY)
- Recursive-safe: uses only DCC covariances from data up to the rebalance date
- Inverse-vol fallback when SLSQP fails

The ERC weight vector w solves:
  min_w  sum_i ( w_i * (H w)_i / (w' H w) - 1/N )^2
  s.t.   sum(w) = 1,  w >= 0

where H is the DCC-implied covariance matrix. Each asset ends up
contributing equally to portfolio variance.
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).parent.parent))
import config


# =============================================================================
# 1. COVARIANCE REGULARIZATION
# =============================================================================

def regularize_covariance(H: np.ndarray) -> np.ndarray:
    """
    Ensure H is symmetric and positive-definite before optimization.
    Clip eigenvalues at 1e-12 to guarantee invertibility.
    Numerical safety net; rarely binds on well-formed DCC covariances.
    """
    H = np.asarray(H, dtype=float)
    H = 0.5 * (H + H.T)                              # force symmetry
    eigvals, eigvecs = np.linalg.eigh(H)
    eigvals = np.clip(eigvals, 1e-12, None)          # PSD clip
    H_reg = eigvecs @ np.diag(eigvals) @ eigvecs.T
    return 0.5 * (H_reg + H_reg.T)


# =============================================================================
# 2. ERC MATH
# =============================================================================

def erc_objective(w: np.ndarray, H: np.ndarray) -> float:
    """
    Sum of squared deviations of risk contribution fractions from 1/N.
    Zero when all N assets contribute equally to portfolio variance.
    """
    port_var = float(w @ H @ w)
    if port_var <= 0 or not np.isfinite(port_var):
        return 1e9
    mrc     = H @ w                                  # marginal risk contribution
    rc_frac = (w * mrc) / port_var                   # risk contribution fractions
    target  = np.full(len(w), 1.0 / len(w))
    return float(np.sum((rc_frac - target) ** 2))


def risk_contribution_fraction(w: np.ndarray, H: np.ndarray) -> np.ndarray:
    """
    Diagnostic: per-asset risk contribution as fraction of total variance.
    Should be ~1/N for every asset if ERC solved correctly.
    """
    port_var = float(w @ H @ w)
    return (w * (H @ w)) / port_var


def inverse_vol_start(H: np.ndarray) -> np.ndarray:
    """
    Starting point: weight inversely proportional to volatility.
    A good baseline that often nearly solves the ERC problem for
    diagonal-ish correlations.
    """
    vols = np.sqrt(np.clip(np.diag(H), 1e-12, None))
    w = 1.0 / vols
    return w / w.sum()


# =============================================================================
# 3. ERC SOLVER (multi-start, robust)
# =============================================================================

def solve_erc(H: np.ndarray, previous_weights: np.ndarray = None) -> dict:
    """
    Solve ERC given covariance H, with multi-start SLSQP.
    
    Starts tried in order:
      1. previous_weights (if provided and finite) - warm start for stability
      2. equal-weight 1/N
      3. inverse-vol
    Best solution (lowest objective) is returned.

    Fallback: if all three starts fail, return inverse-vol weights with
    a flag so the caller knows something went wrong.
    """
    H = regularize_covariance(H)
    n = H.shape[0]

    cap = getattr(config, "ERC_MAX_PER_ASSET_WEIGHT", 1.0)
    cap = max(cap, 1.0 / n + 1e-9)   # must stay feasible
    bounds = [(1e-8, cap) for _ in range(n)]         # long-only, capped
    cons   = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

    # Build list of starting points
    starts = []
    if previous_weights is not None and np.isfinite(previous_weights).all():
        starts.append(("prev", np.asarray(previous_weights, dtype=float)))
    starts.append(("equal_weight",  np.full(n, 1.0 / n)))
    starts.append(("inverse_vol",   inverse_vol_start(H)))

    best_result = None
    best_start_name = None
    for name, w0 in starts:
        w0 = np.clip(w0, 1e-8, 1.0)
        w0 = w0 / w0.sum()
        try:
            res = minimize(erc_objective, w0, args=(H,), method="SLSQP",
                           bounds=bounds, constraints=cons,
                           options={"maxiter": 1000, "ftol": 1e-12})
            if res.success and (best_result is None or res.fun < best_result.fun):
                best_result = res
                best_start_name = name
        except Exception:
            continue

    if best_result is None:
        # All optimizations failed - fall back to inverse-vol
        w_fallback = inverse_vol_start(H)
        return {
            "weights":      w_fallback,
            "converged":    False,
            "method":       "inverse-vol fallback",
            "objective":    erc_objective(w_fallback, H),
            "rc_fraction":  risk_contribution_fraction(w_fallback, H),
        }

    w = np.clip(best_result.x, 1e-8, 1.0)
    w = w / w.sum()

    return {
        "weights":     w,
        "converged":   True,
        "method":      f"SLSQP (start: {best_start_name})",
        "objective":   best_result.fun,
        "rc_fraction": risk_contribution_fraction(w, H),
    }


# =============================================================================
# 4. BUILD H_t FROM DCC OUTPUT
# =============================================================================

def build_H(fits: dict, dcc_result: dict, date: pd.Timestamp) -> tuple[np.ndarray, list]:
    """
    Reconstruct the full covariance matrix H_t = D_t R_t D_t on a given date.

    D_t = diag(sigma_i(t)) using each asset's conditional vol from the
          GARCH fit (or EWMA fallback for US_HY).
    R_t = DCC correlation matrix at that date.
    """
    assets = dcc_result["assets"]
    dates  = dcc_result["dates"]

    # Find date position in the DCC output
    if date not in dates:
        raise KeyError(f"Date {date.date()} not in DCC sample "
                       f"({dates[0].date()} to {dates[-1].date()})")
    t_idx = dates.get_loc(date)

    R_t = dcc_result["R"][t_idx]

    # Gather sigmas at that date, in the same order as DCC assets
    sigmas = np.zeros(len(assets))
    for i, asset in enumerate(assets):
        sigma_series = fits[asset]["sigma"]
        if date in sigma_series.index:
            sigmas[i] = sigma_series.loc[date]
        else:
            # Use nearest preceding value (rare edge case)
            sigmas[i] = sigma_series.loc[:date].iloc[-1]

    D_t = np.diag(sigmas)
    H_t = D_t @ R_t @ D_t
    return H_t, assets

# =============================================================================
# 5. MONTHLY REBALANCE LOOP OVER FULL SAMPLE
# =============================================================================

def erc_weight_history(fits: dict,
                       dcc_result: dict,
                       rebal_freq: str = "ME",
                       verbose: bool = True) -> pd.DataFrame:
    """
    Build the full ERC weight history by rebalancing at the end of each period.

    Parameters
    ----------
    fits, dcc_result : outputs of Phases 3 and 4.
    rebal_freq : pandas offset alias; "ME" = month-end (Maillard/Roncalli standard),
                 "W-FRI" for weekly, "QE" for quarterly.

    Returns
    -------
    pd.DataFrame
        Index = rebalance dates (only dates where DCC has data).
        Columns = asset names.
        Values = weights at that rebalance (sum to 1 per row).
        Extra columns 'converged' and 'method' for diagnostics.
    """
    assets = dcc_result["assets"]
    dates  = dcc_result["dates"]

    # Rebalance dates: last available DCC date <= each period-end
    period_ends = pd.date_range(start=dates[0], end=dates[-1], freq=rebal_freq)
    rebal_dates = []
    for p_end in period_ends:
        # Find the last actual DCC date on-or-before this period-end
        eligible = dates[dates <= p_end]
        if len(eligible) > 0:
            rebal_dates.append(eligible[-1])
    rebal_dates = pd.DatetimeIndex(rebal_dates).unique()

    if verbose:
        print(f"\n[erc_weight_history] rebalancing {rebal_freq} on {len(rebal_dates)} dates")
        print(f"[erc_weight_history] first rebalance: {rebal_dates[0].date()}")
        print(f"[erc_weight_history] last  rebalance: {rebal_dates[-1].date()}")

    # Storage
    weights_rows = []
    prev_weights = None
    n_failed = 0

    for i, rebal_date in enumerate(rebal_dates):
        H_t, _ = build_H(fits, dcc_result, rebal_date)
        result = solve_erc(H_t, previous_weights=prev_weights)

        if not result["converged"]:
            n_failed += 1

        row = {a: w for a, w in zip(assets, result["weights"])}
        row["converged"] = result["converged"]
        row["method"]    = result["method"]
        weights_rows.append((rebal_date, row))

        prev_weights = result["weights"]

        if verbose and (i + 1) % 24 == 0:
            print(f"[erc_weight_history]   {i+1}/{len(rebal_dates)} through {rebal_date.date()}")

    W = pd.DataFrame(
        [r for _, r in weights_rows],
        index=[d for d, _ in weights_rows],
    )
    W.index.name = "date"

    if verbose:
        print(f"[erc_weight_history] done. {len(W)} rebalances, {n_failed} used inverse-vol fallback")

    return W
