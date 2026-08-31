"""
DCC(1,1) correlation filter for the Cross-Asset Regime Monitor.

Dynamic conditional correlation recursion (Engle 2002).
Berardi 2025). Uses fixed parameters (a=0.05, b=0.93) from config,
matching Engle (2002)'s empirical findings across most asset pairs.

Consumes: dict of GARCH fits from src/garch.fit_all.
Produces:
- R: (T, N, N) numpy array of correlation matrices per date.
- H: (T, N, N) numpy array of covariance matrices per date.
- Common index of dates where ALL assets have standardised residuals.
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
import config


# =============================================================================
# 1. ASSEMBLE STANDARDISED RESIDUALS
# =============================================================================

def build_std_resid_matrix(fits: dict, returns: pd.DataFrame) -> pd.DataFrame:
    """
    Combine per-asset standardised residuals from GARCH fits into a wide
    DataFrame aligned on the common index (rows where ALL assets have data).
    """
    parts = []
    for asset, fit in fits.items():
        z = fit["std_resid"].copy()
        z.name = asset
        parts.append(z)

    Z = pd.concat(parts, axis=1)
    n_before = Z.shape[0]
    Z = Z.dropna(how="any")
    n_after = Z.shape[0]
    print(f"[build_std_resid_matrix] {n_before} rows -> {n_after} on common sample")
    print(f"[build_std_resid_matrix] sample: {Z.index.min().date()} to {Z.index.max().date()}")
    return Z


# =============================================================================
# 2. DCC RECURSION
# =============================================================================

def dcc_filter(std_resid: pd.DataFrame,
               a: float = None,
               b: float = None) -> dict:
    """
    Run the DCC(1,1) recursion on standardised residuals.

    Parameters
    ----------
    std_resid : pd.DataFrame
        T rows x N columns, no NaN. Column order is preserved.
    a, b : DCC parameters; defaults to config.DCC_A, config.DCC_B.

    Returns
    -------
    dict:
        assets : list of asset names (column order)
        dates  : DatetimeIndex of the common sample
        Qbar   : (N, N) unconditional correlation matrix
        R      : (T, N, N) numpy array of correlation matrices per date
    """
    a = config.DCC_A if a is None else a
    b = config.DCC_B if b is None else b
    assert a + b < 1, f"DCC stationarity requires a+b<1 (got {a+b:.4f})"

    Z = std_resid.values
    T, N = Z.shape
    assets = list(std_resid.columns)

    print(f"\n[dcc_filter] running DCC(1,1) with a={a}, b={b}")
    print(f"[dcc_filter] T={T} dates, N={N} assets: {assets}")

    # Unconditional correlation (using all residuals)
    Qbar = np.corrcoef(Z.T)

    # Storage
    R = np.zeros((T, N, N))
    Qt = Qbar.copy()

    for t in range(T):
        if t > 0:
            z_prev = Z[t - 1].reshape(-1, 1)
            Qt = (1 - a - b) * Qbar + a * (z_prev @ z_prev.T) + b * Qt

        # Convert Q_t to correlation matrix R_t
        diag_q = np.sqrt(np.clip(np.diag(Qt), 1e-12, None))
        inv_diag = np.diag(1.0 / diag_q)
        Rt = inv_diag @ Qt @ inv_diag
        # Symmetrize (numerical clean-up) and force 1s on diagonal
        Rt = 0.5 * (Rt + Rt.T)
        np.fill_diagonal(Rt, 1.0)
        R[t] = Rt

    print(f"[dcc_filter] done. R shape: {R.shape}")

    return {
        "assets": assets,
        "dates":  std_resid.index,
        "Qbar":   Qbar,
        "R":      R,
    }


# =============================================================================
# 3. PAIRWISE CORRELATION SERIES
# =============================================================================

def correlation_pair(dcc_result: dict, asset_a: str, asset_b: str) -> pd.Series:
    """
    Extract the time series of pairwise DCC correlation for two assets.
    """
    assets = dcc_result["assets"]
    if asset_a not in assets or asset_b not in assets:
        raise KeyError(f"{asset_a} or {asset_b} not in DCC assets: {assets}")

    i = assets.index(asset_a)
    j = assets.index(asset_b)
    s = pd.Series(
        dcc_result["R"][:, i, j],
        index=dcc_result["dates"],
        name=f"DCC({asset_a},{asset_b})",
    )
    return s