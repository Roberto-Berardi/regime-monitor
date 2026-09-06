"""
GARCH(1,1) with Student-t innovations for the Cross-Asset Regime Monitor.

Faithful in spirit to Project 2's hand-rolled GARCH but uses the production
`arch` library (Kevin Sheppard) for reliability under unattended weekly
re-fits on Streamlit Cloud.

Design decisions:
- GARCH(1,1) is fixed as the model order (industry-standard baseline,
  no data-mined order selection).
- Student-t innovations (Project 2 documented excess kurtosis 5-15 across
  assets; Gaussian innovations would misprice tails).
- Input returns multiplied by 100 (percent scale) before fitting, matching
  Project 2's convention exactly - makes reconciliation direct.
- Convergence hardening (added in Block 2): multi-start retry + EWMA(0.94)
  fallback if all attempts fail.
"""
from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
from arch import arch_model

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

# Silence a chatty warning the arch library prints on some Mac setups.
warnings.filterwarnings("ignore", category=FutureWarning, module="arch")


# =============================================================================
# 1. SINGLE-ASSET GARCH FIT
# =============================================================================

def fit_garch(returns: pd.Series, name: str = "asset") -> dict:
    """
    Fit GARCH(1,1) with Student-t innovations on a single return series.

    Parameters
    ----------
    returns : pd.Series
        Daily returns for one asset. NaN values dropped internally.
    name : str
        Asset name for diagnostic printing.

    Returns
    -------
    dict with:
      name        : asset name
      mu, omega   : mean and unconditional-variance-scale parameters (percent scale)
      alpha, beta : ARCH and GARCH coefficients
      persistence : alpha + beta (should be <1 for stationarity)
      nu          : Student-t degrees of freedom
      sigma       : pd.Series of conditional daily vol (DECIMAL units, aligned to input index)
      std_resid   : pd.Series of standardised residuals (aligned to input index)
      loglik      : optimised log-likelihood
      converged   : True/False
      method      : "GARCH-t"  (Block 2 will add "EWMA-fallback" option)
    """
    r = returns.dropna()
    if len(r) < 100:
        raise ValueError(f"[{name}] too few observations ({len(r)}) for GARCH fit")

    # Scale so the series sits inside arch's workable band (std ~10).
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
    res = am.fit(disp="off", show_warning=False)

    # Extract parameters (arch library naming)
    params = res.params
    mu    = float(params["mu"])
    omega = float(params["omega"])
    alpha = float(params["alpha[1]"])
    beta  = float(params["beta[1]"])
    nu    = float(params["nu"])

    # Convert conditional vol back to decimal units
    sigma_pct     = res.conditional_volatility
    sigma_decimal = sigma_pct / SCALE

    # Plausibility guard. `converged` alone is not trustworthy: a diverged
    # fit can report success while implying a volatility hundreds of times
    # the realised one. Reject anything that far from reality so the caller
    # falls back to EWMA.
    # Compare against RECENT realised vol, not the full sample: conditional
    # vol is a statement about today, and a full-sample sigma spanning 2008
    # and 2020 is not the right yardstick for a calm market.
    ann_fit      = float(sigma_decimal.iloc[-1]) * np.sqrt(252)
    ann_realised = float(r.tail(60).std()) * np.sqrt(252)
    ratio = ann_fit / ann_realised if ann_realised > 0 else np.inf
    if not 0.1 <= ratio <= 10.0:
        raise RuntimeError(
            f"[{name}] GARCH fit implausible: implied annualised vol "
            f"{ann_fit:.2%} vs realised {ann_realised:.2%} "
            f"(ratio {ratio:.1f}x, mu={float(params['mu']):.4f}). "
            "Rejecting so the caller can fall back to EWMA."
        )
    sigma_decimal.name = f"{name}_sigma"

    # Standardised residuals (scale-invariant, use arch's directly)
    z = res.std_resid
    z.name = f"{name}_z"

    result = {
        "name":        name,
        "mu":          mu,
        "omega":       omega,
        "alpha":       alpha,
        "beta":        beta,
        "persistence": alpha + beta,
        "nu":          nu,
        "sigma":       sigma_decimal,
        "std_resid":   z,
        "loglik":      float(res.loglikelihood),
        "converged":   bool(res.convergence_flag == 0),
        "method":      "GARCH-t",
    }

    # Diagnostic printout
    print(f"[fit_garch] {name}:")
    print(f"    method={result['method']}  converged={result['converged']}  n={len(r)}")
    print(f"    mu={mu:+.4f}  omega={omega:.4f}  alpha={alpha:.4f}  beta={beta:.4f}")
    print(f"    persistence={result['persistence']:.4f}  nu={nu:.2f}  loglik={result['loglik']:.1f}")

    return result

# =============================================================================
# 2. HARDENING - MULTI-START RETRY + EWMA FALLBACK
# =============================================================================

def _ewma_sigma(returns: pd.Series, lam: float = None) -> pd.Series:
    """
    EWMA (RiskMetrics) conditional vol as a fallback when GARCH fails.
    Recursion: sigma2_t = (1-lam) * r_{t-1}^2 + lam * sigma2_{t-1}
    """
    if lam is None:
        lam = config.EWMA_LAMBDA

    r = returns.dropna()
    r2 = r ** 2
    var = r2.ewm(alpha=(1 - lam), adjust=False).mean()
    sigma = np.sqrt(var)
    sigma.name = returns.name
    return sigma


def fit_garch_hardened(returns: pd.Series, name: str = "asset",
                      n_retries: int = 3) -> dict:
    """
    Robust wrapper around fit_garch:
      1. Try the standard fit.
      2. If it fails OR persistence >= 0.999 (near-IGARCH is fine, but
         >= 1.0 is degenerate), retry from perturbed starting values.
      3. If all retries fail, fall back to EWMA(lambda=0.94) and flag it.

    The app's methodology footnote can display which assets used the
    fallback - full transparency without breaking the pipeline.
    """
    r = returns.dropna()
    if len(r) < 100:
        print(f"[fit_garch_hardened] {name}: too few observations, EWMA fallback")
        sigma = _ewma_sigma(returns)
        z = (returns / sigma).dropna()
        return {
            "name":        name,
            "mu":          float(r.mean() * 100),
            "omega":       np.nan,
            "alpha":       np.nan,
            "beta":        np.nan,
            "persistence": np.nan,
            "nu":          np.nan,
            "sigma":       sigma,
            "std_resid":   z,
            "loglik":      np.nan,
            "converged":   False,
            "method":      "EWMA-fallback",
        }

    # Attempt 1: standard fit
    try:
        result = fit_garch(returns, name=name)
        if result["converged"] and result["persistence"] < 0.9999:
            return result
        print(f"[fit_garch_hardened] {name}: primary fit poor "
              f"(converged={result['converged']}, persistence={result['persistence']:.4f}), retrying")
    except Exception as e:
        print(f"[fit_garch_hardened] {name}: primary fit raised '{e}', retrying")

    # Retries with perturbed starting values via arch's internal 'starting_values'.
    # We pass in different mu/vol starts to nudge the optimizer to a new basin.
    raw_sd = float(r.std())
    SCALE = 10.0 ** round(np.log10(10.0 / raw_sd)) if raw_sd > 0 else 100.0
    r_pct = r * SCALE
    ann_recent = float(r.tail(60).std()) * np.sqrt(252)
    for attempt in range(1, n_retries + 1):
        try:
            # Perturb: use asset's own rolling stats to generate alternative starts
            rng = np.random.default_rng(seed=attempt * 17)
            init_mu    = float(r_pct.mean()) + rng.normal(scale=0.05)
            init_omega = float(r_pct.var()) * rng.uniform(0.02, 0.15)
            init_alpha = rng.uniform(0.03, 0.20)
            init_beta  = rng.uniform(0.70, 0.94)
            starting = [init_mu, init_omega, init_alpha, init_beta, 8.0]

            am = arch_model(r_pct, mean="constant", vol="GARCH", p=1, q=1,
                            dist="t", rescale=False)
            res = am.fit(disp="off", show_warning=False, starting_values=starting)

            p = res.params
            alpha = float(p["alpha[1]"])
            beta  = float(p["beta[1]"])
            sigma_try = res.conditional_volatility / SCALE
            ann_try = float(sigma_try.iloc[-1]) * np.sqrt(252)
            ratio = ann_try / ann_recent if ann_recent > 0 else np.inf
            plausible = 0.1 <= ratio <= 10.0
            if not plausible:
                print(f"[fit_garch_hardened] {name}: retry {attempt} converged but "
                      f"implies {ann_try:.1%} vol vs {ann_recent:.1%} realised "
                      f"({ratio:.0f}x) - rejecting")
            if res.convergence_flag == 0 and (alpha + beta) < 0.9999 and plausible:
                print(f"[fit_garch_hardened] {name}: retry {attempt} converged")
                sigma = sigma_try
                sigma.name = f"{name}_sigma"
                z = res.std_resid
                z.name = f"{name}_z"
                return {
                    "name":        name,
                    "mu":          float(p["mu"]),
                    "omega":       float(p["omega"]),
                    "alpha":       alpha,
                    "beta":        beta,
                    "persistence": alpha + beta,
                    "nu":          float(p["nu"]),
                    "sigma":       sigma,
                    "std_resid":   z,
                    "loglik":      float(res.loglikelihood),
                    "converged":   True,
                    "method":      f"GARCH-t (retry {attempt})",
                }
        except Exception as e:
            print(f"[fit_garch_hardened] {name}: retry {attempt} raised '{e}'")

    # All GARCH attempts failed; fall back to EWMA
    print(f"[fit_garch_hardened] {name}: all GARCH attempts failed, EWMA fallback")
    sigma = _ewma_sigma(returns)
    z = (returns / sigma).dropna()
    return {
        "name":        name,
        "mu":          float(r.mean() * 100),
        "omega":       np.nan,
        "alpha":       np.nan,
        "beta":        np.nan,
        "persistence": np.nan,
        "nu":          np.nan,
        "sigma":       sigma,
        "std_resid":   z,
        "loglik":      np.nan,
        "converged":   False,
        "method":      "EWMA-fallback",
    }

# =============================================================================
# 3. MULTI-ASSET WRAPPER
# =============================================================================

def fit_all(returns: pd.DataFrame, verbose: bool = True) -> tuple[dict, pd.DataFrame]:
    """
    Fit hardened GARCH(1,1)-t on every column of a returns DataFrame.

    Parameters
    ----------
    returns : pd.DataFrame
        One column per asset, daily returns in decimal units.
    verbose : bool
        Print progress per asset.

    Returns
    -------
    fits : dict
        Keyed by asset name -> full result dict from fit_garch_hardened.
    summary : pd.DataFrame
        One row per asset, columns: model params, method, converged.
    """
    if verbose:
        print("\n" + "="*70)
        print(f"[fit_all] fitting GARCH(1,1)-t on {returns.shape[1]} assets")
        print("="*70)

    fits = {}
    rows = []
    for col in returns.columns:
        if verbose:
            print(f"\n----- {col} -----")
        fit = fit_garch_hardened(returns[col], name=col)
        fits[col] = fit
        rows.append({
            "asset":       col,
            "method":      fit["method"],
            "converged":   fit["converged"],
            "mu":          fit["mu"],
            "omega":       fit["omega"],
            "alpha":       fit["alpha"],
            "beta":        fit["beta"],
            "persistence": fit["persistence"],
            "nu":          fit["nu"],
            "loglik":      fit["loglik"],
        })

    summary = pd.DataFrame(rows).set_index("asset")

    if verbose:
        print("\n" + "="*70)
        print("[fit_all] SUMMARY")
        print("="*70)
        cols_to_show = ["method", "converged", "mu", "omega", "alpha", "beta",
                       "persistence", "nu"]
        print(summary[cols_to_show].round(4).to_string())

    return fits, summary


# =============================================================================
# 4. RECONCILIATION vs PROJECT 2
# =============================================================================

# Project 2 GARCH(1,1)-t results, from stored notebook outputs.
# (Note: Project 2 selected order per asset via BIC and sometimes chose GARCH(2,1)
# or ARCH(2,0); we fix GARCH(1,1) as a pre-committed spec, so exact matches on
# omega/alpha/beta aren't expected for those assets. We compare persistence
# and nu which are less order-sensitive.)
PROJECT2_GARCH = {
    "SP500":        {"model": "GARCH(1,1)", "omega": 0.0166, "persistence": 0.9999, "nu": 4.73},
    "EuroStoxx50":  {"model": "GARCH(2,1)", "omega": 0.0355, "persistence": 0.9863, "nu": 5.33},
    "MSCI_EM":      {"model": "GARCH(1,1)", "omega": 0.0271, "persistence": 0.9785, "nu": 7.81},
    "US_IG":        {"model": "GARCH(1,1)", "omega": 0.0008, "persistence": 0.9917, "nu": 9.49},
    "US_HY":        {"model": "GARCH(2,0)", "omega": 0.0152, "persistence": 1.0000, "nu": 3.59},
    "Gold":         {"model": "GARCH(2,1)", "omega": 0.0113, "persistence": 0.9929, "nu": 4.88},
    "Oil_WTI":      {"model": "GARCH(2,1)", "omega": 0.0717, "persistence": 0.9896, "nu": 6.31},
    "US_10Y_proxy": {"model": "GARCH(1,1)", "omega": 0.0019, "persistence": 0.9921, "nu": 10.93},
    # US_2Y_proxy has no Project 2 target - new series
}


def reconcile_garch_vs_project2(summary: pd.DataFrame) -> pd.DataFrame:
    """
    Compare our GARCH persistence and nu against Project 2's stored results.
    """
    rows = []
    for asset in summary.index:
        target = PROJECT2_GARCH.get(asset)
        our_pers = summary.loc[asset, "persistence"]
        our_nu   = summary.loc[asset, "nu"]

        if target is None:
            rows.append({
                "asset":         asset,
                "p2_model":      "n/a",
                "our_pers":      f"{our_pers:.4f}" if pd.notna(our_pers) else "  n/a",
                "p2_pers":       "  n/a",
                "pers_dev":      "  n/a",
                "our_nu":        f"{our_nu:.2f}" if pd.notna(our_nu) else " n/a",
                "p2_nu":         " n/a",
                "nu_dev":        "  n/a",
                "verdict":       "no target",
            })
            continue

        pers_dev = (our_pers - target["persistence"]) / target["persistence"]
        nu_dev   = (our_nu   - target["nu"])          / target["nu"]

        # We compare against Project 2's numbers but note that they used
        # different orders on some assets - flag those with '~' verdict
        order_note = "" if target["model"] == "GARCH(1,1)" else " (P2 used " + target["model"] + ")"
        pers_ok = abs(pers_dev) <= 0.05
        nu_ok   = abs(nu_dev)   <= 0.25   # nu is noisier, wider tolerance
        verdict = ("PASS" if pers_ok and nu_ok else "REVIEW") + order_note

        rows.append({
            "asset":         asset,
            "p2_model":      target["model"],
            "our_pers":      f"{our_pers:.4f}",
            "p2_pers":       f"{target['persistence']:.4f}",
            "pers_dev":      f"{pers_dev:+.1%}",
            "our_nu":        f"{our_nu:.2f}",
            "p2_nu":         f"{target['nu']:.2f}",
            "nu_dev":        f"{nu_dev:+.1%}",
            "verdict":       verdict,
        })
    return pd.DataFrame(rows)

# =============================================================================
# 5. LJUNG-BOX DIAGNOSTICS ON STANDARDISED RESIDUALS
# =============================================================================
from statsmodels.stats.diagnostic import acorr_ljungbox


def ljungbox_diagnostics(fits: dict, lags: tuple = (5, 10)) -> pd.DataFrame:
    """
    Post-GARCH Ljung-Box test on standardised residuals and squared residuals.

    Interpretation:
    - Test on z_t   : are the residuals autocorrelated after subtracting the mean?
                      Failure would suggest the mean model needs AR terms.
    - Test on z_t^2 : is there ARCH effect LEFT after GARCH filtering?
                      Failure means GARCH did not fully capture vol clustering.
                      This is the more diagnostic of the two.

    We report p-values at the requested lags; PASS if all p-values > 0.05.
    """
    rows = []
    for asset, fit in fits.items():
        if fit["method"] == "EWMA-fallback" or fit["std_resid"] is None:
            rows.append({
                "asset":       asset,
                "method":      fit["method"],
                "LB_z_lag5":   "  n/a",
                "LB_z_lag10":  "  n/a",
                "LB_z2_lag5":  "  n/a",
                "LB_z2_lag10": "  n/a",
                "verdict":     "skipped (EWMA)",
            })
            continue

        z = fit["std_resid"].dropna()
        if len(z) < 100:
            rows.append({"asset": asset, "verdict": "too short"})
            continue

        lb_z  = acorr_ljungbox(z,     lags=list(lags), return_df=True)
        lb_z2 = acorr_ljungbox(z ** 2, lags=list(lags), return_df=True)

        p_z_5   = lb_z.loc[lags[0], "lb_pvalue"]
        p_z_10  = lb_z.loc[lags[1], "lb_pvalue"]
        p_z2_5  = lb_z2.loc[lags[0], "lb_pvalue"]
        p_z2_10 = lb_z2.loc[lags[1], "lb_pvalue"]

        arch_removed = (p_z2_5 > 0.05) and (p_z2_10 > 0.05)
        mean_clean   = (p_z_5  > 0.05) and (p_z_10  > 0.05)

        if arch_removed and mean_clean:
            verdict = "PASS"
        elif arch_removed and not mean_clean:
            verdict = "PASS (mean AR)"
        elif not arch_removed:
            verdict = "REVIEW (ARCH left)"
        else:
            verdict = "REVIEW"

        rows.append({
            "asset":       asset,
            "method":      fit["method"],
            "LB_z_lag5":   f"{p_z_5:.3f}",
            "LB_z_lag10":  f"{p_z_10:.3f}",
            "LB_z2_lag5":  f"{p_z2_5:.3f}",
            "LB_z2_lag10": f"{p_z2_10:.3f}",
            "verdict":     verdict,
        })

    return pd.DataFrame(rows)

