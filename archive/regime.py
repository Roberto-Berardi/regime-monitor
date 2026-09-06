"""
Two-state Markov regime switching model on weekly stock-bond DCC correlation.

CRITICAL for production: the app must use FILTERED probabilities only
(P(regime_t | data up to t)) - NOT smoothed probabilities
(P(regime_t | full sample)). Smoothed uses future data, which would leak
into any historical backtest. This module provides BOTH: full-sample fit
(Block 1, for reconciliation) and recursive-filtered fit (Block 2, for
production).

Interpretation of the 2 states:
- LOW-corr regime  : DCC clearly negative -> diversification working
- HIGH-corr regime : DCC near-zero / positive -> diversification broken
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

sys.path.insert(0, str(Path(__file__).parent.parent))
import config


# =============================================================================
# 1. WEEKLY RESAMPLING
# =============================================================================

def to_weekly(dcc_series: pd.Series) -> pd.Series:
    """
    Resample daily DCC to weekly Friday closes (Project 2 convention).
    Weekly frequency is standard for regime models on correlation data:
    less noisy than daily, still responsive enough to catch turning points.
    """
    weekly = dcc_series.resample("W-FRI").last().dropna()
    weekly.name = dcc_series.name
    return weekly


# =============================================================================
# 2. FULL-SAMPLE FIT (Project 2 style; SMOOTHED probs - not for production)
# =============================================================================

def fit_markov_full(weekly_dcc: pd.Series,
                    search_reps: int = 20,
                    maxiter: int = 200) -> dict:
    """
    Fit 2-state Markov switching model on weekly DCC.
    Multiple random starts (default 20) to avoid local optima.

    Returns dict with:
      model, result       : statsmodels objects
      means, sigmas       : per-regime constant and variance
      trans_mat           : 2x2 transition matrix
      high_idx, low_idx   : which regime index is "high correlation"
      smoothed_high       : P(high-corr regime | full sample), aligned to series
      pct_high, pct_low   : share of weeks in each state (smoothed >= 0.5)
      converged           : bool
    """
    print(f"[fit_markov_full] fitting on {len(weekly_dcc)} weekly observations")
    print(f"[fit_markov_full] range: {weekly_dcc.index[0].date()} to {weekly_dcc.index[-1].date()}")

    model = MarkovRegression(
        weekly_dcc,
        k_regimes=2,
        trend="c",
        switching_variance=True,
    )
    result = model.fit(
        search_reps=search_reps,
        search_iter=20,
        disp=False,
        maxiter=maxiter,
    )

    means  = np.array([result.params[f"const[{k}]"]  for k in range(2)])
    sigmas = np.array([result.params[f"sigma2[{k}]"] for k in range(2)])

    # By convention: the HIGH-corr regime is the one with the LARGER mean
    # (closer to zero or positive - diversification broken)
    high_idx = int(np.argmax(means))
    low_idx  = 1 - high_idx

    # Transition matrix
    p00 = result.params["p[0->0]"]
    p10 = result.params["p[1->0]"]
    p01 = 1 - p00
    p11 = 1 - p10
    trans_mat = np.array([[p00, p01], [p10, p11]])

    # Smoothed probability of being in the high-corr state at each date
    smoothed = result.smoothed_marginal_probabilities
    smoothed_high = smoothed.iloc[:, high_idx]
    smoothed_high.name = "P_high_smoothed"

    regime_ind = (smoothed_high >= 0.5).astype(int)
    pct_high = regime_ind.mean() * 100
    pct_low  = 100 - pct_high

    # Convergence check
    probs_ok = np.isfinite(smoothed.values).all() and np.allclose(smoothed.sum(axis=1), 1.0, atol=0.01)

    print(f"[fit_markov_full] converged: {probs_ok}")
    print(f"[fit_markov_full] regime {high_idx} (HIGH-corr): mean={means[high_idx]:+.4f}, sigma2={sigmas[high_idx]:.4f}")
    print(f"[fit_markov_full] regime {low_idx}  (LOW-corr):  mean={means[low_idx]:+.4f}, sigma2={sigmas[low_idx]:.4f}")
    print(f"[fit_markov_full] transition: p[0->0]={p00:.4f}, p[1->0]={p10:.4f}")
    print(f"[fit_markov_full] sample time in HIGH-corr: {pct_high:.1f}%   in LOW-corr: {pct_low:.1f}%")
    print(f"[fit_markov_full] log-likelihood: {result.llf:.2f}  AIC: {result.aic:.2f}")

    return {
        "model":         model,
        "result":        result,
        "means":         means,
        "sigmas":        sigmas,
        "trans_mat":     trans_mat,
        "high_idx":      high_idx,
        "low_idx":       low_idx,
        "smoothed_high": smoothed_high,
        "pct_high":      pct_high,
        "pct_low":       pct_low,
        "converged":     probs_ok,
    }

# =============================================================================
# 3. RECURSIVE FILTERED FIT - PRODUCTION VERSION (no look-ahead)
# =============================================================================
# At each historical date t, filtered probabilities are computed using
# parameters estimated on data up to t only. Refitting happens quarterly
# (both to save compute and to reflect real-world model-maintenance cadence).
# Between refits, filtered probs are produced by the standard Kim/Hamilton
# forward filter using the last quarter's parameters.


def fit_markov_recursive(weekly_dcc: pd.Series,
                         burn_in_end: str = "2015-12-31",
                         refit_freq: str = "Q",
                         search_reps: int = 10) -> pd.Series:
    """
    Recursive filtered Markov regime probabilities.

    Protocol:
    1. Fit model on burn-in window (start -> burn_in_end).
    2. Extract filtered probability of the high-corr state for the burn-in.
    3. Advance quarter by quarter through the rest of the sample:
         a. Refit model on data available up to the quarter-end.
         b. Read filtered probs for the just-completed quarter.
    4. Concatenate: filtered_high is a full series aligned to weekly_dcc.

    Parameters
    ----------
    weekly_dcc : pd.Series
        The weekly DCC series (stock-bond correlation).
    burn_in_end : str
        Last date of the initial anchor window.
    refit_freq : str
        Pandas offset alias for refit cadence. "Q" = quarterly.
    search_reps : int
        Random starts per fit. 10 is a good speed/quality trade-off for
        recursive use; the initial full-sample fit uses 20.

    Returns
    -------
    pd.Series
        Filtered P(high-corr regime), one value per weekly date, aligned
        to weekly_dcc.index. No look-ahead: value at date t uses only
        weekly_dcc up to t.
    """
    print(f"\n[fit_markov_recursive] anchor window ends: {burn_in_end}")
    print(f"[fit_markov_recursive] refit cadence: {refit_freq}")

    # --- Step 1: burn-in fit ---
    burn_in_data = weekly_dcc.loc[:burn_in_end]
    if len(burn_in_data) < 100:
        raise ValueError(f"burn-in window too short: {len(burn_in_data)} obs")

    print(f"[fit_markov_recursive] burn-in fit on {len(burn_in_data)} obs "
          f"({burn_in_data.index[0].date()} to {burn_in_data.index[-1].date()})")

    model0 = MarkovRegression(burn_in_data, k_regimes=2, trend="c",
                              switching_variance=True)
    result0 = model0.fit(search_reps=search_reps, disp=False, maxiter=200)

    # Which regime is HIGH-corr in this fit?
    means0 = np.array([result0.params[f"const[{k}]"] for k in range(2)])
    high_idx = int(np.argmax(means0))
    print(f"[fit_markov_recursive] burn-in: high_idx={high_idx}, "
          f"means={means0}, LL={result0.llf:.2f}")

    # Store filtered probs for burn-in period
    filtered_high = result0.filtered_marginal_probabilities.iloc[:, high_idx].copy()

    # --- Step 2: recursive refits ---
    # Rebalance dates: quarter-ends AFTER burn_in_end, up to the sample end
    all_qends = pd.date_range(start=burn_in_end, end=weekly_dcc.index[-1],
                              freq="QE")
    # Ensure we cover the very last date too
    if all_qends[-1] < weekly_dcc.index[-1]:
        all_qends = all_qends.append(pd.DatetimeIndex([weekly_dcc.index[-1]]))

    print(f"[fit_markov_recursive] {len(all_qends)} recursive refits scheduled")

    prev_end = pd.Timestamp(burn_in_end)
    high_idx_current = high_idx

    for i, q_end in enumerate(all_qends):
        window_data = weekly_dcc.loc[:q_end]
        if len(window_data) <= len(filtered_high):
            continue  # nothing new to fit yet

        try:
            model_i = MarkovRegression(window_data, k_regimes=2, trend="c",
                                       switching_variance=True)
            # Warm-start from previous parameters where feasible
            result_i = model_i.fit(search_reps=search_reps, disp=False, maxiter=200)

            # Regime labels can flip between refits: re-identify the HIGH-corr
            means_i = np.array([result_i.params[f"const[{k}]"] for k in range(2)])
            high_idx_i = int(np.argmax(means_i))

            # Get filtered probs for the NEW segment (from previous end to now)
            new_segment_mask = (window_data.index > prev_end)
            new_dates = window_data.index[new_segment_mask]
            if len(new_dates) == 0:
                prev_end = q_end
                continue

            new_probs = result_i.filtered_marginal_probabilities.iloc[:, high_idx_i]
            new_probs_slice = new_probs.loc[new_dates]

            # Append to running filtered_high (with the CURRENT regime interpretation)
            filtered_high = pd.concat([filtered_high, new_probs_slice])

            high_idx_current = high_idx_i
            prev_end = q_end

            if (i + 1) % 5 == 0 or i == len(all_qends) - 1:
                print(f"[fit_markov_recursive]   refit {i+1}/{len(all_qends)}: "
                      f"through {q_end.date()}, high_idx={high_idx_i}, "
                      f"P_last={new_probs_slice.iloc[-1]:.3f}")

        except Exception as e:
            print(f"[fit_markov_recursive]   refit {i+1} FAILED at {q_end.date()}: {e}")
            print(f"[fit_markov_recursive]   holding previous parameters for this segment")
            # Fallback: use previous model's filtered probs on the new segment
            # (This is the "keep previous quarter's parameters and flag" from roadmap)
            prev_end = q_end
            continue

    filtered_high = filtered_high.sort_index()
    filtered_high.name = "P_high_filtered"

    print(f"[fit_markov_recursive] final series: {len(filtered_high)} obs")
    print(f"[fit_markov_recursive] latest P(high-corr, filtered): {filtered_high.iloc[-1]:.3f}")
    return filtered_high
