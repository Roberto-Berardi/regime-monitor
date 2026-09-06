"""
Tactical tilt + regime gate for the Cross-Asset Regime Monitor.

Combines the three layers:
- Strategic ERC weights (Phase 6): monthly rebalance
- Momentum signals (Phase 7 B1): daily 12-1M + 200DMA agreement gate
- Filtered regime probability (Phase 5): weekly recursive Markov

Tilt logic (config-driven, pre-committed):
1. Weekly rebalance on Friday closes.
2. Active cap: TILT_CAP_PP = 4pp default; halved to 2pp when
   filtered P(high-corr regime) > REGIME_THRESHOLD = 0.70.
3. Tilted weight = ERC weight + (active_cap / 100) * momentum_signal.
4. Long-only floor at 0.
5. Renormalize to sum to 1.

The floor can bind for small ERC weights; the effective tilt is then
asymmetric (feature, not bug - documented in Limitations).
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
import config


def build_tilted_weights(
    erc_weights: pd.DataFrame,
    signals: pd.DataFrame,
    regime_prob: pd.Series,
    verbose: bool = True,
) -> dict:
    """
    Compute weekly tilted weights: ERC + regime-gated momentum tilt.

    Parameters
    ----------
    erc_weights : pd.DataFrame
        Monthly ERC weight history from erc_weight_history().
        Contains asset columns + 'converged' + 'method' diagnostics.
    signals : pd.DataFrame
        Daily momentum signals from build_signal_panel().
    regime_prob : pd.Series
        Weekly filtered P(high-corr regime) from fit_markov_recursive().

    Returns
    -------
    dict with:
        weights       : pd.DataFrame of final weekly tilted weights
        erc_weekly    : pd.DataFrame of ERC weights aligned to same weekly grid
        active_cap    : pd.Series of applied cap (pp) at each date
        deviations    : pd.DataFrame of (tilted - erc) per asset per date
        signals_weekly: pd.DataFrame of Friday-close signals used
        passed_tests  : bool
        test_details  : dict
    """
    # Asset columns (exclude diagnostic columns from ERC output)
    asset_cols = [c for c in erc_weights.columns if c not in ("converged", "method")]

    # Align signals to same asset order
    signals_aligned = signals[asset_cols].copy()

    # Resample signals and regime probability to weekly Friday close
    signals_weekly = signals_aligned.resample("W-FRI").last()
    regime_weekly  = regime_prob.resample("W-FRI").last()

    # ERC weights: forward-fill from monthly to weekly grid
    erc_asset = erc_weights[asset_cols].copy()
    common_index = signals_weekly.index.union(regime_weekly.index).sort_values()
    erc_weekly = erc_asset.reindex(common_index, method="ffill")

    # Common sample: dates where ALL three inputs are valid
    valid_dates = (
        erc_weekly.dropna(how="any").index
        .intersection(signals_weekly.dropna(how="any").index)
        .intersection(regime_weekly.dropna().index)
    )

    if verbose:
        print(f"[build_tilted_weights] {len(valid_dates)} weekly rebalances")
        print(f"[build_tilted_weights] sample: {valid_dates[0].date()} to {valid_dates[-1].date()}")
        print(f"[build_tilted_weights] base cap: +/-{config.TILT_CAP_PP}pp, "
              f"gate threshold: {config.REGIME_THRESHOLD}, halving: {config.REGIME_HALVING}")

    base_cap_pp = config.TILT_CAP_PP
    threshold   = config.REGIME_THRESHOLD
    halving     = config.REGIME_HALVING

    tilted_rows  = []
    active_caps  = []

    for t in valid_dates:
        w_erc  = erc_weekly.loc[t].values
        sig    = signals_weekly.loc[t].values
        p_high = regime_weekly.loc[t]

        # Active cap: halved when regime probability exceeds threshold
        cap_pp = base_cap_pp * (halving if p_high > threshold else 1.0)
        active_caps.append(cap_pp)

        # Apply tilt (convert pp -> decimal)
        w_tilted = w_erc + (cap_pp / 100.0) * sig

        # Long-only floor
        w_tilted = np.maximum(w_tilted, 0.0)

        # Renormalize to sum to 1
        total = w_tilted.sum()
        if total > 0:
            w_final = w_tilted / total
        else:
            # Extreme edge case: everything floored. Fall back to unrenormalized ERC.
            w_final = w_erc / w_erc.sum()

        tilted_rows.append(w_final)

    # Assemble outputs
    W = pd.DataFrame(tilted_rows, index=valid_dates, columns=asset_cols)
    active_cap = pd.Series(active_caps, index=valid_dates, name="active_cap_pp")
    erc_weekly_aligned = erc_weekly.loc[valid_dates]
    deviations = W - erc_weekly_aligned

    # ------------------------------------------------------------
    # Unit tests
    # ------------------------------------------------------------
    test_details = {}

    sum_dev = (W.sum(axis=1) - 1.0).abs().max()
    test_details["weights_sum_to_1"] = {
        "max_deviation": float(sum_dev),
        "passed": sum_dev < 1e-10,
    }

    min_w = W.min().min()
    test_details["non_negative"] = {
        "min_weight": float(min_w),
        "passed": min_w >= 0.0,
    }

    n_gated = (active_cap < base_cap_pp).sum()
    test_details["regime_gate_triggered"] = {
        "n_gated_dates": int(n_gated),
        "pct_gated": float(n_gated / len(active_cap) * 100),
        "passed": n_gated > 0,
    }

    passed_tests = all(d["passed"] for d in test_details.values())

    if verbose:
        print(f"\n[build_tilted_weights] UNIT TESTS")
        for name, d in test_details.items():
            status = "PASS" if d["passed"] else "FAIL"
            print(f"  [{status}] {name}: {d}")

    return {
        "weights":        W,
        "erc_weekly":     erc_weekly_aligned,
        "active_cap":     active_cap,
        "deviations":     deviations,
        "signals_weekly": signals_weekly.loc[valid_dates],
        "passed_tests":   passed_tests,
        "test_details":   test_details,
    }
