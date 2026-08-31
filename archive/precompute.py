"""
Precompute pipeline outputs for the Streamlit dashboard.

The full pipeline (GARCH x9, recursive Markov x44 refits, 231 ERC solves,
5 backtests) takes 2-4 minutes. Streamlit Cloud cannot render that on every
page load. This script runs the pipeline ONCE and writes every artifact the
app needs to data/precomputed/*.parquet.

The app then loads parquet only - it never imports arch or statsmodels.

Run locally:   python precompute.py
Run in CI:     GitHub Actions, weekly, commits the outputs.
"""
from pathlib import Path
import json
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import config
from src.data import get_data, fetch_dnsi, dnsi_summary
from src.returns import compute_returns, reconcile_vs_project2
from src.garch import fit_all
from src.dcc import build_std_resid_matrix, dcc_filter, correlation_pair
from src.regime import to_weekly, fit_markov_recursive, fit_markov_full
from src.erc import erc_weight_history
from src.momentum import build_signal_panel, build_signal_panel_full
from src.tilt import build_tilted_weights
from src.strategy_b import build_strategy_b_weights, run_strategy_b, build_ablation_variants
from src.macro import get_recent_releases
from src.risk import (latest_covariance, risk_contributions, risk_summary,
                      trailing_performance)
from src.narrative import build_narrative, weekly_moves
from src.backtest import (daily_log_to_weekly_simple, run_strategy,
                          build_60_40_weights, build_equal_weight,
                          build_pure_erc_weekly, get_rf_weekly,
                          compare_strategies, lookahead_test, performance_stats,
                          bootstrap_sharpe_ci, cumulative_equity, max_drawdown)

OUT = config.DATA_DIR / "precomputed"


def _save(obj, name: str):
    """Write a DataFrame or Series to parquet under OUT."""
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}.parquet"
    if isinstance(obj, pd.Series):
        obj.to_frame(name=obj.name or name).to_parquet(path)
    else:
        obj.to_parquet(path)
    print(f"  wrote {name}.parquet  {getattr(obj, 'shape', '')}")


def main():
    t0 = datetime.now(timezone.utc)
    print("=" * 70)
    print("PRECOMPUTE — running full pipeline")
    print("=" * 70)

    # ---- 1. Data + returns -------------------------------------------------
    print("\n[1/8] data + returns")
    panel, meta = get_data()
    returns = compute_returns(panel)
    _save(panel, "panel")
    _save(returns, "returns_daily")
    _save(reconcile_vs_project2(returns), "reconciliation")

    # ---- 2. GARCH ----------------------------------------------------------
    print("\n[2/8] GARCH")
    fits, garch_summary = fit_all(returns, verbose=False)
    _save(garch_summary, "garch_summary")
    sigmas = pd.DataFrame({a: f["sigma"] for a, f in fits.items()})
    _save(sigmas, "conditional_vol")

    # ---- 3. DCC ------------------------------------------------------------
    print("\n[3/8] DCC")
    Z = build_std_resid_matrix(fits, returns)
    dcc = dcc_filter(Z)
    sb_daily = correlation_pair(dcc, "SP500", "US_10Y_proxy")
    _save(sb_daily, "dcc_spx_10y_daily")
    # Full current correlation matrix for a heatmap
    R_now = pd.DataFrame(dcc["R"][-1], index=dcc["assets"], columns=dcc["assets"])
    _save(R_now, "corr_matrix_latest")

    # ---- 4. Regime ---------------------------------------------------------
    print("\n[4/8] regime (recursive filtered + full-sample smoothed)")
    sb_weekly = to_weekly(sb_daily)
    regime_filtered = fit_markov_recursive(sb_weekly, burn_in_end="2015-12-31")
    markov_full = fit_markov_full(sb_weekly, search_reps=20)
    regime_smoothed = markov_full["smoothed_high"]
    regime_df = pd.DataFrame({
        "dcc_weekly": sb_weekly,
        "p_high_filtered": regime_filtered,
        "p_high_smoothed": regime_smoothed,
    })
    _save(regime_df, "regime")
    regime_params = {
        "high_mean":  float(markov_full["means"][markov_full["high_idx"]]),
        "low_mean":   float(markov_full["means"][markov_full["low_idx"]]),
        "high_sigma2": float(markov_full["sigmas"][markov_full["high_idx"]]),
        "low_sigma2":  float(markov_full["sigmas"][markov_full["low_idx"]]),
        "pct_high":   float(markov_full["pct_high"]),
        "loglik":     float(markov_full["result"].llf),
    }

    # ---- 5. ERC + tilt -----------------------------------------------------
    print("\n[5/8] ERC + tilt")
    erc_hist = erc_weight_history(fits, dcc, rebal_freq="ME", verbose=False)
    signals = build_signal_panel(panel)
    signals_full = build_signal_panel_full(panel, returns)
    tilt = build_tilted_weights(erc_hist, signals, regime_filtered, verbose=False)
    cols = list(tilt["weights"].columns)
    _save(erc_hist[cols], "erc_weights_monthly")
    _save(tilt["weights"], "tilted_weights_weekly")
    _save(tilt["erc_weekly"], "erc_weights_weekly")
    _save(tilt["active_cap"], "active_cap")
    _save(signals_full[cols], "signals_daily")
    _save(tilt["signals_weekly"], "signals_weekly")

    # ---- 6. Strategy B -----------------------------------------------------
    print("\n[6/8] Strategy B (production: trend + Markov gate)")
    strat_b = build_strategy_b_weights(returns, signals_full, regime_filtered,
                                       cols, verbose=False)
    _save(strat_b["weights"], "strategyb_weights_weekly")
    _save(strat_b["cash_weight"], "strategyb_cash")
    _save(strat_b["leverage"], "strategyb_gate")

    # ---- 7. Backtests ------------------------------------------------------
    print("\n[7/8] backtests + stats")
    wk_ret = daily_log_to_weekly_simple(returns)
    idx = tilt["weights"].index
    rf = get_rf_weekly(panel, idx)
    _save(rf, "rf_weekly")

    runs = {
        "Tilted":    run_strategy(tilt["weights"], wk_ret, cost_bps=5, verbose=False),
        "PureERC":   run_strategy(build_pure_erc_weekly(erc_hist, idx, cols), wk_ret, cost_bps=5, verbose=False),
        "StrategyB": run_strategy_b(strat_b["weights"], strat_b["cash_weight"], wk_ret, rf, cost_bps=5, verbose=False),
        "60/40":     run_strategy(build_60_40_weights(idx, cols), wk_ret, cost_bps=5, verbose=False),
        "EW":        run_strategy(build_equal_weight(idx, cols), wk_ret, cost_bps=5, verbose=False),
    }
    comp = compare_strategies(runs, rf=rf)
    _save(comp, "comparison")

    # Cost sensitivity: the Sharpe edge over buy-and-hold is thin, so it is
    # honest to show at what cost level it disappears. Drawdown is unaffected
    # by cost, which is why the headline claim rests on drawdown not Sharpe.
    cost_rows = []
    for bps in [0, 5, 10, 20, 40]:
        r_t = run_strategy(tilt["weights"], wk_ret, cost_bps=bps, verbose=False)
        r_e = run_strategy(build_equal_weight(idx, cols), wk_ret, cost_bps=bps, verbose=False)
        st_t = performance_stats(r_t["net"], rf=rf, turnover=r_t["turnover"], cost=r_t["cost"])
        st_e = performance_stats(r_e["net"], rf=rf)
        cost_rows.append({
            "cost_bps":      bps,
            "drag_bps":      st_t.get("cost_bps", 0.0),
            "tilted_sharpe": st_t["sharpe_excess"],
            "ew_sharpe":     st_e["sharpe_excess"],
            "edge":          st_t["sharpe_excess"] - st_e["sharpe_excess"],
            "tilted_maxdd":  st_t["max_dd"],
            "ew_maxdd":      st_e["max_dd"],
        })
    _save(pd.DataFrame(cost_rows).set_index("cost_bps"), "cost_sensitivity")

    # Cost sensitivity: the Sharpe edge over buy-and-hold is thin, so it is
    # honest to show at what cost level it disappears. Drawdown is unaffected
    # by cost, which is why the headline claim rests on drawdown not Sharpe.
    cost_rows = []
    for bps in [0, 5, 10, 20, 40]:
        r_t = run_strategy(tilt["weights"], wk_ret, cost_bps=bps, verbose=False)
        r_e = run_strategy(build_equal_weight(idx, cols), wk_ret, cost_bps=bps, verbose=False)
        st_t = performance_stats(r_t["net"], rf=rf, turnover=r_t["turnover"], cost=r_t["cost"])
        st_e = performance_stats(r_e["net"], rf=rf)
        cost_rows.append({
            "cost_bps":      bps,
            "drag_bps":      st_t.get("cost_bps", 0.0),
            "tilted_sharpe": st_t["sharpe_excess"],
            "ew_sharpe":     st_e["sharpe_excess"],
            "edge":          st_t["sharpe_excess"] - st_e["sharpe_excess"],
            "tilted_maxdd":  st_t["max_dd"],
            "ew_maxdd":      st_e["max_dd"],
        })
    _save(pd.DataFrame(cost_rows).set_index("cost_bps"), "cost_sensitivity")

    # Net weekly returns + equity curves + drawdown traces for charts
    nets = pd.DataFrame({k: v["net"] for k, v in runs.items()})
    _save(nets, "net_weekly")
    equity = nets.apply(lambda s: (1 + s).cumprod())
    _save(equity, "equity_curves")
    drawdowns = equity.apply(lambda s: s / s.cummax() - 1.0)
    _save(drawdowns, "drawdowns")

    # ---- 8. Diagnostics: bootstrap, look-ahead, ablation, crisis episodes ---
    print("\n[8/8] diagnostics")
    boot = bootstrap_sharpe_ci(
        {k: v["net"] for k, v in runs.items()},
        rf=rf, n_boots=5000, block_size=52, baseline="StrategyB")
    boot_ci = pd.DataFrame(boot["strategy_cis"]).T
    _save(boot_ci, "bootstrap_ci")
    boot_pairs = pd.DataFrame(boot["pairwise_diffs"]).T
    _save(boot_pairs, "bootstrap_pairs")

    la_a = lookahead_test(tilt["weights"], wk_ret, extra_lag_weeks=1, cost_bps=5)
    la_b = lookahead_test(strat_b["weights"], wk_ret, extra_lag_weeks=1, cost_bps=5)

    variants = build_ablation_variants(returns, signals_full, regime_filtered, cols, verbose=False)
    ablation_runs = {n: run_strategy_b(v["weights"], v["cash"], wk_ret, rf,
                                        cost_bps=5, verbose=False)
                     for n, v in variants.items()}
    ablation = compare_strategies(ablation_runs, rf=rf)
    _save(ablation, "ablation")

    # Crisis episodes (audit Finding 5)
    episodes = {
        "GFC 2008-09":        ("2007-10-01", "2009-03-31"),
        "Eurozone 2011":      ("2011-05-01", "2011-10-31"),
        "Taper 2013":         ("2013-05-01", "2013-09-30"),
        "China/Oil 2015-16":  ("2015-08-01", "2016-02-29"),
        "Q4 2018":            ("2018-10-01", "2018-12-31"),
        "COVID 2020":         ("2020-02-01", "2020-04-30"),
        "Hiking cycle 2022":  ("2022-01-01", "2022-10-31"),
    }
    rows = []
    for name, (s, e) in episodes.items():
        row = {"episode": name, "start": s, "end": e}
        for k, net in nets.items():
            seg = net.loc[s:e]
            if len(seg) < 2:
                row[f"{k}_ret"] = np.nan
                row[f"{k}_dd"] = np.nan
                continue
            eq = (1 + seg).cumprod()
            row[f"{k}_ret"] = float(eq.iloc[-1] - 1)
            row[f"{k}_dd"] = float(max_drawdown(eq))
        rows.append(row)
    _save(pd.DataFrame(rows).set_index("episode"), "crisis_episodes")

    # ---- 8b. Ex-ante risk analytics ----------------------------------------
    print("\n[8b] ex-ante risk")
    cov_now = latest_covariance(fits, dcc, cols)
    _save(cov_now, "covariance_latest")

    rc_tilt = risk_contributions(tilt["weights"].iloc[-1], cov_now)
    rc_erc  = risk_contributions(tilt["erc_weekly"].iloc[-1], cov_now)
    _save(rc_tilt, "risk_contrib_tilted")
    _save(rc_erc,  "risk_contrib_erc")

    # Contributions measured with the covariance AS AT the rebalance date.
    # At that moment ERC is exact (1/N each); the drift visible in the
    # current-covariance version is what motivates monthly rebalancing.
    rebal_date = erc_hist.index[-1]
    dcc_dates = pd.DatetimeIndex(dcc["dates"])
    d_at = dcc_dates[dcc_dates <= rebal_date][-1]
    pos_at = dcc_dates.get_loc(d_at)
    R_at = pd.DataFrame(dcc["R"][pos_at], index=dcc["assets"],
                        columns=dcc["assets"]).loc[cols, cols]
    # NB: pull sigma BY DATE - the returns index and the DCC common-sample
    # index differ in length, so positional lookup silently misaligns them.
    sig_at = np.array([float(fits[a]["sigma"].loc[d_at]) for a in cols])
    cov_at = pd.DataFrame(np.diag(sig_at) @ R_at.values @ np.diag(sig_at),
                          index=cols, columns=cols)
    _save(risk_contributions(erc_hist.loc[rebal_date, cols], cov_at),
          "risk_contrib_erc_at_rebalance")

    risk_now = risk_summary(tilt["weights"].iloc[-1], cov_now)
    trail_12m = trailing_performance(runs["Tilted"]["net"], rf, weeks=52)
    trail_12m_ew = trailing_performance(runs["EW"]["net"], rf, weeks=52)

    # ---- 9. Market-conditions artifacts ------------------------------------
    print("\n[9/10] market conditions")

    # Per-asset conditional vol: latest level + percentile within own history
    vol_rows = []
    for a in cols:
        s_vol = sigmas[a].dropna()
        if len(s_vol) < 100:
            continue
        latest = float(s_vol.iloc[-1]) * np.sqrt(252)
        hist_ann = s_vol * np.sqrt(252)
        vol_rows.append({
            "asset":      a,
            "vol_ann":    latest,
            "pctile":     float((hist_ann <= s_vol.iloc[-1] * np.sqrt(252)).mean() * 100),
            "vol_1m_ago": float(hist_ann.iloc[-21]) if len(hist_ann) > 21 else np.nan,
            "hist_min":   float(hist_ann.min()),
            "hist_med":   float(hist_ann.median()),
            "hist_max":   float(hist_ann.max()),
        })
    _save(pd.DataFrame(vol_rows).set_index("asset"), "vol_percentiles")

    # Weekly moves with own-history percentiles, for the latest week
    _save(weekly_moves(returns, idx[-1]), "weekly_moves")

    # ---- 10. Macro releases + narrative ------------------------------------
    print("\n[10/10] macro releases + narrative")
    releases = get_recent_releases(weeks_back=4)
    if not releases.empty:
        _save(releases, "macro_releases")
    else:
        print("  no releases returned - dashboard will show a fallback line")

    # ---- DNSI --------------------------------------------------------------
    dnsi = fetch_dnsi()
    _save(dnsi, "dnsi")
    dnsi_stats = dnsi_summary(dnsi)

    # ---- Narrative (needs dnsi_stats in metadata, so assembled last) -------
    _tmp_meta = {
        "data_as_of":            str(panel.index.max().date()),
        "p_high_latest":         float(regime_filtered.iloc[-1]),
        "active_cap_latest":     float(tilt["active_cap"].iloc[-1]),
        "dnsi":                  dnsi_stats,
    }
    _D = {
        "tilted_weights_weekly": tilt["weights"],
        "signals_weekly":        tilt["signals_weekly"],
        "returns_daily":         returns,
        "regime":                regime_df,
    }
    try:
        narrative = build_narrative(_D, _tmp_meta, releases)
    except Exception as e:
        print(f"  narrative generation failed: {e}")
        narrative = {}

    # ---- Metadata sidecar --------------------------------------------------
    elapsed = (datetime.now(timezone.utc) - t0).total_seconds()
    metadata = {
        "generated_utc":   datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(elapsed, 1),
        "data_source":     meta.get("source", "unknown"),
        "data_as_of":      str(panel.index.max().date()),
        "latest_rebalance": str(idx[-1].date()),
        "n_weeks":         int(len(idx)),
        "assets":          cols,
        "regime_params":   regime_params,
        "p_high_latest":   float(regime_filtered.iloc[-1]),
        "active_cap_latest": float(tilt["active_cap"].iloc[-1]),
        "strategyb_gate_latest": float(strat_b["leverage"].iloc[-1]),
        "strategyb_gross_latest": float(strat_b["weights"].iloc[-1].sum()),
        "lookahead": {
            "Tilted":    {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                          for k, v in la_a.items()},
            "StrategyB": {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                          for k, v in la_b.items()},
        },
        "dnsi": dnsi_stats,
        "garch_methods": {a: f["method"] for a, f in fits.items()},
        "risk_now": {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                     for k, v in risk_now.items()},
        "trailing_12m":    {k: float(v) for k, v in trail_12m.items()},
        "trailing_12m_ew": {k: float(v) for k, v in trail_12m_ew.items()},
        "rebalance_date":  str(rebal_date.date()),
        "narrative": narrative,
        "n_releases": int(len(releases)) if not releases.empty else 0,
    }
    (OUT / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"  wrote metadata.json")

    print("\n" + "=" * 70)
    print(f"PRECOMPUTE COMPLETE in {elapsed:.0f}s")
    print(f"  data as of:        {metadata['data_as_of']} (source: {metadata['data_source']})")
    print(f"  latest rebalance:  {metadata['latest_rebalance']}")
    print(f"  P(high-corr):      {metadata['p_high_latest']:.3f}")
    print(f"  outputs:           {OUT}")
    print("=" * 70)


if __name__ == "__main__":
    main()