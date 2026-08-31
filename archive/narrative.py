"""
Weekly narrative generator.

Turns precomputed artifacts into readable prose. Template-based, never
generative - the output can only contain numbers that exist in the data.

Deliberate constraint: NO causal language. We report what moved, by how much
relative to that asset's own history, what macro data printed, and how the
positioning changed. We never write "X fell BECAUSE of Y" - weekly causal
attribution is narrative fitting, and asserting it would undermine every
other number on the page.
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

LABELS = {
    "SP500": "S&P 500", "EuroStoxx50": "Euro Stoxx 50", "MSCI_EM": "MSCI EM",
    "Gold": "gold", "Oil_WTI": "WTI crude", "US_IG": "US IG credit",
    "US_HY": "US HY credit", "US_2Y_proxy": "US 2Y", "US_10Y_proxy": "US 10Y",
}


def _ordinal(n: int) -> str:
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def weekly_moves(returns_daily: pd.DataFrame, week_end: pd.Timestamp) -> pd.DataFrame:
    """
    Weekly simple return per asset for the week ending `week_end`, plus the
    percentile of that move's magnitude within the asset's own history.

    Percentile answers "how unusual was this move for THIS asset" - which is
    the only honest way to compare a 2% move in US_2Y against a 6% move in oil.
    """
    weekly = np.exp(returns_daily.resample("W-FRI").sum()) - 1.0
    if week_end not in weekly.index:
        week_end = weekly.index[weekly.index <= week_end][-1]

    this_week = weekly.loc[week_end]
    rows = []
    for a in weekly.columns:
        hist = weekly[a].dropna()
        if len(hist) < 50:
            continue
        move = float(this_week[a])
        pct = float((hist.abs() <= abs(move)).mean() * 100)
        rows.append({
            "asset": a,
            "label": LABELS.get(a, a),
            "ret": move,
            "abs_pctile": pct,
        })
    df = pd.DataFrame(rows).set_index("asset")
    return df.sort_values("ret", ascending=False)


def signal_changes(signals_weekly: pd.DataFrame, week_end: pd.Timestamp) -> list:
    """Assets whose combined trend signal changed state this week."""
    if week_end not in signals_weekly.index:
        return []
    pos = signals_weekly.index.get_loc(week_end)
    if pos == 0:
        return []
    now = signals_weekly.iloc[pos]
    prev = signals_weekly.iloc[pos - 1]

    out = []
    for a in signals_weekly.columns:
        if float(now[a]) != float(prev[a]):
            out.append({
                "asset": a,
                "label": LABELS.get(a, a),
                "from": float(prev[a]),
                "to": float(now[a]),
            })
    return out


def weight_changes(weights: pd.DataFrame, week_end: pd.Timestamp,
                   threshold: float = 0.005) -> list:
    """Assets whose weight moved by more than `threshold` (default 50bp)."""
    if week_end not in weights.index:
        return []
    pos = weights.index.get_loc(week_end)
    if pos == 0:
        return []
    delta = weights.iloc[pos] - weights.iloc[pos - 1]
    out = []
    for a, d in delta.items():
        if abs(d) >= threshold:
            out.append({
                "asset": a,
                "label": LABELS.get(a, a),
                "delta": float(d),
                "new": float(weights.iloc[pos][a]),
            })
    return sorted(out, key=lambda x: -abs(x["delta"]))


def _fmt_signal_word(v: float) -> str:
    return {1.0: "positive", -1.0: "negative", 0.0: "neutral"}.get(v, "neutral")


def build_narrative(D: dict, M: dict, releases: pd.DataFrame = None) -> dict:
    """
    Assemble the weekly brief. Returns a dict of paragraph strings so the
    dashboard can lay them out independently.

    Keys: regime, moves, signals, positioning, macro, sentiment
    """
    tilt_w = D["tilted_weights_weekly"]
    sig_w = D["signals_weekly"]
    rets = D["returns_daily"]
    week_end = tilt_w.index[-1]

    # W-FRI labels a bucket with the UPCOMING Friday, so the final bucket is
    # usually incomplete. Report sessions elapsed rather than implying a full week.
    data_end = pd.Timestamp(M.get("data_as_of", str(rets.index[-1].date())))
    sessions = int(len(pd.bdate_range(week_end - pd.Timedelta(days=4), data_end)))
    sessions = max(1, min(sessions, 5))
    partial = data_end < week_end

    out = {}
    out["partial_week"] = bool(partial)
    out["sessions"] = sessions
    out["period_label"] = (f"week to date ({sessions} of 5 sessions)"
                           if partial else "week")

    # --- Regime -------------------------------------------------------------
    p_high = M["p_high_latest"]
    cap = M["active_cap_latest"]
    regime_df = D["regime"]
    p_series = regime_df["p_high_filtered"].dropna()
    p_prev = float(p_series.iloc[-2]) if len(p_series) > 1 else p_high

    if p_high > 0.70:
        state = "high-correlation"
        implication = ("stock-bond diversification is impaired, so the tactical "
                       f"tilt runs at half strength (±{cap:.0f}pp)")
    else:
        state = "low-correlation"
        implication = (f"stock-bond diversification is working, so the tilt runs "
                       f"at full strength (±{cap:.0f}pp)")

    moved = abs(p_high - p_prev) > 0.10
    change_txt = (f" The probability moved {p_prev:.2f} → {p_high:.2f} over the week."
                  if moved else " Unchanged from last week.")
    out["regime"] = (
        f"The filtered probability of the {state} regime is {p_high:.2f}."
        f"{change_txt} At this level {implication}."
    )

    # --- Moves --------------------------------------------------------------
    mv = weekly_moves(rets, week_end)
    if len(mv):
        top = mv.iloc[0]
        bot = mv.iloc[-1]
        notable = mv[mv["abs_pctile"] >= 85].sort_values("abs_pctile", ascending=False)

        lead_label = top["label"]
        lead_label = lead_label[0].upper() + lead_label[1:]
        span = out["period_label"]
        parts = [f"Over the {span}:"]
        parts += [
            f"{lead_label.lower()} led at {top['ret']:+.1%}; "
            f"{bot['label']} lagged at {bot['ret']:+.1%}."
        ]
        if len(notable):
            n = notable.iloc[0]
            parts.append(
                f"{n['label'].capitalize()}'s {abs(n['ret']):.1%} move was in the "
                f"{_ordinal(int(round(n['abs_pctile'])))} percentile of its own "
                f"weekly moves since 2007."
            )
        if len(notable) > 1:
            others = ", ".join(x["label"] for _, x in notable.iloc[1:].iterrows())
            parts.append(f"Also unusually large for their own history: {others}.")
        out["moves"] = " ".join(parts)
    else:
        out["moves"] = "Insufficient return history to summarise weekly moves."

    # --- Signals ------------------------------------------------------------
    flips = signal_changes(sig_w, week_end)
    if not flips:
        out["signals"] = ("No trend signals changed state this week. Positioning "
                          "moves reflect volatility and correlation updates only.")
    else:
        bits = [
            f"{f['label']} turned {_fmt_signal_word(f['to'])} "
            f"(from {_fmt_signal_word(f['from'])})"
            for f in flips
        ]
        out["signals"] = (
            (f"One trend signal changed state: " if len(flips) == 1
             else f"{len(flips)} trend signals changed state: ") + "; ".join(bits) + ". "
            "A signal requires 12-1 month momentum and the 200-day moving average "
            "to agree; disagreement produces no position."
        )

    # --- Positioning --------------------------------------------------------
    wc = weight_changes(tilt_w, week_end)
    if not wc:
        out["positioning"] = "No single asset weight moved more than 50bp this week."
    else:
        bits = [
            f"{w['label']} {w['delta']*100:+.1f}pp to {w['new']:.1%}"
            for w in wc[:4]
        ]
        out["positioning"] = "Largest weight changes: " + "; ".join(bits) + "."

    # --- Macro --------------------------------------------------------------
    if releases is None or releases.empty:
        out["macro"] = "No tracked macro releases in the past week."
    else:
        from src.macro import format_release_line
        cutoff = pd.Timestamp(week_end).date() - pd.Timedelta(days=7)
        recent = releases[pd.to_datetime(releases["release_date"]).dt.date >= cutoff]
        if recent.empty:
            recent = releases.head(3)
            lead = "Most recent tracked releases: "
        else:
            lead = "Released this week: "
        lines = [format_release_line(r) for _, r in recent.iterrows()]
        out["macro"] = lead + "; ".join(lines) + "."

    # --- Sentiment ----------------------------------------------------------
    d = M.get("dnsi", {})
    if d:
        direction = "improving" if d["delta_1m_12m"] > 0 else "deteriorating"
        out["sentiment"] = (
            f"Economic news sentiment sits at the "
            f"{_ordinal(int(round(d['percentile'])))} percentile of its history "
            f"since 1980, {direction} over the past month "
            f"({d['recent_1m']:+.3f} vs {d['recent_12m']:+.3f} trailing 12M). "
            f"Reference indicator only — not a portfolio input."
        )
    else:
        out["sentiment"] = ""

    out["week_end"] = str(week_end.date())
    return out
