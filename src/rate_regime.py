"""
Monetary regime classification from rates and liquidity.

Two observable inputs, both from FRED, both available same-day:

    DGS2   2-year Treasury constant maturity yield  -> policy direction
    WALCL  Fed total assets                         -> liquidity direction

Every threshold below was fixed IN ADVANCE and justified by an institutional
fact, not by a backtest result. They are not to be revised after seeing
performance. If a value looks wrong, report it as a sensitivity, not a fix.

    Rate lookback      3 months   one FOMC cycle
    Rate band          +/-25bp    one policy increment
    Liquidity lookback 13 weeks   one quarter
    Liquidity band     +/-1%      meaningful change on a ~$7tn balance sheet
    Dwell time         4 weeks    one monthly rebalance cycle

The rate band produces a NEUTRAL state deliberately: when the 2-year has
moved less than a single Fed increment over a quarter, there is no policy
signal worth acting on, and the strategy runs its unmodified core.
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

# ---------------------------------------------------------------- constants --
RATE_LOOKBACK_DAYS = 63          # ~3 months of business days
RATE_BAND_BP = 25.0              # one policy increment
LIQ_LOOKBACK_DAYS = 65           # ~13 weeks of business days
LIQ_BAND_PCT = 1.0               # percent
DWELL_DAYS = 20                  # ~4 weeks of business days

RATE_STATES = ("falling", "neutral", "rising")
LIQ_STATES = ("contracting", "neutral", "expanding")

# Scoring rule, fixed in advance. Rates carry twice the weight of liquidity
# because the discount-rate channel is the direct mechanism in the equity
# duration thesis; liquidity is a second-order support channel.
RATE_SCORE = {"falling": 2, "neutral": 0, "rising": -2}
LIQ_SCORE = {"expanding": 1, "neutral": 0, "contracting": -1}
MAX_SCORE = 3
TILT_PP = 10.0

# Defensive sleeve follows the rate signal: duration helps when rates fall
# and hurts when they rise.
DEFENSIVE_ETF = {"falling": "IEF", "neutral": "IEF", "rising": "SHY"}


def score_to_tilt(score: int) -> float:
    """Percentage points toward growth. Negative means toward value."""
    return TILT_PP * score / MAX_SCORE


def score_label(score: int) -> str:
    if score >= 2:
        return "growth"
    if score == 1:
        return "growth_mild"
    if score == 0:
        return "none"
    if score == -1:
        return "value_mild"
    if score == -2:
        return "value"
    return "value_defensive"


# ------------------------------------------------------------------ helpers --
def _classify_rates(dgs2: pd.Series) -> pd.Series:
    """3-month change in the 2-year yield, banded at +/-25bp."""
    chg_bp = (dgs2 - dgs2.shift(RATE_LOOKBACK_DAYS)) * 100.0
    out = pd.Series("neutral", index=dgs2.index, dtype=object)
    out[chg_bp > RATE_BAND_BP] = "rising"
    out[chg_bp < -RATE_BAND_BP] = "falling"
    out[chg_bp.isna()] = np.nan
    return out


def _classify_liquidity(walcl: pd.Series) -> pd.Series:
    """13-week percentage change in the balance sheet, banded at +/-1%."""
    chg_pct = (walcl / walcl.shift(LIQ_LOOKBACK_DAYS) - 1.0) * 100.0
    out = pd.Series("neutral", index=walcl.index, dtype=object)
    out[chg_pct > LIQ_BAND_PCT] = "expanding"
    out[chg_pct < -LIQ_BAND_PCT] = "contracting"
    out[chg_pct.isna()] = np.nan
    return out


def _apply_dwell(states: pd.Series, dwell: int = DWELL_DAYS) -> pd.Series:
    """
    A new state must persist `dwell` business days before it is adopted.
    Prevents the book flipping on a series hovering near a threshold.
    Uses only past observations, so it introduces no look-ahead.
    """
    vals = states.tolist()
    out, current, run = [], None, 0
    for v in vals:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            out.append(np.nan)
            continue
        if current is None:
            current, run = v, dwell          # adopt the first valid state
        elif v == current:
            run = 0
        else:
            run += 1
            if run >= dwell:
                current, run = v, 0
        out.append(current)
    return pd.Series(out, index=states.index, dtype=object)


# -------------------------------------------------------------------- build --
def build_regime(panel: pd.DataFrame,
                 rate_col: str = "US_2Y",
                 liq_col: str = "WALCL",
                 dwell: int = DWELL_DAYS) -> pd.DataFrame:
    """
    Classify every date into a monetary regime.

    Parameters
    ----------
    panel : DataFrame containing the rate and liquidity columns.
    rate_col, liq_col : column names.
    dwell : business days a new state must persist. Pass 0 to disable
            (used for the sensitivity table).

    Returns
    -------
    DataFrame indexed by date with:
        rate_chg_bp, liq_chg_pct   the raw inputs
        rate_raw, liq_raw          unfiltered states
        rate, liq                  states after the dwell filter
        quadrant                   "rising|contracting" etc
        favours                    growth / value / balanced / value_defensive / none
        defensive                  IEF or SHY
    """
    for col in (rate_col, liq_col):
        if col not in panel.columns:
            raise KeyError(f"[rate_regime] '{col}' missing from panel; "
                           f"available: {list(panel.columns)}")

    dgs2 = panel[rate_col].ffill()
    walcl = panel[liq_col].ffill()

    rate_raw = _classify_rates(dgs2)
    liq_raw = _classify_liquidity(walcl)

    rate = _apply_dwell(rate_raw, dwell) if dwell else rate_raw
    liq = _apply_dwell(liq_raw, dwell) if dwell else liq_raw

    df = pd.DataFrame({
        "rate_chg_bp": (dgs2 - dgs2.shift(RATE_LOOKBACK_DAYS)) * 100.0,
        "liq_chg_pct": (walcl / walcl.shift(LIQ_LOOKBACK_DAYS) - 1.0) * 100.0,
        "rate_raw": rate_raw,
        "liq_raw": liq_raw,
        "rate": rate,
        "liq": liq,
    })
    df = df.dropna(subset=["rate", "liq"])
    df["quadrant"] = df["rate"] + "|" + df["liq"]
    df["score"] = df["rate"].map(RATE_SCORE) + df["liq"].map(LIQ_SCORE)
    df["tilt_pp"] = df["score"].apply(score_to_tilt)
    df["favours"] = df["score"].apply(score_label)
    df["defensive"] = df["rate"].map(DEFENSIVE_ETF)
    return df


def regime_summary(regime: pd.DataFrame) -> pd.DataFrame:
    """Occupancy of each quadrant: days, share of sample, number of spells."""
    rows = []
    for q, grp in regime.groupby("quadrant"):
        spells = (regime["quadrant"] != regime["quadrant"].shift()).cumsum()
        n_spells = spells[regime["quadrant"] == q].nunique()
        rows.append({
            "quadrant": q,
            "favours": grp["favours"].iloc[0],
            "score": int(grp["score"].iloc[0]),
            "tilt_pp": round(float(grp["tilt_pp"].iloc[0]), 1),
            "days": len(grp),
            "pct_of_sample": 100.0 * len(grp) / len(regime),
            "spells": n_spells,
            "avg_spell_days": len(grp) / max(n_spells, 1),
        })
    return pd.DataFrame(rows).sort_values("days", ascending=False)


def regime_spells(regime: pd.DataFrame) -> pd.DataFrame:
    """One row per continuous period in a single quadrant."""
    grp = (regime["quadrant"] != regime["quadrant"].shift()).cumsum()
    rows = []
    for _, g in regime.groupby(grp):
        rows.append({
            "start": g.index[0].date(),
            "end": g.index[-1].date(),
            "days": len(g),
            "quadrant": g["quadrant"].iloc[0],
            "favours": g["favours"].iloc[0],
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import json
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.data import get_data

    panel, meta = get_data()
    reg = build_regime(panel)

    print(f"\nregime series: {reg.index[0].date()} to {reg.index[-1].date()}"
          f"  ({len(reg):,} days)\n")
    print("OCCUPANCY")
    print(regime_summary(reg).round(1).to_string(index=False))

    print("\n\nSPELLS LONGER THAN 60 DAYS")
    sp = regime_spells(reg)
    print(sp[sp["days"] > 60].to_string(index=False))

    print("\n\nCURRENT")
    last = reg.iloc[-1]
    print(f"  as of      {reg.index[-1].date()}")
    print(f"  rate       {last['rate']:<12} ({last['rate_chg_bp']:+.0f}bp over 3m)")
    print(f"  liquidity  {last['liq']:<12} ({last['liq_chg_pct']:+.2f}% over 13w)")
    print(f"  quadrant   {last['quadrant']}")
    print(f"  score      {int(last['score']):+d}  ->  tilt {last['tilt_pp']:+.1f}pp")
    print(f"  favours    {last['favours']}")
    print(f"  defensive  {last['defensive']}")
