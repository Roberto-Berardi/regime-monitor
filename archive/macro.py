"""
Macro release calendar via ALFRED.

FRED indexes observations by REFERENCE period ("January 2026 payrolls").
A weekly market monitor needs the RELEASE date - when the number actually
printed. ALFRED's realtime_start field gives exactly that.

We report what was released, its value, and the prior value for the same
series. We do NOT assert that any release caused any price move: weekly
causal attribution is narrative fitting. The reader sees the releases and
the moves side by side and draws their own link.
"""
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

# Series tracked for the release calendar.
# (series_id, display name, unit hint, decimals)
RELEASE_SERIES = [
    ("PAYEMS",   "Nonfarm Payrolls",      "k jobs",  0),
    ("ICSA",     "Initial Jobless Claims", "k",      0),
    ("CPIAUCSL", "CPI (index)",           "% m/m",   2),
    ("PCEPI",    "PCE Price Index",       "% m/m",   2),
    ("UNRATE",   "Unemployment Rate",     "%",       1),
    ("INDPRO",   "Industrial Production", "% m/m",   2),
    ("RSAFS",    "Retail Sales",          "% m/m",   2),
]

# Series reported as period-over-period percent change rather than level
AS_PCT_CHANGE = {"CPIAUCSL", "PCEPI", "INDPRO", "RSAFS"}

# Series whose level is in thousands and reads better divided
AS_THOUSANDS = {"PAYEMS", "ICSA"}


def _get_fred_client():
    from fredapi import Fred
    from dotenv import load_dotenv
    import os
    load_dotenv(config.PROJECT_ROOT / ".env")
    key = os.getenv("FRED_API_KEY")
    if not key:
        raise RuntimeError("FRED_API_KEY missing from .env")
    return Fred(api_key=key)


def get_recent_releases(weeks_back: int = 4) -> pd.DataFrame:
    """
    Return every tracked macro release published in the last `weeks_back` weeks.

    For each release we report:
      release_date  : when it became public (ALFRED realtime_start)
      ref_period    : the period the number describes
      value         : the number as first published
      prior         : the previous period's value, as known at that time
      change        : value - prior (or pct change for AS_PCT_CHANGE series)

    Returns an empty DataFrame (not an error) if the API is unreachable, so
    the dashboard degrades rather than breaking.
    """
    cutoff = pd.Timestamp.today().normalize() - pd.Timedelta(weeks=weeks_back)
    rows = []

    try:
        fred = _get_fred_client()
    except Exception as e:
        print(f"[get_recent_releases] no FRED client: {e}")
        return pd.DataFrame()

    for sid, name, unit, dp in RELEASE_SERIES:
        try:
            allrel = fred.get_series_all_releases(sid)
            allrel = allrel.dropna(subset=["value"])
            allrel["realtime_start"] = pd.to_datetime(allrel["realtime_start"])
            allrel["date"] = pd.to_datetime(allrel["date"])
            allrel["value"] = pd.to_numeric(allrel["value"], errors="coerce")
            allrel = allrel.dropna(subset=["value"])

            recent = allrel[allrel["realtime_start"] >= cutoff]
            if recent.empty:
                continue

            # First publication of each reference period within the window
            first_pub = (recent.sort_values("realtime_start")
                               .groupby("date", as_index=False)
                               .first())

            for _, r in first_pub.iterrows():
                ref = r["date"]
                val = float(r["value"])

                # Skip benchmark revisions. Agencies periodically re-publish
                # years of history on a single date; those rows look like new
                # releases but describe stale reference periods. A genuine
                # print describes a period ending within ~75 days of release.
                lag_days = (r["realtime_start"] - ref).days
                if lag_days > 75:
                    continue

                # Prior period's value, as first published
                prior_periods = allrel[allrel["date"] < ref]
                prior_val = None
                if not prior_periods.empty:
                    prev_ref = prior_periods["date"].max()
                    prev_rows = (allrel[allrel["date"] == prev_ref]
                                 .sort_values("realtime_start"))
                    if not prev_rows.empty:
                        prior_val = float(prev_rows.iloc[0]["value"])

                if sid in AS_PCT_CHANGE and prior_val:
                    disp_val   = (val / prior_val - 1.0) * 100
                    disp_prior = None
                    change     = disp_val
                elif sid in AS_THOUSANDS:
                    disp_val   = val
                    disp_prior = prior_val
                    change     = (val - prior_val) if prior_val is not None else None
                else:
                    disp_val   = val
                    disp_prior = prior_val
                    change     = (val - prior_val) if prior_val is not None else None

                rows.append({
                    "release_date": r["realtime_start"].date(),
                    "series":       sid,
                    "name":         name,
                    "unit":         unit,
                    "decimals":     dp,
                    "ref_period":   ref.date(),
                    "value":        disp_val,
                    "prior":        disp_prior,
                    "change":       change,
                })

        except Exception as e:
            print(f"[get_recent_releases] {sid} failed: {e}")
            continue

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).sort_values("release_date", ascending=False)
    print(f"[get_recent_releases] {len(df)} releases in the last {weeks_back} weeks")
    return df.reset_index(drop=True)


def format_release_line(row: pd.Series) -> str:
    """One human-readable line per release, for the weekly narrative."""
    dp = int(row["decimals"])
    if row["series"] in AS_PCT_CHANGE:
        return f"{row['name']} {row['value']:+.{dp}f}% m/m ({row['ref_period']})"
    if pd.notna(row.get("prior")) and row["prior"] is not None:
        return (f"{row['name']} {row['value']:,.{dp}f} "
                f"vs {row['prior']:,.{dp}f} prior ({row['ref_period']})")
    return f"{row['name']} {row['value']:,.{dp}f} ({row['ref_period']})"
