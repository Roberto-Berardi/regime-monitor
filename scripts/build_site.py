"""
Render docs/index.html from data/rotation/.

Reads the artifacts, formats every figure, applies the conditional stance
wording, and fills templates/index.html.j2. No computation happens here — if a
number is wrong, it is wrong in precompute_rotation.py.

Writes to a temp file and only replaces the live page once rendering has
succeeded, so a crash mid-build cannot leave a broken page up.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "data" / "rotation"
DOCS = ROOT / "docs"
TPL = ROOT / "templates"

LINKS = {"email": "", "linkedin": "", "cv": "",
         "github": "https://github.com/Roberto-Berardi/regime-monitor"}

# score -> accent colour, per the agreed spec
ACCENT = {3: "#6D4AE0", 2: "#6D4AE0", 1: "#8B6BE8", 0: "#8A99A2",
          -1: "#D9AC58", -2: "#C08A20", -3: "#C08A20"}
ACCENT_SOFT = {3: "#F1EDFE", 2: "#F1EDFE", 1: "#F7F4FE", 0: "#F6F9FA",
               -1: "#FDF9F0", -2: "#FBF4E3", -3: "#FBF4E3"}
ACCENT_DEEP = {3: "#4A2FB8", 2: "#4A2FB8", 1: "#4A2FB8", 0: "#4E5D65",
               -1: "#8A6110", -2: "#8A6110", -3: "#8A6110"}

RATE_PHRASE = {"rising": "risen", "neutral": "been broadly unchanged",
               "falling": "fallen"}
LIQ_PHRASE = {"expanding": "expanded", "neutral": "been broadly unchanged",
              "contracting": "contracted"}


def pct(x, dp=1, sign=False):
    return (f"{{:+.{dp}f}}%" if sign else f"{{:.{dp}f}}%").format(x * 100)


def longdate(d):
    return pd.Timestamp(d).strftime("%-d %b %Y")


def shortdate(d):
    return pd.Timestamp(d).strftime("%a %-d %b %Y")


def action_fragment(score, tilt_pp):
    if score == 0:
        return "No tactical adjustment — the book runs at its policy weights."
    side = "long" if score > 0 else "short"
    txt = (f"The book is tilted <b>{abs(tilt_pp):.1f} percentage points</b> "
           f"toward {side}-duration assets.")
    if score <= -3:
        txt += (" A further <b>10 points</b> move out of risk assets into the "
                "defensive sleeve.")
    return txt


def stance_sentence(score, rate_state, liq_state, tilt_pp):
    action = action_fragment(score, tilt_pp)
    if rate_state == "neutral" and liq_state == "neutral":
        return (f"The current reading is <b>{score:+d}</b>: neither the two-year "
                f"yield nor the Federal Reserve's balance sheet has moved past "
                f"its threshold. {action}")
    return (f"The current reading is <b>{score:+d}</b>: the two-year yield has "
            f"{RATE_PHRASE[rate_state]} over the past quarter while the Federal "
            f"Reserve's balance sheet has {LIQ_PHRASE[liq_state]}. {action}")


def dwell_note(m):
    r = m["rates"]
    inside = abs(r["change_bp"]) < m["params"]["rate_band_bp"]
    if inside and r["state"] != "neutral":
        return (f'<b>A state must persist four weeks before it is recognised, '
                f'which is why a current reading of {r["change_bp"]:+.0f}bp '
                f'continues to register as {r["state"]}.</b> The move is inside '
                f'the ±{m["params"]["rate_band_bp"]:.0f}bp band, but the previous '
                f'state stands until the new one holds — a rule fixed in advance '
                f'to stop the book flipping on noise.')
    return ('<b>A state must persist four weeks before it is recognised.</b> A '
            'reading that crosses a threshold does not move the book until it '
            'has held for a month — a rule fixed in advance to stop the book '
            'flipping on noise.')


def build_chart_json():
    eq = pd.read_parquet(ART / "equity_monthly.parquet")
    lr = np.log(eq / eq.shift(1)).dropna()
    return json.dumps({
        "dates": [d.strftime("%Y-%m") for d in lr.index],
        "strategy": [round(float(v), 6) for v in lr["strategy"]],
        "benchmark": [round(float(v), 6) for v in lr["benchmark"]],
        "spy": [round(float(v), 6) for v in lr["spy"]] if "spy" in lr else [],
    }, separators=(",", ":"))


def main():
    chart = build_chart_json()
    m = json.loads((ART / "metadata.json").read_text())
    periods = pd.read_parquet(ART / "periods.parquet")
    policy = pd.read_parquet(ART / "policy.parquet")
    tests = pd.read_parquet(ART / "tests.parquet")
    weights = pd.read_parquet(ART / "weights_monthly.parquet")

    score = int(m["score"])
    tilt = float(m["tilt_pp"])
    budget = float(m["budget"])
    w = m["weights"]
    b = m["buckets"]
    neutral_bucket = budget / 2

    defensive_is_ief = m["defensive"] == "IEF"
    ief_held = w.get("IEF", 0) > 0.001
    shy_held = w.get("SHY", 0) > 0.001

    r, l = m["rates"], m["liquidity"]
    h, bm, boot = m["headline"], m["benchmark"], m["bootstrap"]

    ctx = {
        # freshness
        "data_as_of": shortdate(m["data_as_of"]),
        "weights_set_on": shortdate(m["weights_set_on"]),
        "obs_long": longdate(m["data_as_of"]),
        "budget_pct": f"{budget*100:.0f}",
        "defensive_pct": f"{(1-budget)*100:.0f}",
        "built_utc": datetime.now(timezone.utc).strftime("%-d %b %Y, %H:%M"),

        # stance
        "score": score,
        "score_signed": f"{score:+d}".replace("-", "−"),
        "tilt_signed": f"{tilt:+.1f}".replace("-", "−"),
        "stance_sentence": stance_sentence(score, r["state"], l["state"], tilt),
        "dwell_note": dwell_note(m),

        # gauge
        "gauge_pct": round((3 - score) / 6 * 100, 2),
        "gauge_label": "no tilt" if score == 0 else f"tilt {abs(tilt):.1f}pp",
        "accent": ACCENT[score],
        "accent_soft": ACCENT_SOFT[score],
        "accent_deep": ACCENT_DEEP[score],

        # allocation
        "neutral_long": pct(neutral_bucket),
        "neutral_short": pct(neutral_bucket),
        "w_long": pct(b["long"]), "w_short": pct(b["short"]), "w_def": pct(b["defensive"]),
        "w_long_raw": round(b["long"] * 100, 1),
        "w_short_raw": round(b["short"] * 100, 1),
        "w_def_raw": round(b["defensive"] * 100, 1),
        "tilt_long": f"{-tilt:+.1f}".replace("-", "−") if score else "—",
        "tilt_short": f"{tilt:+.1f}".replace("-", "−") if score else "—",
        "w": {
            "IWF": pct(w["IWF"]), "GLD": pct(w["GLD"]),
            "IWD": pct(w["IWD"]), "XLE": pct(w["XLE"]),
            "SHY": pct(w["SHY"]) if shy_held else "not held",
            "IEF": pct(w["IEF"]) if ief_held else "not held",
            "IEF_held": ief_held, "SHY_held": shy_held,
        },

        # FRED cards
        "rates": {
            "from_v": f"{r['from_value']:.2f}%", "to_v": f"{r['to_value']:.2f}%",
            "change": f"{r['change_bp']:+.0f}bp".replace("-", "−"),
            "window": f"{longdate(r['from_date'])} → {longdate(r['last_obs'])}",
        },
        "liq": {
            "from_v": f"${r_tn(l['from_value'])}", "to_v": f"${r_tn(l['to_value'])}",
            "change": f"{l['change_pct']:+.2f}%".replace("-", "−"),
            "window": f"{longdate(l['from_date'])} → {longdate(l['last_obs'])}",
        },

        # performance
        "sh_strat": f"{h['sharpe']:.3f}", "sh_bench": f"{bm['sharpe']:.3f}",
        "edge_full": f"{boot['observed']:+.3f}".replace("-", "−"),
        "edge_post09": f"{periods.iloc[1]['gain']:+.3f}".replace("-", "−"),
        "boot_ci": f"{boot['ci_low']:+.3f} to {boot['ci_high']:+.3f}".replace("-", "−"),
        "boot_p": f"{boot['p_value']:.3f}",
        "sample_range": (f"{pd.Timestamp(m['sample']['start']).strftime('%B %Y')} – "
                         f"{pd.Timestamp(m['sample']['end']).strftime('%B %Y')}"),
        "bench_eq": pct(budget / 4, 2), "bench_bond": pct((1 - budget) / 2, 2),
        "rf_rate": round(m["risk_free"]["mean"], 4),
        "periods": [{"period": p["period"], "strategy": f"{p['strategy']:.3f}",
                     "benchmark": f"{p['benchmark']:.3f}",
                     "gain": f"{p['gain']:+.3f}".replace("-", "−")}
                    for _, p in periods.iterrows()],
        "policy": [{"budget": f"{p['budget']*100:.0f}%",
                    "ann_return": pct(p["ann_return"], 2),
                    "ann_vol": pct(p["ann_vol"], 2),
                    "sharpe": f"{p['sharpe']:.3f}",
                    "max_dd": pct(p["max_dd"], 1).replace("-", "−"),
                    "pick": abs(p["budget"] - budget) < 1e-9}
                   for _, p in policy.iterrows()],
        "tests": tests.to_dict("records"),
        "chart_json": chart,
        "chart_first": json.loads(chart)["dates"][0],
        "chart_last": json.loads(chart)["dates"][-1],
        "links": LINKS,
    }

    env = Environment(loader=FileSystemLoader(TPL),
                      autoescape=select_autoescape(["html"]))
    html = env.get_template("index.html.j2").render(**ctx)

    DOCS.mkdir(exist_ok=True)
    tmp = DOCS / "index.html.tmp"
    tmp.write_text(html, encoding="utf-8")
    tmp.replace(DOCS / "index.html")
    (DOCS / ".nojekyll").touch()

    print(f"built docs/index.html  ({len(html):,} bytes)")
    print(f"  data as of {m['data_as_of']}  ·  score {score:+d}  ·  tilt {tilt:+.1f}pp")
    print(f"  defensive sleeve: {m['defensive']}")
    print(f"  sharpe {h['sharpe']:.3f} vs {bm['sharpe']:.3f}  "
          f"edge {boot['observed']:+.3f}  p {boot['p_value']:.3f}")
    print(f"  chart: {len(json.loads(ctx['chart_json'])['dates'])} monthly points")


def r_tn(v):
    """WALCL arrives in millions."""
    return f"{v/1e6:.3f}tn"


if __name__ == "__main__":
    main()
