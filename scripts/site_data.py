"""
Single source of truth for every number the page displays.
Reads only data/precomputed/. Returns one dict. No HTML here.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PRE = ROOT / "data" / "precomputed"

SIGNAL_LABEL = {1.0: "Trending up", 0.0: "Neutral", -1.0: "Trending down"}


def _load(name):
    return pd.read_parquet(PRE / f"{name}.parquet")


def last_completed_friday(as_of: pd.Timestamp) -> pd.Timestamp:
    """Most recent Friday strictly on or before as_of."""
    d = as_of
    while d.weekday() != 4:          # 4 = Friday
        d -= pd.Timedelta(days=1)
    return d.normalize()


def build():
    meta = json.loads((PRE / "metadata.json").read_text())
    as_of = pd.Timestamp(meta["data_as_of"])

    fri = last_completed_friday(as_of)
    mon = fri - pd.Timedelta(days=4)

    rets = _load("returns_daily")
    win = rets.loc[mon:fri]

    # returns_daily are log returns -> compound with exp(sum)-1
    weekly = np.exp(win.sum()) - 1.0

    moves_meta = _load("weekly_moves")           # for display labels
    labels = moves_meta["label"].to_dict()

    sig = _load("signals_weekly")
    # the bucket labelled with the UPCOMING friday holds that week's signal,
    # so the completed week ending `fri` is the bucket labelled `fri`
    sig_now = sig.loc[:fri].iloc[-1]
    sig_prev = sig.loc[:fri].iloc[-2]

    wts = _load("tilted_weights_weekly")
    w_now = wts.loc[:fri].iloc[-1]
    w_prev = wts.loc[:fri].iloc[-2]
    w_delta = w_now - w_prev

    # contributions must use SIMPLE returns: portfolio ret = sum(w_i * r_i)
    contrib = {a: float(w_now[a]) * float(weekly[a]) for a in weekly.index}
    gross = sum(contrib.values())

    rows = []
    for a in weekly.index:
        rows.append({
            "asset": a,
            "label": labels.get(a, a),
            "ret": float(weekly[a]),
            "signal": SIGNAL_LABEL[float(sig_now[a])],
            "flipped": float(sig_now[a]) != float(sig_prev[a]),
            "weight": float(w_now[a]),
            "delta": float(w_delta[a]),
            "contrib": contrib[a],
        })
    rows.sort(key=lambda r: -abs(r["ret"]))

    material = [r for r in rows if abs(r["delta"]) >= 0.005]

    comp = _load("comparison")
    cost = _load("cost_sensitivity")
    crisis = _load("crisis_episodes")
    rc = _load("risk_contrib_erc_at_rebalance")

    return {
        "gross_return": gross,
        "week_start": mon.date().isoformat(),
        "week_end": fri.date().isoformat(),
        "data_as_of": meta["data_as_of"],
        "rebalance_date": meta.get("rebalance_date"),
        "next_rebalance": str(pd.Timestamp(meta["latest_rebalance"]).date()),
        "p_high": float(meta["p_high_latest"]),
        "tilt_cap_pp": float(meta["active_cap_latest"]),
        "gate_active": float(meta["active_cap_latest"]) < 4.0,
        "rows": rows,
        "material_changes": material,
        "any_change": bool(material),
        "risk_now": meta.get("risk_now", {}),
        "stats": comp.loc["Tilted"].to_dict(),
        "stats_ew": comp.loc["EW"].to_dict(),
        "cost_ladder": cost.reset_index().to_dict("records"),
        "crisis": crisis.reset_index()[
            ["episode", "Tilted_dd", "EW_dd"]].to_dict("records"),
        "risk_contrib": rc.reset_index(names="asset").to_dict("records"),
        "erc_target": 1.0 / len(rc),
        "lookahead": meta.get("lookahead", {}).get("Tilted", {}),
    }


if __name__ == "__main__":
    d = build()
    print(f"\nWEEK  {d['week_start']} -> {d['week_end']}   (data as of {d['data_as_of']})")
    print(f"regime P={d['p_high']:.2f}  tilt cap ±{d['tilt_cap_pp']:.0f}pp  "
          f"gate {'ON' if d['gate_active'] else 'off'}")
    print(f"\n{'ASSET':<16}{'RET':>8}  {'SIGNAL':<15}{'WEIGHT':>8}{'CHG':>8}  FLIP")
    print("-" * 66)
    for r in d["rows"]:
        print(f"{r['label']:<16}{r['ret']*100:>7.2f}%  {r['signal']:<15}"
              f"{r['weight']*100:>7.1f}%{r['delta']*100:>7.2f}  "
              f"{'*' if r['flipped'] else ''}")
    print("-" * 66)
    print(f"material weight changes (>=0.5pp): {len(d['material_changes'])}")
    s, e = d["stats"], d["stats_ew"]
    print(f"\nTilted   ret {s['ann_return']*100:.2f}%  vol {s['ann_vol']*100:.2f}%  "
          f"sharpe {s['sharpe_excess']:.2f}  maxDD {s['max_dd']*100:.1f}%")
    print(f"EW       ret {e['ann_return']*100:.2f}%  vol {e['ann_vol']*100:.2f}%  "
          f"sharpe {e['sharpe_excess']:.2f}  maxDD {e['max_dd']*100:.1f}%")
    print(f"\nERC target {d['erc_target']:.2%} | lookahead {d['lookahead']}")
