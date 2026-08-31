"""
Two fixes.

1. The KPI row disagreed with every table because the chart carried MONTHLY
   returns while the artifacts compute on DAILY. Monthly sampling misses
   intra-month drawdowns and understates volatility, so the same strategy
   showed −17.9% and 0.740 in the KPIs against −24.0% and 0.625 in the tables.
   Now the chart carries daily returns and the two agree by construction.

2. Section 4's Tests and Limitations tables were never templated - they still
   held mockup values computed on the old zero-cash-rate basis. Both now come
   from the artifacts.
"""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent


# ─────────────────────────────────────────── 1. precompute: daily series + s4 data
p = ROOT / "precompute_rotation.py"
s = p.read_text()

old = '''    monthly = curves.resample("ME").sum()
    (np.exp(monthly.cumsum()) * 100).to_parquet(OUT / "equity_monthly.parquet")'''
new = '''    monthly = curves.resample("ME").sum()
    (np.exp(monthly.cumsum()) * 100).to_parquet(OUT / "equity_monthly.parquet")

    # daily log returns for the chart. The page recomputes its KPIs from these,
    # so they must be on the same basis as every figure in the tables.
    curves.to_parquet(OUT / "returns_daily.parquet")'''
assert old in s, "equity block"
s = s.replace(old, new, 1)

# the four-sleeve comparison, for the gold/energy row
old = '''    # ---- tests table (verdicts recorded, not recomputed daily) ----'''
new = '''    # ---- four sleeves vs six, for the gold/energy row in section 4 ----
    four = backtest(rets, reg, long_c=["Growth"], short_c=["Value"])["net"].loc[common]
    s4, s6 = stats(four), stats(strat_net)
    gold_energy_note = (
        f"Sharpe {s6['sharpe']:.3f} from {s4['sharpe']:.3f}, worst loss "
        f"{s6['max_dd']*100:.1f}% from {s4['max_dd']*100:.1f}%, turnover flat."
    ).replace("-", "\\u2212")

    gv_corr = float(rets[["Growth", "Value"]].dropna().corr().iloc[0, 1])

    # ---- tests table (verdicts recorded, not recomputed daily) ----'''
assert old in s, "tests anchor"
s = s.replace(old, new, 1)

s = s.replace('''         "note": "Sharpe 0.812 from 0.744, worst loss −24.0% from −29.9%, turnover flat."}''',
              '''         "note": gold_energy_note}''')

# limitations, driven by the artifacts
old = '''    # ---- current readings ----'''
new = '''    # ---- limitations, so the page never carries a stale number ----
    tilt_delta = float(attribution.iloc[-1]["delta"])
    pd.DataFrame([
        {"limit": f"Growth and value correlate {gv_corr:.2f}",
         "note": "This picks a side of one axis, it doesn't diversify."},
        {"limit": "Rates fall in easings and in panics",
         "note": "The model scores both alike; 2008 went the wrong way."},
        {"limit": f"The tilt adds +{tilt_delta:.3f} of the +{boot['observed']:.3f}",
         "note": "Volatility weighting and the bond switch do most of the work."},
        {"limit": "Reads current conditions", "note": "Forecasts nothing."},
    ]).to_parquet(OUT / "limitations.parquet")

    # ---- current readings ----'''
assert old in s, "limitations anchor"
s = s.replace(old, new, 1)

# backtest() needs to accept a custom universe
old = '''def backtest(rets, reg, budget=BUDGET, invvol=True, bonds="switch", tilt=True):
    """One configuration. bonds: switch | split | ief"""
    r = rets[SLEEVES].dropna()'''
new = '''def backtest(rets, reg, budget=BUDGET, invvol=True, bonds="switch", tilt=True,
             long_c=None, short_c=None):
    """One configuration. bonds: switch | split | ief"""
    long_c = long_c or LONG
    short_c = short_c or SHORT
    sleeves = long_c + short_c + ["Bond_Long", "Bond_Short"]
    r = rets[sleeves].dropna()'''
assert old in s, "backtest signature"
s = s.replace(old, new, 1)
body = s[s.index("def backtest("):s.index("def benchmark_e1(")]
fixed = (body
         .replace("inverse_vol(cov, LONG), inverse_vol(cov, SHORT)",
                  "inverse_vol(cov, long_c), inverse_vol(cov, short_c)")
         .replace("lw = pd.Series(1.0 / len(LONG), index=LONG)",
                  "lw = pd.Series(1.0 / len(long_c), index=long_c)")
         .replace("sw = pd.Series(1.0 / len(SHORT), index=SHORT)",
                  "sw = pd.Series(1.0 / len(short_c), index=short_c)")
         .replace("w = pd.Series(0.0, index=SLEEVES)", "w = pd.Series(0.0, index=sleeves)")
         .replace("for c in LONG:", "for c in long_c:")
         .replace("for c in SHORT:", "for c in short_c:"))
s = s.replace(body, fixed, 1)
p.write_text(s)
import ast; ast.parse(s)
print("precompute_rotation.py: daily returns, four-sleeve figures, limitations")


# ─────────────────────────────────────────── 2. build_site: daily chart + s4
p = ROOT / "scripts" / "build_site.py"
s = p.read_text()

old = '''def build_chart_json():
    eq = pd.read_parquet(ART / "equity_monthly.parquet")
    lr = np.log(eq / eq.shift(1)).dropna()
    return json.dumps({
        "dates": [d.strftime("%Y-%m") for d in lr.index],'''
new = '''def build_chart_json():
    """Daily log returns — the same basis the tables are computed on."""
    lr = pd.read_parquet(ART / "returns_daily.parquet").dropna()
    return json.dumps({
        "dates": [d.strftime("%Y-%m-%d") for d in lr.index],'''
assert old in s, "chart json"
s = s.replace(old, new, 1)

s = s.replace('    tests = pd.read_parquet(ART / "tests.parquet")',
              '    tests = pd.read_parquet(ART / "tests.parquet")\n'
              '    limits = pd.read_parquet(ART / "limitations.parquet")', 1)
s = s.replace('        "tests": tests.to_dict("records"),',
              '        "tests": tests.to_dict("records"),\n'
              '        "limits": limits.to_dict("records"),', 1)
s = s.replace('"chart_first": json.loads(chart)["dates"][0],',
              '"chart_first": json.loads(chart)["dates"][0][:7],')
s = s.replace('"chart_last": json.loads(chart)["dates"][-1],',
              '"chart_last": json.loads(chart)["dates"][-1][:7],')
p.write_text(s)
ast.parse(s)
print("build_site.py: daily chart series, limitations passed through")


# ─────────────────────────────────────────── 3. template: s4 tables + daily JS
p = ROOT / "templates" / "index.html.j2"
s = p.read_text()

# tests table rows 1-4 keep their prose; row 0 already reads from the artifact.
# limitations table becomes a loop.
old = re.search(r'        <tr><td>Growth and value correlate.*?</tbody>', s, re.S).group(0)
new = '''        {% for l in limits %}<tr><td>{{ l.limit }}</td><td>{{ l.note }}</td></tr>
        {% endfor %}</tbody>'''
s = s.replace(old, new, 1)

# the other four test rows come from the artifact too
for i in (1, 2, 3, 4):
    pat = re.search(
        r'          <td>(200-day trend filter|PCA correlation warning|'
        r'Scaling bond duration by score|Sixteen others)</td>\n'
        r'          <td><span class="tag r">Rejected</span>(.*?)</td>', s, re.S)
    if pat:
        s = s.replace(pat.group(0),
                      f'          <td>{{{{ tests[{i}].test }}}}</td>\n'
                      f'          <td><span class="tag r">Rejected</span>'
                      f'{{{{ tests[{i}].note }}}}</td>', 1)

# the chart JS parses dates as months; make it parse days
s = s.replace('''  const dates = SERIES.dates.map(d => {
    const [y, m] = d.split('-').map(Number);
    return new Date(y, m - 1, 1);
  });''',
'''  const dates = SERIES.dates.map(d => {
    const [y, m, dd] = d.split('-').map(Number);
    return new Date(y, m - 1, dd);
  });''')
# annualisation factor: daily, not monthly
s = s.replace("const k=lr.length,y=k/12;", "const k=lr.length,y=k/252;")
s = s.replace("vol=sd*Math.sqrt(12)", "vol=sd*Math.sqrt(252)")
# period tabs are in months -> convert to trading days
s = s.replace('data-m="12"', 'data-m="252"').replace('data-m="60"', 'data-m="1260"')
s = s.replace('data-m="120"', 'data-m="2520"')
p.write_text(s)
print("template: section 4 tables looped, chart JS switched to daily")
print("\nnow run:  python precompute_rotation.py && python scripts/build_site.py")
