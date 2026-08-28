"""
One-time converter: page-final.html -> templates/index.html.j2

Replaces every hardcoded figure with a Jinja placeholder. Each replacement is
anchored with enough surrounding text to be unambiguous, and asserts it matched
exactly once, so a silent miss is impossible.

Run once. After this, edit the template, not the HTML.
"""
from pathlib import Path
import sys

SRC = Path(sys.argv[1] if len(sys.argv) > 1 else "page-final.html")
DST = Path("templates/index.html.j2")

s = SRC.read_text()

def sub(old, new, why):
    global s
    n = s.count(old)
    assert n == 1, f"[{why}] matched {n} times, expected 1:\n{old[:110]}"
    s = s.replace(old, new, 1)

# ── hero freshness ────────────────────────────────────────────────
sub('<div class="f-val"><span class="pulse"></span>Mon 24 Aug 2026</div>',
    '<div class="f-val"><span class="pulse"></span>{{ data_as_of }}</div>', "data date")
sub('<div class="f-val">Fri 31 Jul 2026</div>',
    '<div class="f-val">{{ weights_set_on }}</div>', "weights set")
sub('<div class="f-val">55% risk / 45% defensive</div>',
    '<div class="f-val">{{ budget_pct }}% risk / {{ defensive_pct }}% defensive</div>',
    "risk anchor")

# ── stance ────────────────────────────────────────────────────────
sub('<div class="date">Latest observation 24 August 2026 · reviewed monthly</div>',
    '<div class="date">Latest observation {{ obs_long }} · reviewed monthly</div>',
    "stance date")
sub('''      <div class="lbl">The reading today</div>
      <p>
        The current reading is <b>−2</b>: the two-year yield has risen over the
        past quarter while the Federal Reserve's balance sheet has been broadly
        unchanged. The book is tilted <b>6.7 percentage points</b> toward
        short-duration assets.
      </p>''',
'''      <div class="lbl">The reading today</div>
      <p>{{ stance_sentence }}</p>''', "stance sentence")

# ── gauge ─────────────────────────────────────────────────────────
sub('<div class="tag" style="left:83.33%">tilt 6.7pp</div>',
    '<div class="tag" style="left:{{ gauge_pct }}%">{{ gauge_label }}</div>', "gauge tag")
sub('<div class="dot" style="left:83.33%"></div>',
    '<div class="dot" style="left:{{ gauge_pct }}%"></div>', "gauge dot")
for v, sc in [("1%", "+3"), ("16.67%", "+2"), ("33.33%", "+1"),
              ("50%", "0"), ("66.67%", "−1"), ("99%", "−3")]:
    sub(f'<span class="num" style="left:{v}">{sc}</span>',
        f'<span class="num{{{{ \' on\' if score == {sc.replace("−","-").replace("+","")} else \'\' }}}}" '
        f'style="left:{v}">{sc}</span>', f"gauge num {sc}")
sub('<span class="num on" style="left:83.33%">−2</span>',
    '<span class="num{{ \' on\' if score == -2 else \'\' }}" style="left:83.33%">−2</span>',
    "gauge num -2")

# ── regime bands ──────────────────────────────────────────────────
sub('        <div class="band l">', '        <div class="band l{{ \' live\' if score > 0 else \'\' }}">', "band +")
sub('        <div class="band n">', '        <div class="band n{{ \' live\' if score == 0 else \'\' }}">', "band 0")
sub('        <div class="band s live">', '        <div class="band s{{ \' live\' if score < 0 else \'\' }}">', "band -")
sub('''          <span></span>
        </div>
        <div class="band n''',
    '''          <span class="now">{{ 'Current' if score > 0 else '' }}</span>
        </div>
        <div class="band n''', "flag +")
sub('''          <span></span>
        </div>
        <div class="band s''',
    '''          <span class="now">{{ 'Current' if score == 0 else '' }}</span>
        </div>
        <div class="band s''', "flag 0")
sub('<span class="now">Current</span>',
    '<span class="now">{{ \'Current\' if score < 0 else \'\' }}</span>', "flag -")

# ── allocation groups ─────────────────────────────────────────────
sub('<s>27.5%</s> → <b>20.8%</b>', '<s>{{ neutral_long }}</s> → <b>{{ w_long }}</b>', "long move")
sub('<s>27.5%</s> → <b>34.2%</b>', '<s>{{ neutral_short }}</s> → <b>{{ w_short }}</b>', "short move")
sub('<s>45.0%</s> → <b>45.0%</b>', '<s>{{ defensive_pct }}.0%</s> → <b>{{ w_def }}</b>', "def move")
sub('<div class="gbar"><i style="width:20.8%"></i></div>',
    '<div class="gbar"><i style="width:{{ w_long_raw }}%"></i></div>', "long bar")
sub('<div class="gbar"><i style="width:34.2%"></i></div>',
    '<div class="gbar"><i style="width:{{ w_short_raw }}%"></i></div>', "short bar")
sub('<div class="gbar"><i style="width:45%"></i></div>',
    '<div class="gbar"><i style="width:{{ w_def_raw }}%"></i></div>', "def bar")

sub('<div><span>Growth · IWF</span><span class="v">9.7%</span></div>',
    '<div><span>Growth · IWF</span><span class="v">{{ w.IWF }}</span></div>', "IWF")
sub('<div><span>Gold · GLD</span><span class="v">11.1%</span></div>',
    '<div><span>Gold · GLD</span><span class="v">{{ w.GLD }}</span></div>', "GLD")
sub('<div><span>Value · IWD</span><span class="v">24.8%</span></div>',
    '<div><span>Value · IWD</span><span class="v">{{ w.IWD }}</span></div>', "IWD")
sub('<div><span>Energy · XLE</span><span class="v">9.4%</span></div>',
    '<div><span>Energy · XLE</span><span class="v">{{ w.XLE }}</span></div>', "XLE")
sub('<div><span>US 1–3y · SHY</span><span class="v">45.0%</span></div>',
    '<div><span>US 1–3y · SHY</span><span class="v">{{ w.SHY }}</span></div>', "SHY")
sub('<div class="off"><span>US 7–10y · IEF</span><span class="v">not held</span></div>',
    '<div class="{{ \'off\' if w.IEF_held == false else \'\' }}"><span>US 7–10y · IEF</span>'
    '<span class="v">{{ w.IEF }}</span></div>', "IEF")

# ── tilt table + arithmetic ───────────────────────────────────────
sub('''          <div class="rl">Long duration</div>
          <div class="rv">27.5%</div><div class="rv d">−6.7</div>
          <div class="rv now">20.8%</div>''',
    '''          <div class="rl">Long duration</div>
          <div class="rv">{{ neutral_long }}</div><div class="rv d">{{ tilt_long }}</div>
          <div class="rv now">{{ w_long }}</div>''', "tilt row long")
sub('''          <div class="rl">Short duration</div>
          <div class="rv">27.5%</div><div class="rv d">+6.7</div>
          <div class="rv now">34.2%</div>''',
    '''          <div class="rl">Short duration</div>
          <div class="rv">{{ neutral_short }}</div><div class="rv d">{{ tilt_short }}</div>
          <div class="rv now">{{ w_short }}</div>''', "tilt row short")
sub('''          <div class="rl">Defensive</div>
          <div class="rv">45.0%</div><div class="rv">—</div>
          <div class="rv now">45.0%</div>''',
    '''          <div class="rl">Defensive</div>
          <div class="rv">{{ defensive_pct }}.0%</div><div class="rv">—</div>
          <div class="rv now">{{ w_def }}</div>''', "tilt row def")
sub('''        <div class="formula">tilt = 10pp × score ÷ 3<br>
          &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;= 10 × (−2) ÷ 3 = <b>−6.7pp</b></div>''',
    '''        <div class="formula">tilt = 10pp × score ÷ 3<br>
          &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;= 10 × ({{ score_signed }}) ÷ 3 =
          <b>{{ tilt_signed }}pp</b></div>''', "formula")

# ── FRED cards ────────────────────────────────────────────────────
sub('<div class="move">4.01% → 4.19% · <span class="d">+18bp</span></div>',
    '<div class="move">{{ rates.from_v }} → {{ rates.to_v }} · '
    '<span class="d">{{ rates.change }}</span></div>', "rates move")
sub('<div class="win">22 May → 24 Aug 2026</div>',
    '<div class="win">{{ rates.window }}</div>', "rates window")
sub('<div class="move">$6.713tn → $6.746tn · <span class="d">+0.48%</span></div>',
    '<div class="move">{{ liq.from_v }} → {{ liq.to_v }} · '
    '<span class="d">{{ liq.change }}</span></div>', "liq move")
sub('<div class="win">20 May → 19 Aug 2026</div>',
    '<div class="win">{{ liq.window }}</div>', "liq window")
sub('<dt>Risk budget</dt><dd>55 / 45</dd>',
    '<dt>Risk budget</dt><dd>{{ budget_pct }} / {{ defensive_pct }}</dd>', "param budget")
sub('<dt>Tilt applied</dt><dd>−6.7pp</dd>',
    '<dt>Tilt applied</dt><dd>{{ tilt_signed }}pp</dd>', "param tilt")

# ── dwell footnote ────────────────────────────────────────────────
sub('''      <b>A state must persist four weeks before it is recognised, which is why a
      current reading of +18bp continues to register as rising.</b> The move is
      inside the ±25bp band, but the previous state stands until the new one
      holds — a rule fixed in advance to stop the book flipping on noise.''',
    '{{ dwell_note }}', "dwell note")

# ── section 3 ─────────────────────────────────────────────────────
sub('''      <b>The strategy adds +0.139 of a Sharpe point</b> over holding the same six
      assets in fixed equal weights — 0.812 against 0.672, on a shallower
      drawdown. A block bootstrap puts the 95% interval at +0.046 to +0.235,
      p = 0.004. It holds in all three periods.''',
    '''      <b>The strategy adds {{ edge_full }} of a Sharpe point</b> over holding the
      same six assets in fixed equal weights — {{ sh_strat }} against
      {{ sh_bench }}, on a shallower drawdown. A block bootstrap puts the 95%
      interval at {{ boot_ci }}, p = {{ boot_p }}. It holds in all three
      periods.''', "verdict")
sub('''            <tr><td>Full · 2002–2026</td><td>0.812</td><td>0.672</td><td class="gain">+0.139</td></tr>
            <tr><td>Post-crisis · from 2009</td><td>1.025</td><td>0.855</td><td class="gain">+0.170</td></tr>
            <tr><td>Last decade · from 2016</td><td>0.924</td><td>0.818</td><td class="gain">+0.106</td></tr>''',
    '''            {% for p in periods %}<tr><td>{{ p.period }}</td><td>{{ p.strategy }}</td>
              <td>{{ p.benchmark }}</td><td class="gain">{{ p.gain }}</td></tr>
            {% endfor %}''', "period table")
sub('''          largest edge, +0.170 against +0.139 — so the crisis isn't what's
          driving it.''',
    '''          largest edge, {{ edge_post09 }} against {{ edge_full }} — so the crisis
          isn't what's driving it.''', "period note")
sub('''            <tr><td>70%</td><td>8.00%</td><td>10.42%</td><td>0.739</td><td>−30.4%</td></tr>
            <tr><td>60%</td><td>7.31%</td><td>8.87%</td><td>0.795</td><td>−26.0%</td></tr>
            <tr class="pick"><td>55%</td><td>6.89%</td><td>8.21%</td><td>0.812</td><td>−24.0%</td></tr>
            <tr><td>50%</td><td>6.71%</td><td>7.43%</td><td>0.875</td><td>−22.0%</td></tr>''',
    '''            {% for r in policy %}<tr{{ ' class="pick"' if r.pick else '' }}>
              <td>{{ r.budget }}</td><td>{{ r.ann_return }}</td><td>{{ r.ann_vol }}</td>
              <td>{{ r.sharpe }}</td><td>{{ r.max_dd }}</td></tr>
            {% endfor %}''', "policy table")
sub('''      The benchmark is the <b>same six assets held in fixed equal weights</b> —
      13.75% in each risk sleeve, 22.5% in each Treasury. Bought once, never
      traded.''',
    '''      The benchmark is the <b>same six assets held in fixed equal weights</b> —
      {{ bench_eq }} in each risk sleeve, {{ bench_bond }} in each Treasury.
      Bought once, never traded.''', "benchmark desc")
sub('<div class="date">July 2002 – August 2026 · monthly rebalance · transaction costs applied</div>',
    '<div class="date">{{ sample_range }} · monthly rebalance · transaction costs applied</div>',
    "s3 date")

# ── section 4 tests table ─────────────────────────────────────────
sub('''          <td>Gold and energy added</td>
          <td><span class="tag k">Kept</span>Sharpe 0.812 from 0.744, worst loss
            −24.0% from −29.9%, turnover flat.</td>''',
    '''          <td>{{ tests[0].test }}</td>
          <td><span class="tag k">Kept</span>{{ tests[0].note }}</td>''', "test row 0")
sub('Rebuilt every weekday at 06:15 UTC. Page last built 26 Aug 2026, 06:17.',
    'Rebuilt every weekday at 06:15 UTC. Page last built {{ built_utc }}.', "build stamp")

# ── the chart series ──────────────────────────────────────────────
sub("  const START=new Date(2002,6,31), N=290, RF=0.021;",
    "  const SERIES = {{ chart_json }};\n"
    "  const N = SERIES.dates.length, RF = {{ rf_rate }};", "chart data")

DST.parent.mkdir(exist_ok=True)
DST.write_text(s)
print(f"wrote {DST}  ({len(s):,} bytes)")
print(f"placeholders: {s.count('{{')}")
