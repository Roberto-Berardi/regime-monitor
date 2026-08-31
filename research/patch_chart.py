"""Adds: SPY reference toggle, hover crosshair + tooltip, ERC-anchor note."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
bs = ROOT / "scripts" / "build_site.py"
tpl = ROOT / "templates" / "index.html.j2"

# ═══════════════════════════════════════════ build_site.py ═══
s = bs.read_text()

# 1. make_range takes an spy series and emits its line + hover points
old = "def make_range(eq, key, label, years):"
new = "def make_range(eq, spy, key, label, years):"
assert old in s, "make_range signature"
s = s.replace(old, new, 1)

old = """    strat = w["Tilted"] / w["Tilted"].iloc[0] * 100
    bench = w["EW"] / w["EW"].iloc[0] * 100

    lo = min(strat.min(), bench.min())
    hi = max(strat.max(), bench.max())"""
new = """    strat = w["Tilted"] / w["Tilted"].iloc[0] * 100
    bench = w["EW"] / w["EW"].iloc[0] * 100
    sp = spy.reindex(w.index).ffill()
    sp = sp / sp.iloc[0] * 100

    lo = min(strat.min(), bench.min(), sp.min())
    hi = max(strat.max(), bench.max(), sp.max())"""
assert old in s, "normalisation block"
s = s.replace(old, new, 1)

old = "    sy, by = ymap(strat), ymap(bench)"
new = "    sy, by, py = ymap(strat), ymap(bench), ymap(sp)"
assert old in s, "ymap call"
s = s.replace(old, new, 1)

old = """        "strat_line": _pts(xs, sy), "bench_line": _pts(xs, by),"""
new = """        "strat_line": _pts(xs, sy), "bench_line": _pts(xs, by),
        "spy_line": _pts(xs, py),
        "hover": json.dumps([
            {"x": round(x, 1), "d": dt.strftime("%-d %b %Y"),
             "s": round(sv, 1), "b": round(bv, 1), "p": round(pv, 1),
             "sy": round(syy, 1), "by": round(byy, 1), "py": round(pyy, 1)}
            for x, dt, sv, bv, pv, syy, byy, pyy
            in zip(xs, w.index, strat, bench, sp, sy, by, py)
        ]),"""
assert old in s, "return dict lines"
s = s.replace(old, new, 1)

# 2. build the SPY curve and pass it through
old = '''    eq = pd.read_parquet(PRE / "equity_curves.parquet")[["Tilted", "EW"]]'''
new = '''    eq = pd.read_parquet(PRE / "equity_curves.parquet")[["Tilted", "EW"]]

    # SPY reference: cumulative growth from daily log returns, sampled weekly
    spy_daily = pd.read_parquet(PRE / "returns_daily.parquet")["SP500"]
    spy = np.exp(spy_daily.cumsum())'''
assert old in s, "equity load"
s = s.replace(old, new, 1)

old = """        "ranges": [r for r in [
            make_range(eq, "1y", "1Y", 1), make_range(eq, "3y", "3Y", 3),
            make_range(eq, "5y", "5Y", 5), make_range(eq, "full", "Full", None),
        ] if r],"""
new = """        "ranges": [r for r in [
            make_range(eq, spy, "1y", "1Y", 1), make_range(eq, spy, "3y", "3Y", 3),
            make_range(eq, spy, "5y", "5Y", 5), make_range(eq, spy, "full", "Full", None),
        ] if r],"""
assert old in s, "ranges list"
s = s.replace(old, new, 1)

if "import numpy as np" not in s:
    s = s.replace("import pandas as pd", "import numpy as np\nimport pandas as pd", 1)

bs.write_text(s)
import ast; ast.parse(s)
print("build_site.py patched")

# ═══════════════════════════════════════════════ template ═══
t = tpl.read_text()

old = """        <div class="legend">
          <span><i style="background:#0FA396"></i>Strategy</span>
          <span><i style="background:#8A99A2"></i>Doing nothing</span>
        </div>"""
new = """        <div class="legend">
          <span><i style="background:#0FA396"></i>Strategy</span>
          <span><i style="background:#8A99A2"></i>Doing nothing</span>
          <button class="toggle" id="spyToggle" aria-pressed="false"><i></i> SPY reference</button>
        </div>"""
assert old in t, "legend block"
t = t.replace(old, new, 1)

old = """        <polyline fill="none" stroke="#0FA396" stroke-width="3.2" stroke-linejoin="round"
          points="{{ r.strat_line }}"/>"""
new = """        <polyline fill="none" stroke="#0FA396" stroke-width="3.2" stroke-linejoin="round"
          points="{{ r.strat_line }}"/>
        <polyline class="spy" fill="none" stroke="#2563EB" stroke-width="2"
          stroke-dasharray="5 4" stroke-linejoin="round" style="display:none"
          points="{{ r.spy_line }}"/>
        <g class="cross" style="display:none">
          <line class="cx" y1="30" y2="195" stroke="#8A99A2" stroke-width="1" stroke-dasharray="3 3"/>
          <circle class="cs" r="4.5" fill="#0FA396" stroke="#fff" stroke-width="2"/>
          <circle class="cb" r="4.5" fill="#8A99A2" stroke="#fff" stroke-width="2"/>
          <circle class="cp" r="4.5" fill="#2563EB" stroke="#fff" stroke-width="2" style="display:none"/>
        </g>
        <rect class="hit" x="40" y="30" width="1030" height="165" fill="transparent"
              data-points='{{ r.hover }}'/>"""
assert old in t, "strat polyline"
t = t.replace(old, new, 1)

# tooltip element + styles
old = """    <div class="chartcard">"""
new = """    <div class="chartcard" style="position:relative">
      <div id="chartTip" style="position:absolute;display:none;pointer-events:none;
        background:#fff;border:1px solid var(--line);border-radius:10px;padding:10px 13px;
        font-size:12px;box-shadow:0 10px 30px rgba(11,31,42,.15);z-index:20;white-space:nowrap"></div>"""
assert old in t, "chartcard open"
t = t.replace(old, new, 1)

old = """.legend i{display:inline-block;width:20px;height:3px;border-radius:2px;
  margin-right:8px;vertical-align:middle}"""
new = """.legend i{display:inline-block;width:20px;height:3px;border-radius:2px;
  margin-right:8px;vertical-align:middle}
.toggle{display:inline-flex;align-items:center;gap:7px;font-size:12.5px;color:var(--grey);
  background:none;border:1px solid var(--line);border-radius:999px;padding:5px 13px}
.toggle.on{color:var(--blue);border-color:var(--blue);background:var(--blue-soft)}
.toggle i{width:20px;height:3px;border-radius:2px;background:currentColor;display:inline-block}
.hit{cursor:crosshair}"""
assert old in t, "legend css"
t = t.replace(old, new, 1)

# JS: toggle + hover
old = """  document.querySelectorAll('.tip').forEach(function(tip){"""
new = """  var spyOn=false, tipEl=document.getElementById('chartTip');
  var spyBtn=document.getElementById('spyToggle');
  if(spyBtn){
    spyBtn.addEventListener('click',function(){
      spyOn=spyBtn.classList.toggle('on');
      spyBtn.setAttribute('aria-pressed',spyOn?'true':'false');
      document.querySelectorAll('polyline.spy').forEach(function(p){
        p.style.display = spyOn ? '' : 'none';
      });
      document.querySelectorAll('.cp').forEach(function(c){
        c.style.display = spyOn ? '' : 'none';
      });
    });
  }

  document.querySelectorAll('rect.hit').forEach(function(hit){
    var svg=hit.closest('svg'), pts=JSON.parse(hit.dataset.points);
    var g=svg.querySelector('.cross');
    var cx=svg.querySelector('.cx'), cs=svg.querySelector('.cs'),
        cb=svg.querySelector('.cb'), cp=svg.querySelector('.cp');
    function move(ev){
      var r=svg.getBoundingClientRect();
      var px=(ev.clientX-r.left)/r.width*1080;
      var best=pts[0], bd=1e9;
      for(var i=0;i<pts.length;i++){
        var dd=Math.abs(pts[i].x-px);
        if(dd<bd){bd=dd;best=pts[i];}
      }
      g.style.display='';
      cx.setAttribute('x1',best.x); cx.setAttribute('x2',best.x);
      cs.setAttribute('cx',best.x); cs.setAttribute('cy',best.sy);
      cb.setAttribute('cx',best.x); cb.setAttribute('cy',best.by);
      cp.setAttribute('cx',best.x); cp.setAttribute('cy',best.py);
      tipEl.innerHTML='<b>'+best.d+'</b><br>'+
        '<span style="color:#0FA396">&#9632;</span> Strategy '+best.s+
        '<br><span style="color:#8A99A2">&#9632;</span> Doing nothing '+best.b+
        (spyOn?'<br><span style="color:#2563EB">&#9632;</span> SPY '+best.p:'');
      var card=svg.closest('.chartcard').getBoundingClientRect();
      var left=r.left-card.left+best.x/1080*r.width+14;
      if(left>card.width-190) left-=210;
      tipEl.style.left=left+'px';
      tipEl.style.top=(r.top-card.top+30)+'px';
      tipEl.style.display='block';
    }
    hit.addEventListener('mousemove',move);
    hit.addEventListener('mouseleave',function(){
      g.style.display='none'; tipEl.style.display='none';
    });
  });

  document.querySelectorAll('.tip').forEach(function(tip){"""
assert old in t, "js anchor"
t = t.replace(old, new, 1)

# ERC-anchor clarification
old = """        A large allocation of capital is not the same as a large allocation of risk. Calm
        markets need far more money to carry the same weight in the portfolio's fortunes."""
new = """        A large allocation of capital is not the same as a large allocation of risk. Calm
        markets need far more money to carry the same weight in the portfolio's fortunes.
        These are the <b>strategic anchor</b> weights from layer 1, before the trend tilt is
        applied — which is why they differ slightly from the live allocation shown below."""
assert old in t, "money-is-not-risk sub"
t = t.replace(old, new, 1)

tpl.write_text(t)
print("template patched")
print("\nnow run:  python scripts/build_site.py && open docs/index.html")
