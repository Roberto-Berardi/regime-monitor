"""
Preview image built from the real data.

A screenshot of the page fails at preview size: LinkedIn renders a 1200px
image at roughly 400px wide, so axis labels and table text turn to mush, and
there is no name on it.

This draws the actual strategy and benchmark curves from the artifacts, thick
enough to read at a quarter size, with the name, the claim and three numbers
that stay legible when the image is small.
"""
from pathlib import Path
import json

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
ART = ROOT / "data" / "rotation"

W, H = 1200, 630
NAVY = (11, 34, 47)
BLUE, ORANGE, GREY = (31, 119, 180), (255, 127, 14), (127, 127, 127)
WHITE, MUTED, TEAL = (255, 255, 255), (150, 172, 184), (79, 191, 182)


def font(size, bold=False):
    for p in ["/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold
              else "/System/Library/Fonts/Supplemental/Arial.ttf",
              "/System/Library/Fonts/Helvetica.ttc",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold
              else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]:
        if Path(p).exists():
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                pass
    return ImageFont.load_default()


# ── data ──────────────────────────────────────────────────────────
lr = pd.read_parquet(ART / "returns_daily.parquet").dropna()
m = json.loads((ART / "metadata.json").read_text())
h, b = m["headline"], m["benchmark"]

# weekly, to keep the line smooth at small sizes
wk = lr.resample("W").sum()
strat = np.exp(wk["strategy"].cumsum())
bench = np.exp(wk["benchmark"].cumsum())

img = Image.new("RGB", (W, H), NAVY)
d = ImageDraw.Draw(img)
for y in range(H):
    t = y / H
    d.line([(0, y), (W, y)], fill=(int(11 + 8 * t), int(34 + 26 * t), int(47 + 30 * t)))

# ── the chart, right two-thirds ───────────────────────────────────
X0, X1, Y0, Y1 = 470, 1140, 150, 470
lo = min(strat.min(), bench.min()) * 0.98
hi = max(strat.max(), bench.max()) * 1.02
xs = lambda i: X0 + (X1 - X0) * i / (len(strat) - 1)
ys = lambda v: Y1 - (Y1 - Y0) * (np.log(v) - np.log(lo)) / (np.log(hi) - np.log(lo))

for v in [1, 2, 4]:
    if lo <= v <= hi:
        y = ys(v)
        d.line([(X0, y), (X1, y)], fill=(38, 66, 84), width=1)

for series, col, wdt in ((bench, ORANGE, 4), (strat, BLUE, 5)):
    pts = [(xs(i), ys(v)) for i, v in enumerate(series)]
    d.line(pts, fill=col, width=wdt, joint="curve")

d.line([(X0, 118), (X0 + 46, 118)], fill=BLUE, width=5)
d.text((X0 + 58, 106), "Strategy", font=font(22, True), fill=WHITE)
d.line([(X0 + 190, 118), (X0 + 236, 118)], fill=ORANGE, width=4)
d.text((X0 + 248, 106), "Equal weight", font=font(22), fill=MUTED)

d.text((X0, Y1 + 16), "2006", font=font(19), fill=MUTED)
d.text((X1 - 46, Y1 + 16), "2026", font=font(19), fill=MUTED)

# ── the words, left third ─────────────────────────────────────────
d.text((70, 96), "Roberto Berardi", font=font(48, True), fill=WHITE)
d.text((70, 156), "MSc Finance, HEC Lausanne", font=font(22), fill=TEAL)

d.text((70, 232), "A multi-asset book", font=font(33), fill=WHITE)
d.text((70, 274), "positioned by Fed rates", font=font(33), fill=WHITE)
d.text((70, 316), "and liquidity.", font=font(33), fill=WHITE)

d.text((70, 382), "Rebuilt from live data every", font=font(21), fill=MUTED)
d.text((70, 410), "weekday. Not a backtest.", font=font(21), fill=MUTED)

# ── three numbers along the bottom ────────────────────────────────
d.line([(70, 512), (1140, 512)], fill=(38, 66, 84), width=2)
cols = [
    ("SHARPE", f"{h['sharpe']:.2f}", f"vs {b['sharpe']:.2f}"),
    ("MAX DRAWDOWN", f"{h['max_dd']*100:.1f}%", f"vs {b['max_dd']*100:.1f}%"),
    ("ANNUALISED", f"{h['ann_return']*100:.2f}%", f"vs {b['ann_return']*100:.2f}%"),
]
for i, (k, v, sub) in enumerate(cols):
    x = 70 + i * 360
    d.text((x, 536), k, font=font(16, True), fill=MUTED)
    d.text((x, 560), v, font=font(38, True), fill=WHITE)
    d.text((x + 8 + d.textlength(v, font=font(38, True)), 578), sub,
           font=font(19), fill=MUTED)

img.save(ROOT / "docs" / "preview.png", "PNG", optimize=True)
print(f"  preview.png  {(ROOT/'docs'/'preview.png').stat().st_size/1024:.0f}KB")
print(f"  curve: {len(strat)} weekly points, "
      f"{strat.index[0].date()} to {strat.index[-1].date()}")
print("\n  open docs/preview.png to check it")
