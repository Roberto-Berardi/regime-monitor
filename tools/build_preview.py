"""
Build the link preview.

Draws the approved card at 1200x630 with Pillow, adds the Open Graph and
Twitter tags LinkedIn was missing, and makes a favicon from the headshot.

Nothing on the card dates, so this runs once. It is not wired into the daily
build on purpose: LinkedIn caches the scrape and will not re-fetch, so a card
carrying today's numbers would freeze at whatever it said when it was scraped.
"""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
URL = "https://roberto-berardi.github.io/regime-monitor/"

W, H = 1200, 630
TEAL, MUTED, LINE, WHITE = (79, 191, 182), (150, 172, 184), (38, 66, 79), (255, 255, 255)


def font(size, bold=False, serif=False):
    names = (["Georgia Bold.ttf", "Georgia.ttf"] if serif else
             ["Arial Bold.ttf", "Arial.ttf"])
    name = names[0] if bold else names[-1]
    for p in [f"/System/Library/Fonts/Supplemental/{name}",
              f"/Library/Fonts/{name}",
              "/System/Library/Fonts/Helvetica.ttc",
              "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf" if serif else
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]:
        if Path(p).exists():
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                pass
    return ImageFont.load_default()


# ── background: the same gradient and washes as the page hero ─────
img = Image.new("RGB", (W, H))
d = ImageDraw.Draw(img)
for y in range(H):
    for_x = y / H
    d.line([(0, y), (W, y)],
           fill=(int(11 + 9 * for_x), int(34 + 33 * for_x), int(47 + 43 * for_x)))

glow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
g = ImageDraw.Draw(glow)
for r, (cx, cy, col) in [(1, (1120, 40, (15, 163, 150))), (1, (60, 620, (109, 74, 224)))]:
    for i in range(70, 0, -1):
        rad = i * 8
        a = int(26 * (i / 70) ** 2.4)
        g.ellipse([cx - rad, cy - rad, cx + rad, cy + rad], fill=col + (a,))
img = Image.alpha_composite(img.convert("RGBA"), glow).convert("RGB")
d = ImageDraw.Draw(img)

PAD = 70

# ── header ────────────────────────────────────────────────────────
d.text((PAD, 64), "Roberto Berardi", font=font(46, bold=True), fill=WHITE)
d.text((PAD, 122),
       "MSc Finance, HEC Lausanne  ·  Asset and Risk Management",
       font=font(21, bold=True), fill=TEAL)

# live marker, right aligned
f_live_b, f_live = font(17, bold=True), font(15)
txt = "LIVE"
tw = d.textlength(txt, font=f_live_b)
x_right = W - PAD
d.ellipse([x_right - tw - 26, 72, x_right - tw - 14, 84], fill=TEAL)
d.text((x_right - tw, 68), txt, font=f_live_b, fill=WHITE)
sub = "rebuilt every weekday"
d.text((x_right - d.textlength(sub, font=f_live), 96), sub, font=f_live, fill=MUTED)

# ── the claim ─────────────────────────────────────────────────────
f_claim = font(45, serif=True)
d.text((PAD, 262), "A multi-asset book", font=f_claim, fill=WHITE)
d.text((PAD, 318), "positioned by", font=f_claim, fill=WHITE)
w_pos = d.textlength("positioned by ", font=f_claim)
d.text((PAD + w_pos, 318), "Fed rates", font=f_claim, fill=TEAL)
d.text((PAD, 374), "and liquidity.", font=f_claim, fill=TEAL)

# ── the rail ──────────────────────────────────────────────────────
RAIL_Y = 500
d.line([(PAD, RAIL_Y), (W - PAD, RAIL_Y)], fill=LINE, width=1)

cells = [("POSITIONED BY", "Equity duration"),
         ("READS", "2y yield · Fed balance sheet"),
         ("SINCE", "February 2006")]
col_w = (W - 2 * PAD) / 3
f_k, f_v = font(12, bold=True), font(22, bold=True)
for i, (k, v) in enumerate(cells):
    x = PAD + i * col_w
    if i:
        d.line([(x - 26, RAIL_Y + 22), (x - 26, RAIL_Y + 92)], fill=LINE, width=1)
    d.text((x, RAIL_Y + 26), " ".join(k), font=f_k, fill=MUTED)
    d.text((x, RAIL_Y + 54), v, font=f_v, fill=WHITE)

out = ROOT / "docs" / "preview.png"
img.save(out, "PNG", optimize=True)
print(f"  preview.png  {W}x{H}, {out.stat().st_size/1024:.0f}KB")

# ── favicon ───────────────────────────────────────────────────────
photo = ROOT / "docs" / "photo.jpg"
if photo.exists():
    ic = Image.open(photo).convert("RGB")
    ic.thumbnail((64, 64))
    ic.save(ROOT / "docs" / "favicon.png", "PNG", optimize=True)
    print("  favicon.png from the headshot")

# ── meta tags ─────────────────────────────────────────────────────
p = ROOT / "templates" / "index.html.j2"
s = p.read_text()
if "og:image" in s:
    print("  meta tags already present, skipped")
else:
    TAGS = f'''<meta name="description" content="A multi-asset book positioned by Fed rates and liquidity. Rebuilt from live data every weekday, showing the position it holds today.">
<meta name="author" content="Roberto Berardi">
<meta property="og:type" content="website">
<meta property="og:url" content="{URL}">
<meta property="og:title" content="Rate-Regime Multi-Asset Book · Roberto Berardi">
<meta property="og:description" content="Six ETFs positioned by Fed rates and liquidity. Rebuilt from live data every weekday. This is the position it holds today, not a backtest.">
<meta property="og:image" content="{URL}preview.png">
<meta property="og:image:width" content="1200">
<meta property="og:image:height" content="630">
<meta property="og:site_name" content="Roberto Berardi">
<meta name="twitter:card" content="summary_large_image">
<meta name="twitter:title" content="Rate-Regime Multi-Asset Book">
<meta name="twitter:description" content="Six ETFs positioned by Fed rates and liquidity. Rebuilt from live data every weekday.">
<meta name="twitter:image" content="{URL}preview.png">
<link rel="icon" type="image/png" href="favicon.png">
<title>'''
    assert "<title>" in s
    p.write_text(s.replace("<title>", TAGS, 1))
    print("  meta tags added")

print("\n  open docs/preview.png to check it")
print("  then:  python scripts/build_site.py")
