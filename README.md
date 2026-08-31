# Rate-Regime Factor Rotation

A systematic multi-asset portfolio positioned by monetary conditions, rebuilt
from live data every weekday.

**Live page — https://roberto-berardi.github.io/regime-monitor/**

---

## The idea

Interest rates set the discount rate on future cash flows. Companies whose
profits arrive years from now behave like long-duration bonds; companies
earning cash today behave like short-duration bonds. That sensitivity is called
**equity duration**, and it is the single axis this portfolio is built on.

2022 is the clearest illustration: the Fed raised rates 425bp, growth equity
fell 29.8%, value equity fell 8.1%. Same market, same year, 21.7 percentage
points apart.

## The signal

Two observable inputs, both published by FRED, neither revised.

| Input | Series | Window | Threshold | Weight |
|---|---|---|---|---|
| Policy rates | DGS2 | 3 months | ±25bp | ±2 |
| Liquidity | WALCL | 13 weeks | ±1% | ±1 |

Rates count double: the discount-rate effect is arithmetic, while the liquidity
effect depends on where investors choose to put the cash. The two sum to a
score from −3 to +3. A state must persist four weeks before it is recognised.

Every threshold was fixed in advance and justified by an institutional fact
rather than a backtest result — 25bp is one policy increment, three months is
one FOMC cycle, four weeks is one rebalance period.

## The portfolio

| Bucket | Holdings | Favoured when |
|---|---|---|
| Long duration | IWF growth equity, GLD gold | score > 0 |
| Short duration | IWD value equity, XLE energy | score < 0 |
| Defensive | IEF 7–10y **or** SHY 1–3y | always 45% |

**55% risk assets / 45% defensive**, a stated policy rather than an optimiser
output. Within each duration bucket, capital is split inversely to volatility.
The score moves up to 10 percentage points between the two buckets — a transfer,
not a split. The defensive sleeve switches on the rate signal: short duration
while rates rise, long when they fall, because duration is a hedge in a growth
shock and a liability in a rate shock.

Monthly rebalance, 2pp no-trade band, 5bp one-way transaction costs.

## Results

February 2006 – August 2026. Sharpe is excess of the 3-month Treasury
bill. Figures as at August 2026 — the live page is always current.

| | Strategy | Benchmark | SPY |
|---|---|---|---|
| Annualised return | 6.89% | 6.20% | 10.76% |
| Volatility | 8.21% | 8.96% | 18.92% |
| Sharpe | **0.624** | 0.495 | 0.475 |
| Maximum drawdown | −24.0% | −25.2% | −55.2% |

The benchmark is the same six assets held in fixed equal weights, bought once
and never traded. SPY is shown for scale, not as a benchmark.

**The edge is +0.129 Sharpe**, 95% bootstrap interval +0.045 to +0.234,
p = 0.004. It holds in all three sub-periods tested and is largest in the window
that excludes 2008.

### What each part contributes

| | Sharpe | Adds |
|---|---|---|
| Equal weight, both bonds | 0.495 | — |
| + volatility weighting | 0.533 | +0.037 |
| + switching bond sleeve | 0.593 | +0.060 |
| + regime tilt | **0.624** | +0.031 |

The tilt is the smallest of the three contributions.

## What was tested and rejected

Twenty variations, judged against three conditions fixed before each ran:
improve Sharpe by more than 0.05, hold in all three periods, raise turnover by
less than half. **Nineteen failed.**

Rejected: 200-day trend filter · PCA correlation early warning · scaling bond
duration by score · momentum tilt · credit-spread override · volatility
targeting · GJR-GARCH · turbulence · VIX level · VIX term structure ·
yield-curve veto · Taylor rule gap · hierarchical risk parity ·
regime-conditional covariance · regime-conditional risk budget · split
defensive sleeve · safe-haven currencies · dollar index · European equity ·
long Treasuries.

Two are worth reading about on the live page. The 200-day trend filter produced
the best headline of the twenty and lost almost all of it when 2008 was removed.
The correlation early-warning signal fired a year before the financial crisis
and three months before Covid, and still could not be traded — it sees
fragility, not timing.

## Limitations

- Growth and value correlate 0.87. This picks a side of one axis; it is not
  diversification.
- Rates fall when policy eases and when investors panic. The model scores both
  alike, so 2008 went the wrong way.
- The model reads conditions that already exist. It forecasts nothing.

## Running it

```bash
conda create -n regime python=3.11 && conda activate regime
pip install -r requirements.txt
export FRED_API_KEY=your_key          # free from fred.stlouisfed.org
python precompute_rotation.py         # ~20s, writes data/rotation/
python scripts/check_rotation.py      # 41 validation checks
python scripts/build_site.py          # renders docs/index.html
```

Serve the page over HTTP rather than opening the file directly — Chrome blocks
the chart under `file://`.

```bash
cd docs && python -m http.server 8000
```

## How it stays live

A GitHub Actions workflow runs every weekday at 06:15 UTC: pull data, run the
pipeline, validate, rebuild the page, commit. The validation gate checks
structure *and* plausibility — that headline metrics fall inside sensible
bounds, and that the maximum drawdown has not moved more than 5 percentage
points since the previous build. A failure stops the workflow, so the last good
page stays up and the date on the page stops moving.

## Repository

```
precompute_rotation.py     the pipeline
src/rate_regime.py         the threshold rule
src/rotation.py            the backtest
scripts/check_rotation.py  the validation gate
scripts/build_site.py      renders the page from the artifacts
templates/                 the page template
data/rotation/             committed artifacts
research/                  the twenty tests
archive/                   see below
```

### Earlier work

This project began as a nine-asset equal-risk-contribution portfolio. Testing
it properly showed that ERC concentrated the book in whichever asset happened to
be calmest — at one point half the portfolio sat in a single Treasury proxy —
and that the construction suited the method rather than the question. That code
is preserved under `archive/` and no longer runs.

The econometric techniques used here — GARCH volatility modelling, dynamic
conditional correlation, Markov regime estimation — were studied during the MSc
in Finance at HEC Lausanne and applied to this project independently.

## Data

Yahoo Finance for prices, FRED (Federal Reserve Bank of St. Louis) for rates and
the Federal Reserve balance sheet.

## Disclaimer

Research and demonstration purposes. Not investment advice, and not an offer or
recommendation to buy or sell any security.
