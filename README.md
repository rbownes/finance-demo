# Macro Credit Risk Dashboard

A Python toolkit for analysing U.S. macro-credit conditions using live data from the [FRED API](https://fred.stlouisfed.org/).
It tracks five key economic series, computes rolling correlations between interest-rate changes and consumer loan delinquencies, and flags periods of elevated lending risk.

---

## What it does

| Module | Purpose |
|--------|---------|
| `main.py` | CLI analytics engine — fetches FRED data, computes analytics, prints risk reports to the terminal |
| `dashboard.py` | Interactive Streamlit web dashboard with charts, a risk-score gauge, and a natural-language macro summary |

### Key indicators tracked

| FRED Series | Metric |
|-------------|--------|
| `FEDFUNDS` | Federal Funds Rate |
| `DGS10` | 10-Year Treasury Yield |
| `DRALACBN` | Consumer Loan Delinquency Rate |
| `PCE` | Personal Consumption Expenditures |
| `CPIAUCSL` | CPI (All Urban Consumers) |
| `USREC` | NBER Recession Indicator (dashboard only) |

### Analytics computed

- **Yield curve spread** (10Y Treasury − Fed Funds) — inversion signals credit stress
- **Year-over-year and month-over-month changes** for all series
- **Rolling 18-month correlations** between Fed Funds rate changes and delinquency changes at lags 0 – 24 months
- **Risk score (0 – 7)** based on seven threshold conditions (see table below)

### Risk flags

| Flag | Condition |
|------|-----------|
| `inverted_curve` | Yield spread < 0 pp |
| `near_inverted` | Yield spread < 0.5 pp |
| `rising_delinquency` | Delinquency YoY change > +10 bps |
| `high_delinquency` | Delinquency rate > 1.80 % |
| `high_inflation` | CPI YoY > 4.0 % |
| `rapid_rate_hike` | Fed Funds YoY change > +200 bps |
| `high_rates` | Fed Funds rate > 5.0 % |

A **risk score ≥ 3** triggers the `elevated_risk` designation.

---

## Getting started

### Prerequisites

- Python 3.12 or 3.13
- [uv](https://docs.astral.sh/uv/) (recommended) **or** pip
- A free [FRED API key](https://fred.stlouisfed.org/docs/api/api_key.html)

### Installation

```bash
# Clone the repo
git clone https://github.com/rbownes/finance-demo.git
cd finance-demo

# Install dependencies with uv (recommended)
uv sync

# Or with pip
pip install -e .
```

### Set your FRED API key

```bash
export FRED_API_KEY="your_key_here"
```

You can obtain a free key at <https://fred.stlouisfed.org/docs/api/api_key.html>.

---

## Usage

### CLI report (`main.py`)

Prints a full terminal report including latest indicator values, yield curve history, rolling correlations, and elevated-risk episodes since 1990.

```bash
uv run python main.py
# or
python main.py
```

**Sample output sections:**

- `LATEST VALUES` — most recent reading for each series
- `YIELD CURVE SPREAD` — annual snapshots flagging inversions
- `ROLLING 18-MONTH CORRELATIONS` — rate → delinquency lead/lag analysis
- `CORRELATION SUMMARY` — full-sample Pearson correlations at multiple lags
- `ELEVATED RISK PERIODS` — contiguous episodes with risk score ≥ 3

### Interactive dashboard (`dashboard.py`)

Launches a Streamlit web app with interactive Plotly charts.

```bash
uv run streamlit run dashboard.py
# or
streamlit run dashboard.py
```

Open <http://localhost:8501> in your browser.

**Dashboard panels:**

1. **KPI row** — live values for all key metrics with month-over-month deltas
2. **Risk Regime badge** — colour-coded regime (Normal → Caution → Elevated → High Risk)
3. **Yield Curve & Delinquency chart** — time series with NBER recession shading
4. **Risk Score Gauge** — 0–7 dial
5. **Fed Funds → Delinquency Cross-Correlogram** — full-sample Pearson r at each lag
6. **Rolling 18-Month Correlation Heatmap** — lag × time matrix showing how the transmission lag has shifted over cycles
7. **Natural Language Summary** — plain-English macro interpretation

---

## Running the tests

The test suite uses [pytest](https://docs.pytest.org/) and covers boundary conditions in the financial modelling layer.

```bash
# Install dev dependencies
uv sync --extra dev
# or: pip install pytest

# Run all tests
uv run pytest tests/ -v

# Run with coverage
uv run pytest tests/ -v --tb=short
```

### What is tested

| Test class | Coverage |
|------------|---------|
| `TestFetchSeries` | API responses, FRED `.` sentinel → NaN, HTTP errors, negative rates |
| `TestAlignToMonthly` | Monthly alignment, quarterly forward-fill, NaN dropna behaviour |
| `TestComputeAnalytics` | Spread arithmetic, zero/negative base in pct_change, diff boundary NaNs, rolling window requirements |
| `TestFlagRisk` | All seven threshold boundaries (at, above, and below), score 0 vs 7, `elevated_risk` boundary at 3, NaN handling, missing columns |
| `TestClassifyRegime` | All four regime buckets (scores 0-1, 2, 3-4, 5-7) |
| `TestRecessionShapes` | Single, multiple, and trailing recession episodes; no-recession base case |

---

## Project structure

```
finance-demo/
├── main.py           # CLI analytics engine
├── dashboard.py      # Streamlit web dashboard
├── tests/
│   └── test_main.py  # pytest test suite
├── pyproject.toml    # project metadata & dependencies
├── uv.lock           # locked dependency tree
└── .python-version   # pinned Python version (3.12)
```

---

## Environment variables

| Variable | Required | Description |
|----------|----------|-------------|
| `FRED_API_KEY` | Yes | Your FRED API key |

---

## Data sources

All data is sourced from the [Federal Reserve Bank of St. Louis FRED API](https://fred.stlouisfed.org/).
Historical data runs from **January 1990** to the present.
