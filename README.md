# Macro Credit Risk Dashboard

A Python application that pulls macroeconomic data from the [FRED API](https://fred.stlouisfed.org/) and provides two interfaces for analysing U.S. lending risk:

- **`main.py`** — CLI analytics pipeline (terminal output)
- **`dashboard.py`** — Interactive Streamlit dashboard (browser UI)

---

## What It Does

The tool tracks five FRED series and derives a composite risk score that flags periods of elevated credit risk for lending businesses.

| FRED Series | Description | Frequency |
|---|---|---|
| `FEDFUNDS` | Federal Funds Rate | Monthly |
| `DGS10` | 10-Year Treasury Yield | Daily → monthly mean |
| `DRALACBN` | Consumer Loan Delinquency Rate | Quarterly → forward-filled |
| `PCE` | Personal Consumption Expenditures | Monthly |
| `CPIAUCSL` | CPI — All Urban Consumers | Monthly |
| `USREC` | NBER Recession Indicator *(dashboard only)* | Monthly |

### Analytics Computed

- **Yield curve spread**: 10Y Treasury − Fed Funds Rate
- **Month-over-month changes** for all series
- **Year-over-year changes** for Fed Funds, DGS10, delinquency, CPI, and PCE
- **Rolling 18-month Pearson correlations** between rate changes and delinquency changes at lags 0, 3, 6, and 12 months
- **Full-sample cross-correlogram** from −6 to +24 months (dashboard only)

### Risk Flags (7 total)

| Flag | Condition | Financial Rationale |
|---|---|---|
| `inverted_curve` | Spread < 0 pp | Historically precedes recessions by 12–18 months; compresses NIM |
| `near_inverted` | Spread < 0.5 pp | Warning zone before full inversion |
| `rising_delinquency` | Delinquency YoY > +10 bps | Early credit-cycle deterioration signal |
| `high_delinquency` | Delinquency > 1.80% | Elevated credit losses (cf. 2008–2010 crisis) |
| `high_inflation` | CPI YoY > 4.0% | Forces "higher-for-longer" rate policy |
| `rapid_rate_hike` | FEDFUNDS YoY > +200 bps | Rapid transmission to variable-rate borrowers |
| `high_rates` | FEDFUNDS > 5.0% | Absolute affordability pressure on borrowers |

A **risk score ≥ 3** marks a period as **elevated risk**.

---

## Quickstart

### Prerequisites

- Python 3.12–3.13
- [`uv`](https://github.com/astral-sh/uv) (recommended) or `pip`
- A [FRED API key](https://fred.stlouisfed.org/docs/api/api_key.html) (free)

### Installation

```bash
# Clone the repo
git clone https://github.com/rbownes/finance-demo.git
cd finance-demo

# Install dependencies with uv
uv sync
```

### Set Your API Key

```bash
export FRED_API_KEY="your_key_here"
```

Or create a `.env` file and load it with your preferred tool.

### Run the CLI Report

```bash
uv run python main.py
```

Outputs a terminal report covering:
- Latest indicator values
- Yield curve spread history
- Rolling correlation table
- Correlation summary by regime
- Elevated risk period episodes

### Launch the Dashboard

```bash
uv run streamlit run dashboard.py
```

Opens a browser dashboard with:
- KPI cards (live values + MoM delta)
- Yield curve & delinquency chart with NBER recession shading
- Risk score gauge
- Cross-correlogram (Fed Funds → Delinquency at all lags)
- Rolling 18-month correlation heatmap
- Natural language macro summary

---

## Running Tests

```bash
# Install dev dependencies
uv sync --group dev

# Run all tests
uv run pytest tests/ -v

# Run with coverage report
uv run pytest tests/ --cov=main --cov=dashboard --cov-report=term-missing
```

### Test Structure

```
tests/
├── __init__.py
├── test_main.py       # Unit tests for the analytics pipeline
└── test_dashboard.py  # Unit tests for dashboard helper functions
```

Tests cover:
- Data alignment and resampling logic
- Analytics calculations (spread, YoY diffs, rolling correlations)
- Risk flag triggering and score aggregation
- Dashboard regime classification and recession shape generation

#### Financial Edge Cases Tested

| Scenario | Test Coverage |
|---|---|
| **Zero Lower Bound** (FEDFUNDS = 0%) | No false `high_rates` flag; positive spread with long rates |
| **Yield curve inversion** | `inverted_curve` flag fires; deeply inverted triggers both inversion flags |
| **Spread exactly at boundary** (0.0 pp) | `inverted_curve` does NOT fire; `near_inverted` does |
| **Rapid rate-hike cycle** (+500 bps over 12 months) | `rapid_rate_hike` and `high_rates` flags fire |
| **Financial crisis** (high delinquency) | `high_delinquency` flag fires above 1.80% |
| **Rising delinquency** | `rising_delinquency` fires when YoY > 10 bps |
| **COVID forbearance** (delinquency ≈ 0.3%) | No delinquency flags trigger |
| **High inflation** (CPI YoY > 4%) | `high_inflation` flag fires |
| **All flags simultaneous** | Risk score reaches ≥ 5 in worst-case scenario |
| **NaN values in FRED data** | `flag_risk` handles missing data without crashing |
| **Quarterly DRALACBN forward-fill** | No NaN gaps after monthly alignment |
| **Recession at series end** | `recession_shapes` closes rectangle correctly |

---

## Project Structure

```
finance-demo/
├── main.py              # CLI analytics pipeline
├── dashboard.py         # Streamlit dashboard
├── pyproject.toml       # Project metadata and dependencies
├── tests/
│   ├── __init__.py
│   ├── test_main.py
│   └── test_dashboard.py
└── README.md
```

---

## Known Financial Reporting Edge Cases

The following scenarios can affect the accuracy of reports and should be monitored:

1. **FRED data revisions** — FRED occasionally revises historical series. Re-runs on the same date may produce different results.
2. **DRALACBN publication lag** — The delinquency rate is published quarterly with a ~45-day lag. The most recent 1–2 quarters may not yet be available, causing the forward-fill to repeat the last known value.
3. **Regulatory forbearance** — Government programmes (e.g. COVID-era student loan or mortgage forbearance) can suppress reported delinquency rates below economically true distress levels, causing the risk model to under-flag credit stress.
4. **DGS10 and weekend/holiday gaps** — The 10-year yield is a daily series with no weekend observations. Monthly mean aggregation smooths this but can differ from month-end spot readings used elsewhere.
5. **Zero Lower Bound arithmetic** — With FEDFUNDS near 0%, percentage-change calculations become numerically unstable. The pipeline uses basis-point (absolute) differences for `FEDFUNDS_yoy`, avoiding division-by-zero issues.
6. **Yield curve normalisation after inversion** — A rapid re-steepening after inversion (e.g. bull steepener ahead of recession) can cause the `near_inverted` flag to drop out just as credit risk is actually increasing.
7. **Risk score double-counting** — `inverted_curve` and `near_inverted` can both fire simultaneously (inverted implies near-flat), potentially inflating the score by 2 for a single underlying condition.

---

## Data Sources

All data via the [Federal Reserve Bank of St. Louis (FRED)](https://fred.stlouisfed.org/). Series identifiers: `FEDFUNDS`, `DGS10`, `DRALACBN`, `PCE`, `CPIAUCSL`, `USREC`.
