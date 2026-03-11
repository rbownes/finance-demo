# Macro Credit Risk Dashboard

A Python toolkit for analysing U.S. macroeconomic conditions and their impact on lending businesses. It pulls live data from the [FRED API](https://fred.stlouisfed.org/), computes key financial indicators, scores credit-risk regimes, and surfaces the results via an interactive Streamlit dashboard.

---

## Table of Contents

1. [Overview](#overview)
2. [Key Concepts](#key-concepts)
3. [Prerequisites](#prerequisites)
4. [Installation](#installation)
5. [Configuration](#configuration)
6. [Running the Tools](#running-the-tools)
7. [Project Structure](#project-structure)
8. [Risk Model Reference](#risk-model-reference)
9. [Running the Tests](#running-the-tests)

---

## Overview

The project has two entry points:

| File | Purpose |
|------|---------|
| `main.py` | Command-line analysis — fetches FRED data, computes analytics, prints risk report |
| `dashboard.py` | Interactive Streamlit web dashboard with charts and a natural-language summary |

Both tools track the same five FRED series and derive the same risk metrics; the dashboard simply adds visualisation.

---

## Key Concepts

### Tracked FRED Series

| Series ID | Name | Frequency |
|-----------|------|-----------|
| `FEDFUNDS` | Federal Funds Rate | Monthly |
| `DGS10` | 10-Year Treasury Yield | Daily |
| `DRALACBN` | Consumer Loan Delinquency Rate | Quarterly |
| `PCE` | Personal Consumption Expenditures | Monthly |
| `CPIAUCSL` | CPI — All Urban Consumers | Monthly |
| `USREC` | NBER Recession Indicator | Monthly (dashboard only) |

### Derived Metrics

- **Yield Curve Spread** — `DGS10 − FEDFUNDS`. Negative values indicate an inverted curve, which has historically preceded recessions by 12–18 months.
- **Year-over-Year (YoY) Changes** — 12-month differences for rates, delinquency, and price indices.
- **CPI / PCE YoY %** — `pct_change(12) × 100`.
- **Rolling 18-Month Correlations** — Pearson r between Fed Funds rate changes and delinquency changes, computed at lags of 0, 3, 6, and 12 months to capture the monetary transmission lag.

### Risk-Scoring Model

Seven binary flags are evaluated each month. When three or more are active the period is classified as **elevated risk**.

| Flag | Condition | Threshold |
|------|-----------|-----------|
| Inverted yield curve | `spread < 0.0` | strict `<` |
| Near-inverted curve | `spread < 0.5` | strict `<` |
| Rising delinquencies | `DRALACBN_yoy > 0.10` | +10 bps YoY |
| High delinquency | `DRALACBN > 1.80` | % |
| High inflation | `CPI_yoy_pct > 4.0` | % |
| Rapid rate hike | `FEDFUNDS_yoy > 2.0` | +200 bps YoY |
| High absolute rates | `FEDFUNDS > 5.0` | % |

> **Note on boundary values:** all comparisons use strict inequalities — a value exactly equal to a threshold does **not** trigger the flag.

---

## Prerequisites

- Python 3.12 or 3.13
- [uv](https://docs.astral.sh/uv/) package manager (recommended) **or** pip
- A free [FRED API key](https://fred.stlouisfed.org/docs/api/api_key.html)

---

## Installation

```bash
# Clone the repository
git clone https://github.com/rbownes/finance-demo.git
cd finance-demo

# Install runtime dependencies with uv
uv sync

# — or — with pip inside a virtual environment
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e .
```

---

## Configuration

The tools read your FRED API key from the environment:

```bash
export FRED_API_KEY="your_key_here"
```

For persistent configuration you can add this line to your shell profile (`~/.bashrc`, `~/.zshrc`, etc.) or place it in a `.env` file and use a tool such as [python-dotenv](https://pypi.org/project/python-dotenv/).

---

## Running the Tools

### Command-line analysis (`main.py`)

```bash
uv run python main.py
```

Prints four sections to the terminal:

1. **Latest values** — most recent reading for each series
2. **Yield curve history** — annual snapshots with inversion signal
3. **Rolling correlations** — 18-month window at various lags
4. **Elevated risk periods** — contiguous episodes where `risk_score ≥ 3`

### Interactive dashboard (`dashboard.py`)

```bash
uv run streamlit run dashboard.py
```

Then open the URL printed in the terminal (typically `http://localhost:8501`).

The dashboard includes:

- KPI cards (Fed Funds, 10Y yield, spread, delinquency, CPI YoY)
- Risk regime badge and gauge
- Yield curve & delinquency time series with NBER recession bands
- Cross-correlogram (rate-change lead/lag vs. delinquency)
- Rolling correlation heatmap
- Natural-language macro summary

---

## Project Structure

```
finance-demo/
├── main.py            # CLI analysis pipeline
├── dashboard.py       # Streamlit dashboard
├── pyproject.toml     # Project metadata and dependencies
├── tests/
│   ├── conftest.py    # Pytest fixtures (Streamlit mock)
│   ├── test_main.py   # Tests for main.py analytics/risk logic
│   └── test_dashboard.py  # Tests for dashboard.py pure functions
└── .claude/
    └── skills/        # Claude AI skill definitions
```

---

## Risk Model Reference

### `compute_analytics(df)`

Enriches a monthly-aligned DataFrame with:

- Month-over-month changes (`*_chg`)
- Year-over-year changes (`*_yoy`, `CPI_yoy_pct`, `PCE_yoy_pct`)
- Yield curve spread
- Rolling 18-month correlations at lags 0, 3, 6, 12 months

### `flag_risk(df)`

Returns `(df_with_flags, flags_df)` where:

- `flags_df` — boolean DataFrame, one column per risk condition
- `df["risk_score"]` — integer count of active flags (0–7)
- `df["elevated_risk"]` — `True` when `risk_score >= 3`
- `df["active_flags"]` — comma-separated string of active flag names

### `classify_regime(df)` *(dashboard.py)*

Maps `risk_score` to a named regime:

| Score | Regime | Colour |
|-------|--------|--------|
| 0–1 | Normal / Benign | Green |
| 2 | Caution | Yellow |
| 3–4 | Elevated Risk | Orange |
| 5–7 | High Risk | Red |

---

## Running the Tests

The test suite uses [pytest](https://docs.pytest.org/) and covers boundary-value edge cases for all seven risk-flag thresholds, the analytics pipeline, and dashboard utility functions.

```bash
# Install dev dependencies and run all tests
uv sync --group dev
uv run pytest

# Verbose output with short tracebacks
uv run pytest -v --tb=short

# Run only the main.py tests
uv run pytest tests/test_main.py

# Run only the dashboard tests
uv run pytest tests/test_dashboard.py
```

### What is tested

| Module | Test class | Focus |
|--------|------------|-------|
| `main.py` | `TestAlignToMonthly` | Output shape, column completeness, forward-fill |
| `main.py` | `TestComputeAnalytics` | Spread, MoM/YoY maths, rolling corr window |
| `main.py` | `TestFlagRiskBoundaries` | Exact boundary values for all 7 thresholds |
| `main.py` | `TestFlagRiskScoreAndElevated` | Score accumulation, `elevated_risk` flag |
| `main.py` | `TestFlagRiskEdgeCases` | Missing columns, all-NaN data, single-row DataFrame |
| `main.py` | `TestRiskThresholdsConstant` | Threshold config completeness and structure |
| `dashboard.py` | `TestClassifyRegime` | Score boundary conditions (0/1/2/3/4/5/7) |
| `dashboard.py` | `TestRecessionShapes` | No recession, ongoing recession, two episodes |
| `dashboard.py` | `TestMakeNlSummary` | Narrative text for various macro conditions |
