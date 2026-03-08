---
name: fred-dashboard
description: >
  Generates a self-contained interactive HTML dashboard with FRED macroeconomic
  data visualizations for lending risk analysis. Use this skill whenever the
  user asks to generate, update, or view a dashboard, chart, or visualization of
  macro data — yield curve, recession bands, delinquency trends, correlation
  heatmap, risk regime indicator, or natural language macro summary. Also
  triggers on: "show me the dashboard", "update the charts", "visualize the
  FRED data", "open the dashboard", "generate the html", "lending risk
  visualization", or any request to see data in a graphical or interactive
  format rather than tables.
---

# FRED Dashboard Skill

You are generating a self-contained interactive HTML dashboard that visualizes
U.S. macro lending-risk indicators from FRED. The bundled Python script handles
everything — fetching ~35 years of history, aligning time series, computing
analytics, building four Plotly charts, and writing a standalone HTML file.

## Step 1 — Find the API key

```bash
grep -h "FRED" ~/.bashrc ~/.zshrc 2>/dev/null | grep -i "api_key\|FRED_API" | head -3
```

Extract the key. If not found, ask the user to set `FRED_API_KEY` in their
shell config.

## Step 2 — Run the dashboard script

Run from the project root so the venv is found:

```bash
cd /home/motoko/Projects/fin-test
FRED_API_KEY="<key>" .venv/bin/python \
  .claude/skills/fred-dashboard/scripts/generate_dashboard.py
```

If `.venv` doesn't exist or is missing dependencies, fall back to:

```bash
FRED_API_KEY="<key>" python3 \
  .claude/skills/fred-dashboard/scripts/generate_dashboard.py
```

The script takes ~10–15 seconds (6 FRED fetches + chart rendering).

## Step 3 — Present the result

The script prints the output path and automatically opens the file in the
default browser via `xdg-open`. Tell the user:

- The filename (e.g., `fred_dashboard_20260308.html`)
- That it's fully self-contained — one HTML file with Plotly.js embedded via
  CDN, no server needed, can be opened on any machine or shared via email
- What the four sections contain (see below)

If `xdg-open` fails (headless environment), just report the file path.

## Dashboard contents

| Section | What it shows |
|---------|---------------|
| **Yield Curve** | Fed Funds + 10Y Treasury overlay with NBER recession bands (grey) and inversion periods (red tint); spread in lower panel |
| **Correlation Heatmap** | 18-month rolling Pearson r between rate changes and delinquency changes at lags 0/3/6/9/12m; annotated band highlights 6-12m lag |
| **Risk Regime** | Gauge showing current risk score (0–7) + timeline of monthly risk scores with elevated-risk threshold line |
| **Summary** | Natural language paragraph with current snapshot values, regime assessment, and interpretation |

## Error handling

- **Missing module** (`pandas`, `numpy`, `plotly`, `requests`): tell the user
  to run `uv add pandas numpy plotly requests` from the project root
- **FRED HTTP 400**: wrong API key — ask user to verify `FRED_API_KEY`
- **xdg-open not found**: normal in headless/SSH environments; just report the
  file path so the user can copy it to their browser
- **Script stderr warnings**: the script suppresses FutureWarnings; any other
  warnings can be noted but are usually non-fatal
