---
name: fred-macro
description: >
  Fetches and displays key U.S. macroeconomic indicators from the FRED API
  (Federal Reserve Economic Data). Use this skill whenever the user asks to
  pull FRED data, show macro indicators, get economic data, check interest
  rates, look up CPI or PCE, check delinquency rates, or says anything like
  "pull FRED series", "show me the Fed Funds rate", "what's the 10-year yield",
  "get economic indicators", "macro data", or "FRED". Even if the user doesn't
  say "FRED" explicitly, use this skill any time they want current U.S. rate,
  inflation, or credit data.
---

# FRED Macro Indicators Skill

You are fetching live macroeconomic data from the St. Louis Fed's FRED API and
presenting it in a clean, readable format.

## Step 1 — Verify the API key exists

Check that the `FRED_API_KEY` environment variable is set (do **not** print its value):

```bash
[ -n "$FRED_API_KEY" ] && echo "FRED_API_KEY is set" || echo "FRED_API_KEY is NOT set"
```

If the variable is not set, tell the user to set `export FRED_API_KEY="..."` in
their `.zshrc` (or `.bashrc`).

## Step 2 — Fetch the 5 series

Use `curl` to fetch the last **12 observations** for each series, sorted
ascending. Run all 5 fetches in a single bash command (background jobs +
`wait`) so they finish in parallel:

```bash
BASE="https://api.stlouisfed.org/fred/series/observations"
KEY="${FRED_API_KEY}"
PARAMS="&api_key=${KEY}&file_type=json&sort_order=desc&limit=12"

curl -s "${BASE}?series_id=FEDFUNDS${PARAMS}" -o /tmp/fred_FEDFUNDS.json &
curl -s "${BASE}?series_id=DGS10${PARAMS}"    -o /tmp/fred_DGS10.json    &
curl -s "${BASE}?series_id=DRALACBN${PARAMS}" -o /tmp/fred_DRALACBN.json &
curl -s "${BASE}?series_id=PCE${PARAMS}"      -o /tmp/fred_PCE.json      &
curl -s "${BASE}?series_id=CPIAUCSL${PARAMS}" -o /tmp/fred_CPIAUCSL.json &
wait
```

## Step 3 — Parse and display

Use Python (available as `python3`) to parse the JSON files and produce the
formatted output. Here is a template to work from:

```python
import json

SERIES = [
    ("FEDFUNDS",  "Federal Funds Rate",           "%"),
    ("DGS10",     "10-Year Treasury Yield",        "%"),
    ("DRALACBN",  "Consumer Loan Delinquency",     "%"),
    ("PCE",       "Personal Consumption Expend.",  "$B"),
    ("CPIAUCSL",  "CPI (All Urban Consumers)",     "idx"),
]

def load(sid):
    with open(f"/tmp/fred_{sid}.json") as f:
        obs = json.load(f)["observations"]
    # API returns desc (newest first); keep that order for display
    return [(o["date"], o["value"]) for o in obs if o["value"] != "."]

# ── Markdown summary table ─────────────────────────────────────────────────
latest_vals = {}
print("\n| Series | Indicator | Latest | Date | Unit |")
print("|--------|-----------|-------:|------|------|")
for sid, name, unit in SERIES:
    data = load(sid)
    date, val = data[0]
    latest_vals[sid] = float(val)
    print(f"| {sid} | {name} | {float(val):.3f} | {date} | {unit} |")

# ── Per-series markdown tables (recent 6 observations) ────────────────────
for sid, name, unit in SERIES:
    data = load(sid)
    print(f"\n**{name} ({sid})** [{unit}]")
    print(f"\n| Date | Value |")
    print(f"|------|------:|")
    for date, val in data[:6]:
        print(f"| {date} | {float(val):.4f} |")
```

After the per-series tables, add a **Context note** block computing the yield
curve spread and CPI YoY. Example addition to the script:

```python
spread = latest_vals["DGS10"] - latest_vals["FEDFUNDS"]
cpi_data = load("CPIAUCSL")
cpi_yoy = (float(cpi_data[0][1]) / float(cpi_data[-1][1]) - 1) * 100
del_data = load("DRALACBN")
del_trend = "rising" if float(del_data[0][1]) > float(del_data[-1][1]) else "easing"
inv = "⚠ inverted" if spread < 0 else "positive"
print(f"\n---")
print(f"\n**Context:** 10Y–FF spread is **{spread:+.2f}pp** ({inv}). "
      f"CPI YoY ≈ **{cpi_yoy:.2f}%**. "
      f"Delinquency is **{del_trend}** ({latest_vals['DRALACBN']:.2f}%). "
      f"PCE at **${latest_vals['PCE']:,.0f}B**.")
```

Run this with `python3 /tmp/fred_display.py` (write the script first, then run
it).

## Output format

Present the markdown output exactly as the script produces it — summary table
first, then per-series tables, then the context line. The markdown will render
as formatted tables in the conversation.

## Error handling

- If `curl` fails for a series (empty file or missing `observations` key), note
  it inline rather than crashing: print `N/A` for that series and continue.
- If no API key is found, stop and ask the user before proceeding.
