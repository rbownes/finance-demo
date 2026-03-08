---
name: fred-lending-risk
description: >
  Fetches full FRED history (1990–present), aligns five macro time series to
  monthly frequency, computes rolling correlations between rate changes and
  delinquency rates at 0/3/6/9/12-month lags, calculates the yield curve
  spread (10Y − Fed Funds), and flags historical periods of elevated credit
  risk for lending businesses using a composite risk-score model. Use this
  skill whenever the user asks about lending risk, credit cycle analysis,
  yield curve inversion history, rate-delinquency correlations, monetary
  transmission lag, historical risk episodes, macro risk regime, or says
  anything like "flag risk periods", "when was the curve inverted", "how do
  rate hikes affect delinquency", "credit stress analysis", or "macro lending
  environment". Prefer this skill over fred-macro when the user wants analysis
  or historical context rather than just current values.
---

# FRED Lending Risk Skill

You are running a full historical macro-credit risk analysis using FRED data.
The heavy lifting is done by a bundled Python script — your job is to find the
API key, run the script, and present the output clearly.

## Step 1 — Find the API key

```bash
grep -h "FRED" ~/.bashrc ~/.zshrc 2>/dev/null | grep -i "api_key\|FRED_API" | head -3
```

Extract the key value. If not found, ask the user to set `FRED_API_KEY` in
their shell config.

## Step 2 — Run the analysis script

The bundled script handles everything: fetching 35+ years of history,
resampling to monthly frequency, computing analytics, and printing markdown
output. Run it from the project root so it can find the venv:

```bash
cd /path/to/project   # wherever this skill lives (check base directory)
FRED_API_KEY="<key>" .venv/bin/python \
  .claude/skills/fred-lending-risk/scripts/analyze.py
```

If `.venv` doesn't exist or is missing pandas/numpy, fall back to:

```bash
FRED_API_KEY="<key>" python3 \
  .claude/skills/fred-lending-risk/scripts/analyze.py
```

The script takes ~5–10 seconds (6 parallel-ish FRED fetches).

## Step 3 — Present the output

The script prints five markdown sections in order:

1. **Current Snapshot** — latest values + risk regime badge (🟢/🟡/🟠/🔴)
2. **Yield Curve Spread** — annual table from 1990 + recent months, ⚠ flagged inversions
3. **Rolling Correlations** — 18-month rolling r at lags 0/3/6/9/12m for last 24 months
4. **Elevated Risk Episodes** — contiguous periods where risk score ≥ 3, with peak flags
5. **Interpretation** — prose paragraph tying it together

Present each section as-is. The markdown tables render natively in the
conversation. After the script output, you may add a brief follow-up sentence
if the user asked a specific question (e.g., "is now a good time to expand
lending?") — but keep it short; the data speaks for itself.

## What the risk score measures

Each month gets a score 0–7 based on how many of these conditions fire:

| Flag | Condition |
|------|-----------|
| `inverted_curve` | 10Y − FF spread < 0 |
| `near_inverted` | spread < 0.5pp |
| `rising_delinquency` | delinquency YoY > +10bps |
| `high_delinquency` | delinquency > 1.8% |
| `high_inflation` | CPI YoY > 4% |
| `rapid_rate_hike` | FF rate YoY > +200bps |
| `high_rates` | FF rate > 5% |

Score ≥ 3 = elevated risk episode. This threshold is documented so users can
ask you to adjust it if they want a stricter or looser definition.

## Error handling

- If the script fails with a missing-module error, tell the user to run
  `uv add pandas numpy requests` (or `pip install`) from the project root.
- If FRED returns an error for a series, the script will print a stderr
  warning and continue; note it to the user.
- If the API key is wrong (HTTP 400), tell the user to verify `FRED_API_KEY`.
