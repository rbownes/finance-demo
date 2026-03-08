"""
FRED Lending Risk Analyzer
Fetches full history, aligns time series, computes rolling correlations,
yield curve spread, and flags elevated-risk periods for lending businesses.
"""

import os
import sys
import warnings
import requests
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore", category=FutureWarning)

FRED_API_KEY  = os.environ.get("FRED_API_KEY", "")
FRED_BASE_URL = "https://api.stlouisfed.org/fred/series/observations"

RISK_THRESHOLDS = {
    "inverted_curve":     ("spread",        "<",  0.0),
    "near_inverted":      ("spread",        "<",  0.5),
    "rising_delinquency": ("DRALACBN_yoy",  ">",  0.10),
    "high_delinquency":   ("DRALACBN",      ">",  1.80),
    "high_inflation":     ("CPI_yoy_pct",   ">",  4.0),
    "rapid_rate_hike":    ("FEDFUNDS_yoy",  ">",  2.0),
    "high_rates":         ("FEDFUNDS",      ">",  5.0),
}

ROLL = 18   # months for rolling correlations
LAGS = [0, 3, 6, 9, 12]

# ── Fetch ──────────────────────────────────────────────────────────────────

def fetch(series_id, start="1990-01-01"):
    r = requests.get(FRED_BASE_URL, params={
        "series_id": series_id, "api_key": FRED_API_KEY,
        "file_type": "json", "observation_start": start,
        "sort_order": "asc",
    }, timeout=15)
    r.raise_for_status()
    obs = r.json()["observations"]
    df = pd.DataFrame(obs)[["date", "value"]]
    df["date"]  = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df.set_index("date").rename(columns={"value": series_id})

def fetch_all():
    series = ["FEDFUNDS", "DGS10", "DRALACBN", "PCE", "CPIAUCSL", "USREC"]
    print("Fetching FRED series from 1990…", file=sys.stderr)
    return {s: fetch(s) for s in series}

# ── Align ─────────────────────────────────────────────────────────────────

def align(raw):
    df = pd.concat([
        raw["FEDFUNDS"].resample("ME").last(),
        raw["DGS10"].resample("ME").mean(),
        raw["DRALACBN"].resample("ME").last().ffill(),
        raw["PCE"].resample("ME").last(),
        raw["CPIAUCSL"].resample("ME").last(),
        raw["USREC"].resample("ME").max(),
    ], axis=1).dropna(subset=["FEDFUNDS", "DGS10", "CPIAUCSL"])
    df.columns = ["FEDFUNDS", "DGS10", "DRALACBN", "PCE", "CPIAUCSL", "USREC"]
    return df

# ── Analytics ─────────────────────────────────────────────────────────────

def compute(df):
    df = df.copy()
    df["spread"]       = df["DGS10"] - df["FEDFUNDS"]
    df["FEDFUNDS_chg"] = df["FEDFUNDS"].diff()
    df["DGS10_chg"]    = df["DGS10"].diff()
    df["DRALACBN_chg"] = df["DRALACBN"].diff()
    df["FEDFUNDS_yoy"] = df["FEDFUNDS"].diff(12)
    df["DGS10_yoy"]    = df["DGS10"].diff(12)
    df["DRALACBN_yoy"] = df["DRALACBN"].diff(12)
    df["CPI_yoy_pct"]  = df["CPIAUCSL"].pct_change(12) * 100
    df["PCE_yoy_pct"]  = df["PCE"].pct_change(12) * 100

    for lag in LAGS:
        df[f"corr_lag{lag}"] = (
            df["FEDFUNDS_chg"].shift(lag)
            .rolling(ROLL).corr(df["DRALACBN_chg"])
        )
    return df

# ── Risk flagging ─────────────────────────────────────────────────────────

def flag_risk(df):
    df = df.copy()
    flags = pd.DataFrame(index=df.index)
    for name, (col, op, thr) in RISK_THRESHOLDS.items():
        if col not in df.columns:
            continue
        flags[name] = (df[col] < thr) if op == "<" else (df[col] > thr)

    df["risk_score"]    = flags.sum(axis=1)
    df["elevated_risk"] = df["risk_score"] >= 3
    df["active_flags"]  = flags.apply(
        lambda row: ", ".join(f for f in flags.columns if row[f]), axis=1
    )
    return df, flags

# ── Markdown output ───────────────────────────────────────────────────────

def md_latest(df):
    row = df.dropna(subset=["spread", "DRALACBN", "CPI_yoy_pct"]).iloc[-1]
    as_of = df.dropna(subset=["spread"]).index[-1].strftime("%B %Y")

    # Current risk regime
    score = int(row["risk_score"])
    regime = (
        "🟢 Normal / Benign" if score <= 1 else
        "🟡 Caution"         if score == 2 else
        "🟠 Elevated Risk"   if score <= 4 else
        "🔴 High Risk"
    )

    print(f"## Current Snapshot — {as_of}\n")
    print(f"**Risk Regime: {regime}** (score {score}/7)\n")
    print("| Indicator | Value | Unit |")
    print("|-----------|------:|------|")
    print(f"| Federal Funds Rate | {row['FEDFUNDS']:.2f} | % |")
    print(f"| 10-Year Treasury | {row['DGS10']:.2f} | % |")
    print(f"| Yield Curve Spread | {row['spread']:+.2f} | pp |")
    print(f"| Consumer Loan Delinquency | {row['DRALACBN']:.2f} | % |")
    print(f"| CPI YoY | {row['CPI_yoy_pct']:.2f} | % |")
    print(f"| PCE | {row['PCE']:,.0f} | $B |")
    if row["active_flags"]:
        print(f"\n**Active risk flags:** {row['active_flags']}")


def md_yield_curve(df):
    print("\n## Yield Curve Spread (10Y − Fed Funds) — Annual Snapshots\n")
    print("_Negative = inverted. Shaded rows historically precede recessions/credit stress._\n")
    print("| Year | Fed Funds | 10Y | Spread | Signal |")
    print("|------|----------:|----:|-------:|--------|")

    ann = df.resample("YE").last()
    recent = df.tail(12)
    combined = pd.concat([ann, recent]).loc[~pd.concat([ann, recent]).index.duplicated(keep="last")].sort_index()

    for dt, row in combined[["FEDFUNDS", "DGS10", "spread"]].dropna().iterrows():
        signal = "⚠ INVERTED" if row["spread"] < 0 else ("~ near flat" if row["spread"] < 0.5 else "")
        print(f"| {dt.strftime('%Y-%m')} | {row['FEDFUNDS']:.2f} | {row['DGS10']:.2f} | {row['spread']:+.2f} | {signal} |")


def md_correlations(df):
    print("\n## Rolling Rate-Change → Delinquency Correlations (last 24 months)\n")
    print("_18-month rolling Pearson r. Positive = rate hikes predict future delinquency rise._\n")

    cols = [f"corr_lag{l}" for l in LAGS]
    headers = " | ".join(f"Lag {l}m" for l in LAGS)
    seps    = " | ".join("------:" for _ in LAGS)
    print(f"| Date | {headers} |")
    print(f"|------|{seps}|")

    recent = df[cols].dropna().tail(24)
    for dt, row in recent.iterrows():
        vals = " | ".join(f"{row[c]:+.3f}" for c in cols)
        print(f"| {dt.strftime('%Y-%m')} | {vals} |")

    # Full-sample peak lag
    full_corrs = {lag: df["FEDFUNDS_chg"].shift(lag).corr(df["DRALACBN_chg"]) for lag in LAGS}
    best_lag  = max(full_corrs, key=lambda l: full_corrs[l])
    best_r    = full_corrs[best_lag]
    print(f"\n_Full-sample peak correlation: lag **{best_lag}m** (r = {best_r:+.3f})_")


def md_risk_episodes(df):
    print("\n## Elevated Risk Episodes for Lending (risk score ≥ 3)\n")
    print("_Criteria: inverted/near-flat curve, rising/high delinquency, high inflation, rapid hikes, high absolute rates_\n")

    elevated = df[df["elevated_risk"]].copy()
    if elevated.empty:
        print("_No elevated-risk periods identified._")
        return

    elevated["episode"] = (
        elevated.index.to_series().diff() > pd.Timedelta("45D")
    ).cumsum()

    def peak_flags(group):
        return group.loc[group["risk_score"].idxmax(), "active_flags"]

    eps = elevated.groupby("episode").agg(
        start=("risk_score", lambda x: x.index[0].strftime("%Y-%m")),
        end=("risk_score",   lambda x: x.index[-1].strftime("%Y-%m")),
        months=("risk_score", "count"),
        peak=("risk_score", "max"),
    )
    eps["flags"] = elevated.groupby("episode").apply(peak_flags, include_groups=False)

    print("| Start | End | Months | Peak Score | Active Conditions at Peak |")
    print("|-------|-----|-------:|-----------:|---------------------------|")
    for _, ep in eps.iterrows():
        print(f"| {ep['start']} | {ep['end']} | {ep['months']} | {ep['peak']}/7 | {ep['flags']} |")

    # Score distribution
    print("\n**Risk score distribution (all months):**\n")
    score_counts = df["risk_score"].value_counts().sort_index()
    total = score_counts.sum()
    print("| Score | Months | Share | Bar |")
    print("|------:|-------:|------:|-----|")
    for score, count in score_counts.items():
        bar = "█" * int(count / total * 20)
        print(f"| {score} | {count} | {count/total*100:.1f}% | {bar} |")


def md_context(df):
    row    = df.dropna(subset=["spread", "DRALACBN", "CPI_yoy_pct"]).iloc[-1]
    spread = row["spread"]
    as_of  = df.dropna(subset=["spread"]).index[-1].strftime("%B %Y")

    full_corrs = {lag: df["FEDFUNDS_chg"].shift(lag).corr(df["DRALACBN_chg"]) for lag in LAGS}
    best_lag   = max(full_corrs, key=lambda l: full_corrs[l])
    best_r     = full_corrs[best_lag]

    del_data   = df["DRALACBN"].dropna().tail(4)
    del_trend  = "rising" if del_data.iloc[-1] > del_data.iloc[0] else "easing"
    spread_dir = "above" if spread > 0 else "below"

    print("\n## Interpretation\n")
    print(
        f"As of **{as_of}**, the 10Y–FF spread is **{spread:+.2f}pp** ({spread_dir} zero). "
        + ("The curve is **inverted**, historically preceding recessions by 12–18 months and compressing lender net interest margins. " if spread < 0 else "A positive slope supports conventional maturity-transformation lending. ")
        + f"The full-sample cross-correlogram shows Fed Funds changes peak-correlate with delinquency at a **{best_lag}-month lag** "
        f"(r = {best_r:+.3f}), consistent with the well-documented monetary transmission lag — borrowers feel the pinch "
        f"of rate moves only after resets, refinancings, and income pressure accumulate. "
        f"Consumer loan delinquency is currently **{del_trend}** at {row['DRALACBN']:.2f}%, "
        f"and CPI is running at **{row['CPI_yoy_pct']:.2f}% YoY**."
    )


# ── Entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not FRED_API_KEY:
        print("ERROR: FRED_API_KEY not set.", file=sys.stderr)
        sys.exit(1)

    raw = fetch_all()
    df  = align(raw)
    df  = compute(df)
    df, flags = flag_risk(df)

    md_latest(df)
    md_yield_curve(df)
    md_correlations(df)
    md_risk_episodes(df)
    md_context(df)
