import os
import requests
import pandas as pd
import numpy as np

FRED_API_KEY = os.environ.get("FRED_API_KEY")
FRED_BASE_URL = "https://api.stlouisfed.org/fred/series/observations"

SERIES = {
    "FEDFUNDS": "Federal Funds Rate",
    "DGS10":    "10-Year Treasury Yield",
    "DRALACBN": "Consumer Loan Delinquency Rate",
    "PCE":      "Personal Consumption Expenditures",
    "CPIAUCSL": "CPI (All Urban Consumers)",
}

# Thresholds for risk flagging
RISK_THRESHOLDS = {
    "inverted_curve":    ("spread",           "<",   0.0),
    "near_inverted":     ("spread",           "<",   0.5),
    "rising_delinquency":("DRALACBN_yoy",     ">",   0.10),   # +10bps YoY
    "high_delinquency":  ("DRALACBN",         ">",   1.80),
    "high_inflation":    ("CPI_yoy_pct",       ">",   4.0),
    "rapid_rate_hike":   ("FEDFUNDS_yoy",     ">",   2.0),    # +200bps YoY
    "high_rates":        ("FEDFUNDS",         ">",   5.0),
}


# ── Data fetching ──────────────────────────────────────────────────────────────

def fetch_series(series_id: str, start: str = "1990-01-01") -> pd.DataFrame:
    params = {
        "series_id": series_id,
        "api_key": FRED_API_KEY,
        "file_type": "json",
        "observation_start": start,
        "sort_order": "asc",
    }
    r = requests.get(FRED_BASE_URL, params=params)
    r.raise_for_status()
    obs = r.json()["observations"]
    df = pd.DataFrame(obs)[["date", "value"]]
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df.set_index("date").rename(columns={"value": series_id})


def fetch_all() -> dict[str, pd.DataFrame]:
    print("Fetching FRED series from 1990 …")
    return {sid: fetch_series(sid) for sid in SERIES}


# ── Alignment ─────────────────────────────────────────────────────────────────

def align_to_monthly(raw: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Resample everything to month-end:
      - FEDFUNDS / PCE / CPIAUCSL : monthly → last value
      - DGS10                     : daily   → monthly mean
      - DRALACBN                  : quarterly → forward-fill
    """
    resampled = {
        "FEDFUNDS": raw["FEDFUNDS"].resample("ME").last(),
        "DGS10":    raw["DGS10"].resample("ME").mean(),
        "DRALACBN": raw["DRALACBN"].resample("ME").last().ffill(),
        "PCE":      raw["PCE"].resample("ME").last(),
        "CPIAUCSL": raw["CPIAUCSL"].resample("ME").last(),
    }
    df = pd.concat(resampled.values(), axis=1)
    df.columns = list(resampled.keys())
    return df.dropna(subset=["FEDFUNDS", "DGS10", "CPIAUCSL"])


# ── Analytics ─────────────────────────────────────────────────────────────────

def compute_analytics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Month-over-month changes
    df["FEDFUNDS_chg"] = df["FEDFUNDS"].diff()
    df["DGS10_chg"]    = df["DGS10"].diff()
    df["DRALACBN_chg"] = df["DRALACBN"].diff()

    # Year-over-year changes
    df["FEDFUNDS_yoy"]  = df["FEDFUNDS"].diff(12)
    df["DGS10_yoy"]     = df["DGS10"].diff(12)
    df["DRALACBN_yoy"]  = df["DRALACBN"].diff(12)   # ffilled to monthly → diff(12) = true YoY
    df["CPI_yoy_pct"]   = df["CPIAUCSL"].pct_change(12) * 100
    df["PCE_yoy_pct"]   = df["PCE"].pct_change(12) * 100

    # Yield curve spread
    df["spread"] = df["DGS10"] - df["FEDFUNDS"]

    # Rolling 18-month correlation: rate changes → delinquency changes
    roll = 18
    df["corr_ff_delinq"]       = df["FEDFUNDS_chg"].rolling(roll).corr(df["DRALACBN_chg"])
    df["corr_dgs10_delinq"]    = df["DGS10_chg"].rolling(roll).corr(df["DRALACBN_chg"])

    # Lagged correlations: rate change N months ago vs current delinquency change
    for lag in (3, 6, 12):
        df[f"corr_ff_lag{lag}_delinq"] = (
            df["FEDFUNDS_chg"].shift(lag).rolling(roll).corr(df["DRALACBN_chg"])
        )

    return df


# ── Risk flagging ─────────────────────────────────────────────────────────────

def flag_risk(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    flags = pd.DataFrame(index=df.index)

    for name, (col, op, threshold) in RISK_THRESHOLDS.items():
        if col not in df.columns:
            continue
        if op == "<":
            flags[name] = df[col] < threshold
        elif op == ">":
            flags[name] = df[col] > threshold

    df["risk_score"]   = flags.sum(axis=1)
    df["elevated_risk"] = df["risk_score"] >= 3
    df["active_flags"] = flags.apply(
        lambda row: ", ".join(f for f in flags.columns if row[f]), axis=1
    )
    return df, flags


# ── Reporting ─────────────────────────────────────────────────────────────────

def print_section(title: str):
    print(f"\n{'═' * 72}")
    print(f"  {title}")
    print(f"{'═' * 72}")


def report_latest(df: pd.DataFrame):
    print_section("LATEST VALUES (most recent available per series)")
    rows = [
        ("Federal Funds Rate",          "FEDFUNDS",   "%",  df["FEDFUNDS"].last_valid_index()),
        ("10-Year Treasury Yield",      "DGS10",      "%",  df["DGS10"].last_valid_index()),
        ("Consumer Loan Delinquency",   "DRALACBN",   "%",  df["DRALACBN"].last_valid_index()),
        ("PCE ($B)",                    "PCE",        "$B", df["PCE"].last_valid_index()),
        ("CPI",                         "CPIAUCSL",   "",   df["CPIAUCSL"].last_valid_index()),
        ("Yield Curve Spread",          "spread",     "pp", df["spread"].last_valid_index()),
        ("CPI YoY %",                   "CPI_yoy_pct","%", df["CPI_yoy_pct"].last_valid_index()),
    ]
    print(f"  {'Indicator':<35} {'Date':<12} {'Value':>10}  {'Unit'}")
    print(f"  {'-'*65}")
    for label, col, unit, idx in rows:
        val = df.loc[idx, col]
        print(f"  {label:<35} {str(idx.date()):<12} {val:>10.4f}  {unit}")


def report_rolling_correlations(df: pd.DataFrame):
    print_section("ROLLING 18-MONTH CORRELATIONS: Rate Changes → Delinquency Changes")
    print("  (Lag = months by which rate change precedes delinquency change)")
    print()
    print(f"  {'Date':<12}  {'FF→Delinq':>10}  {'DGS10→Delinq':>13}  "
          f"{'FF lag3':>8}  {'FF lag6':>8}  {'FF lag12':>9}")
    print(f"  {'-'*68}")

    # Show last 24 months, then a few historical snapshots
    recent = df[["corr_ff_delinq", "corr_dgs10_delinq",
                 "corr_ff_lag3_delinq", "corr_ff_lag6_delinq",
                 "corr_ff_lag12_delinq"]].dropna().tail(24)

    for idx, row in recent.iterrows():
        print(f"  {str(idx.date()):<12}  "
              f"{row['corr_ff_delinq']:>10.3f}  "
              f"{row['corr_dgs10_delinq']:>13.3f}  "
              f"{row['corr_ff_lag3_delinq']:>8.3f}  "
              f"{row['corr_ff_lag6_delinq']:>8.3f}  "
              f"{row['corr_ff_lag12_delinq']:>9.3f}")


def report_yield_curve(df: pd.DataFrame):
    print_section("YIELD CURVE SPREAD (10Y − Fed Funds)  — Annual Snapshots")
    print("  Negative = inverted (historically precedes recessions/credit stress)\n")
    print(f"  {'Date':<12}  {'FEDFUNDS':>9}  {'DGS10':>7}  {'Spread':>8}  {'Signal'}")
    print(f"  {'-'*58}")

    # Annual snapshot (December of each year) + recent months
    ann = df.resample("YE").last()
    recent_12 = df.tail(12)
    combined = pd.concat([ann, recent_12]).loc[~pd.concat([ann, recent_12]).index.duplicated(keep="last")]
    combined = combined.sort_index()

    for idx, row in combined[["FEDFUNDS", "DGS10", "spread"]].dropna().iterrows():
        signal = "⚠ INVERTED" if row["spread"] < 0 else ("~ near flat" if row["spread"] < 0.5 else "")
        print(f"  {str(idx.date()):<12}  {row['FEDFUNDS']:>9.2f}  {row['DGS10']:>7.2f}  "
              f"{row['spread']:>8.2f}  {signal}")


def report_risk_periods(df: pd.DataFrame, flags: pd.DataFrame):
    print_section("ELEVATED RISK PERIODS FOR LENDING BUSINESSES  (risk_score ≥ 3)")
    print("  Criteria: inverted/near-inverted curve, rising/high delinquency,")
    print("            high inflation, rapid rate hikes, high absolute rates\n")

    elevated = df[df["elevated_risk"]].copy()
    if elevated.empty:
        print("  No periods found.")
        return

    # Collapse consecutive months into contiguous episodes
    elevated["episode"] = (elevated.index.to_series().diff() > pd.Timedelta("45D")).cumsum()
    episodes = elevated.groupby("episode").agg(
        start=("risk_score", lambda x: x.index[0].date()),
        end=("risk_score",   lambda x: x.index[-1].date()),
        peak_score=("risk_score", "max"),
        months=("risk_score", "count"),
    )

    # Representative flags from the peak month
    def peak_flags(group):
        peak_idx = group["risk_score"].idxmax()
        return group.loc[peak_idx, "active_flags"]

    episodes["peak_flags"] = elevated.groupby("episode").apply(peak_flags)

    print(f"  {'Start':<12}  {'End':<12}  {'Months':>6}  {'Peak':>5}  Active conditions at peak")
    print(f"  {'-'*80}")
    for _, ep in episodes.iterrows():
        print(f"  {str(ep['start']):<12}  {str(ep['end']):<12}  "
              f"{ep['months']:>6}  {ep['peak_score']:>5}  {ep['peak_flags']}")

    # Summary: score distribution over time
    print()
    print("  Risk score distribution (all months since 1992):")
    score_counts = df["risk_score"].value_counts().sort_index()
    total = score_counts.sum()
    for score, count in score_counts.items():
        bar = "█" * int(count / total * 40)
        print(f"    score {score}: {count:>4} months  {bar}")


def report_correlation_summary(df: pd.DataFrame):
    print_section("CORRELATION SUMMARY: Full-Sample & By-Regime")

    print("  Full-sample correlations with DRALACBN changes:\n")
    targets = {
        "FF rate change":         "FEDFUNDS_chg",
        "FF rate change (lag 3m)": None,
        "FF rate change (lag 6m)": None,
        "FF rate change (lag 12m)": None,
        "10Y yield change":       "DGS10_chg",
        "Yield curve spread":     "spread",
        "CPI YoY %":              "CPI_yoy_pct",
    }

    sub = df[["DRALACBN_chg", "FEDFUNDS_chg", "DGS10_chg",
              "spread", "CPI_yoy_pct"]].dropna()
    for lag in (3, 6, 12):
        sub[f"FF_lag{lag}"] = sub["FEDFUNDS_chg"].shift(lag)

    corr_map = {
        "FF rate change":          "FEDFUNDS_chg",
        "FF rate change (lag 3m)": "FF_lag3",
        "FF rate change (lag 6m)": "FF_lag6",
        "FF rate change (lag 12m)":"FF_lag12",
        "10Y yield change":        "DGS10_chg",
        "Yield curve spread":      "spread",
        "CPI YoY %":               "CPI_yoy_pct",
    }

    print(f"  {'Variable':<28} {'r':>7}  {'Interpretation'}")
    print(f"  {'-'*70}")
    for label, col in corr_map.items():
        r = sub["DRALACBN_chg"].corr(sub[col])
        interp = (
            "strong positive" if r > 0.4 else
            "moderate positive" if r > 0.2 else
            "weak positive" if r > 0.05 else
            "weak negative" if r > -0.2 else
            "moderate negative" if r > -0.4 else
            "strong negative"
        )
        print(f"  {label:<28} {r:>7.3f}  {interp}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    raw = fetch_all()
    df = align_to_monthly(raw)
    df = compute_analytics(df)
    df, flags = flag_risk(df)

    report_latest(df)
    report_yield_curve(df)
    report_rolling_correlations(df)
    report_correlation_summary(df)
    report_risk_periods(df, flags)


if __name__ == "__main__":
    main()
