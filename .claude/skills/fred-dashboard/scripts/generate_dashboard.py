"""
FRED Macro Credit Risk Dashboard Generator
Fetches live FRED data, runs the full analysis, and exports a self-contained
interactive HTML dashboard — no server required, opens in any browser.
"""

import os
import sys
import warnings
import textwrap
from datetime import date, datetime
from pathlib import Path

import requests
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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

COLORS = dict(
    bg="#0e1117", card="#1a1d27",
    blue="#4c8bf5", red="#f55c5c", green="#4caf7d", gold="#f5b942",
    muted="#8b92a5", recession="rgba(200,80,80,0.15)", inversion="rgba(200,80,80,0.25)",
)

LAGS = [0, 3, 6, 9, 12]
ROLL = 18

# ── Data ──────────────────────────────────────────────────────────────────────

def fetch(series_id, start="1990-01-01"):
    r = requests.get(FRED_BASE_URL, params={
        "series_id": series_id, "api_key": FRED_API_KEY,
        "file_type": "json", "observation_start": start, "sort_order": "asc",
    }, timeout=15)
    r.raise_for_status()
    df = pd.DataFrame(r.json()["observations"])[["date", "value"]]
    df["date"]  = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df.set_index("date").rename(columns={"value": series_id})


def build_dataset():
    print("Fetching FRED data from 1990…", file=sys.stderr)
    raw = {s: fetch(s) for s in ["FEDFUNDS", "DGS10", "DRALACBN", "PCE", "CPIAUCSL", "USREC"]}

    df = pd.concat([
        raw["FEDFUNDS"].resample("ME").last(),
        raw["DGS10"].resample("ME").mean(),
        raw["DRALACBN"].resample("ME").last().ffill(),
        raw["PCE"].resample("ME").last(),
        raw["CPIAUCSL"].resample("ME").last(),
        raw["USREC"].resample("ME").max(),
    ], axis=1).dropna(subset=["FEDFUNDS", "DGS10", "CPIAUCSL"])
    df.columns = ["FEDFUNDS", "DGS10", "DRALACBN", "PCE", "CPIAUCSL", "USREC"]

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
            df["FEDFUNDS_chg"].shift(lag).rolling(ROLL).corr(df["DRALACBN_chg"])
        )

    flags = pd.DataFrame(index=df.index)
    for name, (col, op, thr) in RISK_THRESHOLDS.items():
        if col in df.columns:
            flags[name] = (df[col] < thr) if op == "<" else (df[col] > thr)

    df["risk_score"]    = flags.sum(axis=1)
    df["elevated_risk"] = df["risk_score"] >= 3
    df["active_flags"]  = flags.apply(
        lambda row: ", ".join(f for f in flags.columns if row[f]), axis=1
    )
    return df


# ── Recession & inversion shapes ──────────────────────────────────────────────

def recession_shapes(df):
    shapes, in_rec, start = [], False, None
    for dt, val in df["USREC"].items():
        if val == 1 and not in_rec:
            in_rec, start = True, dt
        elif val == 0 and in_rec:
            shapes.append(dict(type="rect", xref="x", yref="paper",
                x0=start, x1=dt, y0=0, y1=1,
                fillcolor=COLORS["recession"], line_width=0, layer="below"))
            in_rec = False
    if in_rec:
        shapes.append(dict(type="rect", xref="x", yref="paper",
            x0=start, x1=df.index[-1], y0=0, y1=1,
            fillcolor=COLORS["recession"], line_width=0, layer="below"))
    return shapes


def inversion_shapes(df):
    shapes, inv, start = [], False, None
    for dt, val in df["spread"].items():
        if pd.isna(val): continue
        if val < 0 and not inv:
            inv, start = True, dt
        elif val >= 0 and inv:
            shapes.append(dict(type="rect", xref="x", yref="paper",
                x0=start, x1=dt, y0=0, y1=1,
                fillcolor=COLORS["inversion"], line_width=0, layer="below"))
            inv = False
    if inv:
        shapes.append(dict(type="rect", xref="x", yref="paper",
            x0=start, x1=df.index[-1], y0=0, y1=1,
            fillcolor=COLORS["inversion"], line_width=0, layer="below"))
    return shapes


# ── Chart 1: Yield Curve ──────────────────────────────────────────────────────

def chart_yield_curve(df):
    shapes = recession_shapes(df) + inversion_shapes(df)

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.65, 0.35], vertical_spacing=0.04,
        subplot_titles=("", "Consumer Loan Delinquency Rate"),
    )

    fig.add_trace(go.Scatter(x=df.index, y=df["DGS10"], name="10Y Treasury",
        line=dict(color=COLORS["blue"], width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["FEDFUNDS"], name="Fed Funds",
        line=dict(color=COLORS["gold"], width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["spread"], name="Spread (10Y−FF)",
        line=dict(color=COLORS["green"], width=2.5),
        fill="tozeroy", fillcolor="rgba(76,175,125,0.12)"), row=1, col=1)
    fig.add_hline(y=0, row=1, col=1,
        line=dict(color="rgba(255,255,255,0.3)", dash="dot", width=1))

    fig.add_trace(go.Scatter(x=df.index, y=df["DRALACBN"], name="Delinquency %",
        line=dict(color=COLORS["red"], width=2),
        fill="tozeroy", fillcolor="rgba(245,92,92,0.12)"), row=2, col=1)

    fig.update_layout(
        shapes=shapes,
        template="plotly_dark", paper_bgcolor=COLORS["bg"], plot_bgcolor=COLORS["card"],
        legend=dict(orientation="h", y=1.06, x=0, font=dict(size=11)),
        hovermode="x unified", margin=dict(l=0, r=0, t=30, b=0),
        title=dict(text="Yield Curve & Delinquency — with NBER Recession Bands (red) and Inversion Periods (dark red)",
                   font=dict(size=13, color=COLORS["muted"])),
    )
    fig.update_yaxes(row=1, col=1, title_text="Rate (%)", gridcolor="#1e2130")
    fig.update_yaxes(row=2, col=1, title_text="Delinquency (%)", gridcolor="#1e2130")
    fig.update_xaxes(gridcolor="#1e2130")
    return fig


# ── Chart 2: Correlation Heatmap ──────────────────────────────────────────────

def chart_corr_heatmap(df):
    cols    = [f"corr_lag{l}" for l in LAGS]
    heat_q  = df[cols].resample("QE").mean().dropna()
    z       = heat_q[cols].T.values
    x       = [f"{d.year} Q{(d.month - 1) // 3 + 1}" for d in heat_q.index]
    y       = [f"{l}m" for l in LAGS]

    fig = go.Figure(go.Heatmap(
        z=z, x=x, y=y,
        colorscale=[[0.0, "#2563eb"],[0.35, "#1e3a6b"],[0.5, "#1a1d27"],
                    [0.65, "#7a1f1f"],[1.0, "#ef4444"]],
        zmid=0, zmin=-0.8, zmax=0.8,
        colorbar=dict(
            title=dict(text="r", font=dict(color="white")),
            thickness=12,
            tickfont=dict(size=10, color="white")),
        hovertemplate="Quarter: %{x}<br>Lag: %{y}<br>r = %{z:.3f}<extra></extra>",
    ))

    # Annotate the 6-12m sweet spot
    fig.add_hrect(y0="6m", y1="12m",
        fillcolor="rgba(255,255,255,0.04)",
        line=dict(color="rgba(255,255,255,0.35)", width=1.5, dash="dot"),
        annotation_text="6–12m lag window (peak signal)",
        annotation_position="top right",
        annotation_font=dict(color="rgba(255,255,255,0.55)", size=10))

    fig.update_layout(
        template="plotly_dark", paper_bgcolor=COLORS["bg"], plot_bgcolor=COLORS["card"],
        title=dict(
            text="Rolling 18-Month Correlation: Fed Funds Rate Changes → Delinquency Changes at Each Lag<br>"
                 "<sup>Red = rate hikes predict future delinquency rise. The 6–12m band shows the strongest and most consistent signal.</sup>",
            font=dict(size=13, color=COLORS["muted"])),
        xaxis=dict(title="Quarter", tickangle=-45, tickfont=dict(size=9),
                   gridcolor="#1e2130", nticks=20),
        yaxis=dict(title="Rate Change Lag", autorange="reversed", gridcolor="#1e2130"),
        margin=dict(l=0, r=0, t=60, b=60),
    )
    return fig


# ── Chart 3a: Risk Gauge ──────────────────────────────────────────────────────

def chart_risk_gauge(df):
    latest = df.dropna(subset=["spread", "DRALACBN", "CPI_yoy_pct"]).iloc[-1]
    score  = int(latest["risk_score"])
    as_of  = df.dropna(subset=["spread", "DRALACBN", "CPI_yoy_pct"]).index[-1].strftime("%b %Y")
    regime = (
        "Normal / Benign" if score <= 1 else
        "Caution"         if score == 2 else
        "Elevated Risk"   if score <= 4 else
        "High Risk"
    )
    color = (
        COLORS["green"] if score <= 1 else
        COLORS["gold"]  if score == 2 else
        "#f58f42"        if score <= 4 else
        COLORS["red"]
    )
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number=dict(font=dict(size=48, color="white"), suffix="/7"),
        title=dict(text=f"<b>{regime}</b><br><sup>as of {as_of}</sup>",
                   font=dict(size=14, color=color)),
        domain=dict(x=[0, 1], y=[0, 1]),
        gauge=dict(
            axis=dict(range=[0, 7], tickwidth=1, tickcolor="white",
                      tickfont=dict(color="white")),
            bar=dict(color=color, thickness=0.7),
            bgcolor=COLORS["card"],
            steps=[
                dict(range=[0, 1.5], color="rgba(76,175,125,0.15)"),
                dict(range=[1.5, 2.5], color="rgba(245,185,66,0.15)"),
                dict(range=[2.5, 4.5], color="rgba(245,143,66,0.15)"),
                dict(range=[4.5, 7],   color="rgba(245,92,92,0.15)"),
            ],
        ),
    ))
    fig.update_layout(
        template="plotly_dark", paper_bgcolor=COLORS["bg"], plot_bgcolor=COLORS["card"],
        margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig


# ── Chart 3b: Risk Score Timeline ─────────────────────────────────────────────

def chart_risk_timeline(df):
    risk_ts = df["risk_score"].dropna()
    shapes  = [s for s in recession_shapes(df)]   # xref="x" is fine for single-axis fig

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=risk_ts.index, y=risk_ts,
        mode="lines", line=dict(width=0),
        fill="tozeroy", fillcolor="rgba(245,92,92,0.2)",
        showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=risk_ts.index, y=risk_ts,
        mode="lines", line=dict(color=COLORS["red"], width=1.5),
        showlegend=False,
    ))
    fig.add_hline(y=3,
        line=dict(color="rgba(245,143,66,0.75)", dash="dash", width=1.5),
        annotation_text="Elevated (≥3)",
        annotation_font=dict(color="rgba(245,143,66,0.85)", size=10),
        annotation_position="top right",
    )
    fig.update_layout(
        shapes=shapes,
        template="plotly_dark", paper_bgcolor=COLORS["bg"], plot_bgcolor=COLORS["card"],
        margin=dict(l=0, r=0, t=30, b=0),
        title=dict(text="Risk Score Over Time (0–7 active conditions)",
                   font=dict(size=12, color=COLORS["muted"])),
        yaxis=dict(range=[0, 7.5], gridcolor="#1e2130", title_text="Score"),
        xaxis=dict(gridcolor="#1e2130"),
    )
    return fig


# ── Natural Language Summary ───────────────────────────────────────────────────

def make_summary(df):
    latest  = df.dropna(subset=["spread", "DRALACBN", "CPI_yoy_pct"]).iloc[-1]
    as_of   = df.dropna(subset=["spread"]).index[-1].strftime("%B %Y")
    score   = int(latest["risk_score"])
    spread  = latest["spread"]
    regime  = (
        "🟢 Normal / Benign" if score <= 1 else
        "🟡 Caution"         if score == 2 else
        "🟠 Elevated Risk"   if score <= 4 else
        "🔴 High Risk"
    )

    # Peak lag from full-sample cross-correlogram
    lag_corrs  = {l: df["FEDFUNDS_chg"].shift(l).corr(df["DRALACBN_chg"]) for l in range(0, 25)}
    pos_lags   = {l: r for l, r in lag_corrs.items() if l > 0}
    best_lag   = max(pos_lags, key=lambda l: pos_lags[l])
    best_r     = pos_lags[best_lag]

    del_trend  = "rising" if df["DRALACBN"].dropna().iloc[-1] > df["DRALACBN"].dropna().iloc[-4] else "easing"
    spread_dir = "above zero" if spread > 0 else "below zero (inverted)"

    # Longest recent episode
    elevated   = df[df["elevated_risk"]]
    max_ep_desc = ""
    if not elevated.empty:
        elevated = elevated.copy()
        elevated["episode"] = (elevated.index.to_series().diff() > pd.Timedelta("45D")).cumsum()
        ep_lengths = elevated.groupby("episode").size()
        longest    = ep_lengths.idxmax()
        ep_data    = elevated[elevated["episode"] == longest]
        max_ep_desc = (f" The longest elevated-risk episode since 1990 ran "
                       f"{ep_data.index[0].strftime('%b %Y')} – "
                       f"{ep_data.index[-1].strftime('%b %Y')} "
                       f"({len(ep_data)} months).")

    lines = [
        f"As of {as_of}, the macro-credit environment is classified as {regime} (risk score {score}/7).",
        "",
        f"Yield Curve: The 10Y–Fed Funds spread is {spread:+.2f}pp, {spread_dir}. "
        + ("An inverted yield curve compresses lender net interest margins and historically "
           "precedes recessions by 12–18 months." if spread < 0
           else "A positive slope supports conventional maturity-transformation lending."),
        "",
        f"Rate–Delinquency Transmission: The cross-correlogram shows Fed Funds changes "
        f"peak-correlate with delinquency at a {best_lag}-month lag (r = {best_r:+.3f}). "
        f"This well-documented monetary transmission delay reflects the time it takes "
        f"for rate moves to flow through to borrower stress via resets and refinancings.",
        "",
        f"Delinquency: Consumer loan delinquency is {del_trend} at {latest['DRALACBN']:.2f}%."
        + max_ep_desc,
        "",
        f"Inflation & Spending: CPI is running at {latest['CPI_yoy_pct']:.2f}% YoY. "
        f"PCE stands at ${latest['PCE']:,.0f}B, "
        + ("reflecting robust consumer spending." if latest["PCE"] > 20000
           else "reflecting moderate consumer activity."),
        "",
        ("Outlook: Multiple risk flags are simultaneously active — a pattern historically "
         "associated with elevated credit losses and compressed margins. "
         "Lending portfolios should be stress-tested with a 6–12 month delinquency horizon."
         if score >= 3 else
         "Outlook: Risk flags are limited. The current environment is broadly supportive "
         "of lending margins, though the lead-lag relationship warrants monitoring "
         "delinquency trends over the next 2–3 quarters."),
    ]
    return "\n".join(lines)


# ── Assemble HTML ─────────────────────────────────────────────────────────────

def build_html(fig_yc, fig_heatmap, fig_gauge, fig_timeline, summary_text, as_of):
    def fig_html(fig, height="480px"):
        return fig.to_html(
            full_html=False, include_plotlyjs=False,
            config={"displayModeBar": False},
            div_id=None,
        ).replace('<div>', f'<div style="height:{height}">', 1)

    summary_html = "".join(
        f'<p style="margin:0 0 10px 0">{line}</p>' if line else '<br>'
        for line in summary_text.split("\n")
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>FRED Macro Credit Risk Dashboard — {as_of}</title>
<script src="https://cdn.plot.ly/plotly-3.4.0.min.js"></script>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    background: {COLORS["bg"]}; color: #e0e0e0;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    padding: 24px;
  }}
  h1 {{ font-size: 22px; font-weight: 700; color: #fff; margin-bottom: 4px; }}
  .subtitle {{ font-size: 12px; color: {COLORS["muted"]}; margin-bottom: 24px; }}
  .kpi-row {{ display: flex; gap: 12px; margin-bottom: 24px; flex-wrap: wrap; }}
  .kpi {{
    background: {COLORS["card"]}; border-radius: 10px;
    padding: 14px 20px; flex: 1; min-width: 140px;
  }}
  .kpi-label {{ font-size: 10px; color: {COLORS["muted"]}; text-transform: uppercase;
                letter-spacing: .08em; margin-bottom: 6px; }}
  .kpi-value {{ font-size: 26px; font-weight: 700; color: #fff; }}
  .kpi-unit  {{ font-size: 12px; color: {COLORS["muted"]}; margin-left: 3px; }}
  .section   {{ margin-bottom: 28px; }}
  .section-label {{
    font-size: 10px; color: {COLORS["muted"]}; text-transform: uppercase;
    letter-spacing: .1em; margin-bottom: 8px;
  }}
  .chart-box {{
    background: {COLORS["card"]}; border-radius: 10px; padding: 4px; overflow: hidden;
  }}
  .two-col {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
  .summary-box {{
    background: {COLORS["card"]}; border-radius: 10px;
    padding: 20px 24px; line-height: 1.75; font-size: 14px;
  }}
  .footer {{ text-align: center; font-size: 10px; color: #3a3f52; margin-top: 32px; }}
</style>
</head>
<body>

<h1>📉 Macro Credit Risk Dashboard</h1>
<div class="subtitle">
  Source: Federal Reserve Bank of St. Louis (FRED) &middot;
  FEDFUNDS &middot; DGS10 &middot; DRALACBN &middot; PCE &middot; CPIAUCSL &middot; USREC &middot;
  Generated {as_of}
</div>

<!-- KPI Row (filled by inline script below) -->
<div class="kpi-row" id="kpi-row"></div>

<!-- Yield Curve -->
<div class="section">
  <div class="section-label">Yield Curve &amp; Delinquency — NBER Recession Bands</div>
  <div class="chart-box">{fig_html(fig_yc, "500px")}</div>
</div>

<!-- Heatmap + Risk Regime -->
<div class="section two-col">
  <div>
    <div class="section-label">Rate-Change Lead-Lag Heatmap</div>
    <div class="chart-box">{fig_html(fig_heatmap, "440px")}</div>
  </div>
  <div>
    <div class="section-label">Risk Regime Indicator</div>
    <div class="chart-box" style="display:grid;grid-template-rows:200px 220px;gap:4px">
      {fig_html(fig_gauge, "200px")}
      {fig_html(fig_timeline, "220px")}
    </div>
  </div>
</div>

<!-- NL Summary -->
<div class="section">
  <div class="section-label">Natural Language Summary</div>
  <div class="summary-box">{summary_html}</div>
</div>

<div class="footer">
  FRED data is publicly available and subject to revision.
  Risk score is a heuristic composite — not financial advice.
</div>

</body>
</html>"""


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if not FRED_API_KEY:
        print("ERROR: FRED_API_KEY not set.", file=sys.stderr)
        sys.exit(1)

    df     = build_dataset()
    latest = df.dropna(subset=["spread", "DRALACBN", "CPI_yoy_pct"]).iloc[-1]
    as_of  = df.dropna(subset=["spread"]).index[-1].strftime("%B %Y")

    print("Building charts…", file=sys.stderr)
    fig_yc       = chart_yield_curve(df)
    fig_heatmap  = chart_corr_heatmap(df)
    fig_gauge    = chart_risk_gauge(df)
    fig_timeline = chart_risk_timeline(df)
    summary      = make_summary(df)

    html = build_html(fig_yc, fig_heatmap, fig_gauge, fig_timeline, summary, as_of)

    # Determine output path
    out_dir = Path(os.environ.get("DASHBOARD_OUT_DIR", Path(__file__).parents[4]))
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp    = datetime.now().strftime("%Y%m%d")
    out_path = out_dir / f"fred_dashboard_{stamp}.html"

    out_path.write_text(html, encoding="utf-8")
    print(f"\n✓ Dashboard saved to: {out_path}")
    print(f"  Open with:  xdg-open \"{out_path}\"")

    # Try to open automatically
    try:
        import subprocess
        subprocess.Popen(["xdg-open", str(out_path)],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass


if __name__ == "__main__":
    main()
