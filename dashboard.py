import os
import requests
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import date

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Macro Credit Risk Dashboard",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="collapsed",
)

FRED_API_KEY  = os.environ.get("FRED_API_KEY", "")
FRED_BASE_URL = "https://api.stlouisfed.org/fred/series/observations"

COLORS = {
    "background":  "#0e1117",
    "card":        "#1a1d27",
    "accent_blue": "#4c8bf5",
    "accent_red":  "#f55c5c",
    "accent_green":"#4caf7d",
    "accent_gold": "#f5b942",
    "recession":   "rgba(200, 80, 80, 0.15)",
    "inversion":   "rgba(200, 80, 80, 0.25)",
    "text_muted":  "#8b92a5",
}

# ── Data layer ────────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner="Fetching FRED data…")
def fetch_series(series_id: str, start: str = "1990-01-01") -> pd.DataFrame:
    params = {
        "series_id": series_id,
        "api_key": FRED_API_KEY,
        "file_type": "json",
        "observation_start": start,
        "sort_order": "asc",
    }
    r = requests.get(FRED_BASE_URL, params=params, timeout=15)
    r.raise_for_status()
    obs = r.json()["observations"]
    df = pd.DataFrame(obs)[["date", "value"]]
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df.set_index("date").rename(columns={"value": series_id})


@st.cache_data(ttl=3600, show_spinner="Aligning series…")
def build_dataset() -> pd.DataFrame:
    raw = {sid: fetch_series(sid) for sid in
           ["FEDFUNDS", "DGS10", "DRALACBN", "PCE", "CPIAUCSL", "USREC"]}

    df = pd.concat([
        raw["FEDFUNDS"].resample("ME").last(),
        raw["DGS10"].resample("ME").mean(),
        raw["DRALACBN"].resample("ME").last().ffill(),
        raw["PCE"].resample("ME").last(),
        raw["CPIAUCSL"].resample("ME").last(),
        raw["USREC"].resample("ME").max(),
    ], axis=1).dropna(subset=["FEDFUNDS", "DGS10", "CPIAUCSL"])

    # Derived metrics
    df["spread"]        = df["DGS10"] - df["FEDFUNDS"]
    df["FEDFUNDS_chg"]  = df["FEDFUNDS"].diff()
    df["DGS10_chg"]     = df["DGS10"].diff()
    df["DRALACBN_chg"]  = df["DRALACBN"].diff()
    df["FEDFUNDS_yoy"]  = df["FEDFUNDS"].diff(12)
    df["CPI_yoy_pct"]   = df["CPIAUCSL"].pct_change(12) * 100
    df["PCE_yoy_pct"]   = df["PCE"].pct_change(12) * 100
    df["DRALACBN_yoy"]  = df["DRALACBN"].diff(12)

    # Rolling 18m correlations at each lag
    ROLL = 18
    for lag in range(0, 25, 1):
        df[f"corr_lag{lag}"] = (
            df["FEDFUNDS_chg"].shift(lag)
            .rolling(ROLL)
            .corr(df["DRALACBN_chg"])
        )

    # Risk scoring
    df["f_inverted"]    = (df["spread"] < 0).astype(int)
    df["f_near_inv"]    = (df["spread"] < 0.5).astype(int)
    df["f_rising_del"]  = (df["DRALACBN_yoy"] > 0.10).astype(int)
    df["f_high_del"]    = (df["DRALACBN"] > 1.80).astype(int)
    df["f_inflation"]   = (df["CPI_yoy_pct"] > 4.0).astype(int)
    df["f_rate_hike"]   = (df["FEDFUNDS_yoy"] > 2.0).astype(int)
    df["f_high_rates"]  = (df["FEDFUNDS"] > 5.0).astype(int)
    df["risk_score"]    = df[["f_inverted", "f_near_inv", "f_rising_del",
                               "f_high_del", "f_inflation", "f_rate_hike",
                               "f_high_rates"]].sum(axis=1)
    return df


def recession_shapes(df: pd.DataFrame) -> list[dict]:
    """Build Plotly shape rectangles for NBER recession periods."""
    if "USREC" not in df.columns:
        return []
    in_rec   = False
    start    = None
    shapes   = []
    for dt, row in df["USREC"].items():
        if row == 1 and not in_rec:
            in_rec, start = True, dt
        elif row == 0 and in_rec:
            shapes.append(dict(
                type="rect", xref="x", yref="paper",
                x0=start, x1=dt, y0=0, y1=1,
                fillcolor=COLORS["recession"], line_width=0, layer="below",
            ))
            in_rec = False
    if in_rec:
        shapes.append(dict(
            type="rect", xref="x", yref="paper",
            x0=start, x1=df.index[-1], y0=0, y1=1,
            fillcolor=COLORS["recession"], line_width=0, layer="below",
        ))
    return shapes


def classify_regime(df: pd.DataFrame) -> dict:
    row   = df.dropna(subset=["spread", "DRALACBN", "CPI_yoy_pct"]).iloc[-1]
    score = int(row["risk_score"])
    spread_val = row["spread"]
    delinq_val = row["DRALACBN"]
    cpi_val    = row["CPI_yoy_pct"]
    ff_val     = row["FEDFUNDS"]
    dgs10_val  = row["DGS10"]

    if score <= 1:
        regime, color, emoji = "Normal / Benign", "#4caf7d", "🟢"
    elif score == 2:
        regime, color, emoji = "Caution", "#f5b942", "🟡"
    elif score <= 4:
        regime, color, emoji = "Elevated Risk", "#f58f42", "🟠"
    else:
        regime, color, emoji = "High Risk", "#f55c5c", "🔴"

    flags = []
    if row["f_inverted"]:  flags.append("Inverted yield curve")
    if row["f_near_inv"] and not row["f_inverted"]: flags.append("Near-flat yield curve")
    if row["f_rising_del"]: flags.append("Rising delinquencies YoY")
    if row["f_high_del"]:   flags.append("Delinquency rate elevated")
    if row["f_inflation"]:  flags.append("Inflation above 4%")
    if row["f_rate_hike"]:  flags.append("Rapid rate hike cycle")
    if row["f_high_rates"]: flags.append("High absolute rates (>5%)")

    return dict(
        regime=regime, color=color, emoji=emoji, score=score,
        flags=flags, spread=spread_val, delinq=delinq_val,
        cpi=cpi_val, ff=ff_val, dgs10=dgs10_val, row=row,
    )


def make_nl_summary(df: pd.DataFrame, regime: dict) -> str:
    latest     = df.dropna(subset=["spread","DRALACBN","CPI_yoy_pct"]).iloc[-1]
    as_of      = df.dropna(subset=["spread"]).index[-1].strftime("%B %Y")
    spread_dir = "above" if regime["spread"] > 0 else "below"

    # Lag with peak correlation (full-sample)
    lag_corrs  = {lag: df["FEDFUNDS_chg"].shift(lag).corr(df["DRALACBN_chg"])
                  for lag in range(0, 25)}
    best_lag   = max(lag_corrs, key=lambda l: lag_corrs[l])
    best_corr  = lag_corrs[best_lag]

    # Recent spread trajectory
    spread_3m  = df["spread"].dropna().tail(3)
    spread_trend = "steepening" if spread_3m.iloc[-1] > spread_3m.iloc[0] else "flattening"

    # Delinquency trajectory
    del_3m = df["DRALACBN"].dropna().tail(4)
    del_trend = "rising" if del_3m.iloc[-1] > del_3m.iloc[0] else "easing"

    summary = f"""
**As of {as_of}**, the macro-credit environment is classified as **{regime["emoji"]} {regime["regime"]}**
(risk score {regime["score"]}/7).

**Yield Curve:** The 10Y–Fed Funds spread stands at **{regime["spread"]:+.2f}pp**,
{spread_dir} zero and currently {spread_trend}.
{"An inverted yield curve has historically preceded recessions by 12–18 months and compresses net interest margins for lenders." if regime["spread"] < 0 else "A positive slope supports lender profitability through conventional maturity transformation."}

**Delinquencies:** Consumer loan delinquency is **{regime["delinq"]:.2f}%**, with a {del_trend} trend.
The cross-correlogram reveals that Fed Funds rate changes lead delinquency changes by approximately
**{best_lag} months** (r = {best_corr:+.3f}), consistent with the well-documented monetary
transmission lag — borrowers feel the pinch of higher rates only after resets, refinancings, and
income pressure accumulate.

**Inflation:** CPI is running at **{regime["cpi"]:.2f}% YoY**,
{"above the Fed's 2% target, maintaining pressure to hold rates higher for longer." if regime["cpi"] > 2.5 else "near the Fed's 2% target, reducing pressure for further tightening."}

**Outlook for Lending Businesses:**
{"Multiple risk flags are active simultaneously — a pattern historically associated with elevated credit losses and compressed margins. Lending businesses should stress-test portfolios for delinquency increases with a 6–12 month horizon." if regime["score"] >= 3 else "Risk flags are limited. The current environment is broadly supportive of lending margins, though the lead-lag relationship suggests monitoring delinquency trends over the next two to three quarters."}
"""
    return summary.strip()


# ── Charts ────────────────────────────────────────────────────────────────────

def chart_yield_curve(df: pd.DataFrame) -> go.Figure:
    shapes = recession_shapes(df)

    # Shade inverted periods in a different color
    in_inv   = False
    inv_start = None
    for dt, val in df["spread"].items():
        if pd.isna(val):
            continue
        if val < 0 and not in_inv:
            in_inv, inv_start = True, dt
        elif val >= 0 and in_inv:
            shapes.append(dict(
                type="rect", xref="x", yref="paper",
                x0=inv_start, x1=dt, y0=0, y1=1,
                fillcolor=COLORS["inversion"], line_width=0, layer="below",
            ))
            in_inv = False

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.65, 0.35],
        vertical_spacing=0.04,
    )

    # Top: FF rate, DGS10, spread
    fig.add_trace(go.Scatter(
        x=df.index, y=df["DGS10"], name="10Y Treasury",
        line=dict(color=COLORS["accent_blue"], width=1.8),
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=df.index, y=df["FEDFUNDS"], name="Fed Funds",
        line=dict(color=COLORS["accent_gold"], width=1.8),
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=df.index, y=df["spread"], name="Spread (10Y−FF)",
        line=dict(color=COLORS["accent_green"], width=2),
        fill="tozeroy",
        fillcolor="rgba(76,175,125,0.12)",
    ), row=1, col=1)

    # Zero line for spread
    fig.add_hline(y=0, row=1, col=1,
                  line=dict(color="rgba(255,255,255,0.25)", dash="dot", width=1))

    # Bottom: delinquency rate
    fig.add_trace(go.Scatter(
        x=df.index, y=df["DRALACBN"], name="Delinquency Rate",
        line=dict(color=COLORS["accent_red"], width=1.8),
        fill="tozeroy", fillcolor="rgba(245,92,92,0.12)",
    ), row=2, col=1)

    fig.update_layout(
        shapes=shapes,
        template="plotly_dark",
        paper_bgcolor=COLORS["background"],
        plot_bgcolor=COLORS["card"],
        legend=dict(orientation="h", y=1.06, x=0, font=dict(size=11)),
        margin=dict(l=0, r=0, t=10, b=0),
        hovermode="x unified",
    )
    fig.update_yaxes(row=1, col=1, title_text="Rate (%)", gridcolor="#1e2130")
    fig.update_yaxes(row=2, col=1, title_text="Delinquency (%)", gridcolor="#1e2130")
    fig.update_xaxes(gridcolor="#1e2130")
    return fig


def chart_corr_heatmap(df: pd.DataFrame) -> go.Figure:
    lags    = list(range(0, 25))
    cols    = [f"corr_lag{l}" for l in lags]
    heat_df = df[cols].dropna()

    # Downsample to quarterly for readability
    heat_q  = heat_df.resample("QE").mean()

    z = heat_q[cols].T.values   # shape: (lags, time)
    x = [d.strftime("%Y-Q%q") for d in heat_q.index]
    y = [f"{l}m" for l in lags]

    fig = go.Figure(go.Heatmap(
        z=z, x=x, y=y,
        colorscale=[
            [0.0,  "#2563eb"],
            [0.35, "#1e3a6b"],
            [0.5,  "#1a1d27"],
            [0.65, "#7a1f1f"],
            [1.0,  "#ef4444"],
        ],
        zmid=0,
        zmin=-0.8, zmax=0.8,
        colorbar=dict(
            title="Corr", thickness=12,
            tickfont=dict(size=10, color="white"),
            titlefont=dict(color="white"),
        ),
        hoverongaps=False,
        hovertemplate="Quarter: %{x}<br>Lag: %{y}<br>Correlation: %{z:.3f}<extra></extra>",
    ))

    # Annotate the "sweet spot" band at 6-12 months
    fig.add_hrect(
        y0="6m", y1="12m",
        fillcolor="rgba(255,255,255,0.04)",
        line=dict(color="rgba(255,255,255,0.3)", width=1, dash="dot"),
        annotation_text="6–12m lag window",
        annotation_position="top right",
        annotation_font=dict(color="rgba(255,255,255,0.5)", size=10),
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=COLORS["background"],
        plot_bgcolor=COLORS["card"],
        xaxis=dict(
            title="Quarter", tickangle=-45,
            tickfont=dict(size=9), gridcolor="#1e2130",
            nticks=20,
        ),
        yaxis=dict(
            title="Rate Change Lag", autorange="reversed",
            gridcolor="#1e2130",
        ),
        margin=dict(l=0, r=0, t=10, b=60),
    )
    return fig


def chart_cross_correlogram(df: pd.DataFrame) -> go.Figure:
    lags = list(range(-6, 25))
    corrs = []
    for lag in lags:
        if lag < 0:
            # Negative lag: delinquency leads rates
            r = df["FEDFUNDS_chg"].corr(df["DRALACBN_chg"].shift(-lag))
        else:
            r = df["FEDFUNDS_chg"].shift(lag).corr(df["DRALACBN_chg"])
        corrs.append(r)

    colors = [COLORS["accent_red"] if c > 0 else COLORS["accent_blue"] for c in corrs]

    fig = go.Figure(go.Bar(
        x=lags, y=corrs,
        marker_color=colors,
        hovertemplate="Lag: %{x}m<br>Correlation: %{y:.3f}<extra></extra>",
        name="Correlation",
    ))

    # Highlight peak positive lag
    pos_corrs = [(l, c) for l, c in zip(lags, corrs) if l >= 0]
    peak_lag, peak_corr = max(pos_corrs, key=lambda x: x[1])
    fig.add_annotation(
        x=peak_lag, y=peak_corr,
        text=f"Peak: lag {peak_lag}m<br>r={peak_corr:.3f}",
        showarrow=True, arrowhead=2, arrowcolor="white",
        font=dict(color="white", size=10),
        bgcolor="rgba(0,0,0,0.7)", bordercolor="white",
        borderwidth=1,
    )
    fig.add_vline(x=0, line=dict(color="rgba(255,255,255,0.3)", dash="dot"))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=COLORS["background"],
        plot_bgcolor=COLORS["card"],
        xaxis=dict(
            title="Months (+ = rate change leads delinquency)",
            gridcolor="#1e2130", zeroline=False,
        ),
        yaxis=dict(title="Pearson r", gridcolor="#1e2130"),
        margin=dict(l=0, r=0, t=10, b=0),
        showlegend=False,
    )
    return fig


def estimate_bp_impact(df: pd.DataFrame, bp_change: float = 50.0) -> dict:
    """
    OLS regression at lags 0,3,6,9,12m: DRALACBN_chg = a + b*FEDFUNDS_chg(lag).
    Returns dict keyed by lag with beta, r_squared, estimated_impact_bps, ci bounds.
    """
    rate_change = bp_change / 100.0
    results = {}
    for lag in (0, 3, 6, 9, 12):
        sub = df[["FEDFUNDS_chg", "DRALACBN_chg"]].dropna().copy()
        sub["FF_lagged"] = sub["FEDFUNDS_chg"].shift(lag)
        sub = sub[["FF_lagged", "DRALACBN_chg"]].dropna()
        if len(sub) < 24:
            continue
        x = sub["FF_lagged"].values
        y = sub["DRALACBN_chg"].values
        n = len(x)
        x_mean, y_mean = x.mean(), y.mean()
        ss_xx = ((x - x_mean) ** 2).sum()
        ss_xy = ((x - x_mean) * (y - y_mean)).sum()
        if ss_xx == 0:
            continue
        beta = ss_xy / ss_xx
        intercept = y_mean - beta * x_mean
        y_pred = intercept + beta * x
        ss_res = ((y - y_pred) ** 2).sum()
        ss_tot = ((y - y_mean) ** 2).sum()
        r_squared = 1.0 - ss_res / ss_tot if ss_tot != 0 else 0.0
        se_beta = np.sqrt(ss_res / (n - 2) / ss_xx) if n > 2 else np.nan
        estimated = beta * rate_change
        ci_lo = (beta - 1.96 * se_beta) * rate_change
        ci_hi = (beta + 1.96 * se_beta) * rate_change
        results[lag] = dict(
            beta=beta, r_squared=r_squared, n_obs=n,
            estimated_impact_bps=estimated * 100,
            ci_lower_bps=ci_lo * 100,
            ci_upper_bps=ci_hi * 100,
        )
    return results


def chart_bp_impact(results: dict, bp_change: float = 50.0) -> go.Figure:
    """Bar chart of estimated delinquency impact per lag with 95% CI error bars."""
    lags = sorted(results.keys())
    impacts = [results[l]["estimated_impact_bps"] for l in lags]
    ci_lo   = [results[l]["ci_lower_bps"] for l in lags]
    ci_hi   = [results[l]["ci_upper_bps"] for l in lags]
    r2      = [results[l]["r_squared"] for l in lags]

    err_minus = [imp - lo for imp, lo in zip(impacts, ci_lo)]
    err_plus  = [hi - imp for imp, hi in zip(impacts, ci_hi)]
    bar_colors = [COLORS["accent_red"] if v > 0 else COLORS["accent_blue"] for v in impacts]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[f"{l}m lag" for l in lags],
        y=impacts,
        error_y=dict(type="data", symmetric=False,
                     array=err_plus, arrayminus=err_minus,
                     color="rgba(255,255,255,0.5)", thickness=1.5, width=6),
        marker_color=bar_colors,
        text=[f"R²={r:.3f}" for r in r2],
        textposition="outside",
        textfont=dict(size=10, color="white"),
        hovertemplate=(
            "Lag: %{x}<br>"
            "Est. impact: %{y:+.2f} bps<br>"
            "<extra></extra>"
        ),
        name=f"+{bp_change:.0f}bp shock",
    ))
    fig.add_hline(y=0, line=dict(color="rgba(255,255,255,0.3)", dash="dot", width=1))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=COLORS["background"],
        plot_bgcolor=COLORS["card"],
        xaxis=dict(title="Rate Change Lag", gridcolor="#1e2130"),
        yaxis=dict(title="Estimated Δ Delinquency (bps)", gridcolor="#1e2130"),
        margin=dict(l=0, r=0, t=10, b=0),
        showlegend=False,
    )
    return fig


def chart_risk_gauge(score: int) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number=dict(font=dict(size=36, color="white"), suffix="/7"),
        gauge=dict(
            axis=dict(range=[0, 7], tickwidth=1, tickcolor="white",
                      tickfont=dict(color="white")),
            bar=dict(color=(
                "#4caf7d" if score <= 1 else
                "#f5b942" if score == 2 else
                "#f58f42" if score <= 4 else
                "#f55c5c"
            ), thickness=0.7),
            bgcolor=COLORS["card"],
            steps=[
                dict(range=[0, 1.5], color="rgba(76,175,125,0.15)"),
                dict(range=[1.5, 2.5], color="rgba(245,185,66,0.15)"),
                dict(range=[2.5, 4.5], color="rgba(245,143,66,0.15)"),
                dict(range=[4.5, 7],   color="rgba(245,92,92,0.15)"),
            ],
            threshold=dict(
                line=dict(color="white", width=2),
                thickness=0.8, value=score,
            ),
        ),
    ))
    fig.update_layout(
        paper_bgcolor=COLORS["background"],
        font=dict(color="white"),
        height=220,
        margin=dict(l=20, r=20, t=30, b=0),
    )
    return fig


# ── Layout ────────────────────────────────────────────────────────────────────

def main():
    # Header
    st.markdown("""
    <style>
        .block-container { padding-top: 1.5rem; padding-bottom: 1rem; }
        .metric-card {
            background: #1a1d27; border-radius: 10px;
            padding: 14px 18px; text-align: center;
        }
        .metric-label { font-size: 11px; color: #8b92a5; text-transform: uppercase;
                        letter-spacing: .08em; margin-bottom: 4px; }
        .metric-value { font-size: 26px; font-weight: 700; color: white; }
        .metric-delta { font-size: 12px; margin-top: 3px; }
        .section-label { font-size: 11px; color: #8b92a5; text-transform: uppercase;
                         letter-spacing: .1em; margin-bottom: 6px; }
        .flag-pill {
            display: inline-block; padding: 3px 10px; border-radius: 20px;
            font-size: 11px; margin: 3px 3px 3px 0;
            background: rgba(245,92,92,0.2); color: #f5b0b0;
        }
    </style>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns([3, 1])
    with c1:
        st.markdown("## 📉 Macro Credit Risk Dashboard")
    with c2:
        st.markdown(
            f"<div style='text-align:right;color:#8b92a5;font-size:12px;padding-top:12px'>"
            f"Data via FRED · as of {date.today().strftime('%b %d, %Y')}</div>",
            unsafe_allow_html=True,
        )

    df     = build_dataset()
    regime = classify_regime(df)
    latest = df.dropna(subset=["spread", "DRALACBN", "CPI_yoy_pct"]).iloc[-1]
    prev   = df.dropna(subset=["spread", "DRALACBN", "CPI_yoy_pct"]).iloc[-2]

    # ── KPI row ───────────────────────────────────────────────────────────────
    st.markdown("")
    k1, k2, k3, k4, k5, k6 = st.columns(6)

    def kpi(col, label, value, fmt, delta=None, delta_label="MoM"):
        d_html = ""
        if delta is not None:
            color = "#4caf7d" if delta >= 0 else "#f55c5c"
            sign  = "▲" if delta >= 0 else "▼"
            d_html = f"<div class='metric-delta' style='color:{color}'>{sign} {abs(delta):{fmt}} {delta_label}</div>"
        col.markdown(
            f"<div class='metric-card'>"
            f"<div class='metric-label'>{label}</div>"
            f"<div class='metric-value'>{value:{fmt}}</div>"
            f"{d_html}</div>",
            unsafe_allow_html=True,
        )

    kpi(k1, "Fed Funds Rate",    latest["FEDFUNDS"],    ".2f",
        latest["FEDFUNDS"] - prev["FEDFUNDS"])
    kpi(k2, "10Y Treasury",      latest["DGS10"],       ".2f",
        latest["DGS10"] - prev["DGS10"])
    kpi(k3, "Yield Curve Spread",latest["spread"],      "+.2f",
        latest["spread"] - prev["spread"])
    kpi(k4, "Delinquency Rate",  latest["DRALACBN"],    ".2f",
        latest["DRALACBN"] - prev["DRALACBN"])
    kpi(k5, "CPI YoY %",         latest["CPI_yoy_pct"], ".2f",
        latest["CPI_yoy_pct"] - prev["CPI_yoy_pct"])

    # Risk regime badge
    flags_html = "".join(f"<span class='flag-pill'>{f}</span>" for f in regime["flags"]) \
                 or "<span style='color:#8b92a5;font-size:12px'>No active risk flags</span>"
    k6.markdown(
        f"<div class='metric-card' style='border:1px solid {regime['color']}44;'>"
        f"<div class='metric-label'>Risk Regime</div>"
        f"<div class='metric-value' style='color:{regime['color']};font-size:20px'>"
        f"{regime['emoji']} {regime['regime']}</div>"
        f"<div style='margin-top:6px'>{flags_html}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Main charts ───────────────────────────────────────────────────────────
    top_l, top_r = st.columns([2, 1])

    with top_l:
        st.markdown("<div class='section-label'>Yield Curve & Delinquency — with NBER Recession Bands</div>",
                    unsafe_allow_html=True)
        st.plotly_chart(chart_yield_curve(df), use_container_width=True, config={"displayModeBar": False})

    with top_r:
        st.markdown("<div class='section-label'>Risk Score Gauge</div>", unsafe_allow_html=True)
        st.plotly_chart(chart_risk_gauge(regime["score"]), use_container_width=True,
                        config={"displayModeBar": False})

        st.markdown("<div class='section-label' style='margin-top:8px'>Fed Funds → Delinquency Cross-Correlogram</div>",
                    unsafe_allow_html=True)
        st.markdown(
            "<div style='color:#8b92a5;font-size:11px;margin-bottom:6px'>"
            "Full-sample Pearson r at each lag. Positive lags = rate changes precede delinquency.</div>",
            unsafe_allow_html=True,
        )
        st.plotly_chart(chart_cross_correlogram(df), use_container_width=True,
                        config={"displayModeBar": False})

    # ── Heatmap ───────────────────────────────────────────────────────────────
    st.markdown("<div class='section-label'>Rolling 18-Month Correlation Heatmap — Rate Change Lag vs. Delinquency Change</div>",
                unsafe_allow_html=True)
    st.markdown(
        "<div style='color:#8b92a5;font-size:11px;margin-bottom:8px'>"
        "Each cell = rolling 18-month Pearson r between Fed Funds changes (lagged N months) "
        "and delinquency changes. Red = positive (rate hikes predict future delinquency rise). "
        "The 6–12m band historically shows the strongest signal.</div>",
        unsafe_allow_html=True,
    )
    st.plotly_chart(chart_corr_heatmap(df), use_container_width=True,
                    config={"displayModeBar": False})

    # ── NL Summary ────────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='section-label'>Natural Language Summary</div>", unsafe_allow_html=True)
    st.markdown(
        f"<div style='background:{COLORS['card']};border-radius:10px;padding:20px 24px;"
        f"border-left:3px solid {regime['color']};line-height:1.7;font-size:14px'>"
        + make_nl_summary(df, regime).replace("\n\n", "</p><p>").replace("\n", "<br>")
        + "</div>",
        unsafe_allow_html=True,
    )

    # ── 50bp Impact Estimation ────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='section-label'>+50 Basis Point Rate Shock — Estimated Delinquency Impact</div>",
                unsafe_allow_html=True)
    st.markdown(
        "<div style='color:#8b92a5;font-size:11px;margin-bottom:8px'>"
        "OLS regression of monthly delinquency changes on lagged Fed Funds changes "
        "(1990–present). Error bars = 95% confidence interval. R² shown above each bar.</div>",
        unsafe_allow_html=True,
    )

    bp_results = estimate_bp_impact(df, bp_change=50.0)
    if bp_results:
        bp_left, bp_right = st.columns([2, 1])
        with bp_left:
            st.plotly_chart(chart_bp_impact(bp_results, bp_change=50.0),
                            use_container_width=True, config={"displayModeBar": False})

        with bp_right:
            best_lag = max(bp_results, key=lambda l: bp_results[l]["r_squared"])
            best = bp_results[best_lag]
            st.markdown(
                f"<div style='background:{COLORS[\"card\"]};border-radius:10px;"
                f"padding:16px 20px;border-left:3px solid {COLORS[\"accent_red\"]};'>"
                f"<div class='metric-label'>Best-fit lag</div>"
                f"<div class='metric-value' style='font-size:22px'>{best_lag} months</div>"
                f"<div style='color:#8b92a5;font-size:12px;margin-top:8px'>"
                f"R² = {best['r_squared']:.3f}</div>"
                f"<hr style='border-color:#2a2d3a;margin:12px 0'>"
                f"<div class='metric-label'>Estimated Δ Delinquency</div>"
                f"<div class='metric-value' style='font-size:26px;color:{COLORS[\"accent_red\"]}'>"
                f"{best['estimated_impact_bps']:+.2f} bps</div>"
                f"<div style='color:#8b92a5;font-size:11px;margin-top:4px'>"
                f"95% CI: {best['ci_lower_bps']:+.2f} to {best['ci_upper_bps']:+.2f} bps</div>"
                f"<hr style='border-color:#2a2d3a;margin:12px 0'>"
                f"<div style='color:#8b92a5;font-size:11px;line-height:1.6'>"
                f"A +50bp Fed Funds rate increase is historically associated with a "
                f"<b style='color:white'>{best['estimated_impact_bps']:+.2f}bps</b> change in the "
                f"consumer loan delinquency rate ~{best_lag} months later. "
                f"Estimates reflect full-sample averages and will vary across credit regimes.</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

            st.markdown("<br>", unsafe_allow_html=True)
            rows = []
            for lag in sorted(bp_results.keys()):
                r = bp_results[lag]
                rows.append({
                    "Lag": f"{lag}m",
                    "β (pp/pp)": f"{r['beta']:.4f}",
                    "R²": f"{r['r_squared']:.3f}",
                    "Est. Impact": f"{r['estimated_impact_bps']:+.2f} bps",
                    "N": r["n_obs"],
                })
            st.dataframe(rows, use_container_width=True, hide_index=True)

    st.markdown(
        "<div style='color:#3a3f52;font-size:10px;text-align:center;margin-top:24px'>"
        "Source: Federal Reserve Bank of St. Louis (FRED) · "
        "FEDFUNDS · DGS10 · DRALACBN · PCE · CPIAUCSL · USREC</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
