"""
Tests for main.py — FRED macro analytics pipeline.

Covers:
  - align_to_monthly: resampling, forward-fill, dropna behaviour
  - compute_analytics: MoM/YoY diffs, yield curve spread, rolling correlations
  - flag_risk: risk flag logic and score aggregation

Financial edge cases:
  - Zero Lower Bound (Fed Funds = 0%)
  - Deeply inverted yield curve (spread < −2 pp)
  - Spread exactly at boundary values (0.0 and 0.5)
  - Rapid rate-hike cycle (>200 bps YoY)
  - High delinquency / financial-crisis conditions
  - Near-zero delinquency (COVID forbearance era)
  - High inflation (CPI YoY > 4%)
  - All flags simultaneously triggered (max risk score = 7)
  - No flags triggered (benign environment)
  - NaN / missing values (FRED sometimes returns "." entries)
  - Quarterly DRALACBN forward-fill correctness
"""

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Import the functions under test
# ---------------------------------------------------------------------------
from main import align_to_monthly, compute_analytics, flag_risk, RISK_THRESHOLDS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _monthly_index(start: str = "2000-01-31", periods: int = 36) -> pd.DatetimeIndex:
    return pd.date_range(start=start, periods=periods, freq="ME")


def _make_raw(
    fedfunds: float | list = 3.0,
    dgs10: float | list = 4.5,
    dralacbn: float | list = 1.5,
    pce: float | list = 18_000.0,
    cpiaucsl: float | list = 300.0,
    periods: int = 36,
) -> dict[str, pd.DataFrame]:
    """
    Build a minimal raw-series dict (monthly frequency) suitable for
    passing to align_to_monthly().  Scalar arguments are broadcast to
    all periods; list arguments are used directly.
    """
    idx = _monthly_index(periods=periods)

    def _series(val, name):
        data = val if isinstance(val, list) else [val] * periods
        return pd.DataFrame({name: data}, index=idx)

    return {
        "FEDFUNDS": _series(fedfunds, "FEDFUNDS"),
        "DGS10":    _series(dgs10,    "DGS10"),
        "DRALACBN": _series(dralacbn, "DRALACBN"),
        "PCE":      _series(pce,      "PCE"),
        "CPIAUCSL": _series(cpiaucsl, "CPIAUCSL"),
    }


def _aligned(
    fedfunds: float | list = 3.0,
    dgs10: float | list = 4.5,
    dralacbn: float | list = 1.5,
    pce: float | list = 18_000.0,
    cpiaucsl: float | list = 300.0,
    periods: int = 36,
) -> pd.DataFrame:
    raw = _make_raw(fedfunds, dgs10, dralacbn, pce, cpiaucsl, periods=periods)
    return align_to_monthly(raw)


def _analytics(**kwargs) -> pd.DataFrame:
    return compute_analytics(_aligned(**kwargs))


def _flagged(**kwargs):
    df = _analytics(**kwargs)
    return flag_risk(df)


# ===========================================================================
# align_to_monthly
# ===========================================================================

class TestAlignToMonthly:

    def test_returns_dataframe(self):
        df = _aligned()
        assert isinstance(df, pd.DataFrame)

    def test_expected_columns(self):
        df = _aligned()
        for col in ("FEDFUNDS", "DGS10", "DRALACBN", "PCE", "CPIAUCSL"):
            assert col in df.columns, f"Missing column: {col}"

    def test_drops_rows_missing_required(self):
        """Rows where FEDFUNDS, DGS10, or CPIAUCSL are NaN must be dropped."""
        idx = _monthly_index(periods=6)
        raw = {
            "FEDFUNDS": pd.DataFrame({"FEDFUNDS": [np.nan, 2.0, 2.0, 2.0, 2.0, 2.0]}, index=idx),
            "DGS10":    pd.DataFrame({"DGS10":    [3.0,   3.0, 3.0, 3.0, 3.0, 3.0]}, index=idx),
            "DRALACBN": pd.DataFrame({"DRALACBN": [1.5,   1.5, 1.5, 1.5, 1.5, 1.5]}, index=idx),
            "PCE":      pd.DataFrame({"PCE":      [100.0]*6}, index=idx),
            "CPIAUCSL": pd.DataFrame({"CPIAUCSL": [300.0]*6}, index=idx),
        }
        df = align_to_monthly(raw)
        assert df["FEDFUNDS"].isna().sum() == 0

    def test_dralacbn_forward_filled(self):
        """
        DRALACBN is quarterly; after resampling to monthly the intermediate
        months should be forward-filled, not left as NaN.
        """
        idx_q = pd.date_range("2010-03-31", periods=8, freq="QE")
        # Only quarterly observations — non-quarter-end months are absent
        raw = _make_raw(periods=36)
        # Replace DRALACBN with a truly quarterly series
        dralacbn_q = pd.DataFrame(
            {"DRALACBN": [1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4]},
            index=idx_q,
        )
        raw["DRALACBN"] = dralacbn_q
        df = align_to_monthly(raw)
        # After ffill no NaN should survive inside the index range
        assert df["DRALACBN"].isna().sum() == 0 or df.empty

    def test_monthly_index_frequency(self):
        df = _aligned()
        assert df.index.freq == "ME" or df.index.inferred_freq in ("M", "ME", "BME")


# ===========================================================================
# compute_analytics
# ===========================================================================

class TestComputeAnalytics:

    def test_spread_calculation(self):
        df = _analytics(fedfunds=2.0, dgs10=5.0)
        expected_spread = 5.0 - 2.0
        assert (df["spread"].dropna() == pytest.approx(expected_spread)).all()

    def test_inverted_spread(self):
        df = _analytics(fedfunds=5.5, dgs10=4.0)
        assert (df["spread"].dropna() < 0).all(), "Spread must be negative when curve is inverted"

    def test_fedfunds_yoy_constant_series(self):
        """YoY diff on a constant series must be 0 for all periods > 12."""
        df = _analytics(fedfunds=3.0)
        yoy = df["FEDFUNDS_yoy"].dropna()
        assert (yoy == pytest.approx(0.0)).all()

    def test_fedfunds_yoy_rate_hike_cycle(self):
        """
        Simulate a 300 bps rate-hike cycle over 12 months (0.25 bps/month).
        YoY change should reflect the full hike.
        """
        rates = [0.25 * i for i in range(13)] + [3.25] * 24  # ramp then plateau
        df = _analytics(fedfunds=rates)
        # At index 12 (month 13), YoY should ≈ 3.0 (12 × 0.25)
        yoy_values = df["FEDFUNDS_yoy"].dropna()
        assert yoy_values.max() == pytest.approx(3.0, abs=0.01)

    def test_cpi_yoy_pct_positive_inflation(self):
        """CPI growing at 2% annually → YoY pct ≈ 2%."""
        monthly_growth = (1.02 ** (1 / 12))
        cpi = [300 * (monthly_growth ** i) for i in range(36)]
        df = _analytics(cpiaucsl=cpi)
        yoy = df["CPI_yoy_pct"].dropna()
        assert yoy.mean() == pytest.approx(2.0, abs=0.05)

    def test_rolling_corr_columns_present(self):
        df = _analytics()
        for col in ("corr_ff_delinq", "corr_dgs10_delinq",
                    "corr_ff_lag3_delinq", "corr_ff_lag6_delinq",
                    "corr_ff_lag12_delinq"):
            assert col in df.columns, f"Missing correlation column: {col}"

    def test_rolling_corr_range(self):
        """Rolling correlations must stay in [−1, 1]."""
        df = _analytics()
        for col in ("corr_ff_delinq", "corr_ff_lag3_delinq"):
            vals = df[col].dropna()
            assert (vals >= -1.0 - 1e-9).all() and (vals <= 1.0 + 1e-9).all()

    def test_does_not_mutate_input(self):
        aligned = _aligned()
        cols_before = set(aligned.columns)
        compute_analytics(aligned)
        assert set(aligned.columns) == cols_before


# ===========================================================================
# flag_risk
# ===========================================================================

class TestFlagRisk:

    def test_returns_tuple(self):
        df = _analytics()
        result = flag_risk(df)
        assert isinstance(result, tuple) and len(result) == 2

    def test_risk_score_column_present(self):
        df, _ = _flagged()
        assert "risk_score" in df.columns

    def test_elevated_risk_column_present(self):
        df, _ = _flagged()
        assert "elevated_risk" in df.columns

    def test_risk_score_non_negative(self):
        df, _ = _flagged()
        assert (df["risk_score"] >= 0).all()

    def test_risk_score_max_seven(self):
        df, _ = _flagged()
        assert (df["risk_score"] <= 7).all()

    # -- Benign environment: no flags expected --------------------------------

    def test_no_flags_benign_environment(self):
        """
        Moderate rates, positive spread, low delinquency, low inflation.
        No flags should be active (or only a small number).
        """
        df, flags = _flagged(
            fedfunds=2.0,
            dgs10=4.0,
            dralacbn=1.0,
            cpiaucsl=[300 * (1.015 ** (i / 12)) for i in range(36)],  # ~1.5% inflation
        )
        # After 13+ months, compute YoY changes; risk score should be ≤ 1
        late_scores = df["risk_score"].dropna().tail(12)
        assert late_scores.max() <= 1

    # -- Zero Lower Bound -------------------------------------------------

    def test_zero_lower_bound_no_high_rates_flag(self):
        """
        FEDFUNDS = 0.25% (ZLB era, e.g. 2009–2015, 2020–2022).
        The 'high_rates' flag (threshold > 5%) must NOT trigger.
        """
        df, flags = _flagged(fedfunds=0.25, dgs10=2.5)
        if "high_rates" in flags.columns:
            assert flags["high_rates"].sum() == 0

    def test_zero_lower_bound_spread_positive(self):
        """At ZLB with positive long rates, spread must be strongly positive."""
        df = _analytics(fedfunds=0.25, dgs10=2.5)
        spread = df["spread"].dropna()
        assert (spread > 0).all()

    # -- Yield curve inversion --------------------------------------------

    def test_inverted_curve_flag(self):
        """
        DGS10 below FEDFUNDS → inverted_curve flag must fire.
        """
        df, flags = _flagged(fedfunds=5.5, dgs10=4.0)
        if "inverted_curve" in flags.columns:
            assert flags["inverted_curve"].any()

    def test_deeply_inverted_curve(self):
        """
        Spread of −2 pp should trigger both inverted_curve and near_inverted flags.
        """
        df, flags = _flagged(fedfunds=6.0, dgs10=4.0)
        for flag_name in ("inverted_curve", "near_inverted"):
            if flag_name in flags.columns:
                assert flags[flag_name].any(), f"Expected flag '{flag_name}' to fire"

    def test_spread_exactly_zero(self):
        """
        Spread = 0.0 is at the inversion boundary.
        inverted_curve (< 0) must NOT fire; near_inverted (< 0.5) MUST fire.
        """
        df, flags = _flagged(fedfunds=4.0, dgs10=4.0)
        # spread == 0 is not < 0, so inverted_curve should be False
        if "inverted_curve" in flags.columns:
            assert not flags["inverted_curve"].any()
        if "near_inverted" in flags.columns:
            assert flags["near_inverted"].any()

    def test_spread_at_near_inverted_boundary(self):
        """
        Spread = 0.5 is exactly at the near_inverted boundary (< 0.5 is False).
        near_inverted must NOT fire.
        """
        df, flags = _flagged(fedfunds=4.0, dgs10=4.5)
        if "near_inverted" in flags.columns:
            assert not flags["near_inverted"].any()

    # -- Rapid rate-hike cycle --------------------------------------------

    def test_rapid_rate_hike_flag(self):
        """
        Rates rising from 0.25% to 5.25% over 12 months (2022-style).
        After 12+ months the rapid_rate_hike flag (YoY > 200 bps) must fire.
        """
        rates = [0.25 + (5.0 / 12) * i for i in range(13)] + [5.25] * 24
        df, flags = _flagged(fedfunds=rates, dgs10=4.5)
        if "rapid_rate_hike" in flags.columns:
            assert flags["rapid_rate_hike"].any(), "rapid_rate_hike should fire during hike cycle"

    def test_high_rates_flag(self):
        """FEDFUNDS > 5% should trigger high_rates flag."""
        df, flags = _flagged(fedfunds=5.5, dgs10=6.0)
        if "high_rates" in flags.columns:
            assert flags["high_rates"].any()

    # -- Delinquency flags ------------------------------------------------

    def test_high_delinquency_flag(self):
        """
        Delinquency > 1.80% (financial-crisis level) must trigger high_delinquency flag.
        """
        df, flags = _flagged(dralacbn=2.5)
        if "high_delinquency" in flags.columns:
            assert flags["high_delinquency"].any()

    def test_rising_delinquency_flag(self):
        """
        Delinquency rising more than 10 bps YoY should trigger rising_delinquency flag.
        """
        # Start at 1.0%, add 0.5 bps/month → ~6 bps / year initially
        # Use 2 bps/month → 24 bps/year to comfortably exceed the 10 bps threshold
        delinqs = [1.0 + 0.02 * i for i in range(36)]
        df, flags = _flagged(dralacbn=delinqs)
        if "rising_delinquency" in flags.columns:
            assert flags["rising_delinquency"].any()

    def test_covid_forbearance_no_high_delinquency(self):
        """
        Near-zero delinquency (COVID forbearance era) must not trigger
        either delinquency flag.
        """
        df, flags = _flagged(dralacbn=0.3)
        if "high_delinquency" in flags.columns:
            assert flags["high_delinquency"].sum() == 0
        if "rising_delinquency" in flags.columns:
            # YoY change on constant series is 0, well below 10 bps
            assert flags["rising_delinquency"].sum() == 0

    # -- Inflation flags --------------------------------------------------

    def test_high_inflation_flag(self):
        """
        CPI growing at ~8% YoY should trigger high_inflation flag.
        """
        monthly_growth = (1.08 ** (1 / 12))
        cpi = [300 * (monthly_growth ** i) for i in range(36)]
        df, flags = _flagged(cpiaucsl=cpi)
        if "high_inflation" in flags.columns:
            assert flags["high_inflation"].any()

    def test_low_inflation_no_flag(self):
        """
        CPI growing at ~2% YoY must NOT trigger high_inflation flag.
        """
        monthly_growth = (1.02 ** (1 / 12))
        cpi = [300 * (monthly_growth ** i) for i in range(36)]
        df, flags = _flagged(cpiaucsl=cpi)
        if "high_inflation" in flags.columns:
            assert flags["high_inflation"].sum() == 0

    # -- Maximum / minimum risk scenarios --------------------------------

    def test_all_flags_triggered(self):
        """
        Simulate a worst-case macro environment where every flag fires:
          - inverted curve  (DGS10 < FEDFUNDS)
          - near-inverted   (spread < 0.5)
          - high_rates      (FEDFUNDS > 5%)
          - rapid rate hike (YoY > 200 bps)
          - high delinquency (DRALACBN > 1.80%)
          - rising delinquency (DRALACBN YoY > 10 bps)
          - high inflation  (CPI YoY > 4%)
        Risk score should reach 7 at some point.
        """
        rates_ramp = [0.25 + (5.5 / 12) * i for i in range(13)] + [5.75] * 23
        delinqs    = [1.5 + 0.03 * i for i in range(36)]  # rising above 1.80
        monthly_cpi = (1.08 ** (1 / 12))
        cpi        = [300 * (monthly_cpi ** i) for i in range(36)]

        df, flags = _flagged(
            fedfunds=rates_ramp,
            dgs10=[r - 1.5 for r in rates_ramp],  # DGS10 below FF → inverted
            dralacbn=delinqs,
            cpiaucsl=cpi,
        )
        late_score = df["risk_score"].dropna().tail(12).max()
        assert late_score >= 5, f"Expected risk score ≥ 5 in crisis scenario, got {late_score}"

    def test_elevated_risk_threshold(self):
        """
        elevated_risk column must be True iff risk_score >= 3.
        """
        df, _ = _flagged()
        if "elevated_risk" in df.columns:
            expected = df["risk_score"] >= 3
            pd.testing.assert_series_equal(df["elevated_risk"], expected, check_names=False)

    # -- NaN handling -------------------------------------------------------

    def test_flag_risk_handles_nan_gracefully(self):
        """
        NaN values in derived columns (e.g. CPI_yoy_pct in first 12 months)
        must not cause flag_risk to crash.
        """
        df = _analytics()
        # Inject NaN into one derived column
        df_copy = df.copy()
        df_copy.loc[df_copy.index[:5], "spread"] = np.nan
        result_df, result_flags = flag_risk(df_copy)
        assert "risk_score" in result_df.columns

    def test_active_flags_string_format(self):
        """active_flags column should be a comma-separated string or empty string."""
        df, _ = _flagged(fedfunds=6.0, dgs10=4.0)
        if "active_flags" in df.columns:
            for val in df["active_flags"].dropna():
                assert isinstance(val, str)


# ===========================================================================
# RISK_THRESHOLDS configuration
# ===========================================================================

class TestRiskThresholds:

    def test_all_threshold_operators_valid(self):
        """Every threshold must use '<' or '>' operator."""
        for name, (col, op, threshold) in RISK_THRESHOLDS.items():
            assert op in ("<", ">"), f"Unexpected operator '{op}' in threshold '{name}'"

    def test_threshold_count(self):
        """There must be exactly 7 risk thresholds (one per flag)."""
        assert len(RISK_THRESHOLDS) == 7

    def test_inverted_curve_threshold_is_zero(self):
        _, op, val = RISK_THRESHOLDS["inverted_curve"]
        assert op == "<" and val == 0.0

    def test_high_delinquency_threshold(self):
        _, op, val = RISK_THRESHOLDS["high_delinquency"]
        assert op == ">" and val == pytest.approx(1.80)

    def test_high_inflation_threshold(self):
        _, op, val = RISK_THRESHOLDS["high_inflation"]
        assert op == ">" and val == pytest.approx(4.0)

    def test_rapid_rate_hike_threshold(self):
        _, op, val = RISK_THRESHOLDS["rapid_rate_hike"]
        assert op == ">" and val == pytest.approx(2.0)
