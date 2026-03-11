"""
Test suite for main.py — financial analytics and risk-flagging logic.

Focus areas
-----------
- compute_analytics : derived metrics (spread, MoM/YoY changes, rolling corrs)
- flag_risk         : all seven threshold boundary conditions (strict inequalities)
- align_to_monthly  : resampling output shape and column completeness
"""
import numpy as np
import pandas as pd
import pytest

from main import (
    RISK_THRESHOLDS,
    align_to_monthly,
    compute_analytics,
    flag_risk,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _monthly_df(n: int = 36, **overrides) -> pd.DataFrame:
    """
    Return a month-end indexed DataFrame with sensible defaults for all
    columns expected by compute_analytics / flag_risk.

    Pass keyword arguments to override specific column values (scalar or list).
    """
    dates = pd.date_range("2020-01-31", periods=n, freq="ME")
    data = {
        "FEDFUNDS": 2.0,
        "DGS10":    3.0,
        "DRALACBN": 1.5,
        "PCE":      15_000.0,
        "CPIAUCSL": 260.0,
    }
    data.update(overrides)

    # Expand scalars to lists
    expanded = {k: ([v] * n if np.isscalar(v) else v) for k, v in data.items()}
    return pd.DataFrame(expanded, index=dates)


def _analytics_df(n: int = 36, **overrides) -> pd.DataFrame:
    """Return a DataFrame that has already passed through compute_analytics."""
    return compute_analytics(_monthly_df(n, **overrides))


def _flag_df(**col_overrides) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (df_with_flags, flags) ready for assertion."""
    return flag_risk(_analytics_df(**col_overrides))


# ── align_to_monthly ───────────────────────────────────────────────────────────

class TestAlignToMonthly:
    def _make_raw(self, months: int = 24) -> dict[str, pd.DataFrame]:
        """Produce a minimal raw dict mirroring what fetch_all() returns."""
        monthly_idx   = pd.date_range("2020-01-01", periods=months, freq="MS")
        quarterly_idx = pd.date_range("2020-01-01", periods=months // 3 + 1, freq="QS")
        daily_idx     = pd.date_range("2020-01-01", periods=months * 30, freq="D")

        return {
            "FEDFUNDS": pd.DataFrame({"FEDFUNDS": 2.0},  index=monthly_idx),
            "DGS10":    pd.DataFrame({"DGS10":    3.0},  index=daily_idx),
            "DRALACBN": pd.DataFrame({"DRALACBN": 1.5},  index=quarterly_idx),
            "PCE":      pd.DataFrame({"PCE":      15_000.0}, index=monthly_idx),
            "CPIAUCSL": pd.DataFrame({"CPIAUCSL": 260.0},    index=monthly_idx),
        }

    def test_returns_dataframe(self):
        result = align_to_monthly(self._make_raw())
        assert isinstance(result, pd.DataFrame)

    def test_has_all_required_columns(self):
        result = align_to_monthly(self._make_raw())
        for col in ("FEDFUNDS", "DGS10", "DRALACBN", "PCE", "CPIAUCSL"):
            assert col in result.columns, f"Missing column: {col}"

    def test_index_is_month_end(self):
        """All index dates should fall on month-end (MonthEnd offset)."""
        result = align_to_monthly(self._make_raw())
        for dt in result.index:
            # Month-end dates satisfy: dt + 1 day is in the next month
            assert (dt + pd.Timedelta(days=1)).month != dt.month or dt.month == 12, (
                f"Date {dt} is not a month-end"
            )

    def test_no_nan_in_required_columns(self):
        """After dropna on FEDFUNDS/DGS10/CPIAUCSL there should be no NaNs there."""
        result = align_to_monthly(self._make_raw())
        for col in ("FEDFUNDS", "DGS10", "CPIAUCSL"):
            assert result[col].notna().all(), f"NaN found in {col}"

    def test_dralacbn_forward_filled(self):
        """Quarterly DRALACBN values should be forward-filled to monthly frequency."""
        result = align_to_monthly(self._make_raw())
        # After ffill there should be no NaN in DRALACBN
        assert result["DRALACBN"].notna().all()

    def test_empty_series_returns_empty(self):
        """If input series are too short to overlap, result should be empty."""
        raw = {
            "FEDFUNDS": pd.DataFrame({"FEDFUNDS": [2.0]},
                                     index=pd.date_range("2020-01-01", periods=1, freq="MS")),
            "DGS10":    pd.DataFrame({"DGS10":    [3.0]},
                                     index=pd.date_range("2020-06-01", periods=1, freq="D")),
            "DRALACBN": pd.DataFrame({"DRALACBN": [1.5]},
                                     index=pd.date_range("2020-01-01", periods=1, freq="QS")),
            "PCE":      pd.DataFrame({"PCE":      [15_000.0]},
                                     index=pd.date_range("2020-01-01", periods=1, freq="MS")),
            "CPIAUCSL": pd.DataFrame({"CPIAUCSL": [260.0]},
                                     index=pd.date_range("2025-01-01", periods=1, freq="MS")),
        }
        result = align_to_monthly(raw)
        # FEDFUNDS and CPIAUCSL do not overlap → dropna removes all rows
        assert len(result) == 0


# ── compute_analytics ──────────────────────────────────────────────────────────

class TestComputeAnalytics:
    def test_spread_is_dgs10_minus_fedfunds(self):
        df = _analytics_df(FEDFUNDS=2.0, DGS10=3.5)
        assert (df["spread"] == pytest.approx(1.5, abs=1e-9)).all()

    def test_spread_negative_when_inverted(self):
        df = _analytics_df(FEDFUNDS=5.0, DGS10=4.0)
        assert (df["spread"] == pytest.approx(-1.0, abs=1e-9)).all()

    def test_spread_zero_when_equal(self):
        df = _analytics_df(FEDFUNDS=3.0, DGS10=3.0)
        assert (df["spread"] == pytest.approx(0.0, abs=1e-9)).all()

    def test_mom_changes_first_row_is_nan(self):
        """Month-over-month diff → first row must be NaN."""
        df = _analytics_df()
        assert pd.isna(df["FEDFUNDS_chg"].iloc[0])
        assert pd.isna(df["DGS10_chg"].iloc[0])
        assert pd.isna(df["DRALACBN_chg"].iloc[0])

    def test_mom_changes_constant_series_are_zero(self):
        """Constant series should produce zero MoM changes (except first row)."""
        df = _analytics_df(FEDFUNDS=2.0, DGS10=3.0, DRALACBN=1.5)
        assert (df["FEDFUNDS_chg"].dropna() == 0.0).all()
        assert (df["DGS10_chg"].dropna()    == 0.0).all()
        assert (df["DRALACBN_chg"].dropna() == 0.0).all()

    def test_yoy_fedfunds_correct_magnitude(self):
        """FEDFUNDS_yoy = FEDFUNDS[t] − FEDFUNDS[t−12]."""
        n = 24
        # First 12 months at 2 %, next 12 months at 4 %  → YoY = +2
        rates = [2.0] * 12 + [4.0] * 12
        df = _analytics_df(n=n, FEDFUNDS=rates)
        yoy = df["FEDFUNDS_yoy"].dropna()
        assert (yoy == pytest.approx(2.0, abs=1e-9)).all()

    def test_cpi_yoy_pct_correct(self):
        """CPI_yoy_pct = pct_change(12) * 100."""
        n = 25
        # CPI grows from 200 to 220 over 12 months (+10 %)
        base = [200.0] * 12 + [220.0] * (n - 12)
        df = _analytics_df(n=n, CPIAUCSL=base)
        valid = df["CPI_yoy_pct"].dropna()
        assert (valid == pytest.approx(10.0, abs=1e-6)).all()

    def test_output_contains_required_columns(self):
        df = _analytics_df()
        expected = {
            "spread", "FEDFUNDS_chg", "DGS10_chg", "DRALACBN_chg",
            "FEDFUNDS_yoy", "DGS10_yoy", "DRALACBN_yoy",
            "CPI_yoy_pct", "PCE_yoy_pct",
            "corr_ff_delinq", "corr_dgs10_delinq",
            "corr_ff_lag3_delinq", "corr_ff_lag6_delinq", "corr_ff_lag12_delinq",
        }
        for col in expected:
            assert col in df.columns, f"Missing analytics column: {col}"

    def test_does_not_mutate_input(self):
        """compute_analytics must return a copy, not mutate the caller's df."""
        raw = _monthly_df()
        original_cols = set(raw.columns)
        _ = compute_analytics(raw)
        assert set(raw.columns) == original_cols, "Input DataFrame was mutated"

    def test_rolling_corr_requires_sufficient_rows(self):
        """With fewer rows than the rolling window (18), corr columns are all NaN."""
        df = _analytics_df(n=17)
        assert df["corr_ff_delinq"].isna().all()


# ── flag_risk — boundary value analysis ───────────────────────────────────────

class TestFlagRiskBoundaries:
    """
    All thresholds use strict inequalities (< or >).

    Boundary convention
    -------------------
    value == threshold  →  NOT flagged
    value just past threshold  →  flagged
    """

    # ── inverted_curve: spread < 0.0 ──────────────────────────────────────────

    def test_inverted_curve_at_zero_not_flagged(self):
        df, flags = _flag_df(FEDFUNDS=3.0, DGS10=3.0)   # spread == 0.0
        assert not flags["inverted_curve"].any()

    def test_inverted_curve_just_below_zero_flagged(self):
        eps = 0.001
        df, flags = _flag_df(FEDFUNDS=3.0 + eps, DGS10=3.0)  # spread == -eps
        assert flags["inverted_curve"].all()

    def test_inverted_curve_positive_spread_not_flagged(self):
        df, flags = _flag_df(FEDFUNDS=2.0, DGS10=4.0)   # spread == +2.0
        assert not flags["inverted_curve"].any()

    # ── near_inverted: spread < 0.5 ───────────────────────────────────────────

    def test_near_inverted_at_0_5_not_flagged(self):
        df, flags = _flag_df(FEDFUNDS=2.5, DGS10=3.0)   # spread == 0.5
        assert not flags["near_inverted"].any()

    def test_near_inverted_just_below_0_5_flagged(self):
        eps = 0.001
        df, flags = _flag_df(FEDFUNDS=2.5 + eps, DGS10=3.0)  # spread = 0.5 - eps
        assert flags["near_inverted"].all()

    def test_near_inverted_above_0_5_not_flagged(self):
        df, flags = _flag_df(FEDFUNDS=2.0, DGS10=3.0)   # spread == 1.0
        assert not flags["near_inverted"].any()

    # ── high_delinquency: DRALACBN > 1.80 ────────────────────────────────────

    def test_high_delinquency_at_1_80_not_flagged(self):
        df, flags = _flag_df(DRALACBN=1.80)
        assert not flags["high_delinquency"].any()

    def test_high_delinquency_just_above_1_80_flagged(self):
        df, flags = _flag_df(DRALACBN=1.801)
        assert flags["high_delinquency"].all()

    def test_high_delinquency_well_below_not_flagged(self):
        df, flags = _flag_df(DRALACBN=1.0)
        assert not flags["high_delinquency"].any()

    def test_high_delinquency_well_above_flagged(self):
        df, flags = _flag_df(DRALACBN=3.5)
        assert flags["high_delinquency"].all()

    # ── high_rates: FEDFUNDS > 5.0 ───────────────────────────────────────────

    def test_high_rates_at_5_0_not_flagged(self):
        df, flags = _flag_df(FEDFUNDS=5.0)
        assert not flags["high_rates"].any()

    def test_high_rates_just_above_5_0_flagged(self):
        df, flags = _flag_df(FEDFUNDS=5.001)
        assert flags["high_rates"].all()

    def test_high_rates_well_below_not_flagged(self):
        df, flags = _flag_df(FEDFUNDS=2.0)
        assert not flags["high_rates"].any()

    # ── high_inflation: CPI_yoy_pct > 4.0 ────────────────────────────────────
    # CPI_yoy_pct = pct_change(12) * 100, so we manipulate CPIAUCSL values.

    def _df_with_cpi_yoy(self, target_pct: float, n: int = 25) -> pd.DataFrame:
        """Build a DataFrame where CPI YoY % stabilises at target_pct after month 12."""
        base_cpi = 260.0
        factor = 1 + target_pct / 100
        cpi = [base_cpi] * 12 + [round(base_cpi * factor, 6)] * (n - 12)
        return _analytics_df(n=n, CPIAUCSL=cpi)

    def test_high_inflation_at_4_pct_not_flagged(self):
        df = self._df_with_cpi_yoy(4.0)
        df_flagged, flags = flag_risk(df)
        assert not flags["high_inflation"].dropna().any()

    def test_high_inflation_above_4_pct_flagged(self):
        df = self._df_with_cpi_yoy(4.1)
        df_flagged, flags = flag_risk(df)
        assert flags["high_inflation"].dropna().any()

    def test_high_inflation_below_4_pct_not_flagged(self):
        df = self._df_with_cpi_yoy(3.5)
        df_flagged, flags = flag_risk(df)
        assert not flags["high_inflation"].dropna().any()

    # ── rapid_rate_hike: FEDFUNDS_yoy > 2.0 ──────────────────────────────────

    def _df_with_fedfunds_yoy(self, yoy: float, n: int = 25) -> pd.DataFrame:
        """Build a DataFrame where FEDFUNDS YoY change stabilises at yoy after month 12."""
        rates = [2.0] * 12 + [2.0 + yoy] * (n - 12)
        return _analytics_df(n=n, FEDFUNDS=rates)

    def test_rapid_rate_hike_at_2_0_not_flagged(self):
        df = self._df_with_fedfunds_yoy(2.0)
        df_flagged, flags = flag_risk(df)
        assert not flags["rapid_rate_hike"].dropna().any()

    def test_rapid_rate_hike_above_2_0_flagged(self):
        df = self._df_with_fedfunds_yoy(2.01)
        df_flagged, flags = flag_risk(df)
        assert flags["rapid_rate_hike"].dropna().any()

    def test_rapid_rate_hike_below_2_0_not_flagged(self):
        df = self._df_with_fedfunds_yoy(1.5)
        df_flagged, flags = flag_risk(df)
        assert not flags["rapid_rate_hike"].dropna().any()

    # ── rising_delinquency: DRALACBN_yoy > 0.10 ──────────────────────────────

    def _df_with_delinq_yoy(self, yoy: float, n: int = 25) -> pd.DataFrame:
        """Build a DataFrame where DRALACBN YoY change stabilises at yoy after month 12."""
        vals = [1.5] * 12 + [1.5 + yoy] * (n - 12)
        return _analytics_df(n=n, DRALACBN=vals)

    def test_rising_delinquency_at_0_10_not_flagged(self):
        df = self._df_with_delinq_yoy(0.10)
        df_flagged, flags = flag_risk(df)
        assert not flags["rising_delinquency"].dropna().any()

    def test_rising_delinquency_above_0_10_flagged(self):
        df = self._df_with_delinq_yoy(0.11)
        df_flagged, flags = flag_risk(df)
        assert flags["rising_delinquency"].dropna().any()

    def test_rising_delinquency_below_0_10_not_flagged(self):
        df = self._df_with_delinq_yoy(0.05)
        df_flagged, flags = flag_risk(df)
        assert not flags["rising_delinquency"].dropna().any()


# ── flag_risk — risk_score & elevated_risk ────────────────────────────────────

class TestFlagRiskScoreAndElevated:
    def test_risk_score_zero_benign_environment(self):
        """Low rates, positive spread, low delinquency → risk_score = 0."""
        df, flags = _flag_df(FEDFUNDS=2.0, DGS10=4.0, DRALACBN=1.0)
        assert (df["risk_score"] == 0).all()

    def test_risk_score_max_stressed_environment(self):
        """All seven flags active simultaneously → risk_score = 7."""
        # To trigger all flags we need:
        #   spread < 0      → FEDFUNDS > DGS10
        #   spread < 0.5    → same condition
        #   DRALACBN > 1.80
        #   DRALACBN_yoy > 0.10 (need YoY change)
        #   CPI_yoy_pct > 4.0  (need CPI growth)
        #   FEDFUNDS_yoy > 2.0 (need YoY rate hike)
        #   FEDFUNDS > 5.0
        n = 25
        fedfunds = [3.5] * 12 + [6.0] * (n - 12)   # YoY hike of +2.5, and > 5
        dgs10    = [3.0] * n                          # spread = DGS10 - FF < 0
        dralacbn = [1.7] * 12 + [1.9] * (n - 12)    # YoY rise of +0.2, and > 1.80
        cpi      = [260.0] * 12 + [260.0 * 1.05] * (n - 12)   # 5% YoY

        df_raw = _analytics_df(
            n=n,
            FEDFUNDS=fedfunds,
            DGS10=dgs10,
            DRALACBN=dralacbn,
            CPIAUCSL=cpi,
        )
        df_flagged, flags = flag_risk(df_raw)
        # Check the last row (well into the stressed period)
        last = df_flagged.iloc[-1]
        assert last["risk_score"] == 7, f"Expected score 7, got {last['risk_score']}"

    def test_elevated_risk_threshold_is_3(self):
        """elevated_risk must be True iff risk_score >= 3."""
        n = 25
        fedfunds = [3.5] * 12 + [6.0] * (n - 12)
        dgs10    = [3.0] * n
        dralacbn = [1.7] * 12 + [1.9] * (n - 12)
        cpi      = [260.0] * 12 + [260.0 * 1.05] * (n - 12)

        df_raw = _analytics_df(n=n, FEDFUNDS=fedfunds, DGS10=dgs10,
                               DRALACBN=dralacbn, CPIAUCSL=cpi)
        df_flagged, _ = flag_risk(df_raw)

        # For every row: elevated_risk == (risk_score >= 3)
        expected = df_flagged["risk_score"] >= 3
        pd.testing.assert_series_equal(df_flagged["elevated_risk"], expected,
                                       check_names=False)

    def test_active_flags_string_is_empty_when_no_flags(self):
        df, flags = _flag_df(FEDFUNDS=2.0, DGS10=4.0, DRALACBN=1.0)
        assert (df["active_flags"] == "").all()

    def test_active_flags_contains_flag_name(self):
        """When high_rates fires, 'high_rates' should appear in active_flags."""
        df, flags = _flag_df(FEDFUNDS=5.5, DGS10=4.0, DRALACBN=1.0)
        assert df["active_flags"].str.contains("high_rates").any()

    def test_returns_two_element_tuple(self):
        result = flag_risk(_analytics_df())
        assert isinstance(result, tuple) and len(result) == 2

    def test_flag_columns_are_boolean(self):
        _, flags = flag_risk(_analytics_df())
        for col in flags.columns:
            assert flags[col].dtype == bool, f"Column {col} is not bool"


# ── flag_risk — edge cases ─────────────────────────────────────────────────────

class TestFlagRiskEdgeCases:
    def test_missing_column_skipped_gracefully(self):
        """flag_risk should not raise if a column referenced in RISK_THRESHOLDS
        is absent (e.g. DRALACBN_yoy not yet computed)."""
        df = _analytics_df()
        df_no_yoy = df.drop(columns=["DRALACBN_yoy"])
        # Should not raise
        df_flagged, flags = flag_risk(df_no_yoy)
        assert "rising_delinquency" not in flags.columns

    def test_all_nan_values_no_exception(self):
        """NaN-filled columns should not cause flag_risk to raise."""
        df = _analytics_df()
        df["spread"] = np.nan
        df["DRALACBN"] = np.nan
        # Should run without error
        df_flagged, flags = flag_risk(df)

    def test_single_row_dataframe(self):
        """flag_risk should work with a single-row DataFrame."""
        df = _analytics_df(n=1)
        df_flagged, flags = flag_risk(df)
        assert len(df_flagged) == 1

    def test_risk_score_non_negative(self):
        """risk_score must always be >= 0."""
        df, _ = flag_risk(_analytics_df())
        assert (df["risk_score"] >= 0).all()

    def test_risk_score_at_most_seven(self):
        """risk_score must never exceed the total number of flags (7)."""
        df, _ = flag_risk(_analytics_df())
        assert (df["risk_score"] <= 7).all()


# ── RISK_THRESHOLDS constant ──────────────────────────────────────────────────

class TestRiskThresholdsConstant:
    def test_all_thresholds_present(self):
        expected_keys = {
            "inverted_curve", "near_inverted", "rising_delinquency",
            "high_delinquency", "high_inflation", "rapid_rate_hike", "high_rates",
        }
        assert set(RISK_THRESHOLDS.keys()) == expected_keys

    def test_threshold_structure(self):
        """Each entry is a (column_name, operator, numeric_threshold) triple."""
        for name, entry in RISK_THRESHOLDS.items():
            assert len(entry) == 3, f"{name}: expected 3-tuple, got {entry}"
            col, op, val = entry
            assert isinstance(col, str), f"{name}: column name must be str"
            assert op in ("<", ">"), f"{name}: operator must be '<' or '>'"
            assert isinstance(val, (int, float)), f"{name}: threshold must be numeric"
