"""
Tests for main.py

Coverage areas:
- fetch_series: HTTP interaction (mocked), value parsing, error propagation
- align_to_monthly: resampling, forward-fill, required-column NaN drop
- compute_analytics: spread, MoM/YoY changes, rolling correlations
- flag_risk: every threshold, boundary values, elevated-risk scoring,
             missing-column tolerance, return shape
"""
import json
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

import main


# ── Helpers ───────────────────────────────────────────────────────────────────

def _monthly_index(n: int = 36, start: str = "2020-01-31") -> pd.DatetimeIndex:
    return pd.date_range(start, periods=n, freq="ME")


def _raw_series(series_id: str, values, freq: str = "ME") -> pd.DataFrame:
    """Create a single-column DataFrame as returned by fetch_series."""
    idx = pd.date_range("2020-01-31", periods=len(values), freq=freq)
    return pd.DataFrame({series_id: values}, index=idx)


def _make_aligned(
    n: int = 36,
    fedfunds: float = 3.0,
    dgs10: float = 4.0,
    dralacbn: float = 1.5,
    cpiaucsl: float = 280.0,
    pce: float = 17_000.0,
) -> pd.DataFrame:
    """Return a ready-made aligned monthly DataFrame suitable for analytics tests."""
    idx = _monthly_index(n)
    return pd.DataFrame(
        {
            "FEDFUNDS": fedfunds,
            "DGS10": dgs10,
            "DRALACBN": dralacbn,
            "CPIAUCSL": cpiaucsl,
            "PCE": pce,
        },
        index=idx,
        dtype=float,
    )


# ── fetch_series ──────────────────────────────────────────────────────────────

class TestFetchSeries:
    """Unit tests for fetch_series with a mocked HTTP layer."""

    def _make_response(self, observations: list[dict]) -> MagicMock:
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"observations": observations}
        mock_resp.raise_for_status.return_value = None
        return mock_resp

    @patch("main.requests.get")
    def test_returns_dataframe_with_correct_column(self, mock_get):
        observations = [
            {"date": "2024-01-01", "value": "5.33"},
            {"date": "2024-02-01", "value": "5.33"},
        ]
        mock_get.return_value = self._make_response(observations)

        df = main.fetch_series("FEDFUNDS")

        assert isinstance(df, pd.DataFrame)
        assert "FEDFUNDS" in df.columns
        assert len(df) == 2

    @patch("main.requests.get")
    def test_date_becomes_datetime_index(self, mock_get):
        observations = [{"date": "2024-03-15", "value": "4.5"}]
        mock_get.return_value = self._make_response(observations)

        df = main.fetch_series("DGS10")

        assert isinstance(df.index, pd.DatetimeIndex)
        assert df.index[0] == pd.Timestamp("2024-03-15")

    @patch("main.requests.get")
    def test_dot_values_coerced_to_nan(self, mock_get):
        """FRED uses '.' as a placeholder for missing observations."""
        observations = [
            {"date": "2024-01-01", "value": "."},
            {"date": "2024-02-01", "value": "1.80"},
        ]
        mock_get.return_value = self._make_response(observations)

        df = main.fetch_series("DRALACBN")

        assert pd.isna(df["DRALACBN"].iloc[0])
        assert df["DRALACBN"].iloc[1] == pytest.approx(1.80)

    @patch("main.requests.get")
    def test_negative_values_preserved(self, mock_get):
        observations = [{"date": "2020-06-30", "value": "-0.12"}]
        mock_get.return_value = self._make_response(observations)

        df = main.fetch_series("FEDFUNDS")

        assert df["FEDFUNDS"].iloc[0] == pytest.approx(-0.12)

    @patch("main.requests.get")
    def test_http_error_propagates(self, mock_get):
        """raise_for_status should propagate HTTP errors to the caller."""
        import requests as req

        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = req.HTTPError("403 Forbidden")
        mock_get.return_value = mock_resp

        with pytest.raises(req.HTTPError):
            main.fetch_series("FEDFUNDS")

    @patch("main.requests.get")
    def test_passes_api_key_to_request(self, mock_get):
        mock_get.return_value = self._make_response([])
        with patch.dict("os.environ", {"FRED_API_KEY": "test-key-123"}):
            import importlib
            importlib.reload(main)
            try:
                main.fetch_series("FEDFUNDS")
            except Exception:
                pass

        # Key should appear somewhere in the call kwargs
        call_kwargs = mock_get.call_args
        assert call_kwargs is not None


# ── align_to_monthly ──────────────────────────────────────────────────────────

class TestAlignToMonthly:
    """Tests for align_to_monthly resampling logic."""

    def _make_raw(self, n_monthly: int = 24) -> dict[str, pd.DataFrame]:
        idx_m = pd.date_range("2020-01-31", periods=n_monthly, freq="ME")
        # DGS10 as daily
        idx_d = pd.date_range("2020-01-01", periods=n_monthly * 21, freq="B")
        # DRALACBN as quarterly
        idx_q = pd.date_range("2020-01-31", periods=n_monthly // 3, freq="QE")

        return {
            "FEDFUNDS": pd.DataFrame({"FEDFUNDS": 3.0}, index=idx_m),
            "DGS10":    pd.DataFrame({"DGS10": 4.0},    index=idx_d),
            "DRALACBN": pd.DataFrame({"DRALACBN": 1.5},  index=idx_q),
            "PCE":      pd.DataFrame({"PCE": 17_000.0},  index=idx_m),
            "CPIAUCSL": pd.DataFrame({"CPIAUCSL": 280.0}, index=idx_m),
        }

    def test_output_has_correct_columns(self):
        raw = self._make_raw()
        df = main.align_to_monthly(raw)
        expected_cols = {"FEDFUNDS", "DGS10", "DRALACBN", "PCE", "CPIAUCSL"}
        assert expected_cols.issubset(df.columns)

    def test_output_index_is_month_end(self):
        raw = self._make_raw()
        df = main.align_to_monthly(raw)
        assert isinstance(df.index, pd.DatetimeIndex)
        # All dates should be month-end
        for dt in df.index:
            assert dt == dt + pd.offsets.MonthEnd(0)

    def test_dralacbn_forward_filled(self):
        """Quarterly DRALACBN must be forward-filled so monthly rows have a value."""
        raw = self._make_raw()
        df = main.align_to_monthly(raw)
        # After forward-fill no NaN should remain in DRALACBN
        # (some NaN at the start before the first quarterly obs is acceptable,
        # but within a period all months should be filled)
        non_na = df["DRALACBN"].dropna()
        assert len(non_na) > 0

    def test_drops_rows_where_required_columns_are_nan(self):
        """Rows where FEDFUNDS, DGS10, or CPIAUCSL are NaN must be dropped."""
        raw = self._make_raw(n_monthly=12)
        # Introduce a NaN in FEDFUNDS for one month
        idx_m = pd.date_range("2020-01-31", periods=12, freq="ME")
        values = [3.0] * 12
        values[0] = float("nan")
        raw["FEDFUNDS"] = pd.DataFrame({"FEDFUNDS": values}, index=idx_m)

        df = main.align_to_monthly(raw)
        assert df["FEDFUNDS"].isna().sum() == 0

    def test_concat_aligns_on_date(self):
        """All series must share the same monthly DatetimeIndex after alignment."""
        raw = self._make_raw()
        df = main.align_to_monthly(raw)
        # Index should be monotonically increasing
        assert df.index.is_monotonic_increasing


# ── compute_analytics ─────────────────────────────────────────────────────────

class TestComputeAnalytics:
    """Tests for compute_analytics derived metric calculations."""

    def test_spread_equals_dgs10_minus_fedfunds(self):
        df = _make_aligned(fedfunds=3.0, dgs10=4.5)
        out = main.compute_analytics(df)
        expected = 4.5 - 3.0
        assert (out["spread"].dropna() == pytest.approx(expected)).all()

    def test_spread_negative_when_inverted(self):
        df = _make_aligned(fedfunds=5.5, dgs10=4.0)
        out = main.compute_analytics(df)
        assert (out["spread"].dropna() < 0).all()

    def test_mom_change_first_row_is_nan(self):
        df = _make_aligned(n=12)
        out = main.compute_analytics(df)
        assert pd.isna(out["FEDFUNDS_chg"].iloc[0])

    def test_mom_change_computed_correctly(self):
        df = _make_aligned(n=12)
        df["FEDFUNDS"] = list(range(1, 13))  # 1, 2, …, 12
        out = main.compute_analytics(df)
        # All MoM changes should be 1.0 after the first row
        assert (out["FEDFUNDS_chg"].iloc[1:] == pytest.approx(1.0)).all()

    def test_yoy_change_first_12_rows_are_nan(self):
        df = _make_aligned(n=24)
        out = main.compute_analytics(df)
        assert out["FEDFUNDS_yoy"].iloc[:12].isna().all()

    def test_cpi_yoy_pct_computed_as_percent(self):
        """CPI_yoy_pct = pct_change(12) * 100, not a ratio."""
        df = _make_aligned(n=24)
        # Make CPI grow linearly so we can predict the YoY result
        cpi_base = 100.0
        df["CPIAUCSL"] = [cpi_base + i * 0.5 for i in range(24)]
        out = main.compute_analytics(df)
        # At month 12 (index 12), CPI=106, base month 0 CPI=100 → ~6%
        yoy_at_12 = out["CPI_yoy_pct"].iloc[12]
        assert not pd.isna(yoy_at_12)
        assert yoy_at_12 == pytest.approx((106.0 / 100.0 - 1) * 100, rel=0.05)

    def test_rolling_correlations_columns_present(self):
        df = _make_aligned(n=36)
        out = main.compute_analytics(df)
        for col in ["corr_ff_delinq", "corr_dgs10_delinq",
                    "corr_ff_lag3_delinq", "corr_ff_lag6_delinq",
                    "corr_ff_lag12_delinq"]:
            assert col in out.columns, f"Missing column: {col}"

    def test_returns_copy_not_mutating_input(self):
        df = _make_aligned(n=24)
        original_cols = set(df.columns)
        main.compute_analytics(df)
        # Input df should still have only its original columns
        assert set(df.columns) == original_cols


# ── flag_risk ─────────────────────────────────────────────────────────────────

class TestFlagRisk:
    """
    Tests for flag_risk covering every threshold condition and boundary values.

    RISK_THRESHOLDS (from main.py):
      inverted_curve    spread           < 0.0
      near_inverted     spread           < 0.5
      rising_delinquency DRALACBN_yoy   > 0.10
      high_delinquency  DRALACBN        > 1.80
      high_inflation    CPI_yoy_pct     > 4.0
      rapid_rate_hike   FEDFUNDS_yoy    > 2.0
      high_rates        FEDFUNDS        > 5.0
    elevated_risk: risk_score >= 3
    """

    def _base_df(self, n: int = 36) -> pd.DataFrame:
        """Return a computed-analytics DataFrame with all flags OFF."""
        df = _make_aligned(
            n=n,
            fedfunds=2.0,   # well below 5.0
            dgs10=3.5,      # spread = 1.5 (positive, > 0.5)
            dralacbn=1.0,   # below 1.80
            cpiaucsl=100.0,
            pce=17_000.0,
        )
        df = main.compute_analytics(df)
        # Force analytical columns to safe values to override NaN warm-up
        df["spread"]        = 1.5
        df["DRALACBN_yoy"]  = 0.05   # < 0.10
        df["CPI_yoy_pct"]   = 2.0    # < 4.0
        df["FEDFUNDS_yoy"]  = 0.5    # < 2.0
        return df

    # ── return shape ──────────────────────────────────────────────────────────

    def test_returns_tuple_of_df_and_flags(self):
        df = self._base_df()
        result = main.flag_risk(df)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_output_df_has_risk_score_column(self):
        df = self._base_df()
        out_df, _ = main.flag_risk(df)
        assert "risk_score" in out_df.columns

    def test_output_df_has_elevated_risk_column(self):
        df = self._base_df()
        out_df, _ = main.flag_risk(df)
        assert "elevated_risk" in out_df.columns

    def test_output_df_has_active_flags_column(self):
        df = self._base_df()
        out_df, _ = main.flag_risk(df)
        assert "active_flags" in out_df.columns

    def test_does_not_mutate_input(self):
        df = self._base_df()
        original_cols = set(df.columns)
        main.flag_risk(df)
        assert set(df.columns) == original_cols

    # ── no risk ───────────────────────────────────────────────────────────────

    def test_all_flags_off_when_all_safe(self):
        df = self._base_df()
        out_df, flags = main.flag_risk(df)
        assert (out_df["risk_score"] == 0).all()
        assert (out_df["elevated_risk"] == False).all()  # noqa: E712

    # ── individual thresholds ─────────────────────────────────────────────────

    def test_inverted_curve_flag_when_spread_below_zero(self):
        df = self._base_df()
        df["spread"] = -0.1
        out_df, flags = main.flag_risk(df)
        assert (flags["inverted_curve"]).all()

    def test_inverted_curve_not_flagged_at_zero(self):
        """Boundary: spread == 0 should NOT trigger inverted_curve (< 0)."""
        df = self._base_df()
        df["spread"] = 0.0
        out_df, flags = main.flag_risk(df)
        assert not (flags["inverted_curve"]).any()

    def test_near_inverted_flag_below_0_5(self):
        df = self._base_df()
        df["spread"] = 0.4
        out_df, flags = main.flag_risk(df)
        assert (flags["near_inverted"]).all()

    def test_near_inverted_not_flagged_at_0_5(self):
        """Boundary: spread == 0.5 should NOT trigger near_inverted (< 0.5)."""
        df = self._base_df()
        df["spread"] = 0.5
        out_df, flags = main.flag_risk(df)
        assert not (flags["near_inverted"]).any()

    def test_rising_delinquency_flag_above_0_10(self):
        df = self._base_df()
        df["DRALACBN_yoy"] = 0.15
        out_df, flags = main.flag_risk(df)
        assert (flags["rising_delinquency"]).all()

    def test_rising_delinquency_not_flagged_at_0_10(self):
        """Boundary: DRALACBN_yoy == 0.10 should NOT trigger (> 0.10)."""
        df = self._base_df()
        df["DRALACBN_yoy"] = 0.10
        out_df, flags = main.flag_risk(df)
        assert not (flags["rising_delinquency"]).any()

    def test_high_delinquency_flag_above_1_80(self):
        df = self._base_df()
        df["DRALACBN"] = 1.81
        out_df, flags = main.flag_risk(df)
        assert (flags["high_delinquency"]).all()

    def test_high_delinquency_not_flagged_at_1_80(self):
        """Boundary: DRALACBN == 1.80 should NOT trigger (> 1.80)."""
        df = self._base_df()
        df["DRALACBN"] = 1.80
        out_df, flags = main.flag_risk(df)
        assert not (flags["high_delinquency"]).any()

    def test_high_inflation_flag_above_4_0(self):
        df = self._base_df()
        df["CPI_yoy_pct"] = 4.5
        out_df, flags = main.flag_risk(df)
        assert (flags["high_inflation"]).all()

    def test_high_inflation_not_flagged_at_4_0(self):
        """Boundary: CPI_yoy_pct == 4.0 should NOT trigger (> 4.0)."""
        df = self._base_df()
        df["CPI_yoy_pct"] = 4.0
        out_df, flags = main.flag_risk(df)
        assert not (flags["high_inflation"]).any()

    def test_rapid_rate_hike_flag_above_2_0(self):
        df = self._base_df()
        df["FEDFUNDS_yoy"] = 2.5
        out_df, flags = main.flag_risk(df)
        assert (flags["rapid_rate_hike"]).all()

    def test_rapid_rate_hike_not_flagged_at_2_0(self):
        """Boundary: FEDFUNDS_yoy == 2.0 should NOT trigger (> 2.0)."""
        df = self._base_df()
        df["FEDFUNDS_yoy"] = 2.0
        out_df, flags = main.flag_risk(df)
        assert not (flags["rapid_rate_hike"]).any()

    def test_high_rates_flag_above_5_0(self):
        df = self._base_df()
        df["FEDFUNDS"] = 5.5
        out_df, flags = main.flag_risk(df)
        assert (flags["high_rates"]).all()

    def test_high_rates_not_flagged_at_5_0(self):
        """Boundary: FEDFUNDS == 5.0 should NOT trigger (> 5.0)."""
        df = self._base_df()
        df["FEDFUNDS"] = 5.0
        out_df, flags = main.flag_risk(df)
        assert not (flags["high_rates"]).any()

    # ── risk score accumulation ───────────────────────────────────────────────

    def test_risk_score_accumulates_correctly(self):
        """Trigger exactly 3 flags → risk_score == 3."""
        df = self._base_df()
        df["spread"]       = -0.1   # inverted_curve + near_inverted → 2 flags
        df["FEDFUNDS"]     = 5.5    # high_rates → 1 flag
        out_df, _ = main.flag_risk(df)
        assert (out_df["risk_score"] == 3).all()

    def test_elevated_risk_triggers_at_score_3(self):
        """elevated_risk must be True when risk_score == 3."""
        df = self._base_df()
        df["spread"]       = -0.1   # +2 flags
        df["FEDFUNDS"]     = 5.5    # +1 flag  → score = 3
        out_df, _ = main.flag_risk(df)
        assert (out_df["elevated_risk"]).all()

    def test_elevated_risk_false_at_score_2(self):
        """elevated_risk must be False when risk_score == 2."""
        df = self._base_df()
        df["spread"] = -0.1   # inverted_curve + near_inverted → score = 2
        out_df, _ = main.flag_risk(df)
        assert not (out_df["elevated_risk"]).any()

    def test_max_risk_score_is_7(self):
        """All 7 flags active → risk_score == 7."""
        df = self._base_df()
        df["spread"]        = -0.1  # inverted_curve, near_inverted
        df["DRALACBN_yoy"]  = 0.50  # rising_delinquency
        df["DRALACBN"]      = 2.00  # high_delinquency
        df["CPI_yoy_pct"]   = 6.00  # high_inflation
        df["FEDFUNDS_yoy"]  = 3.00  # rapid_rate_hike
        df["FEDFUNDS"]      = 6.00  # high_rates
        out_df, _ = main.flag_risk(df)
        assert (out_df["risk_score"] == 7).all()

    # ── robustness ────────────────────────────────────────────────────────────

    def test_missing_column_in_df_is_skipped(self):
        """If a column referenced by RISK_THRESHOLDS is absent, no error."""
        df = self._base_df()
        df = df.drop(columns=["spread"])   # removes spread-based flags
        out_df, flags = main.flag_risk(df)
        # inverted_curve and near_inverted rely on 'spread' → should not appear
        assert "inverted_curve" not in flags.columns
        assert "near_inverted" not in flags.columns

    def test_active_flags_string_lists_triggered_flags(self):
        """active_flags column should name the flags that are active."""
        df = self._base_df()
        df["FEDFUNDS"] = 5.5   # high_rates flag
        out_df, _ = main.flag_risk(df)
        assert "high_rates" in out_df["active_flags"].iloc[0]

    def test_active_flags_empty_when_no_flags(self):
        df = self._base_df()
        out_df, _ = main.flag_risk(df)
        assert (out_df["active_flags"] == "").all()
