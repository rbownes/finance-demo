"""
Tests for main.py — financial modeling analytics with emphasis on
boundary conditions and failure states.
"""
import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

import main


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_monthly_df(n_months: int = 36, seed: int = 42) -> pd.DataFrame:
    """Return a well-formed monthly DataFrame starting 1990-01-31."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("1990-01-31", periods=n_months, freq="ME")
    return pd.DataFrame(
        {
            "FEDFUNDS": rng.uniform(0.5, 8.0, n_months),
            "DGS10": rng.uniform(1.0, 9.0, n_months),
            "DRALACBN": rng.uniform(0.5, 3.0, n_months),
            "PCE": rng.uniform(8_000, 18_000, n_months),
            "CPIAUCSL": rng.uniform(200, 320, n_months),
        },
        index=idx,
    )


def make_raw_dict(n_months: int = 36) -> dict[str, pd.DataFrame]:
    """Mimic the output of fetch_all() — a dict of single-column DataFrames."""
    base = make_monthly_df(n_months)
    return {col: base[[col]] for col in base.columns}


# ── fetch_series ──────────────────────────────────────────────────────────────

class TestFetchSeries:
    def _mock_response(self, observations: list[dict]) -> MagicMock:
        r = MagicMock()
        r.raise_for_status = MagicMock()
        r.json.return_value = {"observations": observations}
        return r

    def _obs(self, date: str, value: str) -> dict:
        return {"date": date, "value": value, "realtime_start": date, "realtime_end": date}

    def test_normal_response(self):
        obs = [self._obs("1990-01-01", "5.5"), self._obs("1990-02-01", "5.75")]
        with patch("requests.get", return_value=self._mock_response(obs)):
            df = main.fetch_series("FEDFUNDS")
        assert len(df) == 2
        assert df.index.name == "date"
        assert "FEDFUNDS" in df.columns
        assert df["FEDFUNDS"].iloc[0] == pytest.approx(5.5)

    def test_dot_values_become_nan(self):
        """FRED uses '.' for missing values; they should become NaN."""
        obs = [self._obs("1990-01-01", "5.5"), self._obs("1990-02-01", ".")]
        with patch("requests.get", return_value=self._mock_response(obs)):
            df = main.fetch_series("FEDFUNDS")
        assert pd.isna(df["FEDFUNDS"].iloc[1])

    def test_empty_observations(self):
        """An empty observation list should return an empty DataFrame."""
        with patch("requests.get", return_value=self._mock_response([])):
            # fetch_series tries to select ["date","value"] from an empty DF
            # which raises a KeyError — document this boundary
            with pytest.raises((KeyError, IndexError, ValueError)):
                main.fetch_series("FEDFUNDS")

    def test_http_error_propagates(self):
        """A non-2xx response should raise an HTTPError via raise_for_status."""
        import requests
        r = MagicMock()
        r.raise_for_status.side_effect = requests.HTTPError("404 Not Found")
        with patch("requests.get", return_value=r):
            with pytest.raises(requests.HTTPError):
                main.fetch_series("FEDFUNDS")

    def test_non_numeric_value_becomes_nan(self):
        """Unexpected non-numeric text should coerce to NaN, not crash."""
        obs = [self._obs("1990-01-01", "N/A")]
        with patch("requests.get", return_value=self._mock_response(obs)):
            df = main.fetch_series("FEDFUNDS")
        assert pd.isna(df["FEDFUNDS"].iloc[0])

    def test_negative_rate_accepted(self):
        """Negative rates (e.g. ECB-style) must parse without error."""
        obs = [self._obs("2020-01-01", "-0.50")]
        with patch("requests.get", return_value=self._mock_response(obs)):
            df = main.fetch_series("FEDFUNDS")
        assert df["FEDFUNDS"].iloc[0] == pytest.approx(-0.50)


# ── align_to_monthly ──────────────────────────────────────────────────────────

class TestAlignToMonthly:
    def test_normal_alignment_shape(self):
        raw = make_raw_dict(36)
        df = main.align_to_monthly(raw)
        assert set(["FEDFUNDS", "DGS10", "DRALACBN", "PCE", "CPIAUCSL"]).issubset(df.columns)
        assert len(df) > 0

    def test_daily_dgs10_averages_to_monthly(self):
        """DGS10 supplied daily should become one row per month (mean)."""
        idx_daily = pd.date_range("1990-01-01", periods=90, freq="D")
        dgs10_daily = pd.DataFrame({"DGS10": np.linspace(5.0, 6.0, 90)}, index=idx_daily)

        # Use monthly stubs for the others
        idx_mo = pd.date_range("1990-01-31", periods=3, freq="ME")
        stub = lambda col: pd.DataFrame({col: [1.0, 1.5, 2.0]}, index=idx_mo)

        raw = {
            "DGS10": dgs10_daily,
            "FEDFUNDS": stub("FEDFUNDS"),
            "DRALACBN": stub("DRALACBN"),
            "PCE": stub("PCE"),
            "CPIAUCSL": stub("CPIAUCSL"),
        }
        df = main.align_to_monthly(raw)
        assert len(df) == 3

    def test_quarterly_dralacbn_forward_filled(self):
        """Quarterly DRALACBN must be forward-filled to monthly resolution."""
        idx_q = pd.date_range("1990-01-31", periods=4, freq="QE")
        dralacbn = pd.DataFrame({"DRALACBN": [1.0, 1.5, 2.0, 2.5]}, index=idx_q)

        idx_mo = pd.date_range("1990-01-31", periods=12, freq="ME")
        stub = lambda col, val=5.0: pd.DataFrame({col: [val] * 12}, index=idx_mo)

        raw = {
            "FEDFUNDS": stub("FEDFUNDS"),
            "DGS10": stub("DGS10", 6.0),
            "DRALACBN": dralacbn,
            "PCE": stub("PCE", 10_000),
            "CPIAUCSL": stub("CPIAUCSL", 250),
        }
        df = main.align_to_monthly(raw)
        # Between Q1 and Q2 the ffill should propagate the Q1 value
        assert df["DRALACBN"].isna().sum() == 0

    def test_drops_rows_missing_required_columns(self):
        """Rows where FEDFUNDS, DGS10, or CPIAUCSL are NaN must be dropped."""
        raw = make_raw_dict(12)
        # Inject NaNs into required columns
        raw["FEDFUNDS"].iloc[0] = np.nan
        raw["CPIAUCSL"].iloc[1] = np.nan
        df = main.align_to_monthly(raw)
        assert df["FEDFUNDS"].isna().sum() == 0
        assert df["CPIAUCSL"].isna().sum() == 0

    def test_completely_empty_series_raises_or_empty(self):
        """All-NaN required series should produce an empty (or raise) result."""
        raw = make_raw_dict(12)
        raw["FEDFUNDS"][:] = np.nan
        df = main.align_to_monthly(raw)
        assert len(df) == 0


# ── compute_analytics ─────────────────────────────────────────────────────────

class TestComputeAnalytics:
    def test_spread_calculation(self):
        """spread = DGS10 − FEDFUNDS exactly."""
        df = make_monthly_df(24)
        result = main.compute_analytics(df)
        expected = df["DGS10"] - df["FEDFUNDS"]
        pd.testing.assert_series_equal(result["spread"], expected, check_names=False)

    def test_spread_zero_when_rates_equal(self):
        """Boundary: yield curve flat (spread == 0) when DGS10 == FEDFUNDS."""
        df = make_monthly_df(24)
        df["DGS10"] = df["FEDFUNDS"]  # force flat curve
        result = main.compute_analytics(df)
        assert (result["spread"] == 0.0).all()

    def test_spread_negative_when_inverted(self):
        """Boundary: spread is negative when short rate exceeds long rate."""
        df = make_monthly_df(24)
        df["DGS10"] = 2.0
        df["FEDFUNDS"] = 5.5  # inverted
        result = main.compute_analytics(df)
        assert (result["spread"] < 0).all()

    def test_mom_changes_first_row_is_nan(self):
        """First diff value must be NaN — no prior period to diff against."""
        df = make_monthly_df(24)
        result = main.compute_analytics(df)
        assert pd.isna(result["FEDFUNDS_chg"].iloc[0])
        assert pd.isna(result["DGS10_chg"].iloc[0])

    def test_yoy_changes_first_12_rows_are_nan(self):
        """diff(12) requires 12 prior rows; first 12 results must be NaN."""
        df = make_monthly_df(36)
        result = main.compute_analytics(df)
        assert result["FEDFUNDS_yoy"].iloc[:12].isna().all()
        assert not result["FEDFUNDS_yoy"].iloc[12:].isna().all()

    def test_cpi_yoy_pct_with_zero_base(self):
        """pct_change with a zero base row should produce inf or NaN, not crash."""
        df = make_monthly_df(24)
        df["CPIAUCSL"].iloc[0] = 0.0  # zero base — triggers division edge case
        result = main.compute_analytics(df)  # must not raise
        # The resulting value at row 12 may be inf or NaN — both acceptable
        val = result["CPI_yoy_pct"].iloc[12]
        assert np.isinf(val) or pd.isna(val)

    def test_cpi_yoy_pct_with_constant_cpi(self):
        """Constant CPI → YoY % change should be 0%."""
        df = make_monthly_df(36)
        df["CPIAUCSL"] = 250.0
        result = main.compute_analytics(df)
        non_nan = result["CPI_yoy_pct"].dropna()
        assert (non_nan == pytest.approx(0.0)).all()

    def test_rolling_correlation_requires_18_observations(self):
        """Rolling 18-month correlation should be NaN for first 17 rows."""
        df = make_monthly_df(36)
        result = main.compute_analytics(df)
        # First valid corr row: needs 18 rows of diffs → row index 18 (0-based)
        assert result["corr_ff_delinq"].iloc[:18].isna().all()

    def test_single_row_dataframe_does_not_crash(self):
        """A one-row DataFrame should not raise an exception."""
        df = make_monthly_df(1)
        result = main.compute_analytics(df)  # must not raise
        assert len(result) == 1

    def test_all_nan_column_does_not_crash(self):
        """An all-NaN DRALACBN column should produce NaN correlations, not raise."""
        df = make_monthly_df(36)
        df["DRALACBN"] = np.nan
        result = main.compute_analytics(df)  # must not raise
        assert result["corr_ff_delinq"].isna().all()

    def test_lagged_correlations_present(self):
        """Lagged correlation columns must exist for lags 3, 6, 12."""
        df = make_monthly_df(36)
        result = main.compute_analytics(df)
        for lag in (3, 6, 12):
            assert f"corr_ff_lag{lag}_delinq" in result.columns


# ── flag_risk ─────────────────────────────────────────────────────────────────

class TestFlagRisk:
    def _base_df(self, n: int = 36) -> pd.DataFrame:
        """Return a fully-computed analytics DataFrame."""
        return main.compute_analytics(make_monthly_df(n))

    # -- inverted_curve (spread < 0.0) -----------------------------------------

    def test_inverted_curve_flagged(self):
        df = self._base_df()
        df["spread"] = -0.1  # below threshold
        result, flags = main.flag_risk(df)
        assert flags["inverted_curve"].all()

    def test_inverted_curve_not_flagged_at_zero(self):
        """Boundary: spread == 0.0 is NOT inverted (strict less-than)."""
        df = self._base_df()
        df["spread"] = 0.0
        result, flags = main.flag_risk(df)
        assert not flags["inverted_curve"].any()

    def test_inverted_curve_not_flagged_positive(self):
        df = self._base_df()
        df["spread"] = 1.5
        result, flags = main.flag_risk(df)
        assert not flags["inverted_curve"].any()

    # -- near_inverted (spread < 0.5) ------------------------------------------

    def test_near_inverted_flagged_below_threshold(self):
        df = self._base_df()
        df["spread"] = 0.3
        result, flags = main.flag_risk(df)
        assert flags["near_inverted"].all()

    def test_near_inverted_not_flagged_at_threshold(self):
        """Boundary: spread == 0.5 is NOT near-inverted (strict less-than)."""
        df = self._base_df()
        df["spread"] = 0.5
        result, flags = main.flag_risk(df)
        assert not flags["near_inverted"].any()

    # -- high_delinquency (DRALACBN > 1.80) ------------------------------------

    def test_high_delinquency_flagged_above_threshold(self):
        df = self._base_df()
        df["DRALACBN"] = 1.81
        result, flags = main.flag_risk(df)
        assert flags["high_delinquency"].all()

    def test_high_delinquency_not_flagged_at_threshold(self):
        """Boundary: DRALACBN == 1.80 is NOT flagged (strict greater-than)."""
        df = self._base_df()
        df["DRALACBN"] = 1.80
        result, flags = main.flag_risk(df)
        assert not flags["high_delinquency"].any()

    def test_high_delinquency_not_flagged_below_threshold(self):
        df = self._base_df()
        df["DRALACBN"] = 1.79
        result, flags = main.flag_risk(df)
        assert not flags["high_delinquency"].any()

    # -- high_inflation (CPI_yoy_pct > 4.0) ------------------------------------

    def test_high_inflation_flagged(self):
        df = self._base_df()
        df["CPI_yoy_pct"] = 4.1
        result, flags = main.flag_risk(df)
        assert flags["high_inflation"].all()

    def test_high_inflation_not_flagged_at_threshold(self):
        """Boundary: CPI_yoy_pct == 4.0 is NOT flagged (strict greater-than)."""
        df = self._base_df()
        df["CPI_yoy_pct"] = 4.0
        result, flags = main.flag_risk(df)
        assert not flags["high_inflation"].any()

    # -- rapid_rate_hike (FEDFUNDS_yoy > 2.0) ----------------------------------

    def test_rapid_rate_hike_flagged(self):
        df = self._base_df()
        df["FEDFUNDS_yoy"] = 2.1
        result, flags = main.flag_risk(df)
        assert flags["rapid_rate_hike"].all()

    def test_rapid_rate_hike_not_flagged_at_threshold(self):
        """Boundary: FEDFUNDS_yoy == 2.0 is NOT flagged (strict greater-than)."""
        df = self._base_df()
        df["FEDFUNDS_yoy"] = 2.0
        result, flags = main.flag_risk(df)
        assert not flags["rapid_rate_hike"].any()

    # -- high_rates (FEDFUNDS > 5.0) -------------------------------------------

    def test_high_rates_flagged(self):
        df = self._base_df()
        df["FEDFUNDS"] = 5.1
        result, flags = main.flag_risk(df)
        assert flags["high_rates"].all()

    def test_high_rates_not_flagged_at_threshold(self):
        """Boundary: FEDFUNDS == 5.0 is NOT flagged (strict greater-than)."""
        df = self._base_df()
        df["FEDFUNDS"] = 5.0
        result, flags = main.flag_risk(df)
        assert not flags["high_rates"].any()

    # -- rising_delinquency (DRALACBN_yoy > 0.10) ------------------------------

    def test_rising_delinquency_flagged(self):
        df = self._base_df()
        df["DRALACBN_yoy"] = 0.11
        result, flags = main.flag_risk(df)
        assert flags["rising_delinquency"].all()

    def test_rising_delinquency_not_flagged_at_threshold(self):
        """Boundary: DRALACBN_yoy == 0.10 is NOT flagged (strict greater-than)."""
        df = self._base_df()
        df["DRALACBN_yoy"] = 0.10
        result, flags = main.flag_risk(df)
        assert not flags["rising_delinquency"].any()

    # -- risk_score & elevated_risk --------------------------------------------

    def test_risk_score_zero_when_no_flags(self):
        """Benign environment: all indicators well within safe range."""
        df = self._base_df()
        df["spread"] = 2.0          # positive, not near-inverted
        df["DRALACBN"] = 1.0        # below high threshold
        df["DRALACBN_yoy"] = 0.0    # not rising
        df["CPI_yoy_pct"] = 2.0     # below 4%
        df["FEDFUNDS_yoy"] = 0.0    # no rapid hike
        df["FEDFUNDS"] = 3.0        # below 5%
        result, flags = main.flag_risk(df)
        assert (result["risk_score"] == 0).all()

    def test_risk_score_max_seven_when_all_flags(self):
        """Worst-case: all seven conditions breached simultaneously."""
        df = self._base_df()
        df["spread"] = -0.5         # inverted + near_inverted
        df["DRALACBN_yoy"] = 0.5    # rising_delinquency
        df["DRALACBN"] = 3.0        # high_delinquency
        df["CPI_yoy_pct"] = 7.0     # high_inflation
        df["FEDFUNDS_yoy"] = 3.0    # rapid_rate_hike
        df["FEDFUNDS"] = 6.0        # high_rates
        result, flags = main.flag_risk(df)
        assert (result["risk_score"] == 7).all()

    def test_elevated_risk_threshold_is_3(self):
        """elevated_risk fires when risk_score >= 3."""
        df = self._base_df()
        # Score == 2: not elevated
        df["spread"] = -0.5         # inverted + near_inverted → score 2
        df["DRALACBN"] = 1.0
        df["DRALACBN_yoy"] = 0.0
        df["CPI_yoy_pct"] = 2.0
        df["FEDFUNDS_yoy"] = 0.0
        df["FEDFUNDS"] = 3.0
        result, _ = main.flag_risk(df)
        assert not result["elevated_risk"].any()

        # Score == 3: elevated
        df["DRALACBN"] = 2.0        # add high_delinquency flag
        result, _ = main.flag_risk(df)
        assert result["elevated_risk"].all()

    def test_active_flags_string_populated(self):
        """active_flags column must contain comma-separated flag names."""
        df = self._base_df()
        df["spread"] = -1.0  # inverted + near_inverted
        result, _ = main.flag_risk(df)
        non_empty = result["active_flags"].str.len() > 0
        assert non_empty.any()

    def test_empty_dataframe_does_not_crash(self):
        """flag_risk on an empty DataFrame should not raise."""
        df = make_monthly_df(36)
        df = main.compute_analytics(df)
        empty = df.iloc[0:0]  # zero rows
        result, flags = main.flag_risk(empty)
        assert len(result) == 0

    def test_missing_column_is_skipped_gracefully(self):
        """If a column referenced by RISK_THRESHOLDS is absent, skip it."""
        df = self._base_df()
        df = df.drop(columns=["spread"])  # remove 'spread' entirely
        result, flags = main.flag_risk(df)  # must not raise
        assert "inverted_curve" not in flags.columns
        assert "near_inverted" not in flags.columns

    def test_nan_values_in_flag_columns_do_not_count(self):
        """NaN comparisons evaluate to False; NaN rows should score 0."""
        df = self._base_df()
        df["spread"] = np.nan
        df["DRALACBN"] = np.nan
        df["CPI_yoy_pct"] = np.nan
        df["FEDFUNDS_yoy"] = np.nan
        df["FEDFUNDS"] = np.nan
        df["DRALACBN_yoy"] = np.nan
        result, _ = main.flag_risk(df)
        assert (result["risk_score"] == 0).all()

    def test_extreme_positive_rates_trigger_high_rates_flag(self):
        """Extreme rates (e.g. 20%+) should still trigger flags normally."""
        df = self._base_df()
        df["FEDFUNDS"] = 20.0
        df["DGS10"] = 18.0
        result, flags = main.flag_risk(df)
        assert flags["high_rates"].all()


# ── classify_regime (dashboard.py) ───────────────────────────────────────────

class TestClassifyRegime:
    """Tests for dashboard.classify_regime — covers all four regime buckets."""

    def _make_df(self, score: int) -> pd.DataFrame:
        """Build a minimal DataFrame whose risk_score equals `score`."""
        idx = pd.date_range("2020-01-31", periods=2, freq="ME")
        data = {
            "spread": [1.5, 1.5],
            "DRALACBN": [1.0, 1.0],
            "CPI_yoy_pct": [2.0, 2.0],
            "FEDFUNDS": [3.0, 3.0],
            "DGS10": [4.5, 4.5],
            "f_inverted": [0, 0],
            "f_near_inv": [0, 0],
            "f_rising_del": [0, 0],
            "f_high_del": [0, 0],
            "f_inflation": [0, 0],
            "f_rate_hike": [0, 0],
            "f_high_rates": [0, 0],
            "risk_score": [score, score],
        }
        return pd.DataFrame(data, index=idx)

    def test_score_0_is_normal(self):
        import dashboard
        regime = dashboard.classify_regime(self._make_df(0))
        assert regime["regime"] == "Normal / Benign"

    def test_score_1_is_normal(self):
        import dashboard
        regime = dashboard.classify_regime(self._make_df(1))
        assert regime["regime"] == "Normal / Benign"

    def test_score_2_is_caution(self):
        import dashboard
        regime = dashboard.classify_regime(self._make_df(2))
        assert regime["regime"] == "Caution"

    def test_score_3_is_elevated(self):
        import dashboard
        regime = dashboard.classify_regime(self._make_df(3))
        assert regime["regime"] == "Elevated Risk"

    def test_score_4_is_elevated(self):
        import dashboard
        regime = dashboard.classify_regime(self._make_df(4))
        assert regime["regime"] == "Elevated Risk"

    def test_score_5_is_high_risk(self):
        import dashboard
        regime = dashboard.classify_regime(self._make_df(5))
        assert regime["regime"] == "High Risk"

    def test_score_7_is_high_risk(self):
        import dashboard
        regime = dashboard.classify_regime(self._make_df(7))
        assert regime["regime"] == "High Risk"

    def test_returns_required_keys(self):
        import dashboard
        regime = dashboard.classify_regime(self._make_df(0))
        for key in ("regime", "color", "emoji", "score", "flags", "spread", "delinq", "cpi", "ff", "dgs10"):
            assert key in regime


# ── recession_shapes (dashboard.py) ──────────────────────────────────────────

class TestRecessionShapes:
    def test_no_usrec_column_returns_empty(self):
        import dashboard
        df = pd.DataFrame({"other": [1, 2]}, index=pd.date_range("2020", periods=2, freq="ME"))
        assert dashboard.recession_shapes(df) == []

    def test_single_recession_produces_one_shape(self):
        import dashboard
        idx = pd.date_range("2020-01-31", periods=6, freq="ME")
        usrec = [0, 1, 1, 1, 0, 0]
        df = pd.DataFrame({"USREC": usrec}, index=idx)
        shapes = dashboard.recession_shapes(df)
        assert len(shapes) == 1
        assert shapes[0]["type"] == "rect"

    def test_recession_at_end_of_series_is_closed(self):
        """If a recession runs to the last data point it should still emit a shape."""
        import dashboard
        idx = pd.date_range("2020-01-31", periods=4, freq="ME")
        df = pd.DataFrame({"USREC": [0, 1, 1, 1]}, index=idx)
        shapes = dashboard.recession_shapes(df)
        assert len(shapes) == 1

    def test_no_recession_returns_empty(self):
        import dashboard
        idx = pd.date_range("2020-01-31", periods=4, freq="ME")
        df = pd.DataFrame({"USREC": [0, 0, 0, 0]}, index=idx)
        shapes = dashboard.recession_shapes(df)
        assert shapes == []

    def test_multiple_recessions_produce_multiple_shapes(self):
        import dashboard
        idx = pd.date_range("2020-01-31", periods=8, freq="ME")
        usrec = [0, 1, 1, 0, 0, 1, 1, 0]
        df = pd.DataFrame({"USREC": usrec}, index=idx)
        shapes = dashboard.recession_shapes(df)
        assert len(shapes) == 2
