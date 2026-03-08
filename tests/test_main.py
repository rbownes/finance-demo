"""
Tests for main.py analytics functions.

All tests use synthetic in-memory DataFrames — no FRED API calls required.
"""

import numpy as np
import pandas as pd
import pytest

from main import (
    align_to_monthly,
    compute_analytics,
    estimate_bp_impact,
    flag_risk,
    RISK_THRESHOLDS,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_monthly_df(n: int = 60, seed: int = 42) -> pd.DataFrame:
    """Return a minimal aligned monthly DataFrame suitable for analytics."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2010-01-31", periods=n, freq="ME")
    df = pd.DataFrame(
        {
            "FEDFUNDS": np.clip(rng.normal(2.5, 1.5, n), 0.05, 10.0),
            "DGS10": np.clip(rng.normal(3.0, 1.0, n), 0.5, 8.0),
            "DRALACBN": np.clip(rng.normal(1.5, 0.4, n), 0.5, 5.0),
            "PCE": rng.uniform(15_000, 20_000, n),
            "CPIAUCSL": rng.uniform(200, 320, n),
        },
        index=idx,
    )
    return df


def _make_raw_dict(n: int = 60) -> dict[str, pd.DataFrame]:
    """Return a fake raw-series dict as returned by fetch_all()."""
    monthly_idx = pd.date_range("2010-01-31", periods=n, freq="ME")
    rng = np.random.default_rng(0)

    def _series(sid, vals):
        return pd.DataFrame({sid: vals}, index=monthly_idx)

    # DRALACBN is quarterly — simulate by spacing it every 3 months
    quarterly_idx = pd.date_range("2010-03-31", periods=n // 3 + 1, freq="QE")
    dralacbn = pd.DataFrame(
        {"DRALACBN": rng.uniform(1.0, 2.5, len(quarterly_idx))},
        index=quarterly_idx,
    )

    return {
        "FEDFUNDS": _series("FEDFUNDS", np.clip(rng.normal(2.5, 1.0, n), 0.05, 10)),
        "DGS10":    _series("DGS10",    np.clip(rng.normal(3.0, 0.8, n), 0.5, 8.0)),
        "DRALACBN": dralacbn,
        "PCE":      _series("PCE",      rng.uniform(15_000, 20_000, n)),
        "CPIAUCSL": _series("CPIAUCSL", rng.uniform(200, 320, n)),
    }


# ── align_to_monthly ───────────────────────────────────────────────────────────

class TestAlignToMonthly:
    def test_returns_dataframe(self):
        raw = _make_raw_dict()
        result = align_to_monthly(raw)
        assert isinstance(result, pd.DataFrame)

    def test_expected_columns_present(self):
        raw = _make_raw_dict()
        result = align_to_monthly(raw)
        for col in ("FEDFUNDS", "DGS10", "DRALACBN", "PCE", "CPIAUCSL"):
            assert col in result.columns, f"Column {col!r} missing"

    def test_no_nan_in_required_columns(self):
        raw = _make_raw_dict()
        result = align_to_monthly(raw)
        assert result[["FEDFUNDS", "DGS10", "CPIAUCSL"]].isna().sum().sum() == 0

    def test_index_is_monthly_period(self):
        raw = _make_raw_dict()
        result = align_to_monthly(raw)
        # All index dates should be month-end (day = last day of month)
        assert all(result.index == result.index.to_period("M").to_timestamp("M"))


# ── compute_analytics ─────────────────────────────────────────────────────────

class TestComputeAnalytics:
    def setup_method(self):
        self.df = _make_monthly_df(n=60)

    def test_returns_dataframe(self):
        result = compute_analytics(self.df)
        assert isinstance(result, pd.DataFrame)

    def test_derived_columns_exist(self):
        result = compute_analytics(self.df)
        expected = [
            "FEDFUNDS_chg", "DGS10_chg", "DRALACBN_chg",
            "FEDFUNDS_yoy", "DGS10_yoy", "DRALACBN_yoy",
            "CPI_yoy_pct", "PCE_yoy_pct", "spread",
            "corr_ff_delinq", "corr_dgs10_delinq",
            "corr_ff_lag3_delinq", "corr_ff_lag6_delinq", "corr_ff_lag12_delinq",
        ]
        for col in expected:
            assert col in result.columns, f"Column {col!r} missing from compute_analytics output"

    def test_spread_is_dgs10_minus_fedfunds(self):
        result = compute_analytics(self.df)
        expected = result["DGS10"] - result["FEDFUNDS"]
        pd.testing.assert_series_equal(result["spread"], expected, check_names=False)

    def test_fedfunds_chg_is_diff(self):
        result = compute_analytics(self.df)
        expected = self.df["FEDFUNDS"].diff()
        pd.testing.assert_series_equal(
            result["FEDFUNDS_chg"], expected, check_names=False
        )

    def test_does_not_mutate_input(self):
        df_copy = self.df.copy()
        compute_analytics(self.df)
        pd.testing.assert_frame_equal(self.df, df_copy)

    def test_corr_values_in_valid_range(self):
        result = compute_analytics(self.df)
        corr_cols = [c for c in result.columns if c.startswith("corr_")]
        for col in corr_cols:
            valid = result[col].dropna()
            assert (valid >= -1.0).all() and (valid <= 1.0).all(), (
                f"{col} has correlation values outside [-1, 1]"
            )


# ── flag_risk ─────────────────────────────────────────────────────────────────

class TestFlagRisk:
    def setup_method(self):
        base = _make_monthly_df(n=60)
        self.df = compute_analytics(base)

    def test_returns_tuple_of_two_dataframes(self):
        result = flag_risk(self.df)
        assert isinstance(result, tuple) and len(result) == 2
        df_out, flags = result
        assert isinstance(df_out, pd.DataFrame)
        assert isinstance(flags, pd.DataFrame)

    def test_risk_score_column_exists(self):
        df_out, _ = flag_risk(self.df)
        assert "risk_score" in df_out.columns

    def test_elevated_risk_column_exists(self):
        df_out, _ = flag_risk(self.df)
        assert "elevated_risk" in df_out.columns

    def test_risk_score_non_negative(self):
        df_out, _ = flag_risk(self.df)
        assert (df_out["risk_score"] >= 0).all()

    def test_elevated_risk_when_score_gte_3(self):
        df_out, _ = flag_risk(self.df)
        mask = df_out["risk_score"] >= 3
        assert df_out.loc[mask, "elevated_risk"].all()
        assert not df_out.loc[~mask, "elevated_risk"].any()

    def test_inverted_curve_flag_fires_correctly(self):
        base = _make_monthly_df(n=60)
        base_analytics = compute_analytics(base)
        # Force a clearly inverted spread
        base_analytics["spread"] = -1.0
        df_out, flags = flag_risk(base_analytics)
        if "inverted_curve" in flags.columns:
            assert flags["inverted_curve"].all()

    def test_does_not_mutate_input(self):
        df_copy = self.df.copy()
        flag_risk(self.df)
        pd.testing.assert_frame_equal(self.df, df_copy)


# ── estimate_bp_impact ────────────────────────────────────────────────────────

class TestEstimateBpImpact:
    def setup_method(self):
        base = _make_monthly_df(n=120)  # longer series for reliable regressions
        self.df = compute_analytics(base)

    def test_returns_dict(self):
        result = estimate_bp_impact(self.df)
        assert isinstance(result, dict)

    def test_expected_lag_keys_present(self):
        result = estimate_bp_impact(self.df)
        for lag in (0, 3, 6, 9, 12):
            assert lag in result, f"Lag {lag} missing from result"

    def test_each_entry_has_required_keys(self):
        result = estimate_bp_impact(self.df)
        required = {
            "beta", "intercept", "r_squared", "se_beta",
            "estimated_impact_bps", "ci_lower_bps", "ci_upper_bps", "n_obs",
        }
        for lag, entry in result.items():
            missing = required - entry.keys()
            assert not missing, f"Lag {lag} missing keys: {missing}"

    def test_r_squared_in_valid_range(self):
        result = estimate_bp_impact(self.df)
        for lag, entry in result.items():
            assert 0.0 <= entry["r_squared"] <= 1.0, (
                f"R² out of range at lag {lag}: {entry['r_squared']}"
            )

    def test_ci_ordering(self):
        result = estimate_bp_impact(self.df)
        for lag, entry in result.items():
            assert entry["ci_lower_bps"] <= entry["estimated_impact_bps"] <= entry["ci_upper_bps"], (
                f"CI ordering violated at lag {lag}"
            )

    def test_different_bp_change_scales_linearly(self):
        result_50 = estimate_bp_impact(self.df, bp_change=50.0)
        result_100 = estimate_bp_impact(self.df, bp_change=100.0)
        for lag in result_50:
            ratio = result_100[lag]["estimated_impact_bps"] / result_50[lag]["estimated_impact_bps"]
            assert abs(ratio - 2.0) < 1e-9, (
                f"Expected 2× scaling for 100bp vs 50bp at lag {lag}, got {ratio:.6f}"
            )

    def test_zero_bp_change_gives_zero_impact(self):
        result = estimate_bp_impact(self.df, bp_change=0.0)
        for lag, entry in result.items():
            assert entry["estimated_impact_bps"] == pytest.approx(0.0), (
                f"Expected zero impact for 0bp shock at lag {lag}"
            )

    def test_n_obs_is_positive_integer(self):
        result = estimate_bp_impact(self.df)
        for lag, entry in result.items():
            assert isinstance(entry["n_obs"], (int, np.integer))
            assert entry["n_obs"] > 0

    def test_insufficient_data_returns_empty(self):
        # Only 10 rows — below the 24-row minimum
        tiny_df = _make_monthly_df(n=10)
        tiny_df = compute_analytics(tiny_df)
        result = estimate_bp_impact(tiny_df)
        assert result == {}, "Expected empty dict for insufficient data"

    def test_known_regression_values(self):
        """Verify OLS math with a deterministic synthetic relationship."""
        n = 100
        idx = pd.date_range("2000-01-31", periods=n, freq="ME")
        # Construct: DRALACBN_chg = 0.3 * FEDFUNDS_chg + noise
        rng = np.random.default_rng(7)
        ff_chg = rng.normal(0, 0.5, n)
        del_chg = 0.3 * ff_chg + rng.normal(0, 0.05, n)

        df = pd.DataFrame(
            {"FEDFUNDS_chg": ff_chg, "DRALACBN_chg": del_chg},
            index=idx,
        )
        result = estimate_bp_impact(df, bp_change=100.0)
        # At lag=0, beta ≈ 0.3, impact for 100bp ≈ 0.3pp = 30bps
        assert abs(result[0]["beta"] - 0.3) < 0.05, (
            f"Expected beta ≈ 0.30, got {result[0]['beta']:.4f}"
        )
        assert abs(result[0]["estimated_impact_bps"] - 30.0) < 5.0, (
            f"Expected ~30bps impact, got {result[0]['estimated_impact_bps']:.2f}bps"
        )
