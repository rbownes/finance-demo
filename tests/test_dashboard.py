"""
Tests for dashboard.py — Streamlit UI helper functions.

Only the pure-Python functions that do not require a running Streamlit
context or a live FRED API call are tested here:
  - recession_shapes
  - classify_regime
  - make_nl_summary

Financial edge cases covered:
  - No recession periods in data
  - Single-month recession
  - Recession extending to the end of the series (still in-progress)
  - Adjacent recession blocks (multiple episodes)
  - Regime classification for all four risk bands
  - Spread at and around inversion boundary
  - Missing USREC column
"""

import numpy as np
import pandas as pd
import pytest

# dashboard.py imports streamlit at module level; we patch it before import.
import sys
from unittest.mock import MagicMock

# Provide lightweight stubs for Streamlit and Plotly so the module can be
# imported in a headless test environment without those packages installed.
for mod_name in [
    "streamlit",
    "plotly",
    "plotly.graph_objects",
    "plotly.express",
    "plotly.subplots",
]:
    if mod_name not in sys.modules:
        sys.modules[mod_name] = MagicMock()

from dashboard import recession_shapes, classify_regime, make_nl_summary


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _monthly_index(start: str = "2010-01-31", periods: int = 36) -> pd.DatetimeIndex:
    return pd.date_range(start=start, periods=periods, freq="ME")


def _base_df(
    fedfunds: float = 3.0,
    dgs10: float = 4.5,
    dralacbn: float = 1.5,
    cpiaucsl_yoy: float = 2.0,
    risk_score: int = 1,
    periods: int = 36,
    usrec: list | None = None,
) -> pd.DataFrame:
    """
    Minimal DataFrame that satisfies classify_regime and recession_shapes.
    Derived columns are computed from scalars for simplicity.
    """
    idx = _monthly_index(periods=periods)
    n = periods

    df = pd.DataFrame(index=idx)
    df["FEDFUNDS"]    = fedfunds
    df["DGS10"]       = dgs10
    df["DRALACBN"]    = dralacbn
    df["spread"]      = dgs10 - fedfunds
    df["CPI_yoy_pct"] = cpiaucsl_yoy
    df["FEDFUNDS_yoy"] = 0.0
    df["DRALACBN_yoy"] = 0.0

    # Risk flags (integer columns expected by classify_regime)
    df["f_inverted"]   = int(df["spread"].iloc[0] < 0)
    df["f_near_inv"]   = int(df["spread"].iloc[0] < 0.5)
    df["f_rising_del"] = 0
    df["f_high_del"]   = int(dralacbn > 1.80)
    df["f_inflation"]  = int(cpiaucsl_yoy > 4.0)
    df["f_rate_hike"]  = 0
    df["f_high_rates"] = int(fedfunds > 5.0)
    df["risk_score"]   = risk_score

    # MoM change columns needed by make_nl_summary
    df["FEDFUNDS_chg"] = 0.0
    df["DRALACBN_chg"] = 0.0
    for lag in range(0, 25):
        df[f"corr_lag{lag}"] = 0.0

    if usrec is not None:
        df["USREC"] = usrec
    else:
        df["USREC"] = 0

    return df


# ===========================================================================
# recession_shapes
# ===========================================================================

class TestRecessionShapes:

    def test_no_recession_returns_empty(self):
        df = _base_df()
        shapes = recession_shapes(df)
        assert shapes == []

    def test_missing_usrec_column_returns_empty(self):
        df = _base_df()
        df = df.drop(columns=["USREC"])
        shapes = recession_shapes(df)
        assert shapes == []

    def test_single_month_recession(self):
        periods = 12
        usrec = [0] * periods
        usrec[5] = 1  # one month recession
        df = _base_df(periods=periods, usrec=usrec)
        shapes = recession_shapes(df)
        # Should produce one rectangle
        assert len(shapes) == 1

    def test_multi_month_recession(self):
        periods = 24
        usrec = [0] * periods
        for i in range(6, 12):
            usrec[i] = 1  # 6-month recession
        df = _base_df(periods=periods, usrec=usrec)
        shapes = recession_shapes(df)
        assert len(shapes) == 1

    def test_two_separate_recessions(self):
        periods = 36
        usrec = [0] * periods
        for i in range(3, 7):
            usrec[i] = 1  # first recession
        for i in range(20, 26):
            usrec[i] = 1  # second recession
        df = _base_df(periods=periods, usrec=usrec)
        shapes = recession_shapes(df)
        assert len(shapes) == 2

    def test_recession_extending_to_end_of_series(self):
        """
        If recession is still in-progress at the end of the data, a shape
        should still be generated (the function must close the rectangle).
        """
        periods = 18
        usrec = [0] * 10 + [1] * 8  # recession starts and never ends
        df = _base_df(periods=periods, usrec=usrec)
        shapes = recession_shapes(df)
        assert len(shapes) == 1

    def test_shape_dicts_have_required_keys(self):
        periods = 12
        usrec = [0] * 4 + [1] * 4 + [0] * 4
        df = _base_df(periods=periods, usrec=usrec)
        shapes = recession_shapes(df)
        for shape in shapes:
            for key in ("type", "x0", "x1", "y0", "y1", "fillcolor"):
                assert key in shape, f"Shape missing key '{key}'"

    def test_shape_type_is_rect(self):
        periods = 12
        usrec = [0] * 4 + [1] * 4 + [0] * 4
        df = _base_df(periods=periods, usrec=usrec)
        shapes = recession_shapes(df)
        for shape in shapes:
            assert shape["type"] == "rect"

    def test_shape_x0_before_x1(self):
        periods = 12
        usrec = [0] * 3 + [1] * 5 + [0] * 4
        df = _base_df(periods=periods, usrec=usrec)
        shapes = recession_shapes(df)
        for shape in shapes:
            assert shape["x0"] < shape["x1"]


# ===========================================================================
# classify_regime
# ===========================================================================

class TestClassifyRegime:

    def test_returns_dict_with_required_keys(self):
        df = _base_df()
        result = classify_regime(df)
        for key in ("regime", "color", "emoji", "score", "flags",
                    "spread", "delinq", "cpi", "ff", "dgs10"):
            assert key in result, f"Missing key: {key}"

    # -- Four risk bands ---------------------------------------------------

    def test_normal_regime(self):
        df = _base_df(risk_score=1)
        result = classify_regime(df)
        assert "Normal" in result["regime"] or "Benign" in result["regime"]

    def test_caution_regime(self):
        df = _base_df(risk_score=2)
        result = classify_regime(df)
        assert "Caution" in result["regime"]

    def test_elevated_risk_regime(self):
        df = _base_df(risk_score=3)
        result = classify_regime(df)
        assert "Elevated" in result["regime"]

    def test_high_risk_regime(self):
        df = _base_df(risk_score=6)
        result = classify_regime(df)
        assert "High Risk" in result["regime"]

    # -- Flag membership ---------------------------------------------------

    def test_inverted_curve_flag_in_flags_list(self):
        df = _base_df(fedfunds=5.5, dgs10=4.0)
        # Manually set f_inverted to 1 in the df (spread < 0)
        df["f_inverted"] = 1
        df["spread"] = -1.5
        result = classify_regime(df)
        flag_text = " ".join(result["flags"]).lower()
        assert "inverted" in flag_text

    def test_no_flags_in_benign_environment(self):
        df = _base_df(
            fedfunds=2.0, dgs10=4.0,
            dralacbn=1.0, cpiaucsl_yoy=2.0,
            risk_score=0,
        )
        # Ensure all flag columns are 0
        for col in ("f_inverted", "f_near_inv", "f_rising_del",
                    "f_high_del", "f_inflation", "f_rate_hike", "f_high_rates"):
            df[col] = 0
        result = classify_regime(df)
        assert result["flags"] == []

    # -- Spread edge cases ------------------------------------------------

    def test_spread_value_returned(self):
        df = _base_df(fedfunds=3.0, dgs10=4.5)
        result = classify_regime(df)
        assert result["spread"] == pytest.approx(1.5)

    def test_negative_spread_returned(self):
        df = _base_df(fedfunds=5.5, dgs10=4.0)
        df["spread"] = -1.5
        result = classify_regime(df)
        assert result["spread"] < 0

    def test_delinquency_value_returned(self):
        df = _base_df(dralacbn=2.1)
        result = classify_regime(df)
        assert result["delinq"] == pytest.approx(2.1)

    def test_cpi_value_returned(self):
        df = _base_df(cpiaucsl_yoy=7.5)
        result = classify_regime(df)
        assert result["cpi"] == pytest.approx(7.5)


# ===========================================================================
# make_nl_summary
# ===========================================================================

class TestMakeNlSummary:

    def _regime(self, **kwargs) -> dict:
        df = _base_df(**kwargs)
        return classify_regime(df), df

    def test_returns_non_empty_string(self):
        regime, df = self._regime()
        summary = make_nl_summary(df, regime)
        assert isinstance(summary, str) and len(summary) > 50

    def test_contains_risk_score(self):
        regime, df = self._regime(risk_score=3)
        summary = make_nl_summary(df, regime)
        assert "3" in summary

    def test_inverted_curve_mentioned_when_inverted(self):
        df = _base_df(fedfunds=5.5, dgs10=4.0, risk_score=5)
        df["spread"] = -1.5
        df["f_inverted"] = 1
        regime = classify_regime(df)
        summary = make_nl_summary(df, regime)
        # Should mention inversion somewhere
        assert "inverted" in summary.lower() or "below zero" in summary.lower() or "-" in summary

    def test_mentions_as_of_date(self):
        regime, df = self._regime()
        summary = make_nl_summary(df, regime)
        # The summary should include a year reference
        assert "20" in summary  # year like 2010, 2020, etc.

    def test_high_risk_summary_advises_stress_test(self):
        df = _base_df(risk_score=5)
        df["f_inverted"] = 1
        df["f_high_del"] = 1
        df["f_inflation"] = 1
        regime = classify_regime(df)
        summary = make_nl_summary(df, regime)
        # High-risk scenario should advise caution
        assert any(word in summary.lower() for word in
                   ("stress", "risk", "elevated", "delinquency", "caution"))
