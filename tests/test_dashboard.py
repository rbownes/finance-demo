"""
Tests for dashboard.py pure-logic functions.

Streamlit is replaced by a MagicMock stub in conftest.py, so the module-level
st.set_page_config() call is a no-op and @st.cache_data simply returns the
function unchanged — allowing us to call the underlying business logic directly.

Coverage areas:
  - recession_shapes: period detection, open-ended recessions, missing column
  - classify_regime: score → label mapping and active-flag list construction
  - make_nl_summary: text content, inversion/non-inversion wording
"""
import pandas as pd
import numpy as np
import pytest

import dashboard


# ── Helpers ───────────────────────────────────────────────────────────────────

def _monthly_index(n: int = 36, start: str = "2020-01-31") -> pd.DatetimeIndex:
    return pd.date_range(start, periods=n, freq="ME")


def _base_df(
    n: int = 36,
    spread: float = 1.0,
    dralacbn: float = 1.0,
    cpi_yoy_pct: float = 2.5,
    fedfunds: float = 3.0,
    dgs10: float = 4.0,
    risk_score: int = 0,
    usrec: int = 0,
) -> pd.DataFrame:
    """Minimal DataFrame that satisfies classify_regime and make_nl_summary."""
    idx = _monthly_index(n)
    df = pd.DataFrame(
        {
            "FEDFUNDS":     fedfunds,
            "DGS10":        dgs10,
            "DRALACBN":     dralacbn,
            "CPIAUCSL":     280.0,
            "PCE":          17_000.0,
            "USREC":        usrec,
            "spread":       spread,
            "FEDFUNDS_chg": 0.0,
            "DGS10_chg":    0.0,
            "DRALACBN_chg": 0.0,
            "FEDFUNDS_yoy": 0.5,
            "CPI_yoy_pct":  cpi_yoy_pct,
            "PCE_yoy_pct":  2.0,
            "DRALACBN_yoy": 0.05,
            "risk_score":   risk_score,
            # Individual flag columns
            "f_inverted":   int(spread < 0),
            "f_near_inv":   int(spread < 0.5),
            "f_rising_del": 0,
            "f_high_del":   0,
            "f_inflation":  0,
            "f_rate_hike":  0,
            "f_high_rates": 0,
        },
        index=idx,
        dtype=float,
    )
    # Add rolling corr lag columns required by make_nl_summary
    for lag in range(0, 25):
        df[f"corr_lag{lag}"] = 0.3
    return df


# ── recession_shapes ──────────────────────────────────────────────────────────

class TestRecessionShapes:
    """Tests for the recession_shapes() helper."""

    def test_returns_empty_list_when_usrec_column_absent(self):
        idx = _monthly_index(12)
        df = pd.DataFrame({"spread": 1.0}, index=idx)
        shapes = dashboard.recession_shapes(df)
        assert shapes == []

    def test_returns_empty_list_when_no_recession_periods(self):
        idx = _monthly_index(24)
        df = pd.DataFrame({"USREC": 0}, index=idx, dtype=float)
        shapes = dashboard.recession_shapes(df)
        assert shapes == []

    def test_single_recession_produces_one_shape(self):
        idx = _monthly_index(24)
        usrec = [0] * 24
        # Mark months 6-9 as recession
        for i in range(6, 10):
            usrec[i] = 1
        df = pd.DataFrame({"USREC": usrec}, index=idx, dtype=float)
        shapes = dashboard.recession_shapes(df)
        assert len(shapes) == 1

    def test_recession_shape_has_required_keys(self):
        idx = _monthly_index(12)
        usrec = [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
        df = pd.DataFrame({"USREC": usrec}, index=idx, dtype=float)
        shapes = dashboard.recession_shapes(df)
        shape = shapes[0]
        for key in ("type", "x0", "x1", "y0", "y1", "fillcolor"):
            assert key in shape, f"Missing key: {key}"

    def test_two_separate_recessions_produce_two_shapes(self):
        idx = _monthly_index(24)
        usrec = [0] * 24
        for i in range(2, 5):
            usrec[i] = 1
        for i in range(15, 19):
            usrec[i] = 1
        df = pd.DataFrame({"USREC": usrec}, index=idx, dtype=float)
        shapes = dashboard.recession_shapes(df)
        assert len(shapes) == 2

    def test_open_ended_recession_produces_shape_ending_at_last_date(self):
        """A recession that hasn't ended yet must still produce a shape."""
        idx = _monthly_index(12)
        usrec = [0] * 6 + [1] * 6   # starts at month 6, never ends
        df = pd.DataFrame({"USREC": usrec}, index=idx, dtype=float)
        shapes = dashboard.recession_shapes(df)
        assert len(shapes) == 1
        # x1 should be the last index date
        assert shapes[0]["x1"] == idx[-1]

    def test_entire_period_in_recession(self):
        idx = _monthly_index(6)
        df = pd.DataFrame({"USREC": 1}, index=idx, dtype=float)
        shapes = dashboard.recession_shapes(df)
        assert len(shapes) == 1


# ── classify_regime ───────────────────────────────────────────────────────────

class TestClassifyRegime:
    """Tests for classify_regime score-to-label mapping and flag extraction."""

    def _df_with_score(self, score: int, spread: float = 1.5) -> pd.DataFrame:
        return _base_df(n=6, spread=spread, risk_score=score)

    # ── regime labels ─────────────────────────────────────────────────────────

    def test_score_0_is_normal_benign(self):
        regime = dashboard.classify_regime(self._df_with_score(0))
        assert regime["regime"] == "Normal / Benign"

    def test_score_1_is_normal_benign(self):
        regime = dashboard.classify_regime(self._df_with_score(1))
        assert regime["regime"] == "Normal / Benign"

    def test_score_2_is_caution(self):
        regime = dashboard.classify_regime(self._df_with_score(2))
        assert regime["regime"] == "Caution"

    def test_score_3_is_elevated_risk(self):
        regime = dashboard.classify_regime(self._df_with_score(3))
        assert regime["regime"] == "Elevated Risk"

    def test_score_4_is_elevated_risk(self):
        regime = dashboard.classify_regime(self._df_with_score(4))
        assert regime["regime"] == "Elevated Risk"

    def test_score_5_is_high_risk(self):
        regime = dashboard.classify_regime(self._df_with_score(5))
        assert regime["regime"] == "High Risk"

    def test_score_7_is_high_risk(self):
        regime = dashboard.classify_regime(self._df_with_score(7))
        assert regime["regime"] == "High Risk"

    # ── return structure ──────────────────────────────────────────────────────

    def test_returns_dict_with_required_keys(self):
        regime = dashboard.classify_regime(self._df_with_score(0))
        for key in ("regime", "color", "emoji", "score", "flags",
                    "spread", "delinq", "cpi", "ff", "dgs10"):
            assert key in regime, f"Missing key: {key}"

    def test_score_matches_input(self):
        regime = dashboard.classify_regime(self._df_with_score(4))
        assert regime["score"] == 4

    def test_spread_value_returned_correctly(self):
        regime = dashboard.classify_regime(self._df_with_score(0, spread=1.23))
        assert regime["spread"] == pytest.approx(1.23)

    # ── color coding ──────────────────────────────────────────────────────────

    def test_normal_regime_has_green_color(self):
        regime = dashboard.classify_regime(self._df_with_score(0))
        assert regime["color"] == "#4caf7d"

    def test_caution_regime_has_gold_color(self):
        regime = dashboard.classify_regime(self._df_with_score(2))
        assert regime["color"] == "#f5b942"

    def test_high_risk_regime_has_red_color(self):
        regime = dashboard.classify_regime(self._df_with_score(6))
        assert regime["color"] == "#f55c5c"

    # ── active flags ──────────────────────────────────────────────────────────

    def test_inverted_curve_flag_appears_when_f_inverted_set(self):
        df = _base_df(n=6, spread=-0.5, risk_score=3)
        df["f_inverted"] = 1.0
        regime = dashboard.classify_regime(df)
        assert any("nverted" in f for f in regime["flags"])

    def test_near_flat_flag_when_near_inv_but_not_inverted(self):
        df = _base_df(n=6, spread=0.3, risk_score=1)
        df["f_inverted"] = 0.0
        df["f_near_inv"] = 1.0
        regime = dashboard.classify_regime(df)
        assert any("flat" in f.lower() for f in regime["flags"])

    def test_no_flags_when_all_conditions_safe(self):
        df = _base_df(n=6, spread=1.5, risk_score=0)
        for col in ["f_inverted", "f_near_inv", "f_rising_del",
                    "f_high_del", "f_inflation", "f_rate_hike", "f_high_rates"]:
            df[col] = 0.0
        regime = dashboard.classify_regime(df)
        assert regime["flags"] == []

    def test_high_delinquency_flag_when_f_high_del_set(self):
        df = _base_df(n=6, dralacbn=2.0, risk_score=1)
        df["f_high_del"] = 1.0
        regime = dashboard.classify_regime(df)
        assert any("elinquency" in f for f in regime["flags"])

    def test_high_rates_flag_when_f_high_rates_set(self):
        df = _base_df(n=6, fedfunds=6.0, risk_score=1)
        df["f_high_rates"] = 1.0
        regime = dashboard.classify_regime(df)
        assert any("rate" in f.lower() for f in regime["flags"])


# ── make_nl_summary ───────────────────────────────────────────────────────────

class TestMakeNlSummary:
    """Tests for make_nl_summary text generation."""

    def _regime(self, spread: float = 1.5, score: int = 0,
                delinq: float = 1.0, cpi: float = 2.5,
                ff: float = 3.0, dgs10: float = 4.5) -> dict:
        flags = []
        if spread < 0:
            flags.append("Inverted yield curve")
        return dict(
            regime="Normal / Benign" if score <= 1 else "Elevated Risk",
            emoji="🟢",
            color="#4caf7d",
            score=score,
            flags=flags,
            spread=spread,
            delinq=delinq,
            cpi=cpi,
            ff=ff,
            dgs10=dgs10,
        )

    def test_returns_non_empty_string(self):
        df = _base_df()
        regime = self._regime()
        summary = dashboard.make_nl_summary(df, regime)
        assert isinstance(summary, str)
        assert len(summary) > 100

    def test_summary_contains_regime_label(self):
        df = _base_df()
        regime = self._regime(score=0)
        summary = dashboard.make_nl_summary(df, regime)
        assert "Normal / Benign" in summary

    def test_inverted_curve_text_when_spread_negative(self):
        df = _base_df(spread=-0.3)
        regime = self._regime(spread=-0.3, score=3)
        summary = dashboard.make_nl_summary(df, regime)
        assert "inverted" in summary.lower()

    def test_positive_spread_text_when_spread_positive(self):
        df = _base_df(spread=1.5)
        regime = self._regime(spread=1.5)
        summary = dashboard.make_nl_summary(df, regime)
        # Should mention positive slope / maturity transformation
        assert "positive" in summary.lower() or "above" in summary.lower()

    def test_cpi_value_appears_in_summary(self):
        df = _base_df(cpi_yoy_pct=3.5)
        regime = self._regime(cpi=3.5)
        summary = dashboard.make_nl_summary(df, regime)
        # The CPI percentage should appear formatted in the text
        assert "3.5" in summary or "3.50" in summary

    def test_high_inflation_wording_when_cpi_above_2_5(self):
        df = _base_df(cpi_yoy_pct=5.0)
        regime = self._regime(cpi=5.0, score=1)
        summary = dashboard.make_nl_summary(df, regime)
        assert "target" in summary.lower() or "higher" in summary.lower()

    def test_low_inflation_wording_when_cpi_at_2(self):
        df = _base_df(cpi_yoy_pct=2.0)
        regime = self._regime(cpi=2.0)
        summary = dashboard.make_nl_summary(df, regime)
        assert "2%" in summary or "target" in summary.lower()

    def test_elevated_risk_wording_when_score_ge_3(self):
        df = _base_df(risk_score=4)
        regime = self._regime(score=4)
        summary = dashboard.make_nl_summary(df, regime)
        # High-risk summary should mention stress-testing or risk flags
        assert "risk" in summary.lower()

    def test_delinquency_value_appears_in_summary(self):
        df = _base_df(dralacbn=1.75)
        regime = self._regime(delinq=1.75)
        summary = dashboard.make_nl_summary(df, regime)
        assert "1.75" in summary
