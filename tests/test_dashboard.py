"""
Test suite for dashboard.py — pure-computation functions.

Streamlit is mocked in conftest.py before this module is imported,
so no running Streamlit server is required.
"""
import numpy as np
import pandas as pd
import pytest

from dashboard import classify_regime, make_nl_summary, recession_shapes


# ── Helpers ────────────────────────────────────────────────────────────────────

def _base_df(n: int = 36, **overrides) -> pd.DataFrame:
    """
    Build a minimal DataFrame that satisfies classify_regime / make_nl_summary.

    All risk flag columns (f_*) are pre-set to zero; override as needed.
    """
    dates = pd.date_range("2020-01-31", periods=n, freq="ME")
    data = {
        "FEDFUNDS":      2.0,
        "DGS10":         3.5,
        "DRALACBN":      1.5,
        "PCE":           15_000.0,
        "CPIAUCSL":      260.0,
        "USREC":         0,
        "spread":        1.5,       # DGS10 − FEDFUNDS
        "FEDFUNDS_chg":  0.0,
        "DGS10_chg":     0.0,
        "DRALACBN_chg":  0.0,
        "FEDFUNDS_yoy":  0.0,
        "CPI_yoy_pct":   2.0,
        "PCE_yoy_pct":   2.0,
        "DRALACBN_yoy":  0.0,
        # Risk flag columns
        "f_inverted":    0,
        "f_near_inv":    0,
        "f_rising_del":  0,
        "f_high_del":    0,
        "f_inflation":   0,
        "f_rate_hike":   0,
        "f_high_rates":  0,
        "risk_score":    0,
    }
    # Add corr_lag{n} columns (required by make_nl_summary)
    for lag in range(0, 25):
        data[f"corr_lag{lag}"] = 0.0

    data.update(overrides)
    expanded = {k: ([v] * n if np.isscalar(v) else v) for k, v in data.items()}
    return pd.DataFrame(expanded, index=dates)


# ── classify_regime — score boundary conditions ────────────────────────────────

class TestClassifyRegime:
    """
    Score thresholds (from dashboard.py lines 137-144):
        score <= 1  → Normal / Benign
        score == 2  → Caution
        score <= 4  → Elevated Risk   (i.e. 3 or 4)
        score >= 5  → High Risk
    """

    def _df_with_score(self, score: int) -> pd.DataFrame:
        return _base_df(risk_score=score)

    def test_score_0_is_normal(self):
        result = classify_regime(self._df_with_score(0))
        assert result["regime"] == "Normal / Benign"

    def test_score_1_is_normal(self):
        result = classify_regime(self._df_with_score(1))
        assert result["regime"] == "Normal / Benign"

    def test_score_2_is_caution(self):
        result = classify_regime(self._df_with_score(2))
        assert result["regime"] == "Caution"

    def test_score_3_is_elevated(self):
        result = classify_regime(self._df_with_score(3))
        assert result["regime"] == "Elevated Risk"

    def test_score_4_is_elevated(self):
        result = classify_regime(self._df_with_score(4))
        assert result["regime"] == "Elevated Risk"

    def test_score_5_is_high_risk(self):
        result = classify_regime(self._df_with_score(5))
        assert result["regime"] == "High Risk"

    def test_score_7_is_high_risk(self):
        result = classify_regime(self._df_with_score(7))
        assert result["regime"] == "High Risk"

    def test_returns_dict_with_required_keys(self):
        result = classify_regime(_base_df())
        for key in ("regime", "color", "emoji", "score", "flags",
                    "spread", "delinq", "cpi", "ff", "dgs10", "row"):
            assert key in result, f"Missing key in regime dict: {key}"

    def test_score_is_integer(self):
        result = classify_regime(_base_df())
        assert isinstance(result["score"], int)

    def test_flags_is_list(self):
        result = classify_regime(_base_df())
        assert isinstance(result["flags"], list)

    def test_no_flags_in_benign_environment(self):
        df = _base_df(risk_score=0)
        result = classify_regime(df)
        assert result["flags"] == []

    def test_inverted_flag_text_in_flags(self):
        df = _base_df(
            risk_score=1,
            f_inverted=1,
            spread=-0.5,
        )
        result = classify_regime(df)
        flag_texts = " ".join(result["flags"]).lower()
        assert "inverted" in flag_texts

    def test_near_inv_flag_only_when_not_inverted(self):
        """near-flat flag should only show when f_inverted == 0."""
        df_near = _base_df(risk_score=1, f_near_inv=1, f_inverted=0)
        result = classify_regime(df_near)
        flag_texts = " ".join(result["flags"]).lower()
        assert "near" in flag_texts or "flat" in flag_texts

        df_both = _base_df(risk_score=2, f_near_inv=1, f_inverted=1)
        result_both = classify_regime(df_both)
        flag_texts_both = " ".join(result_both["flags"]).lower()
        # near-flat should NOT appear when already inverted
        assert "near" not in flag_texts_both and "flat" not in flag_texts_both

    # ── color / emoji ──────────────────────────────────────────────────────────

    def test_color_is_string(self):
        for score in range(8):
            result = classify_regime(self._df_with_score(score))
            assert isinstance(result["color"], str)
            assert result["color"].startswith("#"), f"score {score}: bad color {result['color']}"

    def test_emoji_is_string(self):
        for score in range(8):
            result = classify_regime(self._df_with_score(score))
            assert isinstance(result["emoji"], str)


# ── recession_shapes ───────────────────────────────────────────────────────────

class TestRecessionShapes:
    def test_no_usrec_column_returns_empty(self):
        df = _base_df()
        df_no_rec = df.drop(columns=["USREC"])
        shapes = recession_shapes(df_no_rec)
        assert shapes == []

    def test_no_recession_returns_empty(self):
        df = _base_df(USREC=0)
        shapes = recession_shapes(df)
        assert shapes == []

    def test_single_month_recession(self):
        """A recession lasting one month produces one rect shape."""
        n = 12
        usrec = [0] * 5 + [1] + [0] * 6
        df = _base_df(n=n, USREC=usrec)
        shapes = recession_shapes(df)
        assert len(shapes) == 1
        assert shapes[0]["type"] == "rect"

    def test_multi_month_recession(self):
        """Consecutive recession months collapse into a single shape."""
        n = 12
        usrec = [0] * 3 + [1, 1, 1] + [0] * 6
        df = _base_df(n=n, USREC=usrec)
        shapes = recession_shapes(df)
        assert len(shapes) == 1

    def test_two_separate_recessions(self):
        """Two non-overlapping recession episodes → two shapes."""
        n = 18
        usrec = [0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
        df = _base_df(n=n, USREC=usrec)
        shapes = recession_shapes(df)
        assert len(shapes) == 2

    def test_ongoing_recession_at_series_end(self):
        """If the series ends during a recession, the open episode is still captured."""
        n = 6
        usrec = [0, 0, 0, 1, 1, 1]
        df = _base_df(n=n, USREC=usrec)
        shapes = recession_shapes(df)
        assert len(shapes) == 1
        # The shape's x1 should be the last index of the DataFrame
        assert shapes[0]["x1"] == df.index[-1]

    def test_shape_has_required_plotly_keys(self):
        n = 6
        usrec = [0, 0, 1, 0, 0, 0]
        df = _base_df(n=n, USREC=usrec)
        shape = recession_shapes(df)[0]
        for key in ("type", "xref", "yref", "x0", "x1", "y0", "y1",
                    "fillcolor", "line_width", "layer"):
            assert key in shape, f"Shape missing key: {key}"

    def test_all_recession_returns_one_shape(self):
        """Entire series is a recession → exactly one shape."""
        n = 6
        usrec = [1] * n
        df = _base_df(n=n, USREC=usrec)
        shapes = recession_shapes(df)
        assert len(shapes) == 1


# ── make_nl_summary ────────────────────────────────────────────────────────────

class TestMakeNlSummary:
    def _regime(self, score: int, spread: float = 1.5, delinq: float = 1.5,
                cpi: float = 2.0, ff: float = 2.0, dgs10: float = 3.5) -> dict:
        flags = []
        if spread < 0:
            flags.append("Inverted yield curve")
        if cpi > 4.0:
            flags.append("Inflation above 4%")
        return dict(regime="Normal / Benign" if score <= 1 else "High Risk",
                    emoji="🟢" if score <= 1 else "🔴",
                    color="#4caf7d", score=score,
                    flags=flags, spread=spread, delinq=delinq,
                    cpi=cpi, ff=ff, dgs10=dgs10)

    def test_returns_non_empty_string(self):
        df = _base_df()
        regime = self._regime(score=0)
        summary = make_nl_summary(df, regime)
        assert isinstance(summary, str) and len(summary) > 0

    def test_contains_regime_name(self):
        df = _base_df()
        regime = self._regime(score=0)
        summary = make_nl_summary(df, regime)
        assert regime["regime"] in summary

    def test_inverted_curve_language_present(self):
        df = _base_df(spread=-0.5)
        regime = self._regime(score=2, spread=-0.5)
        summary = make_nl_summary(df, regime)
        assert "inverted" in summary.lower()

    def test_positive_spread_language(self):
        df = _base_df(spread=1.5)
        regime = self._regime(score=0, spread=1.5)
        summary = make_nl_summary(df, regime)
        assert "positive" in summary.lower() or "above" in summary.lower()

    def test_high_inflation_language(self):
        df = _base_df(CPI_yoy_pct=5.0)
        regime = self._regime(score=2, cpi=5.0)
        summary = make_nl_summary(df, regime)
        assert "2%" in summary or "target" in summary.lower()

    def test_high_risk_stress_language(self):
        df = _base_df(risk_score=5)
        regime = self._regime(score=5)
        summary = make_nl_summary(df, regime)
        assert "stress" in summary.lower() or "risk" in summary.lower()
