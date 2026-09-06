from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nullius.data import detectors as det


def test_numeric_stats_constant_column():
    s = pd.Series([5.0] * 20)
    st = det.numeric_stats(s)
    assert st.mean == 5.0
    assert st.std == 0.0
    assert not st.has_outliers


def test_symmetric_data_has_near_zero_skew():
    rng = np.random.default_rng(1)
    st = det.numeric_stats(pd.Series(rng.normal(0, 1, 4000)))
    assert st.skew is not None and abs(st.skew) < 0.05


def test_lognormal_skew_positive():
    rng = np.random.default_rng(2)
    st = det.numeric_stats(pd.Series(np.exp(rng.normal(0, 1.5, 5000))))
    assert st.skew is not None and st.skew > 1.0


def test_outlier_detection_finds_injected_spikes():
    rng = np.random.default_rng(3)
    vals = rng.normal(0, 1, 5000)
    vals[:5] = 60.0
    st = det.numeric_stats(pd.Series(vals))
    # the 5 injected spikes are far beyond any plausible natural tail
    assert st.n_outliers_tukey >= 5
    assert st.n_outliers_mad >= 5
    assert st.n_outliers_mad <= 25  # natural 3-sigma tail stays small


def test_categorical_entropy_balanced_binary_is_one_bit():
    s = pd.Series(["a"] * 500 + ["b"] * 500)
    st = det.categorical_stats(s)
    assert st.entropy_bits == pytest.approx(1.0, abs=1e-9)
    assert st.imbalance_ratio == pytest.approx(1.0)


def test_categorical_imbalance_ratio():
    s = pd.Series(["a"] * 900 + ["b"] * 100)
    st = det.categorical_stats(s)
    assert st.imbalance_ratio == pytest.approx(9.0)
    assert st.dominant_fraction == pytest.approx(0.9)


def test_missing_encoding_detector_counts():
    s = pd.Series(["ok", "", "NA", " ", "nan", "-", "fine", "null"])
    findings = det.missing_encoding_findings(s, "txt", len(s))
    assert len(findings) == 1
    f = findings[0]
    assert f.code == "MISSING_ENCODED"
    assert f.detail["n_encoded"] == 6  # '', 'NA', ' ', 'nan', '-', 'null'


def test_missing_encoding_ignores_numbers():
    s = pd.Series([1, 2, 3, np.nan, 4])
    assert det.missing_encoding_findings(s, "num", 4) == []


def test_categorical_findings_flags_extreme_imbalance():
    s = pd.Series(["a"] * 950 + ["b"] * 50)
    st = det.categorical_stats(s)
    findings = det.categorical_findings(st, "seg")
    codes = {f.code for f in findings}
    assert "CAT_IMBALANCE" in codes


def test_categorical_findings_flags_near_constant():
    s = pd.Series(["a"] * 980 + ["b"] * 20)
    st = det.categorical_stats(s)
    findings = det.categorical_findings(st, "seg")
    codes = {f.code for f in findings}
    assert "NEAR_CONSTANT_COL" in codes
