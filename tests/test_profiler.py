from __future__ import annotations

import json

import pandas as pd
import pytest

from adversarial import (  # type: ignore[import-not-found]
    binary_categorical_target,
    constants_and_all_missing,
    datetime_gaps,
    datetime_text,
    duplicates,
    high_corr_features,
    identifiers,
    imbalanced_bool_target,
    leak_categorical_target,
    leak_numeric_target,
    missing_encoded,
    numeric_as_string,
    rows_partial_missing,
    skewed_with_outliers,
    tiny,
    wide,
)
from nullius.data import profile_dataframe
from nullius.data.types import ColumnKind, Severity
from util import (  # type: ignore[import-not-found]
    association,
    codes,
    column,
    findings_with_code,
    leak_codes,
)


def test_duplicate_rows_detected():
    df, exp = duplicates()
    report = profile_dataframe(df, name="dup")
    assert "DUP_ROWS" in codes(report)
    f = findings_with_code(report, "DUP_ROWS")[0]
    assert f.detail["n_redundant_rows"] == exp["n_redundant"]


def test_encoded_and_real_missingness_detected():
    df, exp = missing_encoded()
    report = profile_dataframe(df, name="missing")
    assert "COL_MISSING" in codes(report)
    assert "MISSING_ENCODED" in codes(report)
    f = findings_with_code(report, "MISSING_ENCODED")[0]
    assert f.columns == ("free_text",)
    assert f.detail["n_encoded"] == exp["n_encoded"]


def test_constants_and_fully_missing_columns():
    df, exp = constants_and_all_missing()
    report = profile_dataframe(df)
    assert exp["codes"] <= codes(report)
    for f in findings_with_code(report, "CONSTANT_COL"):
        assert f.columns[0] in exp["constant_cols"]
    empty = column(report, "empty_col")
    assert empty.kind == ColumnKind.OTHER
    assert empty.n_missing == len(df)


def test_identifier_columns_flagged():
    df, exp = identifiers()
    report = profile_dataframe(df)
    assert "ID_COLUMN" in codes(report)
    for name in exp["identifier_cols"]:
        assert column(report, name).kind == ColumnKind.IDENTIFIER
        assert any(f.code == "ID_COLUMN" for f in column(report, name).findings)
    assert column(report, "product_code").kind == ColumnKind.CATEGORICAL


def test_numeric_text_and_mixed_columns():
    df, exp = numeric_as_string()
    report = profile_dataframe(df)
    assert column(report, exp["numeric_kind"]).kind == ColumnKind.NUMERIC
    assert column(report, exp["mixed_kind"]).kind == ColumnKind.CATEGORICAL
    assert "NUM_AS_STRING" in codes(report)
    assert "NUM_STRING_MIXED" in codes(report)


def test_datetime_text_and_irregular_gaps():
    df, exp = datetime_text()
    report = profile_dataframe(df)
    assert column(report, exp["kind"]).kind == ColumnKind.DATETIME
    assert "DATETIME_AS_STRING" in codes(report)

    df2, _ = datetime_gaps()
    report2 = profile_dataframe(df2)
    assert "DATETIME_IRREGULAR" in codes(report2)


def test_numeric_target_leakage_heuristics():
    df, exp = leak_numeric_target()
    report = profile_dataframe(df, target=exp["target"])
    assert exp["codes"] <= leak_codes(report)
    dup = findings_with_code(report, "LEAK_TARGET_DUPLICATE")[0]
    assert dup.columns == (exp["dup_col"], exp["target"])
    proxy = findings_with_code(report, "LEAK_NEAR_PERFECT_ASSOC")
    assert any(f.columns[0] == exp["proxy_col"] for f in proxy)
    assert all(f.columns[0] != "tenure" for f in proxy)
    # score shows up as a flagged association record as well
    a = association(report, exp["proxy_col"])
    assert a is not None and a.flagged and abs(a.value) > 0.99


def test_categorical_target_leakage_heuristics():
    df, exp = leak_categorical_target()
    report = profile_dataframe(df, target=exp["target"])
    assert exp["codes"] <= leak_codes(report)
    dup = findings_with_code(report, "LEAK_TARGET_DUPLICATE")[0]
    assert dup.columns == (exp["dup_col"], exp["target"])
    proxy = findings_with_code(report, "LEAK_NEAR_PERFECT_ASSOC")
    assert any(f.columns[0] == exp["proxy_col"] for f in proxy)
    imbalance = findings_with_code(report, "TARGET_IMBALANCE")[0]
    assert imbalance.severity == Severity.MEDIUM
    # association via point-biserial is recorded
    a = association(report, exp["proxy_col"])
    assert a is not None and a.method == "point-biserial"


def test_binary_categorical_target_balanced_has_no_leak_flags():
    df, exp = binary_categorical_target()
    report = profile_dataframe(df, target=exp["target"])
    assert leak_codes(report) == set()
    # a genuinely associated (but not derived) categorical feature is fine
    a = association(report, "segment")
    assert a is not None and a.method == "cramers-v"


def test_imbalanced_target_flagged():
    df, exp = imbalanced_bool_target()
    report = profile_dataframe(df, target=exp["target"])
    assert "TARGET_IMBALANCE" in leak_codes(report)
    f = findings_with_code(report, "TARGET_IMBALANCE")[0]
    assert f.severity == Severity.MEDIUM
    assert f.detail["minority_frac"] < 0.1


def test_skew_and_outliers():
    df, exp = skewed_with_outliers()
    report = profile_dataframe(df)
    assert {"HIGH_SKEW", "OUTLIERS"} <= codes(report)
    rev = column(report, "revenue")
    assert rev.numeric is not None and rev.numeric.skew >= 2.0


def test_feature_correlation_highlights():
    df, exp = high_corr_features()
    report = profile_dataframe(df)
    f = findings_with_code(report, "CORR_HIGH")[0]
    assert f.detail["n_near_perfect_pairs"] >= exp["min_near_perfect"]


def test_partial_and_full_missing_rows():
    df, exp = rows_partial_missing()
    report = profile_dataframe(df)
    assert {"ROWS_WITH_MISSING", "ROWS_ALL_MISSING", "COL_MISSING"} <= codes(report)
    all_rows = findings_with_code(report, "ROWS_ALL_MISSING")[0]
    assert all_rows.detail["n_rows"] == 1


def test_tiny_frame_does_not_crash():
    df = tiny()
    report = profile_dataframe(df, name="tiny")
    assert report.overview.n_rows == 3
    assert report.overview.n_cols == 3
    assert len(json.dumps(report.to_dict())) > 0


def test_wide_frame_bounded():
    df = wide()
    report = profile_dataframe(df)
    assert report.overview.n_cols == len(df.columns)
    assert isinstance(report.render_markdown(), str)


def test_report_is_json_serializable_and_complete():
    df, exp = leak_numeric_target()
    report = profile_dataframe(df, target=exp["target"])
    payload = json.loads(json.dumps(report.to_dict()))
    assert payload["overview"]["n_rows"] == len(df)
    assert len(payload["columns"]) == len(df.columns)
    assert any(f["code"] == "LEAK_TARGET_DUPLICATE" for f in payload["leak_findings"])
    assert len(payload["notes"]) >= 3


def test_input_frame_is_never_modified():
    df, exp = identifiers()
    before = df.copy(deep=True)
    before_types = {c: str(df[c].dtype) for c in df.columns}
    profile_dataframe(df, target=None)
    pd.testing.assert_frame_equal(df, before, check_dtype=True)
    assert {c: str(df[c].dtype) for c in df.columns} == before_types


def test_markdown_rendering_contains_sections_and_is_capped():
    df = wide(n_rows=30, p=70)
    report = profile_dataframe(df, name="wide_demo")
    md = report.render_markdown()
    assert "# Data Understanding Report — wide_demo" in md
    assert "## 1. Finding summary" in md
    assert "Scope and limitations" in md
    short = report.render_markdown(max_rows=4)
    assert "omitted" in short  # truncation note present


def test_profiling_is_deterministic():
    df, _ = high_corr_features()
    a = profile_dataframe(df, name="det").to_dict()
    b = profile_dataframe(df, name="det").to_dict()
    a["overview"].pop("created_utc")
    b["overview"].pop("created_utc")
    assert a == b


def test_target_validation_errors():
    df, _ = leak_numeric_target()
    with pytest.raises(ValueError, match="not found"):
        profile_dataframe(df, target="does_not_exist")
    with pytest.raises(TypeError):
        profile_dataframe([1, 2, 3])


def test_empty_dataframe_ok():
    report = profile_dataframe(pd.DataFrame())
    assert report.overview.n_rows == 0
    assert report.overview.n_cols == 0
    assert report.render_markdown()
