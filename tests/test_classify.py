from __future__ import annotations

import numpy as np
import pandas as pd

from nullius.data.classify import classify_column
from nullius.data.types import ColumnKind


def _kind(name: str, series: pd.Series) -> ColumnKind:
    return classify_column(name, series, int(series.notna().sum())).kind


def test_numeric_float():
    s = pd.Series([1.5, 2.5, np.nan, 4.0])
    assert _kind("price", s) == ColumnKind.NUMERIC


def test_numeric_int_unique_plain_name_stays_numeric():
    s = pd.Series(np.arange(500, dtype="int64"))
    assert _kind("measure", s) == ColumnKind.NUMERIC


def test_numeric_int_unique_id_name_is_identifier():
    s = pd.Series(np.arange(500, dtype="int64"))
    assert _kind("customer_id", s) == ColumnKind.IDENTIFIER


def test_text_parses_as_numeric():
    s = pd.Series(["1.5", "2.5", "3", "4.75"])
    res = classify_column("amount", s, 4)
    assert res.kind == ColumnKind.NUMERIC
    assert res.numeric_ratio == 1.0


def test_text_parses_as_datetime():
    s = pd.Series(["2024-01-01", "2024-06-15", "2024-12-31", "2023-02-28"])
    res = classify_column("ts", s, 4)
    assert res.kind == ColumnKind.DATETIME
    assert res.datetime_ratio == 1.0


def test_integer_codes_do_not_become_datetime():
    s = pd.Series(["20240101", "20240102", "20240103", "20240104"])
    res = classify_column("code", s, 4)
    assert res.kind == ColumnKind.NUMERIC  # numeric beats datetime for int-like text


def test_datetime_dtype():
    s = pd.Series(pd.to_datetime(["2024-01-01", "2024-01-02"]))
    assert _kind("when", s) == ColumnKind.DATETIME


def test_bool_dtype():
    assert _kind("flag", pd.Series([True, False, True])) == ColumnKind.BOOLEAN


def test_category_dtype():
    s = pd.Series(pd.Categorical(["a", "b", "a"]))
    assert _kind("seg", s) == ColumnKind.CATEGORICAL


def test_low_cardinality_text_is_categorical():
    s = pd.Series(["A", "B", "A", "B", "C"] * 40)
    assert _kind("plan", s) == ColumnKind.CATEGORICAL


def test_unique_long_strings_are_text():
    s = pd.Series([f"long free-form note number {i} " + "words " * 25 for i in range(1100)])
    assert _kind("comments", s) == ColumnKind.TEXT


def test_uuid_like_unique_short_strings_are_identifiers():
    import uuid as _uuid

    s = pd.Series([str(_uuid.uuid4()) for _ in range(200)])
    assert _kind("key", s) == ColumnKind.IDENTIFIER


def test_mixed_numeric_text_is_categorical_with_ratio():
    s = pd.Series([str(i) for i in range(50)] + ["x"] * 50)
    res = classify_column("odd", s, 100)
    assert res.kind == ColumnKind.CATEGORICAL
    assert res.numeric_ratio == 0.5


def test_all_missing_string_column_is_other():
    s = pd.Series([np.nan, None, pd.NA], dtype="string")
    res = classify_column("gone", s, 0)
    assert res.kind == ColumnKind.OTHER


def test_single_repeated_string_is_categorical():
    s = pd.Series(["K", "K", "K"])
    assert _kind("const", s) == ColumnKind.CATEGORICAL
