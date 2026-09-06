"""Infer a column's role (ColumnKind) from its stored dtype and content.

The input is never cast: classification only *describes* what the column looks
like. Columns stored as text that parse cleanly as numbers/datetimes are marked
NUMERIC/DATETIME with a finding that a typed parse is required before modeling.

Thresholds here are deliberately conservative and documented; they are
heuristics, not laws, and the profiler says so in its notes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
from pandas.api import types as pdt

from nullius.data.types import ColumnKind

#: Sample cap for expensive content probes (parsing text as numbers/datetimes).
_SAMPLE = 5000

_ID_NAME_RE = re.compile(
    r"(?:^|_)(?:id|uuid|guid|key)(?:_|$)|(?:^|_)(?:rowid|recordid)$", re.IGNORECASE
)
_UUID_LIKE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)
_LONG_HEX = re.compile(r"^[0-9a-fA-F]{16,64}$")
_LONG_TOKEN = re.compile(r"^[A-Za-z0-9_\-\.]{16,128}$")

#: numeric ratio above which a text column is best treated as numeric
FULL_NUMERIC_RATIO = 0.98
#: numeric ratio above which a text column is suspiciously mixed
PARTIAL_NUMERIC_RATIO = 0.5
#: datetime parse success above which a text column is treated as datetime
DATETIME_RATIO = 0.9
#: uniqueness ratio above which a short text column is an identifier candidate
ID_UNIQUE_RATIO = 0.98
#: minimum distinct values for an identifier candidate to be meaningful
ID_MIN_UNIQUE = 100
#: mean length above which a high-cardinality column is free text, not an id
ID_MAX_MEAN_LENGTH = 64
TEXT_MEAN_LENGTH = 80
TEXT_MIN_UNIQUE = 1000


def _name_suggests_id(name: str) -> bool:
    return bool(_ID_NAME_RE.search(name.strip().lower()))


def _is_uuid_like(value: str) -> bool:
    return bool(_UUID_LIKE.match(value))


def _is_hash_like(value: str) -> bool:
    return bool(_LONG_HEX.match(value) or _LONG_TOKEN.match(value))


@dataclass(frozen=True)
class Classification:
    kind: ColumnKind
    #: ratio of non-missing text values that parse as numbers (text cols only)
    numeric_ratio: float | None = None
    #: ratio of non-missing text values that parse as datetimes (text cols only)
    datetime_ratio: float | None = None
    reason: str = ""


def _sample_values(series: pd.Series) -> np.ndarray:
    """Deterministic content probe: up to _SAMPLE non-missing unique values."""
    vals = series.dropna().astype(str)
    if len(vals) > _SAMPLE:
        # unique() is order-preserving in pandas; take a strided sample
        uniq = vals.unique()
        if len(uniq) > _SAMPLE:
            step = max(1, len(uniq) // _SAMPLE)
            vals = uniq[::step]
        else:
            vals = uniq
    return np.asarray(vals)


def _numeric_ratio(series: pd.Series) -> float | None:
    vals = _sample_values(series)
    if len(vals) == 0:
        return None
    parsed = pd.to_numeric(pd.Series(vals), errors="coerce")
    return float(parsed.notna().mean())


def _datetime_ratio(series: pd.Series) -> float | None:
    vals = _sample_values(series)
    if len(vals) == 0:
        return None
    try:
        parsed = pd.to_datetime(pd.Series(vals), errors="coerce", format="mixed")
    except (ValueError, TypeError):
        return 0.0
    if parsed is None:
        return 0.0
    return float(parsed.notna().mean())


def classify_column(name: str, series: pd.Series, n_nonnull: int) -> Classification:
    """Infer the role of ``series``. Never mutates ``series``."""
    if pdt.is_datetime64_any_dtype(series):
        return Classification(ColumnKind.DATETIME, reason="stored as datetime dtype")
    if pdt.is_bool_dtype(series):
        return Classification(ColumnKind.BOOLEAN, reason="stored as boolean dtype")
    if isinstance(series.dtype, pd.CategoricalDtype):
        return Classification(
            ColumnKind.CATEGORICAL, reason="stored as categorical dtype"
        )

    if pdt.is_numeric_dtype(series):
        n_unique = int(series.nunique(dropna=True))
        if n_nonnull > 0 and n_unique == n_nonnull and _name_suggests_id(name):
            return Classification(ColumnKind.IDENTIFIER, reason="numeric + id-like name")
        return Classification(ColumnKind.NUMERIC, reason="stored as numeric dtype")

    if pdt.is_string_dtype(series) or pdt.is_object_dtype(series):
        if n_nonnull == 0:
            return Classification(ColumnKind.OTHER, reason="column is entirely missing")
        n_unique = int(series.nunique(dropna=True))
        # Numeric parsing is probed before datetime parsing: integer codes such as
        # "20240101" parse as dates too, and numeric beats datetime for them.
        num_ratio = _numeric_ratio(series)
        if num_ratio is not None and num_ratio >= FULL_NUMERIC_RATIO:
            return Classification(
                ColumnKind.NUMERIC,
                numeric_ratio=num_ratio,
                reason=f"text parses as numeric ({num_ratio:.0%} of sampled values)",
            )
        dt_ratio = _datetime_ratio(series)
        if dt_ratio is not None and dt_ratio >= DATETIME_RATIO:
            return Classification(
                ColumnKind.DATETIME,
                datetime_ratio=dt_ratio,
                reason=f"text parses as datetime ({dt_ratio:.0%} of sampled values)",
            )

        if _name_suggests_id(name):
            return Classification(ColumnKind.IDENTIFIER, reason="id-like column name")
        if n_unique == 1:
            return Classification(ColumnKind.CATEGORICAL, reason="single repeated value")

        # content-based identifier / text detection on non-missing values
        uniq_ratio = n_unique / n_nonnull if n_nonnull else 0.0
        sample = _sample_values(series)
        lengths = np.asarray([len(v) for v in sample], dtype=float)
        mean_len = float(np.mean(lengths)) if len(lengths) else 0.0

        uuid_frac = float(np.mean([_is_uuid_like(v) for v in sample]))
        hash_frac = float(np.mean([_is_hash_like(v) for v in sample]))
        if n_unique >= ID_MIN_UNIQUE and (
            uuid_frac >= 0.9 or (uniq_ratio >= ID_UNIQUE_RATIO and hash_frac >= 0.9)
        ):
            return Classification(ColumnKind.IDENTIFIER, reason="uuid/hash-like values")
        if (
            n_unique >= ID_MIN_UNIQUE
            and uniq_ratio >= ID_UNIQUE_RATIO
            and mean_len <= ID_MAX_MEAN_LENGTH
        ):
            return Classification(
                ColumnKind.IDENTIFIER,
                reason=(
                    f"short high-cardinality strings "
                    f"({uniq_ratio:.0%} unique, mean length {mean_len:.0f})"
                ),
            )
        if mean_len >= TEXT_MEAN_LENGTH and n_unique >= TEXT_MIN_UNIQUE:
            return Classification(
                ColumnKind.TEXT,
                reason=f"long high-cardinality strings (mean length {mean_len:.0f})",
            )

        if num_ratio is not None and num_ratio >= PARTIAL_NUMERIC_RATIO:
            return Classification(
                ColumnKind.CATEGORICAL,
                numeric_ratio=num_ratio,
                reason=(
                    f"mixed text: only {num_ratio:.0%} of sampled values parse as "
                    "numeric — investigate before modeling"
                ),
            )
        return Classification(
            ColumnKind.CATEGORICAL, reason="low-cardinality or heterogeneous text"
        )

    return Classification(ColumnKind.OTHER, reason=f"unsupported stored dtype {series.dtype}")
