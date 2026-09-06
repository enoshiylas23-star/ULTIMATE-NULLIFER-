"""Shared enums for the Data Understanding Engine."""

from __future__ import annotations

import enum


class Severity(enum.IntEnum):
    """How seriously a finding should be taken.

    Ordering is intentional so findings can be sorted
    ``critical > high > medium > low > info``.
    """

    INFO = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class ColumnKind(enum.StrEnum):
    """Inferred role of a column after inspection.

    This is an *inference*, recorded alongside the stored dtype so the two can
    never be confused. The input is never cast: NUMERIC/DATETIME kinds derived
    from text content carry a finding explaining that a parse is needed.
    """

    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    BOOLEAN = "boolean"
    DATETIME = "datetime"
    TEXT = "text"
    IDENTIFIER = "identifier"
    OTHER = "other"
