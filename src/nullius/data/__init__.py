"""Data Understanding Engine (Phase 1).

Produces a `DataUnderstandingReport` (spec §3) without ever modifying the input
dataset: schema, data types, missingness, duplicates, distributions, outliers,
skewness, categorical imbalance, correlations, identifiers, suspicious columns,
and *heuristic* leakage/proxy candidates (target-aware).
"""

from __future__ import annotations

from nullius.data.profiler import hash_file, load_dataset, profile_dataframe
from nullius.data.records import (
    AssociationRecord,
    CategoricalStats,
    ColumnProfile,
    DataUnderstandingReport,
    DateTimeStats,
    Finding,
    NumericStats,
    Overview,
    TextStats,
)
from nullius.data.types import ColumnKind, Severity

__all__ = [
    "AssociationRecord",
    "CategoricalStats",
    "ColumnKind",
    "ColumnProfile",
    "DataUnderstandingReport",
    "DateTimeStats",
    "Finding",
    "NumericStats",
    "Overview",
    "Severity",
    "TextStats",
    "hash_file",
    "load_dataset",
    "profile_dataframe",
]
