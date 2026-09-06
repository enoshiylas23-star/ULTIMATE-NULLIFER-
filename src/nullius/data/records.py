"""Structured records produced by the Data Understanding Engine.

All records are JSON-serializable via ``to_dict``/``from_dict`` so they can be
persisted by experiment memory (Phase 9) and rendered by report builders
without re-running the analysis.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from nullius.data.types import ColumnKind, Severity


@dataclass(frozen=True)
class Finding:
    """A single machine-readable finding.

    ``method`` is a short, honest label of the numerical procedure that produced
    the finding; the renderer attaches the assumption notes that go with it.
    """

    code: str
    severity: Severity
    message: str
    columns: tuple[str, ...] = ()
    method: str = ""
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["severity"] = self.severity.name
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Finding":
        return cls(
            code=d["code"],
            severity=Severity[d["severity"]],
            message=d["message"],
            columns=tuple(d.get("columns", ())),
            method=d.get("method", ""),
            detail=d.get("detail", {}),
        )


@dataclass(frozen=True)
class NumericStats:
    """Summary statistics for a NUMERIC column.

    Methods and their assumptions (recorded per spec §1):
    - quantiles: linear interpolation; assume the column is at least interval-scaled.
    - skewness: adjusted Fisher–Pearson coefficient of the sample; unreliable for
      very small samples; flag only, never a normality verdict by itself.
    - kurtosis: excess kurtosis (Fisher); requires >= 4 non-missing values.
    - outliers: Tukey fences (1.5 x IQR) and a robust z-score on the median
      absolute deviation (3 x 1.4826 x MAD). Both are generic, distribution-free
      screening rules; whether an extreme value is an *error* is a domain call.
    """

    count: int
    mean: float | None = None
    std: float | None = None
    min: float | None = None
    q1: float | None = None
    median: float | None = None
    q3: float | None = None
    max: float | None = None
    skew: float | None = None
    kurtosis: float | None = None
    n_outliers_tukey: int = 0
    n_outliers_mad: int = 0

    @property
    def has_outliers(self) -> bool:
        return self.n_outliers_tukey > 0 or self.n_outliers_mad > 0


@dataclass(frozen=True)
class CategoricalStats:
    """Summary for CATEGORICAL columns (incl. BOOLEAN-as-category).

    entropy is Shannon entropy over the empirical category distribution, in
    bits. imbalance_ratio = largest category count / smallest nonzero category
    count; a high ratio indicates a skewed distribution where accuracy-style
    summaries can mislead.
    """

    cardinality: int
    top: tuple[tuple[str, int, float], ...] = ()  # (label, count, fraction)
    entropy_bits: float | None = None
    dominant_fraction: float | None = None
    imbalance_ratio: float | None = None


@dataclass(frozen=True)
class DateTimeStats:
    """Summary for DATETIME columns. ``min``/``max`` stored as ISO strings."""

    min: str | None = None
    max: str | None = None
    span_seconds: float | None = None
    median_gap_seconds: float | None = None
    largest_gap_seconds: float | None = None
    timezone_aware: bool = False


@dataclass(frozen=True)
class TextStats:
    """Length summary for TEXT/IDENTIFIER columns."""

    count: int
    mean_length: float | None = None
    max_length: int | None = None


@dataclass(frozen=True)
class ColumnProfile:
    """Everything learned about one column. Original data is never touched."""

    name: str
    stored_dtype: str
    kind: ColumnKind
    n_nonnull: int
    n_missing: int
    n_unique: int
    numeric: NumericStats | None = None
    categorical: CategoricalStats | None = None
    datetime: DateTimeStats | None = None
    text: TextStats | None = None
    findings: tuple[Finding, ...] = ()


@dataclass(frozen=True)
class AssociationRecord:
    """A measured association between two columns or a column and the target.

    ``value`` semantics depend on ``method``: pearson r / point-biserial r /
    Cramer's V. All are descriptive association measures — the presence of an
    AssociationRecord never implies causation.
    """

    left: str
    right: str
    value: float
    method: str
    flagged: bool = False
    note: str = ""


@dataclass(frozen=True)
class Overview:
    """Dataset-level facts. ``source_sha256`` is the sha256 of the raw source
    file when one was provided; it anchors reproducibility."""

    dataset_name: str
    n_rows: int
    n_cols: int
    memory_bytes: int
    target: str | None = None
    source_path: str | None = None
    source_sha256: str | None = None
    created_utc: str = ""
    engine_version: str = ""


@dataclass(frozen=True)
class DataUnderstandingReport:
    """The full Data Understanding Report (spec §3, §16 §4).

    - ``overview``      dataset facts
    - ``columns``       one ColumnProfile per column
    - ``findings``      dataset-level findings (duplicates, missingness rows, …)
    - ``associations``  measured associations (feature–feature and target-aware)
    - ``leak_findings`` heuristic leakage/proxy candidates, target-aware
    - ``notes``         scope and limitations the reader must keep in mind
    """

    overview: Overview
    columns: tuple[ColumnProfile, ...] = ()
    findings: tuple[Finding, ...] = ()
    associations: tuple[AssociationRecord, ...] = ()
    leak_findings: tuple[Finding, ...] = ()
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "overview": asdict(self.overview),
            "columns": [
                {
                    "name": c.name,
                    "stored_dtype": c.stored_dtype,
                    "kind": c.kind.value,
                    "n_nonnull": c.n_nonnull,
                    "n_missing": c.n_missing,
                    "n_unique": c.n_unique,
                    "numeric": asdict(c.numeric) if c.numeric else None,
                    "categorical": asdict(c.categorical) if c.categorical else None,
                    "datetime": asdict(c.datetime) if c.datetime else None,
                    "text": asdict(c.text) if c.text else None,
                    "findings": [f.to_dict() for f in c.findings],
                }
                for c in self.columns
            ],
            "findings": [f.to_dict() for f in self.findings],
            "associations": [asdict(a) for a in self.associations],
            "leak_findings": [f.to_dict() for f in self.leak_findings],
            "notes": list(self.notes),
        }

    def sorted_findings(self) -> list[Finding]:
        """All findings (global + per-column + leak), worst first, stable."""
        all_f: list[Finding] = list(self.findings) + list(self.leak_findings)
        for c in self.columns:
            all_f.extend(c.findings)
        return sorted(
            all_f, key=lambda f: (-int(f.severity), f.code, f.columns, f.message)
        )

    def render_markdown(self, *, max_rows: int = 60) -> str:
        """Human-readable rendering (import deferred to avoid a cycle)."""
        from nullius.data import render

        return render.render_markdown(self, max_rows=max_rows)
