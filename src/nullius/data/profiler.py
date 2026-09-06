"""Data Understanding Engine — orchestrator (Phase 1, spec §3).

``profile_dataframe`` inspects a pandas DataFrame and returns a
``DataUnderstandingReport``. The input frame is treated as read-only: this
module never mutates, casts, or imputes anything. Every transformation the user
makes downstream must be recorded separately (Phase 4+ registry); Phase 1 only
observes and reports.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from pandas.api import types as pdt
from scipy.stats.contingency import chi2_contingency

import nullius
from nullius.data import classify as clf
from nullius.data import detectors as det
from nullius.data.records import (
    AssociationRecord,
    ColumnProfile,
    DataUnderstandingReport,
    DateTimeStats,
    Finding,
    NumericStats,
    Overview,
    Severity,
)
from nullius.data.types import ColumnKind

#: cap on full pairwise correlation matrix size; beyond this we only correlate
#: features against the target and say so (an honest, bounded behavior).
MAX_CORRELATION_COLUMNS = 200
#: |r| thresholds for association reporting (linear correlation screens)
STRONG_R = 0.8
PERFECT_R = 0.99
#: Cramer's V thresholds (categorical association screens)
STRONG_V = 0.6
PERFECT_V = 0.95
#: equality fraction that flags "this column IS the target under another name"
TARGET_DUP_EQUALITY = 0.999
#: cap distinct values before Cramer's V becomes unreliable/costly
CRAMERS_MAX_CATEGORIES = 500
_TOP_ASSOCIATIONS = 25


def hash_file(path: str | Path) -> str:
    """sha256 of the raw file bytes — reproducibility anchor (spec §12)."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_dataset(path: str | Path, **kwargs) -> tuple[pd.DataFrame, str]:
    """Read a CSV/TSV/parquet file and return (frame, sha256 of the file).

    Uses the file extension to pick the reader; ``kwargs`` pass through.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"dataset file not found: {path}")
    digest = hash_file(path)
    suffix = path.suffix.lower()
    if suffix in {".csv", ".tsv", ".txt"}:
        df = pd.read_csv(path, **kwargs)
    elif suffix in {".parquet", ".pq"}:
        try:
            import pyarrow  # noqa: F401
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise RuntimeError(
                "reading parquet requires the optional dependency 'pyarrow' "
                "(install with: pip install nullius[io])"
            ) from exc
        df = pd.read_parquet(path, **kwargs)
    else:
        raise ValueError(
            f"unsupported dataset extension '{suffix}'; use .csv/.tsv/.txt/.parquet"
        )
    return df, digest


# --------------------------------------------------------------------------- #
# Column-level inspection
# --------------------------------------------------------------------------- #
def _constant_finding(name: str, kind: ColumnKind) -> Finding:
    return Finding(
        code="CONSTANT_COL",
        severity=Severity.MEDIUM,
        message=(
            f"'{name}' is constant ({kind.value}); it carries no information and "
            "should be excluded from modeling (or confirmed as a legitimate "
            "all-one/all-zero design column)."
        ),
        columns=(name,),
        method="distinct-value count == 1 over non-missing values",
        detail={"kind": kind.value},
    )


def _profile_column(name: str, series: pd.Series, n_rows: int) -> ColumnProfile:
    """Profile one column. Exceptions are contained so one exotic column cannot
    take down the whole report — the failure is recorded instead."""
    try:
        return _profile_column_impl(name, series, n_rows)
    except Exception as exc:  # pragma: no cover - defensive
        return ColumnProfile(
            name=name,
            stored_dtype=str(series.dtype),
            kind=ColumnKind.OTHER,
            n_nonnull=int(series.notna().sum()),
            n_missing=int(series.isna().sum()),
            n_unique=int(series.nunique(dropna=True)),
            findings=(
                Finding(
                    code="COLUMN_PROFILE_ERROR",
                    severity=Severity.LOW,
                    message=(
                        f"'{name}' could not be profiled ({type(exc).__name__}: {exc}); "
                        "check the column's values and stored type."
                    ),
                    columns=(name,),
                    method="defensive containment of profiling error",
                ),
            ),
        )


def _profile_column_impl(name: str, series: pd.Series, n_rows: int) -> ColumnProfile:
    n_missing = int(series.isna().sum())
    n_nonnull = n_rows - n_missing
    n_unique = int(series.nunique(dropna=True))
    stored_dtype = str(series.dtype)
    if n_rows == 0:
        return ColumnProfile(
            name=name, stored_dtype=stored_dtype, kind=ColumnKind.OTHER,
            n_nonnull=0, n_missing=0, n_unique=0,
        )

    classification = clf.classify_column(name, series, n_nonnull)
    kind = classification.kind
    # A column with no usable values has no observable role regardless of the
    # stored dtype it was read as (e.g. an all-NaN float column).
    if n_nonnull == 0:
        kind = ColumnKind.OTHER
    findings: list[Finding] = []
    numeric: NumericStats | None = None
    categorical: det.CategoricalStats | None = None
    dtime: DateTimeStats | None = None
    text: det.TextStats | None = None

    # --- missingness ------------------------------------------------------ #
    if n_missing == n_rows:
        findings.append(
            Finding(
                code="COL_ALL_MISSING",
                severity=Severity.MEDIUM,
                message=(
                    f"'{name}' is entirely missing. It cannot be used as-is; "
                    "dropping or imputing requires a recorded decision."
                ),
                columns=(name,),
                method="count of missing values == row count",
                detail={"n_missing": n_missing},
            )
        )
    elif n_missing > 0:
        frac = n_missing / n_rows
        sev = Severity.LOW if frac < 0.05 else (Severity.MEDIUM if frac < 0.3 else Severity.HIGH)
        findings.append(
            Finding(
                code="COL_MISSING",
                severity=sev,
                message=(
                    f"'{name}' has {n_missing} missing values ({frac:.1%} of rows). "
                    "Before imputation or exclusion, record the assumed missingness "
                    "mechanism — naive deletion or mean-filling can bias estimates."
                ),
                columns=(name,),
                method="missing-value count; fraction of all rows",
                detail={"n_missing": n_missing, "frac": round(frac, 6)},
            )
        )
    findings.extend(det.missing_encoding_findings(series, name, n_nonnull))

    if n_nonnull == 0:
        return ColumnProfile(
            name, stored_dtype, kind, n_nonnull, n_missing, n_unique,
            findings=tuple(findings),
        )

    # --- content stats per inferred kind ---------------------------------- #
    if kind in (ColumnKind.NUMERIC,):
        numeric = det.numeric_stats(series)
        if n_unique <= 1:
            findings.append(_constant_finding(name, kind))
        else:
            findings.extend(det.skew_findings(numeric, name))
            findings.extend(det.outlier_findings(numeric, name, n_rows))
            if classification.numeric_ratio is not None:
                findings.append(
                    Finding(
                        code="NUM_AS_STRING",
                        severity=Severity.MEDIUM,
                        message=(
                            f"'{name}' is stored as text but {classification.numeric_ratio:.0%} "
                            "of sampled values parse as numbers. String dtype breaks "
                            "vectorized statistics and sorts lexicographically — parse it "
                            "to a numeric dtype in a recorded preprocessing step before "
                            "modeling."
                        ),
                        columns=(name,),
                        method=(
                            "parse-ratio probe over a deterministic sample of unique values"
                        ),
                        detail={"numeric_ratio": round(classification.numeric_ratio, 6)},
                    )
                )
            if pdt.is_integer_dtype(series) and 0 < n_unique <= 10:
                findings.append(
                    Finding(
                        code="LOW_CARD_INT",
                        severity=Severity.INFO,
                        message=(
                            f"'{name}' is integer-typed with only {n_unique} distinct "
                            "values — it may encode categories/binary states rather than "
                            "a measured quantity; consider its measurement level before "
                            "treating it as continuous."
                        ),
                        columns=(name,),
                        method="distinct-value count on integer dtype",
                        detail={"n_unique": n_unique},
                    )
                )
    elif kind == ColumnKind.CATEGORICAL:
        categorical = det.categorical_stats(series)
        if n_unique <= 1:
            findings.append(_constant_finding(name, kind))
        else:
            findings.extend(det.categorical_findings(categorical, name))
            if classification.numeric_ratio is not None:
                findings.append(
                    Finding(
                        code="NUM_STRING_MIXED",
                        severity=Severity.MEDIUM,
                        message=(
                            f"'{name}' mixes text and numeric content "
                            f"({classification.numeric_ratio:.0%} of sampled values parse "
                            "as numbers). Decide on a single representation and record it; "
                            "mixed types silently produce unusable feature encodings."
                        ),
                        columns=(name,),
                        method="parse-ratio probe over sampled unique values",
                        detail={"numeric_ratio": round(classification.numeric_ratio, 6)},
                    )
                )
    elif kind == ColumnKind.BOOLEAN:
        categorical = det.categorical_stats(series)
        if n_unique <= 1:
            findings.append(_constant_finding(name, kind))
    elif kind == ColumnKind.DATETIME:
        dtime = det.datetime_stats(series)
        if classification.datetime_ratio is not None:
            findings.append(
                Finding(
                    code="DATETIME_AS_STRING",
                    severity=Severity.MEDIUM,
                    message=(
                        f"'{name}' stores ISO-like datetimes as text "
                        f"({classification.datetime_ratio:.0%} of sampled values parse). "
                        "Convert to a datetime dtype (recorded step) before time-aware "
                        "analysis — string comparison mis-orders dates and hides gaps."
                    ),
                    columns=(name,),
                    method="datetime parse-ratio probe over sampled unique values",
                    detail={"datetime_ratio": round(classification.datetime_ratio, 6)},
                )
            )
        if dtime.median_gap_seconds is not None and dtime.largest_gap_seconds is not None:
            if (
                dtime.median_gap_seconds > 0
                and dtime.largest_gap_seconds > 10 * dtime.median_gap_seconds
            ):
                findings.append(
                    Finding(
                        code="DATETIME_IRREGULAR",
                        severity=Severity.INFO,
                        message=(
                            f"'{name}' has irregular sampling: the largest gap "
                            f"({_fmt_duration(dtime.largest_gap_seconds)}) is >10× the "
                            "median gap — check for missing observation windows."
                        ),
                        columns=(name,),
                        method="gap analysis on sorted unique timestamps",
                        detail={
                            "median_gap_seconds": dtime.median_gap_seconds,
                            "largest_gap_seconds": dtime.largest_gap_seconds,
                        },
                    )
                )
    elif kind in (ColumnKind.TEXT, ColumnKind.IDENTIFIER):
        text = det.text_stats(series)
        if kind == ColumnKind.IDENTIFIER:
            findings.append(
                Finding(
                    code="ID_COLUMN",
                    severity=Severity.LOW,
                    message=(
                        f"'{name}' looks like an identifier ({classification.reason}). "
                        "Identifiers must not be used as features (they memorize rows); "
                        "keep them for joins and traceability."
                    ),
                    columns=(name,),
                    method="name and content heuristics (uniqueness ratio, shape)",
                    detail={"reason": classification.reason},
                )
            )
        elif text.mean_length is not None and text.mean_length >= 60:
            findings.append(
                Finding(
                    code="TEXT_COL",
                    severity=Severity.INFO,
                    message=(
                        f"'{name}' is free text (mean length {text.mean_length:.0f} "
                        "chars). Text modeling is out of scope for this profiler; the "
                        "column is excluded from numeric association scans."
                    ),
                    columns=(name,),
                    method="mean length and cardinality heuristics",
                    detail={"mean_length": round(text.mean_length, 2)},
                )
            )
    else:
        findings.append(
            Finding(
                code="UNSUPPORTED_COL",
                severity=Severity.LOW,
                message=f"'{name}' has stored type '{stored_dtype}', which this profiler does not model.",
                columns=(name,),
                method="stored dtype inspection",
                detail={"stored_dtype": stored_dtype},
            )
        )

    return ColumnProfile(
        name=name,
        stored_dtype=stored_dtype,
        kind=kind,
        n_nonnull=n_nonnull,
        n_missing=n_missing,
        n_unique=n_unique,
        numeric=numeric,
        categorical=categorical,
        datetime=dtime,
        text=text,
        findings=tuple(findings),
    )


def _fmt_duration(seconds: float) -> str:
    if seconds >= 86400:
        return f"{seconds / 86400:.1f} days"
    if seconds >= 3600:
        return f"{seconds / 3600:.1f} hours"
    return f"{seconds:.0f} s"


# --------------------------------------------------------------------------- #
# Dataset-level inspection
# --------------------------------------------------------------------------- #
def _row_missingness_findings(df: pd.DataFrame) -> list[Finding]:
    out: list[Finding] = []
    n = len(df)
    if n == 0:
        return out
    any_missing = int(df.isna().any(axis=1).sum())
    if any_missing:
        frac = any_missing / n
        out.append(
            Finding(
                code="ROWS_WITH_MISSING",
                severity=Severity.LOW if frac < 0.1 else Severity.MEDIUM,
                message=(
                    f"{any_missing} rows ({frac:.1%}) contain at least one missing value; "
                    "listwise deletion would silently discard them. Record the missingness "
                    "mechanism assumption (MCAR/MAR/MNAR) before choosing how to handle "
                    "missingness in modeling."
                ),
                method="row-wise any-missing count",
                detail={"n_rows_affected": any_missing, "frac": round(frac, 6)},
            )
        )
    all_missing = int(df.isna().all(axis=1).sum())
    if all_missing:
        frac = all_missing / n
        out.append(
            Finding(
                code="ROWS_ALL_MISSING",
                severity=Severity.HIGH if frac >= 0.01 else Severity.MEDIUM,
                message=(
                    f"{all_missing} rows ({frac:.1%}) are entirely empty — a data-import "
                    "artifact in most files. Dropping them is a recorded decision, not "
                    "something the profiler does silently."
                ),
                method="row-wise all-missing count",
                detail={"n_rows": all_missing, "frac": round(frac, 6)},
            )
        )
    return out


def _duplicate_findings(df: pd.DataFrame) -> list[Finding]:
    n = len(df)
    if n < 2:
        return []
    redundant = int(df.duplicated().sum())
    if redundant == 0:
        return []
    involved = int(df.duplicated(keep=False).sum())
    frac = redundant / n
    sev = (
        Severity.LOW
        if frac < 0.001
        else (Severity.MEDIUM if frac < 0.1 else Severity.HIGH)
    )
    return [
        Finding(
            code="DUP_ROWS",
            severity=sev,
            message=(
                f"{redundant} rows ({frac:.1%}) are exact duplicates of an earlier row "
                f"({involved} rows participate in duplicate groups). Confirm duplicates "
                "are errors and not legitimate repeated observations (e.g., the same "
                "customer measured again) before removing them — the distinction "
                "matters for inference."
            ),
            method="exact duplicate detection over all columns (pandas duplicated)",
            detail={"n_redundant_rows": redundant, "n_rows_in_groups": involved,
                    "frac": round(frac, 6)},
        )
    ]


def _pairwise_pearson(df: pd.DataFrame, a: str, b: str) -> float | None:
    x = pd.to_numeric(df[a], errors="coerce")
    y = pd.to_numeric(df[b], errors="coerce")
    mask = x.notna() & y.notna()
    if int(mask.sum()) < 5:
        return None
    sx, sy = x[mask], y[mask]
    if sx.nunique(dropna=True) < 2 or sy.nunique(dropna=True) < 2:
        return None
    return float(np.corrcoef(sx.to_numpy(), sy.to_numpy())[0, 1])


def _feature_correlation_findings(
    df: pd.DataFrame,
    profiles: list[ColumnProfile],
    target: str | None = None,
) -> list[Finding]:
    # Only genuinely numeric *stored dtypes* enter the raw Pearson matrix;
    # columns whose text parses as numbers need an explicit parse first and
    # would otherwise break the pairwise computation.
    numeric_cols = [
        p.name
        for p in profiles
        if p.kind == ColumnKind.NUMERIC
        and p.n_unique > 1
        and p.name != target
        and pdt.is_numeric_dtype(df[p.name])
    ]
    if len(numeric_cols) < 2:
        return []
    out: list[Finding] = []
    if len(numeric_cols) > MAX_CORRELATION_COLUMNS:
        out.append(
            Finding(
                code="CORR_SKIPPED",
                severity=Severity.INFO,
                message=(
                    f"{len(numeric_cols)} numeric columns exceed the pairwise-correlation "
                    f"cap ({MAX_CORRELATION_COLUMNS}); full feature–feature correlation "
                    "was skipped. Correlations against a named target are still computed."
                ),
                method="documented cap on O(k²) pairwise computation",
                detail={"n_numeric": len(numeric_cols), "cap": MAX_CORRELATION_COLUMNS},
            )
        )
        return out

    corr = df[numeric_cols].corr(numeric_only=True)
    pairs: list[tuple[float, str, str]] = []
    for i, a in enumerate(numeric_cols):
        for b in numeric_cols[i + 1 :]:
            v = corr.loc[a, b]
            if v == v and abs(v) >= STRONG_R:  # not NaN
                pairs.append((abs(float(v)), a, b))
    pairs.sort(reverse=True)
    n_strong = len(pairs)
    n_perfect = sum(1 for r, _, _ in pairs if r >= PERFECT_R)
    if n_strong:
        top = pairs[:_TOP_ASSOCIATIONS]
        listed = ", ".join(f"{a}–{b} (r={r:.3f})" for r, a, b in top)
        noun = "pair" if n_strong == 1 else "pairs"
        out.append(
            Finding(
                code="CORR_HIGH",
                severity=Severity.LOW,
                message=(
                    f"{n_strong} feature {noun} have |Pearson r| ≥ {STRONG_R} "
                    f"(near-perfect ≥ {PERFECT_R}: {n_perfect}). "
                    "Such features are largely redundant for linear models. "
                    f"Strongest: {listed}. Association is descriptive — no causal "
                    "reading is implied."
                ),
                method="pairwise Pearson correlation; |r| threshold screening",
                detail={
                    "n_strong_pairs": n_strong,
                    "n_near_perfect_pairs": n_perfect,
                    "top_pairs": [
                        {"a": a, "b": b, "r": round(r, 6)} for r, a, b in top
                    ],
                },
            )
        )
    return out


def _cramers_v(feat: pd.Series, target: pd.Series) -> tuple[float, int] | None:
    """Cramer's V between two categorical series, or None if not computable.

    V = sqrt(chi2 / (n * min(r-1, c-1))) on the contingency table; chi2 is the
    Pearson chi-square statistic. Association measure only.
    """
    table = pd.crosstab(feat, target, dropna=True)
    if table.shape[0] < 2 or table.shape[1] < 2:
        return None
    chi2, _, _, _ = chi2_contingency(table.to_numpy(), correction=False)
    n = int(table.to_numpy().sum())
    denom = n * min(table.shape[0] - 1, table.shape[1] - 1)
    if denom <= 0 or chi2 < 0:
        return None
    return float(np.sqrt(chi2 / denom)), n


def _equality_fraction(a: pd.Series, b: pd.Series) -> float | None:
    """Fraction of aligned non-missing rows where a == b (None if not comparable).

    Values are cast only after the missing mask is applied, so missingness is
    never stringified (``astype(str)`` would turn NaN into "nan").
    """
    a_num, b_num = pdt.is_numeric_dtype(a), pdt.is_numeric_dtype(b)
    a_txt = pdt.is_string_dtype(a) or pdt.is_bool_dtype(a)
    b_txt = pdt.is_string_dtype(b) or pdt.is_bool_dtype(b)
    if not ((a_num and b_num) or (a_txt and b_txt)):
        return None
    mask = a.notna() & b.notna()
    if int(mask.sum()) < 5:
        return None
    xa, xb = a[mask], b[mask]
    if not (a_num and b_num):
        xa = xa.astype(str)
        xb = xb.astype(str)
    return float((xa == xb).mean())


def _target_analysis(
    df: pd.DataFrame,
    profiles: list[ColumnProfile],
    target: str,
    n_rows: int,
) -> tuple[list[AssociationRecord], list[Finding]]:
    """Target-aware association records and leakage heuristics (spec §3, §8)."""
    target_profile = next((p for p in profiles if p.name == target), None)
    if target_profile is None:  # validated by profile_dataframe; defensive
        return [], []
    tcol = df[target]
    tkind = target_profile.kind
    records: list[AssociationRecord] = []
    leaks: list[Finding] = []

    # -- duplicate-of-target: the same values under another column name ------ #
    if tkind in (ColumnKind.CATEGORICAL, ColumnKind.BOOLEAN, ColumnKind.NUMERIC):
        for p in profiles:
            if p.name == target or p.kind not in (ColumnKind.NUMERIC, ColumnKind.CATEGORICAL, ColumnKind.BOOLEAN):
                continue
            eq = _equality_fraction(df[p.name], tcol)
            if eq is not None and eq >= TARGET_DUP_EQUALITY:
                leaks.append(
                    Finding(
                        code="LEAK_TARGET_DUPLICATE",
                        severity=Severity.HIGH,
                        message=(
                            f"'{p.name}' matches the target '{target}' on "
                            f"{eq:.1%} of aligned rows. This is the classic leakage "
                            "pattern (the target shipped under another name). If "
                            "confirmed, exclude it from features."
                        ),
                        columns=(p.name, target),
                        method="aligned elementwise equality fraction ≥ 99.9%",
                        detail={"equality_fraction": round(eq, 6)},
                    )
                )

    def note_perfect_assoc(feat: str, value: float, method: str) -> None:
        if abs(value) >= PERFECT_V or (
            method != "cramers-v" and abs(value) >= PERFECT_R
        ):
            leaks.append(
                Finding(
                    code="LEAK_NEAR_PERFECT_ASSOC",
                    severity=Severity.HIGH,
                    message=(
                        f"'{feat}' has near-perfect association with the target "
                        f"({method} = {value:.3f}). Either it *is* the target under a "
                        "transformation, or it is a near-deterministic proxy — in both "
                        "cases the feature leaks target information. Verify before "
                        "modeling; near-perfect association with the target is the most "
                        "common leakage vector."
                    ),
                    columns=(feat, target),
                    method=f"{method} threshold screening",
                    detail={"value": round(value, 6), "method": method},
                )
            )

    # -- target is numeric: linear correlation with numeric features ---------- #
    if tkind == ColumnKind.NUMERIC:
        for p in profiles:
            if p.name == target or p.kind != ColumnKind.NUMERIC or p.n_unique <= 1:
                continue
            r = _pairwise_pearson(df, p.name, target)
            if r is None:
                continue
            strong = abs(r) >= STRONG_R
            records.append(
                AssociationRecord(
                    left=p.name,
                    right=target,
                    value=round(float(r), 6),
                    method="pearson",
                    flagged=strong,
                    note=(
                        "linear association with the numeric target"
                        if not strong
                        else "strong linear association with the target — candidate "
                        "predictor, but verify it is not target-derived"
                    ),
                )
            )
            note_perfect_assoc(p.name, r, "pearson")

    # -- target is categorical ------------------------------------------------ #
    elif tkind in (ColumnKind.CATEGORICAL, ColumnKind.BOOLEAN):
        if tkind == ColumnKind.BOOLEAN or tcol.nunique(dropna=True) == 2:
            # point-biserial correlation for numeric features vs a binary target
            code_map = {v: i for i, v in enumerate(tcol.dropna().unique())}
            y = tcol.map(code_map)
            for p in profiles:
                if p.name == target or p.kind != ColumnKind.NUMERIC or p.n_unique <= 1:
                    continue
                x = pd.to_numeric(df[p.name], errors="coerce")
                mask = x.notna() & y.notna()
                if int(mask.sum()) < 10:
                    continue
                sx = x[mask]
                if sx.nunique(dropna=True) < 2 or int(y[mask].nunique()) < 2:
                    continue
                r = float(np.corrcoef(sx.to_numpy(), y[mask].to_numpy())[0, 1])
                strong = abs(r) >= STRONG_R
                records.append(
                    AssociationRecord(
                        left=p.name,
                        right=target,
                        value=round(r, 6),
                        method="point-biserial",
                        flagged=strong,
                        note=(
                            "correlation with binary target"
                            if not strong
                            else "strong association with the binary target — verify it "
                            "is not target-derived"
                        ),
                    )
                )
                note_perfect_assoc(p.name, r, "point-biserial")
            # Cramer's V vs categorical/boolean features
            for p in profiles:
                if (
                    p.name == target
                    or p.kind not in (ColumnKind.CATEGORICAL, ColumnKind.BOOLEAN)
                    or p.n_unique <= 1
                ):
                    continue
                if p.n_unique > CRAMERS_MAX_CATEGORIES:
                    continue
                res = _cramers_v(df[p.name], tcol)
                if res is None:
                    continue
                v, _n = res
                strong = v >= STRONG_V
                records.append(
                    AssociationRecord(
                        left=p.name,
                        right=target,
                        value=round(v, 6),
                        method="cramers-v",
                        flagged=strong,
                        note=(
                            "categorical association with target"
                            if not strong
                            else "strong association with the target — candidate "
                            "predictor; verify it is not target-derived"
                        ),
                    )
                )
                if v >= PERFECT_V:
                    note_perfect_assoc(p.name, v, "cramers-v")

        # class-imbalance note for the target itself
        counts = tcol.value_counts(dropna=True)
        if len(counts) >= 2:
            minority_frac = float(counts.iloc[-1] / counts.sum())
            if minority_frac < 0.2:
                sev = (
                    Severity.LOW
                    if minority_frac >= 0.1
                    else (Severity.MEDIUM if minority_frac >= 0.01 else Severity.HIGH)
                )
                leaks.append(
                    Finding(
                        code="TARGET_IMBALANCE",
                        severity=sev,
                        message=(
                            f"The target '{target}' is imbalanced: minority class is "
                            f"{minority_frac:.1%} of rows. Accuracy will be misleading; "
                            "plan for balanced evaluation metrics (PR-AUC, balanced "
                            "accuracy), calibration, and stratified splits."
                        ),
                        columns=(target,),
                        method="empirical class fractions of the target",
                        detail={"minority_frac": round(minority_frac, 6)},
                    )
                )
    return records, leaks


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def profile_dataframe(
    df: pd.DataFrame,
    *,
    name: str | None = None,
    source_path: str | None = None,
    source_sha256: str | None = None,
    target: str | None = None,
    max_correlation_columns: int = MAX_CORRELATION_COLUMNS,
) -> DataUnderstandingReport:
    """Profile ``df`` and return a Data Understanding Report.

    ``df`` is never modified. ``target`` (optional) enables class-imbalance and
    leakage heuristics; the column must exist in ``df``.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"expected pandas DataFrame, got {type(df).__name__}")
    if target is not None and target not in df.columns:
        raise ValueError(
            f"target column '{target}' not found in the data "
            f"(available columns: {list(df.columns)[:20]}{'…' if len(df.columns) > 20 else ''})"
        )

    n_rows, n_cols = df.shape
    dataset_name = name or (
        Path(source_path).stem if source_path else "dataset"
    )
    if max_correlation_columns < 2:
        max_correlation_columns = 2

    profiles: list[ColumnProfile] = [
        _profile_column(col, df[col], n_rows) for col in df.columns
    ]

    findings: list[Finding] = []
    findings.extend(_row_missingness_findings(df))
    findings.extend(_duplicate_findings(df))
    findings.extend(_feature_correlation_findings(df, profiles, target=target))

    associations: list[AssociationRecord] = []
    leak_findings: list[Finding] = []
    if target is not None:
        assoc, leaks = _target_analysis(df, profiles, target, n_rows)
        associations = assoc
        leak_findings = leaks
        tp = next((p for p in profiles if p.name == target), None)
        if tp is not None and tp.n_missing > 0:
            frac = tp.n_missing / n_rows
            leak_findings.append(
                Finding(
                    code="TARGET_MISSING",
                    severity=Severity.HIGH if frac > 0.3 else Severity.MEDIUM,
                    message=(
                        f"The target '{target}' itself is missing in {tp.n_missing} rows "
                        f"({frac:.1%}). Rows without a target cannot be used for "
                        "supervised fitting; record how they are handled."
                    ),
                    columns=(target,),
                    method="missing-value count on the target column",
                    detail={"n_missing": tp.n_missing, "frac": round(frac, 6)},
                )
            )

    associations.sort(key=lambda a: -abs(a.value))
    notes = _report_notes(df, profiles, n_cols, source_path)

    overview = Overview(
        dataset_name=dataset_name,
        n_rows=n_rows,
        n_cols=n_cols,
        memory_bytes=int(df.memory_usage(deep=True).sum()),
        target=target,
        source_path=str(source_path) if source_path else None,
        source_sha256=source_sha256,
        created_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        engine_version=nullius.__version__,
    )
    return DataUnderstandingReport(
        overview=overview,
        columns=tuple(profiles),
        findings=tuple(findings),
        associations=tuple(associations),
        leak_findings=tuple(leak_findings),
        notes=tuple(notes),
    )


def _report_notes(
    df: pd.DataFrame, profiles: list[ColumnProfile], n_cols: int, source_path: str | None
) -> list[str]:
    notes = [
        "Associations in this report are descriptive measures (Pearson r, "
        "point-biserial r, Cramer's V) under the profiler's screening rules. "
        "No causal interpretation is implied; 'associated with' is the strongest "
        "wording this report supports.",
        "Leakage flags are heuristic candidates, not verdicts: near-perfect "
        "association can also arise from legitimate duplicate or derived features "
        "(e.g. total = sum of parts). A human or the Phase-8 critic must confirm "
        "before excluding features.",
        "Outlier and skew flags come from generic screening rules (Tukey fences, "
        "MAD-based robust z-scores, skewness magnitude). They identify candidates "
        "for review — they never justify silently dropping values.",
        "Train/test contamination cannot be assessed from a single file: it "
        "requires provenance of how the file was split. If rows are ordered in "
        "time (even without a datetime column), say so — random splits would leak.",
    ]
    if not any(p.kind == ColumnKind.DATETIME for p in profiles):
        notes.append(
            "No datetime column was detected. Temporal structure and temporal "
            "leakage therefore cannot be assessed; if the data has an implicit "
            "time ordering, provide the ordering column explicitly."
        )
    if any(p.kind in (ColumnKind.TEXT, ColumnKind.IDENTIFIER) for p in profiles):
        notes.append(
            "Identifier and free-text columns are excluded from association and "
            "leakage scans by design: identifiers memorize rows and text modeling "
            "is out of scope for Phase 1."
        )
    if n_cols == 0:
        notes.append("The dataset has no columns; only row-level counts are meaningful.")
    if source_path:
        notes.append(
            f"Source file: {source_path}. The profiler analyzed an in-memory "
            "snapshot; the original file was not modified."
        )
    return notes
