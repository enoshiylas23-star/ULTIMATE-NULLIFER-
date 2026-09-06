"""Per-column summary statistics and specialized detectors.

Every helper documents the method and its assumptions (spec §1). Generic rules
(quantiles, fences, MAD, entropy, Cramer's V) are used as *screening* tools;
their output is reported with severity, never silently applied.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.api import types as pdt
from scipy import stats as sstats

from nullius.data.records import (
    CategoricalStats,
    DateTimeStats,
    Finding,
    NumericStats,
    Severity,
    TextStats,
)

#: label strings that commonly encode missingness in text columns
_MISSING_LITERALS = {
    "na",
    "n/a",
    "n.a.",
    "nan",
    "null",
    "none",
    "nil",
    "-",
    "unknown",
    "missing",
    "#n/a",
    "nul",
    "nat",
}
_MAD_Z_THRESHOLD = 3.0
_TUKEY_K = 1.5
#: categorical entropy is reported in bits
_ENTROPY_LOG_BASE = 2.0


# --------------------------------------------------------------------------- #
# Summary statistics
# --------------------------------------------------------------------------- #
def numeric_stats(series: pd.Series) -> NumericStats:
    """Summary + screening flags for a numeric column (method notes in docstring
    of ``NumericStats``)."""
    x = pd.to_numeric(series, errors="coerce")
    count = int(x.notna().sum())
    if count == 0:
        return NumericStats(count=0)

    mean = float(x.mean())
    std = float(x.std(ddof=1)) if count > 1 else 0.0
    q1, med, q3 = (float(v) for v in x.quantile([0.25, 0.5, 0.75]))
    skew = float(x.skew()) if count >= 8 else None
    kurt = float(x.kurt()) if count >= 4 else None

    # Tukey fences: points beyond [Q1 - k*IQR, Q3 + k*IQR]
    iqr = q3 - q1
    lo_f, hi_f = q1 - _TUKEY_K * iqr, q3 + _TUKEY_K * iqr
    n_out_tukey = int(((x < lo_f) | (x > hi_f)).sum())

    # Robust z on MAD (normal-consistent scale 1.4826). If MAD == 0 (>=50% of
    # values equal the median) the robust screen is degenerate and we skip it.
    n_out_mad = 0
    med_x = float(x.median())
    mad = float(sstats.median_abs_deviation(x.dropna(), scale="normal"))
    if mad > 0:
        dev = (x - med_x).abs()
        n_out_mad = int((dev / mad > _MAD_Z_THRESHOLD).sum())

    return NumericStats(
        count=count,
        mean=mean,
        std=std,
        min=float(x.min()),
        q1=q1,
        median=med,
        q3=q3,
        max=float(x.max()),
        skew=skew,
        kurtosis=kurt,
        n_outliers_tukey=n_out_tukey,
        n_outliers_mad=n_out_mad,
    )


def categorical_stats(series: pd.Series, *, top_k: int = 5) -> CategoricalStats:
    """Counts, Shannon entropy (bits), dominance and imbalance ratio.

    Entropy H = -sum_i p_i log2(p_i) over the empirical distribution of
    non-missing values. imbalance_ratio = max count / min nonzero count.
    """
    counts = series.value_counts(dropna=True)
    total = int(counts.sum())
    cardinality = int(len(counts))
    if total == 0:
        return CategoricalStats(cardinality=0)

    probs = counts.to_numpy(dtype=float) / total
    entropy = float(-(probs * np.log2(probs)).sum()) if cardinality > 1 else 0.0
    dom_frac = float(counts.iloc[0] / total) if cardinality else None
    imbalance = float(counts.iloc[0] / counts.iloc[-1]) if cardinality > 1 else None
    top = tuple(
        (str(label), int(cnt), float(cnt / total)) for label, cnt in counts.head(top_k).items()
    )
    return CategoricalStats(
        cardinality=cardinality,
        top=top,
        entropy_bits=entropy,
        dominant_fraction=dom_frac,
        imbalance_ratio=imbalance,
    )


def datetime_stats(series: pd.Series) -> DateTimeStats:
    """Range, span, and sampling-gap summary for datetime columns."""
    x = pd.to_datetime(series, errors="coerce")
    x = x.dropna().sort_values()
    if len(x) == 0:
        return DateTimeStats()
    tz_aware = bool(x.dt.tz is not None)
    diffs = x.diff().dropna().dt.total_seconds().to_numpy()
    span = float((x.iloc[-1] - x.iloc[0]).total_seconds())
    if len(diffs) == 0:
        return DateTimeStats(
            min=x.iloc[0].isoformat(),
            max=x.iloc[-1].isoformat(),
            span_seconds=span,
            timezone_aware=tz_aware,
        )
    return DateTimeStats(
        min=x.iloc[0].isoformat(),
        max=x.iloc[-1].isoformat(),
        span_seconds=span,
        median_gap_seconds=float(np.median(diffs)),
        largest_gap_seconds=float(np.max(diffs)),
        timezone_aware=tz_aware,
    )


def text_stats(series: pd.Series) -> TextStats:
    vals = series.dropna().astype(str)
    if len(vals) == 0:
        return TextStats(count=0)
    lengths = vals.str.len()
    return TextStats(
        count=int(len(vals)),
        mean_length=float(lengths.mean()),
        max_length=int(lengths.max()),
    )


# --------------------------------------------------------------------------- #
# Specialized detectors (return Findings)
# --------------------------------------------------------------------------- #
def missing_encoding_findings(series: pd.Series, name: str, n_nonnull: int) -> list[Finding]:
    """Find text values that are probably *encoded missingness* ("", "NA", …)."""
    if n_nonnull == 0:
        return []
    vals = series.dropna()
    # Only inspect columns whose non-missing content is genuinely textual;
    # mixed-type object columns (numbers + strings) are not candidates.
    if pdt.infer_dtype(vals, skipna=True) != "string":
        return []
    s = vals.astype("string")

    empty = int((s == "").sum())
    whitespace = int(s.str.fullmatch(r"\s+").fillna(False).sum())
    lowered = s.str.strip().str.lower()
    literal = int(lowered.isin(_MISSING_LITERALS).sum())
    total = empty + whitespace + literal
    if total == 0:
        return []
    frac = total / len(series)
    severity = Severity.MEDIUM if frac > 0.01 else (Severity.LOW if frac > 0 else Severity.INFO)
    if frac > 0.2:
        severity = Severity.HIGH
    return [
        Finding(
            code="MISSING_ENCODED",
            severity=severity,
            message=(
                f"'{name}' contains {total} values ({frac:.1%} of rows) that encode "
                "missingness as text (empty string, whitespace, or 'NA'/'null'/'-' …). "
                "They are not counted as missing by pandas and would silently distort "
                "analysis if parsed as categories or numbers."
            ),
            columns=(name,),
            method="string comparison against a curated missingness-encoding dictionary",
            detail={"n_encoded": total, "frac": round(frac, 6), "n_empty": empty,
                    "n_literal": literal},
        )
    ]


def skew_findings(stats: NumericStats, name: str) -> list[Finding]:
    if stats.skew is None:
        return []
    out: list[Finding] = []
    if abs(stats.skew) >= 2.0:
        sev, code = Severity.MEDIUM, "HIGH_SKEW"
        msg = "highly skewed"
    elif abs(stats.skew) >= 1.0:
        sev, code = Severity.LOW, "MODERATE_SKEW"
        msg = "moderately skewed"
    else:
        return out
    out.append(
        Finding(
            code=code,
            severity=sev,
            message=(
                f"'{name}' is {msg} (adj. Fisher–Pearson skew = {stats.skew:.2f}). "
                "Mean-based summaries and methods assuming approximate symmetry or "
                "normality are unreliable; consider log/power transforms for such "
                "methods — but verify the choice on held-out data."
            ),
            columns=(name,),
            method="adjusted Fisher–Pearson sample skewness (flag only; not a test)",
            detail={"skew": round(stats.skew, 6)},
        )
    )
    if stats.kurtosis is not None and stats.kurtosis > 10:
        out.append(
            Finding(
                code="HEAVY_TAIL",
                severity=Severity.LOW,
                message=(
                    f"'{name}' has very heavy tails (excess kurtosis "
                    f"{stats.kurtosis:.1f}); extreme values are more common than "
                    "under a normal model — use robust summaries and check model "
                    "error distributions."
                ),
                columns=(name,),
                method="Fisher excess kurtosis (flag only)",
                detail={"kurtosis": round(stats.kurtosis, 4)},
            )
        )
    return out


def outlier_findings(stats: NumericStats, name: str, n_rows: int) -> list[Finding]:
    if stats.count == 0:
        return []
    out: list[Finding] = []
    if stats.has_outliers:
        frac = max(stats.n_outliers_tukey, stats.n_outliers_mad) / n_rows
        sev = Severity.LOW if frac < 0.01 else Severity.MEDIUM
        msg = (
            f"'{name}': {stats.n_outliers_tukey} values beyond Tukey fences "
            f"(1.5×IQR) and {stats.n_outliers_mad} beyond a MAD-based robust "
            f"z-score (±{_MAD_Z_THRESHOLD:.0f}σ). Screening rules only — "
            "domain knowledge decides whether any are errors worth handling; "
            "never drop outliers without recording the decision."
        )
        if frac >= 0.2:
            msg += (
                " A flagged fraction this large usually means the column is "
                "multi-modal or clustered rather than error-ridden — inspect its "
                "distribution before treating any values as outliers."
            )
        out.append(
            Finding(
                code="OUTLIERS",
                severity=sev,
                message=msg,
                columns=(name,),
                method=(
                    "Tukey fences (1.5×IQR) and robust z on MAD "
                    "(3 × 1.4826 × MAD)"
                ),
                detail={
                    "n_tukey": stats.n_outliers_tukey,
                    "n_mad": stats.n_outliers_mad,
                    "frac": round(frac, 6),
                },
            )
        )
    return out


def categorical_findings(
    stats: CategoricalStats, name: str
) -> list[Finding]:
    if stats.cardinality <= 1:
        return []
    out: list[Finding] = []
    # very one-sided distributions
    if stats.dominant_fraction is not None and stats.dominant_fraction >= 0.98:
        out.append(
            Finding(
                code="NEAR_CONSTANT_COL",
                severity=Severity.LOW,
                message=(
                    f"'{name}' is near-constant: its most common category covers "
                    f"{stats.dominant_fraction:.1%} of values. Such columns carry "
                    "almost no information and can destabilize some models."
                ),
                columns=(name,),
                method="dominant-category fraction over non-missing values",
                detail={"dominant_fraction": round(stats.dominant_fraction, 6)},
            )
        )
    if stats.imbalance_ratio is not None and stats.imbalance_ratio >= 10:
        sev = Severity.MEDIUM if stats.imbalance_ratio >= 100 else Severity.LOW
        out.append(
            Finding(
                code="CAT_IMBALANCE",
                severity=sev,
                message=(
                    f"'{name}' is imbalanced: largest/smallest category count ratio "
                    f"= {stats.imbalance_ratio:.1f}. Accuracy-style summaries on "
                    "this column alone would be dominated by the majority category."
                ),
                columns=(name,),
                method="max/min nonzero category-count ratio",
                detail={"imbalance_ratio": round(stats.imbalance_ratio, 4)},
            )
        )
    if stats.cardinality >= 5000:
        out.append(
            Finding(
                code="HIGH_CARDINALITY_CAT",
                severity=Severity.LOW,
                message=(
                    f"'{name}' has {stats.cardinality} categories — high cardinality "
                    "for a categorical feature; one-hot encoding is wasteful and "
                    "target encoding risks leakage. Plan encodings with care."
                ),
                columns=(name,),
                method="distinct-value count",
                detail={"cardinality": stats.cardinality},
            )
        )
    return out
