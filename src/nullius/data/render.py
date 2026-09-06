"""Markdown rendering for the Data Understanding Report.

The JSON structure (``DataUnderstandingReport.to_dict``) is the machine-readable
source of truth and is always complete; markdown tables are capped at
``max_rows`` rows with an explicit pointer to the JSON for the rest.
"""

from __future__ import annotations

from nullius.data.records import DataUnderstandingReport, Finding
from nullius.data.types import ColumnKind, Severity

_METHOD_LABELS = {
    "pearson": "Pearson r",
    "point-biserial": "point-biserial r",
    "cramers-v": "Cramer's V",
}
_SEVERITY_ORDER = (
    Severity.CRITICAL,
    Severity.HIGH,
    Severity.MEDIUM,
    Severity.LOW,
    Severity.INFO,
)
_MAX_FINDING_CHARS = 260


def _esc(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _f(value: float | int | None) -> str:
    if value is None:
        return "–"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _pct(frac: float) -> str:
    return f"{100 * frac:.1f}%"


def _mb(n_bytes: int) -> str:
    return f"{n_bytes / (1024 * 1024):.1f} MiB"


def _table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def _finding_line(f: Finding) -> str:
    cols = ", ".join(f.columns) if f.columns else "—"
    if len(f.message) <= _MAX_FINDING_CHARS:
        msg = f.message
    else:
        msg = f.message[:_MAX_FINDING_CHARS] + "…"
    return f"- **[{f.severity.name}]** `{f.code}` ({cols}) {msg}"


def _truncated_note(kind: str, total: int, shown: int) -> str:
    if total > shown:
        return f"\n\n*{total - shown} more {kind} omitted — the JSON report (`--out`) is complete.*"
    return ""


def render_markdown(report: DataUnderstandingReport, *, max_rows: int = 60) -> str:
    ov = report.overview
    lines: list[str] = []
    lines.append(f"# Data Understanding Report — {ov.dataset_name}")
    lines.append("")

    lines.append(f"_Generated {ov.created_utc} UTC · Nullius v{ov.engine_version}_")
    lines.append("")
    rows = [
        ["Rows", _f(ov.n_rows)],
        ["Columns", _f(ov.n_cols)],
        ["Memory (deep)", _mb(ov.memory_bytes)],
        ["Target", ov.target or "— (no target; leakage heuristics disabled)"],
        ["Source file", _esc(ov.source_path or "—")],
        ["Source sha256", ov.source_sha256 or "— (in-memory frame)"],
    ]
    lines.append(_table(["Property", "Value"], rows))
    lines.append("")

    # --- severity summary ------------------------------------------------ #
    counts = {s: 0 for s in _SEVERITY_ORDER}
    for f in report.sorted_findings():
        counts[f.severity] += 1
    summary = ", ".join(
        f"{s.name.lower()} {counts[s]}" for s in _SEVERITY_ORDER if counts[s]
    ) or "no findings"
    lines.append("## 1. Finding summary")
    lines.append("")
    lines.append(f"Counts by severity: {summary}.")
    worst = next(
        (f for f in report.sorted_findings() if f.severity >= Severity.HIGH), None
    )
    if worst is not None:
        lines.append("")
        lines.append(f"**Worst finding:** {_finding_line(worst)}")
    lines.append("")

    # --- schema ---------------------------------------------------------- #
    lines.append("## 2. Schema and inferred column roles")
    lines.append("")
    lines.append(
        "_\"Role\" is an inference from stored dtype and content; stored dtype is never "
        "changed by this report. Missing = pandas-missing values (encoded text like "
        "\"NA\" is reported separately)._"
    )
    lines.append("")
    schema_rows: list[list[str]] = []
    for c in report.columns[:max_rows]:
        codes = ",".join(sorted({f.code for f in c.findings})) or "—"
        frac = c.n_missing / ov.n_rows if ov.n_rows else 0.0
        missing = f"{c.n_missing} ({_pct(frac)})" if c.n_missing else "0"
        schema_rows.append(
            [_esc(c.name), _esc(c.stored_dtype), c.kind.value, missing, _f(c.n_unique), _esc(codes)]
        )
    lines.append(
        _table(
            ["column", "stored dtype", "role", "missing", "unique", "findings"],
            schema_rows,
        )
    )
    lines.append(_truncated_note("columns", len(report.columns), max_rows))
    lines.append("")

    # --- numeric --------------------------------------------------------- #
    num_cols = [c for c in report.columns if c.kind == ColumnKind.NUMERIC and c.numeric]
    if num_cols:
        lines.append("## 3. Numeric columns")
        lines.append("")
        lines.append(
            "_Quantiles via linear interpolation; skew = adjusted Fisher–Pearson; "
            "kurt = excess kurtosis. Outliers: Tukey fences (1.5×IQR) and MAD-based "
            "robust z (3×1.4826×MAD) — screening rules, not verdicts._"
        )
        lines.append("")
        n_rows_n: list[list[str]] = []
        for c in num_cols[:max_rows]:
            st = c.numeric
            assert st is not None
            n_rows_n.append(
                [
                    _esc(c.name),
                    _f(st.count),
                    _f(st.min),
                    _f(st.q1),
                    _f(st.median),
                    _f(st.q3),
                    _f(st.max),
                    _f(st.mean),
                    _f(st.std),
                    _f(st.skew),
                    _f(st.kurtosis),
                    f"{st.n_outliers_tukey}/{st.n_outliers_mad}",
                ]
            )
        lines.append(
            _table(
                [
                    "column", "n", "min", "q1", "median", "q3", "max",
                    "mean", "std", "skew", "kurt", "outliers T/M",
                ],
                n_rows_n,
            )
        )
        lines.append(_truncated_note("numeric columns", len(num_cols), max_rows))
        lines.append("")

    # --- categorical ------------------------------------------------------ #
    cat_cols = [
        c
        for c in report.columns
        if c.kind in (ColumnKind.CATEGORICAL, ColumnKind.BOOLEAN) and c.categorical
    ]
    if cat_cols:
        lines.append("## 4. Categorical columns")
        lines.append("")
        lines.append(
            "_Entropy in bits over the empirical distribution; dominant = most common "
            "category share. Imbalance = max/min category count._"
        )
        lines.append("")
        cat_rows: list[list[str]] = []
        for c in cat_cols[:max_rows]:
            st = c.categorical
            assert st is not None
            top3 = ", ".join(f"{label} {_pct(frac)}" for label, _cnt, frac in st.top[:3])
            cat_rows.append(
                [
                    _esc(c.name),
                    _f(st.cardinality),
                    _f(st.entropy_bits),
                    _pct(st.dominant_fraction) if st.dominant_fraction is not None else "–",
                    _f(st.imbalance_ratio),
                    _esc(top3),
                ]
            )
        lines.append(
            _table(
                ["column", "categories", "entropy (bits)", "dominant", "max/min", "top categories"],
                cat_rows,
            )
        )
        lines.append(_truncated_note("categorical columns", len(cat_cols), max_rows))
        lines.append("")

    # --- datetime --------------------------------------------------------- #
    dt_cols = [c for c in report.columns if c.kind == ColumnKind.DATETIME]
    if dt_cols:
        lines.append("## 5. Datetime columns")
        lines.append("")
        dt_rows: list[list[str]] = []
        for c in dt_cols[:max_rows]:
            st = c.datetime
            assert st is not None
            if st.median_gap_seconds is not None:
                gap = _fmt_duration_secs(st.median_gap_seconds)
            else:
                gap = "–"
            span = (
                _fmt_duration_secs(st.span_seconds) if st.span_seconds is not None else "–"
            )
            dt_rows.append(
                [
                    _esc(c.name),
                    _esc(st.min or "–"),
                    _esc(st.max or "–"),
                    span,
                    gap,
                    "aware" if st.timezone_aware else "naive",
                ]
            )
        lines.append(
            _table(
                ["column", "min", "max", "span", "median gap", "tz"],
                dt_rows,
            )
        )
        lines.append(_truncated_note("datetime columns", len(dt_cols), max_rows))
        lines.append("")

    # --- identifiers / text ----------------------------------------------- #
    id_text = [c for c in report.columns if c.kind in (ColumnKind.IDENTIFIER, ColumnKind.TEXT)]
    if id_text:
        lines.append("## 6. Identifier / free-text columns")
        lines.append("")
        lines.append(
            "_Identifiers must not be used as features; free text is out of scope for "
            "Phase 1 modeling. Both are excluded from numeric association scans._"
        )
        lines.append("")
        it_rows: list[list[str]] = []
        for c in id_text[:max_rows]:
            mean_len = _f(c.text.mean_length) if c.text and c.text.mean_length is not None else "–"
            max_len = _f(c.text.max_length) if c.text and c.text.max_length is not None else "–"
            it_rows.append([_esc(c.name), c.kind.value, _f(c.n_unique), mean_len, max_len])
        lines.append(
            _table(["column", "role", "unique", "mean length", "max length"], it_rows)
        )
        lines.append(_truncated_note("identifier/text columns", len(id_text), max_rows))
        lines.append("")

    # --- associations ------------------------------------------------------ #
    lines.append("## 7. Associations with the target")
    lines.append("")
    if ov.target is None:
        lines.append(
            "No target given, so no target-aware associations or leakage heuristics were "
            "computed. Feature–feature correlation highlights appear in the findings."
        )
    elif not report.associations:
        lines.append(
            f"No reportable associations found with target '{ov.target}' "
            "(no numeric/binary-comparable features or all below screening thresholds)."
        )
    else:
        lines.append(
            "_Association measures only: Pearson r (linear), point-biserial r (numeric "
            "feature vs binary target), Cramer's V (categorical association). ⚠ = at or "
            "above the strong-association threshold. Association ≠ causation._"
        )
        lines.append("")
        a_rows: list[list[str]] = []
        for a in report.associations[:max_rows]:
            a_rows.append(
                [
                    _esc(a.left),
                    _esc(a.right),
                    _METHOD_LABELS.get(a.method, a.method),
                    _f(a.value),
                    "⚠" if a.flagged else "",
                ]
            )
        lines.append(
            _table(["feature", "target", "measure", "value", "strong"], a_rows)
        )
        lines.append(_truncated_note("associations", len(report.associations), max_rows))
    lines.append("")

    # --- leakage ------------------------------------------------------------ #
    lines.append("## 8. Leakage / target-quality heuristics")
    lines.append("")
    if not report.leak_findings:
        lines.append(
            "No leakage heuristics triggered. Remember: absence of a flag is not proof "
            "of safety — contamination across files and post-hoc feature creation cannot "
            "be seen from one table."
        )
    else:
        for f in sorted(report.leak_findings, key=lambda f: (-int(f.severity), f.code)):
            lines.append(_finding_line(f))
        lines.append("")
        lines.append(
            "_These are heuristic candidates requiring human review — near-perfect "
            "association is also compatible with legitimate duplicate features._"
        )
    lines.append("")

    # --- all findings -------------------------------------------------------- #
    lines.append("## 9. All findings")
    lines.append("")
    findings = report.sorted_findings()
    if not findings:
        lines.append("No findings. (A clean profile is still not a blank check: see limitations.)")
    else:
        for f in findings[:max_rows]:
            lines.append(_finding_line(f))
        lines.append(_truncated_note("findings", len(findings), max_rows))
    lines.append("")

    # --- notes ---------------------------------------------------------------- #
    lines.append("## 10. Scope and limitations")
    lines.append("")
    for note in report.notes:
        lines.append(f"- {note}")
    lines.append("")
    return "\n".join(lines)


def _fmt_duration_secs(seconds: float) -> str:
    if seconds >= 86400:
        return f"{seconds / 86400:.1f} d"
    if seconds >= 3600:
        return f"{seconds / 3600:.1f} h"
    if seconds >= 60:
        return f"{seconds / 60:.1f} min"
    return f"{seconds:.0f} s"
