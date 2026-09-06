"""Helpers for asserting on DataUnderstandingReport contents."""

from __future__ import annotations

from nullius.data.records import DataUnderstandingReport, Finding


def all_findings(report: DataUnderstandingReport) -> list[Finding]:
    return report.sorted_findings()


def codes(report: DataUnderstandingReport) -> set[str]:
    return {f.code for f in all_findings(report)}


def findings_with_code(report: DataUnderstandingReport, code: str) -> list[Finding]:
    return [f for f in all_findings(report) if f.code == code]


def column(report: DataUnderstandingReport, name: str):
    for c in report.columns:
        if c.name == name:
            return c
    raise KeyError(f"column {name!r} not profiled")


def leak_codes(report: DataUnderstandingReport) -> set[str]:
    return {f.code for f in report.leak_findings}


def association(report: DataUnderstandingReport, feature: str):
    for a in report.associations:
        if a.left == feature:
            return a
    return None
