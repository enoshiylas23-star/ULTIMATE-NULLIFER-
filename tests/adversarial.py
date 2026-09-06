"""Deterministic generators for failure-first datasets (spec §20).

Each builder returns ``(df, expects)`` where ``expects`` documents which
profiler codes the fixture is designed to trigger. Fixtures are generated in
memory — no committed data blobs — with fixed RNG seeds so runs are stable.
"""

from __future__ import annotations

import uuid

import numpy as np
import pandas as pd


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def duplicates(n_base: int = 80, n_copies: int = 20) -> tuple[pd.DataFrame, dict]:
    rng = _rng(7)
    rows = {
        "id": list(range(n_base)),
        "amount": np.round(rng.normal(100, 15, n_base), 2),
        "b": rng.integers(0, 5, n_base),
        "cat": rng.choice(["A", "B", "C"], n_base, p=[0.5, 0.3, 0.2]),
        "ts": [f"2024-03-{1 + i % 28:02d}" for i in range(n_base)],
    }
    df = pd.DataFrame(rows)
    df = pd.concat([df, df.iloc[:n_copies].reset_index(drop=True)], ignore_index=True)
    return df, {"codes": {"DUP_ROWS"}, "n_redundant": n_copies}


def missing_encoded(n: int = 300) -> tuple[pd.DataFrame, dict]:
    rng = _rng(11)
    amount = [float(np.round(rng.normal(50, 8))) if i % 12 else np.nan for i in range(n)]
    words = ["apple", "banana", "cherry", "date", "elderberry", "fig", "grape"]
    text, expected = [], 0
    for i in range(n):
        if i % 9 == 0:
            text.append("")
            expected += 1
        elif i % 11 == 0:
            text.append("NA")
            expected += 1
        elif i % 13 == 0:
            text.append("nan")
            expected += 1
        elif i % 17 == 0:
            text.append(" ")
            expected += 1
        else:
            text.append(rng.choice(words) + str(i))
    df = pd.DataFrame({"amount": amount, "free_text": text})
    return df, {"codes": {"MISSING_ENCODED", "COL_MISSING"}, "n_encoded": expected}


def constants_and_all_missing(n: int = 200) -> tuple[pd.DataFrame, dict]:
    rng = _rng(13)
    df = pd.DataFrame(
        {
            "const": ["K"] * n,
            "zero": [0] * n,
            "val": rng.normal(0, 1, n),
            "empty_col": [np.nan] * n,
        }
    )
    return df, {
        "codes": {"CONSTANT_COL", "COL_ALL_MISSING"},
        "constant_cols": {"const", "zero"},
    }


def identifiers(n: int = 300) -> tuple[pd.DataFrame, dict]:
    rng = _rng(17)
    df = pd.DataFrame(
        {
            "customer_id": [str(uuid.uuid4()) for _ in range(n)],
            "session_key": [f"{int(rng.integers(0, 2**32)):032x}" for _ in range(n)],
            "product_code": [f"prod-{i % 20}" for i in range(n)],
            "price": np.round(rng.normal(25, 5, n), 2),
        }
    )
    return df, {
        "codes": {"ID_COLUMN"},
        "identifier_cols": {"customer_id", "session_key"},
        "categorical_cols": {"product_code"},
    }


def numeric_as_string(n: int = 500) -> tuple[pd.DataFrame, dict]:
    rng = _rng(19)
    amounts = [f"{rng.normal(100, 20):.2f}" for _ in range(n)]
    words = ["low", "med", "high", "unknown", "n/a?"]
    mixed = [f"{rng.integers(1, 1000)}" if i < n // 2 else rng.choice(words) for i in range(n)]
    df = pd.DataFrame(
        {
            "amount_str": amounts,
            "mixed": mixed,
            "score": rng.normal(0, 1, n),
        }
    )
    return df, {
        "numeric_kind": "amount_str",
        "mixed_kind": "mixed",
        "codes": {"NUM_AS_STRING", "NUM_STRING_MIXED"},
    }


def datetime_text(n: int = 200) -> tuple[pd.DataFrame, dict]:
    days = [1 + i % 28 for i in range(n)]
    ts = [f"2024-01-{d:02d} 08:00:00" for d in days]
    df = pd.DataFrame({"ts_text": ts, "val": np.arange(n, dtype=float)})
    return df, {"codes": {"DATETIME_AS_STRING"}, "kind": "ts_text"}


def datetime_gaps() -> tuple[pd.DataFrame, dict]:
    days = list(range(0, 30)) + list(range(120, 150))
    base = pd.Timestamp("2024-01-01")
    t = base + pd.to_timedelta(np.asarray(days, dtype="int64"), unit="D")
    df = pd.DataFrame({"t": t, "v": np.arange(len(days), dtype=float)})
    return df, {"codes": {"DATETIME_IRREGULAR"}}


def leak_numeric_target(n: int = 2000) -> tuple[pd.DataFrame, dict]:
    rng = _rng(23)
    churn = rng.choice([0, 1], n, p=[0.6, 0.4])
    score = 3.0 * churn + rng.normal(0, 0.05, n)
    df = pd.DataFrame(
        {
            "id": np.arange(n),
            "churn": churn,
            "churn_flag": churn.copy(),
            "score": score,
            "tenure": np.round(rng.exponential(24, n), 1),
            "plan": rng.choice(["A", "B", "C"], n, p=[0.5, 0.3, 0.2]),
        }
    )
    return df, {
        "target": "churn",
        "codes": {"LEAK_TARGET_DUPLICATE", "LEAK_NEAR_PERFECT_ASSOC"},
        "dup_col": "churn_flag",
        "proxy_col": "score",
    }


def leak_categorical_target(n: int = 1500) -> tuple[pd.DataFrame, dict]:
    rng = _rng(29)
    churned = rng.choice(["yes", "no"], n, p=[0.92, 0.08])
    y = (churned == "no").astype(int)
    risk = 3.0 * y + rng.normal(0, 0.05, n)
    df = pd.DataFrame(
        {
            "churned": churned,
            "churned_flag": churned.copy(),
            "risk_score": risk,
            "plan": rng.choice(["X", "Y", "Z"], n, p=[0.5, 0.3, 0.2]),
            "usage": np.round(rng.normal(120, 30, n), 1),
        }
    )
    return df, {
        "target": "churned",
        "codes": {
            "LEAK_TARGET_DUPLICATE",
            "LEAK_NEAR_PERFECT_ASSOC",
            "TARGET_IMBALANCE",
        },
        "dup_col": "churned_flag",
        "proxy_col": "risk_score",
    }


def imbalanced_bool_target(n: int = 5000) -> tuple[pd.DataFrame, dict]:
    rng = _rng(31)
    target = rng.random(n) < 0.02
    df = pd.DataFrame(
        {
            "event": target,
            "x1": rng.normal(0, 1, n),
            "x2": rng.normal(0, 1, n),
            "grp": rng.choice(list("LMNOP"), n),
        }
    )
    return df, {"target": "event", "codes": {"TARGET_IMBALANCE"}}


def skewed_with_outliers(n: int = 3000) -> tuple[pd.DataFrame, dict]:
    rng = _rng(37)
    revenue = np.exp(rng.normal(0, 2.0, n))
    revenue[:3] = 1e7  # injected extreme values
    df = pd.DataFrame({"revenue": revenue, "id": np.arange(n)})
    return df, {"codes": {"HIGH_SKEW", "OUTLIERS"}}

def high_corr_features(n: int = 1500) -> tuple[pd.DataFrame, dict]:
    rng = _rng(41)
    x = rng.normal(0, 1, n)
    y = 3.0 * x + rng.normal(0, 0.05, n)
    z = rng.normal(0, 1, n)
    w = -2.0 * x + rng.normal(0, 0.05, n)
    df = pd.DataFrame({"x": x, "y": y, "z": z, "w": w})
    return df, {"codes": {"CORR_HIGH"}, "min_near_perfect": 1}


def rows_partial_missing(n: int = 300) -> tuple[pd.DataFrame, dict]:
    rng = _rng(43)
    a = rng.normal(0, 1, n)
    a[list(range(0, 20))] = np.nan  # 20 missing in column a
    df = pd.DataFrame({"a": a, "b": rng.normal(0, 1, n), "c": rng.normal(0, 1, n)})
    df.iloc[299] = [np.nan, np.nan, np.nan]  # one fully empty row
    return df, {"codes": {"ROWS_WITH_MISSING", "ROWS_ALL_MISSING", "COL_MISSING"}}


def tiny() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0],
            "b": ["x", "y", "x"],
            "c": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
        }
    )


def wide(n_rows: int = 12, p: int = 70) -> pd.DataFrame:
    rng = _rng(47)
    data = {f"noise_{i:02d}": rng.normal(0, 1, n_rows) for i in range(p)}
    data["signal"] = rng.normal(0, 1, n_rows)
    return pd.DataFrame(data)


def binary_categorical_target(n: int = 1200) -> tuple[pd.DataFrame, dict]:
    """Two-class string target + strong (non-leaky) categorical association."""
    rng = _rng(53)
    seg = rng.choice(["a", "b", "c"], n, p=[0.5, 0.3, 0.2])
    p_yes = {"a": 0.85, "b": 0.5, "c": 0.2}
    outcome = [("yes" if rng.random() < p_yes[s] else "no") for s in seg]
    df = pd.DataFrame(
        {
            "outcome": outcome,
            "segment": seg,
            "value": rng.normal(0, 1, n),
        }
    )
    return df, {"target": "outcome", "codes": set()}
