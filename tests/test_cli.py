from __future__ import annotations

import json

import pandas as pd

from nullius.cli import main


def test_cli_profile_json(tmp_path, capsys):
    df = pd.DataFrame(
        {
            "id": range(50),
            "churn": [0, 1] * 25,
            "score": [1.0, 5.0] * 25,
            "plan": ["a", "b"] * 25,
        }
    )
    p = tmp_path / "data.csv"
    df.to_csv(p, index=False)
    out = tmp_path / "report.json"

    rc = main(
        ["data", "profile", str(p), "--target", "churn", "--format", "json", "--out", str(out)]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["overview"]["source_sha256"]
    assert payload["overview"]["n_rows"] == 50
    assert any(f["code"] == "LEAK_NEAR_PERFECT_ASSOC" for f in payload["leak_findings"])
    on_disk = json.loads(out.read_text(encoding="utf-8"))
    assert on_disk["overview"]["dataset_name"] == "data"


def test_cli_profile_markdown(tmp_path, capsys):
    p = tmp_path / "tiny.csv"
    pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": ["x", "y", "x"]}).to_csv(p, index=False)
    rc = main(["data", "profile", str(p)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "# Data Understanding Report" in out
    assert "## 1. Finding summary" in out
    assert "Scope and limitations" in out


def test_cli_missing_file(tmp_path, capsys):
    rc = main(["data", "profile", str(tmp_path / "nope.csv")])
    assert rc == 2
    assert "not found" in capsys.readouterr().err


def test_cli_bad_target(tmp_path, capsys):
    p = tmp_path / "d.csv"
    pd.DataFrame({"a": [1.0]}).to_csv(p, index=False)
    rc = main(["data", "profile", str(p), "--target", "zzz"])
    assert rc == 2
    assert "target column 'zzz' not found" in capsys.readouterr().err


def test_help():
    import pytest

    with pytest.raises(SystemExit) as exc:
        main(["data", "profile", "--help"])
    assert exc.value.code == 0
