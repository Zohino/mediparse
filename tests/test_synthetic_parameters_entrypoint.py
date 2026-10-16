"""Vstupní bod převzetí parametrů: zápis do configu a odmítnutí s důvodem."""

from __future__ import annotations

import json
import shutil
from typing import TYPE_CHECKING

from mediparse.domain.diagnosis import Diagnosis
from mediparse.entrypoints.exit_code import ExitCode
from mediparse.entrypoints.synthetic_parameters import run
from mediparse.infrastructure.sampler_config import load_sampler_config
from tests.support import REPOSITORY_CONFIG, export_payload

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


EXPECTED_NOTES = 400


def _setup(tmp_path: Path, selection: tuple[str, ...]) -> tuple[Path, Path]:
    export = tmp_path / "export.json"
    export.write_text(
        json.dumps(export_payload(selection, prevalence=0.0636)), encoding="utf-8"
    )
    config = tmp_path / "synthetic_plan.json"
    shutil.copyfile(REPOSITORY_CONFIG, config)
    return export, config


def test_ok_writes_config_and_logs_notes(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Po zápisu je návratový kód OK, počet zpráv v configu i v logu."""
    export, config = _setup(tmp_path, tuple(Diagnosis))
    caplog.set_level("INFO")

    code = run(["--export", str(export), "--config", str(config)])

    assert code == ExitCode.OK
    assert load_sampler_config(config).patients.notes == EXPECTED_NOTES
    assert str(EXPECTED_NOTES) in caplog.text


def test_missing_export_is_refused(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Chybějící export odmítne příkaz s důvodem a config zůstane."""
    _, config = _setup(tmp_path, tuple(Diagnosis))
    before = config.read_bytes()

    code = run(["--export", str(tmp_path / "missing.json"), "--config", str(config)])

    assert code == ExitCode.REFUSED
    assert "neexistuje" in caplog.text
    assert config.read_bytes() == before


def test_refused_config_is_refused_with_reason(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Pětice mimo doménu: odmítnutí s důvodem a nezměněný config."""
    export, config = _setup(tmp_path, ("a", "b", "c", "d", "e"))
    before = config.read_bytes()

    code = run(["--export", str(export), "--config", str(config)])

    assert code == ExitCode.REFUSED
    assert "schématu" in caplog.text
    assert config.read_bytes() == before


def test_non_converging_config_is_refused_and_untouched(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Config, jehož IPF s převzatými parametry nezkonverguje, zůstane bajtově stejný."""
    export, config = _setup(tmp_path, tuple(Diagnosis))
    correlated = export_payload(tuple(Diagnosis), prevalence=0.2)
    for row, columns in correlated["conditional"].items():
        for column in columns:
            columns[column] = 1.0 if row == column else 0.15
    export.write_text(json.dumps(correlated), encoding="utf-8")
    payload = json.loads(config.read_text(encoding="utf-8"))
    payload["labels"]["ipf_max_passes"] = 1
    config.write_text(json.dumps(payload), encoding="utf-8")
    before = config.read_bytes()

    code = run(["--export", str(export), "--config", str(config)])

    assert code == ExitCode.REFUSED
    assert "IPF" in caplog.text
    assert config.read_bytes() == before
