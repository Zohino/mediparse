"""Provenance renderu EDA: revize, otisk zámku, zdroj dat a verze knihoven."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from importlib.metadata import version
from typing import TYPE_CHECKING, Final

import pytest

from mediparse.entrypoints import eda_provenance
from mediparse.entrypoints.eda_provenance import provenance
from mediparse.entrypoints.exit_code import ExitCode
from mediparse.infrastructure.synthetic_marker import SyntheticMarker, write_marker
from tests.git_support import git_output, init_repo, needs_git

if TYPE_CHECKING:
    from pathlib import Path

SHA: Final = "b" * 40
CSV_SHA: Final = "c" * 64
ENVIRONMENT: Final = {"MEDIPARSE_COMMIT": SHA, "MEDIPARSE_DIRTY": "true"}


def _root(tmp_path: Path) -> Path:
    root = tmp_path / "root"
    (root / "config").mkdir(parents=True)
    (root / "pixi.lock").write_text("zámek\n", encoding="utf-8")
    tables = {
        "mimic_tables": [
            {
                "url": "https://physionet.org/files/mimic-iv-note/2.2/note/discharge.csv.gz",
                "sha256": CSV_SHA,
            }
        ]
    }
    (root / "config" / "mimic_tables.json").write_text(
        json.dumps(tables), encoding="utf-8"
    )
    return root


def _parquet(tmp_path: Path) -> Path:
    parquet = tmp_path / "parquet"
    parquet.mkdir()
    return parquet


def test_synthetic_tables_are_named_without_mimic_hashes(tmp_path: Path) -> None:
    """Adresář se synthetic.json je zdroj syntetický a otisky MIMIC v provenance nejsou."""
    parquet = _parquet(tmp_path)
    write_marker(parquet, SyntheticMarker(prefix="canary", rows=12))

    result = provenance(ENVIRONMENT, _root(tmp_path), parquet)

    assert result["data_source"] == "syntetické tabulky (prefix canary)"
    assert CSV_SHA not in result.values()
    assert not [key for key in result if key.startswith("csv_sha256")]


def test_real_tables_carry_csv_hashes_from_config(tmp_path: Path) -> None:
    """Bez synthetic.json jsou zdrojem otisky CSV z configu."""
    result = provenance(ENVIRONMENT, _root(tmp_path), _parquet(tmp_path))

    assert result["data_source"] == "otisky CSV z config/mimic_tables.json"
    assert result["csv_sha256 discharge.csv.gz"] == CSV_SHA


def test_commit_and_dirty_come_from_environment(tmp_path: Path) -> None:
    """Proměnné MEDIPARSE_COMMIT a MEDIPARSE_DIRTY mají přednost před gitem."""
    result = provenance(ENVIRONMENT, _root(tmp_path), _parquet(tmp_path))

    assert result["commit"] == SHA
    assert result["dirty"] == "true"


@needs_git
def test_commit_and_dirty_come_from_git_without_environment(tmp_path: Path) -> None:
    """Bez proměnných se revize čte z gitu v kořeni."""
    root = tmp_path / "repo"
    init_repo(root, "a\n")
    (root / "config").mkdir()
    (root / "config" / "mimic_tables.json").write_text(
        '{"mimic_tables": []}', encoding="utf-8"
    )
    (root / "pixi.lock").write_text("zámek\n", encoding="utf-8")

    result = provenance({}, root, _parquet(tmp_path))

    assert result["commit"] == git_output(root, "rev-parse", "HEAD")
    assert result["dirty"] == "true"


def test_lock_hash_matches_file(tmp_path: Path) -> None:
    """Otisk pixi.lock je sha256 souboru."""
    root = _root(tmp_path)

    result = provenance(ENVIRONMENT, root, _parquet(tmp_path))

    assert (
        result["pixi_lock_sha256"]
        == hashlib.sha256((root / "pixi.lock").read_bytes()).hexdigest()
    )


def test_versions_of_python_and_duckdb(tmp_path: Path) -> None:
    """Provenance nese verzi Pythonu a duckdb."""
    result = provenance(ENVIRONMENT, _root(tmp_path), _parquet(tmp_path))

    assert result["python"] == platform.python_version()
    assert result["duckdb"] == version("duckdb")


def test_run_writes_ordered_provenance(tmp_path: Path) -> None:
    """Nástroj zapíše pořadí položek a nezávisle ověřené údaje renderu."""
    root = _root(tmp_path)
    parquet = _parquet(tmp_path)
    write_marker(parquet, SyntheticMarker(prefix="canary", rows=12))
    out = tmp_path / "output" / "provenance.json"
    eda_provenance.run(root, parquet, out, ENVIRONMENT)
    actual = json.loads(out.read_text(encoding="utf-8"))
    assert list(actual.items()) == list(provenance(ENVIRONMENT, root, parquet).items())
    assert actual["commit"] == SHA
    assert actual["dirty"] == "true"
    assert actual["pixi_lock_sha256"] == hashlib.sha256("zámek\n".encode()).hexdigest()
    assert actual["data_source"] == "syntetické tabulky (prefix canary)"


@pytest.mark.parametrize("valid", [True, False])
def test_main_exit_codes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, valid: bool
) -> None:
    """CLI předá cesty a prostředí, odmítnutý vstup vrací REFUSED."""
    root = _root(tmp_path)
    parquet = _parquet(tmp_path)
    out = tmp_path / "provenance.json"
    if not valid:
        (parquet / "synthetic.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mediparse-eda-provenance",
            "--root",
            str(root),
            "--parquet",
            str(parquet),
            "--out",
            str(out),
        ],
    )
    for name, value in ENVIRONMENT.items():
        monkeypatch.setenv(name, value)
    assert eda_provenance.main() is (ExitCode.OK if valid else ExitCode.REFUSED)
    assert out.exists() is valid


def test_run_preserves_json_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Zápis zachovává konkrétní bajty UTF-8, pořadí, odsazení i koncový řádek."""
    details = {
        "commit": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "dirty": "false",
        "pixi_lock_sha256": "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
        "data_source": "syntetické tabulky (prefix ukázka)",
        "python": "3.13.9",
        "duckdb": "1.5.6",
    }
    monkeypatch.setattr(eda_provenance, "provenance", lambda *_: details)
    out = tmp_path / "output" / "provenance.json"
    eda_provenance.run(tmp_path, tmp_path, out, {})
    assert (
        out.read_bytes()
        == (
            "{\n"
            '  "commit": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",\n'
            '  "dirty": "false",\n'
            '  "pixi_lock_sha256": "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",\n'
            '  "data_source": "syntetické tabulky (prefix ukázka)",\n'
            '  "python": "3.13.9",\n'
            '  "duckdb": "1.5.6"\n'
            "}\n"
        ).encode()
    )
