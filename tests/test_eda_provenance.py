"""Provenance renderu EDA: revize, otisk zámku, zdroj dat a verze knihoven."""

from __future__ import annotations

import hashlib
import json
import platform
from importlib.metadata import version
from pathlib import Path
from typing import Final

import pytest

from mediparse.domain.eda_provenance import provenance_rows
from mediparse.domain.inputs import InvalidInputError
from mediparse.domain.thesis_table import ThesisTable, latex_tabular, table_markdown
from mediparse.entrypoints.eda_provenance import appendix, provenance
from mediparse.infrastructure.synthetic_marker import SyntheticMarker, write_marker
from tests.git_support import git_output, init_repo, needs_git

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


FIXTURES = Path(__file__).parent / "fixtures" / "thesis_table"


def test_provenance_rows_match_r_golden() -> None:
    """České popisky a rozdělení otisků dávají stejný LaTeX jako R příloha."""
    details = json.loads((FIXTURES / "provenance.json").read_text(encoding="utf-8"))
    table = ThesisTable(
        header=("Položka", "Hodnota"),
        rows=provenance_rows(details),
        align="ll",
        widths=(28, 72),
    )
    assert (latex_tabular(table) + "\n").encode() == (
        FIXTURES / "provenance.tex"
    ).read_bytes()


@pytest.mark.parametrize(("dirty", "expected"), [("true", "ano"), ("false", "ne")])
def test_provenance_dirty_is_czech(dirty: str, expected: str) -> None:
    """Příznak dirty se překládá na ano nebo ne."""
    assert provenance_rows({"dirty": dirty}) == (("Necommitnuté změny", expected),)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("a" * 64, " ".join(["a" * 16] * 4)),
        ("a" * 40, "a" * 40),
        ("A" * 64, "A" * 64),
    ],
)
def test_provenance_digest_groups(value: str, expected: str) -> None:
    """Pouze 64 malých hexadecimálních znaků se dělí po šestnácti."""
    assert provenance_rows({"jiny": value}) == (("jiny", expected),)


def test_provenance_keeps_key_order_and_knows_r() -> None:
    """Popisky včetně R a neznámých klíčů zachovávají pořadí vstupu."""
    assert provenance_rows({"r": "4.5", "jiny": "hodnota", "commit": SHA}) == (
        ("R", "4.5"),
        ("jiny", "hodnota"),
        ("Revize kódu (commit)", SHA),
    )


def test_python_appendix_exports_provenance(tmp_path: Path) -> None:
    """Python příloha uloží tabulku a vrátí markdown s popiskem, id a šířkami."""
    root = _root(tmp_path)
    parquet = _parquet(tmp_path)
    write_marker(parquet, SyntheticMarker(prefix="canary", rows=12))
    environ = {
        **ENVIRONMENT,
        "PIXI_PROJECT_ROOT": str(root),
        "MEDIPARSE_PARQUET": str(parquet),
        "QUARTO_DOCUMENT_PATH": str(root / "eda"),
        "QUARTO_DOCUMENT_FILE": "1-dataset.qmd",
    }
    result = appendix(environ)
    rows = provenance_rows(provenance(environ, root, parquet))
    table = ThesisTable(
        header=("Položka", "Hodnota"), rows=rows, align="ll", widths=(28, 72)
    )
    destination = root / "results" / "eda" / "1-dataset" / "tbl-provenance.tex"
    assert destination.read_bytes() == (latex_tabular(table) + "\n").encode()
    assert result.text == table_markdown(
        table, label="tbl-provenance", caption="Provenance renderu."
    )
    assert '{#tbl-provenance tbl-colwidths="[28,72]"}' in result.text
    assert "syntetické tabulky (prefix canary)" in result.text
    assert "|R " not in result.text


def test_python_appendix_requires_project_root() -> None:
    """Příloha nepokračuje bez explicitního kořene projektu ani při zadaném parquetu."""
    with pytest.raises(InvalidInputError, match="PIXI_PROJECT_ROOT"):
        appendix({"MEDIPARSE_PARQUET": "parquet"})
