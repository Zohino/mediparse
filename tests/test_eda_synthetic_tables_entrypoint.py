"""CLI EDA zapisuje celý manifest až po validaci všech zdrojů korpusu."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Final

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from mediparse.entrypoints.eda_synthetic_tables import run
from mediparse.entrypoints.exit_code import ExitCode
from mediparse.infrastructure.mimic_inventory import read_manifest
from mediparse.infrastructure.plans_file import PlansFile
from mediparse.infrastructure.synthetic_marker import SyntheticMarker, read_marker
from tests.support import (
    REPOSITORY_CORPUS,
    REPOSITORY_DIAGNOSIS_CANDIDATES,
    REPOSITORY_MIMIC_MANIFEST,
    REPOSITORY_PLANS,
    REPOSITORY_SECTION_HEADERS,
)
from tests.synthetic_support import PLANS

if TYPE_CHECKING:
    from pathlib import Path

PREFIX: Final = "eda-sentinel"
EXPECTED_NOTES: Final = 400


def _arguments(out: Path, **overrides: Path) -> list[str]:
    paths = {
        "plans": REPOSITORY_PLANS,
        "corpus": REPOSITORY_CORPUS,
        "manifest": REPOSITORY_MIMIC_MANIFEST,
        "candidates": REPOSITORY_DIAGNOSIS_CANDIDATES,
        "lexicon": REPOSITORY_SECTION_HEADERS,
        "out": out,
    } | overrides
    return [
        argument
        for name, path in paths.items()
        for argument in (f"--{name}", str(path))
    ] + ["--prefix", PREFIX]


def test_writes_manifest_tables_and_marker(tmp_path: Path) -> None:
    """Všech 13 tabulek má řetězcové sloupce manifestu; značka počítá zprávy korpusu."""
    out = tmp_path / "tables"
    assert run(_arguments(out)) is ExitCode.OK
    shapes = read_manifest(REPOSITORY_MIMIC_MANIFEST)
    assert {path.stem for path in out.glob("*.parquet")} == set(shapes)
    for name, shape in shapes.items():
        table = pq.read_table(out / f"{name}.parquet")
        assert table.column_names == list(shape.columns)
        assert all(field.type == pa.string() for field in table.schema)
        assert table.num_rows > 0
    assert pq.read_table(out / "discharge.parquet").num_rows == EXPECTED_NOTES
    assert read_marker(out) == SyntheticMarker(prefix=PREFIX, rows=EXPECTED_NOTES)
    originals = {
        path.stem: path.read_text(encoding="utf-8")
        for path in (REPOSITORY_CORPUS / "en").glob("*.txt")
    }
    notes = pq.read_table(out / "discharge.parquet").to_pylist()
    for note in notes:
        note_id = note["note_id"].removeprefix(f"{PREFIX}-discharge-note_id-")
        assert note["text"].split("\n", 1)[1] == originals[note_id]


@pytest.mark.parametrize("source", ["manifest", "lexicon", "plans"])
def test_invalid_source_is_refused_without_output(tmp_path: Path, source: str) -> None:
    """Chybějící manifest, neplatný lexikon a nespárované plány nezanechají výstup."""
    broken = tmp_path / "broken.json"
    if source == "lexicon":
        broken.write_text("{", encoding="utf-8")
    elif source == "plans":
        PlansFile(broken).save(PLANS)
    out = tmp_path / "tables"
    assert run(_arguments(out, **{source: broken})) is ExitCode.REFUSED
    assert not out.exists()


def test_invalid_last_table_leaves_no_partial_output(tmp_path: Path) -> None:
    """Validace poslední tabulky předchází zápisu první tabulky."""
    manifest = json.loads(REPOSITORY_MIMIC_MANIFEST.read_text(encoding="utf-8"))
    manifest["tables"][-1]["columns"] = []
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    out = tmp_path / "tables"
    assert run(_arguments(out, manifest=path)) is ExitCode.REFUSED
    assert not out.exists()


@pytest.mark.parametrize("missing", ["column", "table"])
def test_manifest_omitting_corpus_facts_leaves_no_output(
    tmp_path: Path,
    missing: str,
) -> None:
    """Odmítnutí ztráty korpusového sloupce nebo tabulky předchází všem zápisům."""
    manifest = json.loads(REPOSITORY_MIMIC_MANIFEST.read_text(encoding="utf-8"))
    if missing == "column":
        admission = next(
            table
            for table in manifest["tables"]
            if table["file"] == "admissions.csv.gz"
        )
        admission["columns"].remove("race")
    else:
        manifest["tables"] = [
            table for table in manifest["tables"] if table["file"] != "patients.csv.gz"
        ]
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    out = tmp_path / "tables"
    assert run(_arguments(out, manifest=path)) is ExitCode.REFUSED
    assert not out.exists()
