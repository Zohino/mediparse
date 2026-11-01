"""Kanárek pokrývá doplněné buňky a vazby korpusu bez změny jeho textu."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Final

import pytest

from mediparse.domain.inputs import InvalidInputError
from mediparse.domain.mimic_table import TableShape
from mediparse.domain.synthetic_mimic import corpus_tables
from mediparse.domain.synthetic_sentinel import RARE_ROWS, sentinel_tables
from mediparse.infrastructure.diagnosis_candidates_file import load_diagnosis_codes
from mediparse.infrastructure.mimic_inventory import read_manifest
from tests.support import REPOSITORY_DIAGNOSIS_CANDIDATES, REPOSITORY_MIMIC_MANIFEST
from tests.synthetic_support import LEXICON, NOTES, PLANS, repository_tables

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

PREFIX: Final = "eda-sentinel"
SHAPES: Final = read_manifest(REPOSITORY_MIMIC_MANIFEST)
TEXT_CHARS: Final = 100
EXPECTED_RARE_ROWS: Final = 2
KEYS: Final = {"subject_id", "hadm_id", "note_id"}


@pytest.fixture(params=["F", "repository"])
def corpus(request: pytest.FixtureRequest) -> dict[str, list[dict[str, str]]]:
    """F i repozitářový korpus prochází stejnými invarianty kanárku.

    Returns:
        Neoznačené tabulky společného jádra.
    """
    if request.param == "repository":
        return repository_tables()
    return corpus_tables(
        PLANS, NOTES, load_diagnosis_codes(REPOSITORY_DIAGNOSIS_CANDIDATES), LEXICON
    )


def _sources(
    table: str, corpus: Mapping[str, Sequence[Mapping[str, str]]]
) -> Sequence[Mapping[str, str]]:
    if table in corpus:
        return corpus[table]
    return [{key: row[key] for key in KEYS} for row in corpus["discharge"]]


@pytest.mark.parametrize("table", SHAPES)
def test_missing_cells_are_unique_and_manifest_order_is_preserved(
    corpus: dict[str, list[dict[str, str]]],
    table: str,
) -> None:
    """A5a,e: každá nedodaná buňka nese prefix, každý sloupec pořadí manifestu."""
    rows = sentinel_tables(corpus, SHAPES, PREFIX, TEXT_CHARS)[table]
    assert all(tuple(row) == SHAPES[table].columns for row in rows)
    cells = []
    for row, source in zip(rows, _sources(table, corpus), strict=False):
        for column, value in row.items():
            if column not in source and column not in KEYS:
                assert value.startswith(f"{PREFIX}-")
                cells.append(value)
            elif column not in KEYS and column != "text":
                assert value == source[column]
    assert len(cells) == len(set(cells))
    if table not in corpus:
        assert len(rows) == len(corpus["discharge"])


def test_identity_links_are_consistent(corpus: dict[str, list[dict[str, str]]]) -> None:
    """A5b: stejný klíč pacienta, přijetí i zprávy odkazuje na stejnou buňku."""
    tables = sentinel_tables(corpus, SHAPES, PREFIX, TEXT_CHARS)
    patients = {
        old["subject_id"]: new["subject_id"]
        for old, new in zip(corpus["patients"], tables["patients"], strict=False)
    }
    admissions = {
        old["hadm_id"]: new["hadm_id"]
        for old, new in zip(corpus["admissions"], tables["admissions"], strict=False)
    }
    notes = {
        old["note_id"]: new["note_id"]
        for old, new in zip(corpus["discharge"], tables["discharge"], strict=True)
    }
    assert all(
        value.startswith(f"{PREFIX}-")
        for value in (*patients.values(), *admissions.values(), *notes.values())
    )
    for table, rows in tables.items():
        for source, row in zip(_sources(table, corpus), rows, strict=False):
            if "subject_id" in row:
                assert row["subject_id"] == patients[source["subject_id"]]
            if "hadm_id" in row:
                assert row["hadm_id"] == admissions[source["hadm_id"]]
            if table == "mimic-iv-bhc":
                assert row["note_id"] == notes[source["note_id"]]
    assert {row["note_id"] for row in tables["radiology"]}.isdisjoint(notes.values())


def test_discharge_text_is_unchanged_after_canary_line(
    corpus: dict[str, list[dict[str, str]]],
) -> None:
    """A5c: jediný nový řádek obsahuje buňku a různý token ze samých písmen."""
    tables = sentinel_tables(corpus, SHAPES, PREFIX, TEXT_CHARS)
    tokens = []
    for original, decorated in zip(
        corpus["discharge"], tables["discharge"], strict=True
    ):
        head, text = decorated["text"].split("\n", 1)
        cell, token = head.split()
        assert cell.startswith(f"{PREFIX}-")
        assert token.isalpha()
        assert text == original["text"]
        tokens.append(token)
    assert len(tokens) == len(set(tokens))
    assert all(len(row["text"]) >= TEXT_CHARS for row in tables["radiology"])


def test_rare_witnesses_and_links(corpus: dict[str, list[dict[str, str]]]) -> None:
    """A5d: právě dva svědci mimo korpus sdílejí vazby a ostatní typové hodnoty."""
    tables = sentinel_tables(corpus, SHAPES, PREFIX, TEXT_CHARS)
    assert RARE_ROWS == EXPECTED_RARE_ROWS
    for table, columns in {
        "patients": ("anchor_year_group",),
        "diagnoses_icd": ("icd_version", "icd_code"),
    }.items():
        for column in columns:
            assert (
                sum(row[column].startswith(f"{PREFIX}-") for row in tables[table])
                == EXPECTED_RARE_ROWS
            )
    rare_subjects = {row["subject_id"] for row in tables["patients"][-2:]}
    rare_hadms = {row["hadm_id"] for row in tables["admissions"][-2:]}
    assert rare_subjects.isdisjoint(row["subject_id"] for row in tables["discharge"])
    assert rare_hadms.isdisjoint(row["hadm_id"] for row in tables["discharge"])
    for table in ("patients", "admissions", "diagnoses_icd"):
        assert len(tables[table]) == len(corpus[table]) + EXPECTED_RARE_ROWS
        assert {row["subject_id"] for row in tables[table][-2:]} == rare_subjects
    assert {row["hadm_id"] for row in tables["diagnoses_icd"][-2:]} == rare_hadms
    for row in tables["patients"][-2:]:
        assert row["gender"] == tables["patients"][0]["gender"]
        assert row["anchor_age"] == tables["patients"][0]["anchor_age"]


def test_decoration_is_deterministic_and_does_not_mutate_corpus() -> None:
    """Dva běhy se shodují a nesdílejí mutace s korpusovou vrstvou."""
    corpus = repository_tables()
    original = deepcopy(corpus)
    assert sentinel_tables(corpus, SHAPES, PREFIX, TEXT_CHARS) == sentinel_tables(
        corpus, SHAPES, PREFIX, TEXT_CHARS
    )
    assert set(sentinel_tables(corpus, SHAPES, PREFIX, TEXT_CHARS)) == set(SHAPES)
    assert corpus == original


@pytest.mark.parametrize("empty_prefix", [False, True])
def test_invalid_input_is_refused(*, empty_prefix: bool) -> None:
    """Prázdný prefix a prázdné sloupce nemohou vyrobit zdánlivě platnou tabulku."""
    shapes = SHAPES if empty_prefix else {"empty": TableShape(records=0, columns=())}
    with pytest.raises(InvalidInputError):
        sentinel_tables(
            repository_tables(), shapes, "" if empty_prefix else PREFIX, TEXT_CHARS
        )
