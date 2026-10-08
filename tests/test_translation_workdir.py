"""Pracovní adresář překladu: čtení požadavků, běhů a výstupů z disku."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Final

import pytest

from mediparse.domain.inputs import InvalidInputError
from mediparse.infrastructure.translation_workdir import TranslationWorkdir
from tests.workdir_support import (
    FIRST_START,
    LAST_START,
    PROBE_START,
    output,
    request,
    run_line,
    sha256_text,
    write_workdir,
)

if TYPE_CHECKING:
    from pathlib import Path

REQUESTS_SHA: Final = "a" * 64
WINDOW: Final = 2048


def _workdir(tmp_path: Path) -> Path:
    requests = [
        request("n1", "k1", "one"),
        request("n2", "k2a", "two"),
        request("n2", "k2b", "two"),
    ]
    runs = [
        run_line(PROBE_START, REQUESTS_SHA),
        run_line(FIRST_START, REQUESTS_SHA),
        run_line(LAST_START, "b" * 64),
    ]
    outputs = [
        output("probe-only", "x", PROBE_START),
        output("k1", "old", PROBE_START),
        output("k1", "jedna", FIRST_START),
        output("k2a", "dva a", LAST_START),
        output("k2b", "dva b", LAST_START),
    ]
    write_workdir(tmp_path, requests, runs, outputs)
    return tmp_path


def test_runs_are_only_those_that_produced_used_outputs(tmp_path: Path) -> None:
    """Běh, z něhož nepochází žádný výstup pro klíč požadavku, do provenance nepatří."""
    translation = TranslationWorkdir(_workdir(tmp_path)).translation()

    assert [run.started.isoformat() for run in translation.runs] == [
        FIRST_START,
        LAST_START,
    ]
    assert translation.runs[0].requests_sha256 == REQUESTS_SHA
    assert translation.runs[0].gpu == "NVIDIA A40"


def test_translation_carries_model_and_requests_digest(tmp_path: Path) -> None:
    """Model, revize, okno i dekódování jdou z requests.json, otisk z jeho bajtů."""
    workdir = _workdir(tmp_path)

    translation = TranslationWorkdir(workdir).translation()

    assert translation.model == "google/translategemma-12b-it"
    assert translation.window == WINDOW
    assert translation.decoding.stop_token_ids == (1, 106)
    assert translation.requests_sha256 == sha256_text(
        (workdir / "requests.json").read_text(encoding="utf-8")
    )
    assert translation.marker_mismatches == ()


def test_notes_join_parts_in_request_order_from_latest_output(tmp_path: Path) -> None:
    """Zprávy nesou části v pořadí požadavků; k jednomu klíči platí poslední výstup."""
    notes = TranslationWorkdir(_workdir(tmp_path)).notes()

    assert [(note.note_id, note.parts) for note in notes] == [
        ("n1", ("jedna",)),
        ("n2", ("dva a", "dva b")),
    ]
    assert notes[0].source_sha256 == sha256_text("one")


def test_missing_output_for_key_is_invalid_input(tmp_path: Path) -> None:
    """Požadavek bez výstupu znamená, že překlad není úplný."""
    write_workdir(
        tmp_path,
        [request("n1", "k1", "one")],
        [run_line(LAST_START, REQUESTS_SHA)],
        [],
    )

    with pytest.raises(InvalidInputError, match="k1"):
        TranslationWorkdir(tmp_path).translation()


def test_output_of_unknown_run_is_invalid_input(tmp_path: Path) -> None:
    """Výstup z běhu, který runs.jsonl nezná, nejde zapsat do provenance."""
    write_workdir(
        tmp_path,
        [request("n1", "k1", "one")],
        [run_line(FIRST_START, REQUESTS_SHA)],
        [output("k1", "jedna", LAST_START)],
    )

    with pytest.raises(InvalidInputError, match=LAST_START[:19]):
        TranslationWorkdir(tmp_path).translation()


@pytest.mark.parametrize("name", ["requests.json", "runs.jsonl", "outputs.jsonl"])
def test_missing_file_is_invalid_input(tmp_path: Path, name: str) -> None:
    """Chybějící soubor pracovního adresáře se přeloží na InvalidInputError."""
    workdir = _workdir(tmp_path)
    (workdir / name).unlink()

    with pytest.raises(InvalidInputError, match=re.escape(name)):
        TranslationWorkdir(workdir).translation()


def test_invalid_json_is_invalid_input(tmp_path: Path) -> None:
    """Soubor, který neodpovídá schématu, se přeloží na InvalidInputError."""
    workdir = _workdir(tmp_path)
    (workdir / "runs.jsonl").write_text("{}\n", encoding="utf-8")

    with pytest.raises(InvalidInputError, match=r"runs\.jsonl"):
        TranslationWorkdir(workdir).translation()


@pytest.mark.parametrize(
    ("name", "old", "new"),
    [
        ("requests.json", '"revision": "' + "d" * 40 + '"', '"revision": "abc"'),
        ("runs.jsonl", '"script_sha256": "' + "5" * 64 + '"', '"script_sha256": "xyz"'),
    ],
    ids=["short-revision", "non-hex-sha"],
)
def test_invalid_values_are_invalid_input(
    tmp_path: Path, name: str, old: str, new: str
) -> None:
    """Zkrácená revize a nehexový otisk se odmítnou jako chybný vstup, ne holá chyba pydantic."""
    workdir = _workdir(tmp_path)
    path = workdir / name
    content = path.read_text(encoding="utf-8")
    assert old in content
    path.write_text(content.replace(old, new), encoding="utf-8")

    with pytest.raises(InvalidInputError, match=re.escape(name)):
        TranslationWorkdir(workdir).translation()
