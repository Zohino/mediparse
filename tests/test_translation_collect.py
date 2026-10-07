"""Testy sběru překladů: skládání částí a hlášení problémů po zprávách."""

from __future__ import annotations

import hashlib
import json
import logging
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import pytest

from tests.support import REPOSITORY, load_script

if TYPE_CHECKING:
    from pathlib import Path
    from types import ModuleType

SCRIPT: Final = REPOSITORY / "translation" / "collect.py"
NOTE: Final = "n1"
OTHER: Final = "n2"


@pytest.fixture
def collect(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Načte skript collect.

    Returns:
        Načtený modul.
    """
    return load_script(SCRIPT, monkeypatch)


def _request(request_id: str, note_id: str, source: str) -> dict[str, str]:
    return {
        "id": request_id,
        "note_id": note_id,
        "key": f"key-{request_id}",
        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
    }


def _output(request_id: str, text: str, finish: str = "stop") -> dict[str, str]:
    return {"key": f"key-{request_id}", "text": text, "finish_reason": finish}


@dataclass(frozen=True)
class Case:
    """Originály, požadavky a výstupy jednoho běhu sběru."""

    sources: dict[str, str]
    requests: list[dict[str, str]]
    outputs: list[dict[str, str]]


def _run(
    collect: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    case: Case,
) -> tuple[int, Path]:
    source = tmp_path / "en"
    source.mkdir()
    for note_id, text in case.sources.items():
        (source / f"{note_id}.txt").write_text(text, encoding="utf-8")
    workdir = tmp_path / "work"
    workdir.mkdir()
    (workdir / "requests.json").write_text(json.dumps({"requests": case.requests}))
    (workdir / "outputs.jsonl").write_text(
        "".join(f"{json.dumps(output)}\n" for output in case.outputs)
    )
    target = tmp_path / "cs"
    argv = [
        "collect",
        "--workdir",
        str(workdir),
        "--source",
        str(source),
        "--target",
        str(target),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    return collect.main(), target


def _split_note() -> tuple[dict[str, str], list[dict[str, str]]]:
    source = "A\n\nB\n"
    requests = [
        _request(f"{NOTE}/1", NOTE, source),
        _request(f"{NOTE}/2", NOTE, source),
    ]
    return {NOTE: source}, requests


def test_parts_are_joined_in_request_order(
    collect: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Části se zapíšou spojené v pořadí požadavků, ne výstupů."""
    sources, requests = _split_note()
    outputs = [_output(f"{NOTE}/2", "druhá\n"), _output(f"{NOTE}/1", "první\n")]

    code, target = _run(
        collect, monkeypatch, tmp_path, Case(sources, requests, outputs)
    )

    assert code == 0
    assert (target / f"{NOTE}.txt").read_text() == "první\n\ndruhá\n"


def test_whole_note_is_written_as_before(
    collect: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Celá zpráva s jedním požadavkem se zapíše jako dosud."""
    requests = [_request(NOTE, NOTE, "A\n")]

    code, target = _run(
        collect,
        monkeypatch,
        tmp_path,
        Case({NOTE: "A\n"}, requests, [_output(NOTE, "x \n")]),
    )

    assert code == 0
    assert (target / f"{NOTE}.txt").read_text() == "x\n"


def test_truncated_part_flags_note_once(
    collect: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Uříznutá část vyřadí zprávu a ta je v hlášení jednou."""
    sources, requests = _split_note()
    outputs = [_output(f"{NOTE}/1", "a"), _output(f"{NOTE}/2", "b", "length")]

    with caplog.at_level(logging.INFO):
        code, target = _run(
            collect, monkeypatch, tmp_path, Case(sources, requests, outputs)
        )

    assert code == 1
    assert not (target / f"{NOTE}.txt").exists()
    assert f"Uříznutý překlad: {NOTE}\n" in f"{caplog.text}\n"


def test_missing_part_is_reported_once_per_note(
    collect: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Chybějící část se hlásí jednou za zprávu, i když chybí obě."""
    sources, requests = _split_note()

    with caplog.at_level(logging.INFO):
        code, _ = _run(collect, monkeypatch, tmp_path, Case(sources, requests, []))

    assert code == 1
    assert f"Chybí překlad: {NOTE}\n" in f"{caplog.text}\n"


def test_changed_source_is_reported_once_per_note(
    collect: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Změněný originál rozdělené zprávy se hlásí jednou."""
    _, requests = _split_note()
    outputs = [_output(f"{NOTE}/1", "a"), _output(f"{NOTE}/2", "b")]

    with caplog.at_level(logging.INFO):
        code, target = _run(
            collect, monkeypatch, tmp_path, Case({NOTE: "jiný\n"}, requests, outputs)
        )

    assert code == 1
    assert not (target / f"{NOTE}.txt").exists()
    assert f"Originál se od přípravy změnil: {NOTE}\n" in f"{caplog.text}\n"


def test_marker_mismatch_is_counted_over_joined_note(
    collect: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Počet značek se porovnává nad celou zprávou, ne nad částmi."""
    source = "A ___\n\nB ___\n"
    requests = [
        _request(f"{NOTE}/1", NOTE, source),
        _request(f"{NOTE}/2", NOTE, source),
    ]
    outputs = [_output(f"{NOTE}/1", "a ___"), _output(f"{NOTE}/2", "b ___")]

    with caplog.at_level(logging.INFO):
        code, _ = _run(
            collect, monkeypatch, tmp_path, Case({NOTE: source}, requests, outputs)
        )

    assert code == 0
    assert "Jiný počet" not in caplog.text


def test_written_count_counts_notes(
    collect: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Zapsáno X z Y počítá zprávy, ne požadavky."""
    sources, requests = _split_note()
    sources[OTHER] = "C\n"
    requests.append(_request(OTHER, OTHER, "C\n"))
    outputs = [_output(f"{NOTE}/1", "a"), _output(f"{NOTE}/2", "b")]

    with caplog.at_level(logging.INFO):
        _run(collect, monkeypatch, tmp_path, Case(sources, requests, outputs))

    assert "Zapsáno 1 z 2 překladů" in caplog.text
