"""Parita skriptů translation/ s doménou: tytéž vstupy projdou skriptem i doménou stejně.

Skripty jsou samostatné soubory PEP 723 a balíček neimportují, takže znalost, kterou
sdílejí s doménou, drží pohromadě jen tyto testy.
"""

from __future__ import annotations

import json
import sys
import typing
from types import ModuleType
from typing import TYPE_CHECKING, Final

import pytest

from mediparse.domain.corpus_provenance import Masking
from mediparse.domain.note_text import deid_markers
from mediparse.domain.translation_consistency import (
    binding_violations,
    restored_markers,
)
from mediparse.infrastructure.translation_workdir import TranslationWorkdir
from tests.support import REPOSITORY, load_script
from tests.workdir_support import (
    LAST_START,
    collect_argv,
    output,
    request,
    run_line,
    write_single_part_workdir,
    write_workdir,
)

if TYPE_CHECKING:
    from pathlib import Path

TRANSLATION: Final = REPOSITORY / "translation"
LINE_BREAKS: Final = {"U+2028": "\u2028", "U+0085": "\x85"}


@pytest.fixture
def collect(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Načte skript collect.

    Returns:
        Načtený modul.
    """
    return load_script(TRANSLATION / "collect.py", monkeypatch)


@pytest.fixture
def translate(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Načte translate.py s náhradními moduly vllm a torch, které `_read` nepoužívá.

    Returns:
        Načtený modul.
    """
    vllm = ModuleType("vllm")
    inputs = ModuleType("vllm.inputs")
    vllm.LLM = vllm.SamplingParams = object  # ty: ignore[unresolved-attribute]
    inputs.TokensPrompt = dict  # ty: ignore[unresolved-attribute]
    for name, module in (
        ("vllm", vllm),
        ("vllm.inputs", inputs),
        ("torch", ModuleType("torch")),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    return load_script(TRANSLATION / "translate.py", monkeypatch)


def _write_notes(
    tmp_path: Path, notes: dict[str, tuple[str, str]]
) -> tuple[Path, Path]:
    """Pracovní adresář s výstupy s ensure_ascii=False a prázdným řádkem mezi záznamy.

    Args:
        tmp_path: Kořen testu.
        notes: note_id → dvojice anglický originál a text výstupu překladu.

    Returns:
        Dvojice (pracovní adresář, adresář originálů).
    """
    english = tmp_path / "en"
    english.mkdir()
    for note_id, (source, _) in notes.items():
        (english / f"{note_id}.txt").write_text(source, encoding="utf-8")
    workdir = tmp_path / "work"
    write_single_part_workdir(workdir, notes)
    lines = (
        json.dumps(output(f"key-{note_id}", text, LAST_START), ensure_ascii=False)
        for note_id, (_, text) in notes.items()
    )
    (workdir / "outputs.jsonl").write_text("\n\n".join(lines) + "\n", encoding="utf-8")
    return workdir, english


def _read_outputs(translate: ModuleType, path: Path) -> list[dict[str, str]]:
    # Čtení výstupů je soukromá funkce skriptu a právě ta je předmětem parity.
    return translate._read(path)  # ruff: ignore[private-member-access]


def _collect(
    collect: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    workdir: Path,
    english: Path,
) -> tuple[int, Path]:
    target = workdir.parent / "cs"
    monkeypatch.setattr(sys, "argv", collect_argv(workdir, english, target))
    return collect.main(), target


@pytest.mark.parametrize("separator", LINE_BREAKS.values(), ids=LINE_BREAKS.keys())
def test_json_lines_with_unicode_line_breaks_are_read_alike(
    translate: ModuleType,
    collect: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    separator: str,
) -> None:
    """Znak U+2028 či U+0085 uvnitř textu výstupu a prázdný řádek čtení nerozlomí."""
    texts = {"n1": f"první{separator}řádek\n", "n2": "druhá\n"}
    notes = {note_id: ("A\n", text) for note_id, text in texts.items()}
    workdir, english = _write_notes(tmp_path, notes)

    read = _read_outputs(translate, workdir / "outputs.jsonl")
    code, target = _collect(collect, monkeypatch, workdir, english)
    domain = TranslationWorkdir(workdir).notes()

    assert [record["text"] for record in read] == list(texts.values())
    assert code == 0
    assert sorted(path.name for path in target.iterdir()) == ["n1.txt", "n2.txt"]
    assert [note.parts for note in domain] == [(text,) for text in texts.values()]
    assert (
        binding_violations(
            domain,
            dict.fromkeys(texts, "A\n"),
            {
                note_id: (target / f"{note_id}.txt").read_text(encoding="utf-8")
                for note_id in texts
            },
        )
        == ()
    )


def test_translate_reads_missing_outputs_as_empty(
    translate: ModuleType, tmp_path: Path
) -> None:
    """Bez výstupů z dřívějšího běhu nezbývá nic přeskočit."""
    assert _read_outputs(translate, tmp_path / "outputs.jsonl") == []


def test_translate_read_keeps_record_order(
    translate: ModuleType, tmp_path: Path
) -> None:
    """Záznamy se vrací v pořadí řádků, prázdné a mezerové řádky se přeskočí."""
    path = tmp_path / "outputs.jsonl"
    path.write_text('{"key": "b"}\n\n   \n{"key": "a"}\n', encoding="utf-8")

    assert [record["key"] for record in _read_outputs(translate, path)] == ["b", "a"]


@pytest.fixture
def markers(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Načte skript markers.

    Returns:
        Načtený modul.
    """
    return load_script(TRANSLATION / "markers.py", monkeypatch)


MARKER_TEXTS: Final = [
    "___",
    "______",
    "____ year old",
    "__",
    "x___y",
    "a ___ b ___",
    "[[1]]",
    "[[ 2 ]]",
    "[[10]]",
    "[[x]]",
    "[[1]] a [[2]] ___",
    "",
]


@pytest.mark.parametrize("text", MARKER_TEXTS)
def test_marker_count_matches_domain(markers: ModuleType, text: str) -> None:
    """Skript a doména poznají značku ___ ve stejných textech."""
    assert len(markers.MARKER.findall(text)) == deid_markers(text)


@pytest.mark.parametrize("text", MARKER_TEXTS)
def test_restore_matches_domain(markers: ModuleType, text: str) -> None:
    """Obnova [[n]] na ___ dopadne ve skriptu i v doméně stejně."""
    assert markers.restore(text) == restored_markers(text)


def test_masking_placeholder_is_the_only_domain_masking(markers: ModuleType) -> None:
    """Hodnota, kterou skript zapisuje do plánu, je jediná, kterou provenance zná."""
    assert typing.get_args(Masking.__value__) == (markers.PLACEHOLDER,)


SPLIT_SOURCE: Final = "A ___\n\nB ___\n"


@pytest.mark.parametrize("masked", [True, False], ids=["masked", "plain"])
def test_collected_split_note_is_accepted_by_provenance(
    collect: ModuleType,
    markers: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    masked: bool,
) -> None:
    """Rozdělená zpráva s výstupy mimo pořadí a mezerami se složí, jak ji čeká doména."""
    first, second = ("první [[1]]  \nvnitřní řádek  \n", "  druhá [[2]]\n\n")
    if not masked:
        first, second = markers.restore(first), markers.restore(second)
    english = tmp_path / "en"
    english.mkdir()
    (english / "n1.txt").write_text(SPLIT_SOURCE, encoding="utf-8")
    workdir = tmp_path / "work"
    write_workdir(
        workdir,
        [request("n1", "n1/1", SPLIT_SOURCE), request("n1", "n1/2", SPLIT_SOURCE)],
        [run_line(LAST_START, "0" * 64)],
        [output("n1/2", second, LAST_START), output("n1/1", first, LAST_START)],
        markers.PLACEHOLDER if masked else None,
    )

    code, target = _collect(collect, monkeypatch, workdir, english)

    adapter = TranslationWorkdir(workdir)
    assert code == 0
    assert (
        binding_violations(
            adapter.notes(),
            {"n1": SPLIT_SOURCE},
            {"n1": (target / "n1.txt").read_text(encoding="utf-8")},
            adapter.translation().masking,
        )
        == ()
    )
