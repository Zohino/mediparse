"""Parita skriptů translation/ s doménou: tytéž vstupy projdou skriptem i doménou stejně.

Skripty jsou samostatné soubory PEP 723 a balíček neimportují, takže znalost, kterou
sdílejí s doménou, drží pohromadě jen tyto testy.
"""

from __future__ import annotations

import json
import sys
from types import ModuleType
from typing import TYPE_CHECKING, Final

import pytest

from mediparse.domain.translation_consistency import binding_violations
from mediparse.infrastructure.translation_workdir import TranslationWorkdir
from tests.support import REPOSITORY, load_script
from tests.workdir_support import (
    LAST_START,
    output,
    request,
    run_line,
    write_workdir,
)

if TYPE_CHECKING:
    from pathlib import Path

TRANSLATION: Final = REPOSITORY / "translation"
LINE_BREAKS: Final = {"U+2028": chr(0x2028), "U+0085": chr(0x85)}


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
    tmp_path: Path, notes: dict[str, tuple[str, str]], masking: str | None = None
) -> tuple[Path, Path]:
    """Pracovní adresář a anglické originály, výstupy jsou psané jako ``translate._write``.

    Args:
        tmp_path: Kořen testu.
        notes: note_id → dvojice anglický originál a text výstupu překladu.
        masking: Hodnota ``masking`` v plánu.

    Returns:
        Dvojice (pracovní adresář, adresář originálů).
    """
    english = tmp_path / "en"
    english.mkdir()
    for note_id, (source, _) in notes.items():
        (english / f"{note_id}.txt").write_text(source, encoding="utf-8")
    workdir = tmp_path / "work"
    write_workdir(
        workdir,
        [
            request(note_id, f"key-{note_id}", source)
            for note_id, (source, _) in notes.items()
        ],
        [run_line(LAST_START, "0" * 64)],
        [],
        masking,
    )
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
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "collect",
            "--workdir",
            str(workdir),
            "--source",
            str(english),
            "--target",
            str(target),
        ],
    )
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


def test_domain_accepts_collected_translation(
    collect: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Provenance přijme cs/, které zapsal sběr z výstupů s U+2028."""
    notes = {"n1": ("A\n", f"první{LINE_BREAKS['U+2028']}řádek\n")}
    workdir, english = _write_notes(tmp_path, notes)

    code, target = _collect(collect, monkeypatch, workdir, english)

    assert code == 0
    assert (
        binding_violations(
            TranslationWorkdir(workdir).notes(),
            {"n1": "A\n"},
            {"n1": (target / "n1.txt").read_text(encoding="utf-8")},
        )
        == ()
    )
