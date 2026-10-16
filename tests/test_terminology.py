"""Terminologie hranice MIMIC: „shoda“ je jen překryv auditu, kladný význam nese „soulad“."""

import re
from pathlib import Path

import mediparse

PACKAGE = Path(mediparse.__file__).parent
AUDIT_MODULES = frozenset({
    "domain/corpus_audit.py",
    "application/corpus_audit.py",
    "entrypoints/corpus_audit.py",
    "infrastructure/overlap_report.py",
})
# Tvary slov shoda, neshoda, shodný a shodovat; „shodit“ a „shodí“ s nimi nesouvisí.
OVERLAP_WORD = re.compile(r"(?<!\w)(?:ne)?shod(?![ií])\w*", re.IGNORECASE)


def _modules_outside_audit() -> list[Path]:
    return [
        path
        for path in sorted(PACKAGE.rglob("*.py"))
        if path.relative_to(PACKAGE).as_posix() not in AUDIT_MODULES
    ]


def test_audit_modules_exist() -> None:
    """Výjimka míří na skutečné moduly auditu, jinak by po přejmenování platila naprázdno."""
    assert all((PACKAGE / module).is_file() for module in AUDIT_MODULES)


def test_overlap_word_is_reserved_for_audit() -> None:
    """Mimo audit se „shoda“ nepoužívá, aby výstup kontrol nešel číst jako únik z MIMIC."""
    offenders = [
        f"{path.relative_to(PACKAGE)}: {match.group()}"
        for path in _modules_outside_audit()
        for match in OVERLAP_WORD.finditer(path.read_text(encoding="utf-8"))
    ]

    assert offenders == []
