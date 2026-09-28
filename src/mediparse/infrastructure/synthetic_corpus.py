"""Syntetický korpus na disku: zprávy, jejich otisk a záznam auditu."""

from __future__ import annotations

from pathlib import Path
from typing import Final

from mediparse.domain.corpus_audit import AuditRecord, fingerprint

CORPUS_ROOT: Final = Path("resources/synthetic")
RECORD_NAME: Final = "audit.json"
_NOTES: Final = "*/*.txt"


def note_paths(root: Path) -> list[Path]:
    """Soubory zpráv korpusu ve tvaru ``<jazyk>/<note_id>.txt``.

    Returns:
        Cesty seřazené podle jména; prázdný seznam, když korpus neexistuje.
    """
    return sorted(root.glob(_NOTES))


def read_notes(root: Path) -> dict[str, str]:
    """Texty zpráv klíčované cestou relativní ke kořeni korpusu.

    Returns:
        Slovník relativní cesta → text zprávy.
    """
    return {
        _relative(path, root): path.read_text(encoding="utf-8")
        for path in note_paths(root)
    }


def corpus_sha256(root: Path) -> str | None:
    """Otisk zpráv korpusu tak, jak leží na disku.

    Returns:
        SHA-256 otisk, nebo None, když korpus neobsahuje žádnou zprávu.
    """
    paths = note_paths(root)
    if not paths:
        return None
    return fingerprint((_relative(path, root), path.read_bytes()) for path in paths)


def load_record(root: Path) -> AuditRecord | None:
    """Záznam posledního auditu; obsah, který neodpovídá schématu, vyhodí ValidationError.

    Returns:
        Záznam auditu, nebo None, když chybí.
    """
    path = root / RECORD_NAME
    if not path.exists():
        return None
    return AuditRecord.model_validate_json(path.read_text(encoding="utf-8"))


def save_record(root: Path, record: AuditRecord) -> None:
    """Zapíše záznam auditu do kořene korpusu."""
    (root / RECORD_NAME).write_text(
        f"{record.model_dump_json(indent=2)}\n", encoding="utf-8"
    )


def _relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()
