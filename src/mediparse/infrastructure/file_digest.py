"""Otisk souboru na disku, srovnatelný s výstupem ``sha256sum``."""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

from mediparse.domain.inputs import InvalidInputError

if TYPE_CHECKING:
    from pathlib import Path


def file_sha256(path: Path) -> str:
    """SHA-256 souboru čteného proudově, bez načtení celého souboru do paměti.

    Returns:
        Hexadecimální otisk.

    Raises:
        InvalidInputError: Vstupní soubor nelze přečíst.
    """
    try:
        with path.open("rb") as stream:
            return hashlib.file_digest(stream, "sha256").hexdigest()
    except OSError as error:
        msg = f"Soubor {path} nelze přečíst pro výpočet SHA-256."
        raise InvalidInputError(msg) from error
