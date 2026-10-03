"""Otisk souboru na disku, srovnatelný s výstupem ``sha256sum``."""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


def file_sha256(path: Path) -> str:
    """SHA-256 souboru čteného proudově, bez načtení celého souboru do paměti.

    Returns:
        Hexadecimální otisk.
    """
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()
