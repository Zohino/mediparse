"""Otisky: SHA-256 souboru, SHA-1 commitu a otisk sady souborů."""

from __future__ import annotations

import hashlib
import re
from typing import TYPE_CHECKING, Annotated, Final

from pydantic import AfterValidator, Field

if TYPE_CHECKING:
    from collections.abc import Iterable

Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
_COMMIT: Final = re.compile(r"[0-9a-f]{40}")


def commit_sha(value: str) -> str:
    """Ověří, že hodnota je celý SHA-1 hash commitu.

    Returns:
        Hodnota beze změny.

    Raises:
        ValueError: Hodnota není 40 hexadecimálních znaků, například zkrácený hash.
    """
    if _COMMIT.fullmatch(value) is None:
        msg = "Commit musí být celý SHA-1 hash (40 hexadecimálních znaků), například z `git rev-parse HEAD`."
        raise ValueError(msg)
    return value


CommitSha = Annotated[str, AfterValidator(commit_sha)]


def fingerprint(files: Iterable[tuple[str, bytes]]) -> str:
    """Otisk dvojic jméno a obsah: SHA-256 seznamu ve tvaru výstupu ``sha256sum`` seřazeného podle jména.

    Returns:
        Hexadecimální SHA-256 otisk.
    """
    listing = "".join(
        f"{hashlib.sha256(content).hexdigest()}  {name}\n"
        for name, content in sorted(files)
    )
    return hashlib.sha256(listing.encode()).hexdigest()
