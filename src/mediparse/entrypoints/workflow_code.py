"""Otisky kódu a vstupů pravidel pro parsování workflow Snakemake."""

from __future__ import annotations

from pathlib import Path

import mediparse
from mediparse.infrastructure.code_fingerprint import code_fingerprint as fingerprint
from mediparse.infrastructure.file_digest import file_sha256


def code_fingerprint(script: str) -> str:
    """Doplní umístění instalovaného balíčku.

    Returns:
        Hexadecimální SHA-256 kódu skriptu a jeho závislostí v balíčku.
    """
    return fingerprint(Path(script), Path(mediparse.__file__).parent)


def file_fingerprint(path: str) -> str:
    """Otiskne obsah vstupu mimo závislosti DAG.

    Returns:
        Hexadecimální SHA-256 obsahu souboru.

    Raises:
        InvalidInputError: Soubor chybí nebo jej nelze přečíst.
    """
    return file_sha256(Path(path))
