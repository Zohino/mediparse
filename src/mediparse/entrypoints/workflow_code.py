"""Otisky kódu pravidel pro parsování workflow Snakemake."""

from __future__ import annotations

from pathlib import Path

import mediparse
from mediparse.infrastructure.code_fingerprint import code_fingerprint as fingerprint


def code_fingerprint(script: str) -> str:
    """Doplní umístění instalovaného balíčku.

    Returns:
        Hexadecimální SHA-256 kódu skriptu a jeho závislostí v balíčku.
    """
    return fingerprint(Path(script), Path(mediparse.__file__).parent)
