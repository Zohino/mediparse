"""Otisky kódu pravidel pro parsování workflow Snakemake."""

from __future__ import annotations

from functools import cache
from pathlib import Path

import mediparse
from mediparse.infrastructure.code_fingerprint import code_fingerprint as fingerprint


@cache
def code_fingerprint(script: str) -> str:
    """Doplní umístění instalovaného balíčku a sdílí otisk mezi děděnými pravidly.

    Returns:
        Hexadecimální SHA-256 kódu skriptu a jeho závislostí v balíčku.
    """
    return fingerprint(Path(script), Path(mediparse.__file__).parent)
