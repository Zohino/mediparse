"""Sdílená stráž přístupu nad adresáři datových vstupů."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mediparse.domain.data_access import check_data_access
from mediparse.infrastructure.data_access_rules import load_data_access_rules
from mediparse.infrastructure.synthetic_marker import MARKER_NAME, read_marker

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping
    from pathlib import Path


def check_tables_access(environ: Mapping[str, str], tables: Iterable[Path]) -> None:
    """Ověří každý jedinečný adresář vstupů v pořadí jejich předání.

    Raises:
        DataAccessError: Neoznačená data v prostředí, které nesmí číst MIMIC.
        InvalidInputError: Značka syntetiky nebo pravidla přístupu jsou neplatná.
    """
    rules = load_data_access_rules()
    for directory in dict.fromkeys(table.parent for table in tables):
        check_data_access(
            environ,
            rules,
            synthetic=read_marker(directory) is not None,
            location=str(directory),
            marker=MARKER_NAME,
        )
