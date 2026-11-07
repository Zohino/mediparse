"""Zápis a ověření vazby CSV → parquet v souboru vedle převedené tabulky."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mediparse.domain.inputs import InvalidInputError
from mediparse.domain.mimic_table import TableProvenance
from mediparse.infrastructure.file_digest import file_sha256
from mediparse.infrastructure.input_file import parse_file
from mediparse.infrastructure.json_file import write_model

if TYPE_CHECKING:
    from pathlib import Path

    from mediparse.domain.digest import Sha256


def provenance_path(parquet: Path) -> Path:
    """Odvodí cestu vazby vedle parquetu.

    Returns:
        Cesta ``<stem>.source.json`` ve stejném adresáři.
    """
    return parquet.with_suffix(".source.json")


def write_table_provenance(
    parquet: Path, *, source: str, source_sha256: Sha256
) -> TableProvenance:
    """Spočte otisk zapsaného parquetu a uloží ho spolu s původem CSV.

    Returns:
        Zapsaná vazba zdroje a parquetu.
    """
    provenance = TableProvenance(
        source=source, source_sha256=source_sha256, sha256=file_sha256(parquet)
    )
    write_model(provenance_path(parquet), provenance)
    return provenance


def verified_table_provenance(parquet: Path) -> TableProvenance:
    """Přečte vazbu a porovná ji s nově spočteným otiskem parquetu.

    Returns:
        Ověřená vazba zdroje a skutečně čteného parquetu.

    Raises:
        InvalidInputError: Vazba chybí, není platná nebo neodpovídá parquetu.
    """
    msg = f"Tabulku {parquet} nelze ověřit; převeďte tabulku znovu."
    try:
        provenance = parse_file(
            provenance_path(parquet), TableProvenance.model_validate_json
        )
        actual = file_sha256(parquet)
    except (OSError, InvalidInputError) as error:
        raise InvalidInputError(msg) from error
    if actual != provenance.sha256:
        raise InvalidInputError(msg)
    return provenance
