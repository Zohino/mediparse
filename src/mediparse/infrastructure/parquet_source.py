"""Zápis a ověření vazby CSV → parquet v souboru vedle převedené tabulky."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import ValidationError

from mediparse.domain.inputs import InvalidInputError
from mediparse.domain.mimic_table import ParquetSource
from mediparse.infrastructure.file_digest import file_sha256
from mediparse.infrastructure.input_file import parse_file
from mediparse.infrastructure.json_file import write_model

if TYPE_CHECKING:
    from pathlib import Path


def source_path(parquet: Path) -> Path:
    """Odvodí cestu vazby vedle parquetu.

    Returns:
        Cesta ``<stem>.source.json`` ve stejném adresáři.
    """
    return parquet.with_suffix(".source.json")


def write_parquet_source(
    parquet: Path, *, source: str, source_sha256: str
) -> ParquetSource:
    """Spočte otisk zapsaného parquetu a uloží ho spolu s původem CSV.

    Returns:
        Zapsaná vazba zdroje a parquetu.

    Raises:
        InvalidInputError: Parquet nelze přečíst, původ není platný nebo vazbu nelze zapsat.
    """
    try:
        provenance = ParquetSource(
            source=source, source_sha256=source_sha256, sha256=file_sha256(parquet)
        )
        write_model(source_path(parquet), provenance)
    except (OSError, ValidationError) as error:
        msg = f"Vazbu zdroje tabulky {parquet} nelze zapsat."
        raise InvalidInputError(msg) from error
    return provenance


def verified_parquet_source(parquet: Path) -> ParquetSource:
    """Přečte vazbu a porovná ji s nově spočteným otiskem parquetu.

    Returns:
        Ověřená vazba zdroje a skutečně čteného parquetu.

    Raises:
        InvalidInputError: Vazba chybí, není platná nebo neodpovídá parquetu.
    """
    msg = f"Tabulku {parquet} nelze ověřit; převeďte tabulku znovu."
    try:
        provenance = parse_file(source_path(parquet), ParquetSource.model_validate_json)
        actual = file_sha256(parquet)
    except (OSError, InvalidInputError) as error:
        raise InvalidInputError(msg) from error
    if actual != provenance.sha256:
        raise InvalidInputError(msg)
    return provenance
