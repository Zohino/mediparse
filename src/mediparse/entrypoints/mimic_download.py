"""Vstupní bod stahování MIMIC pro Snakemake: tabulky z configu po validaci mediparse."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mediparse.domain.mimic_table import table_name
from mediparse.infrastructure.mimic_tables import TABLES_PATH, load_mimic_tables

if TYPE_CHECKING:
    from mediparse.infrastructure.mimic_tables import MimicTable


def mimic_tables() -> dict[str, MimicTable]:
    """Tabulky ke stažení, které pravidla Snakemake rozbalí na cíle.

    Returns:
        Slovník jméno souboru → tabulka s adresou a otiskem.
    """
    return load_mimic_tables(TABLES_PATH)


def mimic_table_files() -> dict[str, str]:
    """Stažené soubory pod jménem tabulky, kterým pravidla převodu pojmenují parquet.

    Returns:
        Slovník jméno tabulky → jméno souboru, v pořadí configu.
    """
    return {table_name(file): file for file in mimic_tables()}
