"""Katalog korpusu a přehled úplnosti pro první sešit EDA."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from mediparse.domain.czech import format_number
from mediparse.domain.inputs import InvalidInputError
from mediparse.domain.thesis_table import ThesisTable

if TYPE_CHECKING:
    from collections.abc import Mapping

    from mediparse.domain.mimic_table import TableShape


@dataclass(frozen=True)
class CatalogEntry:
    """Tabulka, veřejný zdroj a její využití v práci."""

    table: str
    source: str
    description: str


CATALOG: Final[tuple[CatalogEntry, ...]] = (
    CatalogEntry(
        "discharge",
        "MIMIC-IV-Note (note)",
        "Propouštěcí zprávy; primární korpus práce.",
    ),
    CatalogEntry(
        "radiology", "MIMIC-IV-Note (note)", "Radiologické zprávy; doplňkový korpus."
    ),
    CatalogEntry(
        "diagnoses_icd",
        "MIMIC-IV (hosp)",
        "Diagnózy ICD přiřazené hospitalizacím; zdroj štítků klasifikace.",
    ),
    CatalogEntry(
        "d_icd_diagnoses", "MIMIC-IV (hosp)", "Číselník diagnóz ICD; názvy ke kódům."
    ),
    CatalogEntry(
        "procedures_icd",
        "MIMIC-IV (hosp)",
        "Výkony ICD přiřazené hospitalizacím; kontext zprávy.",
    ),
    CatalogEntry(
        "d_icd_procedures", "MIMIC-IV (hosp)", "Číselník výkonů ICD; názvy ke kódům."
    ),
    CatalogEntry(
        "patients",
        "MIMIC-IV (hosp)",
        "Pacienti: pohlaví, věk a rok ukotvení; podklad dělení dat po pacientech.",
    ),
    CatalogEntry(
        "admissions",
        "MIMIC-IV (hosp)",
        "Hospitalizace: čas přijetí a propuštění, typ přijetí; vazba zpráv na pacienty.",
    ),
    CatalogEntry(
        "services",
        "MIMIC-IV (hosp)",
        "Služby, na kterých pacient během hospitalizace ležel; kontext zprávy.",
    ),
    CatalogEntry("prescriptions", "MIMIC-IV (hosp)", "Předpisy léků; kontext zprávy."),
    CatalogEntry(
        "pharmacy", "MIMIC-IV (hosp)", "Lékárenské záznamy o lécích; kontext zprávy."
    ),
    CatalogEntry(
        "omr",
        "MIMIC-IV (hosp)",
        "Ambulantní měření (například výška, hmotnost, tlak); kontext zprávy.",
    ),
    CatalogEntry(
        "mimic-iv-bhc",
        "MIMIC-IV-Ext-BHC",
        "Brief Hospital Course vytažený regexem autorů datasetu; reference validace parseru sekcí.",
    ),
)


@dataclass(frozen=True)
class CorpusOverview:
    """Tabulka a neprázdné inline hodnoty přehledu korpusu."""

    table: ThesisTable
    tables: str
    discharge: str
    radiology: str
    records: str
    summary: str


def summarize_corpus(
    counts: Mapping[str, int],
    shapes: Mapping[str, TableShape],
    sizes: Mapping[str, int],
) -> CorpusOverview:
    """Složí přehled počtů databáze proti manifestu validace.

    Args:
        counts: Počty řádků pohledů databáze.
        shapes: Validované počty řádků a jména sloupců.
        sizes: Velikosti stažených souborů v bajtech.

    Returns:
        Rozvržená tabulka a texty pro sešit.

    Raises:
        InvalidInputError: Množiny tabulek si neodpovídají, tabulka je mimo katalog
            nebo chybí discharge či radiology.
    """
    _check_tables(set(counts), set(shapes), set(sizes))
    catalog = {entry.table: entry for entry in CATALOG}
    names = sorted(counts)
    differences = [name for name in names if counts[name] != shapes[name].records]
    table = ThesisTable(
        header=("Tabulka", "Zdroj", "Obsah a využití", "Záznamů", "Sloupců", "MB"),
        rows=tuple(
            (
                name,
                catalog[name].source,
                catalog[name].description,
                format_number(counts[name]),
                format_number(len(shapes[name].columns)),
                format_number(round(sizes[name] / 1_000_000)),
            )
            for name in names
        ),
        align="lllrrr",
        widths=(22, 16, 30, 10, 12, 10),
        code=frozenset({"Tabulka"}),
    )
    return CorpusOverview(
        table=table,
        tables=format_number(len(counts)),
        discharge=format_number(counts["discharge"]),
        radiology=format_number(counts["radiology"]),
        records=_records(counts, agrees=not differences),
        summary=_summary(differences),
    )


def _check_tables(database: set[str], shapes: set[str], sizes: set[str]) -> None:
    catalog = {entry.table for entry in CATALOG}
    _refuse_tables((database | shapes | sizes) - catalog, "Tabulky mimo katalog")
    _refuse_tables(
        (shapes | sizes) - database,
        "Tabulky v manifestu validace, ale ne v databázi",
    )
    _refuse_tables(
        database - (shapes & sizes),
        "Tabulky v databázi, ale ne v manifestu validace",
    )
    _refuse_tables(
        {"discharge", "radiology"} - database,
        "Povinné tabulky chybí v databázi",
    )


def _refuse_tables(names: set[str], source: str) -> None:
    if names:
        msg = f"{source}: {', '.join(sorted(names))}."
        raise InvalidInputError(msg)


def _records(counts: Mapping[str, int], *, agrees: bool) -> str:
    maximum = max(counts.values(), default=0)
    largest = [name for name, count in counts.items() if count == maximum]
    prefix = f"Nejvíce záznamů má tabulka {largest[0]}. " if len(largest) == 1 else ""
    verb = "odpovídají" if agrees else "neodpovídají"
    # Prázdný inline Markdown Quarto vypíše jako kód; věta o manifestu je vždy přítomná.
    return prefix + f"Počty záznamů v databázi {verb} počtům v manifestu validace."


def _summary(differences: list[str]) -> str:
    if differences:
        return (
            "Počty záznamů se liší od manifestu validace u tabulek: "
            f"{', '.join(differences)}. Data se proto nesmějí považovat za úplná "
            "a před dalším použitím je nutné rozdíl vysvětlit."
        )
    return (
        "Počty záznamů u všech tabulek odpovídají manifestu validace. "
        "Data jsou proto úplná a další části analýzy mohou stavět na parquetu "
        "bez další kontroly počtu záznamů."
    )
