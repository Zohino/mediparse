"""Syntetické řádky tabulky ve tvaru MIMIC: buňky s prefixem a typové hodnoty po blocích."""

from __future__ import annotations

import math
from string import ascii_lowercase
from typing import TYPE_CHECKING, Final

from mediparse.domain.cohort import MIN_CELL
from mediparse.domain.diagnosis import Diagnosis
from mediparse.domain.icd import IcdVersion
from mediparse.domain.inputs import InvalidInputError

if TYPE_CHECKING:
    from collections.abc import Callable

    from mediparse.domain.icd import DiagnosisCodes
    from mediparse.domain.mimic_table import TableShape

TEXT_COLUMN: Final = "text"
RARE_ROWS: Final = 2
BLOCKS: Final = 4
DEFAULT_ROWS: Final = BLOCKS * MIN_CELL + RARE_ROWS

# Sloupce sešitů 2 a 5-7 přetypovávají nebo porovnávají hodnoty (TRY_CAST, CASE, icd_version = '9'),
# takže buňka s prefixem by všude vyšla jako NULL nebo "jiné" a větve výpočtu by zůstaly
# nevyzkoušené. Typový sloupec proto nese slovník MIMIC. Řádek patří do bloku
# (řádek // MIN_CELL) % 4: bloky 0 a 1 jsou ICD-9 s obdobím 2008 - 2010, bloky 2 a 3 ICD-10
# s obdobím 2017 - 2019, bloky 0 a 2 nesou diagnózu A, bloky 1 a 3 diagnózu B. Každá kategorie
# tak má n = MIN_CELL (a dvojice skupin po MIN_CELL), takže prevalence a demografie projdou
# potlačením a výstup je k porovnání před přepisem a po něm.
_BLOCK_DIAGNOSES: Final = (Diagnosis.DIABETES, Diagnosis.COPD)
_BLOCK_VERSIONS: Final = (
    IcdVersion.NINE,
    IcdVersion.NINE,
    IcdVersion.TEN,
    IcdVersion.TEN,
)
_BLOCK_DIAGNOSIS_INDEX: Final = (0, 1, 0, 1)
_BLOCK_VALUES: Final = {
    ("patients", "gender"): ("F", "M", "F", "M"),
    ("patients", "anchor_year_group"): ("2008 - 2010",) * 2 + ("2017 - 2019",) * 2,
    ("admissions", "insurance"): ("Medicare", "Private", "Medicare", "Private"),
    ("admissions", "marital_status"): ("MARRIED", "SINGLE", "MARRIED", "SINGLE"),
    ("admissions", "race"): ("WHITE", "BLACK/AFRICAN AMERICAN") * 2,
    ("admissions", "language"): ("English", "Spanish", "English", "Spanish"),
    ("diagnoses_icd", "icd_version"): ("9", "9", "10", "10"),
}
# Vzácné řádky (po blocích) nesou buňku s prefixem jen v těchto sloupcích: kategorie pod
# MIN_CELL, kterou potlačení musí sloučit, a když selže, kanárek ji najde. Jinde mají hodnoty
# bloku 0, protože třetí demografická kategorie s n = 2 by spustila doplňkové potlačení
# (dua_hidden) a zmizelo by pohlaví. Platná verze s kódem s prefixem by kód zařadila do
# tříznakové skupiny, kde se prefix ztratí, proto vzácný řádek nemá platnou verzi ani kód.
_RARE_WITNESSES: Final = frozenset({
    ("patients", "anchor_year_group"),
    ("diagnoses_icd", "icd_version"),
    ("diagnoses_icd", "icd_code"),
})
# Věk při přijetí je pak 20 + řádek % 40, takže se výpočet anchor_age + rok přijetí - anchor_year
# opravdu provede a desetiletá pásma sedí na bloky.
_ROW_VALUES: Final[dict[tuple[str, str], Callable[[int], str]]] = {
    ("patients", "anchor_age"): lambda row: str(15 + row % (BLOCKS * MIN_CELL)),
    ("patients", "anchor_year"): lambda _: "2150",
    ("admissions", "admittime"): lambda _: "2155-01-01 00:00:00",
}

# Klíč patří tabulce, která entitu v MIMIC zakládá. Řádek r každé tabulky tak patří pacientovi
# a hospitalizaci r a spojení přes klíč dá neprázdný vstupní korpus. Klíč není kategorie
# (v tabulce je jedinečný), takže každá kategorie má dál n = 1 a buňka klíče nese prefix
# s pomlčkou, který kanárek najde.
_KEY_OWNERS: Final = {"subject_id": "patients", "hadm_id": "admissions"}
# note_id je v každé tabulce zpráv vlastní (propouštěcí a radiologické se v MIMIC nepotkají),
# jen MIMIC-IV-Ext-BHC odkazuje na propouštěcí zprávu.
_REFERENCES: Final = {("mimic-iv-bhc", "note_id"): "discharge"}


# Tabulka, tvar, prefix, počet řádků a kódy tvoří jedno volání; objekt by jen přesunul arity.
def synthetic_rows(  # ruff: ignore[too-many-arguments]
    table: str,
    shape: TableShape,
    prefix: str,
    rows: int,
    *,
    text_chars: int,
    codes: DiagnosisCodes,
) -> list[dict[str, str]]:
    """Vytvoří řádky se sloupci v pořadí tabulky a buňkami ``prefix-vlastník-sloupec-řádek``.

    Vlastníkem je tabulka sama, u klíčů pacienta a hospitalizace tabulka, která je zakládá,
    a u ``note_id`` z MIMIC-IV-Ext-BHC tabulka ``discharge``. Mimo typové sloupce je každá
    buňka v tabulce jedinečná, takže každá kategorie má n = 1 a únik hodnoty do výstupu je
    nalezitelný hledáním prefixu. Typové sloupce nesou slovník MIMIC po blocích
    (``_typed_value``). Sloupec ``text`` za buňkou opakuje token zprávy, dokud nemá aspoň
    ``text_chars`` znaků (viz ``_text_token``).

    Args:
        table: Jméno tabulky bez přípony.
        shape: Sloupce tabulky v pořadí souboru.
        prefix: Řetězec, kterým začíná každá buňka.
        rows: Počet řádků.
        text_chars: Nejmenší délka textu, aby zpráva prošla filtrem délky vstupního korpusu.
        codes: Předpony pětice; kódy bloků se berou z nich.

    Returns:
        Řádky jako slovníky sloupec na buňku.

    Raises:
        InvalidInputError: Prefix je prázdný, počet řádků není kladný, tabulka nemá sloupce
            nebo diagnóza bloku nemá předponu v některé verzi ICD.
    """
    if not prefix:
        msg = "Prefix syntetických buněk nesmí být prázdný."
        raise InvalidInputError(msg)
    if rows < 1:
        msg = (
            f"Syntetická tabulka {table} potřebuje aspoň jeden řádek, ne {rows} řádků."
        )
        raise InvalidInputError(msg)
    if not shape.columns:
        msg = f"Tabulka {table} nemá žádné sloupce."
        raise InvalidInputError(msg)
    for block in range(BLOCKS):
        _block_code(codes, block)
    return [
        {
            column: _typed_value(table, column, row, codes)
            or _cell(table, column, prefix, row, text_chars)
            for column in shape.columns
        }
        for row in range(rows)
    ]


def _block_code(codes: DiagnosisCodes, block: int) -> str:
    diagnosis = _BLOCK_DIAGNOSES[_BLOCK_DIAGNOSIS_INDEX[block]]
    version = _BLOCK_VERSIONS[block]
    prefixes = codes.prefixes[diagnosis].get(version, ())
    if not prefixes:
        msg = f"Diagnóza {diagnosis.value} nemá předponu ICD-{version.value}."
        raise InvalidInputError(msg)
    # Kód je první předpona z configu, ne skutečný kód: znalost kódů zůstává jen v configu.
    # Charlson (comorbidity) pak označí 491 a J41 jako cpd, ale 249 a E08 jako diab ne.
    return prefixes[0]


def _typed_value(
    table: str, column: str, row: int, codes: DiagnosisCodes
) -> str | None:
    """Vrátí typovou hodnotu buňky, nebo None, když platí buňka s prefixem.

    Returns:
        Hodnota ve slovníku MIMIC nebo None pro sloupec bez typové hodnoty.
    """
    key = (table, column)
    if table == "d_icd_diagnoses":
        return _dictionary_value(column, row, codes)
    if key in _ROW_VALUES:
        return _ROW_VALUES[key](row)
    if row >= BLOCKS * MIN_CELL and key in _RARE_WITNESSES:
        return None
    block = (row // MIN_CELL) % BLOCKS
    if key == ("diagnoses_icd", "icd_code"):
        return _block_code(codes, block)
    values = _BLOCK_VALUES.get(key)
    return values[block] if values else None


def _dictionary_value(column: str, row: int, codes: DiagnosisCodes) -> str | None:
    # Číselník kódů pětice je veřejná konstanta: jeho řádky 0-3 jsou čtyři kódy bloků.
    if row >= BLOCKS:
        return None
    version = _BLOCK_VERSIONS[row]
    diagnosis = _BLOCK_DIAGNOSES[_BLOCK_DIAGNOSIS_INDEX[row]]
    values = {
        "icd_code": _block_code(codes, row),
        "icd_version": str(version.value),
        "long_title": f"{diagnosis.value} ICD-{version.value}",
    }
    return values.get(column)


def _text_token(prefix: str, row: int) -> str:
    """Vrátí token zprávy: písmena prefixu a číslo řádku zapsané písmeny.

    Tokenizace zahodí pomlčky a n-gramy EDA berou jen písmena (``[a-z]{2,}``), takže buňku
    s pomlčkami a číslem řádku ve výstupech po tokenech (n-gramy, slovník) kanárek nenajde.
    Token ze samých písmen projde beze změny a v každé zprávě je jiný (DF = 1); s číslicí by
    se v n-gramech zkrátil na kmen, který mají všechny zprávy. Výstup, ve kterém se token
    objeví, proto pustil token jediné zprávy.

    Returns:
        Token ze samých písmen.
    """
    stem = "".join(character for character in prefix if character.isalpha())
    return stem + _letters(row)


def _cell(table: str, column: str, prefix: str, row: int, text_chars: int) -> str:
    owner = _REFERENCES.get((table, column)) or _KEY_OWNERS.get(column, table)
    cell = f"{prefix}-{owner}-{column}-{row}"
    if column != TEXT_COLUMN:
        return cell
    token = _text_token(prefix, row)
    copies = max(1, math.ceil((text_chars - len(cell)) / (len(token) + 1)))
    return " ".join([cell, *[token] * copies])


def _letters(number: int) -> str:
    """Zapíše nezáporné číslo písmeny jako číslice soustavy o 26 znacích.

    Returns:
        Číslo zapsané písmeny, například 0 → a, 25 → z, 26 → ba.
    """
    digits = ""
    while True:
        number, digit = divmod(number, len(ascii_lowercase))
        digits = ascii_lowercase[digit] + digits
        if number == 0:
            return digits
