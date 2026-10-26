"""Syntetické řádky tabulky ve tvaru MIMIC: buňky s prefixem a typové hodnoty po blocích."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Final

from mediparse.domain.diagnosis import Diagnosis
from mediparse.domain.dua import MIN_CELL
from mediparse.domain.icd import IcdVersion
from mediparse.domain.inputs import InvalidInputError
from mediparse.domain.synthetic_discharge import (
    padded_text,
    require_bhc_layout,
    synthetic_discharge,
    text_token,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from mediparse.domain.icd import DiagnosisCodes
    from mediparse.domain.mimic_table import TableShape
    from mediparse.domain.section_headers import SectionLexicon

TEXT_COLUMN: Final = "text"

# Sloupce sešitů 2 a 5-7 přetypovávají nebo porovnávají hodnoty (TRY_CAST, CASE, icd_version = '9'),
# takže buňka s prefixem by všude vyšla jako NULL nebo "jiné" a větve výpočtu by zůstaly
# nevyzkoušené. Typový sloupec proto nese slovník MIMIC. Řádek patří do bloku
# (řádek // MIN_CELL) % BLOCKS a blok určuje verzi ICD a diagnózu. Každá hodnota bez prefixu
# tak sdílí aspoň MIN_CELL řádků (pravidlo kanárka platí pro rows >= BLOCKS * MIN_CELL),
# dvě skupiny po MIN_CELL v každé verzi projdou potlačením a prevalence diabetu je vidět.
# CHOPN a Charlsonova kategorie cpd mají nad syntetikou stejných 20 zpráv, rozdíl je pod
# MIN_CELL, takže je reconcile_charlson (eda/R/diagnoses.R) obě skryje; je to zamýšlená větev,
# na které se ověřuje parita migrace EDA.
_BLOCKS: Final = (
    (IcdVersion.NINE, Diagnosis.DIABETES),
    (IcdVersion.NINE, Diagnosis.COPD),
    (IcdVersion.TEN, Diagnosis.DIABETES),
    (IcdVersion.TEN, Diagnosis.COPD),
)
BLOCKS: Final = len(_BLOCKS)
RARE_ROWS: Final = 2
DEFAULT_ROWS: Final = BLOCKS * MIN_CELL + RARE_ROWS
_ERAS: Final = {IcdVersion.NINE: "2008 - 2010", IcdVersion.TEN: "2017 - 2019"}
_BLOCK_VALUES: Final = {
    ("patients", "gender"): ("F", "M", "F", "M"),
    ("admissions", "insurance"): ("Medicare", "Private", "Medicare", "Private"),
    ("admissions", "marital_status"): ("MARRIED", "SINGLE", "MARRIED", "SINGLE"),
    ("admissions", "race"): ("WHITE", "BLACK/AFRICAN AMERICAN") * 2,
    ("admissions", "language"): ("English", "Spanish", "English", "Spanish"),
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
# Věk při přijetí (anchor_age + rok přijetí - anchor_year) je 20/30/40/50 podle bloku, takže
# se výpočet opravdu provede a desetiletá pásma sedí na bloky.
_DERIVED: Final[dict[tuple[str, str], Callable[[int, SyntheticRun], str]]] = {
    ("patients", "anchor_age"): lambda block, _: str(15 + 10 * block),
    ("patients", "anchor_year"): lambda _block, _run: "2150",
    ("admissions", "admittime"): lambda _block, _run: "2155-01-01 00:00:00",
    ("patients", "anchor_year_group"): lambda block, _: _ERAS[_BLOCKS[block][0]],
    ("diagnoses_icd", "icd_version"): lambda block, _: str(_BLOCKS[block][0].value),
    ("diagnoses_icd", "icd_code"): lambda block, run: run.block_codes[block],
}

# Klíč patří tabulce, která entitu v MIMIC zakládá. Řádek r každé tabulky tak patří pacientovi
# a hospitalizaci r a spojení přes klíč dá neprázdný vstupní korpus. Klíč není kategorie
# (v tabulce je jedinečný), takže každá kategorie má dál n = 1 a buňka klíče nese prefix
# s pomlčkou, který kanárek najde.
_KEY_OWNERS: Final = {"subject_id": "patients", "hadm_id": "admissions"}
# note_id je v každé tabulce zpráv vlastní (propouštěcí a radiologické se v MIMIC nepotkají),
# jen MIMIC-IV-Ext-BHC odkazuje na propouštěcí zprávu.
_REFERENCES: Final = {("mimic-iv-bhc", "note_id"): "discharge"}
# Text zprávy a její target BHC vznikají spolu z jedné struktury částí (domain/synthetic_discharge).
_NOTE_COLUMNS: Final = frozenset({
    ("discharge", TEXT_COLUMN),
    ("mimic-iv-bhc", "target"),
})


@dataclass(frozen=True)
class SyntheticRun:
    """Parametry běhu generátoru, ověřené při vytvoření.

    Attributes:
        prefix: Řetězec, kterým začíná každá buňka s prefixem.
        rows: Počet řádků každé tabulky.
        text_chars: Nejmenší délka textu, aby zpráva prošla filtrem délky vstupního korpusu.
        codes: Předpony pětice; kódy bloků se berou z nich.
        lexicon: Lexikon sekcí, podle kterého se skládá text zprávy.
        block_codes: Kód každého bloku, první předpona diagnózy bloku v jeho verzi.
    """

    prefix: str
    rows: int
    text_chars: int
    codes: DiagnosisCodes
    lexicon: SectionLexicon
    block_codes: tuple[str, ...] = field(init=False)

    def __post_init__(self) -> None:
        """Ověří prefix, počet řádků a předpony diagnóz bloků.

        Raises:
            InvalidInputError: Prefix je prázdný, počet řádků není kladný, diagnóza
                bloku nemá předponu v jeho verzi ICD nebo lexikon nemá rozvržení kolem BHC.
        """
        if not self.prefix:
            msg = "Prefix syntetických buněk nesmí být prázdný."
            raise InvalidInputError(msg)
        if self.rows < 1:
            msg = (
                f"Syntetická tabulka potřebuje aspoň jeden řádek, ne {self.rows} řádků."
            )
            raise InvalidInputError(msg)
        require_bhc_layout(self.lexicon)
        object.__setattr__(
            self,
            "block_codes",
            tuple(
                self._block_code(version, diagnosis) for version, diagnosis in _BLOCKS
            ),
        )

    def _block_code(self, version: IcdVersion, diagnosis: Diagnosis) -> str:
        prefixes = self.codes.prefixes[diagnosis].get(version, ())
        if not prefixes:
            msg = f"Diagnóza {diagnosis.value} nemá předponu ICD-{version.value}."
            raise InvalidInputError(msg)
        # Kód je první předpona z configu, ne skutečný kód: znalost kódů zůstává jen v configu.
        # Charlson (comorbidity) pak označí 491 a J41 jako cpd, ale 249 a E08 jako diab ne.
        return prefixes[0]


def synthetic_rows(
    table: str, shape: TableShape, run: SyntheticRun
) -> list[dict[str, str]]:
    """Vytvoří řádky se sloupci v pořadí tabulky a buňkami ``prefix-vlastník-sloupec-řádek``.

    Vlastníkem je tabulka sama, u klíčů pacienta a hospitalizace tabulka, která je zakládá,
    a u ``note_id`` z MIMIC-IV-Ext-BHC tabulka ``discharge``. Mimo typové sloupce je každá
    buňka v tabulce jedinečná, takže každá kategorie má n = 1 a únik hodnoty do výstupu je
    nalezitelný hledáním prefixu. Typové sloupce nesou slovník MIMIC po blocích
    (``_typed_value``); hodnota bez prefixu v nich sdílí aspoň ``MIN_CELL`` řádků, když je
    řádků aspoň ``BLOCKS * MIN_CELL``. Sloupec ``text`` za buňkou opakuje token zprávy,
    dokud nemá aspoň ``run.text_chars`` znaků (viz ``text_token``). Text propouštěcí zprávy
    bloku a target BHC jdou ze ``synthetic_discharge``.

    Args:
        table: Jméno tabulky bez přípony.
        shape: Sloupce tabulky v pořadí souboru.
        run: Ověřené parametry běhu.

    Returns:
        Řádky jako slovníky sloupec na buňku.

    Raises:
        InvalidInputError: Tabulka nemá sloupce.
    """
    if not shape.columns:
        msg = f"Tabulka {table} nemá žádné sloupce."
        raise InvalidInputError(msg)
    return [
        {
            column: _typed_value(table, column, row, run)
            or _note_value(table, column, row, run)
            or _cell(table, column, run.prefix, row, run.text_chars)
            for column in shape.columns
        }
        for row in range(run.rows)
    ]


def _typed_value(table: str, column: str, row: int, run: SyntheticRun) -> str | None:
    """Vrátí typovou hodnotu buňky, nebo None, když platí buňka s prefixem.

    Returns:
        Hodnota ve slovníku MIMIC nebo None pro sloupec bez typové hodnoty.
    """
    key = (table, column)
    if table == "d_icd_diagnoses":
        return _dictionary_value(column, row, run)
    if row >= BLOCKS * MIN_CELL and key in _RARE_WITNESSES:
        return None
    block = (row // MIN_CELL) % BLOCKS
    if key in _DERIVED:
        return _DERIVED[key](block, run)
    values = _BLOCK_VALUES.get(key)
    return values[block] if values else None


def _note_value(table: str, column: str, row: int, run: SyntheticRun) -> str | None:
    """Vrátí text zprávy nebo target BHC řádku v bloku, jinak None (platí buňka s prefixem).

    Returns:
        Text zprávy, target BHC, nebo None pro jiný sloupec a pro vzácný řádek.
    """
    if (table, column) not in _NOTE_COLUMNS or row >= BLOCKS * MIN_CELL:
        return None
    note = synthetic_discharge(
        run.prefix, run.lexicon, row, row // MIN_CELL, run.text_chars
    )
    return note.text if column == TEXT_COLUMN else note.bhc_target


def _dictionary_value(column: str, row: int, run: SyntheticRun) -> str | None:
    # Číselník kódů pětice je veřejná konstanta: jeho řádky 0-3 jsou kódy bloků.
    if row >= BLOCKS:
        return None
    version, diagnosis = _BLOCKS[row]
    values = {
        "icd_code": run.block_codes[row],
        "icd_version": str(version.value),
        "long_title": f"{diagnosis.value} ICD-{version.value}",
    }
    return values.get(column)


def _cell(table: str, column: str, prefix: str, row: int, text_chars: int) -> str:
    owner = _REFERENCES.get((table, column)) or _KEY_OWNERS.get(column, table)
    cell = f"{prefix}-{owner}-{column}-{row}"
    if column != TEXT_COLUMN:
        return cell
    return padded_text(cell, text_token(prefix, row), text_chars)
