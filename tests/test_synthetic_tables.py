"""Syntetické řádky tabulky ve tvaru MIMIC: buňky s prefixem, typové hodnoty po blocích a pravidlo kanárka."""

from __future__ import annotations

import re
from collections import Counter
from typing import Final

import pytest

from mediparse.domain.clinical_text import (
    find_diagnosis_acronyms,
    normalize_clinical_text,
)
from mediparse.domain.diagnosis import Diagnosis
from mediparse.domain.dua import MIN_CELL
from mediparse.domain.icd import DiagnosisCodes, IcdVersion
from mediparse.domain.inputs import InvalidInputError
from mediparse.domain.mimic_table import TableShape
from mediparse.domain.section_headers import SectionLexicon
from mediparse.domain.synthetic_discharge import (
    BHC_SECTION,
    synthetic_discharge,
)
from mediparse.domain.synthetic_mentions import (
    ACRONYM_SHARE_MIN_OCCURRENCES,
    MentionState,
    mention_lines,
)
from mediparse.domain.synthetic_tables import (
    BLOCKS,
    DEFAULT_ROWS,
    RARE_ROWS,
    SyntheticRun,
    synthetic_rows,
)
from mediparse.infrastructure.mimic_inventory import read_manifest
from tests.cohort_support import CODES, PREFIXES
from tests.mention_support import ACRONYMS, REPOSITORY_VOCABULARY, TEXT_MENTIONS
from tests.section_support import REPOSITORY_LEXICON
from tests.support import REPOSITORY_MIMIC_MANIFEST

ROWS: Final = 12
TEXT_CHARS: Final = 100
SHAPE = TableShape(records=5, columns=("note_id", "subject_id", "text"))
NOTE_SHAPE = TableShape(records=5, columns=("note_id", "subject_id", "hadm_id", "text"))
PREFIX: Final = "eda-sentinel"
FIRST_KEY: Final = next(iter(REPOSITORY_LEXICON.sections))
COLUMNS: Final = {
    table: shape.columns
    for table, shape in read_manifest(REPOSITORY_MIMIC_MANIFEST).items()
}
BLOCK_DIAGNOSES: Final = (
    Diagnosis.DIABETES,
    Diagnosis.COPD,
    Diagnosis.DIABETES,
    Diagnosis.COPD,
    Diagnosis.DIABETES,
    Diagnosis.COPD,
)
BLOCK_VERSIONS: Final = (
    IcdVersion.NINE,
    IcdVersion.NINE,
    IcdVersion.TEN,
    IcdVersion.TEN,
    IcdVersion.NINE,
    IcdVersion.TEN,
)
BLOCK_CODES: Final = tuple(
    PREFIXES[BLOCK_DIAGNOSES[block]][BLOCK_VERSIONS[block]][0]
    for block in range(len(BLOCK_VERSIONS))
)
CELLS_PER_DIAGNOSIS: Final = 6
# Nezávislé orákulum: ceil(1000 / (MIN_CELL * 2 tvrzené bloky)), ne přepočet vzorce z implementace.
ACRONYM_REPEATS: Final = 50
BLOCK_STATES: Final = (
    MentionState.ASSERTED,
    MentionState.ASSERTED,
    MentionState.NEGATED,
    MentionState.NEGATED,
    MentionState.ABSENT,
    MentionState.ABSENT,
)
# sloupec -> hodnoty bloků 0-5 (slovník MIMIC)
EXPECTED: Final = {
    ("patients", "gender"): ("F", "M") * 3,
    ("patients", "anchor_year_group"): tuple(
        "2008 - 2010" if version is IcdVersion.NINE else "2017 - 2019"
        for version in BLOCK_VERSIONS
    ),
    ("admissions", "insurance"): ("Medicare", "Private") * 3,
    ("admissions", "marital_status"): ("MARRIED", "SINGLE") * 3,
    ("admissions", "race"): ("WHITE", "BLACK/AFRICAN AMERICAN") * 3,
    ("admissions", "language"): ("English", "Spanish") * 3,
    ("diagnoses_icd", "icd_version"): tuple(
        str(version.value) for version in BLOCK_VERSIONS
    ),
    ("diagnoses_icd", "icd_code"): BLOCK_CODES,
    ("patients", "anchor_age"): tuple(str(15 + 10 * block) for block in range(BLOCKS)),
    ("patients", "anchor_year"): ("2150",) * 6,
    ("admissions", "admittime"): ("2155-01-01 00:00:00",) * 6,
}
WITNESSES: Final = {
    ("patients", "anchor_year_group"),
    ("diagnoses_icd", "icd_version"),
    ("diagnoses_icd", "icd_code"),
}
TYPED_TABLES: Final = ("patients", "admissions", "diagnoses_icd")


def test_typed_keys_exist_in_the_public_manifest() -> None:
    """Typové klíče (tabulka, sloupec) z testu existují ve veřejném manifestu, překlep nebo přejmenování test shodí."""
    keys = set(EXPECTED) | WITNESSES

    missing = {(t, c) for t, c in keys if c not in COLUMNS.get(t, ())}

    assert missing == set()


def _run(prefix: str, rows: int, codes: DiagnosisCodes = CODES) -> SyntheticRun:
    return SyntheticRun(
        prefix=prefix,
        rows=rows,
        text_chars=TEXT_CHARS,
        codes=codes,
        lexicon=REPOSITORY_LEXICON,
        mentions=REPOSITORY_VOCABULARY,
    )


def _table(table: str, rows: int = DEFAULT_ROWS) -> list[dict[str, str]]:
    shape = TableShape(records=rows, columns=COLUMNS[table])
    return synthetic_rows(table, shape, _run(PREFIX, rows))


def _column(table: str, column: str, rows: int = DEFAULT_ROWS) -> list[str]:
    return [row[column] for row in _table(table, rows)]


def test_rows_have_requested_count_and_columns_in_manifest_order() -> None:
    """Každý řádek má všechny sloupce v pořadí manifestu a řádků je tolik, kolik se žádá."""
    rows = synthetic_rows("discharge", SHAPE, _run("sentinel", ROWS))

    assert len(rows) == ROWS
    assert all(tuple(row) == SHAPE.columns for row in rows)


def test_cells_are_unique_within_table_and_carry_prefix() -> None:
    """Každá buňka tabulky je v ní jedinečná a obsahuje prefix, takže únik zachytí hledání prefixu."""
    rows = synthetic_rows("discharge", SHAPE, _run("sentinel", ROWS))
    cells = [cell for row in rows for cell in row.values()]

    assert len(set(cells)) == len(cells)
    assert all("sentinel" in cell for cell in cells)


def test_cells_name_table_column_and_row() -> None:
    """Buňka je ve tvaru prefix-tabulka-sloupec-řádek."""
    rows = synthetic_rows("radiology", NOTE_SHAPE, _run("sentinel", 2))

    assert rows[1]["note_id"] == "sentinel-radiology-note_id-1"


def test_rows_are_deterministic() -> None:
    """Stejné vstupy dávají stejné řádky."""
    assert synthetic_rows("t", SHAPE, _run("p", 3)) == synthetic_rows(
        "t", SHAPE, _run("p", 3)
    )


def test_zero_rows_are_refused() -> None:
    """Nulový počet řádků je chyba."""
    with pytest.raises(InvalidInputError, match="řádků"):
        synthetic_rows("t", SHAPE, _run("p", 0))


def test_table_without_columns_is_refused() -> None:
    """Tabulka bez sloupců je chyba a jmenuje tabulku."""
    with pytest.raises(InvalidInputError, match="empty"):
        synthetic_rows("empty", TableShape(records=0, columns=()), _run("p", 3))


def test_empty_prefix_is_refused() -> None:
    """Prázdný prefix je chyba, protože kanárek by nic nehledal."""
    with pytest.raises(InvalidInputError, match="Prefix"):
        synthetic_rows("t", SHAPE, _run("", ROWS))


def test_patient_and_admission_keys_are_shared_across_tables() -> None:
    """Řádek r každé tabulky patří pacientovi a hospitalizaci r, takže se tabulky spojí jako v MIMIC."""
    notes = synthetic_rows("discharge", NOTE_SHAPE, _run("p", ROWS))
    codes = synthetic_rows(
        "diagnoses_icd",
        TableShape(records=5, columns=("subject_id", "hadm_id", "icd_code")),
        _run("p", ROWS),
    )

    assert [row["hadm_id"] for row in notes] == [row["hadm_id"] for row in codes]
    assert [row["subject_id"] for row in notes] == [row["subject_id"] for row in codes]
    assert notes[3]["hadm_id"] == "p-admissions-hadm_id-3"
    assert notes[3]["subject_id"] == "p-patients-subject_id-3"


def test_bhc_note_id_refers_to_discharge_and_radiology_keeps_its_own() -> None:
    """BHC se páruje s propouštěcí zprávou přes note_id, radiologie má note_id vlastní jako v MIMIC."""
    shape = TableShape(records=5, columns=("note_id", "input"))

    bhc = synthetic_rows("mimic-iv-bhc", shape, _run("p", 2))
    radiology = synthetic_rows("radiology", NOTE_SHAPE, _run("p", 2))

    assert bhc[1]["note_id"] == "p-discharge-note_id-1"
    assert radiology[1]["note_id"] == "p-radiology-note_id-1"


def test_text_starts_with_its_cell_and_reaches_minimum_length() -> None:
    """Text začíná svou buňkou s tokenem na prvním řádku a má aspoň požadovaný počet znaků."""
    rows = synthetic_rows("discharge", NOTE_SHAPE, _run("p", DEFAULT_ROWS))

    for index, row in enumerate(rows):
        assert row["text"].splitlines()[0].startswith(f"p-discharge-text-{index} ")
        assert len(row["text"]) >= TEXT_CHARS


def test_text_token_is_letters_only_and_unique_per_row() -> None:
    """Token z písmen přežije tokenizaci a normalizaci čísel a v každé zprávě je jiný (DF = 1)."""
    rows = synthetic_rows("discharge", NOTE_SHAPE, _run("eda-sentinel", 30))

    tokens = [set(re.findall(r"edasentinel[a-z]+", row["text"])) for row in rows]

    assert all(len(found) == 1 for found in tokens)
    assert len({next(iter(found)) for found in tokens}) == len(tokens)


def test_block_texts_and_targets_come_from_the_discharge_module() -> None:
    """Text propouštěcí zprávy a target BHC bloků jdou ze ``synthetic_discharge``, jedna zpráva na řádek."""
    run = _run(PREFIX, DEFAULT_ROWS)
    texts = synthetic_rows("discharge", NOTE_SHAPE, run)
    targets = synthetic_rows(
        "mimic-iv-bhc", TableShape(records=5, columns=("note_id", "target")), run
    )

    for row in range(BLOCKS * MIN_CELL):
        head = texts[row]["text"].splitlines()[0]
        block = row // MIN_CELL
        lines = mention_lines(
            REPOSITORY_VOCABULARY,
            BLOCK_STATES[block],
            ACRONYM_REPEATS,
        )
        note = synthetic_discharge(
            head, head.split()[-1], REPOSITORY_LEXICON, block, lines
        )
        assert texts[row]["text"] == note.text
        assert targets[row]["target"] == note.bhc_target


@pytest.mark.parametrize("text_chars", [1, 2000])
def test_every_note_has_text_chars_and_first_line_is_the_cell_text(
    text_chars: int,
) -> None:
    """Délku ``text_chars`` splní první řádek (buňka a token) i s krátkým lexikonem, ne až sekce."""
    lexicon = SectionLexicon(
        sections={
            key: REPOSITORY_LEXICON.sections[key]
            for key in (
                FIRST_KEY,
                BHC_SECTION,
                "discharge_medications",
                "discharge_diagnosis",
            )
        },
        preamble=(),
    )
    run = SyntheticRun(
        prefix=PREFIX,
        rows=DEFAULT_ROWS,
        text_chars=text_chars,
        codes=CODES,
        lexicon=lexicon,
        mentions=REPOSITORY_VOCABULARY,
    )

    rows = synthetic_rows("discharge", NOTE_SHAPE, run)

    for index, row in enumerate(rows):
        first = row["text"].splitlines()[0]
        assert first.startswith(f"{PREFIX}-discharge-text-{index}")
        assert len(first) >= text_chars
        assert len(row["text"]) >= text_chars


def test_token_is_added_once_when_the_cell_is_already_long_enough() -> None:
    """Buňka delší než ``text_chars`` dostane token právě jednou."""
    run = SyntheticRun(
        prefix=PREFIX,
        rows=1,
        text_chars=1,
        codes=CODES,
        lexicon=REPOSITORY_LEXICON,
        mentions=REPOSITORY_VOCABULARY,
    )

    text = synthetic_rows("radiology", NOTE_SHAPE, run)[0]["text"]

    assert len(re.findall(r"edasentinel[a-z]+", text)) == 1


def test_rare_rows_keep_text_and_target_as_cells_without_headings() -> None:
    """Vzácné řádky zůstávají bez nadpisů, jejich target je buňka."""
    run = _run(PREFIX, DEFAULT_ROWS)
    texts = synthetic_rows("discharge", NOTE_SHAPE, run)
    targets = synthetic_rows(
        "mimic-iv-bhc", TableShape(records=5, columns=("note_id", "target")), run
    )

    for row in range(BLOCKS * MIN_CELL, DEFAULT_ROWS):
        assert texts[row]["text"].count("\n") == 0
        assert targets[row]["target"] == f"{PREFIX}-mimic-iv-bhc-target-{row}"


def test_radiology_text_is_unchanged() -> None:
    """Radiologie a ostatní textové sloupce zůstávají buňkou a tokeny."""
    rows = synthetic_rows("radiology", NOTE_SHAPE, _run("p", 3))

    assert rows[1]["text"].startswith("p-radiology-text-1 ")
    assert "\n" not in rows[1]["text"]


def test_lexicon_without_bhc_section_is_refused() -> None:
    """Lexikon bez sekce BHC je InvalidInputError s klíčem sekce."""
    sections = {
        key: section
        for key, section in REPOSITORY_LEXICON.sections.items()
        if key != BHC_SECTION
    }

    with pytest.raises(InvalidInputError, match=BHC_SECTION):
        SyntheticRun(
            prefix=PREFIX,
            rows=1,
            text_chars=TEXT_CHARS,
            codes=CODES,
            lexicon=SectionLexicon(
                sections=sections, preamble=REPOSITORY_LEXICON.preamble
            ),
            mentions=REPOSITORY_VOCABULARY,
        )


@pytest.mark.parametrize(("key", "values"), EXPECTED.items())
def test_typed_column_has_block_values_and_rare_rows_follow_rule(
    key: tuple[str, str], values: tuple[str, ...]
) -> None:
    """Typový sloupec nese hodnoty bloků (neprázdná množina), vzácné řádky jen ve svědcích prefix."""
    table, column = key
    cells = _column(table, column)

    for block in range(BLOCKS):
        assert (
            cells[block * MIN_CELL : (block + 1) * MIN_CELL]
            == [values[block]] * MIN_CELL
        )
    rare = cells[BLOCKS * MIN_CELL :]
    assert len(rare) == RARE_ROWS
    if key in WITNESSES:
        assert all(cell.startswith(PREFIX) for cell in rare)
    else:
        assert rare == [values[0]] * RARE_ROWS


@pytest.mark.parametrize("table", TYPED_TABLES)
def test_values_without_prefix_are_shared_by_min_cell_rows(table: str) -> None:
    """Pravidlo kanárka: kategorie bez prefixu má aspoň MIN_CELL výskytů, ostatní nesou prefix."""
    for column in COLUMNS[table]:
        cells = _column(table, column)
        bare = [cell for cell in cells if PREFIX not in cell]
        assert all(bare.count(cell) >= MIN_CELL for cell in set(bare)), column
        if (table, column) not in EXPECTED:
            assert not bare, column


def test_patient_age_at_admission_is_shared_within_block() -> None:
    """Věk při přijetí je 20-70 podle bloku a vzácné řádky mají věk bloku 0."""
    patients = _table("patients")
    admissions = _table("admissions")
    ages = [
        int(patient["anchor_age"])
        + int(admission["admittime"][:4])
        - int(patient["anchor_year"])
        for patient, admission in zip(patients, admissions, strict=True)
    ]

    expected = [20 + 10 * ((row // MIN_CELL) % BLOCKS) for row in range(DEFAULT_ROWS)]
    assert ages == expected
    assert [ages.count(age) for age in range(20, 80, 10)] == [
        MIN_CELL + RARE_ROWS,
        *[MIN_CELL] * (BLOCKS - 1),
    ]


def test_era_follows_version_of_same_row() -> None:
    """Období pacienta sleduje verzi ICD diagnózy téhož řádku."""
    eras = _column("patients", "anchor_year_group")
    versions = _column("diagnoses_icd", "icd_version")

    pairs = {
        (era, version)
        for era, version in zip(eras, versions, strict=True)
        if PREFIX not in era
    }
    assert pairs == {("2008 - 2010", "9"), ("2017 - 2019", "10")}


def test_dictionary_holds_block_codes_with_versions_and_titles() -> None:
    """Číselník: řádky 0-3 nesou čtyři různé kódy bloků s verzí a názvem bez prefixu, ostatní prefix."""
    rows = _table("d_icd_diagnoses")

    assert [row["icd_code"] for row in rows[:4]] == list(dict.fromkeys(BLOCK_CODES))
    assert [row["icd_version"] for row in rows[:4]] == ["9", "9", "10", "10"]
    assert [row["long_title"] for row in rows[:4]] == [
        "diabetes ICD-9",
        "copd ICD-9",
        "diabetes ICD-10",
        "copd ICD-10",
    ]
    assert all(PREFIX in cell for row in rows[4:] for cell in row.values())


def test_block_codes_are_first_prefixes_of_the_config() -> None:
    """Kód bloku je první předpona z configu, ne natvrdo zapsaný kód."""
    other = {
        diagnosis: {
            version: tuple(f"Z{index}" for index in range(2))
            if diagnosis is Diagnosis.DIABETES and version is IcdVersion.TEN
            else prefixes
            for version, prefixes in by_version.items()
        }
        for diagnosis, by_version in PREFIXES.items()
    }
    rows = synthetic_rows(
        "diagnoses_icd",
        TableShape(records=1, columns=("icd_code",)),
        _run(PREFIX, 3 * MIN_CELL, DiagnosisCodes(prefixes=other)),
    )

    assert rows[2 * MIN_CELL]["icd_code"] == "Z0"


def test_fewer_rows_give_partial_blocks() -> None:
    """Menší počet řádků dá bloky jen částečně a nespadne."""
    assert _column("patients", "gender", 12) == ["F"] * MIN_CELL + ["M"] * 2


def test_table_without_typed_columns_is_unchanged() -> None:
    """Tabulka bez typových sloupců se nemění, buňky jsou jedinečné s prefixem."""
    rows = synthetic_rows("discharge", SHAPE, _run(PREFIX, DEFAULT_ROWS))
    cells = [cell for row in rows for cell in row.values()]

    assert len(set(cells)) == len(cells)
    assert all(PREFIX in cell for cell in cells)


@pytest.mark.parametrize(
    ("diagnosis", "version"),
    [(Diagnosis.DIABETES, IcdVersion.NINE), (Diagnosis.COPD, IcdVersion.TEN)],
)
def test_block_diagnosis_without_prefix_is_refused(
    diagnosis: Diagnosis, version: IcdVersion
) -> None:
    """Diagnóza bloku bez předpony v některé verzi je InvalidInputError."""
    prefixes = {
        d: {v: p for v, p in by_version.items() if (d, v) != (diagnosis, version)}
        for d, by_version in PREFIXES.items()
    }

    with pytest.raises(
        InvalidInputError, match=rf"{diagnosis.value}.*ICD-{version.value}"
    ):
        _run(PREFIX, 1, DiagnosisCodes(prefixes=prefixes))


def _notes() -> list[str]:
    return _column("discharge", "text")


def _state(text: str, phrase: str) -> MentionState:
    lines = text.splitlines()
    if phrase in lines:
        return MentionState.ASSERTED
    if f"{REPOSITORY_VOCABULARY.cue} {phrase}" in lines:
        return MentionState.NEGATED
    return MentionState.ABSENT


def _mentioned_words(text: str) -> list[str]:
    """Fráze a zkratky slovníku, které text obsahuje jako celé slovo (bez ohledu na velikost písmen).

    Returns:
        Nalezené fráze a zkratky.
    """
    return [
        word
        for word in (*REPOSITORY_VOCABULARY.phrases, *REPOSITORY_VOCABULARY.acronyms)
        if re.search(rf"\b{re.escape(word)}\b", text, re.IGNORECASE)
    ]


def _has_code(code: str, diagnosis: Diagnosis) -> bool:
    return code in {prefixes[0] for prefixes in PREFIXES[diagnosis].values()}


def _first_phrase(diagnosis: Diagnosis) -> str:
    return {
        Diagnosis.DIABETES: "diabetes",
        Diagnosis.COPD: "chronic obstructive pulmonary disease",
    }[diagnosis]


@pytest.mark.parametrize("diagnosis", BLOCK_DIAGNOSES[:2])
def test_every_icd_by_mention_cell_of_a_block_diagnosis_has_min_cell_notes(
    diagnosis: Diagnosis,
) -> None:
    """Řádek souladu ICD x zmínka sešitu 8 má všech šest buněk (kód, bez kódu) x (tvrzená, negovaná, bez zmínky) aspoň MIN_CELL."""
    phrase = _first_phrase(diagnosis)
    cells = Counter(
        (_has_code(code, diagnosis), _state(text, phrase))
        for code, text in zip(
            _column("diagnoses_icd", "icd_code"), _notes(), strict=True
        )
    )

    assert len(cells) == CELLS_PER_DIAGNOSIS
    assert min(cells.values()) >= MIN_CELL


def test_mention_state_follows_the_block_table() -> None:
    """Stav zmínky řádku je stav jeho bloku a vzácné řádky zmínky nemají."""
    notes = _notes()

    for row, text in enumerate(notes):
        expected = (
            BLOCK_STATES[row // MIN_CELL]
            if row < BLOCKS * MIN_CELL
            else MentionState.ABSENT
        )
        assert _state(text, "diabetes") == expected
        if expected is MentionState.ABSENT:
            assert not _mentioned_words(text)


def test_acronyms_reach_the_share_threshold_with_expected_code_cells() -> None:
    """Všech šest zkratek má aspoň ACRONYM_SHARE_MIN_OCCURRENCES výskytů; DM, DM2 a COPD mají buňky zkratka x kód 20/10/20/12."""
    codes = _column("diagnoses_icd", "icd_code")
    found = [
        find_diagnosis_acronyms(normalize_clinical_text(text), ACRONYMS)
        for text in _notes()
    ]
    occurrences = Counter(acronym for acronyms in found for acronym in acronyms)

    assert set(occurrences) == set(ACRONYMS)
    assert min(occurrences.values()) >= ACRONYM_SHARE_MIN_OCCURRENCES
    for acronym, diagnosis in (
        ("DM", Diagnosis.DIABETES),
        ("DM2", Diagnosis.DIABETES),
        ("COPD", Diagnosis.COPD),
    ):
        cells = Counter(
            (_has_code(code, diagnosis), acronym in present)
            for code, present in zip(codes, found, strict=True)
        )
        assert cells == {
            (True, True): 2 * MIN_CELL,
            (True, False): MIN_CELL,
            (False, True): 2 * MIN_CELL,
            (False, False): MIN_CELL + RARE_ROWS,
        }


def test_every_symptom_has_min_cell_notes_in_each_state() -> None:
    """U příznaku jsou tvrzená, jen negovaná i bez zmínky zprávy aspoň MIN_CELL."""
    for symptom in TEXT_MENTIONS.symptoms:
        states = Counter(_state(text, symptom) for text in _notes())
        assert min(states[state] for state in MentionState) >= MIN_CELL
