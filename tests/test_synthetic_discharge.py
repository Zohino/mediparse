"""Text syntetické propouštěcí zprávy: preambule, sekce lexikonu, varianty bloků a target BHC."""

from __future__ import annotations

import re
from typing import Final

import pytest

from mediparse.domain.inputs import InvalidInputError
from mediparse.domain.note_text import DEID
from mediparse.domain.section_headers import (
    OTHER,
    NoteSection,
    SectionLexicon,
    split_sections,
)
from mediparse.domain.synthetic_discharge import (
    BHC_SECTION,
    SyntheticDischarge,
    require_bhc_layout,
    synthetic_discharge,
)
from tests.section_support import REPOSITORY_LEXICON

PREFIX: Final = "eda-sentinel"
SENTINEL_LETTERS: Final = "edasentinel"
LEXICON: Final = REPOSITORY_LEXICON
KEYS: Final = tuple(LEXICON.sections)
BHC_INDEX: Final = KEYS.index(BHC_SECTION)
SKIPPED: Final = KEYS[BHC_INDEX + 1]
REPEATING_VARIANT: Final = 2
SKIPPING_VARIANT: Final = 3
FIRST_DIAGNOSTIC: Final = next(
    key for key, section in LEXICON.sections.items() if section.diagnostic
)
FIRST_KEY: Final = KEYS[0]


def _token(row: int) -> str:
    return SENTINEL_LETTERS + chr(ord("a") + row % 26)


def _note(row: int, block: int) -> SyntheticDischarge:
    return synthetic_discharge(_cell(row), _token(row), LEXICON, block)


def _cell(row: int) -> str:
    return f"{PREFIX}-discharge-text-{row}"


def _header_lines(text: str) -> list[str]:
    """Řádky hlaviček sekcí lexikonu bez pseudohlaviček tokenu.

    Returns:
        Řádky přesně rovné synonymu s dvojtečkou.
    """
    synonyms = {
        f"{synonym}:"
        for section in LEXICON.sections.values()
        for synonym in section.synonyms
    }
    return [line for line in text.splitlines() if line in synonyms]


@pytest.mark.parametrize("block", range(4))
def test_text_starts_with_cell_and_has_preamble_fields(block: int) -> None:
    """Text začíná buňkou na samostatném řádku a nese každé pole preambule se značkou ___."""
    lines = _note(block * 10, block).text.splitlines()

    assert lines[0] == _cell(block * 10)
    for field in LEXICON.preamble:
        assert f"{field}: {DEID}" in lines


@pytest.mark.parametrize("block", [0, 1])
def test_text_has_every_section_header_in_lexicon_order(block: int) -> None:
    """Hlavičky sekcí jdou v pořadí lexikonu se synonymem podle bloku (modulo počet synonym)."""
    headers = _header_lines(_note(block, block).text)

    expected = [
        f"{section.synonyms[block % len(section.synonyms)]}:"
        for section in LEXICON.sections.values()
    ]
    assert headers == expected


def test_variant_three_omits_only_the_section_after_bhc() -> None:
    """Varianta 3 vynechá právě sekci hned za BHC."""
    headers = _header_lines(_note(3, SKIPPING_VARIANT).text)

    expected = [
        f"{section.synonyms[SKIPPING_VARIANT % len(section.synonyms)]}:"
        for key, section in LEXICON.sections.items()
        if key != SKIPPED
    ]
    assert headers == expected


def test_parser_finds_every_section_and_each_body_has_the_token() -> None:
    """Křížová kontrola s parserem: varianta 0 má všechny sekce lexikonu a v každém těle token zprávy."""
    note = _note(0, 0)

    parts = [p for p in split_sections(note.text, LEXICON) if p.section != OTHER]

    assert [p.section for p in parts] == list(KEYS)
    token = _token(0)
    assert all(token in p.body for p in parts)


@pytest.mark.parametrize("block", range(4))
def test_every_word_is_token_or_public_word_of_the_block(block: int) -> None:
    """Pravidlo kanárka pro text: slovo je token zprávy, nebo slovo synonym, polí preambule či buňky."""
    row = block * 10 + 3
    public = {
        word
        for section in LEXICON.sections.values()
        for synonym in section.synonyms
        for word in re.findall(r"[a-z]{2,}", synonym.lower())
    }
    public |= {
        word
        for field in LEXICON.preamble
        for word in re.findall(r"[a-z]{2,}", field.lower())
    }
    public |= set(re.findall(r"[a-z]{2,}", _cell(row)))
    public.add(_token(row))

    words = set(re.findall(r"[a-z]{2,}", _note(row, block).text.lower()))

    assert words <= public


def test_variant_two_repeats_diagnostic_header_in_last_section() -> None:
    """Varianta 2 opakuje hlavičku diagnostické sekce v těle poslední sekce."""
    synonym = LEXICON.sections[FIRST_DIAGNOSTIC].synonyms[
        REPEATING_VARIANT % len(LEXICON.sections[FIRST_DIAGNOSTIC].synonyms)
    ]

    parts = split_sections(_note(2, REPEATING_VARIANT).text, LEXICON)

    assert f"{synonym}:" in parts[-1].body
    assert parts[-1].section == KEYS[-1]


@pytest.mark.parametrize("block", [0, 1, 3])
def test_other_variants_do_not_repeat_the_header(block: int) -> None:
    """Hlavičku diagnostické sekce opakuje jen varianta 2."""
    synonyms = LEXICON.sections[FIRST_DIAGNOSTIC].synonyms
    parts = split_sections(_note(block, block).text, LEXICON)

    assert f"{synonyms[block % len(synonyms)]}:" not in parts[-1].body


def test_pseudo_header_and_synonym_line_carry_the_token_in_every_body() -> None:
    """Tělo sekce má pseudohlavičku ``Token:`` a řádek se synonymem a tokenem malými písmeny."""
    token = _token(5)

    lines = _note(5, 0).text.splitlines()

    assert lines.count(f"{token.capitalize()}:") == len(KEYS)
    first_synonym = LEXICON.sections[FIRST_KEY].synonyms[0].lower()
    assert lines.count(f"{first_synonym} {token}") == 1


@pytest.mark.parametrize(
    ("block", "skips", "repeats"),
    [(4, False, False), (5, False, False), (6, False, True), (7, True, False)],
)
def test_blocks_beyond_variants_wrap_around(
    block: int, *, skips: bool, repeats: bool
) -> None:
    """Bloky 4-7 dají varianty bloků 0-3, takže rozvržení vydrží víc bloků než variant."""
    parts = split_sections(_note(block * 10, block).text, LEXICON)

    keys = [p.section for p in parts if p.section != OTHER]
    assert (SKIPPED not in keys) is skips
    diagnostic = LEXICON.sections[FIRST_DIAGNOSTIC].synonyms
    repeated = f"{diagnostic[block % len(diagnostic)]}:" in parts[-1].body
    assert repeated is repeats


def _flat(text: str) -> str:
    return " ".join(text.split())


def _bhc_and_next(block: int) -> tuple[SyntheticDischarge, NoteSection, NoteSection]:
    """Zpráva bloku, její sekce BHC a sekce hned za ní podle parseru.

    Returns:
        Zpráva, BHC a následující sekce v textu.
    """
    note = _note(block, block)
    parts = [p for p in split_sections(note.text, LEXICON) if p.section != OTHER]
    index = [p.section for p in parts].index(BHC_SECTION)
    return note, parts[index], parts[index + 1]


def test_target_of_variant_zero_is_the_whole_bhc_body() -> None:
    """Varianta 0: target je celé tělo BHC (shoda)."""
    note, bhc, _ = _bhc_and_next(0)

    assert note.bhc_target == _flat(bhc.body)


def test_target_of_variant_one_continues_through_the_whole_next_section() -> None:
    """Varianta 1: target je tělo BHC a celá následující sekce (hlavička a tělo)."""
    note, bhc, following = _bhc_and_next(1)

    assert note.bhc_target.startswith(_flat(bhc.body))
    assert note.bhc_target == _flat(bhc.body + following.header + following.body)


def test_target_of_variant_two_is_a_proper_prefix_of_the_body() -> None:
    """Varianta 2 (náš konec později): target je vlastní prefix našeho úseku."""
    note, bhc, _ = _bhc_and_next(2)

    assert note.bhc_target
    assert _flat(bhc.body).startswith(note.bhc_target)
    assert note.bhc_target != _flat(bhc.body)


def test_target_of_variant_three_runs_into_the_section_after_the_skipped_one() -> None:
    """Varianta 3: target je tělo BHC, hlavička další sekce a první řádek jejího těla."""
    note, bhc, following = _bhc_and_next(3)

    first_line = following.body.strip().splitlines()[0]
    assert following.section == KEYS[BHC_INDEX + 2]
    assert note.bhc_target == _flat(f"{bhc.body} {following.header} {first_line}")


@pytest.mark.parametrize("block", range(4))
def test_target_has_only_single_spaces(block: int) -> None:
    """Target nemá dvě mezery za sebou ani jiný bílý znak než mezeru (``necisty_target`` = 0)."""
    target = _note(block, block).bhc_target

    assert target
    assert re.fullmatch(r"\S+( \S+)*", target)


def _lexicon_of(*keys: str) -> SectionLexicon:
    return SectionLexicon(
        sections={key: LEXICON.sections[key] for key in keys},
        preamble=LEXICON.preamble,
    )


def test_layout_without_bhc_is_refused_naming_the_key() -> None:
    """Lexikon bez sekce BHC je InvalidInputError a hláška jmenuje klíč."""
    with pytest.raises(InvalidInputError, match=r"nemá sekci hospital_course"):
        require_bhc_layout(_lexicon_of(FIRST_DIAGNOSTIC, FIRST_KEY))


def test_layout_with_one_section_after_bhc_is_refused() -> None:
    """Za BHC musí být aspoň dvě sekce, jedna se vynechá a řez targetu končí v další."""
    with pytest.raises(InvalidInputError, match=r"sekcí: 1,"):
        require_bhc_layout(_lexicon_of(FIRST_DIAGNOSTIC, BHC_SECTION, SKIPPED))


def test_layout_with_two_sections_after_bhc_is_accepted() -> None:
    """Dvě sekce za BHC stačí."""
    require_bhc_layout(_lexicon_of(BHC_SECTION, SKIPPED, FIRST_DIAGNOSTIC))


def test_layout_of_the_repository_lexicon_is_accepted() -> None:
    """Lexikon z configu rozvržení splňuje."""
    require_bhc_layout(LEXICON)
