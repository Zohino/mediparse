"""Property testy rozdělení zprávy na sekce nad lexikonem z config/."""

from __future__ import annotations

import string
from dataclasses import replace
from itertools import pairwise
from typing import TYPE_CHECKING, Final

from hypothesis import given
from hypothesis import strategies as st

from mediparse.domain.section_headers import (
    HEADER_PATTERN,
    OTHER,
    NoteSection,
    find_section_headers,
    normalized,
    split_sections,
    strip_diagnostic_sections,
)
from tests.section_support import REPOSITORY_LEXICON

if TYPE_CHECKING:
    from collections.abc import Sequence

type _Note = tuple[str, tuple[NoteSection, ...]]

_SECTIONS: Final = REPOSITORY_LEXICON.sections
_SECTION_OF: Final = {
    normalized(synonym): key
    for key, section in _SECTIONS.items()
    for synonym in section.synonyms
}
_PLAIN_LINE: Final = st.text(
    alphabet=string.ascii_letters + string.digits + " _-/,.()", max_size=20
)
_DIAGNOSTIC: Final = {k for k, v in _SECTIONS.items() if v.diagnostic}
_EOL: Final = st.sampled_from(("\n", "\r\n"))


@st.composite
def _header_line(draw: st.DrawFn, name: str) -> str:
    """Hlavička s dvojtečkou v povolených variantách odsazení, velikosti a konce názvu.

    Malá písmena se nevolí: hlavička musí začínat velkým písmenem.

    Returns:
        Text od začátku řádku po dvojtečku.
    """
    indent = draw(st.sampled_from(("", " ", "\t")))
    spelled = draw(st.sampled_from((name, name.upper())))
    suffix = draw(st.sampled_from(("", ".", " ", " .")))
    return f"{indent}{spelled}{suffix}:"


@st.composite
def _field_line(draw: st.DrawFn, name: str) -> str:
    """Řádek pole preambule s de-identifikovanou hodnotou.

    Returns:
        Hlavička pole, mezera a hodnota bez konce řádku.
    """
    return f"{draw(_header_line(name))} {draw(st.sampled_from(('___', '___.')))}"


@st.composite
def _repeat_line(draw: st.DrawFn, names: Sequence[str]) -> str:
    """Řádek, který opakuje hlavičku sekce nebo pole z `names`.

    Returns:
        Hlavička, případně s hodnotou na témže řádku.
    """
    header = draw(_header_line(draw(st.sampled_from(names))))
    return header + draw(st.sampled_from(("", " ___")))


@st.composite
def _body_lines(draw: st.DrawFn, eol: str, repeatable: Sequence[str]) -> list[str]:
    """Řádky těla bez dvojtečky, mezi nimi případně opakování dřívějších hlaviček.

    Returns:
        Řádky zakončené `eol`.
    """
    kinds = [_PLAIN_LINE, *([_repeat_line(repeatable)] if repeatable else [])]
    return [line + eol for line in draw(st.lists(st.one_of(kinds), max_size=4))]


@st.composite
def _preamble(draw: st.DrawFn, eol: str) -> tuple[str, list[str]]:
    """Preambule z řádků polí a řádků bez dvojtečky.

    Returns:
        Text preambule a pole, která v ní stála na začátku řádku.
    """
    lines: list[str] = []
    used: list[str] = []
    for item in draw(
        st.lists(st.one_of(st.sampled_from(REPOSITORY_LEXICON.preamble), st.none()))
    ):
        if item is None:
            lines.append(draw(_PLAIN_LINE) + eol)
            continue
        used.append(item)
        lines.append(draw(_field_line(item)) + eol)
    return "".join(lines), used


@st.composite
def _notes(draw: st.DrawFn) -> _Note:
    """Zpráva složená z dílů a dělení, které z konstrukce vyplývá.

    Args:
        draw: Zdroj Hypothesis.

    Returns:
        Text zprávy a očekávané části. Pole za sekcí otevře `OTHER` jen tehdy,
        když dosud nestálo na začátku řádku, jinak by první výskyt byl jinde.
    """
    eol = draw(_EOL)
    preamble, used = draw(_preamble(eol))
    parts = [NoteSection(OTHER, "", preamble)] if preamble else []
    synonyms: list[str] = []
    for key in draw(st.lists(st.sampled_from(tuple(_SECTIONS)), unique=True)):
        synonym = draw(st.sampled_from(_SECTIONS[key].synonyms))
        synonyms.extend(_SECTIONS[key].synonyms)
        repeatable = [*synonyms, *used]
        body = eol + "".join(draw(_body_lines(eol, repeatable)))
        parts.append(NoteSection(key, draw(_header_line(synonym)), body))
        fresh = [name for name in REPOSITORY_LEXICON.preamble if name not in used]
        opened = draw(
            st.lists(st.sampled_from(fresh), unique=True, max_size=2)
            if fresh
            else st.just([])
        )
        if not opened:
            continue
        lines = []
        for name in opened:
            lines.append(draw(_field_line(name)) + eol)
            lines.extend(draw(_body_lines(eol, repeatable)))
        used.extend(opened)
        parts.append(NoteSection(OTHER, "", "".join(lines)))
    if parts and draw(st.booleans()):
        previous = parts.pop()
        last = replace(previous, body=previous.body.removesuffix(eol))
        # Prázdný other by parser nevrátil: z textu "" nevznikne žádná část.
        if last.body or last.section != OTHER:
            parts.append(last)
    return "".join(p.header + p.body for p in parts), tuple(parts)


@st.composite
def _header_fragment(draw: st.DrawFn) -> str:
    """Hlavička libovolného synonyma, někdy s hodnotou na témže řádku.

    Returns:
        Hlavička s dvojtečkou a volitelnou hodnotou.
    """
    header = draw(_header_line(draw(st.sampled_from(sorted(_SECTION_OF))).title()))
    return header + draw(st.sampled_from(("", " ___")))


_FRAGMENTS: Final = st.one_of(
    st.text(alphabet=st.characters(codec="ascii"), max_size=12),
    st.sampled_from(("\n", "\r\n", "___")),
    _header_fragment(),
    st.sampled_from(REPOSITORY_LEXICON.preamble).flatmap(_field_line),
)


@given(st.lists(_FRAGMENTS, max_size=30).map("".join))
def test_split_of_any_text_keeps_the_shape_invariants(text: str) -> None:
    """Libovolný text se složí zpět a části mají tvar, který parser slibuje."""
    sections = split_sections(text, REPOSITORY_LEXICON)
    named = [s for s in sections if s.section != OTHER]

    assert "".join(s.header + s.body for s in sections) == text
    assert len({s.section for s in named}) == len(named)
    assert all(s.section in _SECTIONS for s in named)
    assert all(s.body and not s.header for s in sections if s.section == OTHER)
    assert all((a.section, b.section) != (OTHER, OTHER) for a, b in pairwise(sections))
    for section in named:
        match = HEADER_PATTERN.fullmatch(section.header)
        assert match is not None
        assert _SECTION_OF[normalized(match["header"])] == section.section
    starts = [h.start for h in find_section_headers(text, REPOSITORY_LEXICON)]
    assert starts == sorted(set(starts))


@given(_notes())
def test_note_built_from_known_parts_splits_into_exactly_them(note: _Note) -> None:
    """Zpráva z hlaviček, polí a těl se rozpadne přesně na své díly.

    Opakování dřívějších hlaviček a polí v tělech jsou volitelná, takže jde o text těla.
    """
    text, expected = note

    assert split_sections(text, REPOSITORY_LEXICON) == expected


@given(_notes())
def test_strip_equals_composition_of_expected_parts_without_diagnostic(
    note: _Note,
) -> None:
    """Strip je složení očekávaných částí bez diagnostických."""
    text, parts = note
    expected = "".join(p.header + p.body for p in parts if p.section not in _DIAGNOSTIC)

    assert strip_diagnostic_sections(text, REPOSITORY_LEXICON) == expected
