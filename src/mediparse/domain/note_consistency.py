"""Shoda textu syntetické zprávy s plánem: hlavičky sekcí, de-identifikační značky a zmínky diagnóz."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Final

from mediparse.domain.note_structure import DEID, PreambleValue

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from mediparse.domain.labels import Diagnosis
    from mediparse.domain.mentions import MentionModel
    from mediparse.domain.note_plan import NotePlan
    from mediparse.domain.note_structure import PreambleField, StructureModel

_AGE_SUFFIX: Final = r"[ -]?(?:years?[ -]old|y/?o)\b"
_AGE_MARKER: Final = re.compile(rf"___{_AGE_SUFFIX}", re.IGNORECASE)
_NUMERIC_AGE: Final = re.compile(rf"\b\d{{1,3}}{_AGE_SUFFIX}", re.IGNORECASE)
_FOREIGN_MARK: Final = re.compile(r"\[\*\*|\bXXX\b|(?<!_)(?:_{1,2}|_{4,})(?!_)")

type Sections = tuple[tuple[str, str], ...]
type Patterns = Mapping[Diagnosis, re.Pattern[str]]


def note_violations(
    text: str, plan: NotePlan, structure: StructureModel, mentions: MentionModel
) -> tuple[str, ...]:
    """Důvody, proč text zprávy neodpovídá plánu; prázdný výsledek znamená shodu.

    Returns:
        Popisy porušení pravidel jedné zprávy ze specifikace korpusu.
    """
    headers = {section.key: section.header for section in structure.sections}
    preamble, sections = _segment(text, headers)
    bodies = dict(sections)
    patterns = {
        diagnosis: _keywords(parameters.keywords)
        for diagnosis, parameters in mentions.diagnoses.items()
    }
    section = mentions.diagnosis_section
    return (
        *_section_violations(sections, plan, headers),
        *_preamble_violations(preamble, plan, structure.preamble),
        *_marked_place_violations(bodies, plan, structure),
        *_form_violations(text, plan),
        *_mention_violations(text, bodies, plan, patterns),
        *_diagnosis_section_violations(
            bodies.get(section), plan, patterns, section, headers[section]
        ),
    )


def _segment(text: str, headers: Mapping[str, str]) -> tuple[str, Sections]:
    keys = {header: key for key, header in headers.items()}
    alternatives = "|".join(map(re.escape, sorted(keys, key=len, reverse=True)))
    matches = list(re.finditer(rf"^(?P<header>{alternatives}):", text, re.MULTILINE))
    ends = [match.start() for match in matches[1:]] + [len(text)]
    sections = tuple(
        (keys[match["header"]], text[match.end() : end])
        for match, end in zip(matches, ends, strict=True)
    )
    return text[: matches[0].start()] if matches else text, sections


def _section_violations(
    sections: Sections, plan: NotePlan, headers: Mapping[str, str]
) -> list[str]:
    found = tuple(key for key, _ in sections)
    if found == plan.sections:
        return []
    expected = ", ".join(headers[key] for key in plan.sections)
    actual = ", ".join(headers[key] for key in found)
    return [f"Hlavičky sekcí neodpovídají plánu: plán {expected}; text {actual}."]


def _preamble_violations(
    preamble: str, plan: NotePlan, fields: Sequence[PreambleField]
) -> list[str]:
    values = _preamble_values(preamble, fields)
    violations: list[str] = []
    for field in fields:
        value = values.get(field.name)
        if field.value is PreambleValue.DEID and value != DEID:
            violations.append(f"Pole preambule {field.name} nemá hodnotu {DEID}.")
        if field.value is PreambleValue.SEX and value != plan.sex.code:
            violations.append(f"Pole {field.name} neodpovídá plánu ({plan.sex.code}).")
        if field.value is PreambleValue.TEXT and value in {None, "", DEID}:
            violations.append(f"Pole {field.name} nemá hodnotu.")
    return violations


def _preamble_values(preamble: str, fields: Sequence[PreambleField]) -> dict[str, str]:
    names = "|".join(re.escape(field.name) for field in fields)
    pattern = rf"\b(?P<name>{names}):[ \t]*(?P<value>.*?)[ \t]*(?=\b(?:{names}):|$)"
    return {
        match["name"]: match["value"]
        for match in re.finditer(pattern, preamble, re.MULTILINE)
    }


def _marked_place_violations(
    bodies: Mapping[str, str], plan: NotePlan, structure: StructureModel
) -> list[str]:
    body_reasons = [
        f"Tělo sekce {section.header} není {DEID}."
        for section in structure.sections
        if section.deid_body
        and section.key in plan.sections
        and bodies.get(section.key, "").strip() != DEID
    ]
    subheading_reasons = [
        f"Podnadpis {header} nemá hodnotu {DEID}."
        for key, header in _deid_subheadings(structure)
        if header in plan.subheadings
        and re.search(
            rf"^{re.escape(header)}:\s*{DEID}", bodies.get(key, ""), re.MULTILINE
        )
        is None
    ]
    return body_reasons + subheading_reasons


def _deid_subheadings(structure: StructureModel) -> list[tuple[str, str]]:
    return [
        (section.key, subheading.header)
        for section in structure.sections
        for group in section.subheadings
        for subheading in group
        if subheading.deid_value
    ]


def _form_violations(text: str, plan: NotePlan) -> list[str]:
    violations: list[str] = []
    if (_AGE_MARKER.search(text) is not None) != plan.age_marker:
        violations.append(
            "Věková značka chybí, ačkoli ji plán má."
            if plan.age_marker
            else "Text uvádí věk, ačkoli plán věkovou značku nemá."
        )
    if _NUMERIC_AGE.search(text) is not None:
        violations.append("Text uvádí číselný věk.")
    if _FOREIGN_MARK.search(text) is not None:
        violations.append(
            f"Text obsahuje jinou formu de-identifikační značky než {DEID}."
        )
    return violations


def _mention_violations(
    text: str, bodies: Mapping[str, str], plan: NotePlan, patterns: Patterns
) -> list[str]:
    planned = {mention.diagnosis for mention in plan.mentions}
    unplanned = [
        f"Diagnóza {diagnosis} nemá v plánu zmínku, ale text obsahuje její klíčové slovo."
        for diagnosis, pattern in patterns.items()
        if diagnosis not in planned and pattern.search(text)
    ]
    missing = [
        f"Plánovaná zmínka {mention.diagnosis} chybí v sekcích {', '.join(mention.sections)}."
        for mention in plan.mentions
        if not any(
            patterns[mention.diagnosis].search(bodies.get(key, ""))
            for key in mention.sections
        )
    ]
    return unplanned + missing


def _diagnosis_section_violations(
    body: str | None, plan: NotePlan, patterns: Patterns, section: str, header: str
) -> list[str]:
    if body is None or section not in plan.sections:
        return []
    planned = {m.diagnosis for m in plan.mentions if section in m.sections}
    return [
        f"{header} {'neobsahuje' if diagnosis in planned else 'obsahuje'} "
        f"klíčové slovo {diagnosis} v rozporu s plánem."
        for diagnosis, pattern in patterns.items()
        if (pattern.search(body) is not None) != (diagnosis in planned)
    ]


def _keywords(keywords: Iterable[str]) -> re.Pattern[str]:
    alternatives = "|".join(map(re.escape, sorted(keywords, key=len, reverse=True)))
    return re.compile(rf"\b(?:{alternatives})\b", re.IGNORECASE)
