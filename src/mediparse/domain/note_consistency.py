"""Shoda textu syntetické zprávy s plánem: hlavičky sekcí, de-identifikační značky a zmínky diagnóz."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Final

from mediparse.domain.note_structure import Sex

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from mediparse.domain.labels import Diagnosis
    from mediparse.domain.mentions import MentionModel
    from mediparse.domain.note_plan import NotePlan, PlannedMention
    from mediparse.domain.note_structure import StructureModel

DEID: Final = "___"

_PREAMBLE_MARKED: Final = (
    "Name",
    "Unit No",
    "Admission Date",
    "Discharge Date",
    "Date of Birth",
    "Attending",
)
_MARKED_SECTIONS: Final = ("social_history", "followup_instructions")
_MARKED_SUBHEADING: Final = "Facility"
_SEX_CODES: Final = {Sex.FEMALE: "F", Sex.MALE: "M"}
_AGE_MARKER: Final = re.compile(r"___[ -]?(?:years?[ -]old|y/?o)\b", re.IGNORECASE)
_NUMERIC_AGE: Final = re.compile(
    r"\b\d{1,3}[ -]?(?:years?[ -]old|y/?o)\b", re.IGNORECASE
)
_FOREIGN_MARK: Final = re.compile(r"\[\*\*|\bXXX\b|(?<!_)(?:_{1,2}|_{4,})(?!_)")
_SEX_VALUE: Final = re.compile(r"\bSex:[ \t]*(\S*)")
_SERVICE_VALUE: Final = re.compile(r"\bService:[ \t]*(\S*)")

type Sections = tuple[tuple[str, str], ...]


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
    return (
        *_section_violations(sections, plan, headers),
        *_preamble_violations(preamble, plan),
        *_mark_violations(text, bodies, plan, headers),
        *_mention_violations(text, bodies, plan, mentions),
        *_diagnosis_section_violations(bodies, plan, mentions),
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


def _preamble_violations(preamble: str, plan: NotePlan) -> list[str]:
    violations = [
        f"Pole preambule {field} nemá hodnotu {DEID}."
        for field in _PREAMBLE_MARKED
        if re.search(rf"\b{re.escape(field)}:[ \t]*{DEID}", preamble) is None
    ]
    if _value(_SEX_VALUE, preamble) != _SEX_CODES[plan.sex]:
        violations.append(f"Pole Sex neodpovídá plánu ({_SEX_CODES[plan.sex]}).")
    service = _value(_SERVICE_VALUE, preamble)
    if not service or service == DEID or service.endswith(":"):
        violations.append("Pole Service nemá hodnotu.")
    return violations


def _mark_violations(
    text: str, bodies: Mapping[str, str], plan: NotePlan, headers: Mapping[str, str]
) -> list[str]:
    violations = [
        f"Tělo sekce {headers[key]} není {DEID}."
        for key in _MARKED_SECTIONS
        if key in plan.sections and bodies.get(key, "").strip() != DEID
    ]
    facility = re.search(rf"^{_MARKED_SUBHEADING}:\s*{DEID}", text, re.MULTILINE)
    if _MARKED_SUBHEADING in plan.subheadings and facility is None:
        violations.append(f"Podnadpis {_MARKED_SUBHEADING} nemá hodnotu {DEID}.")
    if (_AGE_MARKER.search(text) is not None) != plan.age_marker:
        violations.append(_age_reason(planned=plan.age_marker))
    if _NUMERIC_AGE.search(text) is not None:
        violations.append("Text uvádí číselný věk.")
    if _FOREIGN_MARK.search(text) is not None:
        violations.append(
            f"Text obsahuje jinou formu de-identifikační značky než {DEID}."
        )
    return violations


def _mention_violations(
    text: str, bodies: Mapping[str, str], plan: NotePlan, mentions: MentionModel
) -> list[str]:
    planned = {mention.diagnosis: mention for mention in plan.mentions}
    return [
        reason
        for diagnosis, parameters in mentions.diagnoses.items()
        if (
            reason := _mention_reason(
                diagnosis,
                planned.get(diagnosis),
                _keywords(parameters.keywords),
                text,
                bodies,
            )
        )
        is not None
    ]


def _mention_reason(
    diagnosis: Diagnosis,
    mention: PlannedMention | None,
    keywords: re.Pattern[str],
    text: str,
    bodies: Mapping[str, str],
) -> str | None:
    if mention is None:
        if keywords.search(text) is None:
            return None
        return f"Diagnóza {diagnosis} nemá v plánu zmínku, ale text obsahuje její klíčové slovo."
    if any(keywords.search(bodies.get(key, "")) for key in mention.sections):
        return None
    return (
        f"Plánovaná zmínka {diagnosis} chybí v sekcích {', '.join(mention.sections)}."
    )


def _diagnosis_section_violations(
    bodies: Mapping[str, str], plan: NotePlan, mentions: MentionModel
) -> list[str]:
    section = mentions.diagnosis_section
    if section not in plan.sections:
        return []
    body = bodies.get(section, "")
    planned = {m.diagnosis for m in plan.mentions if section in m.sections}
    return [
        f"Discharge Diagnosis {'obsahuje' if diagnosis not in planned else 'neobsahuje'} "
        f"klíčové slovo {diagnosis} v rozporu s plánem."
        for diagnosis, parameters in mentions.diagnoses.items()
        if (_keywords(parameters.keywords).search(body) is not None)
        != (diagnosis in planned)
    ]


def _keywords(keywords: Iterable[str]) -> re.Pattern[str]:
    alternatives = "|".join(map(re.escape, sorted(keywords, key=len, reverse=True)))
    return re.compile(rf"\b(?:{alternatives})\b", re.IGNORECASE)


def _value(pattern: re.Pattern[str], text: str) -> str | None:
    match = pattern.search(text)
    return match[1] if match else None


def _age_reason(*, planned: bool) -> str:
    if planned:
        return "Věková značka chybí, ačkoli ji plán má."
    return "Text uvádí věk, ačkoli plán věkovou značku nemá."
