"""Zadání verbalizace: ze šablony instrukcí, plánu zprávy a configu vznikne přesný text pro model."""

from __future__ import annotations

from string import Template
from typing import TYPE_CHECKING, Final

from mediparse.domain.mentions import MentionStatus
from mediparse.domain.note_structure import DEID, PreambleValue

if TYPE_CHECKING:
    from collections.abc import Mapping
    from collections.abc import Set as AbstractSet

    from mediparse.domain.labels import Diagnosis
    from mediparse.domain.mentions import MentionModel
    from mediparse.domain.note_plan import NotePlan, PlannedMention
    from mediparse.domain.note_structure import (
        PreambleField,
        SectionModel,
        StructureModel,
    )
    from mediparse.domain.synthetic_plan import SamplerConfig

_STATUS_INSTRUCTIONS: Final = {
    MentionStatus.AFFIRMED: "state that the patient has it",
    MentionStatus.AFFIRMED_UNCODED: "state that the patient has it",
    MentionStatus.NEGATED: (
        "explicitly negate it, with a negation cue ({cues}) at most {window} characters"
        " before the keyword in the same sentence"
    ),
    MentionStatus.FAMILY_HISTORY: (
        "mention it only as a condition of a family member, with `family history`"
        " at most {window} characters before the keyword in the same sentence"
    ),
    MentionStatus.UNCERTAIN: (
        "mention it as possible or suspected, not confirmed, and without a negation cue"
    ),
}


def render_prompt(template: str, plan: NotePlan, config: SamplerConfig) -> str:
    """Vyplní šablonu instrukcí údaji z plánu zprávy a z configu vzorkovače.

    Returns:
        Zadání pro model; stejné vstupy dávají stejný text.
    """
    structure = config.structure
    return Template(template).substitute(
        note_id=plan.note_id,
        preamble=_preamble(plan, structure),
        sections=_sections(plan, structure),
        narrative_deid=plan.narrative_deid,
        age=_age(plan),
        diagnoses=_diagnoses(plan, structure.headers, config.mentions),
    )


def _preamble(plan: NotePlan, structure: StructureModel) -> str:
    return "\n".join(
        "  ".join(f"{field.name}: {_preamble_value(field, plan)}" for field in line)
        for line in structure.preamble
    )


def _preamble_value(field: PreambleField, plan: NotePlan) -> str:
    if field.value is PreambleValue.DEID:
        return DEID
    if field.value is PreambleValue.SEX:
        return plan.sex.code
    return f"<{field.name.lower()} consistent with the note>"


def _sections(plan: NotePlan, structure: StructureModel) -> str:
    return "\n".join(
        line
        for section in structure.sections
        if section.key in plan.sections
        for line in _section_lines(plan, section)
    )


def _section_lines(plan: NotePlan, section: SectionModel) -> list[str]:
    if section.deid_body:
        body = f"the body is exactly {DEID}"
    elif section.key in plan.section_words:
        body = f"about {plan.section_words[section.key]} words"
    else:
        body = "short"
    subheadings = [
        subheading
        for group in section.subheadings
        for subheading in group
        if subheading.header in plan.subheadings
    ]
    return [
        f"- {section.header}: {body}",
        *(
            f"  - {subheading.header}: {DEID if subheading.deid_value else ''}".rstrip()
            for subheading in subheadings
        ),
    ]


def _age(plan: NotePlan) -> str:
    if plan.age_marker:
        return "State the patient's age exactly once as `___ year old` (or `___ y/o`)."
    return "Do not state the patient's age in any form."


def _diagnoses(
    plan: NotePlan, headers: Mapping[str, str], mentions: MentionModel
) -> str:
    mentioned = {mention.diagnosis for mention in plan.mentions}
    blocks = (
        _mentions(plan, mentioned, headers, mentions),
        _discharge_diagnosis(plan, headers, mentions),
        _forbidden(mentioned, mentions),
    )
    return "\n\n".join(block for block in blocks if block)


def _mentions(
    plan: NotePlan,
    mentioned: AbstractSet[Diagnosis],
    headers: Mapping[str, str],
    mentions: MentionModel,
) -> str:
    lines = [_mention_line(mention, headers, mentions) for mention in plan.mentions]
    lines = lines or ["- Mention none of the diagnoses below by keyword."]
    unmentioned = [_name(d, mentions) for d in plan.labels if d not in mentioned]
    if unmentioned:
        lines.append(
            f"- The patient also has {', '.join(unmentioned)}, but never name it; you may"
            " describe it indirectly, for example through laboratory values or treatment."
        )
    return "\n".join(lines)


def _mention_line(
    mention: PlannedMention, headers: Mapping[str, str], mentions: MentionModel
) -> str:
    instruction = _STATUS_INSTRUCTIONS[mention.status].format(
        cues=", ".join(f"`{cue}`" for cue in mentions.negation.cues),
        window=mentions.negation.window_chars,
    )
    keywords = ", ".join(
        f"`{k}`" for k in mentions.diagnoses[mention.diagnosis].keywords
    )
    sections = ", ".join(headers[key] for key in mention.sections)
    return (
        f"- {_name(mention.diagnosis, mentions)}: {instruction}; use one of {keywords};"
        f" mention it in these sections only: {sections}."
    )


def _discharge_diagnosis(
    plan: NotePlan, headers: Mapping[str, str], mentions: MentionModel
) -> str:
    section = mentions.diagnosis_section
    if section not in plan.sections:
        return ""
    named = [
        _name(mention.diagnosis, mentions)
        for mention in plan.mentions
        if section in mention.sections
    ]
    if not named:
        return f"The {headers[section]} section names none of the diagnoses above; list other diagnoses."
    return f"The {headers[section]} section names {', '.join(named)} and none of the other diagnoses."


def _forbidden(mentioned: AbstractSet[Diagnosis], mentions: MentionModel) -> str:
    keywords = [
        f"`{keyword}`"
        for diagnosis, parameters in mentions.diagnoses.items()
        if diagnosis not in mentioned
        for keyword in parameters.keywords
    ]
    if not keywords:
        return ""
    return (
        "Never write any of these words or abbreviations anywhere in the note,"
        f" in any letter case: {', '.join(keywords)}."
    )


def _name(diagnosis: Diagnosis, mentions: MentionModel) -> str:
    return mentions.diagnoses[diagnosis].keywords[0]
