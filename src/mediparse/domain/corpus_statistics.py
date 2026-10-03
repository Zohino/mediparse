"""Korpus jako celek proti modelu: úplnost, délka narativu, hustota značek a prevalence labelů."""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import TYPE_CHECKING

from mediparse.domain.labels import label_shares, prevalence_outliers
from mediparse.domain.note_text import AGE_MARKER, DEID, line_headers, segment

if TYPE_CHECKING:
    import re
    from collections.abc import Mapping, Sequence

    from mediparse.domain.labels import Diagnosis, LabelModel
    from mediparse.domain.note_plan import NotePlan
    from mediparse.domain.note_structure import NarrativeModel, StructureModel
    from mediparse.domain.synthetic_plan import SamplerConfig


@dataclass(frozen=True)
class CorpusStatistics:
    """Vlastnosti korpusu změřené na textu zpráv a na jejich labelech."""

    median_words: float
    log_sigma: float
    deid_density: float
    prevalence: Mapping[Diagnosis, float]


def corpus_violations(
    texts: Mapping[str, str], plans: Mapping[str, NotePlan], config: SamplerConfig
) -> tuple[str, ...]:
    """Porušení korpusu jako celku; zprávy bez plánu hlásí kontroly jedné zprávy.

    Returns:
        Nic pro korpus bez zpráv, neúplnost pro korpus, kterému chybí zprávy k plánům,
        jinak vlastnosti mimo tolerance modelu.
    """
    missing = plans.keys() - texts.keys()
    if texts and missing:
        return (f"Korpus není úplný: chybí {len(missing)} z {len(plans)} zpráv.",)
    notes = [
        (plan, texts[note_id]) for note_id, plan in plans.items() if note_id in texts
    ]
    if not notes:
        return ()
    measured = corpus_statistics(notes, config.structure)
    return statistics_violations(measured, config.structure.narrative, config.labels)


def corpus_statistics(
    notes: Sequence[tuple[NotePlan, str]], structure: StructureModel
) -> CorpusStatistics:
    """Změří neprázdný korpus; narativ tvoří sekce s délkou v plánu bez podnadpisů.

    Returns:
        Medián a směrodatnou odchylku logaritmu délky narativu ve slovech, hustotu
        narativních značek bez věkové značky a podíl zpráv s každým labelem.
    """
    subheadings = line_headers(
        subheading.header for _, subheading in structure.subheadings
    )
    words, marks = zip(
        *(
            _narrative(plan, text, structure.headers, subheadings)
            for plan, text in notes
        ),
        strict=True,
    )
    logs = [math.log(count) for count in words if count > 0]
    return CorpusStatistics(
        median_words=statistics.median(words),
        log_sigma=statistics.stdev(logs) if len(logs) > 1 else 0.0,
        deid_density=sum(marks) / max(sum(words), 1),
        prevalence=label_shares([plan.labels for plan, _ in notes]),
    )


def statistics_violations(
    measured: CorpusStatistics, narrative: NarrativeModel, labels: LabelModel
) -> tuple[str, ...]:
    """Vlastnosti korpusu mimo tolerance configu.

    Returns:
        Popisy porušení s naměřenou hodnotou a cílem; prázdný výsledek znamená shodu.
    """
    violations: list[str] = []
    if (
        abs(measured.median_words / narrative.median_words - 1)
        > narrative.median_tolerance
    ):
        violations.append(
            f"Medián délky narativu {measured.median_words:.1f} slov je mimo"
            f" {narrative.median_words} ± {narrative.median_tolerance:.0%}."
        )
    if abs(measured.log_sigma - narrative.sigma) > narrative.sigma_tolerance:
        violations.append(
            f"Směrodatná odchylka logaritmu délky narativu {measured.log_sigma:.3f} je mimo"
            f" {narrative.sigma} ± {narrative.sigma_tolerance}."
        )
    if (
        abs(measured.deid_density / narrative.deid_per_word - 1)
        > narrative.deid_tolerance
    ):
        violations.append(
            f"Hustota narativních značek {measured.deid_density:.4f} na slovo je mimo"
            f" {narrative.deid_per_word} ± {narrative.deid_tolerance:.0%}."
        )
    violations.extend(
        f"Prevalence {diagnosis} {measured.prevalence[diagnosis]:.3f} je mimo"
        f" {labels.prevalence[diagnosis]:.3f} ± {labels.prevalence_tolerance}."
        for diagnosis in prevalence_outliers(measured.prevalence, labels)
    )
    return tuple(violations)


def _narrative(
    plan: NotePlan,
    text: str,
    headers: Mapping[str, str],
    subheadings: re.Pattern[str],
) -> tuple[int, int]:
    _, sections = segment(text, headers)
    bodies = [
        subheadings.sub("", body) for key, body in sections if key in plan.section_words
    ]
    words = sum(len(body.split()) for body in bodies)
    marks = sum(body.count(DEID) - len(AGE_MARKER.findall(body)) for body in bodies)
    return words, marks
