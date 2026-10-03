"""Konfigurace vzorkovače v repu: reprodukuje čísla z notes-synthesis."""

from __future__ import annotations

from math import isclose
from random import Random
from typing import TYPE_CHECKING

from mediparse.domain.labels import Diagnosis, max_entropy_joint
from mediparse.domain.mentions import sample_mentions
from mediparse.domain.note_structure import sample_structure
from mediparse.domain.synthetic_patients import sample_notes

if TYPE_CHECKING:
    from mediparse.domain.synthetic_plan import SamplerConfig

NO_DIAGNOSIS = 0.478
NOTES_SYNTHESIS_ROUNDING = 0.001


def test_max_entropy_joint_matches_notes_synthesis(
    sampler_config: SamplerConfig,
) -> None:
    """Rozdělení z EDA parametrů dává podíl zpráv bez diagnózy uvedený v notes-synthesis."""
    joint = max_entropy_joint(sampler_config.labels)

    assert isclose(joint[frozenset()], NO_DIAGNOSIS, abs_tol=NOTES_SYNTHESIS_ROUNDING)


def test_repository_config_samples_structure(sampler_config: SamplerConfig) -> None:
    """S parametry z repa dostane každá zpráva strukturu s délkou v mezích a přesným rozdělením."""
    notes = sample_notes(sampler_config.patients, sampler_config.labels, Random(1))

    structures = sample_structure(notes, sampler_config.structure, Random(1))

    assert len(structures) == len(notes)
    for structure in structures:
        assert (
            sum(words for _, words in structure.section_words)
            == structure.narrative_words
        )


NO_MENTION_AFTER_ABLATION = {
    Diagnosis.DIABETES: 0.262,
    Diagnosis.CKD: 0.316,
    Diagnosis.HEART_FAILURE: 0.327,
    Diagnosis.ATRIAL_FIBRILLATION: 0.261,
    Diagnosis.AKI: 0.628,
}
ABLATION_SLACK = 0.03
LARGE_CORPUS = 20000


def test_repository_config_matches_ablation_shares(
    sampler_config: SamplerConfig,
) -> None:
    """Podíl pozitivních bez zmínky po odstranění DD odpovídá notes-synthesis."""
    patients = sampler_config.patients.model_copy(update={"notes": LARGE_CORPUS})
    labels = sampler_config.labels.model_copy(update={"prevalence_tolerance": 1.0})
    notes = sample_notes(patients, labels, Random(1))
    structures = sample_structure(notes, sampler_config.structure, Random(2))
    mentions = sample_mentions(notes, structures, sampler_config.mentions, Random(3))

    for diagnosis, expected in NO_MENTION_AFTER_ABLATION.items():
        positives = [
            note_mentions
            for note, note_mentions in zip(notes, mentions, strict=True)
            if diagnosis in note.labels
        ]
        hidden = sum(
            not any(
                m.diagnosis is diagnosis and set(m.sections) - {"discharge_diagnosis"}
                for m in note_mentions
            )
            for note_mentions in positives
        )
        assert abs(hidden / len(positives) - expected) <= ABLATION_SLACK
