"""Konfigurace vzorkovače v repu: dá se z ní vylosovat korpus se strukturou."""

from __future__ import annotations

from random import Random
from typing import TYPE_CHECKING

from mediparse.domain.note_structure import sample_structure
from mediparse.domain.synthetic_patients import sample_notes

if TYPE_CHECKING:
    from mediparse.domain.synthetic_plan import SamplerConfig


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
