"""Konfigurace vzorkovače v repu: dá se z ní vylosovat korpus se strukturou."""

from __future__ import annotations

from random import Random
from typing import TYPE_CHECKING

from mediparse.domain.note_structure import sample_structure
from mediparse.domain.synthetic_patients import sample_notes

if TYPE_CHECKING:
    from mediparse.domain.synthetic_plan import SamplerConfig


def test_repository_config_samples_structure(sampler_config: SamplerConfig) -> None:
    """S parametry z repa dostane každá zpráva strukturu s délkou narativu v mezích configu."""
    notes = sample_notes(sampler_config.patients, sampler_config.labels, Random(1))

    structures = sample_structure(notes, sampler_config.structure, Random(1))

    assert len(structures) == len(notes)
    assert any(structure.section_words for structure in structures)
    narrative = sampler_config.structure.narrative
    for structure in structures:
        assert structure.section_words.keys() <= set(structure.sections)
        if structure.section_words:
            allocated = sum(structure.section_words.values())
            assert narrative.min_words <= allocated <= narrative.max_words
