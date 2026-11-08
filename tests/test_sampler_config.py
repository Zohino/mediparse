"""Konfigurace vzorkovače v repu: dá se z ní vylosovat korpus se strukturou."""

from __future__ import annotations

import json
from random import Random

import pytest
from pydantic import ValidationError

from mediparse.synthetic.domain.note_structure import sample_structure
from mediparse.synthetic.domain.synthetic_patients import sample_notes
from mediparse.synthetic.domain.synthetic_plan import SamplerConfig
from tests.support import REPOSITORY_CONFIG


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


@pytest.mark.parametrize(
    "field",
    ["diagnosis_section", "family_section", "chronic_sections", "acute_sections"],
)
def test_unknown_mention_section_is_refused(field: str) -> None:
    """Odkaz na sekci mimo structure se odmítne už při validaci configu."""
    tree = json.loads(REPOSITORY_CONFIG.read_text(encoding="utf-8"))
    unknown = "neexistujici-sekce"
    if field in {"chronic_sections", "acute_sections"}:
        weights = tree["mentions"][field]
        weights[unknown] = weights.pop(next(iter(weights)))
    else:
        tree["mentions"][field] = unknown

    with pytest.raises(ValidationError, match=unknown):
        SamplerConfig.model_validate_json(json.dumps(tree))
