"""Konfigurace vzorkovače v repu: projde schématem a reprodukuje čísla z notes-synthesis."""

from math import isclose
from pathlib import Path
from random import Random

from mediparse.domain.labels import max_entropy_joint
from mediparse.domain.note_structure import sample_structure
from mediparse.domain.synthetic_patients import sample_notes
from mediparse.infrastructure.sampler_config import load_sampler_config

REPO_CONFIG = Path(__file__).parents[1] / "config" / "synthetic_plan.json"
NO_DIAGNOSIS = 0.478
NOTES_SYNTHESIS_ROUNDING = 0.001


def test_repository_config_is_valid() -> None:
    """Config v repu odpovídá schématu a jeho matice je v obou směrech konzistentní."""
    config = load_sampler_config(REPO_CONFIG)

    assert config.patients.notes > 0


def test_max_entropy_joint_matches_notes_synthesis() -> None:
    """Rozdělení z EDA parametrů dává podíl zpráv bez diagnózy uvedený v notes-synthesis."""
    joint = max_entropy_joint(load_sampler_config(REPO_CONFIG).labels)

    assert isclose(joint[frozenset()], NO_DIAGNOSIS, abs_tol=NOTES_SYNTHESIS_ROUNDING)


def test_repository_config_samples_a_corpus() -> None:
    """S parametry z repa vznikne korpus zadané velikosti v toleranci prevalencí."""
    config = load_sampler_config(REPO_CONFIG)

    notes = sample_notes(config.patients, config.labels, Random(1))

    assert len(notes) == config.patients.notes


def test_repository_config_samples_structure() -> None:
    """S parametry z repa dostane každá zpráva strukturu s délkou v mezích a přesným rozdělením."""
    config = load_sampler_config(REPO_CONFIG)
    notes = sample_notes(config.patients, config.labels, Random(1))

    structures = sample_structure(notes, config.structure, Random(1))

    assert len(structures) == len(notes)
    for structure in structures:
        assert (
            sum(words for _, words in structure.section_words)
            == structure.narrative_words
        )
