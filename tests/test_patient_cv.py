"""Sdílený výběr vzorku předkola."""

import pytest

from mediparse.application.patient_cv import development_sample, patient_holdouts
from mediparse.infrastructure.sklearn_split import StratifiedGroupFolds
from tests.screening_support import notes


@pytest.mark.parametrize("parts", [1, 3])
def test_development_sample_matches_first_part(parts: int) -> None:
    """Vzorek je celá množina nebo první pacientská část."""
    corpus = notes(visits=30)
    partitioner = StratifiedGroupFolds()
    expected = (
        corpus if parts == 1 else patient_holdouts(corpus, 3, 7, partitioner)[0].test
    )
    assert development_sample(corpus, parts, 7, partitioner) == expected
