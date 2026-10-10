"""Doména evaluace: ověření, že pravda diagnózy obsahuje obě třídy."""

from __future__ import annotations

import pytest

from mediparse.domain.evaluation import ensure_both_classes
from mediparse.domain.inputs import InvalidInputError
from mediparse.domain.labels import Diagnosis


def test_mixed_truth_passes() -> None:
    """Smíšená pravda projde."""
    ensure_both_classes(Diagnosis.CKD, [True, False, True])


@pytest.mark.parametrize("truth", [[True, True], [False, False, False]])
def test_single_class_is_rejected(truth: list[bool]) -> None:
    """Samé True i samé False skončí InvalidInputError s diagnózou."""
    with pytest.raises(InvalidInputError, match="ckd"):
        ensure_both_classes(Diagnosis.CKD, truth)
