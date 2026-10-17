"""Prostředí běhu: přísná validace revize kódu a verzí knihoven."""

from __future__ import annotations

from typing import Final

import pytest
from pydantic import ValidationError

from mediparse.domain.environment import Environment, SourceRevision

COMMIT: Final = "a" * 40


@pytest.mark.parametrize("commit", ["a" * 7, "A" * 40, "a" * 41])
def test_invalid_commit_is_rejected(commit: str) -> None:
    """Zkrácený, velkými písmeny psaný i příliš dlouhý commit je odmítnut."""
    with pytest.raises(ValidationError):
        SourceRevision(commit=commit, dirty=False)


def test_environment_rejects_unknown_field_and_non_strict_types() -> None:
    """Prostředí je přísné: cizí pole a balíček s jinou verzí než text neprojdou."""
    source = SourceRevision(commit=COMMIT, dirty=False)

    with pytest.raises(ValidationError):
        Environment.model_validate({"packages": {}, "source": source, "extra": 1})
    with pytest.raises(ValidationError):
        Environment.model_validate({"packages": {"python": 3}, "source": source})
