"""Sdílené volby vstupu modelu a význam úrovní předzpracování."""

import pytest

from mediparse.domain.model_input import TextPreprocessing


@pytest.mark.parametrize(
    ("level", "normalizes", "rewrites_variants"),
    [
        (TextPreprocessing.NONE, False, False),
        (TextPreprocessing.NORMALIZED, True, False),
        (TextPreprocessing.DICTIONARY, True, True),
    ],
)
def test_preprocessing_level_properties(
    level: TextPreprocessing, *, normalizes: bool, rewrites_variants: bool
) -> None:
    """Žebříček rozlišuje původní text, normalizaci a přepis variant slovníkem."""
    assert level.normalizes is normalizes
    assert level.rewrites_variants is rewrites_variants
