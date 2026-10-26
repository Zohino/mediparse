"""Text syntetické propouštěcí zprávy a target BHC odvozený z téhož textu."""

from __future__ import annotations

import re

from mediparse.domain.synthetic_discharge import text_token


def test_text_token_is_letters_only_and_unique_per_row() -> None:
    """Token z písmen přežije tokenizaci a v každé zprávě je jiný (DF = 1)."""
    tokens = [text_token("eda-sentinel", row) for row in range(60)]

    assert all(re.fullmatch(r"edasentinel[a-z]+", token) for token in tokens)
    assert len(set(tokens)) == len(tokens)
    assert text_token("p", 0) == "pa"
    assert text_token("p", 26) == "pba"
