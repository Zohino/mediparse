"""Výběr rodin předkola a doladění na vývojové části."""

from __future__ import annotations

from dataclasses import replace

import pytest

from mediparse.domain.inputs import InvalidInputError
from mediparse.domain.stage_tuning import StageTrialRecord, family_candidates


def _record(number: int, branch: str, value: float) -> StageTrialRecord:
    return StageTrialRecord(
        number,
        {"branch": branch},
        default=False,
        value=value,
        converged=True,
        seconds=1.0,
    )


def test_candidates_match_r_rule() -> None:
    """Příklad R A2: a b a b c c, při shodě vítězí nižší číslo."""
    families = {"a": "logreg", "b": "linear_svm", "c": "nb_svm"}
    records = tuple(
        _record(i, f"word/{families[family]}", value)
        for i, (family, value) in enumerate(
            zip("ababcc", (0.80, 0.78, 0.82, 0.82, 0.70, 0.70), strict=True)
        )
    )
    assert tuple(r.number for r in family_candidates(records)) == (2, 3)
    assert tuple(
        r.number
        for r in family_candidates(tuple(replace(r, value=0.8) for r in records))
    ) == (0, 1)


def test_candidates_group_by_family_not_branch() -> None:
    """Dvě reprezentace NB-SVM obsadí jen jedno místo kontroly."""
    records = (
        _record(0, "word/nb_svm", 0.9),
        _record(1, "char_wb/nb_svm", 0.8),
        _record(2, "word/linear_svm", 0.7),
    )
    assert tuple(r.number for r in family_candidates(records)) == (0, 2)
    with pytest.raises(InvalidInputError):
        family_candidates(records[:2])
