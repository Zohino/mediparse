"""Výběr ukázky predikcí pro souhrn smoketestu podle labelů kohorty."""

import pytest

from mediparse.domain.diagnosis import Diagnosis
from mediparse.domain.input_note import InputNote
from mediparse.domain.language import Language
from mediparse.synthetic.domain.smoketest_summary import demo_examples
from tests.model_support import prediction

NOTES = (
    InputNote("2-DS-1", 2, Language.EN, "text", frozenset({Diagnosis.DIABETES})),
    InputNote("4-DS-1", 4, Language.EN, "text", frozenset({Diagnosis.DIABETES})),
    InputNote("1-DS-1", 1, Language.EN, "text", frozenset({Diagnosis.CKD})),
    InputNote("3-DS-1", 3, Language.EN, "text", frozenset()),
)


def test_demo_examples_positive_before_negative_smallest_note_id() -> None:
    """Ukázka je nejmenší pozitivní a pak nejmenší negativní note_id podle labelů."""
    items = [
        prediction(name, score=0.5) for name in ("3-DS-1", "2-DS-1", "1-DS-1", "4-DS-1")
    ]

    assert [(item.note_id, truth) for item, truth in demo_examples(items, NOTES)] == [
        ("2-DS-1", True),
        ("1-DS-1", False),
    ]


@pytest.mark.parametrize(
    ("names", "expected"),
    [
        (("3-DS-1", "1-DS-1"), [("1-DS-1", False)]),
        (("4-DS-1", "2-DS-1"), [("2-DS-1", True)]),
    ],
)
def test_demo_examples_missing_class_returns_one(
    names: tuple[str, ...], expected: list[tuple[str, bool]]
) -> None:
    """Chybí-li třída, vrátí se jen druhá."""
    items = [prediction(name, score=0.5) for name in names]

    assert [
        (item.note_id, truth) for item, truth in demo_examples(items, NOTES)
    ] == expected


def test_demo_of_empty_predictions_is_empty() -> None:
    """Prázdný vstup dá prázdnou n-tici."""
    assert demo_examples([], NOTES) == ()
