"""Výběr ukázky predikcí pro souhrn smoketestu."""

from mediparse.domain.smoketest_summary import demo_examples
from tests.model_support import prediction


def test_demo_examples_positive_before_negative_smallest_note_id() -> None:
    """Ukázka je nejmenší pozitivní a pak nejmenší negativní note_id."""
    items = [
        prediction("3-DS-1", truth=False, score=0.5),
        prediction("2-DS-1", truth=True, score=0.5),
        prediction("1-DS-1", truth=False, score=0.5),
        prediction("4-DS-1", truth=True, score=0.5),
    ]

    assert [item.note_id for item in demo_examples(items)] == ["2-DS-1", "1-DS-1"]


def test_demo_examples_missing_class_returns_one() -> None:
    """Chybí-li třída, vrátí se jen druhá."""
    items = [
        prediction("2-DS-1", truth=False, score=0.5),
        prediction("1-DS-1", truth=False, score=0.5),
    ]

    assert [item.note_id for item in demo_examples(items)] == ["1-DS-1"]


def test_demo_of_empty_predictions_is_empty() -> None:
    """Prázdný vstup dá prázdnou n-tici."""
    assert demo_examples([]) == ()
