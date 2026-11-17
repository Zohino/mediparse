"""Manifest ladění hlídá rozpočet, prostor i soulad předzpracování."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from mediparse.domain.clinical_text import TextNormalization
from mediparse.domain.inputs import InvalidInputError
from mediparse.domain.model_input import TextPreprocessing
from mediparse.domain.run_manifest import (
    TuningManifest,
    ensure_tuning_lineage,
)
from tests.tuning_support import SVM_PARAMS, tuning_manifest, tuning_manifest_fields


def _fields(**changes: object) -> dict[str, object]:
    return tuning_manifest_fields(**changes)


def test_tuning_manifest_roundtrips() -> None:
    """Povinné agregáty a provenance přežijí JSON beze změny."""
    manifest = TuningManifest.model_validate(_fields())
    assert TuningManifest.model_validate_json(manifest.model_dump_json()) == manifest


@pytest.mark.parametrize(
    "changes",
    [
        {"pruned_trials": 0},
        {"best": SVM_PARAMS | {"c": 1000.0}},
        {"nonconverged_trials": 4},
        {"completed_trials": -1, "pruned_trials": 4},
        {"best_macro_f1": 1.1},
        {"seconds": -1.0},
        {"text_preprocessing": TextPreprocessing.DICTIONARY},
        {"text_normalization": TextNormalization(acronyms={}, variants={})},
        {
            "text_preprocessing": TextPreprocessing.NORMALIZED,
            "text_normalization": TextNormalization(
                acronyms={}, variants={"kindey": "kidney"}
            ),
        },
    ],
)
def test_tuning_manifest_rejects_inconsistent_record(
    changes: dict[str, object],
) -> None:
    """Rozpočet 3 odmítne 2+0, nejlepší hodnoty mimo prostor i rozpornou normalizaci."""
    with pytest.raises(ValidationError):
        TuningManifest.model_validate(_fields(**changes))


def _lineage_manifest(**changes: object) -> TuningManifest:
    return tuning_manifest(
        best=None,
        **(
            {
                "text_preprocessing": "dictionary",
                "text_normalization": {"acronyms": {}, "variants": {}},
            }
            | changes
        ),
    )


def _ensure_lineage(tuning: TuningManifest) -> None:
    ensure_tuning_lineage(
        tuning,
        row_id="row",
        fold=0,
        dataset_sha256="b" * 64,
        split_sha256="c" * 64,
        diagnostic_strip=None,
        text_preprocessing=TextPreprocessing.DICTIONARY,
        text_normalization=TextNormalization(acronyms={}, variants={}),
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("row_id", "other"),
        ("fold", 1),
        ("dataset_sha256", "f" * 64),
        ("split_sha256", "f" * 64),
        (
            "diagnostic_strip",
            {"sections": ["diagnosis"], "lexicon_sha256": "f" * 64},
        ),
        ("text_preprocessing", TextPreprocessing.NORMALIZED),
        (
            "text_normalization",
            {"acronyms": {"DM": "diabetes"}, "variants": {}},
        ),
    ],
)
def test_tuning_lineage_refuses_each_mismatch(field: str, value: object) -> None:
    """Každá z jednotlivých sedmi neshod musí uvést přesně příslušné pole."""
    with pytest.raises(InvalidInputError) as error:
        _ensure_lineage(_lineage_manifest(**{field: value}))
    assert field in str(error.value)
    assert all(
        other not in str(error.value)
        for other in (
            "row_id",
            "fold",
            "dataset_sha256",
            "split_sha256",
            "diagnostic_strip",
            "text_preprocessing",
            "text_normalization",
        )
        if other != field
    )


def test_tuning_lineage_reports_all_mismatches() -> None:
    """Neshody se neukončí prvním polem a zpráva nevypisuje hodnoty."""
    with pytest.raises(InvalidInputError) as error:
        _ensure_lineage(_lineage_manifest(fold=1, row_id="other"))
    assert "fold" in str(error.value)
    assert "row_id" in str(error.value)
    assert "other" not in str(error.value)


def test_tuning_lineage_accepts_matching_inputs() -> None:
    """Shodný původ projde bez výjimky."""
    _ensure_lineage(_lineage_manifest())
