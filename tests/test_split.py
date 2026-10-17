"""Pravidla splitu: pacienti po foldech, třídy stratifikace a dost pacientů ve třídě."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from mediparse.domain.diagnosis import Diagnosis
from mediparse.domain.input_note import InputNote
from mediparse.domain.inputs import InvalidInputError
from mediparse.domain.language import Language
from mediparse.domain.split import (
    RESIDUAL_STRATUM,
    Holdout,
    Split,
    SplitConfig,
    SubjectLeakError,
    ensure_patients_per_class,
    strata,
)

FOLDS = 2
GOLDEN_SHA256 = "fc0a6d02e61973f417172558b5ada2551fb8e6057c08699741f4a388273cb912"
NUMERIC_ORDER_SHA256 = (
    "91f9e8d3b27133625651802f23ad0fbbdb182b788cbf7058427385a493b5c85d"
)


def _notes(*subjects: int) -> list[InputNote]:
    return [
        InputNote(f"{subject}-DS-{index}", subject, Language.EN, "text", frozenset())
        for index, subject in enumerate(subjects)
    ]


def _labelled(*patients: tuple[int, set[Diagnosis]]) -> list[InputNote]:
    return [
        InputNote(
            f"{subject}-DS-{index}", subject, Language.EN, "text", frozenset(labels)
        )
        for index, (subject, labels) in enumerate(patients)
    ]


def test_disjoint_folds_pass() -> None:
    """Foldy bez společného pacienta tvoří platný split."""
    split = Split((frozenset({1, 2}), frozenset({3})))

    assert split.holdout(_notes(1, 2, 3), 1).test == tuple(_notes(1, 2, 3)[2:])


def test_shared_subject_is_a_leak() -> None:
    """Pacienti ve dvou foldech vyhodí SubjectLeakError s počtem, ne s identifikátory."""
    folds = (frozenset({1, 2, 987654}), frozenset({2, 3}), frozenset({987654, 4}))

    with pytest.raises(
        SubjectLeakError, match=r"^Pacientů ve více foldech: 2\.$"
    ) as raised:
        Split(folds)

    assert "987654" not in str(raised.value)


def test_holdout_splits_notes_by_subject_in_input_order() -> None:
    """Test jsou zprávy pacientů foldu, trénink ostatní; pořadí zůstává a fold se přenese."""
    notes = _notes(1, 2, 1, 3, 2)
    split = Split((frozenset({1, 3}), frozenset({2})))

    holdout = split.holdout(notes, 0)

    assert holdout == Holdout(
        fold=0,
        train=(notes[1], notes[4]),
        test=(notes[0], notes[2], notes[3]),
    )
    assert split.holdout(notes, 1).fold == 1
    assert split.holdout(notes, 1).test == (notes[1], notes[4])


@pytest.mark.parametrize("fold", [2, -1])
def test_fold_out_of_range_is_invalid_input(fold: int) -> None:
    """Fold mimo 0..počet-1 je neplatný vstup, záporný index nevybere poslední."""
    split = Split((frozenset({1}), frozenset({2})))

    with pytest.raises(
        InvalidInputError, match=rf"^Fold {fold} neexistuje, počet foldů splitu: 2\.$"
    ):
        split.holdout(_notes(1, 2), fold)


def test_note_of_subject_outside_split_is_invalid_input() -> None:
    """Zpráva pacienta mimo split nese v hlášce počet, ne identifikátor."""
    split = Split((frozenset({1}), frozenset({2})))

    with pytest.raises(
        InvalidInputError, match=r"^Zprávy pacientů mimo split: 2\.$"
    ) as raised:
        split.holdout(_notes(1, 2, 987654, 987654), 0)

    assert "987654" not in str(raised.value)


def test_enough_patients_in_each_class_pass() -> None:
    """Každá třída má aspoň tolik pacientů jako foldů."""
    ensure_patients_per_class(
        [True, True, False, False, False], [1, 2, 3, 4, 4], Diagnosis.CKD, FOLDS
    )


def test_class_without_enough_patients_is_invalid_input() -> None:
    """Zpráva jmenuje diagnózu a počty pacientů pozitivních i negativních."""
    with pytest.raises(InvalidInputError, match=r"ckd.*1 pozitivních.*3 negativních"):
        ensure_patients_per_class(
            [True, True, False, False, False], [1, 1, 2, 3, 4], Diagnosis.CKD, FOLDS
        )


def test_class_counts_patients_not_notes() -> None:
    """Pacient s více zprávami se počítá jednou."""
    with pytest.raises(InvalidInputError):
        ensure_patients_per_class(
            [True, True, False, False], [1, 1, 2, 3], Diagnosis.PNEUMONIA, FOLDS
        )


def _stratum_by_subject(notes: list[InputNote], folds: int) -> dict[int, int]:
    return {
        note.subject_id: stratum
        for note, stratum in zip(notes, strata(notes, folds), strict=True)
    }


def test_same_combination_shares_stratum_and_different_ones_do_not() -> None:
    """Pacienti se stejnou kombinací mají stejnou třídu, s různou různou; bez diagnóz je také třída."""
    notes = _labelled(
        (1, {Diagnosis.DIABETES}),
        (2, {Diagnosis.DIABETES}),
        (3, {Diagnosis.CKD}),
        (4, {Diagnosis.CKD}),
        (5, set()),
        (6, set()),
    )

    stratum = _stratum_by_subject(notes, FOLDS)

    assert stratum[1] == stratum[2]
    assert stratum[3] == stratum[4]
    assert stratum[5] == stratum[6]
    assert len({stratum[1], stratum[3], stratum[5]}) == 3  # ruff: ignore[magic-value-comparison]


def test_stratum_is_union_of_patient_labels() -> None:
    """Pacient se zprávami {diabetes} a {ckd} má obě zprávy ve třídě pacientů s oběma."""
    notes = _labelled(
        (1, {Diagnosis.DIABETES}),
        (1, {Diagnosis.CKD}),
        (2, {Diagnosis.DIABETES, Diagnosis.CKD}),
        (3, {Diagnosis.DIABETES}),
        (4, {Diagnosis.CKD}),
        (5, {Diagnosis.DIABETES}),
        (6, {Diagnosis.CKD}),
    )

    stratum = strata(notes, FOLDS)

    assert stratum[0] == stratum[1] == stratum[2]
    assert stratum[3] != stratum[0]
    assert stratum[4] != stratum[0]


def test_rare_combination_goes_to_residual_stratum() -> None:
    """Kombinace s méně pacienty než foldů tvoří zbytkovou třídu, pokud je jich dost."""
    notes = _labelled(
        (1, {Diagnosis.DIABETES}),
        (2, {Diagnosis.DIABETES}),
        (3, {Diagnosis.CKD}),
        (4, {Diagnosis.COPD}),
    )

    stratum = _stratum_by_subject(notes, FOLDS)

    assert stratum[1] == stratum[2] != RESIDUAL_STRATUM
    assert stratum[3] == stratum[4] == RESIDUAL_STRATUM


def test_small_residual_joins_most_common_combination() -> None:
    """Zbytek menší než foldy převezme nejčastější kombinaci."""
    notes = _labelled(
        (1, {Diagnosis.DIABETES}),
        (2, {Diagnosis.DIABETES}),
        (3, {Diagnosis.CKD}),
        (4, {Diagnosis.CKD}),
        (5, {Diagnosis.CKD}),
        (6, {Diagnosis.COPD}),
    )

    stratum = _stratum_by_subject(notes, FOLDS)

    assert stratum[6] == stratum[3] != stratum[1]


def test_small_residual_tie_joins_lower_mask() -> None:
    """Při shodě četností převezme zbytek kombinaci dřívější v pořadí Diagnosis."""
    notes = _labelled(
        (1, {Diagnosis.CKD}),
        (2, {Diagnosis.CKD}),
        (3, {Diagnosis.DIABETES}),
        (4, {Diagnosis.DIABETES}),
        (5, {Diagnosis.COPD}),
    )

    stratum = _stratum_by_subject(notes, FOLDS)

    assert stratum[5] == stratum[3] != stratum[1]


def test_strata_do_not_depend_on_note_order() -> None:
    """Permutace zpráv dá stejnou třídu u každé zprávy."""
    notes = _labelled(
        (1, {Diagnosis.CKD}),
        (2, {Diagnosis.CKD}),
        (3, {Diagnosis.DIABETES}),
        (4, {Diagnosis.DIABETES}),
        (5, {Diagnosis.COPD}),
        (5, set()),
        (6, set()),
    )
    reverse = notes[::-1]

    forward = dict(zip((n.note_id for n in notes), strata(notes, FOLDS), strict=True))
    backward = dict(
        zip((n.note_id for n in reverse), strata(reverse, FOLDS), strict=True)
    )

    assert forward == backward


def test_valid_split_config_is_parsed() -> None:
    """Platný JSON dá config se seedem a počtem foldů."""
    config = SplitConfig.model_validate_json('{"seed": 7, "folds": 5}')

    assert (config.seed, config.folds) == (7, 5)


@pytest.mark.parametrize(
    "body",
    [
        {"seed": 0, "folds": 1},
        {"seed": 0, "folds": 0},
        {"seed": 0, "folds": 2, "fold": 0},
        {"seed": 0, "folds": "5"},
        {"folds": 2},
        {"seed": -1, "folds": 2},
        {"seed": 2**32, "folds": 2},
    ],
    ids=[
        "one-fold",
        "zero-folds",
        "extra-field",
        "text-folds",
        "no-seed",
        "negative-seed",
        "seed-over-generator-range",
    ],
)
def test_invalid_split_config_is_rejected(body: dict[str, object]) -> None:
    """Méně než dva foldy, pole navíc, text místo čísla i chybějící seed config odmítnou."""
    with pytest.raises(ValidationError):
        SplitConfig.model_validate_json(json.dumps(body))


def test_sha256_matches_golden_value() -> None:
    """Otisk ukotvuje tvar řádků ``subject_id TAB fold LF``; stejný musí vyjít v S26 ze souboru."""
    split = Split((frozenset({1, 3}), frozenset({2})))

    assert split.sha256() == GOLDEN_SHA256


def test_sha256_changes_when_subject_moves_fold() -> None:
    """Přesun pacienta do jiného foldu otisk změní."""
    moved = Split((frozenset({1}), frozenset({2, 3})))

    assert moved.sha256() != GOLDEN_SHA256


def test_sha256_changes_when_folds_swap_order() -> None:
    """Číslo foldu je součástí přiřazení, takže prohození pořadí foldů otisk změní."""
    swapped = Split((frozenset({2}), frozenset({1, 3})))

    assert swapped.sha256() != GOLDEN_SHA256


def test_sha256_ignores_construction_order() -> None:
    """Stejné přiřazení z jiné konstrukce frozensetů dá týž otisk."""
    rebuilt = Split((frozenset([3, 1]), frozenset([2])))

    assert rebuilt.sha256() == GOLDEN_SHA256


def test_sha256_orders_subjects_numerically() -> None:
    """Pacient 10 je za pacientem 2, ne před ním jako při textovém řazení."""
    split = Split((frozenset({2}), frozenset({10})))

    assert split.sha256() == NUMERIC_ORDER_SHA256
