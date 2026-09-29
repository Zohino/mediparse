"""Labely diagnóz: schéma, společné rozdělení s maximální entropií a podmíněné AKI."""

from itertools import combinations
from math import isclose, prod

import pytest
from pydantic import ValidationError

from mediparse.domain.labels import (
    Diagnosis,
    IpfNotConvergedError,
    LabelModel,
    acute_given_chronic,
    label_sets,
    max_entropy_joint,
)

PREVALENCE = {
    Diagnosis.DIABETES: 0.30,
    Diagnosis.CKD: 0.20,
    Diagnosis.HEART_FAILURE: 0.15,
    Diagnosis.ATRIAL_FIBRILLATION: 0.10,
    Diagnosis.AKI: 0.05,
}
CELLS = 2 ** len(Diagnosis)
CHRONIC_STATES = 2 ** sum(d.chronic for d in Diagnosis)
CORRELATED = {
    (Diagnosis.DIABETES, Diagnosis.CKD): 0.10,
    (Diagnosis.CKD, Diagnosis.HEART_FAILURE): 0.06,
    (Diagnosis.HEART_FAILURE, Diagnosis.ATRIAL_FIBRILLATION): 0.05,
    (Diagnosis.CKD, Diagnosis.AKI): 0.03,
}


def _model(
    pairs: dict[tuple[Diagnosis, Diagnosis], float] | None = None, passes: int = 500
) -> LabelModel:
    pairs = pairs or {}
    joint = {
        frozenset(pair): pairs.get(pair, PREVALENCE[pair[0]] * PREVALENCE[pair[1]])
        for pair in combinations(Diagnosis, 2)
    }
    conditional = {
        first: {
            second: 1.0
            if first is second
            else joint[frozenset((first, second))] / PREVALENCE[first]
            for second in Diagnosis
        }
        for first in Diagnosis
    }
    return LabelModel(
        prevalence=PREVALENCE,
        conditional=conditional,
        prevalence_tolerance=0.02,
        ipf_tolerance=1e-12,
        ipf_max_passes=passes,
    )


def _marginal(joint: dict[frozenset[Diagnosis], float], *diagnoses: Diagnosis) -> float:
    return sum(p for cell, p in joint.items() if set(diagnoses) <= cell)


def test_label_sets_cover_all_combinations_once() -> None:
    """Pět binárních labelů dává 32 různých kombinací včetně žádné diagnózy."""
    cells = label_sets()

    assert len(set(cells)) == CELLS
    assert frozenset() in cells


def test_joint_matches_prevalences_and_pairs() -> None:
    """Rozdělení splní všechny prevalence i párové výskyty a sečte se na jedna."""
    model = _model(CORRELATED)

    joint = max_entropy_joint(model)

    assert isclose(sum(joint.values()), 1.0, abs_tol=1e-9)
    for diagnosis, prevalence in PREVALENCE.items():
        assert isclose(_marginal(joint, diagnosis), prevalence, abs_tol=1e-9)
    for first, second in combinations(Diagnosis, 2):
        expected = model.co_occurrence(first, second)
        assert isclose(_marginal(joint, first, second), expected, abs_tol=1e-9)


def test_independent_pairs_give_product_distribution() -> None:
    """Bez korelací je rozdělením s maximální entropií součin nezávislých labelů."""
    joint = max_entropy_joint(_model())

    for cell, probability in joint.items():
        expected = prod(
            PREVALENCE[d] if d in cell else 1 - PREVALENCE[d] for d in Diagnosis
        )
        assert isclose(probability, expected, abs_tol=1e-9)


def test_ipf_without_enough_passes_fails() -> None:
    """Nekonvergované rozdělení se nevrací potichu."""
    with pytest.raises(IpfNotConvergedError):
        max_entropy_joint(_model(CORRELATED, passes=1))


def test_acute_given_chronic_is_a_probability_per_chronic_state() -> None:
    """Pro každou kombinaci chronických diagnóz vznikne pravděpodobnost AKI."""
    conditional = acute_given_chronic(max_entropy_joint(_model(CORRELATED)))

    assert len(conditional) == CHRONIC_STATES
    assert all(0.0 <= p <= 1.0 for p in conditional.values())
    assert all(Diagnosis.AKI not in chronic for chronic in conditional)


def test_inconsistent_conditionals_are_rejected() -> None:
    """Matice, jejíž dva směry dávají jiný společný výskyt, neprojde schématem."""
    valid = _model(CORRELATED).model_dump()
    valid["conditional"][Diagnosis.DIABETES][Diagnosis.CKD] = 0.9

    with pytest.raises(ValidationError, match="obou směrů"):
        LabelModel.model_validate(valid)


def test_missing_diagnosis_is_rejected() -> None:
    """Config bez některé z pěti diagnóz neprojde."""
    valid = _model().model_dump()
    del valid["prevalence"][Diagnosis.AKI]

    with pytest.raises(ValidationError, match="pět diagnóz"):
        LabelModel.model_validate(valid)
