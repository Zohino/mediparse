"""Kód ICD a jeho přiřazení k diagnózám pětice podle předpon."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from mediparse.domain.diagnosis import Diagnosis
from mediparse.domain.icd import DiagnosisCodes, IcdCode, IcdVersion

NINE = IcdVersion.NINE
TEN = IcdVersion.TEN


def _prefixes(
    **overrides: dict[IcdVersion, tuple[str, ...]],
) -> dict[Diagnosis, dict[IcdVersion, tuple[str, ...]]]:
    base: dict[Diagnosis, dict[IcdVersion, tuple[str, ...]]] = {
        Diagnosis.DIABETES: {NINE: ("250",), TEN: ("E11",)},
        Diagnosis.COPD: {NINE: ("496",), TEN: ("J44",)},
        Diagnosis.CKD: {NINE: ("585",), TEN: ("N18",)},
        Diagnosis.PNEUMONIA: {NINE: ("486",), TEN: ("J18",)},
        Diagnosis.ATRIAL_FIBRILLATION: {NINE: ("4273",), TEN: ("I48",)},
    }
    base.update({Diagnosis(key): value for key, value in overrides.items()})
    return base


def test_code_is_normalized() -> None:
    """Mezery a malá písmena zmizí, aby kód odpovídal zápisu v MIMIC."""
    assert IcdCode(version=TEN, code=" e11 ").code == "E11"


@pytest.mark.parametrize("code", ["", "   ", "E11.9", "E11-9", "É11"])
def test_invalid_code_is_rejected(code: str) -> None:
    """Prázdný kód, tečka a jiné znaky jsou chyba vstupu."""
    with pytest.raises(ValidationError):
        IcdCode(version=TEN, code=code)


def test_diagnosis_of_matches_group_none_and_other_version() -> None:
    """Kód jedné skupiny dostane diagnózu, cizí kód ne a předpona platí jen ve své verzi."""
    codes = DiagnosisCodes(prefixes=_prefixes())

    assert codes.diagnosis_of(IcdCode(version=TEN, code="E119")) is Diagnosis.DIABETES
    assert codes.diagnosis_of(IcdCode(version=TEN, code="Z000")) is None
    assert codes.diagnosis_of(IcdCode(version=TEN, code="250")) is None
    assert codes.diagnosis_of(IcdCode(version=NINE, code="25000")) is (
        Diagnosis.DIABETES
    )


def test_four_character_prefix_is_exact() -> None:
    """Čtyřznaková předpona přijme 42731 a odmítne 4270."""
    codes = DiagnosisCodes(prefixes=_prefixes())

    assert codes.diagnosis_of(IcdCode(version=NINE, code="42731")) is (
        Diagnosis.ATRIAL_FIBRILLATION
    )
    assert codes.diagnosis_of(IcdCode(version=NINE, code="4270")) is None


def test_nested_prefixes_of_two_diagnoses_are_rejected() -> None:
    """Dvě diagnózy s předponami E11 a E1 v téže verzi: kód by padl do obou skupin."""
    with pytest.raises(ValidationError, match=r"diabetes.*copd|copd.*diabetes"):
        DiagnosisCodes(prefixes=_prefixes(copd={NINE: ("496",), TEN: ("E1",)}))


def test_equal_prefix_in_two_diagnoses_is_rejected() -> None:
    """Stejná předpona ve dvou diagnózách je chyba."""
    with pytest.raises(ValidationError, match="E11"):
        DiagnosisCodes(prefixes=_prefixes(copd={NINE: ("496",), TEN: ("E11",)}))


def test_same_prefix_in_different_versions_is_valid() -> None:
    """E11 ve verzi 9 u jedné diagnózy a ve verzi 10 u druhé se nepřekrývá."""
    codes = DiagnosisCodes(
        prefixes=_prefixes(
            copd={NINE: ("E11",), TEN: ("J44",)},
        )
    )

    assert codes.diagnosis_of(IcdCode(version=NINE, code="E11")) is Diagnosis.COPD
    assert codes.diagnosis_of(IcdCode(version=TEN, code="E11")) is Diagnosis.DIABETES


def test_missing_diagnosis_is_rejected() -> None:
    """Chybějící diagnóza pětice je chyba."""
    prefixes = _prefixes()
    del prefixes[Diagnosis.CKD]

    with pytest.raises(ValidationError, match="ckd"):
        DiagnosisCodes(prefixes=prefixes)


def test_diagnosis_without_prefixes_is_rejected() -> None:
    """Diagnóza bez jediné předpony je chyba."""
    with pytest.raises(ValidationError, match="ckd"):
        DiagnosisCodes(prefixes=_prefixes(ckd={NINE: (), TEN: ()}))


def test_invalid_prefix_is_rejected() -> None:
    """Předpona s tečkou není kód."""
    with pytest.raises(ValidationError):
        DiagnosisCodes(prefixes=_prefixes(ckd={NINE: ("585",), TEN: ("N18.",)}))


@pytest.mark.parametrize("prefix", ["e11", " E11", "E11 "])
def test_unnormalized_prefix_is_rejected(prefix: str) -> None:
    """Předpona s malými písmeny nebo mezerou by se v EDA neshodovala, proto je chyba."""
    with pytest.raises(ValidationError, match="normaliz"):
        DiagnosisCodes(prefixes=_prefixes(ckd={NINE: ("585",), TEN: (prefix,)}))
