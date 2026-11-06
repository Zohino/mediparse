"""Striktní konfigurace variant pětice a její chyby."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from mediparse.domain.diagnosis import Diagnosis
from mediparse.domain.inputs import InvalidInputError
from mediparse.infrastructure.diagnosis_variants_file import load_diagnosis_variants
from tests.support import REPOSITORY_DIAGNOSIS_VARIANTS

if TYPE_CHECKING:
    from pathlib import Path


def test_repository_variants_load_sorted() -> None:
    """Repozitní config pokrývá pětici a zachová zadanou podobu variant."""
    variants = load_diagnosis_variants(REPOSITORY_DIAGNOSIS_VARIANTS)
    assert set(variants.values()) == set(Diagnosis)
    assert list(variants) == sorted(variants)
    assert variants["T2DM"] is Diagnosis.DIABETES
    assert variants["a fib"] is Diagnosis.ATRIAL_FIBRILLATION


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("ckd", None),
        ("other", ["other"]),
        ("ckd", []),
        ("ckd", [""]),
        ("ckd", [" "]),
        ("ckd", ["AFIB"]),
        ("ckd", ["CRI", "cri"]),
        ("ckd", [1]),
        ("ckd", "CRI"),
    ],
)
def test_invalid_variants_are_rejected(tmp_path: Path, key: str, value: object) -> None:
    """Chybějící, neznámé, prázdné a duplicitní varianty jsou chyba vstupu."""
    tree = json.loads(REPOSITORY_DIAGNOSIS_VARIANTS.read_text(encoding="utf-8"))
    if value is None:
        del tree["variants_en"][key]
    else:
        tree["variants_en"][key] = value
    path = tmp_path / "variants.json"
    path.write_text(json.dumps(tree), encoding="utf-8")
    with pytest.raises(InvalidInputError):
        load_diagnosis_variants(path)


def test_extra_top_level_field_is_rejected(tmp_path: Path) -> None:
    """Konfigurace odmítne i neznámé pole na nejvyšší úrovni."""
    tree = json.loads(REPOSITORY_DIAGNOSIS_VARIANTS.read_text(encoding="utf-8"))
    tree["other"] = 1
    path = tmp_path / "variants.json"
    path.write_text(json.dumps(tree), encoding="utf-8")
    with pytest.raises(InvalidInputError):
        load_diagnosis_variants(path)
