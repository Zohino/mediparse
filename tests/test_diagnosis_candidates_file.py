"""Načtení předpon pětice z configu kandidátů."""

from __future__ import annotations

import json
from functools import partial
from typing import TYPE_CHECKING, Any, Final

import duckdb
import pytest

from mediparse.domain.diagnosis import Diagnosis
from mediparse.domain.icd import IcdCode, IcdVersion
from mediparse.domain.inputs import InvalidInputError
from mediparse.infrastructure.diagnosis_candidates_file import (
    load_diagnosis_codes,
    load_diagnosis_variants,
    load_text_normalization,
)
from tests.cohort_support import candidate_prefixes
from tests.support import REPOSITORY_DIAGNOSIS_CANDIDATES

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


def _candidates(skip: str | None = None) -> dict[str, Any]:
    candidates: dict[str, Any] = {
        key: {"label_cs": key, "charlson": ["x"], **prefixes}
        for key, prefixes in candidate_prefixes().items()
        if key != skip
    }
    candidates["sepsis"] = {"icd9_prefixes": ["038"], "icd10_prefixes": ["A41"]}
    return {"candidates": candidates}


def _write(tmp_path: Path, tree: dict[str, Any]) -> Path:
    path = tmp_path / "candidates.json"
    path.write_text(json.dumps(tree), encoding="utf-8")
    return path


def test_loads_five_and_ignores_other_candidates(tmp_path: Path) -> None:
    """Pětice se načte a kandidát navíc s dalšími poli nevadí."""
    codes = load_diagnosis_codes(_write(tmp_path, _candidates()))

    assert codes.diagnosis_of(IcdCode(version=IcdVersion.TEN, code="J449")) is (
        Diagnosis.COPD
    )
    assert codes.diagnosis_of(IcdCode(version=IcdVersion.TEN, code="A419")) is None


def test_missing_diagnosis_is_invalid_input(tmp_path: Path) -> None:
    """Chybějící diagnóza pětice je chyba vstupu."""
    with pytest.raises(InvalidInputError, match="Chybí diagnózy: ckd"):
        load_diagnosis_codes(_write(tmp_path, _candidates(skip="ckd")))


def test_missing_file_is_invalid_input(tmp_path: Path) -> None:
    """Chybějící soubor je chyba vstupu."""
    with pytest.raises(InvalidInputError):
        load_diagnosis_codes(tmp_path / "none.json")


def test_invalid_json_is_invalid_input(tmp_path: Path) -> None:
    """Neplatný JSON je chyba vstupu."""
    path = tmp_path / "candidates.json"
    path.write_text("{", encoding="utf-8")

    with pytest.raises(InvalidInputError):
        load_diagnosis_codes(path)


def test_repository_config_loads() -> None:
    """Config kandidátů v repu se načte."""
    codes = load_diagnosis_codes(REPOSITORY_DIAGNOSIS_CANDIDATES)

    assert set(codes.prefixes) == set(Diagnosis)


_NAMES: dict[str, dict[str, list[str]]] = {
    "diabetes": {"keywords_en": ["diabetes", "diabetic"], "acronyms_en": ["DM", "DM2"]},
    "copd": {
        "keywords_en": ["chronic obstructive pulmonary disease"],
        "acronyms_en": ["COPD"],
    },
    "ckd": {
        "keywords_en": ["chronic kidney disease", "end-stage renal disease"],
        "acronyms_en": ["CKD", "ESRD"],
    },
    "pneumonia": {"keywords_en": ["pneumonia"], "acronyms_en": []},
    "atrial_fibrillation": {
        "keywords_en": ["atrial fibrillation", "afib"],
        "acronyms_en": ["AF"],
    },
}


def _named(**changes: dict[str, list[str]] | None) -> dict[str, Any]:
    names = {**_NAMES, "heart_failure": {"keywords_en": ["x"], "acronyms_en": ["MI"]}}
    for key, value in changes.items():
        if value is None:
            del names[key]
        else:
            names[key] = value
    return {"candidates": names}


def test_repository_config_gives_exactly_the_five_acronym_pairs() -> None:
    """Reálný config dá šest dvojic pětice a zkratky ostatních kandidátů vynechá."""
    normalization = load_text_normalization(
        REPOSITORY_DIAGNOSIS_CANDIDATES, variants=False
    )

    assert normalization.acronyms == {
        "AF": "atrial fibrillation",
        "CKD": "chronic kidney disease",
        "COPD": "chronic obstructive pulmonary disease",
        "DM": "diabetes",
        "DM2": "diabetes",
        "ESRD": "chronic kidney disease",
    }


def test_canonical_form_is_first_keyword_and_other_candidates_are_ignored(
    tmp_path: Path,
) -> None:
    """Tvar je první klíčové slovo; zkratka ostatních kandidátů se nenačte."""
    normalization = load_text_normalization(_write(tmp_path, _named()), variants=False)

    assert normalization.acronyms["ESRD"] == "chronic kidney disease"
    assert "MI" not in normalization.acronyms


def test_missing_diagnosis_in_names_is_invalid_input(tmp_path: Path) -> None:
    """Chybějící diagnóza pětice je chyba vstupu."""
    with pytest.raises(InvalidInputError, match="Chybí diagnózy: ckd"):
        load_text_normalization(_write(tmp_path, _named(ckd=None)), variants=False)


def test_acronym_without_keywords_is_invalid_input(tmp_path: Path) -> None:
    """Zkratka bez klíčových slov nemá kanonický tvar."""
    tree = _named(copd={"keywords_en": [], "acronyms_en": ["COPD"]})

    with pytest.raises(InvalidInputError, match="keywords_en"):
        load_text_normalization(_write(tmp_path, tree), variants=False)


def test_acronym_of_two_diagnoses_is_invalid_input(tmp_path: Path) -> None:
    """Zkratka u dvou diagnóz by tiše přepsala slovník."""
    tree = _named(pneumonia={"keywords_en": ["pneumonia"], "acronyms_en": ["AF"]})

    with pytest.raises(InvalidInputError, match="u více diagnóz"):
        load_text_normalization(_write(tmp_path, tree), variants=False)


def test_missing_acronyms_field_is_invalid_input(tmp_path: Path) -> None:
    """Pole ``acronyms_en`` je povinné, chybějící se nedoplní prázdným výchozím."""
    tree = _named(copd={"keywords_en": ["chronic obstructive pulmonary disease"]})

    with pytest.raises(InvalidInputError, match="acronyms_en"):
        load_text_normalization(_write(tmp_path, tree), variants=False)


def test_candidate_without_names_and_acronyms_is_accepted(tmp_path: Path) -> None:
    """Diagnóza bez klíčových slov i zkratek slovník nezměří, ale nevadí."""
    tree = _named(pneumonia={"keywords_en": [], "acronyms_en": []})

    normalization = load_text_normalization(_write(tmp_path, tree), variants=False)

    assert "AF" in normalization.acronyms


def test_names_missing_file_and_invalid_json_are_invalid_input(
    tmp_path: Path,
) -> None:
    """Chybějící soubor a neplatný JSON jsou chyba vstupu."""
    broken = tmp_path / "broken.json"
    broken.write_text("{", encoding="utf-8")

    with pytest.raises(InvalidInputError):
        load_text_normalization(tmp_path / "none.json", variants=False)
    with pytest.raises(InvalidInputError):
        load_text_normalization(broken, variants=False)


_VARIANT_COUNT: Final = 15
_NORMALIZATION_VARIANT_COUNT: Final = 43


def test_repository_variants_load_sorted() -> None:
    """Repozitní config dá patnáct původních variant pětice včetně afib z keywords."""
    variants = load_diagnosis_variants(REPOSITORY_DIAGNOSIS_CANDIDATES)
    assert len(variants) == _VARIANT_COUNT
    assert variants == {
        "CRF": Diagnosis.CKD,
        "CRI": Diagnosis.CKD,
        "DM1": Diagnosis.DIABETES,
        "DMII": Diagnosis.DIABETES,
        "ESKD": Diagnosis.CKD,
        "IDDM": Diagnosis.DIABETES,
        "NIDDM": Diagnosis.DIABETES,
        "PAF": Diagnosis.ATRIAL_FIBRILLATION,
        "PNA": Diagnosis.PNEUMONIA,
        "T1DM": Diagnosis.DIABETES,
        "T2DM": Diagnosis.DIABETES,
        "a fib": Diagnosis.ATRIAL_FIBRILLATION,
        "a-fib": Diagnosis.ATRIAL_FIBRILLATION,
        "afib": Diagnosis.ATRIAL_FIBRILLATION,
        "chronic obstructive lung disease": Diagnosis.COPD,
    }
    assert list(variants) == sorted(variants)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("ckd", None),
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
    """Chybějící, prázdné, netypované a duplicitní varianty jsou chyba vstupu."""
    tree = json.loads(REPOSITORY_DIAGNOSIS_CANDIDATES.read_text(encoding="utf-8"))
    if value is None:
        del tree["candidates"][key]
    else:
        tree["candidates"][key]["variants_en"] = value
    with pytest.raises(InvalidInputError):
        load_diagnosis_variants(_write(tmp_path, tree))


def test_missing_variants_field_is_rejected(tmp_path: Path) -> None:
    """Diagnóza bez pole variant není úplná konfigurace měření."""
    tree = json.loads(REPOSITORY_DIAGNOSIS_CANDIDATES.read_text(encoding="utf-8"))
    del tree["candidates"]["ckd"]["variants_en"]
    with pytest.raises(InvalidInputError, match="variants_en"):
        load_diagnosis_variants(_write(tmp_path, tree))


@pytest.mark.parametrize("key", ["diabetes", "ckd"])
def test_variant_colliding_with_acronym_is_rejected(tmp_path: Path, key: str) -> None:
    """Varianta dm nesmí zdvojit DM ani u vlastní, ani u jiné diagnózy pětice."""
    tree = json.loads(REPOSITORY_DIAGNOSIS_CANDIDATES.read_text(encoding="utf-8"))
    tree["candidates"][key]["variants_en"] = ["dm"]
    with pytest.raises(InvalidInputError, match="zkrat"):
        load_diagnosis_variants(_write(tmp_path, tree))


def test_variant_loader_ignores_other_candidates_and_fields(tmp_path: Path) -> None:
    """Loader čte jen varianty a zkratky pětice z podmnožiny sdíleného configu."""
    tree = json.loads(REPOSITORY_DIAGNOSIS_CANDIDATES.read_text(encoding="utf-8"))
    tree["other"] = 1
    tree["candidates"]["other"] = {"variants_en": ["dm"]}
    tree["candidates"]["ckd"]["variants_en"] = ["CHF"]
    variants = load_diagnosis_variants(_write(tmp_path, tree))
    assert variants["CHF"] is Diagnosis.CKD
    assert "dm" not in variants


def test_other_loaders_ignore_variants(tmp_path: Path) -> None:
    """Přidání variant nemění předpony ICD ani slovník zkratek normalizace."""
    tree = json.loads(REPOSITORY_DIAGNOSIS_CANDIDATES.read_text(encoding="utf-8"))
    selected = {
        key for key, entry in tree["candidates"].items() if "variants_en" in entry
    }
    assert selected == set(Diagnosis)
    for entry in tree["candidates"].values():
        entry.pop("variants_en", None)
        entry.pop("misspellings_en", None)
    path = _write(tmp_path, tree)
    assert load_text_normalization(
        REPOSITORY_DIAGNOSIS_CANDIDATES, variants=False
    ) == load_text_normalization(path, variants=False)
    assert load_diagnosis_codes(
        REPOSITORY_DIAGNOSIS_CANDIDATES
    ) == load_diagnosis_codes(path)


def test_repository_normalization_includes_only_approved_forms() -> None:
    """Patnáct variant a 28 překlepů odpovídá ručně schválenému slovníku."""
    enabled = load_text_normalization(REPOSITORY_DIAGNOSIS_CANDIDATES, variants=True)
    disabled = load_text_normalization(REPOSITORY_DIAGNOSIS_CANDIDATES, variants=False)
    assert len(enabled.variants) == _NORMALIZATION_VARIANT_COUNT
    for form, word in {
        "T2DM": "diabetes",
        "CRF": "chronic kidney disease",
        "chronic obstructive lung disease": "chronic obstructive pulmonary disease",
        "kindey": "kidney",
        "pumonary": "pulmonary",
        "mellitis": "mellitus",
        "atrail": "atrial",
    }.items():
        assert enabled.variants[form] == word
    assert not set(enabled.variants).intersection({
        "pneumoniae",
        "pneumonic",
        "atria",
        "flatter",
        "clutter",
        "pneumonias",
        "diabetics",
        "fibrillations",
        "flutters",
        "aterial",
        "arial",
        "atial",
    })
    assert disabled.variants == {}
    assert disabled.acronyms == enabled.acronyms


@pytest.mark.parametrize(
    "misspellings", [None, {"kindey": "liver"}, {"CRF": "kidney"}, {"kindey": 1}, []]
)
def test_invalid_misspellings_are_rejected(
    tmp_path: Path, misspellings: object
) -> None:
    """Cíl patří do slov keywords téže diagnózy; pole je povinné, striktní a bez kolizí."""
    tree = json.loads(REPOSITORY_DIAGNOSIS_CANDIDATES.read_text(encoding="utf-8"))
    if misspellings is None:
        del tree["candidates"]["ckd"]["misspellings_en"]
    else:
        tree["candidates"]["ckd"]["misspellings_en"] = misspellings
    with pytest.raises(InvalidInputError):
        load_text_normalization(_write(tmp_path, tree), variants=True)


def test_duplicate_variant_before_dictionary_construction_is_rejected(
    tmp_path: Path,
) -> None:
    """Přesná duplicita mezi diagnózami se nesmí ztratit konstrukcí dict."""
    tree = json.loads(REPOSITORY_DIAGNOSIS_CANDIDATES.read_text(encoding="utf-8"))
    tree["candidates"]["ckd"]["variants_en"] = ["T2DM"]
    with pytest.raises(InvalidInputError, match="Duplicitní"):
        load_text_normalization(_write(tmp_path, tree), variants=True)


def test_variant_loader_rejects_equivalent_whitespace_forms(tmp_path: Path) -> None:
    """Dvě různá rozložení bílých znaků nesmějí vytvořit kolidující varianty."""
    tree = json.loads(REPOSITORY_DIAGNOSIS_CANDIDATES.read_text(encoding="utf-8"))
    tree["candidates"]["atrial_fibrillation"]["variants_en"] = ["a fib", "a  fib"]
    with pytest.raises(InvalidInputError):
        load_diagnosis_variants(_write(tmp_path, tree))


def test_misspelling_keys_are_stripped_at_config_boundary(tmp_path: Path) -> None:
    """Okrajové mezery klíče překlepu odstraní striktní loader před doménou."""
    tree = json.loads(REPOSITORY_DIAGNOSIS_CANDIDATES.read_text(encoding="utf-8"))
    tree["candidates"]["ckd"]["misspellings_en"] = {" kindey ": "kidney"}
    normalization = load_text_normalization(_write(tmp_path, tree), variants=True)
    assert normalization.variants["kindey"] == "kidney"
    assert " kindey " not in normalization.variants


@pytest.mark.parametrize(
    "loader",
    [load_diagnosis_variants, partial(load_text_normalization, variants=True)],
    ids=["measurement", "normalization"],
)
def test_variant_loaders_reject_empty_keywords_without_acronyms(
    tmp_path: Path, loader: Callable[[Path], object]
) -> None:
    """Varianta potřebuje kanonický název i u diagnózy bez zkratek."""
    tree = json.loads(REPOSITORY_DIAGNOSIS_CANDIDATES.read_text(encoding="utf-8"))
    tree["candidates"]["pneumonia"]["keywords_en"] = []
    tree["candidates"]["pneumonia"]["misspellings_en"] = {}
    with pytest.raises(InvalidInputError):
        loader(_write(tmp_path, tree))


def test_repository_misspellings_are_one_edit_from_their_targets() -> None:
    """Každá schválená dvojice splňuje nezávislou definici blízkého tvaru S37a."""
    tree = json.loads(REPOSITORY_DIAGNOSIS_CANDIDATES.read_text(encoding="utf-8"))
    with duckdb.connect() as connection:
        for diagnosis in Diagnosis:
            pairs = tree["candidates"][diagnosis.value]["misspellings_en"]
            for form, target in pairs.items():
                distance = connection.execute(
                    "SELECT damerau_levenshtein(?, ?)", [form, target]
                ).fetchone()
                assert distance == (1,), (diagnosis, form, target, distance)
