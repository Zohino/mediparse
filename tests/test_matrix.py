"""Schéma experimentální matice: platný manifest projde, porušení invariantů spadne."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
from pydantic import ValidationError

from mediparse.domain.inputs import InvalidInputError
from mediparse.domain.language import Language
from mediparse.domain.matrix import ExperimentMatrix, ModelName, Tier
from mediparse.infrastructure.matrix_manifest import load_matrix
from tests.support import REPOSITORY_MATRIX

if TYPE_CHECKING:
    from pathlib import Path


def _row(**overrides: object) -> dict[str, object]:
    return {
        "id": "en-logreg",
        "language": "en",
        "model": "logreg",
        "strip_diagnostic_sections": False,
        "tuning_trials": 20,
    } | overrides


def _validate(*rows: dict[str, object]) -> ExperimentMatrix:
    return ExperimentMatrix.model_validate_json(json.dumps({"rows": rows}))


def test_single_row_matrix_is_valid() -> None:
    """Jeden úplný řádek projde a hodnoty os se převedou na výčty."""
    (row,) = _validate(_row()).rows

    assert row.language is Language.EN
    assert row.model is ModelName.LOGREG


@pytest.mark.parametrize(
    ("model", "tier"),
    [
        (ModelName.LOGREG, Tier.CLASSICAL),
        (ModelName.SVM, Tier.CLASSICAL),
        (ModelName.FASTTEXT, Tier.INTERMEDIATE),
        (ModelName.XLM_R_BASE, Tier.TRANSFORMER),
        (ModelName.MDEBERTA_V3_BASE, Tier.TRANSFORMER),
    ],
)
def test_model_tier_is_derived(model: ModelName, tier: Tier) -> None:
    """Stupeň se odvozuje z modelu, v manifestu se neuvádí."""
    assert model.tier is tier


def test_every_model_has_tier() -> None:
    """Nový model bez přiřazeného stupně shodí test, ne až běh pipeline."""
    assert all(model.tier in Tier for model in ModelName)


@pytest.mark.parametrize(
    "overrides",
    [
        {"language": "de"},
        {"model": "random-forest"},
        {"tuning_trials": 0},
        {"tuning_trials": "20"},
        {"strip_diagnostic_sections": "true"},
        {"id": "EN logreg"},
        {"tier": "classical"},
    ],
    ids=[
        "unknown-language",
        "unknown-model",
        "zero-trials",
        "string-trials",
        "string-bool",
        "bad-id",
        "extra-field",
    ],
)
def test_invalid_row_is_rejected(overrides: dict[str, object]) -> None:
    """Neznámé hodnoty os, špatné typy, špatné id i pole navíc spadnou."""
    with pytest.raises(ValidationError):
        _validate(_row(**overrides))


def test_diagnosis_is_not_matrix_axis() -> None:
    """Diagnóza je metriková osa: jeden model nese všech pět výstupů, řádek ji nenese."""
    with pytest.raises(ValidationError, match="diagnosis"):
        _validate(_row(diagnosis="diabetes"))


def test_missing_axis_is_rejected() -> None:
    """Každá osa je povinná, manifest nesmí spoléhat na výchozí hodnoty."""
    row = _row()
    del row["strip_diagnostic_sections"]

    with pytest.raises(ValidationError):
        _validate(row)


def test_empty_matrix_is_rejected() -> None:
    """Matice bez řádků nic nedeklaruje."""
    with pytest.raises(ValidationError):
        _validate()


def test_duplicate_ids_are_rejected() -> None:
    """Id řádku je klíč výsledků, dva řádky ho sdílet nesmí."""
    with pytest.raises(ValidationError, match=r"Duplicitní id řádků: \['en-logreg'\]"):
        _validate(_row(), _row(language="cs"))


def test_duplicate_experiments_are_rejected() -> None:
    """Dva řádky se stejnými osami jsou tentýž experiment pod dvěma jmény."""
    with pytest.raises(ValidationError, match=r"stejné osy: \[\("):
        _validate(_row(), _row(id="en-logreg-copy"))


def test_models_of_same_tier_are_distinct_experiments() -> None:
    """LR a SVM jsou oba klasické, ale jsou to dva různé řádky."""
    matrix = _validate(_row(), _row(id="en-svm", model="svm"))

    assert {row.model.tier for row in matrix.rows} == {Tier.CLASSICAL}


def test_unequal_tuning_budget_is_rejected() -> None:
    """Sedmý invariant: porovnávané řádky mají stejný rozpočet ladění."""
    transformer = _row(id="en-xlm-r", model="xlm-r-base", tuning_trials=50)

    with pytest.raises(ValidationError, match="rozpočet"):
        _validate(_row(), transformer)


def test_load_matrix_reads_json(tmp_path: Path) -> None:
    """Loader přečte JSON ze souboru a předá ho validaci."""
    manifest = tmp_path / "matrix.json"
    row = _row(id="cs-fasttext", language="cs", model="fasttext")
    manifest.write_text(json.dumps({"rows": [row]}), encoding="utf-8")

    (loaded,) = load_matrix(manifest).rows

    assert loaded.model.tier is Tier.INTERMEDIATE


def test_load_matrix_rejects_invalid_content(tmp_path: Path) -> None:
    """Neplatný obsah souboru spadne na validaci jako doménová chyba vstupu se jménem souboru."""
    manifest = tmp_path / "matrix.json"
    manifest.write_text('{"rows": [{"id": "en-logreg"}]}', encoding="utf-8")

    with pytest.raises(InvalidInputError, match=r"matrix\.json"):
        load_matrix(manifest)


def test_czech_row_with_strip_is_rejected_with_row_id() -> None:
    """Lexikon je anglický, strip české zprávy by nic neodstranil; hláška nese id řádku."""
    row = _row(id="cs-logreg-strip", language="cs", strip_diagnostic_sections=True)

    with pytest.raises(ValidationError) as error:
        _validate(row)

    assert error.value.errors()[0]["msg"].startswith(
        "Value error, Řádek cs-logreg-strip: strip diagnostických sekcí"
    )


def test_czech_row_without_strip_is_valid() -> None:
    """Česká zpráva bez stripu je hlavní řádek, projde."""
    (row,) = _validate(_row(id="cs-logreg", language="cs")).rows

    assert row.language is Language.CS


def test_repository_manifest_declares_main_and_ablation_rows() -> None:
    """Hlavní řádek je bez stripu, ablační se stripem a ostatní osy jsou stejné."""
    rows = {row.id: row for row in load_matrix(REPOSITORY_MATRIX).rows}

    main, ablation = rows["en-logreg"], rows["en-logreg-strip"]
    assert main.strip_diagnostic_sections is False
    assert ablation.strip_diagnostic_sections is True
    assert (main.language, main.model, main.tuning_trials) == (
        ablation.language,
        ablation.model,
        ablation.tuning_trials,
    )


def test_repository_manifest_is_valid() -> None:
    """Manifest v repu je platný — neplatná matice neprojde CI."""
    assert load_matrix(REPOSITORY_MATRIX).rows
