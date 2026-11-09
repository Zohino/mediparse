"""Schéma experimentální matice: platný manifest projde, porušení invariantů spadne."""

from __future__ import annotations

import json
from itertools import product
from typing import TYPE_CHECKING

import pytest
from pydantic import ValidationError

from mediparse.domain.inputs import InvalidInputError
from mediparse.domain.language import Language
from mediparse.domain.matrix import ExperimentMatrix, MatrixRow, ModelName, Tier
from mediparse.domain.model_input import TextPreprocessing
from mediparse.infrastructure.matrix_manifest import load_matrix
from tests.support import REPOSITORY_MATRIX

if TYPE_CHECKING:
    from pathlib import Path


def _row(**overrides: object) -> dict[str, object]:
    return {
        "id": "en-logreg",
        "language": "en",
        "model": "logreg",
        "input": {"strip_diagnostic_sections": False, "text_preprocessing": "none"},
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
        {"input": {"strip_diagnostic_sections": "true", "text_preprocessing": "none"}},
        {"input": {"strip_diagnostic_sections": False, "text_preprocessing": "full"}},
        {"normalize_clinical_text": False},
        {
            "input": {
                "strip_diagnostic_sections": False,
                "text_preprocessing": "none",
                "extra": 0,
            }
        },
        {"id": "EN logreg"},
        {"tier": "classical"},
    ],
    ids=[
        "unknown-language",
        "unknown-model",
        "zero-trials",
        "string-trials",
        "string-bool",
        "unknown-preprocessing",
        "legacy-field",
        "extra-input-field",
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


@pytest.mark.parametrize(
    "field", ["input", "strip_diagnostic_sections", "text_preprocessing"]
)
def test_missing_axis_is_rejected(field: str) -> None:
    """Vstup i každá jeho osa jsou povinné, bez výchozích hodnot."""
    row = _row()
    if field == "input":
        del row[field]
    else:
        input_fields: dict[str, object] = {
            "strip_diagnostic_sections": False,
            "text_preprocessing": "none",
        }
        del input_fields[field]
        row["input"] = input_fields

    with pytest.raises(ValidationError, match=field):
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


def test_rows_differing_only_in_normalization_are_distinct() -> None:
    """Normalizace je osa: dva řádky lišící se jen jí projdou."""
    matrix = _validate(
        _row(
            input={
                "strip_diagnostic_sections": False,
                "text_preprocessing": "normalized",
            }
        ),
        _row(id="en-logreg-nonorm"),
    )

    assert len({row.axes for row in matrix.rows}) == len(matrix.rows)
    assert matrix.rows[0].axes[2].text_preprocessing is TextPreprocessing.NORMALIZED


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
    row = _row(
        id="cs-logreg-strip",
        language="cs",
        input={"strip_diagnostic_sections": True, "text_preprocessing": "none"},
    )

    with pytest.raises(ValidationError) as error:
        _validate(row)

    assert error.value.errors()[0]["msg"].startswith(
        "Value error, Řádek cs-logreg-strip: strip diagnostických sekcí"
    )


def test_czech_row_with_normalization_is_rejected_with_row_id() -> None:
    """Pravidla normalizace jsou anglická; hláška nese id řádku."""
    row = _row(
        id="cs-logreg-norm",
        language="cs",
        input={"strip_diagnostic_sections": False, "text_preprocessing": "normalized"},
    )

    with pytest.raises(ValidationError) as error:
        _validate(row)

    assert error.value.errors()[0]["msg"].startswith(
        "Value error, Řádek cs-logreg-norm: normalizace klinického textu"
    )


@pytest.mark.parametrize("model", ["fasttext", "xlm-r-base", "mdeberta-v3-base"])
def test_non_classical_row_with_normalization_is_rejected(model: str) -> None:
    """Normalizace je příznak klasického modelu; hláška nese id řádku."""
    row = _row(
        id=f"en-{model}",
        model=model,
        input={"strip_diagnostic_sections": False, "text_preprocessing": "normalized"},
    )

    with pytest.raises(ValidationError, match=rf"Řádek en-{model}: .*klasick"):
        _validate(row)


def test_czech_row_without_strip_is_valid() -> None:
    """Česká zpráva bez stripu je hlavní řádek, projde."""
    (row,) = _validate(_row(id="cs-logreg", language="cs")).rows

    assert row.language is Language.CS


@pytest.mark.parametrize(
    ("row_id", "strip", "preprocessing", "changed_axis"),
    [
        (
            "en-logreg-strip",
            True,
            TextPreprocessing.DICTIONARY,
            "strip_diagnostic_sections",
        ),
        ("en-logreg-nonorm", False, TextPreprocessing.NONE, "text_preprocessing"),
        (
            "en-logreg-novariants",
            False,
            TextPreprocessing.NORMALIZED,
            "text_preprocessing",
        ),
    ],
)
def test_repository_ablation_changes_one_input_axis(
    row_id: str, *, strip: bool, preprocessing: TextPreprocessing, changed_axis: str
) -> None:
    """Ruční ablační řádky mění právě jednu osu vstupu oproti hlavnímu řádku."""
    rows = {row.id: row for row in load_matrix(REPOSITORY_MATRIX).rows}
    main, ablation = rows["en-logreg"], rows[row_id]
    assert main.input.strip_diagnostic_sections is False
    assert main.input.text_preprocessing is TextPreprocessing.DICTIONARY
    assert ablation.input.strip_diagnostic_sections is strip
    assert ablation.input.text_preprocessing is preprocessing
    assert main.model_dump(exclude={"id", "input"}) == ablation.model_dump(
        exclude={"id", "input"}
    )
    assert main.input.model_dump(exclude={changed_axis}) == ablation.input.model_dump(
        exclude={changed_axis}
    )


def test_repository_manifest_is_valid() -> None:
    """Manifest v repu je platný — neplatná matice neprojde CI."""
    assert load_matrix(REPOSITORY_MATRIX).rows


def test_matrix_accepts_exactly_23_of_60_input_combinations() -> None:
    """Ruční orákulum: CS 5, EN klasika 12, EN ostatní 6 platných kombinací."""
    total_combinations, total_accepted = 60, 23
    czech_accepted, english_classical, english_other = 5, 12, 6
    accepted = []
    combinations = list(product(Language, ModelName, (False, True), TextPreprocessing))
    assert len(combinations) == total_combinations
    for language, model, strip, preprocessing in combinations:
        row = _row(
            language=language,
            model=model,
            input={
                "strip_diagnostic_sections": strip,
                "text_preprocessing": preprocessing,
            },
        )
        try:
            accepted.append(MatrixRow.model_validate_json(json.dumps(row)))
        except ValidationError:
            continue

    assert len(accepted) == total_accepted
    assert sum(row.language is Language.CS for row in accepted) == czech_accepted
    assert (
        sum(
            row.language is Language.EN and row.model.tier is Tier.CLASSICAL
            for row in accepted
        )
        == english_classical
    )
    assert (
        sum(
            row.language is Language.EN and row.model.tier is not Tier.CLASSICAL
            for row in accepted
        )
        == english_other
    )


def test_multiple_duplicated_input_axes_are_rejected() -> None:
    """Více opakovaných vstupů stejného modelu skončí validační chybou."""
    normalized = {
        "strip_diagnostic_sections": False,
        "text_preprocessing": "normalized",
    }
    with pytest.raises(ValidationError, match="stejné osy"):
        _validate(
            _row(),
            _row(id="copy-none"),
            _row(id="normalized", input=normalized),
            _row(id="copy-normalized", input=normalized),
        )
