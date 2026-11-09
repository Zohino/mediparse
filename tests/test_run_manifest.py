"""Schéma run manifestu: přísná validace otisků a vnoření prostředí."""

from __future__ import annotations

from typing import Final

import pytest
from pydantic import ValidationError

from mediparse.domain.clinical_text import TextNormalization
from mediparse.domain.environment import (
    Environment,
    GpuDevice,
    Hardware,
    SourceRevision,
)
from mediparse.domain.inputs import InvalidInputError
from mediparse.domain.run_manifest import (
    DiagnosticStrip,
    EvaluationManifest,
    RunManifest,
    ensure_training_lineage,
)

COMMIT: Final = "a" * 40
DIGEST: Final = "b" * 64
HARDWARE: Final = {
    "cpu_model": "Synthetic CPU",
    "logical_cores": 8,
    "available_cores": 2,
    "memory_bytes": 1024,
    "gpus": (),
}


def _fields(**changes: object) -> dict[str, object]:
    fields: dict[str, object] = {
        "row_id": "row",
        "dataset_sha256": DIGEST,
        "predictions_sha256": DIGEST,
        "config_sha256": DIGEST,
        "seed": 0,
        "split_seed": 1,
        "split_folds": 5,
        "split_sha256": DIGEST,
        "diagnostic_strip": None,
        "normalize_clinical_text": None,
        "model": "sklearn TfidfVectorizer+LinearSVC",
        "model_revision": None,
        "tokenizer_revision": None,
        "hardware": HARDWARE,
        "environment": Environment(
            packages={"python": "3.13.0"},
            source=SourceRevision(commit=COMMIT, dirty=False),
        ),
    }
    return fields | changes


def test_valid_manifest_with_null_revisions() -> None:
    """Platný manifest projde a revize modelu i tokenizeru smí být null."""
    manifest = RunManifest.model_validate(_fields())

    assert manifest.model_revision is None
    assert manifest.tokenizer_revision is None
    assert manifest.environment.source.commit == COMMIT


def test_manifest_requires_hardware() -> None:
    """Manifest bez údajů o stroji nelze vydávat za úplnou provenance."""
    fields = _fields()
    del fields["hardware"]

    with pytest.raises(ValidationError, match="hardware"):
        RunManifest.model_validate(fields)


def test_manifest_records_hardware() -> None:
    """Manifest zachová údaje CPU, paměti a GPU o stroji."""
    fields = _fields()
    dumped = RunManifest.model_validate(fields).model_dump(mode="json")

    assert dumped["hardware"] == HARDWARE | {"gpus": []}


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("logical_cores", 0),
        ("available_cores", 0),
        ("memory_bytes", 0),
        ("cpu_model", ""),
        ("latency_seconds", 0.1),
    ],
)
def test_manifest_rejects_invalid_hardware(field: str, value: object) -> None:
    """Manifest odmítne nulové kapacity, prázdný model CPU a měření navíc v HW."""
    with pytest.raises(ValidationError, match=rf"hardware\.{field}"):
        RunManifest.model_validate(_fields(hardware=HARDWARE | {field: value}))


@pytest.mark.parametrize(
    "field", ["dataset_sha256", "config_sha256", "split_sha256", "predictions_sha256"]
)
@pytest.mark.parametrize("digest", ["b" * 63, "b" * 65])
def test_digest_of_other_length_is_rejected(field: str, digest: str) -> None:
    """Hash jiné délky než 64 znaků je odmítnut."""
    with pytest.raises(ValidationError):
        RunManifest.model_validate(_fields(**{field: digest}))


@pytest.mark.parametrize("field", ["seed", "split_seed"])
@pytest.mark.parametrize("seed", [-1, 2**32])
def test_seed_outside_generator_range_is_rejected(field: str, seed: int) -> None:
    """Seed mimo rozsah generátoru manifest odmítne."""
    with pytest.raises(ValidationError):
        RunManifest.model_validate(_fields(**{field: seed}))


@pytest.mark.parametrize("folds", [0, 1])
def test_fewer_than_two_split_folds_are_rejected(folds: int) -> None:
    """Split má aspoň dva foldy, manifest s méně je odmítnut."""
    with pytest.raises(ValidationError):
        RunManifest.model_validate(_fields(split_folds=folds))


def test_extra_field_is_rejected() -> None:
    """Pole navíc je odmítnuto."""
    with pytest.raises(ValidationError):
        RunManifest.model_validate(_fields(started="now"))


@pytest.mark.parametrize("row_id", ["", "Row 1"])
def test_invalid_row_id_is_rejected(row_id: str) -> None:
    """Prázdné id i id mimo skladbu řádku matice manifest odmítne."""
    with pytest.raises(ValidationError):
        RunManifest.model_validate(_fields(row_id=row_id))


def test_manifest_nests_environment_instead_of_flat_fields() -> None:
    """Serializovaný manifest nese ``environment`` a nemá ``packages`` ani ``source`` nahoře."""
    dumped = RunManifest.model_validate(_fields()).model_dump(mode="json")

    assert dumped["environment"] == {
        "packages": {"python": "3.13.0"},
        "source": {"commit": COMMIT, "dirty": False},
    }
    assert "packages" not in dumped
    assert "source" not in dumped


def test_manifest_without_strip_serializes_null() -> None:
    """Bez stripu je ``diagnostic_strip`` ``null``."""
    dumped = RunManifest.model_validate(_fields()).model_dump(mode="json")

    assert dumped["diagnostic_strip"] is None


def test_strip_without_sections_is_rejected() -> None:
    """Záznam stripu bez sekcí nelze zapsat."""
    with pytest.raises(ValidationError):
        DiagnosticStrip.model_validate({"sections": (), "lexicon_sha256": DIGEST})


def test_strip_serializes_sections_as_sorted_list() -> None:
    """Sekce zadané v jiném pořadí jdou do JSON jako seřazený seznam."""
    strip = DiagnosticStrip(sections=("b_section", "a_section"), lexicon_sha256=DIGEST)
    manifest = RunManifest.model_validate(_fields(diagnostic_strip=strip))

    dumped = manifest.model_dump(mode="json")

    assert dumped["diagnostic_strip"] == {
        "sections": ["a_section", "b_section"],
        "lexicon_sha256": DIGEST,
    }


def test_manifest_without_strip_key_is_rejected() -> None:
    """Klíč ``diagnostic_strip`` je povinný, chybějící klíč není totéž co ``null``."""
    fields = _fields()
    del fields["diagnostic_strip"]

    with pytest.raises(ValidationError, match="diagnostic_strip"):
        RunManifest.model_validate(fields)


def test_strip_with_extra_field_is_rejected() -> None:
    """Záznam stripu s polem navíc je odmítnut."""
    with pytest.raises(ValidationError, match="extra"):
        DiagnosticStrip.model_validate({
            "sections": ("a",),
            "lexicon_sha256": DIGEST,
            "extra": 1,
        })


def test_strip_duplicate_sections_are_merged() -> None:
    """Opakovaný klíč sekce se sloučí, záznam říká, které sekce se odstranily."""
    strip = DiagnosticStrip(sections=("b", "a", "b"), lexicon_sha256=DIGEST)

    assert strip.sections == ("a", "b")


def test_manifest_records_normalization_as_sorted_pairs() -> None:
    """Záznam normalizace se zapíše jako objekt se seřazenými dvojicemi zkratka a tvar."""
    dumped = RunManifest.model_validate(
        _fields(
            normalize_clinical_text=TextNormalization(
                variants={"kindey": "kidney", "afib": "atrial fibrillation"},
                acronyms={"DM2": "diabetes", "AF": "atrial fibrillation"},
            )
        )
    ).model_dump(mode="json")

    assert list(dumped["normalize_clinical_text"]["variants"].items()) == [
        ("afib", "atrial fibrillation"),
        ("kindey", "kidney"),
    ]
    assert list(dumped["normalize_clinical_text"]["acronyms"].items()) == [
        ("AF", "atrial fibrillation"),
        ("DM2", "diabetes"),
    ]


def test_manifest_without_normalization_is_none() -> None:
    """Bez normalizace pole nese ``None`` a v JSON je ``null``."""
    dumped = RunManifest.model_validate(_fields()).model_dump(mode="json")

    assert dumped["normalize_clinical_text"] is None


def test_manifest_without_normalization_key_is_rejected() -> None:
    """Klíč ``normalize_clinical_text`` je povinný, bez výchozí hodnoty."""
    fields = _fields()
    del fields["normalize_clinical_text"]

    with pytest.raises(ValidationError, match="normalize_clinical_text"):
        RunManifest.model_validate(fields)


def test_normalization_as_string_is_rejected() -> None:
    """Bool ani řetězec nejsou záznam normalizace."""
    for value in ("true", True, False):
        with pytest.raises(ValidationError, match="normalize_clinical_text"):
            RunManifest.model_validate(_fields(normalize_clinical_text=value))


def test_hardware_requires_gpu_devices() -> None:
    """Chybějící pole GPU není totéž co seznam bez zařízení."""
    with pytest.raises(ValidationError, match="gpus"):
        Hardware.model_validate({
            key: value for key, value in HARDWARE.items() if key != "gpus"
        })


@pytest.mark.parametrize(("name", "memory"), [("", 1000), ("Fake GPU", 0)])
def test_gpu_device_requires_name_and_capacity(name: str, memory: int) -> None:
    """Prázdné jméno a nulová kapacita nemohou popsat GPU."""
    with pytest.raises(ValidationError):
        GpuDevice(name=name, memory_total_bytes=memory)


def _evaluation_fields() -> dict[str, object]:
    return {
        "row_id": "row",
        "training_manifest_sha256": DIGEST,
        "predictions_sha256": DIGEST,
        "dataset_sha256": DIGEST,
        "split_sha256": DIGEST,
        "metrics_sha256": DIGEST,
        "environment": _fields()["environment"],
    }


@pytest.mark.parametrize("field", list(_evaluation_fields()))
def test_evaluation_manifest_requires_every_field(field: str) -> None:
    """Každý článek provenance evaluace je povinný."""
    fields = _evaluation_fields()
    del fields[field]
    with pytest.raises(ValidationError, match=field):
        EvaluationManifest.model_validate(fields)


@pytest.mark.parametrize(
    "field",
    [
        "training_manifest_sha256",
        "predictions_sha256",
        "dataset_sha256",
        "split_sha256",
        "metrics_sha256",
    ],
)
@pytest.mark.parametrize("value", ["b" * 63, "g" * 64, 1])
def test_evaluation_manifest_rejects_invalid_digest(field: str, value: object) -> None:
    """Otisky jsou přísné řetězce SHA-256, nikoli libovolné hodnoty."""
    with pytest.raises(ValidationError, match=field):
        EvaluationManifest.model_validate(_evaluation_fields() | {field: value})


def test_evaluation_manifest_forbids_extra_fields() -> None:
    """Manifest neznámý klíč nepřijme."""
    with pytest.raises(ValidationError, match="extra"):
        EvaluationManifest.model_validate(_evaluation_fields() | {"extra": 1})


def test_evaluation_manifest_is_frozen() -> None:
    """Manifest po validaci nelze měnit."""
    manifest = EvaluationManifest.model_validate(_evaluation_fields())
    with pytest.raises(ValidationError, match="frozen"):
        manifest.row_id = "other"  # ty: ignore[invalid-assignment]


@pytest.mark.parametrize(
    "matching", ["predictions_sha256", "dataset_sha256", "split_sha256"]
)
def test_lineage_names_all_mismatched_fields(matching: str) -> None:
    """Dvě neshody vrátí obě jména, bez otisků nebo identifikátorů."""
    digests = dict.fromkeys(
        ("predictions_sha256", "dataset_sha256", "split_sha256"), "c" * 64
    )
    digests[matching] = DIGEST
    with pytest.raises(InvalidInputError) as error:
        ensure_training_lineage(RunManifest.model_validate(_fields()), **digests)
    for field in digests:
        assert (field in str(error.value)) == (field != matching)
    assert "c" * 64 not in str(error.value)


def test_matching_training_lineage_is_accepted() -> None:
    """Shodné vstupy pokračují bez výjimky."""
    ensure_training_lineage(
        RunManifest.model_validate(_fields()),
        predictions_sha256=DIGEST,
        dataset_sha256=DIGEST,
        split_sha256=DIGEST,
    )


def test_training_manifest_requires_prediction_digest() -> None:
    """Manifest bez vazby na predikce není platná provenance."""
    fields = _fields()
    del fields["predictions_sha256"]
    with pytest.raises(ValidationError, match="predictions_sha256"):
        RunManifest.model_validate(fields)
