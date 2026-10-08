"""Otisk zadání v provenance: vzniká z textu, který model dostane."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Final

import pytest

from mediparse.domain.corpus_audit import fingerprint
from mediparse.domain.corpus_provenance import (
    ProvenanceRecord,
    Translation,
    model_id,
    prompts_sha256,
    provenance_violations,
    version,
)
from mediparse.domain.verbalization import render_prompt

if TYPE_CHECKING:
    from collections.abc import Callable

    from mediparse.domain.synthetic_plan import SamplerConfig
    from tests.conftest import PlannedNote

from tests.support import translation_record

AUDIT: Final = "9" * 64
RECORD: Final = {
    "generation": {
        "seed": 0,
        "sampler_config_sha256": "1" * 64,
        "prompts_sha256": "3" * 64,
        "model": "claude-sonnet-5-5",
        "claude_code_version": "2.1.5",
        "generated_on": "2026-10-10",
        "specification_commit": "d" * 40,
    },
    "audit_sha256": AUDIT,
}


def test_prompts_fingerprint_covers_rendered_prompt(
    planned_note: PlannedNote,
    sampler_config: SamplerConfig,
    verbalization_template: str,
) -> None:
    """Otisk zadání vzniká z vyrenderovaného textu, tedy i z vět, které skládá kód."""
    plan = planned_note.plan
    prompt = render_prompt(verbalization_template, plan, sampler_config)

    assert prompts_sha256(
        verbalization_template, [plan], sampler_config
    ) == fingerprint([(plan.note_id, prompt.encode())])


@pytest.mark.parametrize(
    ("validate", "valid", "invalid", "message"),
    [
        (model_id, "claude-sonnet-5-5", "gpt-4o", "Model"),
        (version, "2.1.288", "2.1", "Verze"),
    ],
    ids=["model", "version"],
)
def test_generation_validators(
    validate: Callable[[str], str], valid: str, invalid: str, message: str
) -> None:
    """Validátor platnou hodnotu vrátí beze změny a neplatnou odmítne."""
    assert validate(valid) == valid
    with pytest.raises(ValueError, match=message):
        validate(invalid)


def test_record_without_translation_loads() -> None:
    """Dnešní provenance.json bez části translation se načte, překlad je volitelný."""
    record = ProvenanceRecord.model_validate_json(json.dumps(RECORD))

    assert record.translation is None


def test_translation_survives_json_round_trip() -> None:
    """Část translation se zapíše a načte beze změny, včetně data běhu."""
    record = ProvenanceRecord.model_validate_json(
        json.dumps(RECORD | {"translation": None})
    ).model_copy(update={"translation": translation_record(marker_mismatches=("a",))})

    loaded = ProvenanceRecord.model_validate_json(record.model_dump_json())

    assert loaded == record
    assert isinstance(loaded.translation, Translation)


def test_translation_rejects_short_revision() -> None:
    """Revize modelu je celý SHA, zkrácená se odmítne."""
    with pytest.raises(ValueError, match="Commit"):
        translation_record(revision="abc")


def _record(*, translation: bool) -> ProvenanceRecord:
    return ProvenanceRecord.model_validate_json(json.dumps(RECORD)).model_copy(
        update={"translation": translation_record() if translation else None}
    )


def test_czech_corpus_without_translation_provenance_is_reported() -> None:
    """Korpus s cs/ a provenance bez překladu neprojde."""
    violations = provenance_violations(
        _record(translation=False), AUDIT, translated=True
    )

    assert violations == ("Český korpus nemá provenance překladu.",)


@pytest.mark.parametrize(
    ("translation", "translated"),
    [(True, True), (False, False), (True, False)],
    ids=["translated", "english-only", "stale-translation"],
)
def test_translation_provenance_matches_corpus(
    *, translation: bool, translated: bool
) -> None:
    """S cs/ musí být translation, bez cs/ nevadí."""
    record = _record(translation=translation)

    assert provenance_violations(record, AUDIT, translated=translated) == ()
