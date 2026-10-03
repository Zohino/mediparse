"""Otisk zadání v provenance: vzniká z textu, který model dostane."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mediparse.domain.corpus_audit import fingerprint
from mediparse.domain.corpus_provenance import prompts_sha256
from mediparse.domain.verbalization import render_prompt

if TYPE_CHECKING:
    from mediparse.domain.synthetic_plan import SamplerConfig
    from tests.conftest import PlannedNote


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
