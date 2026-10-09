"""Pracovní adresář překladu na disku: požadavky, běhy a výstupy ve tvaru, jaký píše translation/."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Final

from pydantic import BaseModel, ConfigDict, PositiveInt

from mediparse.domain.corpus_audit import CommitSha, Sha256
from mediparse.domain.corpus_provenance import (
    Decoding,
    Masking,
    Translation,
    TranslationRun,
)
from mediparse.domain.inputs import InvalidInputError
from mediparse.domain.translation_consistency import TranslatedNote
from mediparse.infrastructure.file_digest import file_sha256
from mediparse.infrastructure.input_file import parse_file

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

REQUESTS_NAME: Final = "requests.json"
RUNS_NAME: Final = "runs.jsonl"
OUTPUTS_NAME: Final = "outputs.jsonl"
WORKDIR_PATH: Final = "build/translation"
_LENIENT: Final = ConfigDict(frozen=True, extra="ignore", strict=True)


class _Request(BaseModel):
    model_config = _LENIENT

    note_id: str
    key: str
    source_sha256: Sha256


class _Plan(BaseModel):
    model_config = _LENIENT

    model: str
    revision: CommitSha
    dtype: str
    window: PositiveInt
    decoding: Decoding
    requests: tuple[_Request, ...]
    masking: Masking | None = None


class _RunLine(BaseModel):
    model_config = _LENIENT

    started: datetime
    requests_sha256: Sha256
    script_sha256: Sha256
    gpu: str
    cuda: str
    packages: dict[str, str]


class _Output(BaseModel):
    model_config = _LENIENT

    key: str
    run: datetime
    text: str


@dataclass(frozen=True)
class TranslationWorkdir:
    """Pracovní adresář překladu; chybějící nebo neplatný soubor je chybný vstup."""

    root: Path

    def translation(self) -> Translation:
        """Vstupy překladu z požadavků, běhů a výstupů; neshody značek doplňuje use case.

        Returns:
            Provenance překladu s běhy, z nichž pochází výstup pro klíč požadavku.

        Raises:
            InvalidInputError: Soubor chybí či neodpovídá schématu, požadavek nemá
                výstup nebo výstup pochází z běhu, který runs.jsonl nezná.
        """
        plan = self._plan()
        runs = self._runs()
        started = sorted({output.run for output in self._latest(plan).values()})
        unknown = [moment.isoformat() for moment in started if moment not in runs]
        if unknown:
            msg = f"Výstupy pocházejí z běhů, které {RUNS_NAME} nezná: {', '.join(unknown)}."
            raise InvalidInputError(msg)
        return Translation(
            model=plan.model,
            revision=plan.revision,
            dtype=plan.dtype,
            window=plan.window,
            decoding=plan.decoding,
            requests_sha256=file_sha256(self.root / REQUESTS_NAME),
            runs=tuple(runs[moment] for moment in started),
            marker_mismatches=(),
            masking=plan.masking,
        )

    def notes(self) -> tuple[TranslatedNote, ...]:
        """Zprávy z požadavků s přeloženými částmi v pořadí požadavků.

        Returns:
            Zprávy v pořadí prvního požadavku; u klíče platí poslední výstup.
        """
        plan = self._plan()
        outputs = self._latest(plan)
        grouped: dict[str, list[_Request]] = {}
        for request in plan.requests:
            grouped.setdefault(request.note_id, []).append(request)
        return tuple(
            TranslatedNote(
                note_id,
                requests[0].source_sha256,
                tuple(outputs[request.key].text for request in requests),
            )
            for note_id, requests in grouped.items()
        )

    def _plan(self) -> _Plan:
        return parse_file(self.root / REQUESTS_NAME, _Plan.model_validate_json)

    def _runs(self) -> dict[datetime, TranslationRun]:
        lines = parse_file(self.root / RUNS_NAME, _lines(_RunLine))
        return {line.started: TranslationRun(**line.model_dump()) for line in lines}

    def _latest(self, plan: _Plan) -> dict[str, _Output]:
        outputs = {
            output.key: output
            for output in parse_file(self.root / OUTPUTS_NAME, _lines(_Output))
        }
        missing = [
            request.key for request in plan.requests if request.key not in outputs
        ]
        if missing:
            msg = f"Chybí výstup překladu pro klíče: {', '.join(missing)}."
            raise InvalidInputError(msg)
        return {request.key: outputs[request.key] for request in plan.requests}


def _lines[T: BaseModel](model: type[T]) -> Callable[[str], tuple[T, ...]]:
    def parse(content: str) -> tuple[T, ...]:
        return tuple(
            model.model_validate_json(line)
            for line in content.splitlines()
            if line.strip()
        )

    return parse
