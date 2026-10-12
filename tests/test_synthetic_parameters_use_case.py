"""Převzetí parametrů pětice: výstup se předá configu a odmítnutí nese důvod."""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from mediparse.application.synthetic_parameters import (
    ParametersRefused,
    ParametersWritten,
    ParameterTransfer,
)
from mediparse.domain.inputs import InvalidInputError
from mediparse.domain.synthetic_parameters import SyntheticParameters
from tests.support import export_payload

KEYS = ("a", "b")
EXPECTED_NOTES = 400


def _parameters() -> SyntheticParameters:
    payload = export_payload(KEYS, prevalence=0.0636)
    return SyntheticParameters.model_validate_json(json.dumps(payload))


@dataclass
class FakeExport:
    """Export, který vrátí zadané parametry, nebo odmítne."""

    parameters: SyntheticParameters | None = None
    error: str = ""

    def load(self) -> SyntheticParameters:
        """Vrátí parametry.

        Returns:
            Parametry zadané při vytvoření.

        Raises:
            InvalidInputError: Export nemá parametry.
        """
        if self.parameters is None:
            raise InvalidInputError(self.error)
        return self.parameters


@dataclass
class FakeStore:
    """Config, který si pamatuje zápisy, nebo odmítne."""

    error: str = ""
    applied: list[tuple[SyntheticParameters, int]] = field(default_factory=list)

    def apply(self, parameters: SyntheticParameters, notes: int) -> None:
        """Zaznamená zápis.

        Raises:
            InvalidInputError: Config je nastaven na odmítnutí.
        """
        if self.error:
            raise InvalidInputError(self.error)
        self.applied.append((parameters, notes))


def test_apply_receives_parameters_and_required_notes() -> None:
    """Config dostane parametry a počet zpráv odvozený z nejvzácnější prevalence."""
    parameters = _parameters()
    store = FakeStore()

    result = ParameterTransfer(FakeExport(parameters), store).run()

    assert store.applied == [(parameters, EXPECTED_NOTES)]
    assert result == ParametersWritten(notes=EXPECTED_NOTES)


def test_store_refusal_carries_its_message() -> None:
    """Chybu configu use case vrátí jako odmítnutí s její hláškou."""
    store = FakeStore(error="IPF nekonverguje")

    result = ParameterTransfer(FakeExport(_parameters()), store).run()

    assert result == ParametersRefused(reason="IPF nekonverguje")


def test_invalid_export_is_refused_without_touching_config() -> None:
    """Neplatný export skončí odmítnutím a config se nevolá."""
    store = FakeStore()

    result = ParameterTransfer(FakeExport(error="export chybí"), store).run()

    assert result == ParametersRefused(reason="export chybí")
    assert store.applied == []
