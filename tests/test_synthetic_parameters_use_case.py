"""Převzetí parametrů pětice: výstup se předá configu, odmítnutí vyletí jako výjimka."""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import pytest

from mediparse.application.synthetic_parameters import (
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

    notes = ParameterTransfer(FakeExport(parameters), store).run()

    assert store.applied == [(parameters, EXPECTED_NOTES)]
    assert notes == EXPECTED_NOTES


def test_store_refusal_propagates() -> None:
    """Chyba configu vyletí k vstupnímu bodu s vlastní hláškou."""
    store = FakeStore(error="IPF nekonverguje")

    with pytest.raises(InvalidInputError, match="IPF nekonverguje"):
        ParameterTransfer(FakeExport(_parameters()), store).run()


def test_invalid_export_propagates_without_touching_config() -> None:
    """Neplatný export vyletí a config se nevolá."""
    store = FakeStore()

    with pytest.raises(InvalidInputError, match="export chybí"):
        ParameterTransfer(FakeExport(error="export chybí"), store).run()

    assert store.applied == []
