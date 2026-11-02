"""Export parametrů pětice ze souboru JSON, který zapsal sešit 8."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

from mediparse.domain.synthetic_parameters import SyntheticParameters
from mediparse.infrastructure.input_file import parse_file

PARAMETERS_PATH: Final = Path("results/eda/8-mentions/parametry-syntetiky.json")


@dataclass(frozen=True)
class SyntheticParametersFile:
    """Adaptér exportu: čte soubor a překládá chyby na doménovou chybu vstupu."""

    path: Path

    def load(self) -> SyntheticParameters:
        """Přečte export a ověří ho schématem.

        Returns:
            Parametry pětice.
        """
        return parse_file(self.path, SyntheticParameters.model_validate_json)
