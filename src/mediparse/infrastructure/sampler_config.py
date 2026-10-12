"""Načtení a aktualizace konfigurace vzorkovače plánů v JSON souboru."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

from pydantic import BaseModel, ConfigDict, ValidationError

from mediparse.domain.inputs import InvalidInputError
from mediparse.domain.labels import IpfNotConvergedError, max_entropy_joint
from mediparse.domain.synthetic_plan import SamplerConfig
from mediparse.infrastructure.input_file import parse_file

if TYPE_CHECKING:
    from mediparse.domain.synthetic_parameters import SyntheticParameters

SAMPLER_CONFIG_PATH: Final = Path("config/synthetic_plan.json")
_JSON_INDENT: Final = 2


def load_sampler_config(path: Path) -> SamplerConfig:
    """Přečte JSON konfiguraci a předá ji validaci.

    Returns:
        Konfigurace, která prošla schématem i invarianty.
    """
    return parse_file(path, SamplerConfig.model_validate_json)


@dataclass(frozen=True)
class SamplerConfigFile:
    """Config vzorkovače v souboru; zápis parametrů zachová pořadí klíčů a formát."""

    path: Path

    def apply(self, parameters: SyntheticParameters, notes: int) -> None:
        """Nahradí bloky labelů a zmínek a počet zpráv; zapíše jen platný config.

        Config, který nejde přečíst, nebo výsledek, který neprojde schématem či IPF,
        skončí chybou vstupu a soubor zůstane beze změny.
        """
        tree = parse_file(self.path, _read_tree)
        keys = parameters.selection
        tree["labels"]["prevalence"] = {k: parameters.prevalence[k] for k in keys}
        tree["labels"]["conditional"] = {
            row: {column: parameters.conditional[row][column] for column in keys}
            for row in keys
        }
        tree["mentions"]["diagnoses"] = {
            key: parameters.mentions[key].model_dump(mode="json") for key in keys
        }
        tree["patients"]["notes"] = notes
        text = _dump(tree)
        _validate(text, self.path)
        self.path.write_text(text, encoding="utf-8")


class _ConfigTree(BaseModel):
    """Tvar souboru, který příkaz mění: tři bloky, ostatní klíče projdou beze změny."""

    model_config = ConfigDict(extra="allow")

    labels: dict[str, Any]
    mentions: dict[str, Any]
    patients: dict[str, Any]


def _read_tree(text: str) -> dict[str, Any]:
    _ConfigTree.model_validate_json(text)
    return json.loads(text)


def _dump(tree: dict[str, Any]) -> str:
    return json.dumps(tree, indent=_JSON_INDENT, ensure_ascii=False) + "\n"


def _validate(text: str, path: Path) -> None:
    try:
        config = SamplerConfig.model_validate_json(text)
        max_entropy_joint(config.labels)
    except ValidationError as error:
        msg = f"Config {path} by po převzetí neodpovídal schématu:\n{error}"
        raise InvalidInputError(msg) from error
    except IpfNotConvergedError as error:
        msg = f"Config {path} by po převzetí neměl konvergující IPF: {error}"
        raise InvalidInputError(msg) from error
