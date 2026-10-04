# /// script
# requires-python = ">=3.13"
# ///
"""Zapíše přeložené zprávy do korpusu a ohlásí, co chybí nebo nesedí s originálem.

Výstup se k požadavku páruje klíčem a požadavek s originálem hashem, takže do
korpusu se nedostane překlad jiné verze zprávy. Text se normalizuje stejně jako
hooky repa, aby hash korpusu po commitu seděl s tím, co prošlo auditem.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

if TYPE_CHECKING:
    from collections.abc import Mapping

REQUESTS: Final = "requests.json"
OUTPUTS: Final = "outputs.jsonl"
MARKER: Final = re.compile(r"(?<!_)___(?!_)")
LOG_FORMAT: Final = "%(asctime)s %(levelname)s %(message)s"

logger = logging.getLogger(__name__)

type Record = dict[str, Any]


def main() -> int:
    """Zapíše úplné překlady aktuálních originálů a ohlásí problémy.

    Returns:
        0, když má každý požadavek úplný překlad aktuálního originálu se stejným
        počtem značek ___, jinak 1.
    """
    logging.basicConfig(format=LOG_FORMAT)
    logger.setLevel(logging.INFO)
    args = _parser().parse_args()
    plan = json.loads((args.workdir / REQUESTS).read_text(encoding="utf-8"))
    requests = plan["requests"]
    outputs = _outputs(args.workdir / OUTPUTS)
    problems = {
        "Originál se od přípravy změnil": [
            request["id"] for request in requests if _changed(request, args.source)
        ],
        "Chybí překlad": [
            request["id"] for request in requests if request["key"] not in outputs
        ],
        "Uříznutý překlad": [
            request["id"]
            for request in requests
            if request["key"] in outputs
            and outputs[request["key"]]["finish_reason"] != "stop"
        ],
    }
    flagged = set(itertools.chain.from_iterable(problems.values()))
    complete = {
        request["id"]: _normalized(outputs[request["key"]]["text"])
        for request in requests
        if request["id"] not in flagged
    }
    problems["Jiný počet značek ___ než originál"] = _mismatched(complete, args.source)
    args.target.mkdir(parents=True, exist_ok=True)
    for note_id, text in complete.items():
        (args.target / f"{note_id}.txt").write_text(text, encoding="utf-8")
    logger.info(
        "Zapsáno %d z %d překladů do %s.", len(complete), len(requests), args.target
    )
    for label, note_ids in problems.items():
        if note_ids:
            logger.error("%s: %s", label, ", ".join(note_ids))
    return 1 if any(problems.values()) else 0


def _outputs(path: Path) -> dict[str, Record]:
    if not path.exists():
        return {}
    lines = path.read_text(encoding="utf-8").splitlines()
    return {output["key"]: output for output in map(json.loads, lines)}


def _changed(request: Record, source: Path) -> bool:
    path = source / f"{request['id']}.txt"
    return (
        not path.exists()
        or hashlib.sha256(path.read_bytes()).hexdigest() != request["source_sha256"]
    )


def _normalized(text: str) -> str:
    return "".join(f"{line.rstrip()}\n" for line in text.strip().splitlines())


def _mismatched(complete: Mapping[str, str], source: Path) -> list[str]:
    return [
        note_id
        for note_id, text in complete.items()
        if _markers(text)
        != _markers((source / f"{note_id}.txt").read_text(encoding="utf-8"))
    ]


def _markers(text: str) -> int:
    return len(MARKER.findall(text))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workdir",
        type=Path,
        default=Path("build/translation"),
        help=f"adresář s {REQUESTS} a {OUTPUTS}",
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("resources/synthetic/en"),
        help="adresář anglických originálů",
    )
    parser.add_argument(
        "--target",
        type=Path,
        default=Path("resources/synthetic/cs"),
        help="adresář, kam se zapíšou české zprávy",
    )
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
