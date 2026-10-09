# /// script
# requires-python = ">=3.13"
#
# [tool.ty.environment]
# extra-paths = ["."]
# ///
"""Zapíše přeložené zprávy do korpusu a ohlásí, co chybí nebo nesedí s originálem.

Části rozdělené zprávy se skládají podle ``note_id`` v pořadí požadavků. Když plán
nese maskování, hlásí se chybějící, přebývající a zdvojená čísla značek [[n]]
a značky se vrátí na ___ dřív, než se porovná jejich počet. Výstup se
k požadavku páruje klíčem a požadavek s originálem hashem, takže do
korpusu se nedostane překlad jiné verze zprávy. Text se normalizuje stejně jako
hooky repa, aby hash korpusu po commitu seděl s tím, co prošlo auditem.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

from markers import MARKER, PLACEHOLDER, placeholder_problems, restore
from note_parts import join

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

REQUESTS: Final = "requests.json"
OUTPUTS: Final = "outputs.jsonl"
LOG_FORMAT: Final = "%(asctime)s %(levelname)s %(message)s"

logger = logging.getLogger(__name__)

type Record = dict[str, Any]


def main() -> int:
    """Zapíše úplné překlady aktuálních originálů a ohlásí problémy.

    Returns:
        0, když má každá zpráva úplný překlad aktuálního originálu se stejným
        počtem značek ___, jinak 1.
    """
    logging.basicConfig(format=LOG_FORMAT)
    logger.setLevel(logging.INFO)
    args = _parser().parse_args()
    plan = json.loads((args.workdir / REQUESTS).read_text(encoding="utf-8"))
    masking = plan.get("masking")
    if masking not in {None, PLACEHOLDER}:
        logger.error("Neznámé maskování v %s: %r", REQUESTS, masking)
        return 1
    notes = _notes(plan["requests"])
    outputs = _outputs(args.workdir / OUTPUTS)
    problems = {
        "Originál se od přípravy změnil": [
            note_id
            for note_id, requests in notes.items()
            if any(_changed(request, args.source) for request in requests)
        ],
        "Chybí překlad": [
            note_id
            for note_id, requests in notes.items()
            if any(request["key"] not in outputs for request in requests)
        ],
        "Uříznutý překlad": [
            note_id
            for note_id, requests in notes.items()
            if any(_truncated(request, outputs) for request in requests)
        ],
    }
    flagged = set(itertools.chain.from_iterable(problems.values()))
    joined = {
        note_id: join([outputs[request["key"]]["text"] for request in requests])
        for note_id, requests in notes.items()
        if note_id not in flagged
    }
    expected = {
        note_id: _markers((args.source / f"{note_id}.txt").read_text(encoding="utf-8"))
        for note_id in joined
    }
    if masking == PLACEHOLDER:
        problems.update(_placeholder_problems(joined, expected))
        joined = {note_id: restore(text) for note_id, text in joined.items()}
    complete = {note_id: _normalized(text) for note_id, text in joined.items()}
    problems["Jiný počet značek ___ než originál"] = _mismatched(complete, expected)
    args.target.mkdir(parents=True, exist_ok=True)
    for note_id, text in complete.items():
        (args.target / f"{note_id}.txt").write_text(text, encoding="utf-8")
    logger.info(
        "Zapsáno %d z %d překladů do %s.", len(complete), len(notes), args.target
    )
    for label, note_ids in problems.items():
        if note_ids:
            logger.error("%s: %s", label, ", ".join(note_ids))
    return 1 if any(problems.values()) else 0


def _notes(requests: Sequence[Record]) -> dict[str, list[Record]]:
    notes: dict[str, list[Record]] = {}
    for request in requests:
        notes.setdefault(request["note_id"], []).append(request)
    return notes


def _truncated(request: Record, outputs: Mapping[str, Record]) -> bool:
    output = outputs.get(request["key"])
    return output is not None and output["finish_reason"] != "stop"


def _outputs(path: Path) -> dict[str, Record]:
    if not path.exists():
        return {}
    lines = path.read_text(encoding="utf-8").splitlines()
    return {output["key"]: output for output in map(json.loads, lines)}


def _changed(request: Record, source: Path) -> bool:
    path = source / f"{request['note_id']}.txt"
    return (
        not path.exists()
        or hashlib.sha256(path.read_bytes()).hexdigest() != request["source_sha256"]
    )


def _normalized(text: str) -> str:
    return "".join(f"{line.rstrip()}\n" for line in text.strip().splitlines())


def _mismatched(complete: Mapping[str, str], expected: Mapping[str, int]) -> list[str]:
    return [
        note_id
        for note_id, text in complete.items()
        if _markers(text) != expected[note_id]
    ]


def _placeholder_problems(
    joined: Mapping[str, str], expected: Mapping[str, int]
) -> dict[str, list[str]]:
    labels = {
        "missing": "Chybí značky [[n]]",
        "extra": "Značky [[n]] navíc",
        "duplicated": "Zdvojené značky [[n]]",
    }
    found: dict[str, list[str]] = {label: [] for label in labels.values()}
    for note_id, text in joined.items():
        problems = placeholder_problems(text, expected[note_id])
        for field, label in labels.items():
            numbers = getattr(problems, field)
            if numbers:
                found[label].append(f"{note_id} ({', '.join(map(str, numbers))})")
    return found


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
