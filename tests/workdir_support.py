"""Pracovní adresář překladu pro testy: požadavky, běhy a výstupy ve tvaru, jaký píše translation/."""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

REVISION: Final = "d" * 40
PROBE_START: Final = "2026-10-06T19:06:45.188215+00:00"
FIRST_START: Final = "2026-10-06T19:21:27.202468+00:00"
LAST_START: Final = "2026-10-07T17:21:12.787980+00:00"

type Record = dict[str, object]


def sha256_text(text: str) -> str:
    """SHA-256 textu v UTF-8.

    Returns:
        Hexadecimální otisk.
    """
    return hashlib.sha256(text.encode()).hexdigest()


def request(note_id: str, key: str, source: str) -> Record:
    """Požadavek překladu jedné části zprávy, bez tokenů zadání.

    Returns:
        Záznam požadavku.
    """
    return {
        "id": key,
        "note_id": note_id,
        "key": key,
        "source_sha256": sha256_text(source),
        "prompt_token_ids": [2, 105],
    }


def run_line(started: str, requests_sha256: str) -> Record:
    """Řádek runs.jsonl včetně polí, která provenance nenese.

    Returns:
        Záznam běhu.
    """
    return {
        "started": started,
        "requests_sha256": requests_sha256,
        "script_sha256": "5" * 64,
        "gpu": "NVIDIA A40",
        "memory_gb": 47.7,
        "capability": "8.6",
        "cuda": "13.0",
        "packages": {"vllm": "0.30.0"},
    }


def output(key: str, text: str, started: str) -> Record:
    """Řádek outputs.jsonl.

    Returns:
        Záznam výstupu.
    """
    return {
        "key": key,
        "id": key,
        "run": started,
        "text": text,
        "finish_reason": "stop",
        "output_tokens": 3,
    }


def write_workdir(
    root: Path,
    requests: Sequence[Record],
    runs: Sequence[Record],
    outputs: Sequence[Record],
) -> str:
    """Zapíše requests.json, runs.jsonl a outputs.jsonl.

    Returns:
        SHA-256 zapsaného requests.json.
    """
    root.mkdir(parents=True, exist_ok=True)
    plan = {
        "model": "google/translategemma-12b-it",
        "revision": REVISION,
        "dtype": "bfloat16",
        "window": 2048,
        "decoding": {"temperature": 0.0, "stop_token_ids": [1, 106]},
        "requests": list(requests),
    }
    content = json.dumps(plan)
    (root / "requests.json").write_text(content, encoding="utf-8")
    for name, records in (("runs.jsonl", runs), ("outputs.jsonl", outputs)):
        (root / name).write_text(
            "".join(f"{json.dumps(record)}\n" for record in records), encoding="utf-8"
        )
    return sha256_text(content)


def write_single_part_workdir(root: Path, notes: Mapping[str, tuple[str, str]]) -> str:
    """Pracovní adresář s jedním během a jednou částí na zprávu.

    Args:
        root: Cílový adresář.
        notes: note_id → dvojice anglický originál a text výstupu překladu.

    Returns:
        SHA-256 zapsaného requests.json.
    """
    requests = [
        request(note_id, f"key-{note_id}", en) for note_id, (en, _) in notes.items()
    ]
    outputs = [
        output(f"key-{note_id}", cs, LAST_START) for note_id, (_, cs) in notes.items()
    ]
    return write_workdir(root, requests, [run_line(LAST_START, "0" * 64)], outputs)
