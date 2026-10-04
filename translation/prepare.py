# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "jinja2>=3.1.6",
#     "transformers>=5.18.0",
# ]
#
# [tool.ty.analysis]
# allowed-unresolved-imports = ["transformers"]
# ///
"""Sestaví požadavky na překlad syntetických zpráv a ověří, že se vejdou do okna modelu.

Prompt skládá chat šablona modelu v pevné revizi, takže ID tokenů jsou přesně ta,
která model čeká, a překladový skript je na GPU jen předá. Klíč požadavku je hash
všeho, co určuje překlad, a slouží zároveň jako cache. Požadavky se zapíšou, jen
když se do okna vejde prompt každé zprávy. Zprávy, kterým na překlad zbude méně
místa, než je odhad jeho délky, skript vypíše jako riziko uříznutí.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final

from transformers import AutoTokenizer, GenerationConfig

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from transformers import PreTrainedTokenizerBase

MODEL: Final = "google/translategemma-12b-it"
REVISION: Final = "d1b225e1caa17f1ddc7e62065d8637d0923f34e2"
WINDOW: Final = 2048
OUTPUT_RATIO: Final = 1.5
DTYPE: Final = "bfloat16"
SOURCE_LANG: Final = "en"
TARGET_LANG: Final = "cs"
REQUESTS: Final = "requests.json"
LOG_FORMAT: Final = "%(asctime)s %(levelname)s %(message)s"

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Prompt:
    """Prompt jedné zprávy a kolikrát se do zbytku okna vejde délka originálu."""

    note_id: str
    source_sha256: str
    token_ids: list[int]
    room: float


def main() -> int:
    """Zapíše požadavky na překlad, když se do okna vejdou prompty všech zpráv.

    Returns:
        0 po zápisu požadavků, 1 když korpus nemá zprávy nebo se prompt některé
        zprávy do okna nevejde.
    """
    logging.basicConfig(format=LOG_FORMAT)
    logger.setLevel(logging.INFO)
    args = _parser().parse_args()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    prompts = [_prompt(tokenizer, path) for path in sorted(args.source.glob("*.txt"))]
    if not prompts:
        logger.error("V %s nejsou žádné zprávy.", args.source)
        return 1
    lengths = [len(prompt.token_ids) for prompt in prompts]
    logger.info(
        "Zpráv %d, prompt má medián %s a nejvýš %d tokenů z okna %d.",
        len(prompts),
        statistics.median(lengths),
        max(lengths),
        WINDOW,
    )
    overflowing = [
        prompt.note_id for prompt in prompts if len(prompt.token_ids) >= WINDOW
    ]
    if overflowing:
        logger.error("Prompt se do okna nevejde: %s", ", ".join(overflowing))
        return 1
    risky = sorted(
        (prompt for prompt in prompts if prompt.room < OUTPUT_RATIO),
        key=lambda prompt: prompt.room,
    )
    if risky:
        listed = ", ".join(f"{prompt.note_id} ({prompt.room:.3f})" for prompt in risky)
        logger.warning(
            "Riziko uříznutí (místo na překlad / originál < %s): %s",
            OUTPUT_RATIO,
            listed,
        )
    path = args.workdir / REQUESTS
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_requests(prompts)), encoding="utf-8")
    logger.info("Požadavky zapsány do %s.", path)
    return 0


def _prompt(tokenizer: PreTrainedTokenizerBase, path: Path) -> Prompt:
    raw = path.read_bytes()
    text = raw.decode("utf-8")
    content = {
        "type": "text",
        "source_lang_code": SOURCE_LANG,
        "target_lang_code": TARGET_LANG,
        "text": text,
    }
    token_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": [content]}],
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
    )["input_ids"]
    if token_ids[0] != tokenizer.bos_token_id or token_ids[1] == tokenizer.bos_token_id:
        msg = f"Prompt zprávy {path.stem} nezačíná právě jedním BOS."
        raise ValueError(msg)
    source = len(tokenizer(text.strip(), add_special_tokens=False)["input_ids"])
    room = (WINDOW - len(token_ids)) / source
    return Prompt(path.stem, hashlib.sha256(raw).hexdigest(), token_ids, room)


def _requests(prompts: Sequence[Prompt]) -> dict[str, object]:
    stop = GenerationConfig.from_pretrained(MODEL, revision=REVISION).eos_token_id
    header = {
        "model": MODEL,
        "revision": REVISION,
        "dtype": DTYPE,
        "window": WINDOW,
        "decoding": {"temperature": 0.0, "stop_token_ids": stop},
    }
    requests = [
        {
            "id": prompt.note_id,
            "key": _key(header, prompt.token_ids),
            "source_sha256": prompt.source_sha256,
            "prompt_token_ids": prompt.token_ids,
        }
        for prompt in prompts
    ]
    return {**header, "requests": requests}


def _key(header: Mapping[str, object], token_ids: Sequence[int]) -> str:
    canonical = json.dumps(
        {**header, "prompt_token_ids": token_ids},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("resources/synthetic/en"),
        help="adresář anglických zpráv",
    )
    parser.add_argument(
        "--workdir",
        type=Path,
        default=Path("build/translation"),
        help=f"adresář, kam se zapíše {REQUESTS}",
    )
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
