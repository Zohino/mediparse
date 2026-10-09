# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "jinja2>=3.1.6",
#     "transformers>=5.18.0",
# ]
#
# [tool.ty.environment]
# extra-paths = ["."]
#
# [tool.ty.analysis]
# allowed-unresolved-imports = ["transformers"]
# ///
"""Sestaví požadavky na překlad syntetických zpráv a rozdělí ty, které se nevejdou.

Prompt skládá chat šablona modelu v pevné revizi, takže ID tokenů jsou přesně ta,
která model čeká, a překladový skript je na GPU jen předá. Klíč požadavku je hash
všeho, co určuje překlad, a slouží zároveň jako cache. Zpráva, které v okně zbude
na překlad méně než SPLIT_RATIO násobek tokenů originálu, se rozdělí na části po
blocích a každá část jde jako samostatný požadavek. Dělení se počítá nad originálem,
značky ___ se nahradí číslovanými [[n]] až v částech. Zprávy, které rozdělit nejde,
skript vypíše a požadavky nezapíše.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import logging
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final

from markers import PLACEHOLDER, mask
from note_parts import UnsplittableError, part_ids, split
from transformers import AutoTokenizer, GenerationConfig

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from transformers import PreTrainedTokenizerBase

MODEL: Final = "google/translategemma-27b-it"
REVISION: Final = "7d10f0b72f89a2d0f268cea30727d8b77c0d25c2"
WINDOW: Final = 2048
SPLIT_RATIO: Final = 2.2
DTYPE: Final = "bfloat16"
SOURCE_LANG: Final = "en"
TARGET_LANG: Final = "cs"
REQUESTS: Final = "requests.json"
LOG_FORMAT: Final = "%(asctime)s %(levelname)s %(message)s"

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Prompt:
    """Prompt jedné části zprávy a ID požadavku."""

    request_id: str
    note_id: str
    source_sha256: str
    token_ids: list[int]


def main() -> int:
    """Zapíše požadavky na překlad, když jde rozdělit každá zpráva, která se nevejde.

    Returns:
        0 po zápisu požadavků, 1 když korpus nemá zprávy nebo se některé zprávě
        nevejde do okna ani jeden řádek.
    """
    logging.basicConfig(format=LOG_FORMAT)
    logger.setLevel(logging.INFO)
    args = _parser().parse_args()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    paths = sorted(args.source.glob("*.txt"))
    if not paths:
        logger.error("V %s nejsou žádné zprávy.", args.source)
        return 1
    prompts: list[Prompt] = []
    unsplittable: list[str] = []
    for path in paths:
        try:
            prompts.extend(_prompts(tokenizer, path))
        except UnsplittableError:
            unsplittable.append(path.stem)
    if unsplittable:
        logger.error("Nejde rozdělit: %s", ", ".join(unsplittable))
        return 1
    lengths = [len(prompt.token_ids) for prompt in prompts]
    logger.info(
        "Požadavků %d, prompt má medián %s a nejvýš %d tokenů z okna %d.",
        len(prompts),
        statistics.median(lengths),
        max(lengths),
        WINDOW,
    )
    divided = [prompt for prompt in prompts if prompt.request_id != prompt.note_id]
    notes = {prompt.note_id for prompt in divided}
    listed = ", ".join(prompt.request_id for prompt in divided)
    logger.info(
        "Rozděleno %d zpráv na %d částí (%s) z %d zpráv.",
        len(notes),
        len(divided),
        listed,
        len(paths),
    )
    path = args.workdir / REQUESTS
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_requests(prompts)), encoding="utf-8")
    logger.info("Požadavky zapsány do %s.", path)
    return 0


def _prompts(tokenizer: PreTrainedTokenizerBase, path: Path) -> list[Prompt]:
    raw = path.read_bytes()
    fits = functools.partial(_fits, tokenizer)
    digest = hashlib.sha256(raw).hexdigest()
    try:
        texts = mask(split(raw.decode("utf-8"), fits))
        return [
            Prompt(request_id, path.stem, digest, _token_ids(tokenizer, text))
            for request_id, text in zip(
                part_ids(path.stem, len(texts)), texts, strict=True
            )
        ]
    except RuntimeError as error:
        msg = f"Prompt zprávy {path.stem} nezačíná právě jedním BOS."
        raise RuntimeError(msg) from error


def _fits(tokenizer: PreTrainedTokenizerBase, text: str) -> bool:
    room = WINDOW - len(_token_ids(tokenizer, text))
    source = len(tokenizer(text.strip(), add_special_tokens=False)["input_ids"])
    return room >= SPLIT_RATIO * source


def _token_ids(tokenizer: PreTrainedTokenizerBase, text: str) -> list[int]:
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
        msg = "Prompt nezačíná právě jedním BOS."
        raise RuntimeError(msg)
    return token_ids


def _requests(prompts: Sequence[Prompt]) -> dict[str, object]:
    stop = GenerationConfig.from_pretrained(MODEL, revision=REVISION).eos_token_id
    header = {
        "model": MODEL,
        "revision": REVISION,
        "dtype": DTYPE,
        "window": WINDOW,
        "masking": PLACEHOLDER,
        "decoding": {"temperature": 0.0, "stop_token_ids": stop},
    }
    requests = [
        {
            "id": prompt.request_id,
            "note_id": prompt.note_id,
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
