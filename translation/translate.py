# /// script
# requires-python = "==3.13.*"
# dependencies = [
#     "vllm>=0.30.0",
# ]
#
# [tool.ty.analysis]
# allowed-unresolved-imports = ["torch", "vllm", "vllm.**"]
# ///
"""Přeloží požadavky z prepare.py modelem ve vLLM a po přerušení naváže podle klíčů.

Skript nezná zprávy ani jazyky, jen klíče a ID tokenů promptu. Výstupy každé dávky
se hned uloží, takže nové spuštění přeloží jen požadavky, jejichž klíč ve výstupech
ještě není. Každé spuštění připíše do záznamu běhů GPU a verze knihoven, se kterými
překládalo, a každý výstup nese čas začátku běhu, který ho vytvořil.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import logging
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import torch
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from vllm import RequestOutput

REQUESTS: Final = "requests.json"
OUTPUTS: Final = "outputs.jsonl"
RUNS: Final = "runs.jsonl"
BATCH: Final = 25
MIN_MEMORY_GB: Final = 26
LOG_FORMAT: Final = "%(asctime)s %(levelname)s %(message)s"

logger = logging.getLogger(__name__)

type Record = dict[str, Any]


def main() -> int:
    """Přeloží požadavky, které ještě nemají výstup, případně jen vybrané.

    Returns:
        0, když má výstup každý vybraný požadavek.
    """
    logging.basicConfig(format=LOG_FORMAT)
    logger.setLevel(logging.INFO)
    args = _parser().parse_args()
    raw = (args.workdir / REQUESTS).read_bytes()
    plan = json.loads(raw)
    outputs = _read(args.workdir / OUTPUTS)
    done = {output["key"] for output in outputs}
    pending = [
        request
        for request in plan["requests"]
        if request["key"] not in done and (not args.ids or request["id"] in args.ids)
    ]
    logger.info("Požadavků %d, k překladu %d.", len(plan["requests"]), len(pending))
    if not pending:
        return 0
    run = _run(hashlib.sha256(raw).hexdigest(), _checked_gpu())
    runs = args.workdir / RUNS
    _write(runs, [*_read(runs), run])
    llm = LLM(
        model=plan["model"],
        revision=plan["revision"],
        tokenizer_revision=plan["revision"],
        dtype=plan["dtype"],
        max_model_len=plan["window"],
        seed=0,
        limit_mm_per_prompt={"image": 0},
    )
    translated = 0
    for batch in itertools.batched(pending, BATCH, strict=False):
        outputs.extend(_translate(llm, plan, batch, run["started"]))
        _write(args.workdir / OUTPUTS, outputs)
        translated += len(batch)
        logger.info("Přeloženo %d z %d.", translated, len(pending))
    return 0


def _checked_gpu() -> Record:
    if not torch.cuda.is_available():
        logger.error("Překlad potřebuje GPU s CUDA a žádnou nevidí.")
        raise SystemExit(1)
    gpu = torch.cuda.get_device_properties(0)
    memory = gpu.total_memory / 10**9
    if memory < MIN_MEMORY_GB:
        logger.error(
            "GPU %s má %.0f GB, model potřebuje aspoň %d GB.",
            gpu.name,
            memory,
            MIN_MEMORY_GB,
        )
        raise SystemExit(1)
    return {
        "gpu": gpu.name,
        "memory_gb": round(memory, 1),
        "capability": f"{gpu.major}.{gpu.minor}",
    }


def _run(requests_sha256: str, gpu: Record) -> Record:
    return {
        "started": datetime.now(UTC).isoformat(timespec="microseconds"),
        "requests_sha256": requests_sha256,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        **gpu,
        "cuda": torch.version.cuda,
        "packages": {name: version(name) for name in ("vllm", "torch", "transformers")},
    }


def _translate(
    llm: LLM, plan: Mapping[str, Any], batch: Sequence[Record], run: str
) -> list[Record]:
    prompts = [
        TokensPrompt(prompt_token_ids=request["prompt_token_ids"]) for request in batch
    ]
    sampling = [
        SamplingParams(
            **plan["decoding"],
            max_tokens=plan["window"] - len(request["prompt_token_ids"]),
        )
        for request in batch
    ]
    results = llm.generate(prompts, sampling, use_tqdm=False)
    return [
        _output(request, result, run)
        for request, result in zip(batch, results, strict=True)
    ]


def _output(request: Record, result: RequestOutput, run: str) -> Record:
    completion = result.outputs[0]
    return {
        "key": request["key"],
        "id": request["id"],
        "run": run,
        "text": completion.text,
        "finish_reason": completion.finish_reason,
        "output_tokens": len(completion.token_ids),
    }


def _read(path: Path) -> list[Record]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _write(path: Path, records: Iterable[Record]) -> None:
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(
        "".join(f"{json.dumps(record, ensure_ascii=False)}\n" for record in records),
        encoding="utf-8",
    )
    temporary.replace(path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workdir",
        type=Path,
        default=Path("build/translation"),
        help=f"adresář s {REQUESTS}, kam se zapíšou {OUTPUTS} a {RUNS}",
    )
    parser.add_argument(
        "--ids",
        nargs="+",
        help="přeloží jen požadavky s těmito ID, třeba v sondě",
    )
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
