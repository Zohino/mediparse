"""Inventář stažené tabulky MIMIC skriptem workflow/scripts/inventory.sh.

Skript běží nad dočasnými soubory ``.csv`` a ``.csv.gz``. Komprimovanost určuje
přípona, záznamy se počítají paritou uvozovek, ne fyzickými řádky.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Final

import pytest

from tests.support import BLOCKING_NAMES, blocking_reason

SCRIPT: Final = Path(__file__).parents[1] / "workflow" / "scripts" / "inventory.sh"
# Hlavička s CRLF, zdvojená uvozovka a zalomení uvnitř pole: dva záznamy na pěti řádcích.
TRICKY: Final = 'note_id,text\r\n1,"řekl ""ne""\nkonec"\n2,"a\n\nb"\n'
# Tolik záznamů, aby useknutý gzip ještě vydal celé záznamy se sudou paritou uvozovek.
MANY_RECORDS: Final = 20_000


def _write(directory: Path, name: str, text: str) -> Path:
    path = directory / name
    data = text.encode()
    path.write_bytes(gzip.compress(data, mtime=0) if name.endswith(".gz") else data)
    return path


def _inventory(
    table: Path, sha256: str | None = None, **environ: str
) -> tuple[subprocess.CompletedProcess[str], Path]:
    output = table.with_name(table.name + ".json")
    result = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            str(table),
            sha256
            if sha256 is not None
            else hashlib.sha256(table.read_bytes()).hexdigest(),
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        env={**os.environ, **dict.fromkeys(BLOCKING_NAMES, ""), **environ},
        timeout=60,
    )
    return result, output


def test_inventory_blocks_before_reading_table(tmp_path: Path) -> None:
    """Blokace předchází i čtení otisku neexistující tabulky."""
    result, output = _inventory(tmp_path / "missing.csv.gz", "00", CODEX_THREAD_ID="1")

    assert result.returncode == 1
    assert blocking_reason("CODEX_THREAD_ID") in result.stderr
    assert "nejde přečíst" not in result.stderr
    assert not output.exists()


def test_inventory_blocks_when_access_check_fails_silently(tmp_path: Path) -> None:
    """Nenulový kód stráže zastaví inventář i bez diagnostiky."""
    table = _write(tmp_path, "table.csv", "a,b\n1,2\n")
    command = tmp_path / "mediparse-data-access-check"
    command.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
    command.chmod(0o755)

    result, output = _inventory(
        table, PATH=f"{tmp_path}{os.pathsep}{os.environ['PATH']}"
    )

    assert result.returncode != 0
    assert not output.exists()


def _refused(table: Path, message: str) -> None:
    result, output = _inventory(table)
    assert result.returncode == 1
    assert f"{table} {message}" in result.stderr
    assert not output.exists()


@pytest.mark.parametrize("name", ["discharge.csv", "discharge.csv.gz"])
def test_inventory_counts_records_by_quote_parity(tmp_path: Path, name: str) -> None:
    """Prosté i komprimované CSV dají záznamy, ne fyzické řádky; velikost je na disku."""
    table = _write(tmp_path, name, TRICKY)

    result, output = _inventory(table)

    assert result.returncode == 0, result.stderr
    assert json.loads(output.read_text(encoding="utf-8")) == {
        "file": name,
        "bytes": table.stat().st_size,
        "records": 2,
        "columns": ["note_id", "text"],
        "sha256": hashlib.sha256(table.read_bytes()).hexdigest(),
    }


def test_changed_digest_is_refused(tmp_path: Path) -> None:
    """Otisk lišící se v posledním znaku odmítne CSV bez vytvoření inventáře."""
    table = _write(tmp_path, "discharge.csv.gz", TRICKY)
    expected = hashlib.sha256(table.read_bytes()).hexdigest()
    changed = expected[:-1] + ("0" if expected[-1] != "0" else "1")

    result, output = _inventory(table, changed)

    assert result.returncode != 0
    assert f"{table}: otisk SHA-256 nesouhlasí s configem" in result.stderr
    assert not output.exists()


@pytest.mark.parametrize("name", ["discharge.csv", "discharge.csv.gz"])
def test_unclosed_quote_is_refused(tmp_path: Path, name: str) -> None:
    """Neuzavřené pole v uvozovkách by spojilo zbytek souboru do jednoho záznamu."""
    _refused(
        _write(tmp_path, name, 'a,b\n1,"x\n2,y\n'), "má neuzavřené pole v uvozovkách"
    )


def test_header_without_records_is_refused(tmp_path: Path) -> None:
    """Tabulka jen s hlavičkou nemá co převést."""
    _refused(_write(tmp_path, "empty.csv", "a,b\n"), "nemá žádný záznam")


def test_empty_file_is_refused(tmp_path: Path) -> None:
    """Prázdný soubor nemá hlavičku, ze které by šly sloupce."""
    _refused(_write(tmp_path, "empty.csv", ""), "nemá hlavičku")


def test_truncated_gzip_is_refused(tmp_path: Path) -> None:
    """Useknutý gzip neprojde, ani když celé záznamy před koncem mají sudou paritu."""
    rows = "".join(f'{n},"zpráva\n{n}"\n' for n in range(MANY_RECORDS))
    table = _write(tmp_path, "discharge.csv.gz", "note_id,text\n" + rows)
    data = table.read_bytes()
    table.write_bytes(data[: len(data) // 2])

    _refused(table, "nejde přečíst ani rozbalit")


def test_plain_csv_named_as_gzip_is_refused(tmp_path: Path) -> None:
    """Přípona .gz bez gzipu je chyba čtení, ne prosté CSV ani tabulka bez hlavičky."""
    table = tmp_path / "discharge.csv.gz"
    table.write_text(TRICKY, encoding="utf-8")

    _refused(table, "nejde přečíst ani rozbalit")


def test_missing_output_argument_is_refused(tmp_path: Path) -> None:
    """Bez výstupní cesty skript vypíše použití a nic nezapíše."""
    table = _write(tmp_path, "discharge.csv", TRICKY)

    result = subprocess.run(
        ["bash", str(SCRIPT), str(table)],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
    )

    assert result.returncode == 1
    assert "použití:" in result.stderr
