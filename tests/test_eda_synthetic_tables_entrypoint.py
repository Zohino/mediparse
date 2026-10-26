"""Vstupní bod syntetických tabulek: jeden parquet na tabulku manifestu a značka syntetiky."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Final

import duckdb
import pyarrow.parquet as pq
import pytest

from mediparse.domain.dua import MIN_CELL
from mediparse.domain.synthetic_tables import BLOCKS, DEFAULT_ROWS
from mediparse.entrypoints.eda_synthetic_tables import run
from mediparse.entrypoints.exit_code import ExitCode
from mediparse.infrastructure.duckdb_cohort import MIN_CHARS
from tests.cohort_support import candidate_prefixes
from tests.support import (
    REPOSITORY_DIAGNOSIS_CANDIDATES,
    REPOSITORY_SECTION_HEADERS,
    REPOSITORY_TEXT_MENTIONS,
)

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

ROWS: Final = 3
ARGPARSE_USAGE_ERROR: Final = 2
MANIFEST = {
    "tables": [
        {"file": "discharge.csv.gz", "bytes": 1, "records": 3, "columns": ["a", "b"]},
        {"file": "admissions.csv.gz", "bytes": 1, "records": 3, "columns": ["c"]},
    ]
}


def _manifest(root: Path, manifest: Mapping[str, object] = MANIFEST) -> Path:
    path = root / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def _write_candidates(
    path: Path, prefixes: Mapping[str, Mapping[str, object]] | None = None
) -> Path:
    """Zapíše config kandidátů s předponami a s názvy a zkratkami z configu v repu.

    Returns:
        Cesta k zapsanému souboru.
    """
    names = json.loads(REPOSITORY_DIAGNOSIS_CANDIDATES.read_text(encoding="utf-8"))[
        "candidates"
    ]
    merged = {
        diagnosis: {
            **fields,
            "keywords_en": names[diagnosis]["keywords_en"],
            "acronyms_en": names[diagnosis]["acronyms_en"],
        }
        for diagnosis, fields in (prefixes or candidate_prefixes()).items()
    }
    path.write_text(json.dumps({"candidates": merged}), encoding="utf-8")
    return path


def _arguments(
    manifest: Path,
    out: Path,
    *,
    rows: int | None = None,
    omit: str | None = None,
    **paths: Path,
) -> list[str]:
    """Úplné platné argumenty běhu; ``paths`` přepíše ``candidates``, ``lexicon`` nebo ``mentions``.

    Returns:
        Argumenty bez volby ``omit``.
    """
    candidates = paths.get("candidates") or _write_candidates(
        manifest.parent / "candidates.json"
    )
    options: dict[str, object] = {
        "manifest": manifest,
        "candidates": candidates,
        "lexicon": REPOSITORY_SECTION_HEADERS,
        "mentions": REPOSITORY_TEXT_MENTIONS,
        "out": out,
        "prefix": "canary",
        **paths,
    }
    if rows is not None:
        options["rows"] = rows
    return [f"--{name}={value}" for name, value in options.items() if name != omit]


def test_writes_one_all_string_parquet_per_manifest_table(tmp_path: Path) -> None:
    """Každá tabulka manifestu dostane parquet se samými řetězci a požadovaným počtem řádků."""
    out = tmp_path / "synthetic" / "tables"

    code = run(_arguments(_manifest(tmp_path), out, rows=ROWS))

    assert code is ExitCode.OK
    discharge = pq.read_table(out / "discharge.parquet")
    assert discharge.schema.names == ["a", "b"]
    assert {str(field.type) for field in discharge.schema} == {"string"}
    assert discharge.num_rows == ROWS
    assert pq.read_table(out / "admissions.parquet").num_rows == ROWS


def test_writes_marker_with_prefix_and_rows(tmp_path: Path) -> None:
    """Značka synthetic.json nese prefix a počet řádků."""
    out = tmp_path / "tables"

    run(_arguments(_manifest(tmp_path), out, rows=ROWS))

    marker = json.loads((out / "synthetic.json").read_text(encoding="utf-8"))
    assert marker == {"prefix": "canary", "rows": ROWS}


def test_marker_carries_default_rows_from_domain(tmp_path: Path) -> None:
    """Bez ``--rows`` značka nese ``DEFAULT_ROWS`` z domény."""
    out = tmp_path / "tables"

    run(_arguments(_manifest(tmp_path), out, rows=None))

    marker = json.loads((out / "synthetic.json").read_text(encoding="utf-8"))
    assert marker["rows"] == DEFAULT_ROWS


def test_missing_manifest_is_refused(tmp_path: Path) -> None:
    """Chybějící manifest ukončí krok kódem REFUSED a nic nezapíše."""
    out = tmp_path / "tables"

    code = run(_arguments(tmp_path / "none.json", out, rows=ROWS))

    assert code is ExitCode.REFUSED
    assert not out.exists()


def test_rows_default_when_not_given(tmp_path: Path) -> None:
    """Bez ``--rows`` se zapíše ``DEFAULT_ROWS`` řádků."""
    out = tmp_path / "tables"

    run(_arguments(_manifest(tmp_path), out, rows=None))

    assert pq.read_table(out / "admissions.parquet").num_rows == DEFAULT_ROWS


@pytest.mark.parametrize("option", ["candidates", "lexicon", "mentions"])
def test_missing_required_option_exits_with_usage_error(
    tmp_path: Path, option: str, capsys: pytest.CaptureFixture[str]
) -> None:
    """Bez povinné volby argparse skončí kódem 2 a hláška volbu jmenuje."""
    with pytest.raises(SystemExit) as raised:
        run(_arguments(_manifest(tmp_path), tmp_path / "tables", omit=option))

    assert raised.value.code == ARGPARSE_USAGE_ERROR
    assert f"--{option}" in capsys.readouterr().err


def test_invalid_candidates_are_refused_without_output(tmp_path: Path) -> None:
    """Neplatný config skončí REFUSED a nevznikne žádný parquet."""
    out = tmp_path / "tables"
    broken = tmp_path / "broken.json"
    broken.write_text("{", encoding="utf-8")

    code = run(_arguments(_manifest(tmp_path), out, rows=ROWS, candidates=broken))

    assert code is ExitCode.REFUSED
    assert not out.exists()


def test_candidates_without_icd9_prefix_of_block_diagnosis_are_refused(
    tmp_path: Path,
) -> None:
    """Config platný pro loader, ale bez předpony ICD-9 u diabetu, skončí REFUSED bez parquetu."""
    prefixes = candidate_prefixes()
    prefixes["diabetes"]["icd9_prefixes"] = []
    prefixes["diabetes"]["icd10_prefixes"] = ["E11"]
    out = tmp_path / "tables"

    code = run(
        _arguments(
            _manifest(tmp_path),
            out,
            rows=ROWS,
            candidates=_write_candidates(tmp_path / "candidates.json", prefixes),
        )
    )

    assert code is ExitCode.REFUSED
    assert not out.exists()


def test_table_without_columns_is_refused_before_any_output(tmp_path: Path) -> None:
    """Druhá tabulka bez sloupců shodí běh dřív, než se zapíše první parquet."""
    manifest = {
        "tables": [
            {"file": "discharge.csv.gz", "bytes": 1, "records": 3, "columns": ["a"]},
            {"file": "admissions.csv.gz", "bytes": 1, "records": 3, "columns": []},
        ]
    }
    out = tmp_path / "tables"

    code = run(_arguments(_manifest(tmp_path, manifest), out, rows=ROWS))

    assert code is ExitCode.REFUSED
    assert not out.exists()


def test_notes_join_codes_and_pass_the_length_filter(tmp_path: Path) -> None:
    """Zprávy sdílí hadm_id s kódy ICD a mají aspoň MIN_CHARS znaků, takže vstupní korpus není prázdný."""
    manifest = {
        "tables": [
            {
                "file": "discharge.csv.gz",
                "bytes": 1,
                "records": 3,
                "columns": ["hadm_id", "text"],
            },
            {
                "file": "diagnoses_icd.csv.gz",
                "bytes": 1,
                "records": 3,
                "columns": ["hadm_id", "icd_version"],
            },
        ]
    }
    out = tmp_path / "tables"

    run(_arguments(_manifest(tmp_path, manifest), out, rows=None))

    notes = pq.read_table(out / "discharge.parquet").to_pydict()
    codes = pq.read_table(out / "diagnoses_icd.parquet").to_pydict()
    assert notes["hadm_id"] == codes["hadm_id"]
    assert min(len(text) for text in notes["text"]) >= MIN_CHARS
    assert len(notes["text"]) == DEFAULT_ROWS
    with duckdb.connect() as connection:
        joined = connection.execute(
            "SELECT TRY_CAST(c.icd_version AS INTEGER), count(*) "
            "FROM read_parquet(?) n JOIN read_parquet(?) c USING (hadm_id) "
            "GROUP BY 1",
            [str(out / "discharge.parquet"), str(out / "diagnoses_icd.parquet")],
        ).fetchall()
    assert dict(joined) == {
        9: 3 * MIN_CELL,
        10: 3 * MIN_CELL,
        None: DEFAULT_ROWS - BLOCKS * MIN_CELL,
    }


def test_invalid_lexicon_is_refused_without_output(tmp_path: Path) -> None:
    """Neplatný lexikon skončí REFUSED a nevznikne žádný parquet."""
    out = tmp_path / "tables"
    broken = tmp_path / "broken.json"
    broken.write_text("{", encoding="utf-8")

    code = run(_arguments(_manifest(tmp_path), out, rows=ROWS, lexicon=broken))

    assert code is ExitCode.REFUSED
    assert not out.exists()


def test_lexicon_without_bhc_section_is_refused_without_output(tmp_path: Path) -> None:
    """Platný lexikon bez sekce ``hospital_course`` skončí REFUSED bez parquetu."""
    lexicon = json.loads(REPOSITORY_SECTION_HEADERS.read_text(encoding="utf-8"))
    del lexicon["sections"]["hospital_course"]
    path = tmp_path / "lexicon.json"
    path.write_text(json.dumps(lexicon), encoding="utf-8")
    out = tmp_path / "tables"

    code = run(_arguments(_manifest(tmp_path), out, rows=ROWS, lexicon=path))

    assert code is ExitCode.REFUSED
    assert not out.exists()


def test_notes_have_bhc_headers_and_bhc_pairs_refer_to_notes(tmp_path: Path) -> None:
    """Přes DuckDB: 60 zpráv s hlavičkou BHC a 62 párů BHC s ``note_id`` propouštěcí zprávy."""
    manifest = {
        "tables": [
            {
                "file": "discharge.csv.gz",
                "bytes": 1,
                "records": 3,
                "columns": ["note_id", "text"],
            },
            {
                "file": "mimic-iv-bhc.csv.gz",
                "bytes": 1,
                "records": 3,
                "columns": ["note_id", "target"],
            },
        ]
    }
    out = tmp_path / "tables"

    run(_arguments(_manifest(tmp_path, manifest), out, rows=None))

    with duckdb.connect() as connection:
        with_header = connection.execute(
            "SELECT count(*) FROM read_parquet(?) WHERE text LIKE '%Brief Hospital Course:%'",
            [str(out / "discharge.parquet")],
        ).fetchone()
        pairs = connection.execute(
            "SELECT count(*) FROM read_parquet(?) b JOIN read_parquet(?) d USING (note_id)",
            [str(out / "mimic-iv-bhc.parquet"), str(out / "discharge.parquet")],
        ).fetchone()
    assert with_header == (BLOCKS * MIN_CELL,)
    assert pairs == (DEFAULT_ROWS,)


def test_invalid_mentions_are_refused_without_output(tmp_path: Path) -> None:
    """Neplatný config zmínek skončí REFUSED a nevznikne žádný parquet."""
    out = tmp_path / "tables"
    broken = tmp_path / "broken.json"
    broken.write_text("{", encoding="utf-8")

    code = run(_arguments(_manifest(tmp_path), out, rows=ROWS, mentions=broken))

    assert code is ExitCode.REFUSED
    assert not out.exists()


def test_notes_carry_mentions_from_the_configs(tmp_path: Path) -> None:
    """Text tvrzeného bloku nese klíčové slovo diabetu z configu kandidátů a text negovaného bloku negaci z configu zmínek."""
    manifest = {
        "tables": [
            {
                "file": "discharge.csv.gz",
                "bytes": 1,
                "records": 3,
                "columns": ["text"],
            }
        ]
    }
    out = tmp_path / "tables"

    run(_arguments(_manifest(tmp_path, manifest), out))

    texts = pq.read_table(out / "discharge.parquet").to_pydict()["text"]
    assert "diabetes" in texts[0].splitlines()
    assert "no diabetes" in texts[2 * MIN_CELL].splitlines()
    assert "diabetes" not in texts[BLOCKS * MIN_CELL - 1].splitlines()
