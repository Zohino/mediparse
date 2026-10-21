"""Tabulky, které pravidla Snakemake stahují a převádějí na parquet."""

from __future__ import annotations

from mediparse.entrypoints.mimic_download import mimic_table_files, mimic_tables


def test_every_downloaded_file_has_its_table() -> None:
    """Převod pokryje každý stažený soubor a jméno tabulky ho najde zpět."""
    files = mimic_table_files()

    assert sorted(files.values()) == sorted(mimic_tables())
    assert files["discharge"] == "discharge.csv.gz"


def test_bhc_is_downloaded_as_plain_csv() -> None:
    """MIMIC-IV-Ext-BHC je na PhysioNetu nekomprimované CSV a tabulka nese jeho jméno."""
    assert mimic_table_files()["mimic-iv-bhc"] == "mimic-iv-bhc.csv"
    assert mimic_tables()["mimic-iv-bhc.csv"].url == (
        "https://physionet.org/files/labelled-notes-hospital-course/1.2.0/mimic-iv-bhc.csv"
    )
