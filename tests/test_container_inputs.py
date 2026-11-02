"""Soubory nezbytné pro testy spouštěné v obrazu kontejneru."""

import pytest

from tests.support import REPOSITORY


@pytest.mark.skipif(
    not (REPOSITORY / "Containerfile").is_file(),
    reason="Kontext sestavení ověřuje hostitelské CI; v běhovém obrazu není.",
)
def test_translation_prepare_is_available_to_container_tests() -> None:
    """Příprava překladu projde kontextem sestavení i COPY do běhového obrazu."""
    required = "translation/prepare.py"
    included = (REPOSITORY / ".dockerignore").read_text(encoding="utf-8").splitlines()
    copied = {
        source
        for line in (REPOSITORY / "Containerfile")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.startswith("COPY ")
        for source in line.split()[1:-1]
    }
    assert f"!{required}" in included
    assert required in copied
